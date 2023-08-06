import numpy as np
import tensorflow as tf
from tensorflow import keras

import argparse
import json
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Any, Final, Optional

from projects import Projects
from data.loader import TfDatasetMaker
from model.edge import EdgeModel
from model.model_data import get_model_data
from model.optimizer import ModelOptimizer
from model.tuner import tune_model
from model.utilities import ModelManager
from utils.constants import (
    MOD_COMPILE_ARGS,
    MOD_HIDDEN_ARGS,
    MOD_OUTPUT_ARGS,
    MOD_X_SHAPE,
    MOD_Y_SHAPE
)
from utils.custom_types import Filepath
from utils.defaults import Defaults


class ServerModelGenerator(ModelManager):

    def __init__(self, project: str, untrained: bool, server_dir: Filepath, learning_rate: float, seed: int,
                 train_ds: tf.data.Dataset, test_ds: tf.data.Dataset, val_ds: tf.data.Dataset) -> None:
        super().__init__(project, untrained, server_dir)
        self.learning_rate = learning_rate
        self.seed = seed
        self.train_ds = train_ds
        self.test_ds = test_ds
        self.val_ds = val_ds

        self.model_data: dict[str, Any] = get_model_data(self.project, seed=self.seed, learning_rate=self.learning_rate)
        self.compile_args: dict[str, Any] = self.model_data[MOD_COMPILE_ARGS]
        self.model: Optional[keras.Sequential] = None
        self.optimized_model: Optional[keras.Sequential] = None

        np.random.seed(self.seed)
        tf.random.set_seed(self.seed)

    def _evaluate_compressed_model(self, comp_mode: str) -> None:
        ModelOptimizer.print_model_weights_sparsity(self.optimized_model, comp_mode)
        self.optimized_model.compile(**self.compile_args)
        results = self.optimized_model.evaluate(self.test_ds, verbose=0, return_dict=True)
        loss, accuracy = results['loss'], results['accuracy']
        print(f'\nModel metrics after {comp_mode}:\tloss = {loss:.4f}, accuracy = {accuracy:.4f}')

    def _get_tuned_model(self) -> keras.Sequential:
        try:
            return keras.models.load_model(self.tuned_model_dir)
        except (ValueError, OSError, FileNotFoundError):
            print(f'WARNING: Tuned model not found, generating tuned model for project "{self.project}"...')
            return tune_model(
                project=self.project,
                train_ds=self.train_ds,
                val_ds=self.val_ds,
                learning_rate=self.learning_rate,
                seed=self.seed,
                max_layers=Defaults.TUNE_MAX_LAYERS,
                max_units=Defaults.TUNE_MAX_UNITS,
                max_epochs=Defaults.TUNE_MAX_EPOCHS,
                factor=Defaults.TUNE_FACTOR,
                iterations=Defaults.TUNE_ITERATIONS,
                save_dir=self.tuned_model_dir
            )

    def _get_default_model(self) -> keras.Sequential:
        try:
            return keras.models.load_model(self.default_model_dir)
        except (ValueError, OSError, FileNotFoundError):
            print(f'WARNING: Default model not found, generating new default model for {self.project}...')
            x_shape, y_shape = self.model_data[MOD_X_SHAPE], self.model_data[MOD_Y_SHAPE]
            hidden_args = self.model_data[MOD_HIDDEN_ARGS]
            output_args = self.model_data[MOD_OUTPUT_ARGS]
            model: keras.Sequential = keras.Sequential([
                keras.Input(shape=(np.product(x_shape),), name='input'),
                keras.layers.Reshape(x_shape, name='input_2D_reshape'),
                keras.layers.Conv2D(32, 3, padding='valid', **hidden_args),
                keras.layers.MaxPooling2D(),
                keras.layers.Conv2D(32, 3, padding='valid', **hidden_args),
                keras.layers.MaxPooling2D(),
                keras.layers.Conv2D(64, 3, padding='valid', **hidden_args),
                keras.layers.MaxPooling2D(),
                keras.layers.Dropout(0.1),
                keras.layers.Flatten(),
                keras.layers.Dense(128, **hidden_args),
                keras.layers.Dense(y_shape, name='output', **output_args)
            ])
            model.compile(**self.compile_args)
            model.save(self.default_model_dir)
            return model

    def optimize_model(self, tune_epochs: int, target_sparsity: float, target_clusters: int) -> None:
        model_optimizer = ModelOptimizer(self.compile_args, self.train_ds, self.val_ds)
        # prune_epochs: int = round(0.65 * tune_epochs)
        # cluster_epochs: int = tune_epochs - prune_epochs
        try:
            self.optimized_model = keras.models.clone_model(self.model)
            self.optimized_model.set_weights(self.model.get_weights())
            self.optimized_model = model_optimizer.prune_model(self.optimized_model, target_sparsity, tune_epochs)
            self._evaluate_compressed_model('pruning')
            self.optimized_model = model_optimizer.cluster_model(self.optimized_model, target_clusters, tune_epochs)
            self._evaluate_compressed_model('clustering')
            self._init_dir(self.server_model_dir)
            with open(self.server_model_dir / 'comp_params.json', 'w') as f:
                f.write(json.dumps({
                    'comp_params': {
                        'target_sparsity': target_sparsity,
                        'target_clusters': target_clusters,
                        'comp_epochs': tune_epochs,
                    }
                }))
        except ValueError as e:
            self.optimized_model = None
            print(f'Error: unable to compress model\n{e.args[0]}')

    def generate_server_model(self, epochs: int, tuned_model: bool = False) -> None:
        try:
            self.model = self._get_tuned_model() if tuned_model else self._get_default_model()
        except KeyError:
            raise OSError(f'No model defined under "{self.project}" in `utils/models.py`')
        if not self.untrained:
            self.model.fit(self.train_ds, epochs=epochs, validation_data=self.val_ds)

    def save_server_model(self) -> None:
        if self.optimized_model is not None:
            self.optimized_model.save(self.server_model_dir)
        else:
            self.model.save(self.server_model_dir)
        print(f'saved server model to \"{self.server_model_dir}\"')


class EdgeModelConverter:
    Q_INT_8: Final[str] = 'int8'
    Q_DYNAMIC: Final[str] = 'dynamic'
    Q_FLOAT_16: Final[str] = 'float16'
    Q_CHOICES: Final[dict[int, str]] = {i: q_type for i, q_type in
                                        enumerate([None, Q_INT_8, Q_DYNAMIC, Q_FLOAT_16], 0xD0)}

    def __init__(self,
                 project: str,
                 server_model: keras.Sequential,
                 untrained: bool,
                 repr_ds: tf.data.Dataset,
                 edge_model_dir: Optional[Filepath] = None,
                 basic_model_dir: Optional[Filepath] = None,
                 quantization: Optional[str] = None
                 ) -> None:
        self.project: str = project
        self.p_upper: str = self.project.upper()
        self.p_lower: str = self.project.lower()
        self.server_model: keras.Sequential = server_model
        self.untrained: bool = untrained
        if edge_model_dir is not None:
            self.edge_model_dir: Path = Path(edge_model_dir)
        else:
            self.edge_model_dir: Path = Defaults.CLIENT_MODEL_DIR
        self.basic_model_dir: Path = Path(basic_model_dir or self.edge_model_dir.parent / 'basic' /
                                          (('untrained' if self.untrained else 'trained') + '_models'))
        if not self.edge_model_dir.exists():
            self.edge_model_dir.mkdir(parents=True)
        if not self.basic_model_dir.exists():
            self.basic_model_dir.mkdir(parents=True)
        self.repr_ds: tf.data.Dataset = repr_ds
        self.quantization: Optional[str] = quantization
        self.basic_model_path: Optional[Path] = None
        self.quant_id_map = {q_type: i for i, q_type in EdgeModelConverter.Q_CHOICES.items()}
        self.server_model.summary()

    def _repr_data_generator(self):
        for x, _ in self.repr_ds:
            yield (
                'infer', {
                    'x': x
                }
            )

    def save_trainable_model(self) -> None:
        model_path: Path = self.edge_model_dir / (f'{self.project}_' + ('untrained' if self.untrained else 'trained'))
        self.server_model.save(model_path)
        print(f'saved trainable client model to \"{model_path}\"')

    def generate_basic_model(self) -> None:
        client_model: EdgeModel = EdgeModel(self.server_model)
        with TemporaryDirectory() as temp_dir:
            tf.saved_model.save(
                client_model,
                temp_dir,
                signatures={
                    'infer': client_model.infer.get_concrete_function(),
                }
            )
            converter: tf.lite.TFLiteConverter = tf.lite.TFLiteConverter.from_saved_model(temp_dir)
            if self.quantization is not None:
                converter.optimizations = [tf.lite.Optimize.DEFAULT]
                if self.quantization == EdgeModelConverter.Q_FLOAT_16:
                    converter.target_spec.supported_types = [tf.float16]
                elif self.quantization == EdgeModelConverter.Q_INT_8:
                    converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
                    converter.inference_input_type = tf.int8
                    converter.inference_output_type = tf.int8
                    converter.representative_dataset = self._repr_data_generator
            converter._experimental_unfold_large_splat_constant = True
            tflm_model = converter.convert()
        tf.lite.experimental.Analyzer.analyze(model_content=tflm_model)
        quant_type_flag: bytes = bytes([0xD0 if self.quantization is None else self.quant_id_map[self.quantization]])
        print(f'Model Quantization Type: {self.quantization} ({quant_type_flag})')
        self.basic_model_path: Path = self.basic_model_dir / (self.p_lower + '_model.tflite')
        with open(self.basic_model_path, 'wb') as f:
            f.write(tflm_model + quant_type_flag)

    def generate_comparison_models(self) -> None:
        comp_dir: Path = self.edge_model_dir / 'comparison'
        if not comp_dir.exists():
            comp_dir.mkdir(parents=True)
        for f in comp_dir.iterdir():
            f.unlink()
        client_model: EdgeModel = EdgeModel(self.server_model)
        tflm_models: dict[str, bytes] = {}
        with TemporaryDirectory() as temp_dir:
            tf.saved_model.save(
                client_model,
                temp_dir,
                signatures={
                    'infer': client_model.infer.get_concrete_function(),
                }
            )
            converter: tf.lite.TFLiteConverter = tf.lite.TFLiteConverter.from_saved_model(temp_dir)
            tflm_models[f'{self.p_lower}_model_with_no_quantization.tflite'] = converter.convert() + bytes([0xD0])
            converter = tf.lite.TFLiteConverter.from_saved_model(temp_dir)
            converter.optimizations = [tf.lite.Optimize.DEFAULT]
            tflm_models[f'{self.p_lower}_model_with_dynamic_range_quantization.tflite'] = converter.convert() + bytes(
                [0xD2])
            converter = tf.lite.TFLiteConverter.from_saved_model(temp_dir)
            converter.optimizations = [tf.lite.Optimize.DEFAULT]
            converter.target_spec.supported_types = [tf.float16]
            tflm_models[f'{self.p_lower}_model_with_float_fallback_quantization.tflite'] = converter.convert() + bytes(
                [0xD3])
            converter = tf.lite.TFLiteConverter.from_saved_model(temp_dir)
            converter.optimizations = [tf.lite.Optimize.DEFAULT]
            converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
            converter.inference_input_type = tf.int8
            converter.inference_output_type = tf.int8
            converter.representative_dataset = self._repr_data_generator
            tflm_models[f'{self.p_lower}_model_with_full_integer_quantization.tflite'] = converter.convert() + bytes(
                [0xD1])
            print(f'Generated comparison models at {comp_dir}')
        for model_name, model in tflm_models.items():
            with open(comp_dir / model_name, 'wb') as f:
                f.write(model)


def main():
    args: argparse.Namespace = parser.parse_args()
    project: str = args.project.replace("'", "")
    ds_loader: TfDatasetMaker = TfDatasetMaker.from_project(project, val_split=args.val_split, seed=args.seed)
    train_ds, val_ds, test_ds = ds_loader.train_ds, ds_loader.val_ds, ds_loader.test_ds
    server_gen: ServerModelGenerator = ServerModelGenerator(
        project=project,
        untrained=args.untrained,
        server_dir=args.server_model_dir.replace("'", ""),
        learning_rate=args.learning_rate,
        seed=args.seed,
        train_ds=train_ds,
        test_ds=test_ds,
        val_ds=val_ds,
    )
    server_gen.generate_server_model(args.epochs, args.tuned_model)
    if not args.no_compress:
        server_gen.optimize_model(args.comp_epochs, args.sparsity / 100, args.clusters)
    server_gen.save_server_model()

    if not args.server_only:
        client_conv: EdgeModelConverter = EdgeModelConverter(
            project=server_gen.project,
            server_model=keras.models.load_model(server_gen.server_model_dir),
            untrained=server_gen.untrained,
            edge_model_dir=args.client_model_dir.replace("'", ""),
            basic_model_dir=None,
            repr_ds=ds_loader.val_ds,
            quantization=args.client_quant,
        )
        client_conv.save_trainable_model()
        client_conv.generate_basic_model()
        if args.comparison:
            client_conv.generate_comparison_models()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(prog='python -m model.generator')
    parser.add_argument(
        '-p', '--project',
        type=str,
        required=True,
        choices=Projects.all(),
        metavar='NAME',
        help='the project name (available: %(choices)s)'
    )
    parser.add_argument(
        '-e', '--epochs',
        type=int,
        required=True,
        help='training epochs (must be >=1)'
    )
    parser.add_argument(
        '--untrained',
        action='store_true',
        help='generate untrained server and client models (overrides "--epochs"!)'
    )
    parser.add_argument(
        '-c', '--client_model_dir',
        type=str,
        metavar='DIR',
        default=Defaults.CLIENT_MODEL_DIR.as_posix(),
        help='directory for storing generated client model (default: %(default)s)'
    )
    parser.add_argument(
        '-s', '--server_model_dir',
        type=str,
        metavar='DIR',
        default=Defaults.SERVER_MODEL_DIR.as_posix(),
        help='directory for storing generated server model (default: %(default)s)'
    )
    parser.add_argument(
        '--server_only',
        action='store_true',
        help='generate only server model and skip client model conversion'
    )
    parser.add_argument(
        '--tuned_model',
        action='store_true',
        help='generate KerasTuner tuned model instead of default model'
    )
    parser.add_argument(
        '--comparison',
        action='store_true',
        help='generate TFLM comparison models'
    )
    comp_args = parser.add_argument_group('model compression parameters')
    comp_args.add_argument(
        '--no_compress',
        action='store_true',
        help='skips server model compression'
    )
    comp_args.add_argument(
        '--comp_epochs',
        type=int,
        default=1,
        help='number of epochs to tune model with during each mode of compression (default: %(default)s)'
    )
    comp_args.add_argument(
        '--client_quant',
        type=str,
        choices=[c for c in EdgeModelConverter.Q_CHOICES.values() if c],
        metavar='TYPE',
        help='choose type of client model quantization (available: %(choices)s)'
    )
    comp_args.add_argument(
        '--sparsity',
        type=float,
        default=Defaults.OPT_TARGET_SPARSITY,
        help='choose target weights sparsity for model pruning (default: %(default)s)'
    )
    comp_args.add_argument(
        '--clusters',
        type=int,
        default=Defaults.OPT_TARGET_CLUSTERS,
        help='choose target number of clusters for model clustering (default: %(default)s)'
    )
    gen_args = parser.add_argument_group('additional model generation parameters')
    gen_args.add_argument(
        '--learning_rate',
        type=float,
        metavar='RATE',
        default=Defaults.ML_LEARNING_RATE,
        help='the model\'s optimizer learning rate (default: %(default)s)'
    )
    gen_args.add_argument(
        '--seed',
        type=int,
        default=Defaults.ML_SEED,
        help='an integer to use for seeding RNG operations (default: %(default)s)'
    )
    gen_args.add_argument(
        '--val_split',
        type=float,
        metavar='FRACTION',
        default=Defaults.DS_VAL_SPLIT,
        help='fraction of dataset to use for validation (default: %(default)s)'
    )

    main()
