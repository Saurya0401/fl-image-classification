import keras_tuner as kt
import tensorflow as tf
from tensorflow import keras

import argparse
from pathlib import Path
from typing import Any, Optional

from projects import Projects
from data.loader import TfDatasetMaker
from model.model_data import get_model_data
from model.utilities import ModelManager
from utils.defaults import Defaults


class TunableClassifierModel(kt.HyperModel):
    def __init__(
            self,
            project: str,
            max_layers: int,
            max_units: int,
            x_shape: int,
            y_shape: int,
            compile_args: dict[str, Any],
            hidden_args: Optional[dict[str, Any]] = None,
            output_args: Optional[dict[str, Any]] = None,
    ) -> None:
        self.max_layers: int = max_layers
        self.max_units: int = max_units
        self.x_shape: int = x_shape
        self.y_shape: int = y_shape
        self.compile_args: dict[str, Any] = compile_args
        self.hidden_args: dict[str, Any] = hidden_args or {}
        self.output_args: dict[str, Any] = output_args or {}
        super().__init__(name=f'{project}_hyper_model')

    def build(self, hp: kt.HyperParameters, **kwargs) -> keras.Sequential:
        model: keras.Sequential = keras.Sequential()
        model.add(keras.Input(shape=self.x_shape, name='input'))
        for i in range(hp.Int('num_conv2d_layers', 1, self.max_layers)):
            model.add(
                keras.layers.Conv2D(
                    filters=hp.Int(f'filters_{i}', min_value=16, max_value=self.max_units, step=16),
                    kernel_size=3,
                    padding='same',
                    name=f'conv2d_{i}',
                    **self.hidden_args,
                )
            )
            model.add(keras.layers.MaxPooling2D())
        model.add(keras.layers.Dropout(hp.Float(f'dropout_rate', min_value=0.1, max_value=0.5, step=0.1)))
        model.add(keras.layers.Flatten())
        model.add(keras.layers.Dense(hp.Int(f'dense_feat_units', min_value=128, max_value=256, step=128),
                                     name='feature_combination', **self.hidden_args)),
        model.add(keras.layers.Dense(self.y_shape, name='output', **self.output_args))
        model.compile(**self.compile_args)
        return model

    def fit(self, hp: kt.HyperParameters, model: keras.Sequential, *args, **kwargs) -> keras.callbacks.History:
        return model.fit(
            *args,
            verbose=0,
            **kwargs,
        )


def tune_model(project: str, train_ds: tf.data.Dataset, val_ds: tf.data.Dataset, learning_rate: float, seed: int,
               max_layers: int, max_units: int, max_epochs: int, factor: int, iterations: int, save_dir: Path) \
        -> keras.Sequential:
    tunable_model = TunableClassifierModel(
        project=project,
        max_layers=max_layers,
        max_units=max_units,
        **get_model_data(project, learning_rate=learning_rate, seed=seed)
    )
    tuner: kt.Hyperband = kt.Hyperband(
        hypermodel=tunable_model,
        objective='val_accuracy',
        max_epochs=max_epochs,
        factor=factor,
        hyperband_iterations=iterations,
        overwrite=True,
        directory='./server/models/tuned',
        project_name='kt_test'
    )
    tuner.search_space_summary()
    stop_early = keras.callbacks.EarlyStopping(monitor='val_accuracy', mode='max', patience=5)
    tuner.search(train_ds, validation_data=val_ds, epochs=20, callbacks=[stop_early])
    best_hps: kt.HyperParameters = tuner.get_best_hyperparameters(num_trials=1)[0]
    for param, value in best_hps.values.items():
        print(f'{param} -> {value}')

    best_hp = tuner.get_best_hyperparameters()[0]
    tuned_model = tunable_model.build(best_hp)
    keras.models.save_model(tuned_model, save_dir)
    return tuned_model


def main():
    args = parser.parse_args()
    project = args.project.replace("'", "")
    ds_loader = TfDatasetMaker.from_project(project)
    manager = ModelManager(project, False)
    tune_model(
        project=project,
        train_ds=ds_loader.train_ds,
        val_ds=ds_loader.val_ds,
        learning_rate=args.learning_rate,
        seed=args.seed,
        max_layers=args.max_layers,
        max_units=args.max_units,
        max_epochs=args.max_epochs,
        factor=args.factor,
        iterations=args.iterations,
        save_dir=manager.tuned_model_dir
    ).summary()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(prog='python -m utils.model_tuner')
    parser.add_argument(
        '-p', '--project',
        type=str,
        required=True,
        choices=Projects.all(),
        metavar='NAME',
        help='the project name (available: %(choices)s)'
    )
    hb_args = parser.add_argument_group('Hyperband tuning algorithm options')
    hb_args.add_argument(
        '-l', '--max_layers',
        type=int,
        default=Defaults.TUNE_MAX_LAYERS,
        help='maximum allowable number of layers in the model (default: %(default)s)'
    )
    hb_args.add_argument(
        '-u', '--max_units',
        type=int,
        default=Defaults.TUNE_MAX_UNITS,
        help='maximum allowable number of units per model layer (default: %(default)s)'
    )
    hb_args.add_argument(
        '-e', '--max_epochs',
        type=int,
        default=Defaults.TUNE_MAX_EPOCHS,
        help='the maximum number of epochs to train one model while tuning (default: %(default)s)'
    )
    hb_args.add_argument(
        '-f', '--factor',
        type=int,
        default=Defaults.TUNE_FACTOR,
        help='the reduction factor for the Hyperband tuning algorithm (default: %(default)s)'
    )
    hb_args.add_argument(
        '-i', '--iterations',
        type=int,
        default=Defaults.TUNE_ITERATIONS,
        help='number of times to iterate over the full Hyperband algorithm (default: %(default)s)'
    )
    parser.add_argument(
        '--learning_rate',
        type=float,
        metavar='RATE',
        default=Defaults.ML_LEARNING_RATE,
        help='the model\'s optimizer learning rate (default: %(default)s)'
    )
    parser.add_argument(
        '--seed',
        type=int,
        default=Defaults.ML_SEED,
        help='an integer to use for seeding RNG operations (default: %(default)s)'
    )

    main()
