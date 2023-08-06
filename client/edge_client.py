import flwr as fl
import psutil
from grpc import RpcError
from pympler import asizeof
from tensorflow import keras

import argparse
from os import getpid
from time import sleep
from typing import Optional

from projects import Projects
from utils.constants import (
    MET_LOSS,
    MET_ACCURACY,
    MET_VAL_LOSS,
    MET_VAL_ACC,
    MET_CPU_TIME,
    MET_BYTES_DOWN,
    MET_BYTES_UP,
    MET_RAM_USAGE
)
from data.loader import TfDatasetMaker
from data.preprocessor import Preprocessor


class EdgeClient(fl.client.NumPyClient):

    def __init__(self, model_path: str, is_emulated: bool, ds_maker: Optional[TfDatasetMaker] = None,
                 local_epochs: Optional[int] = None, *args, **kwargs) -> None:
        """
        Represents an Edge Client running on an IoT device.
        :param project: the project name
        :param model_path: the location of the TFLite model
        :param ds_maker: the location of the device's local dataset
        :param local_epochs: how many epochs to train for, if training
        :param args: additional args
        :param kwargs: additional keyword args
        """

        # todo: training error when non-profiled client is used
        super().__init__(*args, **kwargs)
        self.model_path: str = model_path
        self.is_emulated: bool = is_emulated
        self.model: keras.Sequential = keras.models.load_model(self.model_path)
        self.ds_maker: Optional[TfDatasetMaker] = ds_maker
        if self.ds_maker is not None:
            self.local_epochs: int = local_epochs
            self.preprocessor: Preprocessor = self.ds_maker.data_loader.preprocessor

    def predict(self, inp) -> fl.common.NDArray:
        """
        Runs inference on provided inputs.
        :param inp: the provided input attributes
        :return: the predicted output
        """

        if self.ds_maker is None:
            raise ValueError('Cannot infer, local dataset not provided')
        output = self.model.predict(self.preprocessor.normalize_input(inp))
        return self.preprocessor.decode_output(output)

    def get_parameters(self, config) -> fl.common.NDArrays:
        """
        Returns the model weights as a list.
        :param config: an optional config dict
        :return: list of model weights
        """

        return self.model.get_weights()

    def set_parameters(self, parameters) -> None:
        """
        Set custom model weights.
        :param parameters: the model weights to set
        :return: None
        """

        self.model.set_weights(parameters)

    def fit(self, parameters, config) -> tuple[list[fl.common.NDArrays], int, dict[str, fl.common.Scalar]]:
        """
        Updates model weights, then trains model with new weights.
        :param parameters: the provided model weights
        :param config: an optional configuration dict
        :return: the updated weights after training
        """

        if self.ds_maker is None:
            raise ValueError('Cannot train, local dataset not provided')

        self.model.set_weights(parameters)
        epochs: int = config.get('local_epochs', self.local_epochs)
        history = self.model.fit(self.ds_maker.train_ds, epochs=epochs, validation_data=self.ds_maker.val_ds, verbose=2)
        parameters_prime = self.model.get_weights()
        results = {
            MET_LOSS: history.history['loss'][0],
            MET_ACCURACY: history.history['accuracy'][0],
            MET_VAL_LOSS: history.history['val_loss'][0],
            MET_VAL_ACC: history.history['val_accuracy'][0],
        }
        return parameters_prime, self.ds_maker.num_train_samples, results

    def evaluate(self, parameters, config) -> tuple[float, int, dict[str, fl.common.Scalar]]:
        """
        Evaluates model and returns evaluated metrics
        :param parameters: the provided model weights
        :param config: an optional configuration dict
        :return: the evaluated metrics
        """

        if self.ds_maker is None:
            raise ValueError('Cannot evaluate, local dataset not provided')

        self.model.set_weights(parameters)
        loss, accuracy = self.model.evaluate(self.ds_maker.test_ds, verbose=2)
        if not self.is_emulated:
            self.model.save(self.model_path)
        return loss, self.ds_maker.num_test_samples, {MET_ACCURACY: accuracy}


class ProfiledEdgeClient(EdgeClient):

    def __init__(self, model_path: str, is_emulated: bool, ds_maker: Optional[TfDatasetMaker] = None,
                 local_epochs: Optional[int] = None, *args, **kwargs) -> None:
        super().__init__(model_path, is_emulated, ds_maker, local_epochs, *args, **kwargs)
        self.proc = psutil.Process(getpid())

    def fit(self, parameters, config):
        self.set_parameters(parameters)
        cpu_time_s = self.proc.cpu_times()
        ram_usage_s = self.proc.memory_info().rss
        params, num_ex, results = super().fit(parameters, config)
        ram_usage_end = self.proc.memory_info().rss
        cpu_time_end = self.proc.cpu_times()
        return params, num_ex, {
            **results,
            MET_CPU_TIME: ((cpu_time_end.user + cpu_time_end.system) - (cpu_time_s.user + cpu_time_s.system)) * 1000,
            MET_BYTES_DOWN: sum(asizeof.asizesof(parameters, config)),
            MET_BYTES_UP: sum(asizeof.asizesof(params, num_ex, results)),
            MET_RAM_USAGE: ram_usage_end - ram_usage_s,
        }

    def evaluate(self, parameters, config):
        self.set_parameters(parameters)
        ct_start = self.proc.cpu_times()
        ru_start = self.proc.memory_info().rss
        loss, num_ex, metrics = super().evaluate(parameters, config)
        ru_end = self.proc.memory_info().rss
        ct_end = self.proc.cpu_times()
        eval_ct = ((ct_end.user + ct_end.system) - (ct_start.user + ct_start.system)) * 1000
        return loss, num_ex, {
            **metrics,
            MET_CPU_TIME: eval_ct if eval_ct > 0.0 else 0.0,
            MET_BYTES_DOWN: sum(asizeof.asizesof(parameters, config)),
            MET_BYTES_UP: sum(asizeof.asizesof(loss, num_ex, metrics)),
            MET_RAM_USAGE: ru_end - ru_start,
        }


def get_client(project: str, data_path: str, epochs: int, untrained: bool, is_emulated: bool, no_profile: bool) \
        -> EdgeClient:
    model_path: str = f'./client/models/{project}_untrained' if untrained \
        else f'./client/models/{project}_trained'
    ds_maker: TfDatasetMaker = TfDatasetMaker.from_project(project, data_path=data_path)
    edge_client = EdgeClient(model_path, is_emulated, ds_maker, epochs) if no_profile \
        else ProfiledEdgeClient(model_path, is_emulated, ds_maker, epochs)
    return edge_client


def run_client(client, server, retry_limit) -> None:
    cnxn_attempts: int = 0
    while True:
        try:
            fl.client.start_numpy_client(server_address=server, client=client)
        except RpcError:
            if cnxn_attempts < retry_limit:
                cnxn_attempts += 1
                print(f'\nError connecting to server, retrying ... attempt {cnxn_attempts}/{retry_limit}')
                sleep(5)
            else:
                print('\nError: Could not connect to server (max retry attempts exceeded)')
                break
        else:
            break


def main():
    args = parser.parse_args()
    edge_client = get_client(
        project=args.project,
        data_path=args.data_path,
        epochs=args.epochs,
        untrained=args.untrained,
        is_emulated=args.emulated,
        no_profile=args.no_profile,
    )
    run_client(edge_client, args.server, args.retry_limit)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Federated Training Client')
    parser.add_argument(
        '-p', '--project',
        type=str,
        required=True,
        choices=Projects.all(),
        metavar='NAME',
        help='the project name (available: %(choices)s)'
    )
    parser.add_argument(
        '-d', '--data_path',
        type=str,
        required=True,
        metavar='PATH',
        help='path to client dataset'
    )
    parser.add_argument(
        '-e', '--epochs',
        type=int,
        default=1,
        help='number of local training epochs during each training round (default: %(default)s)'
    )
    parser.add_argument(
        '-s', '--server',
        type=str,
        metavar='ADDRESS',
        required=True,
        help='federated training server address'
    )
    parser.add_argument(
        '-r', '--retry_limit',
        type=int,
        default=5,
        help='server connection retry attempts limit (default: %(default)s)'
    )
    parser.add_argument(
        '--untrained',
        action='store_true',
        help='use untrained client model for federated training (model is pre-trained by default)'
    )
    parser.add_argument(
        '--emulated',
        action='store_true',
        help='use emulated client (trained model is not saved)'
    )
    parser.add_argument(
        '--no_profile',
        action='store_true',
        help='turn off client profiling while training'
    )

    main()
