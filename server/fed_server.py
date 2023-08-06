import flwr as fl
from flwr.common.logger import log
from flwr.common.parameter import parameters_to_ndarrays
from flwr.server.client_manager import SimpleClientManager
from flwr.server.server import History
from tensorflow import keras

import argparse
import timeit
from logging import INFO, WARN
from pathlib import Path
from time import time
from typing import Any, Callable, Optional

from projects import Projects
from data.loader import DataLoader, TfDatasetMaker
from model.utilities import ModelManager
from training.result import TrainingResult
from training.strategy import FedStrategy
from server.provider import update_basic_model
from utils.constants import (
    MET_ACCURACY,
    MET_AVG_CPU_TIME,
    MET_AVG_BYTES_DOWN,
    MET_AVG_BYTES_UP,
    MET_AVG_RAM_USAGE,
    MET_CPU_TIME,
    MET_BYTES_DOWN,
    MET_BYTES_UP,
    MET_RAM_USAGE,
    MET_ROUND_DURATION
)
from utils.custom_types import Filepath


class FedServer(fl.server.Server):

    def __init__(self, strategy: FedStrategy) -> None:
        super().__init__(client_manager=SimpleClientManager())
        self.strategy = strategy

    def fit(self, num_rounds: int, timeout: Optional[float]) -> History:
        """Run federated averaging for a number of rounds."""
        history = History()

        # Initialize parameters
        log(INFO, 'Initializing global parameters')
        self.parameters = self._get_initial_parameters(timeout=timeout)
        log(INFO, 'Evaluating initial parameters')
        init_res_cen = self.strategy.evaluate(0, parameters=self.parameters)
        if init_res_cen is not None:
            log(
                INFO,
                'Pre-train server metrics (loss, other metrics): %s, %s',
                init_res_cen[0],
                init_res_cen[1],
            )
            history.add_loss_centralized(server_round=0, loss=init_res_cen[0])
            history.add_metrics_centralized(server_round=0, metrics=init_res_cen[1])
        init_res_fed = self.evaluate_round(0, timeout=timeout)
        if init_res_fed:
            log(
                INFO,
                'Pre-train federated metrics (loss, other metrics): %s, %s',
                init_res_fed[0],
                init_res_fed[1],
            )
            history.add_loss_distributed(server_round=0, loss=init_res_fed[0])
            history.add_metrics_distributed(server_round=0, metrics={
                **init_res_fed[1],
                MET_AVG_CPU_TIME: 0.0,
                MET_AVG_BYTES_DOWN: 0.0,
                MET_AVG_BYTES_UP: 0.0,
                MET_AVG_RAM_USAGE: 0.0,
                MET_ROUND_DURATION: 0.0,
            })

        # Run federated learning for num_rounds
        log(INFO, "FL starting")
        start_time = timeit.default_timer()

        for current_round in range(1, num_rounds + 1):
            avg_cpu_time: float = 0.0
            avg_bytes_down: float = 0.0
            avg_bytes_up: float = 0.0
            avg_ram_usage: float = 0.0
            round_start = timeit.default_timer()

            # Train model and replace previous global model
            res_fit = self.fit_round(server_round=current_round, timeout=timeout)
            if res_fit:
                parameters_prime, fit_metrics_fed, _ = res_fit  # fit_metrics_aggregated
                log(INFO, 'fit metrics: %s', str(fit_metrics_fed))
                if parameters_prime:
                    self.parameters = parameters_prime
                avg_cpu_time += self.get_fed_metric(fit_metrics_fed, MET_CPU_TIME, 'Fit CPU time')
                avg_bytes_down += self.get_fed_metric(fit_metrics_fed, MET_BYTES_DOWN, 'Fit Bytes downloaded')
                avg_bytes_up += self.get_fed_metric(fit_metrics_fed, MET_BYTES_UP, 'Fit Bytes uploaded')
                avg_ram_usage += self.get_fed_metric(fit_metrics_fed, MET_RAM_USAGE, 'Fit RAM usage')

            # Evaluate model using strategy implementation
            res_cen = self.strategy.evaluate(current_round, parameters=self.parameters)
            if res_cen is not None:
                loss_cen, metrics_cen = res_cen
                log(
                    INFO,
                    'fit progress: (%s, %s, %s, %s)',
                    current_round,
                    loss_cen,
                    metrics_cen,
                    timeit.default_timer() - start_time,
                )
                history.add_loss_centralized(server_round=current_round, loss=loss_cen)
                history.add_metrics_centralized(
                    server_round=current_round, metrics=metrics_cen
                )

            # Evaluate model on a sample of available clients
            res_fed = self.evaluate_round(server_round=current_round, timeout=timeout)
            round_end = timeit.default_timer()
            if res_fed:
                loss_fed, evaluate_metrics_fed, _ = res_fed
                log(INFO, 'eval metrics: %s', str(evaluate_metrics_fed))
                accuracy = self.get_fed_metric(evaluate_metrics_fed, MET_ACCURACY, 'Accuracy')
                avg_cpu_time += self.get_fed_metric(evaluate_metrics_fed, MET_CPU_TIME, 'Eval CPU time')
                avg_bytes_down += self.get_fed_metric(evaluate_metrics_fed, MET_BYTES_DOWN, 'Eval Bytes downloaded')
                avg_bytes_up += self.get_fed_metric(evaluate_metrics_fed, MET_BYTES_UP, 'Eval Bytes downloaded')
                avg_ram_usage += self.get_fed_metric(evaluate_metrics_fed, MET_RAM_USAGE, 'Eval RAM usage')
                if loss_fed:
                    history.add_loss_distributed(
                        server_round=current_round, loss=loss_fed
                    )
                    history.add_metrics_distributed(
                        server_round=current_round, metrics={
                            MET_ACCURACY: accuracy,
                            MET_AVG_CPU_TIME: avg_cpu_time,
                            MET_AVG_BYTES_DOWN: avg_bytes_down,
                            MET_AVG_BYTES_UP: avg_bytes_up,
                            MET_AVG_RAM_USAGE: avg_ram_usage,
                            MET_ROUND_DURATION: round_end - round_start,
                        }
                    )

            if self.strategy.stop_training:
                log(WARN, 'Stopping training due to degradation in model performance')
                break

        # Bookkeeping
        end_time = timeit.default_timer()
        elapsed = end_time - start_time
        log(INFO, 'FL finished in %s', elapsed)
        return history

    @staticmethod
    def get_fed_metric(metrics: dict[str, int | float], metric_key, metric_name) -> float:
        metric = metrics.get(metric_key, None)
        if metric is None:
            log(WARN, 'Federated metric "%s" not found for training round', metric_name)
            return 0.0
        return metric


def get_evaluate_fn(project: str, model: keras.Sequential) \
        -> Callable[[int, fl.common.NDArrays, dict[str, Any]], tuple[float, dict[str, float]]]:
    """Return an evaluation function for server-side evaluation."""

    data_loader: DataLoader = DataLoader.from_project(project)
    ds_maker: TfDatasetMaker = TfDatasetMaker(data_loader)

    def evaluate(
            server_round: int, parameters: fl.common.NDArrays, config: dict[str, fl.common.Scalar],
    ) -> Optional[tuple[float, dict[str, fl.common.Scalar]]]:
        model.set_weights(parameters)  # Update model with the latest parameters
        loss, accuracy = model.evaluate(ds_maker.test_ds)
        return loss, {"accuracy": accuracy}

    return evaluate


def train(
        address: str,
        rounds: int,
        clients: int,
        client_epochs: int,
        fraction: float,
        identifier: int,
        project: str,
        untrained: bool,
        server_evaluation: bool,
        no_model_update: bool,
        training_eval_rounds: int,
        results_dir: Filepath,
        no_plot: bool
) -> TrainingResult:
    """
    Conducts federated training.
    :param address: the federated server address
    :param rounds: how many rounds of training to conduct
    :param clients: how many clients are participating in the training
    :param client_epochs: local epochs of training per client
    :param fraction: the fraction of clients that must be available during each training round
    :param identifier: a unique identifier for the training
    :param project: the project name
    :param training_eval_rounds: the nth rounds to evaluate training in
    :param untrained: whether to use an initial untrained model
    :param server_evaluation: perform server evaluation during training
    :param results_dir: the folder for storing training training_result
    :param no_model_update: skip updating server model after training
    :param no_plot: do not plot training results
    :return: the result of training
    """

    mod_manager: ModelManager = ModelManager(project, untrained)
    model: keras.Sequential = mod_manager.get_server_model()

    strategy: FedStrategy = FedStrategy(
        fraction_fit=fraction,
        fraction_evaluate=fraction,
        initial_parameters=model.get_weights(),
        min_available_clients=clients,
        evaluate_fn=get_evaluate_fn(project, model) if server_evaluation else None,
        client_epochs=client_epochs,
        training_eval_rounds=training_eval_rounds
    )
    fed_server: FedServer = FedServer(strategy=strategy)
    hist: fl.server.History = fl.server.start_server(
        server_address=address,
        config=fl.server.ServerConfig(num_rounds=rounds),
        server=fed_server,
    )

    training_result: TrainingResult = TrainingResult(hist, clients, identifier, results_dir)
    print(training_result)
    if not no_plot:
        training_result.plot()
    training_result.to_csv()
    print(f'Training results saved to: {training_result.results_dir.absolute()}')

    if not no_model_update:
        server_model: keras.Sequential = mod_manager.get_server_model()
        trained_weights: fl.common.NDArrays = parameters_to_ndarrays(fed_server.parameters)
        server_model.set_weights(trained_weights)
        server_model.save(mod_manager.server_model_dir)
        update_basic_model(project)

    return training_result


def main():
    args = parser.parse_args()
    train(
        project=args.project,
        address=args.address,
        rounds=args.num_rounds,
        clients=args.num_clients,
        client_epochs=args.client_epochs,
        fraction=args.clients_fraction,
        untrained=args.untrained,
        server_evaluation=args.server_evaluation,
        identifier=args.identifier,
        no_model_update=args.no_model_update,
        training_eval_rounds=args.training_eval_rounds,
        results_dir=Path(f'./results/{args.project}_{args.num_rounds}_{args.num_clients}_{args.identifier}'),
        no_plot=args.no_plot,
    )


if __name__ == '__main__':
    # todo: add arg for explicitly including server metrics
    parser = argparse.ArgumentParser(prog='python -m server.fed_server')
    parser.add_argument(
        '-p', '--project',
        type=str,
        required=True,
        choices=Projects.all(),
        metavar='NAME',
        help='the project name (available: %(choices)s)'
    )
    parser.add_argument(
        '-a', '--address',
        type=str,
        required=True,
        help='federated training server address'
    )
    parser.add_argument(
        '-r', '--num_rounds',
        type=int,
        default=50,
        help='number of training rounds (default: %(default)s)'
    )
    parser.add_argument(
        '-c', '--num_clients',
        type=int,
        default=5,
        help='minimum number of clients for training (default: %(default)s)'
    )
    parser.add_argument(
        '-e', '--client_epochs',
        type=int,
        metavar='EPOCHS',
        help='local epochs of training for each client'
    )
    parser.add_argument(
        '-f', '--clients_fraction',
        type=float,
        default=1.0,
        help='fraction of clients used for training if available clients >= minimum clients (default: %(default)s)'
    )
    parser.add_argument(
        '--untrained',
        action='store_true',
        help='use untrained server model for federated training (uses pre-trained model by default)'
    )
    parser.add_argument(
        '-i', '--identifier',
        type=int,
        default=int(time()),
        help='an unique integer identifier used when generating logs and results (default: unix timestamp)'
    )
    parser.add_argument(
        '--server_evaluation',
        action='store_true',
        help='enable server-side evaluation while training'
    )
    parser.add_argument(
        '--no_model_update',
        action='store_true',
        help='skip updating server model after training'
    )
    parser.add_argument(
        '--no_plot',
        action='store_true',
        help='do not plot training results'
    )
    parser.add_argument(
        '--training_eval_rounds',
        type=int,
        metavar='N',
        default=0,
        help='evaluate training performance every Nth round (default: %(default)s)'
    )

    main()
