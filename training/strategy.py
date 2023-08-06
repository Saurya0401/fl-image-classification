import flwr.server.strategy
from flwr.common import (
    FitIns,
    NDArrays,
    Parameters,
    Scalar,
    ndarrays_to_parameters,
)
from flwr.common.logger import log
from flwr.server.client_manager import ClientManager
from flwr.server.client_proxy import ClientProxy

from logging import INFO
from random import randint
from typing import Callable, Optional, Union

from utils.constants import (
    MET_ACCURACY,
    MET_CPU_TIME,
    MET_BYTES_DOWN,
    MET_BYTES_UP,
    MET_RAM_USAGE
)


def aggregate_fit_metrics(metrics: list[tuple[int, dict[str, float]]]) -> dict[str, float]:
    """
    Aggregates the metrics returned by clients after local training in each federated training round.

    Currently, these metrics consist of the cpu time, bytes uploaded, and bytes downloaded.
    :param metrics: the metrics returned by the client
    :return: the aggregated metrics
    """

    cpu_times: list[float] = [m.get(MET_CPU_TIME, None) for _, m in metrics]
    bytes_up: list[float] = [m.get(MET_BYTES_UP, None) for _, m in metrics]
    bytes_down: list[float] = [m.get(MET_BYTES_DOWN, None) for _, m in metrics]
    ram_usages: list[float] = [m.get(MET_RAM_USAGE, None) for _, m in metrics]

    return {
        MET_CPU_TIME: sum(cpu_times) / len(cpu_times) if all(cpu_times) else None,
        MET_BYTES_DOWN: sum(bytes_down) / len(bytes_down) if all(bytes_down) else None,
        MET_BYTES_UP: sum(bytes_up) / len(bytes_up) if all(bytes_up) else None,
        MET_RAM_USAGE: sum(ram_usages) / len(ram_usages) if all(ram_usages) else None,
    }


def aggregate_eval_metrics(metrics: list[tuple[int, dict[str, float]]]) -> dict[str, float]:
    """
    Aggregates the metrics returned by clients after local evaluation in each federated training round.

    Currently, these metrics consist of the accuracy, cpu time, bytes uploaded, and bytes downloaded.
    :param metrics: the metrics returned by the client
    :return: the aggregated metrics
    """

    # Weigh accuracy of each client by number of examples used
    accuracies: list[float] = [m.get(MET_ACCURACY, None) * n for n, m in metrics]
    cpu_times: list[float] = [m.get(MET_CPU_TIME, None) for _, m in metrics]
    bytes_down: list[float] = [m.get(MET_BYTES_DOWN, None) for _, m in metrics]
    bytes_up: list[float] = [m.get(MET_BYTES_UP, None) for _, m in metrics]
    ram_usages: list[float] = [m.get(MET_RAM_USAGE, None) for _, m in metrics]
    examples: list[int] = [n for n, _ in metrics]

    # Aggregate and return custom metric
    return {
        MET_ACCURACY: sum(accuracies) / sum(examples) if all(accuracies) else None,
        MET_CPU_TIME: sum(cpu_times) / len(cpu_times) if all(cpu_times) else None,
        MET_BYTES_DOWN: sum(bytes_down) / len(bytes_down) if all(bytes_down) else None,
        MET_BYTES_UP: sum(bytes_up) / len(bytes_up) if all(bytes_up) else None,
        MET_RAM_USAGE: sum(ram_usages) / len(ram_usages) if all(ram_usages) else None,
    }


class FedStrategy(flwr.server.strategy.FedAvg):

    def __init__(
            self,
            fraction_fit: float = 1.0,
            fraction_evaluate: float = 1.0,
            initial_parameters: Optional[NDArrays] = None,
            min_fit_clients: int = 1,
            min_evaluate_clients: int = 1,
            min_available_clients: int = 1,
            evaluate_fn: Optional[Callable] = None,
            client_epochs: Optional[Union[int, str]] = 1,
            epochs_limit: int = 10,
            training_eval_rounds: int = 5,
    ) -> None:
        super().__init__(
            fraction_fit=fraction_fit,
            fraction_evaluate=fraction_evaluate,
            initial_parameters=ndarrays_to_parameters(initial_parameters) if initial_parameters else None,
            min_fit_clients=min_fit_clients,
            min_evaluate_clients=min_evaluate_clients,
            min_available_clients=min_available_clients,
            evaluate_fn=evaluate_fn,
            fit_metrics_aggregation_fn=aggregate_fit_metrics,
            evaluate_metrics_aggregation_fn=aggregate_eval_metrics
        )
        self.client_epochs: Optional[Union[int, str]] = client_epochs
        self.epochs_limit: int = epochs_limit
        self.training_eval_rounds: int = training_eval_rounds
        self.metrics_buffer: list[tuple[float, float]] = []
        self.prev_avg_loss: float = 1000.0
        self.prev_avg_accuracy: float = 0.0
        self.stop_training: bool = False

    def __repr__(self) -> str:
        return 'Federated Strategy\n' + \
            '\n'.join([f'\t{str(k):22} -> {str(v)}' for k, v in self.__dict__.items() if not callable(v)])

    def _get_fit_config(self, server_round: int) -> dict[str, Union[int, float]]:
        config = {'server_round': server_round}
        if self.client_epochs is None:
            return config
        config['local_epochs'] = randint(1, self.epochs_limit) if self.client_epochs == 'random' else self.client_epochs
        return config

    def _add_training_metrics(self, server_round: int, loss: float, accuracy: float):
        if loss is None or accuracy is None:
            return
        if server_round == 0:
            self.prev_avg_loss = loss
            self.prev_avg_accuracy = accuracy
            return
        self.metrics_buffer.append((loss, accuracy))
        if server_round % self.training_eval_rounds == 0:
            self._evaluate_training(server_round)
            self.metrics_buffer.clear()

    def _evaluate_training(self, server_round: int):
        losses, accuracies = zip(*self.metrics_buffer)
        avg_loss: float = sum(losses) / len(losses)
        avg_accuracy: float = sum(accuracies) / len(accuracies)
        loss_dev: float = (avg_loss - self.prev_avg_loss) / self.prev_avg_loss * 100
        accuracy_dev: float = (avg_accuracy - self.prev_avg_accuracy) / self.prev_avg_accuracy * 100
        log(INFO, f'Evaluating training for round {server_round}')
        log(INFO, f'Average Loss     -> current: {avg_loss:.10f}\tprev: {self.prev_avg_loss:.10f} '
                  f'({"+" if loss_dev > 0 else ""}{loss_dev:.5f}%)')
        log(INFO, f'Average Accuracy -> current: {avg_accuracy:.10f}\tprev: {self.prev_avg_accuracy:.10f} '
                  f'({"+" if accuracy_dev > 0 else ""}{accuracy_dev:.5f}%)\n\n')
        if avg_loss >= self.prev_avg_loss and avg_accuracy <= self.prev_avg_accuracy:
            self.stop_training = True
        self.prev_avg_loss = avg_loss
        self.prev_avg_accuracy = avg_accuracy

    def configure_fit(self, server_round: int, parameters: Parameters, client_manager: ClientManager) \
            -> list[tuple[ClientProxy, FitIns]]:
        sample_size, min_num_clients = self.num_fit_clients(client_manager.num_available())
        clients = client_manager.sample(num_clients=sample_size, min_num_clients=min_num_clients)
        return [(client, FitIns(parameters, self._get_fit_config(server_round))) for client in clients]

    def evaluate(self, server_round: int, parameters: Parameters) -> Optional[tuple[float, dict[str, Scalar]]]:
        loss, metrics = super().evaluate(server_round, parameters)
        if self.training_eval_rounds != 0:
            self._add_training_metrics(server_round, loss, metrics[MET_ACCURACY])
        return loss, metrics


if __name__ == '__main__':
    print(FedStrategy())
