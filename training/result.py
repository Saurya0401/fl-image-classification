from flwr.server import History
from matplotlib import pyplot as plt
from pandas import read_csv, DataFrame

import argparse
import csv
import json
from pathlib import Path
from typing import Optional, Union

from utils.constants import *
from utils.custom_types import Filepath


class TrainingResult:

    METRIC_LABELS: list[str] = [
        'Server Round',
        'Server Loss',
        'Server Accuracy',
        'Client Loss',
        'Client Accuracy',
        'Avg. CPU Time',
        'Avg. Bytes Down',
        'Avg. Bytes Up',
        'Avg. RAM Usage',
        'Round Duration',
    ]

    def __init__(self, results_data: Union[History, DataFrame], num_clients: int, identifier: int,
                 results_dir: Filepath) -> None:
        if isinstance(results_data, History):
            self._init_from_history(results_data)
        elif isinstance(results_data, DataFrame):
            self._init_from_dataframe(results_data)
        else:
            raise TypeError('`results_data` must be a flwr.server.History or a pandas.DataFrame')
        self.num_clients: int = num_clients
        self.identifier: int = identifier
        self.results_dir: Path = Path(results_dir)
        if not self.results_dir.exists():
            self.results_dir.mkdir(parents=True)
        self._iter_idx: int = 0
        self.saved_csv: Optional[Path] = None

    def __getitem__(self, item) -> list[float]:
        return self._results_dict[item]

    @property
    def pre_train_client_accuracy(self) -> float:
        return self.client_accuracies[0]

    @property
    def pre_train_client_loss(self) -> float:
        return self.client_losses[0]

    @property
    def pre_train_server_loss(self) -> float:
        return self.server_losses[0]

    @property
    def pre_train_server_accuracy(self) -> float:
        return self.server_accuracies[0]

    @property
    def num_rounds(self) -> int:
        return self.server_rounds[-1]

    @property
    def server_rounds(self) -> list[int]:
        return self._results_dict['Server Round']

    @property
    def server_losses(self) -> list[float]:
        return self._results_dict['Server Loss']

    @property
    def server_accuracies(self) -> list[float]:
        return self._results_dict['Server Accuracy']

    @property
    def client_losses(self) -> list[float]:
        return self._results_dict['Client Loss']

    @property
    def client_accuracies(self) -> list[float]:
        return self._results_dict['Client Accuracy']

    @property
    def avg_cpu_times(self) -> list[float]:
        return self._results_dict['Avg. CPU Time']

    @property
    def avg_bytes_downs(self) -> list[float]:
        return self._results_dict['Avg. Bytes Down']

    @property
    def avg_bytes_ups(self) -> list[float]:
        return self._results_dict['Avg. Bytes Up']

    @property
    def avg_ram_usages(self) -> list[float]:
        return self._results_dict['Avg. RAM Usage']

    @property
    def round_durations(self) -> list[float]:
        return self._results_dict['Round Duration']

    def __iter__(self) -> 'TrainingResult':
        return self

    def __next__(self) -> tuple[int, float, float, float, float, float, float, float, float, float]:
        if self._iter_idx <= self.num_rounds:
            self._iter_idx += 1
            return (
                self.server_rounds[self._iter_idx - 1],
                self.server_losses[self._iter_idx - 1],
                self.server_accuracies[self._iter_idx - 1],
                self.client_losses[self._iter_idx - 1],
                self.client_accuracies[self._iter_idx - 1],
                self.avg_cpu_times[self._iter_idx - 1],
                self.avg_bytes_downs[self._iter_idx - 1],
                self.avg_bytes_ups[self._iter_idx - 1],
                self.avg_ram_usages[self._iter_idx - 1],
                self.round_durations[self._iter_idx - 1],
            )
        self._iter_idx: int = 0
        raise StopIteration

    def __repr__(self) -> str:
        res_str = '\nFEDERATED TRAINING RESULTS\n'
        res_str += '-' * 82 + '\n'
        res_str += f'\nTraining info:\n\tNum Rounds = {self.num_rounds}\t\tNum Clients = {self.num_clients}\n'
        try:
            pre_f_loss: float = self.pre_train_client_loss
            pre_f_acc: float = self.pre_train_client_accuracy
            pre_s_loss: float = self.pre_train_server_loss
            pre_s_acc: float = self.pre_train_server_accuracy
            res_str += f'\nBefore training:\n\tClient loss = {pre_f_loss:.8f}\tClient accuracy = {pre_f_acc:.8f}'
            res_str += f'\n\tServer loss = {pre_s_loss:.8f}\tServer accuracy = {pre_s_acc:.8f}\n'
        except ValueError as e:
            res_str += f'\nBefore training:\n\t{e}\n'
        res_str += f'\nTraining ({self.num_rounds} rounds):\n'
        res_str += f'Round\tServer Loss\tServer Accuracy\t\tClient Loss\tClient Accuracy\n'
        res_str += '-' * 82 + '\n'
        for s_round, s_loss, s_acc, f_loss, f_acc, _, _, _, _, _ in self:
            if s_round == 0:
                continue
            res_str += f'{s_round:03d}\t{s_loss:.8f}\t{s_acc:.8f}\t\t{f_loss:.8f}\t{f_acc:.8f}\n'
        res_str += '-' * 82
        return res_str

    def _load_train_params(self):
        try:
            with open(self.results_dir / 'train_params.json') as f:
                params: dict[str, str | int | float] = json.load(f)['train_params']
        except FileNotFoundError:
            params = {}
        return params

    def _init_from_history(self, results_data: History):
        self._results_dict: dict[str, list[int | float]] = {
            metric: values for metric, values in zip(TrainingResult.METRIC_LABELS, [
                [r[0] for r in results_data.losses_centralized],
                [r[1] for r in results_data.losses_centralized],
                [r[1] for r in results_data.metrics_centralized[MET_ACCURACY]],
                [r[1] for r in results_data.losses_distributed],
                [r[1] for r in results_data.metrics_distributed[MET_ACCURACY]],
                [r[1] for r in results_data.metrics_distributed[MET_AVG_CPU_TIME]],
                [r[1] for r in results_data.metrics_distributed[MET_AVG_BYTES_DOWN]],
                [r[1] for r in results_data.metrics_distributed[MET_AVG_BYTES_UP]],
                [r[1] for r in results_data.metrics_distributed[MET_AVG_RAM_USAGE]],
                [r[1] for r in results_data.metrics_distributed[MET_ROUND_DURATION]],
            ])
        }

    def _init_from_dataframe(self, results_data: DataFrame):
        results_data = results_data.rename(columns={
            'Federated Accuracy': 'Client Accuracy',
            'Federated Loss': 'Client Loss'
        })
        self._results_dict: dict[str, list[int | float]] = results_data.to_dict(orient='list')

    def plot(self) -> None:

        def _annotate_final_metrics(final_vals: dict[str, list[Union[float, str]]]):
            if final_vals['c_loss'][0] < final_vals['s_loss'][0]:
                final_vals['c_loss'][1], final_vals['s_loss'][1] = final_vals['s_loss'][1], final_vals['c_loss'][1]
            if final_vals['c_acc'][0] > final_vals['s_acc'][0]:
                final_vals['c_acc'][1], final_vals['s_acc'][1] = final_vals['s_acc'][1], final_vals['c_acc'][1]
            for value, y_pos, color in final_vals.values():
                plt.annotate(
                    f'{value:.6f}',
                    xy=(1, value),
                    xytext=(1.01, y_pos),
                    xycoords=('axes fraction', 'data'),
                    textcoords='axes fraction',
                    color=color,
                    size=10
                )

        plot_data = self._results_dict
        train_params = self._load_train_params()
        fig: plt.Figure = plt.figure(figsize=(8, 6))
        line_s_loss: plt.Line2D = plt.plot('Server Round', 'Server Loss', data=plot_data)[0]
        line_s_acc: plt.Line2D = plt.plot('Server Round', 'Server Accuracy', data=plot_data)[0]
        line_c_loss: plt.Line2D = plt.plot('Server Round', 'Client Loss', data=plot_data)[0]
        line_c_acc: plt.Line2D = plt.plot('Server Round', 'Client Accuracy', data=plot_data)[0]
        _annotate_final_metrics({
            's_acc': [line_s_acc.get_data(orig=True)[1][-1], 0.90, line_s_acc.get_color()],
            'c_acc': [line_c_acc.get_data(orig=True)[1][-1], 0.86, line_c_acc.get_color()],
            'c_loss': [line_c_loss.get_data(orig=True)[1][-1], 0.04, line_c_loss.get_color()],
            's_loss': [line_s_loss.get_data(orig=True)[1][-1], 0.00, line_s_loss.get_color()]
        })
        plt.xlim(plot_data['Server Round'][0], plot_data['Server Round'][-1])
        plt.ylim(0.0, 1.1)
        plt.xlabel('Training Round')
        plt.legend(title="Metrics:")
        title: str = f'Federated Training Results\n({self.num_rounds} rounds, {self.num_clients} clients'
        if (client_epochs := train_params.get('client_epochs', None)) is not None:
            title += f', {client_epochs} epochs/round'
        if (untrained := train_params.get('untrained', None)) is not None:
            title += f', {"UNTRAINED" if untrained else "TRAINED"} models'
        title += ')'
        plt.title(title)
        plt.tight_layout()
        plt.grid()
        plt.show()
        fig.savefig(str(self.results_dir / 'results_plot.png'))

    def to_csv(self, save_dir: Optional[str] = None) -> str:
        save_dir_: Path = Path(save_dir) if save_dir else self.results_dir
        csv_file: Path = save_dir_ / 'results_data.csv'
        with open(csv_file, mode='w', newline='') as file:
            writer = csv.writer(file, delimiter=',', quotechar="'", quoting=csv.QUOTE_MINIMAL)
            writer.writerow(TrainingResult.METRIC_LABELS)
            for result in self:
                writer.writerow(result)
        self.saved_csv = csv_file
        return self.saved_csv.as_posix()

    @classmethod
    def from_saved_results(cls, results_dir: Filepath) -> 'TrainingResult':
        results_dir: Path = Path(results_dir)
        file_info: list[str] = results_dir.name.split('_')
        return cls(read_csv(results_dir / 'results_data.csv'), int(file_info[3]), int(file_info[4]), results_dir)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(prog='python -m utils.training_utils')
    parser.add_argument(
        '-f', '--from_saved',
        type=str,
        metavar='DIR',
        help='Load results from saved results directory'
    )
    args = parser.parse_args()

    if args.from_saved is not None:
        results = TrainingResult.from_saved_results(args.from_saved)
        print(results)
        results.plot()
