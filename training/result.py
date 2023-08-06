import argparse
import csv
import json
from pathlib import Path
from typing import Final, Optional, Union

from flwr.server import History
from matplotlib import pyplot as plt
from pandas import read_csv, DataFrame

from utils.constants import (
    MET_ACCURACY,
    MET_AVG_CPU_TIME,
    MET_AVG_BYTES_DOWN,
    MET_AVG_BYTES_UP,
    MET_AVG_RAM_USAGE,
    MET_ROUND_DURATION
)
from utils.custom_types import Filepath

_LABEL_TRAINING_ROUND: Final[str] = 'Training Round'
_LABEL_LOSS: Final[str] = 'Loss'
_LABEL_ACCURACY: Final[str] = 'Accuracy'
_LABEL_AVG_CPU_TIME: Final[str] = 'Avg. CPU Time'
_LABEL_AVG_BYTES_DOWN: Final[str] = 'Avg. Bytes Down'
_LABEL_AVG_BYTES_UP: Final[str] = 'Avg. Bytes Up'
_LABEL_AVG_RAM_USAGE: Final[str] = 'Avg. RAM Usage'
_LABEL_ROUND_DURATION: Final[str] = 'Round Duration'
_LABEL_SERVER_LOSS: Final[str] = 'Server Loss'
_LABEL_SERVER_ACCURACY: Final[str] = 'Server Accuracy'


class TrainingResult:

    def __init__(self, results_data: Union[History, DataFrame], num_clients: int, identifier: int,
                 results_dir: Filepath) -> None:
        if isinstance(results_data, History):
            results_dict = TrainingResult._init_from_history(results_data)
        elif isinstance(results_data, DataFrame):
            results_dict = TrainingResult._init_from_dataframe(results_data)
        else:
            raise TypeError('`results_data` must be a flwr.server.History or a pandas.DataFrame')
        self._rounds: list[int] = results_dict[_LABEL_TRAINING_ROUND]
        self._loss: list[float] = results_dict[_LABEL_LOSS]
        self._accuracy: list[float] = results_dict[_LABEL_ACCURACY]
        self._avg_cpu_time: list[float] = results_dict[_LABEL_AVG_CPU_TIME]
        self._avg_bytes_down: list[float] = results_dict[_LABEL_AVG_BYTES_DOWN]
        self._avg_bytes_up: list[float] = results_dict[_LABEL_AVG_BYTES_UP]
        self._avg_ram_usage: list[float] = results_dict[_LABEL_AVG_RAM_USAGE]
        self._round_duration: list[float] = results_dict[_LABEL_ROUND_DURATION]
        self._server_loss: Optional[list[float]] = results_dict.get(_LABEL_SERVER_LOSS, None)
        self._server_accuracy: Optional[list[float]] = results_dict.get(_LABEL_SERVER_ACCURACY, None)
        self._no_server_metrics: bool = self._server_loss is None and self._server_accuracy is None
        self.num_clients: int = num_clients
        self.identifier: int = identifier
        self.results_dir: Path = Path(results_dir)
        if not self.results_dir.exists():
            self.results_dir.mkdir(parents=True)
        self._iter_idx: int = 0
        self.saved_csv: Optional[Path] = None

    def __getitem__(self, item) -> list[float]:
        return self.__dict__[item]

    @property
    def num_rounds(self) -> int:
        return self._rounds[-1]

    def __iter__(self) -> 'TrainingResult':
        return self

    def __next__(self) -> tuple[Union[int, float], ...]:
        if self._iter_idx <= self.num_rounds:
            self._iter_idx += 1
            metrics: list[Union[int, float]] = [
                self._rounds[self._iter_idx - 1],
                self._loss[self._iter_idx - 1],
                self._accuracy[self._iter_idx - 1],
                self._avg_cpu_time[self._iter_idx - 1],
                self._avg_bytes_down[self._iter_idx - 1],
                self._avg_bytes_up[self._iter_idx - 1],
                self._avg_ram_usage[self._iter_idx - 1],
                self._round_duration[self._iter_idx - 1],
            ]
            if not self._no_server_metrics:
                metrics.extend([
                    self._server_loss[self._iter_idx - 1],
                    self._server_accuracy[self._iter_idx - 1],
                ])
            return tuple(metrics)
        self._iter_idx: int = 0
        raise StopIteration

    def __repr__(self) -> str:
        res_str = '\nFEDERATED TRAINING RESULTS\n'
        res_str += '-' * 82 + '\n'
        res_str += f'\nTraining info:\n\tNum Rounds = {self.num_rounds}\t\tNum Clients = {self.num_clients}\n'
        try:
            pre_loss: float = self._loss[0]
            pre_acc: float = self._accuracy[0]
            res_str += f'\nBefore training:'
            if self._no_server_metrics:
                res_str += f'\n\tLoss = {pre_loss:.6f}\t\tAccuracy = {pre_acc:.6f}\n'
            else:
                pre_s_loss: float = self._server_loss[0]
                pre_s_acc: float = self._server_accuracy[0]
                res_str += f'\n\tClient loss = {pre_loss:.6f}\tClient accuracy = {pre_acc:.6f}'
                res_str += f'\n\tServer loss = {pre_s_loss:.6f}\tServer accuracy = {pre_s_acc:.6f}\n'
        except ValueError as e:
            res_str += f'\nBefore training:\n\t{e}\n'
        res_str += f'\nTraining ({self.num_rounds} rounds):\n'
        if self._no_server_metrics:
            res_str += f'Round\tLoss\t\tAccuracy\n'
        else:
            res_str += f'Round\tClient Loss\tClient Accuracy\t\tServer Loss\tServer Accuracy\n'
        res_str += '-' * 82 + '\n'
        for metrics in self:
            t_round, loss, acc = metrics[0], metrics[1], metrics[2]
            if t_round == 0:
                continue
            res_str += f'{t_round:03d}\t{loss:.6f}\t{acc:.6f}'
            if not self._no_server_metrics:
                s_loss, s_acc = metrics[-2], metrics[-1]
                res_str += f'\t\t{s_loss:.6f}\t{s_acc:.6f}\n'
            else:
                res_str += '\n'
        res_str += '-' * 82
        return res_str

    def _load_train_params(self):
        try:
            with open(self.results_dir / 'train_params.json') as f:
                params: dict[str, str | int | float] = json.load(f)['train_params']
        except FileNotFoundError:
            params = {}
        return params

    @staticmethod
    def _init_from_history(results_data: History) -> dict[str, list[int | float]]:
        results_dict: dict[str, list[int | float]] = {
            _LABEL_TRAINING_ROUND: [r[0] for r in results_data.losses_distributed],
            _LABEL_LOSS: [r[1] for r in results_data.losses_distributed],
            _LABEL_ACCURACY: [r[1] for r in results_data.metrics_distributed[MET_ACCURACY]],
            _LABEL_AVG_CPU_TIME: [r[1] for r in results_data.metrics_distributed[MET_AVG_CPU_TIME]],
            _LABEL_AVG_BYTES_DOWN: [r[1] for r in results_data.metrics_distributed[MET_AVG_BYTES_DOWN]],
            _LABEL_AVG_BYTES_UP: [r[1] for r in results_data.metrics_distributed[MET_AVG_BYTES_UP]],
            _LABEL_AVG_RAM_USAGE: [r[1] for r in results_data.metrics_distributed[MET_AVG_RAM_USAGE]],
            _LABEL_ROUND_DURATION: [r[1] for r in results_data.metrics_distributed[MET_ROUND_DURATION]],
        }
        if server_loss := results_data.losses_centralized:
            results_dict[_LABEL_SERVER_LOSS] = [r[1] for r in server_loss]
        if (server_acc := results_data.metrics_centralized.get(MET_ACCURACY, None)) is not None:
            results_dict[_LABEL_SERVER_ACCURACY] = [r[1] for r in server_acc]
        return results_dict

    @staticmethod
    def _init_from_dataframe(results_data: DataFrame) -> dict[str, list[int | float]]:
        results_data = results_data.rename(columns={
            'Server Round': _LABEL_TRAINING_ROUND,
            'Client Accuracy': _LABEL_ACCURACY,
            'Client Loss': _LABEL_LOSS
        })
        return results_data.to_dict(orient='list')

    def plot(self) -> None:

        def _annotate_final_metrics(final_vals_: list[list[Union[float, str]]]):
            for ann_data in final_vals_:
                value, y_offset, color = ann_data
                plt.annotate(
                    f'{value:.4f}',
                    xy=(1, value),
                    xytext=(1.01, value + y_offset),
                    xycoords=('axes fraction', 'data'),
                    color=color,
                    size=10
                )

        fig: plt.Figure = plt.figure(figsize=(8, 6))
        plot_data = {
            _LABEL_TRAINING_ROUND: self._rounds,
        }
        if self._no_server_metrics:
            plot_data[_LABEL_LOSS] = self._loss
            plot_data[_LABEL_ACCURACY] = self._accuracy
            line_loss: plt.Line2D = plt.plot(_LABEL_TRAINING_ROUND, _LABEL_LOSS, data=plot_data, color='#2ca02c')[0]
            line_acc: plt.Line2D = plt.plot(_LABEL_TRAINING_ROUND, _LABEL_ACCURACY, data=plot_data, color='#d62728')[0]
            final_vals: list[list[Union[str, float]]] = [
                [line_loss.get_data(orig=True)[1][-1], 0.0, line_loss.get_color()],
                [line_acc.get_data(orig=True)[1][-1], 0.0, line_acc.get_color()],
            ]
        else:
            plot_data['Client Loss'] = self._loss
            plot_data['Client Accuracy'] = self._accuracy
            plot_data[_LABEL_SERVER_LOSS] = self._server_loss
            plot_data[_LABEL_SERVER_ACCURACY] = self._server_accuracy
            line_loss: plt.Line2D = plt.plot(_LABEL_TRAINING_ROUND, 'Client Loss', data=plot_data,
                                             color='#2ca02c')[0]
            line_acc: plt.Line2D = plt.plot(_LABEL_TRAINING_ROUND, 'Client Accuracy', data=plot_data,
                                            color='#d62728')[0]
            line_s_loss: plt.Line2D = plt.plot(_LABEL_TRAINING_ROUND, _LABEL_SERVER_LOSS, data=plot_data,
                                               color='#1f77b4')[0]
            line_s_acc: plt.Line2D = plt.plot(_LABEL_TRAINING_ROUND, _LABEL_SERVER_ACCURACY, data=plot_data,
                                              color='#ff7f0e')[0]
            final_loss, final_s_loss = line_loss.get_data(orig=True)[1][-1], line_s_loss.get_data(orig=True)[1][-1]
            final_acc, final_s_acc = line_acc.get_data(orig=True)[1][-1], line_s_acc.get_data(orig=True)[1][-1]
            loss_y_offset: float = 0.04 if final_loss > final_s_loss else -0.04
            acc_y_offset: float = 0.04 if final_acc > final_s_acc else -0.04
            final_vals: list[list[Union[str, float]]] = [
                [line_loss.get_data(orig=True)[1][-1], loss_y_offset, line_loss.get_color()],
                [line_acc.get_data(orig=True)[1][-1], acc_y_offset, line_acc.get_color()],
                [line_s_loss.get_data(orig=True)[1][-1], -loss_y_offset, line_s_loss.get_color()],
                [line_s_acc.get_data(orig=True)[1][-1], -acc_y_offset, line_s_acc.get_color()],
            ]
        train_params = self._load_train_params()
        _annotate_final_metrics(final_vals)
        plt.xlim(plot_data[_LABEL_TRAINING_ROUND][0], plot_data[_LABEL_TRAINING_ROUND][-1])
        plt.xlabel(_LABEL_TRAINING_ROUND)
        plt.ylim(bottom=0)
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
            header: list[str] = [
                _LABEL_TRAINING_ROUND, 
                _LABEL_LOSS, 
                _LABEL_ACCURACY,
                _LABEL_AVG_CPU_TIME, 
                _LABEL_AVG_BYTES_DOWN, 
                _LABEL_AVG_BYTES_UP,
                _LABEL_AVG_RAM_USAGE,
                _LABEL_ROUND_DURATION
            ]
            if not self._no_server_metrics:
                header.extend([_LABEL_SERVER_LOSS, _LABEL_SERVER_ACCURACY])
            writer.writerow(header)
            for result in self:
                writer.writerow(result)
        self.saved_csv = csv_file
        return self.saved_csv.as_posix()

    @classmethod
    def from_saved_results(cls, results_dir: Filepath) -> 'TrainingResult':
        results_dir: Path = Path(results_dir)
        file_info: list[str] = results_dir.name.split('_')
        return cls(read_csv(results_dir / 'results_data.csv'), int(file_info[2]), int(file_info[3]), results_dir)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(prog='python -m training.result')
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
