import argparse
import json
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
from random import randint
from time import time
from typing import Optional

from projects import Projects
from client.runner import client_runner
from server.fed_server import train
from utils.custom_types import Filepath


class _ClientDataPool:

    def __init__(self, data_dir: Filepath):
        self.data_dir: Path = Path(data_dir)
        self.client_data_paths: tuple[Path] = tuple(self.data_dir.glob('data_batch_[0-9]'))
        self.num_client_data: int = len(self.client_data_paths)
        self.curr_data_idx: int = 0

    def get_client_data_path(self) -> str:
        if self.curr_data_idx < self.num_client_data:
            client_data_path: Path = self.client_data_paths[self.curr_data_idx]
            self.curr_data_idx += 1
        else:
            client_data_path: Path = self.client_data_paths[randint(0, self.num_client_data - 1)]
        print(client_data_path)
        return client_data_path.as_posix()


class Session:

    def __init__(self,
                 project: str,
                 address: str,
                 num_rounds: int,
                 num_clients: int,
                 client_epochs: int,
                 clients_fraction: float,
                 untrained: bool,
                 identifier: int,
                 server_evaluation: bool,
                 no_model_update: bool,
                 no_plot: bool,
                 training_eval_rounds: int = 0,
                 alt_address: Optional[str] = None
                 ) -> None:

        self.project: str = project
        self.address: str = address
        self.num_rounds: int = num_rounds
        self.num_clients: int = num_clients
        self.client_epochs: int = client_epochs
        self.clients_fraction: float = clients_fraction
        self.untrained: bool = untrained
        self.identifier: int = identifier
        self.server_evaluation: bool = server_evaluation
        self.no_model_update: bool = no_model_update
        self.no_plot: bool = no_plot
        self.training_eval_rounds: int = training_eval_rounds
        self.alt_address: Optional[str] = alt_address
        self.results_dir: Path = Path(f'results/{self.project}_{self.num_rounds}_{self.num_clients}_{self.identifier}')
        self._client_data_pool: _ClientDataPool = _ClientDataPool(Projects.get_project_spec(self.project).data_dir)
        self._client_retry_lim: int = 5

    def _init_dirs(self):
        if not self.results_dir.exists():
            self.results_dir.mkdir(parents=True)

    def _start_client(self, data_path: str):
        client_runner(
            server=self.alt_address or self.address,
            project=self.project,
            data_path=data_path,
            epochs=self.client_epochs,
            untrained=self.untrained,
            retry_limit=self._client_retry_lim,
        )

    def _start_server(self):
        train(
            address=self.address,
            rounds=self.num_rounds,
            clients=self.num_clients,
            client_epochs=self.client_epochs,
            fraction=self.clients_fraction,
            identifier=self.identifier,
            server_evaluation=self.server_evaluation,
            no_model_update=self.no_model_update,
            no_plot=self.no_plot,
            project=self.project,
            untrained=self.untrained,
            training_eval_rounds=self.training_eval_rounds,
            results_dir=self.results_dir,
        )

    def start(self):
        print(f'Starting federated training: \n\t{self.num_rounds} rounds\n\t{self.num_clients} clients'
              f'\n\t{self.client_epochs} epochs/round\n\t{"UNTRAINED" if self.untrained else "TRAINED"} models\n')
        train_params = {
            'train_params': {k: v if isinstance(v, (int, float)) else str(v)
                             for k, v in self.__dict__.items() if not k.startswith('_') and not callable(v)}
        }
        self._init_dirs()
        with open(self.results_dir / 'train_params.json', 'w') as f:
            f.write(json.dumps(train_params))

        with ProcessPoolExecutor(max_workers=61) as executor:
            executor.submit(self._start_server)
            for i in range(int(self.num_clients)):
                executor.submit(self._start_client, data_path=self._client_data_pool.get_client_data_path())


def main() -> None:
    args: argparse.Namespace = parser.parse_args()
    session: Session = Session(
        project=args.project,
        address=args.address,
        num_rounds=args.num_rounds,
        num_clients=args.num_clients,
        client_epochs=args.client_epochs,
        clients_fraction=args.clients_fraction,
        untrained=args.untrained,
        identifier=args.identifier,
        server_evaluation=args.server_evaluation,
        no_model_update=args.no_model_update,
        no_plot=args.no_plot,
        training_eval_rounds=args.training_eval_rounds,
        alt_address=args.alt_address,
    )
    session.start()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(prog='python -m training.session')
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
        default='localhost:8080',
        help='federated training server address (default: %(default)s)'
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
        default=1,
        help='number of client train epochs during each training round (default: %(default)s)'
    )
    parser.add_argument(
        '-f', '--clients_fraction',
        type=float,
        metavar='FRACTION',
        default=1.0,
        help='fraction of clients used for training if available clients >= minimum clients (default: %(default)s)'
    )
    parser.add_argument(
        '--untrained',
        action='store_true',
        help='use untrained server and client models for federated training (models are pre-trained by default)'
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
        help='do not update server model after training'
    )
    parser.add_argument(
        '--no_plot',
        action='store_true',
        help='do not plot training results'
    )
    parser.add_argument(
        '--alt_address',
        type=str,
        metavar='ADDRESS',
        default=None,
        help='alternate server address to pass to clients (default: follows --address)'
    )
    parser.add_argument(
        '--training_eval_rounds',
        type=int,
        metavar='N',
        default=0,
        help='evaluate training performance every Nth round (default: %(default)s)'
    )

    main()
