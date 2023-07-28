from tensorflow import keras

import argparse
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Optional, Final
from urllib.parse import urlparse, parse_qs

from projects import Projects
from data.loader import TfDatasetMaker
from model.generator import EdgeModelConverter
from utils.defaults import Defaults

SERVING_DIR: Final[str] = './server/provider'
TRAINED_DIR: Final[str] = './client/basic/trained_models'
UNTRAINED_DIR: Final[str] = './client/basic/untrained_models'


def update_basic_model(project: str, serving_dir: Path = SERVING_DIR, search_dirs: Optional[list[Path]] = None) -> None:
    ds_loader: TfDatasetMaker = TfDatasetMaker.from_project(project)
    latest_basic_model: str = get_latest_model(project, search_dirs).as_posix()
    is_untrained: bool = 'untrained' in latest_basic_model
    with open(latest_basic_model, 'rb') as f:
        quant_flag: int = bytes(f.read())[-1]
    server_model: keras.Sequential = keras.models.load_model(
        Defaults.CLIENT_MODEL_DIR / (project + '_' + ('untrained' if is_untrained else 'trained')),
    )
    server_model.summary()
    converter: EdgeModelConverter = EdgeModelConverter(
        project=project,
        server_model=server_model,
        untrained=is_untrained,
        basic_model_dir=serving_dir,
        repr_ds=ds_loader.val_ds,
        quantization=EdgeModelConverter.Q_CHOICES[quant_flag]
    )
    converter.generate_basic_model()
    print(f'updated model \'{server_model}\' to \'{serving_dir}\' as \'{project}_model.tflite\'')


def get_latest_model(project: str, search_dirs: Optional[list[Path]] = None) -> Path:
    models: list[Path] = []
    search_dirs: list[Path] = search_dirs or [Path(TRAINED_DIR), Path(UNTRAINED_DIR)]
    for dir_ in search_dirs:
        models.extend([mod for mod in list(dir_.glob('*.tflite')) if project in mod.name])
    if not models:
        raise ValueError(f'No models found for project "{project}"')
    latest_model: Path = sorted(models, key=lambda mod: mod.stat().st_mtime)[-1]
    return latest_model


class BinaryDataHandler(BaseHTTPRequestHandler):

    def __init__(self, *args, **kwargs) -> None:
        self.latest_dir: Path = Path(c_args.latest_dir)
        if not self.latest_dir.exists():
            self.latest_dir.mkdir(parents=True)
        super().__init__(*args, **kwargs)

    def do_GET(self):
        query_components = parse_qs(urlparse(self.path).query)
        project: Optional[str] = query_components.get('project', [None])[0]
        if project is None or project not in Projects.all():
            self.send_response(400, message=f'Invalid project name "{project}"')
            self.end_headers()
            return
        if query_components.get('command', [None])[0] == 'update':
            update_basic_model(project)
            self.send_response(200, message=f'Latest model for {project} updated')
            self.end_headers()
            return
        model: Path = self.latest_dir / f'{project}_model.tflite'
        if not model.exists():
            self.send_response(404, message='Latest model not located')
            self.end_headers()
            return
        self.send_response(200)
        self.send_header('Content-type', 'application/octet-stream')
        self.send_header('Content-Disposition', f'attachment; filename="{model.name}"')
        self.send_header('Content-Length', str(model.stat().st_size))
        self.end_headers()
        with open(model, 'rb') as f:
            binary_data = f.read()
        self.wfile.write(binary_data)


def main():
    ip, port = c_args.host.replace("'", ""), c_args.port
    server = ThreadingHTTPServer((ip, port), BinaryDataHandler)
    serving_dir = Path(c_args.serving_dir)
    if not serving_dir.exists():
        serving_dir.mkdir(parents=True)
    update_basic_model(c_args.project.replace("'", ""), Path(c_args.serving_dir), [Path(d) for d in c_args.search_dirs])
    if not c_args.update_only:
        try:
            print(f'Model provider server started at http://{ip}:{port}')
            server.serve_forever()
        except KeyboardInterrupt:
            server.shutdown()
            server.server_close()
            print('Model provider server terminated by Ctrl + C.')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(prog='python -m server.model_provider')
    parser.add_argument(
        '-p', '--project',
        type=str,
        choices=Projects.all(),
        metavar='NAME',
        required=True,
        help='the project name (available: %(choices)s)'
    )
    parser.add_argument(
        '--host',
        type=str,
        default='0.0.0.0',
        help='the server host IP (default: %(default)s)'
    )
    parser.add_argument(
        '--port',
        type=int,
        default=8050,
        help='the server port (default: %(default)s)'
    )
    parser.add_argument(
        '--update_only',
        action='store_true',
        help='only update the model and skip starting the server'
    )
    parser.add_argument(
        '--serving_dir',
        type=str,
        metavar='DIR',
        default=SERVING_DIR,
        help='the directory from which models are served (default: %(default)s)'
    )
    parser.add_argument(
        '--search_dirs',
        type=str,
        nargs='+',
        metavar='DIR',
        default=[TRAINED_DIR, UNTRAINED_DIR],
        help='the directories to search for the latest models (default: %(default)s)'
    )

    c_args = parser.parse_args()
    main()
