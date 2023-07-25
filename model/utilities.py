import keras
from pathlib import Path
from typing import Optional

from utils.defaults import Defaults
from utils.custom_types import Filepath


class ModelManager:
    def __init__(self, project: str, untrained: bool, server_model_dir: Optional[Filepath] = None,
                 tuned_model_dir: Optional[Filepath] = None, default_model_dir: Optional[Filepath] = None) -> None:
        self.project: str = project
        self.untrained: bool = untrained
        self.server_model_dir: Path = Path(server_model_dir) if server_model_dir else Defaults.SERVER_MODEL_DIR
        self.tuned_model_dir: Path = Path(tuned_model_dir) if tuned_model_dir else Defaults.TUNED_MODEL_DIR
        self.default_model_dir: Path = Path(default_model_dir) if default_model_dir else Defaults.DEFAULT_MODEL_DIR

        self._init_dir(self.server_model_dir)
        self._init_dir(self.tuned_model_dir)
        self._init_dir(self.default_model_dir)

        self.server_model_dir /= self.project + '_' + ('untrained' if self.untrained else 'trained')
        self.tuned_model_dir /= self.project
        self.default_model_dir /= self.project

    def get_server_model(self) -> keras.Sequential:
        try:
            return keras.models.load_model(self.server_model_dir)
        except (OSError, ValueError, FileNotFoundError):
            raise OSError('Could not load model. Generate model with `server.model_generator`')

    @staticmethod
    def _init_dir(model_dir: Path) -> None:
        if not model_dir.exists():
            model_dir.mkdir(parents=True)
