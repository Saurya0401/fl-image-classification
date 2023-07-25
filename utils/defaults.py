from pathlib import Path
from typing import Final


class Defaults:
    # dataset handler variables
    DS_BATCH_SIZE: Final[int] = 200
    DS_SHUFFLE_BUFFER: Final[int] = 100
    DS_REPEAT_EPOCHS: Final[int] = 2
    DS_VAL_SPLIT: Final[float] = 0.2
    DS_TEST_SPLIT: Final[float] = 0.2

    # model hyperparameters
    ML_SEED: Final[int] = 42
    ML_LEARNING_RATE: Final[float] = 0.001

    # model tuning parameters
    TUNE_MAX_LAYERS: Final[int] = 4
    TUNE_MAX_UNITS: Final[int] = 256
    TUNE_MAX_EPOCHS: Final[int] = 10
    TUNE_FACTOR: Final[int] = 3
    TUNE_ITERATIONS: Final[int] = 2

    # model optimization defaults
    OPT_TARGET_SPARSITY: Final[float] = 50
    OPT_TARGET_CLUSTERS: Final[int] = 8

    # saved model directories
    TUNED_MODEL_DIR: Final[Path] = Path('./server/tuned')
    DEFAULT_MODEL_DIR: Final[Path] = Path('./server/default')
    SERVER_MODEL_DIR: Final[Path] = Path('./server/models')
    CLIENT_MODEL_DIR: Final[Path] = Path('./client/models')
