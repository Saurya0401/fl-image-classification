from tensorflow import keras

from random import randrange
from typing import Any, Callable

from utils.constants import *


def get_cifar10_model_data(**kwargs) -> dict[str, int | dict[str, Any]]:
    print(f'Model hyper-parameters: {kwargs}\n')
    x_shape: tuple[int, ...] = (32, 32, 3)
    y_shape: int = 10
    hidden_layer_args = {
        'activation': keras.activations.relu,
        'kernel_initializer': keras.initializers.HeNormal(seed=kwargs.get('seed', None) or randrange(1000)),
    }
    output_layer_args = {
        'activation': keras.activations.softmax,
    }
    compile_args = {
        'loss': keras.losses.CategoricalCrossentropy(),
        'metrics': [keras.metrics.CategoricalAccuracy(name='accuracy')],
        'optimizer': keras.optimizers.legacy.Adam(learning_rate=kwargs.get('learning_rate', None) or 0.001),
    }
    return {
        MOD_X_SHAPE: x_shape,
        MOD_Y_SHAPE: y_shape,
        MOD_HIDDEN_ARGS: hidden_layer_args,
        MOD_OUTPUT_ARGS: output_layer_args,
        MOD_COMPILE_ARGS: compile_args,
    }


_MODEL_DATA_FUNCS: dict[str, Callable[[dict[str, Any]], dict[str, int | dict[str, Any]]]] = {
    'cifar10': get_cifar10_model_data
}


def get_model_data(project: str, **kwargs) -> dict[str, int | dict[str, Any]]:
    return _MODEL_DATA_FUNCS[project](**kwargs)


def get_compile_args(project: str, **kwargs) -> dict[str, Any]:
    return get_model_data(project, **kwargs)[MOD_COMPILE_ARGS]
