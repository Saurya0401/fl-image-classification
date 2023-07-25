from numpy.typing import NDArray
from numpy import uint8, float32

from os import PathLike
from typing import Union

Filepath = Union[str, bytes, PathLike[str], PathLike[bytes]]
SplitDataset = tuple[NDArray[float32], NDArray[uint8], NDArray[float32], NDArray[uint8]]
