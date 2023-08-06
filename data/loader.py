import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
import pickle

from pathlib import Path
from typing import Any, Optional

from projects import Projects
from data.preprocessor import Preprocessor
from utils.custom_types import Filepath, SplitDataset
from utils.defaults import Defaults


def _unpickle_data(filepath: Filepath) -> dict[bytes, Any]:
    with open(filepath, 'rb') as f:
        return pickle.load(f, encoding='bytes')


class DataLoader:

    def __init__(self, data_dir: Filepath, preprocessor: Preprocessor) -> None:
        self.data_dir: Path = Path(data_dir)
        self.preprocessor: Preprocessor = preprocessor
        self.metadata: dict[bytes, Any] = _unpickle_data(self.data_dir / 'batches.meta')
        self.label_names: list[str] = [label.decode('utf-8') for label in self.metadata[b'label_names']]
        images: list[np.ndarray] = []
        labels: list[list[list[int]]] = []
        for data_batch in self.data_dir.glob('data_batch_[0-9]'):
            data = _unpickle_data(data_batch)
            images.append(data[b'data'])
            labels.append([[lbl] for lbl in data[b'labels']])
        self.images_train: np.ndarray[np.uint8] = np.concatenate(images)
        self.labels_train: np.ndarray[np.uint8] = np.concatenate(labels).astype(np.uint8)
        self.images_test, self.labels_test = self.load_images_and_labels(self.data_dir / 'test_batch')

    def data_as_numpy(self, data_path: Optional[Filepath] = None) -> SplitDataset:
        if data_path is not None:
            images, labels = self.load_images_and_labels(Path(data_path))
            x = self.preprocessor.normalize_images(images)
            y = self.preprocessor.one_hot_encode(labels)
            return self.train_test_split(x, y)
        x_train = self.preprocessor.normalize_images(self.images_train)
        y_train = self.preprocessor.one_hot_encode(self.labels_train)
        x_test = self.preprocessor.normalize_images(self.images_test)
        y_test = self.preprocessor.one_hot_encode(self.labels_test)
        return x_train, y_train, x_test, y_test

    def visualize_random(self, rows: int = 5, columns: int = 5) -> None:
        images: np.ndarray = self.images_train.reshape((len(self.images_train), 3, 32, 32))
        images = images.transpose((0, 2, 3, 1))
        labels = self.labels_train

        image_id = np.random.randint(0, len(images), rows * columns)
        plt_images = images[image_id]
        plt_labels = [labels[i][0] for i in image_id]

        fig = plt.figure(figsize=(rows, columns))
        for i in range(1, columns * rows + 1):
            fig.add_subplot(rows, columns, i)
            plt.imshow(plt_images[i - 1])
            plt.xticks([])
            plt.yticks([])
            plt.title(str(self.label_names[plt_labels[i - 1]]))
        plt.tight_layout()
        plt.show()

    @classmethod
    def from_project(cls, project_name: str) -> 'DataLoader':
        preprocessor: Preprocessor = Preprocessor.from_project(project_name)
        return cls(Projects.get_project_spec(project_name).data_dir, preprocessor)

    @staticmethod
    def load_images_and_labels(data_file: Filepath) -> tuple[np.ndarray[np.uint8], np.ndarray[np.uint8]]:
        data: dict[bytes, Any] = _unpickle_data(data_file)
        return data[b'data'], np.array([[lbl] for lbl in data[b'labels']], dtype=np.uint8)

    @staticmethod
    def train_test_split(x: np.ndarray, y: np.ndarray, shuffle: bool = True, split: Optional[float] = 0.8) \
            -> SplitDataset:
        if len(x) != len(y):
            raise ValueError('X and y data do not have the same number of samples')
        if shuffle:
            num_samples: int = len(x)
            permutation: np.ndarray[int] = np.random.permutation(num_samples)
            x, y = x[permutation], y[permutation]
        split_idx = int(split * x.shape[0])
        x_train, x_test = np.array_split(x, [split_idx], axis=0)
        y_train, y_test = np.array_split(y, [split_idx], axis=0)
        print(x_train.shape, x_test.shape)
        print(y_train.shape, y_test.shape)
        return x_train, y_train, x_test, y_test


class TfDatasetMaker:

    def __init__(self, data_loader: DataLoader, /, shuffle: bool = True, val_split: float = 0.2,
                 data_path: Optional[Filepath] = None, **kwargs) -> None:
        self.data_loader: DataLoader = data_loader
        x_train, y_train, x_test, y_test = self.data_loader.data_as_numpy(data_path)
        x_train, y_train, x_val, y_val = self.data_loader.train_test_split(x_train, y_train, shuffle, 1 - val_split)
        self.num_train_samples: int = len(x_train)
        self.num_test_samples: int = len(x_test)
        self.num_val_samples: int = len(x_val)
        self.seed = kwargs.get('seed', Defaults.ML_SEED)
        self.batch_size: int = kwargs.get('batch_size', Defaults.DS_BATCH_SIZE)
        self.shuffle_buffer: int = kwargs.get('shuffle_buffer', Defaults.DS_SHUFFLE_BUFFER)
        self.repeat_epochs: int = kwargs.get('repeat_epochs', Defaults.DS_REPEAT_EPOCHS)

        np.random.seed(self.seed)
        tf.random.set_seed(self.seed)

        self.train_ds = tf.data.Dataset.from_tensor_slices((x_train, y_train)) \
            .repeat(self.repeat_epochs) \
            .shuffle(self.shuffle_buffer, self.seed) \
            .batch(self.batch_size)
        self.test_ds = tf.data.Dataset.from_tensor_slices((x_test, y_test)) \
            .repeat(self.repeat_epochs) \
            .shuffle(self.shuffle_buffer, self.seed) \
            .batch(self.batch_size)
        self.val_ds = tf.data.Dataset.from_tensor_slices((x_val, y_val)) \
            .repeat(self.repeat_epochs) \
            .shuffle(self.shuffle_buffer, self.seed) \
            .batch(self.batch_size)

    @classmethod
    def from_project(cls, project: str, shuffle: bool = True, val_split: float = 0.2,
                     data_path: Optional[Filepath] = None, **kwargs) -> 'TfDatasetMaker':
        return cls(DataLoader.from_project(project), shuffle, val_split, data_path, **kwargs)


if __name__ == '__main__':
    pass
