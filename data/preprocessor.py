import numpy as np

from projects import Projects, ProjectSpec


class Preprocessor:

    def __init__(self, label_encoder_map: dict[str, int], image_shape: tuple[int, int], color_channels: int,
                 reshape_2d: bool) -> None:
        self.label_encoder_map: dict[str, int] = label_encoder_map
        self.label_decoder_map: dict[int, str] = {v: k for k, v in self.label_encoder_map.items()}
        self.image_shape: tuple[int, int] = image_shape
        self.color_channels: int = color_channels
        self.reshape_2d: bool = reshape_2d

    def one_hot_encode(self, labels: np.ndarray[np.uint8]) -> np.ndarray[np.int8]:
        encoded = np.zeros((labels.shape[0], len(self.label_encoder_map)), dtype=np.int8)
        for i in range(labels.shape[0]):
            idx = labels[i][0]
            encoded[i, idx] = 1
        return encoded

    def decode_labels(self, outputs: np.ndarray[np.int8]) -> np.ndarray[str]:
        labels = np.empty((outputs.shape[0], 1), dtype=object)
        for i in range(outputs.shape[0]):
            output = outputs[i]
            idx = int(np.argmax(output))
            labels[i][0] = self.label_decoder_map[idx]
        return labels.astype(str)

    def decode_output(self, output: np.ndarray[np.int8]) -> np.ndarray[str]:
        return self.decode_labels(np.expand_dims(output, axis=0))

    def normalize_images(self, images: np.ndarray[np.uint8]) -> np.ndarray[np.float32]:
        if self.reshape_2d:
            images = images.reshape((len(images), self.color_channels, *self.image_shape))
            images = images.transpose((0, 2, 3, 1))
        return images.astype(np.float32) / 255.

    def normalize_input(self, image: np.ndarray[np.uint8]) -> np.ndarray[np.float32]:
        return self.normalize_images(np.expand_dims(image, axis=0))

    @classmethod
    def from_project(cls, project: str, reshape_2d: bool = False) -> 'Preprocessor':
        proj_spec: ProjectSpec = Projects.get_project_spec(project)
        return cls(proj_spec.label_map, proj_spec.image_shape, proj_spec.color_channels, reshape_2d)


if __name__ == '__main__':
    pass
