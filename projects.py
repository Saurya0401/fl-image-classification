from dataclasses import dataclass

from utils.custom_types import Filepath


@dataclass(frozen=True, eq=False)
class ProjectSpec:
    name: str
    desc: str
    data_dir: Filepath
    label_map: dict[str, int]
    image_shape: tuple[int, int]
    color_channels: int


class Projects:

    cifar10: ProjectSpec = ProjectSpec(
        name='cifar10',
        desc='project for benchmarking image classification with the CIFAR-10 dataset',
        data_dir='data/raw_data/cifar_10_py',
        label_map={
            'airplane': 0,
            'automobile': 1,
            'bird': 2,
            'cat': 3,
            'deer': 4,
            'dog': 5,
            'frog': 6,
            'horse': 7,
            'ship': 8,
            'truck': 9
        },
        image_shape=(32, 32),
        color_channels=3,
    )

    __proj_specs: dict[str, ProjectSpec] = {
        cifar10.name: cifar10
    }

    def __repr__(self) -> str:
        proj_info: str = ''
        for proj in [Projects.cifar10]:
            proj_info += str(proj) + '\n'
        return proj_info

    @staticmethod
    def all() -> list[str]:
        return list(Projects.__proj_specs.keys())

    @staticmethod
    def get_project_spec(name: str) -> ProjectSpec:
        return Projects.__proj_specs[name]


def main():
    print(Projects())


if __name__ == '__main__':
    main()
