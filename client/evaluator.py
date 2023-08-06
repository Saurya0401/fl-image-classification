import psutil
import numpy as np
import tensorflow as tf

import argparse
import gzip
from pathlib import Path
from os import getpid, stat

from projects import Projects
from data.loader import DataLoader
from utils.custom_types import Filepath


class Evaluator:

    def __init__(self, project: str, eval_data: Filepath, model_paths: list[Path], eval_rounds: int = 1) -> None:
        self.model_paths: list[Path] = model_paths
        self.eval_rounds: int = eval_rounds
        self.data_loader: DataLoader = DataLoader.from_project(project)
        self.x_test, self.y_test, _, _ = self.data_loader.data_as_numpy(eval_data)
        self.proc = psutil.Process(getpid())

    @staticmethod
    def _infer(interpreter: tf.lite.Interpreter, input_details, output_details, x):
        interpreter.set_tensor(input_details['index'], np.expand_dims(x, axis=0))
        interpreter.invoke()
        return np.squeeze(interpreter.get_tensor(output_details['index'])[0])

    def evaluate(self, verbose: bool = False) -> dict[str, dict[str, str | int | float | list]]:
        eval_results: dict[str, dict[str, str | int | float | list]] = {}
        for model_path in self.model_paths:
            with open(model_path, 'rb') as model_f:
                model: bytes = bytes(model_f.read())
                comp_model: bytes = gzip.compress(model)
                comp_ratio: float = len(model) / len(comp_model)
                print(f'original size: {len(model)} bytes, compressed size: {len(comp_model)} bytes, CR: {comp_ratio}')
            eval_results[model_path.as_posix()] = {
                'model': model_path.name.split('.')[0].replace('_', ' '),
                'size': stat(model_path).st_size,
                'compressed_size': len(comp_model),
                'compression_ratio': comp_ratio,
                'rounds': self.eval_rounds,
                'accuracies': [],
                'num_samples': [],
                'cpu_times': [],
                'ram_usages': [],
            }
        print(f'Performing {self.eval_rounds} rounds of evaluation...')
        for i in range(self.eval_rounds):
            if verbose:
                print(f'Evaluation round {i}:')
            for model_path in eval_results.keys():
                interpreter = tf.lite.Interpreter(model_path=model_path)
                interpreter.allocate_tensors()
                input_details = interpreter.get_input_details()[0]
                output_details = interpreter.get_output_details()[0]
                quant_params = input_details['quantization_parameters']
                scale = quant_params['scales'][0] if quant_params['scales'].size > 0 else None
                zero_point = quant_params['zero_points'][0] if quant_params['zero_points'].size > 0 else None
                num_samples: int = self.x_test.shape[0]
                num_correct: int = 0
                cpu_times: list[float] = []
                ru_start = self.proc.memory_info().rss
                for x, y in zip(self.x_test, self.y_test):
                    quantised: bool = scale is not None and zero_point is not None
                    if quantised:
                        x = np.array([round((x_ / scale) + zero_point) for x_ in x], dtype=np.int8)

                    ct_start = self.proc.cpu_times()
                    output = self._infer(interpreter, input_details, output_details, x)
                    ct_end = self.proc.cpu_times()
                    cpu_time = ((ct_end.user + ct_end.system) - (ct_start.user + ct_start.system)) * 1000
                    cpu_times.append(cpu_time)

                    if quantised:
                        output = [(q_o - zero_point) * scale for q_o in output]
                    if y[np.argmax(output)] == 1:
                        num_correct += 1

                ru_end = self.proc.memory_info().rss
                accuracy: float = num_correct / num_samples
                eval_results[model_path]['accuracies'].append(accuracy)
                eval_results[model_path]['num_samples'].append(num_samples)
                eval_results[model_path]['cpu_times'].append(sum(cpu_times) / len(cpu_times))
                eval_results[model_path]['ram_usages'].append((ru_end - ru_start) / num_samples)
                if verbose:
                    model_path: str = eval_results[model_path]['model']
                    print(f'Model: {model_path}\t-> Accuracy: {accuracy:.10f} ({num_samples} samples)\n')
        return eval_results


def main():
    args = parser.parse_args()
    model_paths = [Path(f) for f in args.model_files]
    for i, fp in enumerate(model_paths):
        if fp.is_dir():
            model_paths.pop(i)
            model_paths.extend(list(fp.glob('*.tflite')))

    eval_results_dir: Path = Path('client/model_opt_eval')
    if not eval_results_dir.exists():
        eval_results_dir.mkdir(parents=True)

    evaluator = Evaluator(args.project, args.eval_data, model_paths, args.eval_rounds)
    with open(eval_results_dir / 'res.csv', 'w') as rf, open(eval_results_dir / 'size.csv', 'w') as sf:
        rf.write('Quantization Type,Average Accuracy,Average CPU Time (ms),Average RAM Usage (Bytes)\n')
        sf.write('Quantization Type,Model Size (Bytes),Compressed Model Size (Bytes),Compression Ratio\n')
        for result in evaluator.evaluate().values():
            avg_accuracy: float = sum(result['accuracies']) / result['rounds']
            avg_cpu_time: float = sum(result['cpu_times']) / result['rounds']
            avg_ram_usage: float = sum(result['ram_usages']) / result['rounds']
            total_samples: int = sum(result['num_samples'])
            samples_per_round: int = round(total_samples / result['rounds'])
            print(f'Model: {result["model"]}')
            print(f'\t-> Size: {result["size"]} Bytes')
            print(f'\t-> Compression Ratio: {result["compression_ratio"]}')
            print(f'\t-> Avg. accuracy: {avg_accuracy:.10f}')
            print(f'\t-> Avg. CPU time: {avg_cpu_time:.5f} ms')
            print(f'\t-> Avg. RAM usage: {avg_ram_usage:.5f} Bytes')
            print(f'\t-> Total samples: {total_samples} ({samples_per_round} samples/round)\n')
            model_name: str = result['model'].replace(' ', '_')
            rf.write(f'{model_name},{avg_accuracy},{avg_cpu_time},{avg_ram_usage}\n')
            sf.write(f'{model_name},{result["size"]},{result["compressed_size"]},{result["compression_ratio"]}\n')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Edge Client Interpreter')
    parser.add_argument(
        '-p', '--project',
        type=str,
        required=True,
        choices=Projects.all(),
        metavar='NAME',
        help='the project name (available: %(choices)s)'
    )
    parser.add_argument(
        '-d', '--eval_data',
        type=str,
        metavar='FILEPATH',
        required=True,
        help='path to evaluation dataset'
    )
    parser.add_argument(
        '-f', '--model_files',
        type=str,
        metavar='FILEPATH',
        required=True,
        nargs='+',
        help='path to TFLite model or directory containing TFLite models'
    )
    parser.add_argument(
        '--eval_rounds',
        type=int,
        default=1,
        help='how many rounds of evaluation to perform (default: %(default)s)'
    )

    main()
