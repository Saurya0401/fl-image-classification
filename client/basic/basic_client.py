import psutil
import tensorflow as tf
from numpy.typing import NDArray

from os import getpid
from time import perf_counter
from typing import Union

from data.preprocessor import Preprocessor


class BasicClient:

    def __init__(self, project: str, model_path: str) -> None:
        self.interpreter: tf.lite.Interpreter = tf.lite.Interpreter(model_path)
        self.interpreter.allocate_tensors()
        self.infer = self.interpreter.get_signature_runner('infer')
        self.preprocessor: Preprocessor = Preprocessor.from_project(project)

    def predict(self, inp) -> NDArray[str]:
        """
        Runs inference on provided inputs.
        :param inp: the provided input attributes
        :return: the predicted output
        """

        output = self.infer(x=self.preprocessor.normalize_input(inp))
        return self.preprocessor.decode_output(output['output'])


class ProfiledBasicClient(BasicClient):

    def __init__(self, project: str, model_path: str) -> None:
        super().__init__(project, model_path)
        self.proc = psutil.Process(getpid())

    def predict(self, inp) -> dict[str, Union[NDArray[str], int, float]]:
        norm_inp = self.preprocessor.normalize_input(inp)
        cpu_time_s = self.proc.cpu_times()
        ram_usage_s = self.proc.memory_info().rss
        inf_time_s = perf_counter()
        output = self.infer(x=norm_inp)
        cpu_time_e = self.proc.cpu_times()
        ram_usage_e = self.proc.memory_info().rss
        inf_time_e = perf_counter()
        return {
            'output': self.preprocessor.decode_output(output['output']),
            'cpu_time': ((cpu_time_e.user + cpu_time_e.system) - (cpu_time_s.user + cpu_time_s.system)) * 1000,
            'ram_usage': ram_usage_e - ram_usage_s,
            'inf_time': inf_time_e - inf_time_s,
        }
