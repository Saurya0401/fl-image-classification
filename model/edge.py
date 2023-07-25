import tensorflow as tf
import keras


class EdgeModel(tf.Module):

    def __init__(self, model: keras.Sequential):
        """
        Edge client model capable of only inference, unable to train.

        Used in low-power devices such as Microcontrollers.
        :param model: the server model to use as a base
        """

        super().__init__()
        self.model = model

        self.infer = tf.function(
            self.__infer__,
            input_signature=[
                tf.TensorSpec(self.model.input_shape, tf.float32),
            ]
        )

    def __infer__(self, x):
        logits = self.model(x)
        return {'output': logits}
