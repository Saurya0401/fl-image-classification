import numpy as np
import tensorflow_model_optimization as tfmot
import tensorflow as tf
from tensorflow import keras

from typing import Any, Optional


class ModelOptimizer:

    def __init__(self, compile_args: dict[str, Any], train_ds: Optional[tf.data.Dataset] = None,
                 val_ds: Optional[tf.data.Dataset] = None) -> None:
        self.compile_args: dict[str, Any] = compile_args
        self.train_ds: Optional[tf.data.Dataset] = train_ds
        self.val_ds: Optional[tf.data.Dataset] = val_ds

    def cluster_model(self, model: keras.Sequential, target_clusters: int, tune_epochs: Optional[int] = None) \
            -> keras.Sequential:
        print(f'\nTargeting {target_clusters} clusters for clustering')
        from tensorflow_model_optimization.python.core.clustering.keras.experimental import cluster
        cluster_weights = cluster.cluster_weights
        centroid_init = tfmot.clustering.keras.CentroidInitialization
        clustering_params = {
            'number_of_clusters': target_clusters,
            'cluster_centroids_init': centroid_init.KMEANS_PLUS_PLUS,
            'preserve_sparsity': True
        }
        clustered_model = cluster_weights(model, **clustering_params)
        clustered_model.compile(**self.compile_args)
        if tune_epochs is not None and self.train_ds is not None:
            clustered_model.fit(self.train_ds, epochs=tune_epochs, validation_data=self.val_ds)
        return tfmot.clustering.keras.strip_clustering(clustered_model)

    def prune_model(self, model: keras.Sequential, target_sparsity: float, tune_epochs: Optional[int] = None) \
            -> keras.Sequential:
        print(f'\nTargeting {target_sparsity * 100:.2f}% sparsity for pruning')
        prune = tfmot.sparsity.keras.prune_low_magnitude
        prune_params = {
            'pruning_schedule': tfmot.sparsity.keras.ConstantSparsity(
                target_sparsity=target_sparsity, begin_step=0, frequency=100
            ),
        }
        pruned_model = prune(model, **prune_params)
        pruned_model.compile(**self.compile_args)
        if tune_epochs is not None and self.train_ds is not None:
            pruned_model.fit(self.train_ds, epochs=tune_epochs, validation_data=self.val_ds,
                             callbacks=[tfmot.sparsity.keras.UpdatePruningStep()])
        return tfmot.sparsity.keras.strip_pruning(pruned_model)

    @staticmethod
    def print_model_weights_sparsity(stripped_model: keras.Sequential, desc: str = '') -> None:
        print(f'\nModel weights sparsity {desc}:')
        for layer in stripped_model.layers:
            if isinstance(layer, keras.layers.Wrapper):
                weights = layer.trainable_weights
            else:
                weights = layer.weights
            for weight in weights:
                # ignore auxiliary quantization weights
                if 'quantize_layer' in weight.name:
                    continue
                weight_size = weight.numpy().size
                zero_num = np.count_nonzero(weight == 0)
                print(f'{weight.name}: {zero_num / weight_size:.2%} sparsity, ({zero_num}/{weight_size})')
