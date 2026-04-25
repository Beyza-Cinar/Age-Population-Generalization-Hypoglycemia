from scipy.special import softmax
import numpy as np

from tensorflow.keras.layers import Conv1D, BatchNormalization, Dense,  Input, Activation, Add, GlobalAveragePooling1D

import tensorflow as tf
from tensorflow.keras.models import Model

from sklearn.metrics import f1_score
from numpy.random import default_rng


# resnet model of wang et al. 
# -*- coding: utf-8 -*-
"""
Created on Sat Nov 12 01:09:17 2016

@author: stephen
"""
def build_resnet_1d(input_shape, n_feature_maps, nb_classes):
    inputs = Input(shape=input_shape)

    # Block 1
    x = BatchNormalization()(inputs)
    x = Conv1D(n_feature_maps, 8, padding="same")(x)
    x = BatchNormalization()(x)
    x = Activation("relu")(x)

    x = Conv1D(n_feature_maps, 5, padding="same")(x)
    x = BatchNormalization()(x)
    x = Activation("relu")(x)

    x = Conv1D(n_feature_maps, 3, padding="same")(x)
    x = BatchNormalization()(x)

    shortcut = Conv1D(n_feature_maps, 1, padding="same")(inputs)
    shortcut = BatchNormalization()(shortcut)

    x = Add()([shortcut, x])
    x = Activation("relu")(x)

    # Block 2
    x1 = x
    x = Conv1D(n_feature_maps * 2, 8, padding="same")(x1)
    x = BatchNormalization()(x)
    x = Activation("relu")(x)

    x = Conv1D(n_feature_maps * 2, 5, padding="same")(x)
    x = BatchNormalization()(x)
    x = Activation("relu")(x)

    x = Conv1D(n_feature_maps * 2, 3, padding="same")(x)
    x = BatchNormalization()(x)

    shortcut = Conv1D(n_feature_maps * 2, 1, padding="same")(x1)
    shortcut = BatchNormalization()(shortcut)

    x = Add()([shortcut, x])
    x = Activation("relu")(x)

    # Block 3
    x1 = x
    x = Conv1D(n_feature_maps * 2, 8, padding="same")(x1)
    x = BatchNormalization()(x)
    x = Activation("relu")(x)

    x = Conv1D(n_feature_maps * 2, 5, padding="same")(x)
    x = BatchNormalization()(x)
    x = Activation("relu")(x)

    x = Conv1D(n_feature_maps * 2, 3, padding="same")(x)
    x = BatchNormalization()(x)

    shortcut = Conv1D(n_feature_maps * 2, 1, padding="same")(x1)
    shortcut = BatchNormalization()(shortcut)

    x = Add()([shortcut, x])
    x = Activation("relu")(x)

    # Output
    x = GlobalAveragePooling1D()(x)
    output = Dense(nb_classes, activation="softmax")(x)

    model = Model(inputs=inputs, outputs=output)
    return model

# FCN model of Wang et al. 
# -*- coding: utf-8 -*-
"""
Created on Sat Nov 12 01:09:17 2016

@author: stephen
"""
def build_FCN(input_shape,  nb_classes):
    x = Input(shape=input_shape)

    # Block 1
    conv1 = tf.keras.layers.Conv1D(128, 8,  padding="same")(x)
    conv1 = tf.keras.layers.BatchNormalization()(conv1)
    conv1 = tf.keras.layers.Activation("relu")(conv1)
    
#    drop_out = Dropout(0.2)(conv1)
    conv2 = tf.keras.layers.Conv1D(256, 5, padding="same")(conv1)
    conv2 = tf.keras.layers.BatchNormalization()(conv2)
    conv2 = tf.keras.layers.Activation("relu")(conv2)
    
#    drop_out = Dropout(0.2)(conv2)
    conv3 = tf.keras.layers.Conv1D(128, 3, padding="same")(conv2)
    conv3 = tf.keras.layers.BatchNormalization()(conv3)
    conv3 = tf.keras.layers.Activation("relu")(conv3)
    
    full = tf.keras.layers.GlobalAveragePooling1D()(conv3)
    out = tf.keras.layers.Dense(nb_classes, activation="softmax")(full)
    
    
    model = tf.keras.models.Model(inputs=x, outputs=out)

    return model



# expected calibration error in batches, since the data is too large 
def expected_calibration_error_batched(samples, true_labels, M=15, batch_size=100_000):
    #samples = np.asarray(samples, dtype=np.float32)

    samples = np.asarray(samples, dtype=np.float32)
    true_labels = np.asarray(true_labels).reshape(-1)

    # Softmax if needed
    row_sums = samples.sum(axis=1)
    if not np.allclose(row_sums, 1.0, atol=1e-3):
        samples = softmax(samples, axis=1)

    bin_edges = np.linspace(0, 1, M + 1)
    total_counts = np.zeros(M)
    total_confs = np.zeros(M)
    total_accs = np.zeros(M)
    total_samples = 0

    N = samples.shape[0]
    for start in range(0, N, batch_size):
        end = min(start + batch_size, N)
        batch = samples[start:end]
        labels = true_labels[start:end]

        confidences = np.max(batch, axis=1)
        predictions = np.argmax(batch, axis=1)
        accuracies = (predictions == labels).astype(np.float32)

        bin_ids = np.digitize(confidences, bin_edges, right=True) - 1
        bin_ids = np.clip(bin_ids, 0, M - 1)

        for i in range(M):
            in_bin = bin_ids == i
            count = np.sum(in_bin)
            if count > 0:
                total_counts[i] += count
                total_confs[i] += np.sum(confidences[in_bin])
                total_accs[i] += np.sum(accuracies[in_bin])

        total_samples += len(batch)

    ece = 0.0
    for i in range(M):
        if total_counts[i] > 0:
            avg_conf = total_confs[i] / total_counts[i]
            avg_acc = total_accs[i] / total_counts[i]
            ece += (total_counts[i] / total_samples) * np.abs(avg_conf - avg_acc)

    return ece


# expected calibration error per class
def compute_ece_per_class(samples, true_labels, M=10):
    n_classes = samples.shape[1]
    ece_per_class = []

    for c in range(n_classes):
        # extracts the predicted probabilities for class c
        p_c = samples[:, c]

        # constructs new "samples" input for binary classification:
        # confidence for "not c" and "is c"
        binary_samples = np.stack([1 - p_c, p_c], axis=1)

        # binary true labels: 1 if true class is c, else 0
        binary_true = (true_labels == c).astype(int)

        # computes ECE using the original function
        ece_c = expected_calibration_error_batched(binary_samples, binary_true, M=M, batch_size=100_000)

        # converts from array to scalar if needed
        ece_per_class.append(float(ece_c))
    # returns the error
    return np.array(ece_per_class)

# macro f1 via bootstrapping
def bootstrap_macro_f1(y_true, y_pred, n_bootstraps=1000, ci=95, seed=42):
    # initializes random number generator with fixed seed
    rng = default_rng(seed)
    # stores number of samples
    n = len(y_true)
    # creates empty list to store F1 scores from each bootstrap iteration
    scores = []

    # performs bootstrap resampling n_bootstraps times
    for _ in range(n_bootstraps):
        # randomly samples indices with replacement
        indices = rng.choice(n, size=n, replace=True)
        # calculates macro F1-score for the sampled data
        f1 = f1_score(y_true[indices], y_pred[indices], average="macro")
        # appends the F1 score to the list
        scores.append(f1)

    # calculates lower confidence interval boundary
    lower = np.percentile(scores, (100 - ci) / 2)
    # calculates upper confidence interval boundary
    upper = np.percentile(scores, 100 - (100 - ci) / 2)
    # returns mean F1 score and confidence interval boundaries
    return np.mean(scores), lower, upper