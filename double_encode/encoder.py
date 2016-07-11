import time
import numpy as np
import tensorflow as tf
import util
from model_saver import ModelSaver
from tensorflow.models.rnn import rnn_cell


class RNNEncoder(ModelSaver):

    PARAMS = [
        "s_output_dim",
        "f_output_dim",
        "backprop_truncate_after",
        "batch_size",
        "dropout_keep_prob_s_output",
        "dropout_keep_prob_f_output",
        "dropout_keep_prob_cell_input",
        "dropout_keep_prob_cell_output",

