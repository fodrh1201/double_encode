from tqdm import trange
import time
import numpy as np
import tensorflow as tf
import util
from model_saver import ModelSaver
from tensorflow.python.ops import rnn_cell


class RNNEncoder(ModelSaver):

    PARAMS = [
        "s_output_dim",
        "f_output_dim",
        "feature_dim",
        "backprop_truncate_after",
        "batch_size",
        "cell_class",
        "dropout_keep_prob_s_output",
        "dropout_keep_prob_f_output",
        "dropout_keep_prob_cell_input",
        "dropout_keep_prob_cell_output",
        "dropout_keep_prob_embedding",
        "embedding_dim",
        "hidden_dim",
        "num_layers",
        "sequence_length",
        "vocabulary_size"
    ]

    def __init__(self,
                 sequence_length,
                 vocabulary_size,
                 batch_size=64,
                 s_output_dim=256,
                 f_output_dim=256,
                 feature_dim=1024,
                 backprop_truncate_after=256,
                 embedding_dim=256,
                 hidden_dim=256,
                 num_layers=2,
                 cell_class="LSTM",
                 dropout_keep_prob_s_output=1.0,
                 dropout_keep_prob_f_output=1.0,
                 dropout_keep_prob_cell_input=1.0,
                 dropout_keep_prob_cell_output=1.0,
                 dropout_keep_prob_embedding=1.0):

        self.sequence_length = sequence_length
        self.vocabulary_size = vocabulary_size
        self.batch_size = batch_size
        self.s_output_dim = s_output_dim
        self.f_output_dim = f_output_dim
        self.feature_dim = feature_dim
        self.backprop_truncate_after = backprop_truncate_after
        self.embedding_dim = embedding_dim
        self.hidden_dim = hidden_dim
        self.cell_class = cell_class
        self.dropout_keep_prob_s_output = dropout_keep_prob_s_output
        self.dropout_keep_prob_f_output = dropout_keep_prob_f_output
        self.dropout_keep_prob_cell_input = dropout_keep_prob_cell_input
        self.dropout_keep_prob_cell_output = dropout_keep_prob_cell_output
        self.dropout_keep_prob_embedding = dropout_keep_prob_embedding
        self.num_layers = num_layers
        self.cell_class_map = {
            "LSTM": rnn_cell.BasicLSTMCell,
            "GRU": rnn_cell.GRUCell,
            "BasicRNN": rnn_cell.BasicRNNCell,
        }

    @staticmethod
    def add_flags():
        tf.flags.DEFINE_integer("batch_size", 64, "Size for one batch of training/dev examples")
        tf.flags.DEFINE_integer("s_output_dim", 256, "output dimension for sentences.")
        tf.flags.DEFINE_integer("f_output_dim", 256, "output dimension for features.")
        tf.flags.DEFINE_integer("backprop_truncate_after", 256, "Truncated backpropagation after this many steps")
        tf.flags.DEFINE_integer("embedding_dim", 256, "Dimensionality of embedding layer")
        tf.flags.DEFINE_integer("hidden_dim", 256, "Dimensionality of the RNN cells")
        tf.flags.DEFINE_integer("num_layers", 2, "Number of stacked RNN cells")
        tf.flags.DEFINE_string("cell_class", "LSTM", "LSTM, GRU or BasicRNN")
        tf.flags.DEFINE_float("dropout_keep_prob_s_output", 1.0, "output dropout for sentences")
        tf.flags.DEFINE_float("dropout_keep_prob_f_output", 1.0, "output dropout for features")
        tf.flags.DEFINE_float("dropout_keep_prob_cell_input", 1.0, "RNN cell input connection dropout")
        tf.flags.DEFINE_float("dropout_keep_prob_cell_output", 1.0, "RNN cell output connection dropout")
        tf.flags.DEFINE_float("dropout_keep_prob_embedding", 1.0, "Embedding dropout")

    def build_graph(self, input_sents, input_features, margin):
        self.input_sents = input_sents
        self.input_features = input_features
        self.margin = margin

        self.dropout_keep_prob_cell_input_t = tf.constant(self.dropout_keep_prob_cell_input)
        self.dropout_keep_prob_cell_output_t = tf.constant(self.dropout_keep_prob_cell_output)
        self.dropout_keep_prob_embedding_t = tf.constant(self.dropout_keep_prob_embedding)
        self.dropout_keep_prob_f_output_t = tf.constant(self.dropout_keep_prob_f_output)
        self.dropout_keep_prob_s_output_t = tf.constant(self.dropout_keep_prob_s_output)

        with tf.variable_scope("embedding"):
            W = tf.get_variable(
                "W",
                [self.vocabulary_size, self.embedding_dim],
                initializer=tf.random_uniform_initializer(-1.0, 1.0))
            self.embbeded_chars = tf.nn.embedding_lookup(W, self.input_sents)
            self.embedded_chars_drop = tf.nn.dropout(self.embbeded_chars, self.dropout_keep_prob_embedding_t)

        with tf.variable_scope("rnn") as scope:
            # RNN cell
            cell_class = self.cell_class_map[self.cell_class]
            one_cell = rnn_cell.DropoutWrapper(
                cell_class(self.hidden_dim),
                input_keep_prob=self.dropout_keep_prob_cell_input_t,
                output_keep_prob=self.dropout_keep_prob_cell_output_t)
            self.cell = rnn_cell.MultiRNNCell([one_cell] * self.num_layers)
            self.initial_state = tf.zeros([self.batch_size, self.cell.state_size])
            self.rnn_states = [self.initial_state]
            self.rnn_outputs = []
            for i in range(self.sequence_length):
                if i > 0:
                    scope.reuse_variables()
                new_output, new_state = self.cell(self.embedded_chars_drop[:, i, :], self.rnn_states[-1])
                if i < max(0, self.sequence_length - self.backprop_truncate_after):
                    new_state = tf.stop_gradient(new_state)
                self.rnn_outputs.append(new_output)
                self.rnn_states.append(new_state)
            self.final_state = self.rnn_states[-1]
            self.final_output = self.rnn_outputs[-1]

        with tf.variable_scope("s_output"):
            W = tf.get_variable(
                "W",
                [self.hidden_dim, self.s_output_dim],
                initializer=tf.truncated_normal_initializer(stddev=0.1))
            b = tf.get_variable(
                "b",
                [self.s_output_dim],
                initializer=tf.constant_initializer(0.1))
            self.s_output = tf.nn.xw_plus_b(self.final_output, W, b)

        with tf.variable_scope("f_output"):
            W = tf.get_variable(
                "W",
                [self.feature_dim, self.f_output_dim],
                initializer=tf.truncated_normal_initializer(stddev=0.1))
            b = tf.get_variable(
                "b",
                [self.f_output_dim],
                initializer=tf.truncated_normal_initializer(stddev=0.1))
            self.f_output = tf.nn.xw_plus_b(self.input_features, W, b)

        with tf.variable_scope("loss"):
            self.scores = tf.matmul(self.f_output, self.s_output, transpose_b = True)
            print(self.scores.get_shape())
            diagonal = tf.diag_part(self.scores)
            cost_s = tf.maximum(0.0, margin - diagonal + self.scores)
            cost_im = tf.maximum(0.0, margin - tf.reshape(diagonal, [-1, 1]) + self.scores)
            self.total_loss = tf.reduce_mean(cost_s) + tf.reduce_mean(cost_im)


class RNNEncoderTrainer:

    def __init__(self, model, optimizer=None, train_summary_dir=None, sess=None, max_grad_norm=5):
        sess = sess or tf.get_default_session()
        self.model = model
        with tf.variable_scope("training"):
            self.global_step = tf.Variable(0, name="global_step", trainable=False)

            tvars = tf.trainable_variables()
            grads, _ = tf.clip_by_global_norm(tf.gradients(model.total_loss, tvars), max_grad_norm)
            self.optimizer = optimizer or tf.train.AdamOptimizer()
            self.train_op = self.optimizer.apply_gradients(zip(grads, tvars), global_step=self.global_step)

            summary_total_loss = tf.scalar_summary("total_loss", model.total_loss)
            self.train_summaries = tf.merge_summary([summary_total_loss])
            self.train_summary_writer = None
            if train_summary_dir is not None:
                self.train_summary_writer = tf.train.SummaryWriter(train_summary_dir, sess.graph)

    def train_loop(self, sfi_iter, margin, sess=None):
        sess = sess or tf.get_default_session()
        for features, sents in sfi_iter:
            start_ts = time.time()
            feed_dict = {
                self.model.input_sents: sents,
                self.model.input_features: features,
                self.model.margin: margin
            }

            _, train_loss, current_step, summaries, scores = sess.run(
                [self.train_op, self.model.total_loss, self.global_step, self.train_summaries, self.model.s_output],
                feed_dict=feed_dict)

            if self.train_summary_writer is not None:
                self.train_summary_writer.add_summary(summaries, current_step)
            end_ts = time.time()
            yield train_loss, current_step, (end_ts - start_ts)


class RNNEncoderEvaluator:

    def __init__(self, model, summary_dir=None):
        self.model = model
        with tf.variable_scope("evaluation"):
            self.summary_writer = None
            if summary_dir is not None:
                self.summary_writer = tf.train.SummaryWriter(summary_dir)
            self.build_eval_graph()

    def build_eval_graph(self):
        self.total_loss = tf.Variable(0.0, trainable=False, collections=[])
        self.example_count = tf.Variable(0.0, trainable=False, collections=[])
        self.mean_loss = self.total_loss / self.example_count

        inc_total_loss = self.total_loss.assign_add(self.model.total_loss)
        inc_example_count = self.example_count.assign_add(1)

        with tf.control_dependencies(
            [self.total_loss.initializer, self.example_count.initializer]):
            self.eval_reset = tf.no_op()

        with tf.control_dependencies([inc_total_loss, inc_example_count]):
            self.eval_step = tf.no_op()

        summary_mean_loss = tf.scalar_summary("mean_loss", self.mean_loss)
        self.summary = tf.merge_summary([summary_mean_loss])

    def eval(self, xy_iter, margin, global_step=None, sess=None):
        sess = sess or tf.get_default_session()
        global_step = global_step or tf.no_op()
        sess.run(self.eval_reset)
        for features, sents in xy_iter:
            feed_dict = {
                self.model.input_features: features,
                self.model.input_sents: sents,
                self.model.margin: margin,
                self.model.dropout_keep_prob_s_output_t: 1.0,
                self.model.dropout_keep_prob_f_output_t: 1.0,
                self.model.dropout_keep_prob_cell_input_t: 1.0,
                self.model.dropout_keep_prob_cell_output_t: 1.0,
                self.model.dropout_keep_prob_embedding_t: 1.0
            }
            sess.run(self.eval_step, feed_dict=feed_dict)
        loss, summaries, current_step, total_loss = sess.run([self.mean_loss, self.summary, global_step, self.example_count])
        if self.summary_writer is not None:
            self.summary_writer.add_summary(summaries, current_step)
        return [loss, current_step]
