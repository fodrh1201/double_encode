import time
import os
import pprint
import util
import tensorflow as tf
import datetime
from encoder import RNNEncoder, RNNEncoderTrainer

pp = pprint.PrettyPrinter(indent=2)

# Encoder parameters

RNNEncoder.add_flags()

# Training parameters

tf.flags.DEFINE_integer("max_sequence_length", 525, "Examples will be padded/truncated to this length")
tf.flags.DEFINE_integer("num_epochs", 20, "Number of training epochs")
tf.flags.DEFINE_integer("checkpoint_every", 1, "Evaluate model after this number of steps")

# Session Parameters

tf.flags.DEFINE_boolean("allow_soft_placement", False, "Allow soft device placement (e.g. no GPU)")
tf.flags.DEFINE_boolean("log_device_placement", True, "Log placement of ops on devices")

FLAGS = tf.flags.FLAGS
FLAGS.batch_size
print("\nParameters:")
for attr, value in sorted(FLAGS.__flags.items()):
    print("{}={}".format(attr.upper(), value))
print("")

data_util = util.DataUtil()

SEQUENCE_LENGTH = 25
VOCABULARY_SIZE = 9486
FEATURE_DIM = 1024
train_data_iter = data_util.batch_iter(FLAGS.batch_size, FLAGS.num_epochs, fill=True)
input_indices = data_util.get_cluster_indices()[:3]

# Create a graph and session

graph = tf.Graph()
session_conf = tf.ConfigProto(
    allow_soft_placement=FLAGS.allow_soft_placement,
    log_device_placement=FLAGS.log_device_placement)
sess = tf.Session(graph=graph, config=session_conf)

with graph.as_default(), sess.as_default():
    model_params = {"sequence_length": SEQUENCE_LENGTH, "vocabulary_size": VOCABULARY_SIZE, "feature_dim": FEATURE_DIM}
    model_params.update(FLAGS.__flags)
    model = RNNEncoder.from_dict(model_params)
    model.print_params()
    input_sents = tf.placeholder(tf.int32, [None, SEQUENCE_LENGTH])
    input_features = tf.placeholder(tf.float32, [None, FEATURE_DIM])
    margin = tf.placeholder(tf.float32)
    model.build_graph(input_sents, input_features, margin)

    # Directory for training summaries

    timestamp = str(int(time.time()))
    rundir = os.path.abspath(os.path.join(os.path.curdir, "runs", timestamp))
    train_dir = os.path.join(rundir, "train")

    # Build the Trainer
    trainer = RNNEncoderTrainer(model, train_summary_dir=train_dir)

    # Saving/Checkpointing
    checkpoint_dir = os.path.join(rundir, "checkpoints")
    if not os.path.exists(checkpoint_dir):
        os.makedirs(checkpoint_dir)
    checkpoint_file = os.path.join(checkpoint_dir, "model.ckpt")
    saver = tf.train.Saver(keep_checkpoint_every_n_hours=2)

    # Initialization, optionally load from checkpoint
    sess.run(tf.initialize_all_variables())
    latest_checkpoint = tf.train.latest_checkpoint(checkpoint_dir)
    if latest_checkpoint:
        print("Restoring checkpoint from {}".format(latest_checkpoint))
        saver.restore(sess, latest_checkpoint)

    # Initialize variables
    sess.run(tf.initialize_all_variables())
    # Training loop
    for train_loss, current_step, time_delta in trainer.train_loop(train_data_iter, 1):
        print("{}: step {}, loss {:g}, ({} epoch/sec)".format(
            datetime.datetime.now().isoformat(), current_step, train_loss, time_delta))

        # Checkpoint Model
        if current_step % FLAGS.checkpoint_every == 0:
            save_path = saver.save(sess, checkpoint_file, global_step=trainer.global_step)
            print("Saved {}".format(save_path))
