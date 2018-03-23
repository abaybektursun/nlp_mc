import os
import sys 
import numpy as np
import tensorflow as tf

# From lm_1b
import language_model.lm_1b.data_utils as data_utils

from six.moves       import xrange
from google.protobuf import text_format

#-------------------------------------------------------------------------------
# For saving demo resources, use batch size 1 and step 1.
BATCH_SIZE = 1
NUM_TIMESTEPS = 1
MAX_WORD_LEN = 50

# File Paths
vocab_file = "language_model/data/vocab-2016-09-10.txt"
save_dir   = "language_model/output"
pbtxt      = "language_model/data/graph-2016-09-10.pbtxt"
ckpt       = "language_model/data/ckpt-*"

#Vocabulary containing character-level information.
vocab = data_utils.CharsVocabulary(vocab_file, MAX_WORD_LEN)

targets  = np.zeros([BATCH_SIZE, NUM_TIMESTEPS], np.int32)
weights  = np.ones([BATCH_SIZE, NUM_TIMESTEPS], np.float32)
inputs   = np.zeros([BATCH_SIZE, NUM_TIMESTEPS], np.int32)
char_ids_inputs = np.zeros([BATCH_SIZE, NUM_TIMESTEPS, vocab.max_word_length], np.int32)

#-------------------------------------------------------------------------------
def LoadModel(gd_file, ckpt_file):
    """Load the model from GraphDef and Checkpoint.

    Args:
    gd_file: GraphDef proto text file.
    ckpt_file: TensorFlow Checkpoint file.

    Returns:
    TensorFlow session and tensors dict.
    """
    with tf.Graph().as_default():
        #class FastGFile: File I/O wrappers without thread locking.
        with tf.gfile.FastGFile(gd_file, 'r') as f:
            # Py 2: s = f.read().decode()
            s = f.read()
            # Serialized version of Graph
            gd = tf.GraphDef()
            # Merges an ASCII representation of a protocol message into a message.
            text_format.Merge(s, gd)

        tf.logging.info('Recovering Graph %s', gd_file)

        t = {}
        [t['states_init'], t['lstm/lstm_0/control_dependency'],
         t['lstm/lstm_1/control_dependency'], t['softmax_out'], t['class_ids_out'],
         t['class_weights_out'], t['log_perplexity_out'], t['inputs_in'],
         t['targets_in'], t['target_weights_in'], t['char_inputs_in'],
         t['all_embs'], t['softmax_weights'], t['global_step']
        ] = tf.import_graph_def(gd, {}, ['states_init',
                                         'lstm/lstm_0/control_dependency:0',
                                         'lstm/lstm_1/control_dependency:0',
                                         'softmax_out:0',
                                         'class_ids_out:0',
                                         'class_weights_out:0',
                                         'log_perplexity_out:0',
                                         'inputs_in:0',
                                         'targets_in:0',
                                         'target_weights_in:0',
                                         'char_inputs_in:0',
                                         'all_embs_out:0',
                                         'Reshape_3:0',
                                         'global_step:0'], name='')

        sys.stderr.write('Recovering checkpoint %s\n' % ckpt_file)
        sess = tf.Session(config=tf.ConfigProto(allow_soft_placement=True))
        sess.run('save/restore_all', {'save/Const:0': ckpt_file})
        sess.run(t['states_init'])

    return sess, t
#-------------------------------------------------------------------------------
# Recovers the model from protobuf
sess, t = LoadModel(pbtxt, ckpt)

sentence = "I wish you were "

word_ids = [vocab.word_to_id(w) for w in sentence.split()]
char_ids = [vocab.word_to_char_ids(w) for w in sentence.split()]

if sentence.find('<S>') != 0:
    sentence = '<S> ' + sentence

for i in xrange(len(word_ids)):
    inputs[0, 0] = word_ids[i]
    char_ids_inputs[0, 0, :] = char_ids[i]

    # Add 'lstm/lstm_0/control_dependency' if you want to dump previous layer
    # LSTM.
    lstm_emb = sess.run(t['lstm/lstm_1/control_dependency'],
                        feed_dict={t['char_inputs_in']: char_ids_inputs,
                                   t['inputs_in']: inputs,
                                   t['targets_in']: targets,
                                   t['target_weights_in']: weights})

    #fname = os.path.join(FLAGS.save_dir, 'lstm_emb_step_%d.npy' % i)
    #with tf.gfile.Open(fname, mode='w') as f:
      #np.save(f, lstm_emb)
    #sys.stderr.write('LSTM embedding step %d file saved\n' % i)
