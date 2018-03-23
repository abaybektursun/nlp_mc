import os
import sys 
import numpy as np
import tensorflow as tf

# From lm_1b
import language_model.lm_1b.data_utils as data_utils

from six.moves       import xrange
from google.protobuf import text_format

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

        sess = tf.Session(config=tf.ConfigProto(allow_soft_placement=True))
        sess.run('save/restore_all', {'save/Const:0': ckpt_file})
        sess.run(t['states_init'])

    return sess, t

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

# Recovers the model from protobuf
sess, t = LoadModel(pbtxt, ckpt)

#-------------------------------------------------------------------------------

def forward(sentence):
    # Tokenize characters and words
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
        return lstm_emb

#-------------------------------------------------------------------------------

# Run the model with some data and retrieve sentence level embeddings

final_embeds = []; annot = []
gov_data = [
        "Privately owned, subsidized housing in which landlords are subsidized to offer reduced rents to low-income tenants",
        "Even if you are ineligible for benefits through these agencies, they may be able to provide referrals to community organizations that might offer help",
        "Each state or city may have different eligibility requirements for housing programs. Contact your local Public Housing Agency to learn about your eligibility for Housing Choice Vouchers",
        "The Eldercare Locator is a free service that can connect you with resources and programs designed to help seniors in your area",
        "The Housing Choice Voucher Program (formerly known as Section 8) is a program administered by the Department of Housing and Urban Development (HUD) that helps pay for rental housing for low-income families or people who are elderly or disabled throughout the United States",
        "The federal government typically awards grants to state and local governments, universities, researchers, law enforcement, organizations, and institutions planning major projects that will benefit specific parts of the population or the community as a whole",
        "A grant is one of the ways the government funds ideas and projects to provide public services and stimulate the economy",
        "Grants support critical recovery initiatives, innovative research, and many other programs listed in the Catalog of Federal Domestic Assistance",
        "Government loans serve a specific purpose such as paying for education, helping with housing or business needs, or responding to an emergency or crisis",
        "The most common type of financial help from the government for home repairs or modifications is through home improvement loans programs backed by the government",
        "Reach out to the federal, state, or county government agency that administers the program. Loans are made by traditional lenders, but the government programs help these lenders make loans that they might normally not fulfill"
        ]
cosm_data = [
        "A shocking number of women have trouble mentally letting go and enjoying oral sex when their partner goes down on them",
        "While very few things are going to recreate the feeling of a tongue exactly, some newer vibrators come pretty close",
        "If your partner is super into the idea of full-penis sensation, you can deliver that easily, without deep-throating",
        "When, where, and how your partner ejaculates during a blow job should be something both of you discuss and agree upon",
        "Lube can add extra sensation to a blow job, and be enjoyable to you too",
        "Neither one of you can read each other's mind during sex, so speak up if there's something that you want that he's not delivering",
        "You should never be doing anything in bed that feels uncomfortable, but if you're coming up against a bit of muscle fatigue, try any of these hand job techniques",
        "Being stimulated in multiple areas will help a woman reach climax more quickly",
        "Your temperature rises slightly when you're aroused, so anything cool will be a pleasurable jolt to your senses",
        "If you're having trouble orgasming in a standard partner-in-between-your-legs position, switch it up and mount his face, being careful not to apply too much pressure",
        "Here's a unique way to give his frenulum—the tiny bump on the underside of his penis where the shaft meets the tip—some special attention: Place the tip of your finger on it, then take his shaft (along with your finger) into your mouth",
        "You hundred percent do not need to be on your knees to give your partner oral sex",
        ]


for text in gov_data:
    final_embeds.append(forward(text).reshape(-1))
    annot.append("usa.gov")

for text in cosm_data:
    final_embeds.append(forward(text).reshape(-1))
    annot.append("cosm")

color= ['cyan' if l == 'cosm' else 'grey' for l in annot]
final_embeds = np.array(final_embeds)

#-------------------------------------------------------------------------------

# Visualize
from sklearn.decomposition import PCA
import matplotlib as mpl
# Disable X server mode
mpl.use('Agg')
import matplotlib.pyplot as plt

# Apply PCA
X_embedded = PCA(n_components=2).fit_transform(final_embeds)

# Plot the projected sentence embeddings
fig, ax = plt.subplots()
ax.scatter(X_embedded[:, 0], X_embedded[:, 1], color=color)
for i, txt in enumerate(annot):
    ax.annotate(txt, (X_embedded[:, 0][i],X_embedded[:, 1][i]))

plt.savefig('words.png')
