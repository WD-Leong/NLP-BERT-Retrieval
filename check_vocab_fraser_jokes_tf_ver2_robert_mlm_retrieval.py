
import time
import numpy as np
import pickle as pkl

import tensorflow as tf
import tensorflow_addons as tfa
import tf_ver2_bert_keras_v1 as bert
#import tf_ver2_bert_keras as bert

# Model Parameters. #
seq_length = 25
num_heads  = 4
num_layers = 3
prob_keep  = 0.9
hidden_size = 256
ffwd_size   = 4*hidden_size

model_ckpt_dir  = "../TF_Models/robert_fraser_retrieval"
train_loss_file = "train_loss_robert_fraser_retrieval.csv"

# Load the data. #
tmp_pkl_file = "../Data/jokes/short_jokes_words.pkl"
with open(tmp_pkl_file, "rb") as tmp_load_file:
    jokes_data = pkl.load(tmp_load_file)
    word_vocab = pkl.load(tmp_load_file)
    word_2_idx = pkl.load(tmp_load_file)
    idx_2_word = pkl.load(tmp_load_file)

vocab_size = len(word_2_idx)
print("Vocabulary Size:", str(vocab_size) + ".")

num_data  = len(jokes_data)
CLS_token = word_2_idx["[CLS]"]
EOS_token = word_2_idx["[EOS]"]
PAD_token = word_2_idx["[PAD]"]
UNK_token = word_2_idx["[UNK]"]
TRU_token = word_2_idx["[TRU]"]
MSK_token = word_2_idx["[MSK]"]
print("Total of", num_data, "rows loaded.")

# Set the number of threads to use. #
tf.config.threading.set_intra_op_parallelism_threads(1)
tf.config.threading.set_inter_op_parallelism_threads(1)

# Build the BERT Model. #
print("Building the BERT Model.")
start_time = time.time()

bert_model = bert.BERT(
    num_layers, num_heads, hidden_size, ffwd_size, 
    vocab_size, seq_length+2, rate=1.0-prob_keep)
bert_optim = tfa.optimizers.AdamW(weight_decay=1.0e-3)

elapsed_time = (time.time()-start_time) / 60
print("BERT Model Built", "(" + str(elapsed_time) + " mins).")

# Create the model checkpoint. #
ckpt = tf.train.Checkpoint(
    step=tf.Variable(0), 
    bert_model=bert_model, 
    bert_optim=bert_optim)

manager = tf.train.CheckpointManager(
    ckpt, model_ckpt_dir, max_to_keep=1)

# Restore the model. #
ckpt.restore(manager.latest_checkpoint)
if manager.latest_checkpoint:
    print("Model restored from {}".format(
        manager.latest_checkpoint))
else:
    print("Error: No latest checkpoint found.")
n_iter = ckpt.step.numpy().astype(np.int32)

# Extract the embeddings. #
word_vocab_indices = np.array([x for x in range(vocab_size)])

word_embedding_matrix = \
    bert_model.bert_model.enc_embed(
        word_vocab_indices).numpy()

print("Word embedding matrix has shape:", word_embedding_matrix.shape)
print("=" * 50)

while True:
    word_input = input("Enter word: ")
    word_input = word_input.strip().lower()

    if word_input == "":
        break
    else:
        word_idx = word_2_idx.get(word_input, UNK_token)
        if word_idx == UNK_token:
            print("Word not in vocabulary.")
        else:
            tmp_index  = np.expand_dims(word_idx, axis=0)
            word_embed = bert_model.bert_model.enc_embed(tmp_index)
            word_dist  = np.sqrt(np.sum(np.square(
                word_embed - word_embedding_matrix), axis=1))
            
            # Sort the distances. #
            sort_idx = np.argsort(word_dist)
            print("Top match (Distance:", 
                  str(round(word_dist[sort_idx[1]], 3)) + ")", 
                  " -- ", idx_2_word[sort_idx[1]])
            print("2nd match (Distance:", 
                  str(round(word_dist[sort_idx[2]], 3)) + ")", 
                  " -- ", idx_2_word[sort_idx[2]])
            print("3rd match (Distance:", 
                  str(round(word_dist[sort_idx[3]], 3)) + ")", 
                  " -- ", idx_2_word[sort_idx[3]])
            print("=" * 50)

