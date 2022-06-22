import time
import numpy as np
import pickle as pkl

import tensorflow as tf
import tf_ver2_bert_keras as bert
from nltk.tokenize import wordpunct_tokenize

# Model Parameters. #
topk = 5
batch_size = 256
seq_length = 25
num_heads  = 4
num_layers = 3
prob_keep  = 0.9
hidden_size = 256
ffwd_size   = 4*hidden_size

model_ckpt_dir  = "TF_Models/bert_reddit"
train_loss_file = "train_loss_bert_reddit.csv"

# Load the data. #
tmp_pkl_file = "../../Data/reddit_jokes/reddit_jokes.pkl"
with open(tmp_pkl_file, "rb") as tmp_load_file:
    tmp_data = pkl.load(tmp_load_file)
    idx2word = pkl.load(tmp_load_file)
    word2idx = pkl.load(tmp_load_file)

vocab_size = len(word2idx)
print("Vocabulary Size:", str(vocab_size) + ".")

num_data  = len(tmp_data)
CLS_token = vocab_size
EOS_token = vocab_size + 1
PAD_token = vocab_size + 2
UNK_token = vocab_size + 3
MSK_token = vocab_size + 4
TRU_token = vocab_size + 5
print("Total of", str(len(tmp_data)), "rows loaded.")

# Set the number of threads to use. #
tf.config.threading.set_intra_op_parallelism_threads(4)
tf.config.threading.set_inter_op_parallelism_threads(4)

# Build the BERT Model. #
print("Building the BERT Model.")
start_time = time.time()

bert_model = bert.BERT(
    num_layers, num_heads, hidden_size, ffwd_size, 
    vocab_size+6, seq_length+2, rate=1.0-prob_keep)
bert_optim = tf.keras.optimizers.Adam(
    beta_1=0.9, beta_2=0.98, epsilon=1.0e-9)

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

# Generate the document embeddings of the BERT model. #
doc_embed_list = []
tmp_anchor_idx = np.zeros(
    [1, seq_length+2], dtype=np.int32)

print("-" * 50)
print("Generating the document embeddings.")
print("-" * 50)

if num_data < batch_size:
    tot_batch = 1
elif num_data % batch_size == 0:
    tot_batch = int(num_data / batch_size)
else:
    tot_batch = int(num_data / batch_size) + 1

start_tm = time.time()
for n_batch in range(tot_batch):
    id_st = n_batch * batch_size
    if n_batch == (tot_batch-1):
        id_en = num_data
    else:
        id_en = (n_batch+1) * batch_size
    
    tmp_seq_in = np.zeros(
        [id_en-id_st, seq_length+2], dtype=np.int32)
    
    # Reset the batch input arrays. #
    tmp_seq_in[:, :] = PAD_token
    tmp_seq_in[:, 0] = CLS_token
    
    for n_index in range(id_en-id_st):
        tmp_index = id_st + n_index
        tmp_p_tok = tmp_data[tmp_index].split(" ")
        num_token = len(tmp_p_tok)

        if num_token <= seq_length:
            end_token = EOS_token
        else:
            tmp_p_tok = tmp_p_tok[:seq_length]
            end_token = TRU_token
        
        tmp_i_tok = [CLS_token]
        tmp_i_tok += tmp_p_tok
        tmp_i_tok += [end_token]
        tmp_p_idx = [word2idx.get(
            x, UNK_token) for x in tmp_i_tok]
        
        n_input = len(tmp_p_idx)
        tmp_seq_in[n_index, :n_input] = tmp_p_idx
    
    # Extract the document embedding. #
    tmp_seq_emb = bert_model(
        tmp_seq_in, training=False)[1][:, 0, :]
    doc_embed_list.append(tmp_seq_emb.numpy())

    if (n_batch+1) % 50 == 0:
        print(n_batch+1, "out of", tot_batch, "processed.")

# Concatenate into a numpy array to utilise #
# its optimised array operations codes.     #
doc_embeddings = np.concatenate(
    tuple(doc_embed_list), axis=0)
del doc_embed_list

elapsed_tm = (time.time() - start_tm) / 60.0
print("Elapsed time:", str(elapsed_tm), "mins.")

print("Testing the BERT Network", 
      "(" + str(n_iter) + " iterations).")
print(str(num_data), "training samples.")
print("-" * 50)

while True:
    tmp_in_phrase = input("Enter phrase: ")
    
    tmp_phrase = tmp_in_phrase.lower().strip()
    del tmp_in_phrase

    if tmp_phrase == "":
        break
    else:
        tmp_tokens = [x for x in \
            wordpunct_tokenize(tmp_phrase) if x != ""]
        
        if len(tmp_tokens) > seq_length:
            print("Too many tokens, truncating the input.")
            tmp_tokens = tmp_tokens[:seq_length]
            end_token = TRU_token
        else:
            end_token = EOS_token
        
        tmp_idx = [CLS_token]
        tmp_idx += [word2idx.get(
            x, UNK_token) for x in tmp_tokens]
        tmp_idx += [end_token]
        n_input = len(tmp_idx)
        
        # Reset the input array. #
        tmp_anchor_idx[:, :] = PAD_token
        tmp_anchor_idx[:, 0] = CLS_token
        tmp_anchor_idx[0, :n_input] = tmp_idx
        
        # Perform the model inference. #
        tmp_test = bert_model(
            tmp_anchor_idx, training=False)[1][:, 0, :]
        tmp_test = tmp_test.numpy()
        tmp_diff = np.square(tmp_test - doc_embeddings)
        tmp_dist = np.mean(tmp_diff, axis=1)
        
        # Show the top-k best matches. #
        tmp_sort_idx = list(np.argsort(tmp_dist))
        for n_display in range(topk):
            tmp_idx_display = tmp_sort_idx[n_display]

            tmp_str_display = "Top " + str(n_display+1)
            tmp_str_display += " match (distance = "
            tmp_str_display += str(tmp_dist[tmp_idx_display])

            print(tmp_str_display  + "):")
            print(tmp_data[tmp_idx_display])
            
            if n_display == (topk-1):
                print("-" * 50)
            else:
                print("=" * 50)
