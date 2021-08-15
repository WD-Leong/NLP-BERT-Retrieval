import time
import numpy as np
import pickle as pkl

import tensorflow as tf
import tensorflow_addons as tfa
import tf_ver2_bert_network as bert
from nltk.tokenize import wordpunct_tokenize

# Model Parameters. #
topk = 5
batch_size = 128
seq_length = 100
num_heads  = 4
num_layers = 3
prob_keep  = 0.9
hidden_size = 256
ffwd_size   = 4*hidden_size

tmp_path = "C:/Users/admin/Desktop/Codes/"
model_ckpt_dir  = "../../TF_Models/yelp_bert"
train_loss_file = "train_loss_yelp_bert.csv"

# Load the data. #
tmp_pkl_file = "C:/Users/admin/Desktop/Data/Yelp/"
tmp_pkl_file += "yelp_review_polarity_csv/train_gpt_data.pkl"
with open(tmp_pkl_file, "rb") as tmp_load_file:
    train_data = pkl.load(tmp_load_file)
    word_vocab = pkl.load(tmp_load_file)
    idx_2_word = pkl.load(tmp_load_file)
    word_2_idx = pkl.load(tmp_load_file)

vocab_size = len(word_vocab)
print("Vocabulary Size:", str(vocab_size) + ".")

num_data  = len(train_data)
CLS_token = word_2_idx["CLS"]
EOS_token = word_2_idx["EOS"]
PAD_token = word_2_idx["PAD"]
UNK_token = word_2_idx["UNK"]
print("Total of", str(len(train_data)), "rows loaded.")

# Set the number of threads to use. #
tf.config.threading.set_intra_op_parallelism_threads(1)
tf.config.threading.set_inter_op_parallelism_threads(1)

# Build the Transformer. #
print("Building the BERT Model.")
start_time = time.time()

bert_model = bert.BERT_Network(
    num_layers, num_heads, 
    hidden_size, ffwd_size, word_2_idx, 
    seq_length+2, p_keep=prob_keep)
bert_optim = tfa.optimizers.AdamW(
    beta_1=0.9, beta_2=0.98, 
    epsilon=1.0e-9, weight_decay=1.0e-4)

elapsed_time = (time.time()-start_time) / 60
print("BERT Model Built", 
      "(" + str(elapsed_time) + " mins).")

# Create the model checkpoint. #
ckpt = tf.train.Checkpoint(
    step=tf.Variable(0), 
    bert_model=bert_model, 
    bert_optim=bert_optim)

manager = tf.train.CheckpointManager(
    ckpt, model_ckpt_dir, max_to_keep=1)

ckpt.restore(manager.latest_checkpoint)
if manager.latest_checkpoint:
    print("Model restored from {}".format(
        manager.latest_checkpoint))
n_iter = ckpt.step.numpy().astype(np.int32)

# Generate the document embeddings of the BERT model. #
doc_embed_list = []
tmp_anchor_idx = np.zeros(
    [1, seq_length+2], dtype=np.int32)

print("-" * 50)
print("Testing the BERT Network", 
      "(" + str(n_iter) + " iterations).")
print(str(num_data), "training samples.")
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
    
    tmp_seq_in[:, :] = PAD_token
    tmp_seq_in[:, 0] = CLS_token
    
    for n_index in range(id_en-id_st):
        tmp_index = id_st + n_index
        tmp_p_tok = train_data[tmp_index][1]
        tmp_p_idx = [word_2_idx.get(
            x, UNK_token) for x in tmp_p_tok]
        
        n_input = len(tmp_p_idx) + 1
        tmp_seq_in[n_index, 1:n_input] = tmp_p_idx
        tmp_seq_in[n_index, n_input] = EOS_token
    
    tmp_seq_emb = bert_model(
        tmp_seq_in, training=False)[1][:, 0, :]
    doc_embed_list.append(tmp_seq_emb.numpy())
    
    if (n_batch+1) % 100 == 0:
        percent_complete = (n_batch+1) / tot_batch
        print(str(round(percent_complete*100, 2)) + "% complete.")
    
doc_embeddings = np.concatenate(
    tuple(doc_embed_list), axis=0)
del doc_embed_list

yelp_data = [" ".join(x[1]) for x in train_data]
elapsed_tm = (time.time() - start_tm) / 60.0
print("Elapsed time:", str(elapsed_tm), "mins.")

while True:
    tmp_in_phrase = input("Enter phrase: ")
    
    tmp_phrase = tmp_in_phrase.lower().strip()
    if tmp_phrase == "":
        break
    else:
        tmp_tokens = [
            x for x in wordpunct_tokenize(tmp_phrase) if x != ""]
        
        if len(tmp_tokens) > seq_length:
            print("Too many tokens.")
            continue
        else:
            tmp_idx = [word_2_idx.get(
                x, UNK_token) for x in tmp_tokens]
            n_input = len(tmp_idx) + 1
            
            tmp_anchor_idx[:, :] = PAD_token
            tmp_anchor_idx[:, 0] = CLS_token
            tmp_anchor_idx[0, n_input] = EOS_token
            tmp_anchor_idx[0, 1:n_input] = tmp_idx
            
            tmp_test = bert_model(
                tmp_anchor_idx, training=False)[1][:, 0, :]
            tmp_test = tmp_test.numpy()
            tmp_diff = np.square(tmp_test - doc_embeddings)
            tmp_dist = np.mean(tmp_diff, axis=1)
            
            # Show the top 5 best match. #
            tmp_sort_idx = list(np.argsort(tmp_dist))
            for n_display in range(topk):
                tmp_idx_display = tmp_sort_idx[n_display]
                print("Top", str(n_display+1), "match (distance =", 
                      str(tmp_dist[tmp_idx_display]) + "):")
                print(yelp_data[tmp_idx_display])
                
                if n_display == (topk-1):
                    print("-" * 50)
                else:
                    print("=" * 50)

