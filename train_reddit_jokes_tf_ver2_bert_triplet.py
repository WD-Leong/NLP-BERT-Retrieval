import time
import numpy as np
import pandas as pd
import pickle as pkl

import tensorflow as tf
import tf_ver2_bert_network as bert

# Define the weight update step for multiple sub-batches. #
#@tf.function
def sub_batch_train_step(
    model, sub_batch_sz, 
    x_anchor, x_positive, x_negative, optimizer, 
    learning_rate=1.0e-3, grad_clip=1.0, alpha=5.0):
    optimizer.lr.assign(learning_rate)
    
    batch_size = x_anchor.shape[0]
    if batch_size <= sub_batch_sz:
        sub_batch = 1
    elif batch_size % sub_batch_sz == 0:
        sub_batch = int(batch_size / sub_batch_sz)
    else:
        sub_batch = int(batch_size / sub_batch_sz) + 1
    
    model_params  = model.trainable_variables
    acc_gradients = [
        tf.zeros_like(var) for var in model_params]
    
    tot_losses = 0.0
    for n_sub in range(sub_batch):
        id_st = n_sub*sub_batch_sz
        if n_sub != (sub_batch-1):
            id_en = (n_sub+1)*sub_batch_sz
        else:
            id_en = batch_size
        
        tmp_anchor = x_anchor[id_st:id_en, :]
        tmp_positive = x_positive[id_st:id_en, :]
        tmp_negative = x_negative[id_st:id_en, :]
        
        with tf.GradientTape() as grad_tape:
            embed_anc = model(
                tmp_anchor, training=True)[1][:, 0, :]
            embed_pos = model(
                tmp_positive, training=True)[1][:, 0, :]
            embed_neg = model(
                tmp_negative, training=True)[1][:, 0, :]
            
            # Triplet loss. #
            tmp_pos_dist = tf.reduce_mean(tf.square(
                embed_anc - embed_pos), axis=1)
            tmp_neg_dist = tf.reduce_mean(tf.square(
                embed_anc - embed_neg), axis=1)
            triplet_loss = tf.maximum(
                0.0, tmp_pos_dist - tmp_neg_dist + alpha)
            
            tmp_losses = tf.reduce_sum(triplet_loss)
#            tmp_losses = tf.reduce_sum(tf.reduce_sum(
#                tf.nn.sparse_softmax_cross_entropy_with_logits(
#                    labels=tmp_output, logits=output_logits), axis=1))
        
        # Accumulate the gradients. #
        tot_losses += tmp_losses
        tmp_gradients = grad_tape.gradient(
            tmp_losses, model_params)
        acc_gradients = [tf.add(
            acc_grad, grad) for acc_grad, grad \
                in zip(acc_gradients, tmp_gradients)]
    
    # Update using the optimizer. #
    average_loss  = tot_losses / batch_size
    acc_gradients = [tf.math.divide_no_nan(
        acc_grad, batch_size) for acc_grad in acc_gradients]
    
    clipped_gradients, _ = tf.clip_by_global_norm(
        acc_gradients, grad_clip)
    optimizer.apply_gradients(
        zip(clipped_gradients, model_params))
    return average_loss

# Model Parameters. #
batch_size = 256
sub_batch  = 64
seq_length = 30
num_heads  = 4
num_layers = 3

gradient_clip = 1.00
maximum_iter  = 3000
restore_flag  = True
save_step     = 500
warmup_steps  = 2000
display_step  = 50
anneal_step   = 2500
anneal_rate   = 0.75

prob_keep = 0.9
hidden_size = 256
ffwd_size   = 4*hidden_size
warmup_flag = True
cooling_step = 500

model_ckpt_dir  = "../../TF_Models/bert_reddit"
train_loss_file = "train_loss_bert_reddit.csv"

# Load the data. #
tmp_pkl_file = \
    "C:/Users/admin/Desktop/Codes/reddit_jokes.pkl"
with open(tmp_pkl_file, "rb") as tmp_load_file:
    full_data = pkl.load(tmp_load_file)
    idx2word = pkl.load(tmp_load_file)
    word2idx = pkl.load(tmp_load_file)

vocab_size = len(word2idx)
print("Vocabulary Size:", str(vocab_size)+".")

tmp_data = []
for tmp_row in full_data:
    if len(tmp_row.split(" ")) > 1 and \
        len(tmp_row.split(" ")) <= seq_length:
        tmp_data.append(tmp_row)

num_data  = len(tmp_data)
SOS_token = word2idx["SOS"]
EOS_token = word2idx["EOS"]
PAD_token = word2idx["PAD"]
UNK_token = word2idx["UNK"]
print("Total of", str(len(tmp_data)), "rows loaded.")

# Set the number of threads to use. #
tf.config.threading.set_intra_op_parallelism_threads(1)
tf.config.threading.set_inter_op_parallelism_threads(1)

# Build the Transformer. #
print("Building the BERT Model.")
start_time = time.time()

bert_model = bert.BERT_Network(
    num_layers, num_heads, 
    hidden_size, ffwd_size, word2idx, 
    seq_length+2, p_keep=prob_keep)
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

if restore_flag:
    ckpt.restore(manager.latest_checkpoint)
    if manager.latest_checkpoint:
        print("Model restored from {}".format(
            manager.latest_checkpoint))
    else:
        print("Error: No latest checkpoint found.")
    
    train_loss_df = pd.read_csv(train_loss_file)
    train_loss_list = [tuple(
        train_loss_df.iloc[x].values) \
        for x in range(len(train_loss_df))]
else:
    print("Training a new model.")
    train_loss_list = []

# Train the Transformer model. #
tmp_seq_anc = np.zeros(
    [batch_size, seq_length+2], dtype=np.int32)
tmp_seq_pos = np.zeros(
    [batch_size, seq_length+2], dtype=np.int32)
tmp_seq_neg = np.zeros(
    [batch_size, seq_length+2], dtype=np.int32)

tmp_test_anc = np.zeros(
    [1, seq_length+2], dtype=np.int32)
tmp_test_pos = np.zeros(
    [1, seq_length+2], dtype=np.int32)
tmp_test_neg = np.zeros(
    [1, seq_length+2], dtype=np.int32)

# Warmup learning schedule. #
n_iter = ckpt.step.numpy().astype(np.int32)
if warmup_flag:
    step_min = float(max(n_iter, warmup_steps))**(-0.5)
    learning_rate = float(hidden_size)**(-0.5) * step_min
else:
    initial_lr = 0.001
    anneal_pow = int(n_iter / anneal_step)
    learning_rate = max(np.power(
        anneal_rate, anneal_pow) * initial_lr, 1.0e-5)

print("-" * 50)
print("Training the BERT Network", 
      "(" + str(n_iter) + " iterations).")
print(str(num_data), "training samples.")
print("-" * 50)

# Update the neural network's weights. #
tot_loss = 0.0
start_tm = time.time()
while n_iter < maximum_iter:
    if warmup_flag:
        step_min = float(max(n_iter, warmup_steps))**(-0.5)
        learning_rate = float(hidden_size)**(-0.5) * step_min
    else:
        if n_iter % anneal_step == 0:
            anneal_pow = int(n_iter / anneal_step)
        learning_rate = max(np.power(
            anneal_rate, anneal_pow) * initial_lr, 1.0e-6)
    
    # Select a sample from the data. #
    batch_sample = np.random.choice(
        num_data, size=batch_size, replace=False)
    batch_add_int = np.random.randint(
        1, num_data, size=batch_size)
    
    # For simplicity, take SOS to be CLS. #
    tmp_seq_anc[:, :] = PAD_token
    tmp_seq_pos[:, :] = PAD_token
    tmp_seq_neg[:, :] = PAD_token
    
    tmp_seq_anc[:, 0] = SOS_token
    tmp_seq_pos[:, 0] = SOS_token
    tmp_seq_neg[:, 0] = SOS_token
    
    for n_index in range(batch_size):
        tmp_index1 = batch_sample[n_index]
        tmp_index2 = \
            (tmp_index1 + batch_add_int[n_index]) % num_data
        
        tmp_p_tok1 = tmp_data[tmp_index1].split(" ")
        tmp_p_tok2 = tmp_data[tmp_index2].split(" ")
        tmp_p_idx1 = [word2idx.get(
            x, UNK_token) for x in tmp_p_tok1]
        tmp_p_idx2 = [word2idx.get(
            x, UNK_token) for x in tmp_p_tok2]
        
        n_input1 = len(tmp_p_idx1) + 1
        n_input2 = len(tmp_p_idx2) + 1
        
        # Randomly sample the positive input #
        # to be used as the anchor.          #
        num_sample = np.random.randint(1, n_input1-1)
        tmp_sample = list(sorted(list(
            np.random.permutation(n_input1-1)[:num_sample])))
        
        tmp_p_anc = list(
            np.array(tmp_p_idx1)[tmp_sample])
        n_anchor  = len(tmp_p_anc) + 1
        del tmp_sample, num_sample
        
        tmp_seq_anc[n_index, 1:n_anchor] = tmp_p_anc
        tmp_seq_pos[n_index, 1:n_input1] = tmp_p_idx1
        tmp_seq_neg[n_index, 1:n_input2] = tmp_p_idx2
        
        tmp_seq_anc[n_index, n_anchor] = EOS_token
        tmp_seq_pos[n_index, n_input1] = EOS_token
        tmp_seq_neg[n_index, n_input2] = EOS_token
    
    tmp_loss = sub_batch_train_step(
        bert_model, sub_batch, 
        tmp_seq_anc, tmp_seq_pos, tmp_seq_neg, 
        bert_optim, learning_rate=learning_rate)

    n_iter += 1
    ckpt.step.assign_add(1)
    
    tot_loss += tmp_loss.numpy()
    if n_iter % display_step == 0:
        end_tm = time.time()
        
        avg_loss = tot_loss / display_step
        tot_loss = 0.0
        elapsed_tm = (end_tm - start_tm) / 60
        
        tmp_test_anc[:, :] = PAD_token
        tmp_test_pos[:, :] = PAD_token
        tmp_test_neg[:, :] = PAD_token
        
        tmp_test_anc[:, 0] = SOS_token
        tmp_test_pos[:, 0] = SOS_token
        tmp_test_neg[:, 0] = SOS_token
        
        sample_add = np.random.randint(
            1, num_data, size=1)
        sample_pos = np.random.choice(num_data, size=1)
        sample_neg = (sample_pos + sample_add) % num_data
        
        tmp_in_pos = tmp_data[sample_pos[0]]
        tmp_in_neg = tmp_data[sample_neg[0]]
        
        tmp_p_pos = [word2idx.get(
            x, UNK_token) for x in tmp_in_pos.split(" ")]
        tmp_p_neg = [word2idx.get(
            x, UNK_token) for x in tmp_in_neg.split(" ")]
        
        n_pos_toks = len(tmp_p_pos) + 1
        n_neg_toks = len(tmp_p_neg) + 1
        num_sample = np.random.randint(1, n_pos_toks-1)
        tmp_sample = list(sorted(list(
            np.random.permutation(n_pos_toks-1)[:num_sample])))
        tmp_in_anc = tmp_in_pos.split(" ")
        tmp_in_anc = " ".join(
            [tmp_in_anc[x] for x in tmp_sample])
        
        tmp_p_anc = list(
            np.array(tmp_p_pos)[tmp_sample])
        n_anc_toks = len(tmp_p_anc) + 1
        del tmp_sample, num_sample
        
        tmp_test_anc[0, 1:n_anc_toks] = tmp_p_anc
        tmp_test_pos[0, 1:n_pos_toks] = tmp_p_pos
        tmp_test_neg[0, 1:n_neg_toks] = tmp_p_neg
        
        tmp_test_anc[0, n_anc_toks] = EOS_token
        tmp_test_pos[0, n_pos_toks] = EOS_token
        tmp_test_neg[0, n_neg_toks] = EOS_token
        
        tmp_anc_emb = bert_model(
            tmp_test_anc, training=False)[1][:, 0, :]
        tmp_pos_emb = bert_model(
            tmp_test_pos, training=False)[1][:, 0, :]
        tmp_neg_emb = bert_model(
            tmp_test_neg, training=False)[1][:, 0, :]
        del sample_pos, sample_neg, sample_add
        del n_pos_toks, n_neg_toks, n_anc_toks
        
        pos_dist = tf.reduce_mean(tf.square(
            tmp_anc_emb - tmp_pos_emb), axis=1)[0]
        neg_dist = tf.reduce_mean(tf.square(
            tmp_anc_emb - tmp_neg_emb), axis=1)[0]
        
        print("Iteration", str(n_iter)+".")
        print("Elapsed Time:", str(elapsed_tm), "mins.")
        print("Gradient Clip:", str(gradient_clip) + ".")
        print("Learning Rate:", str(learning_rate) + ".")
        print("Average Loss:", str(avg_loss) + ".")
        print("")
        
        print("Anchor:", tmp_in_anc)
        print("Positive:", tmp_in_pos)
        print("Positive Dist:", str(pos_dist.numpy()))
        print("Negative:", tmp_in_neg)
        print("Negative Dist:", str(neg_dist.numpy()))
        
        train_loss_list.append((n_iter, avg_loss))
        start_tm = time.time()
        print("-" * 50)
    
    # Save the model. #
    if n_iter % save_step == 0:
        # Save the model. #
        save_path = manager.save()
        print("Saved model to {}".format(save_path))
        
        tmp_df_losses = pd.DataFrame(
            train_loss_list, columns=["n_iter", "xent_loss"])
        tmp_df_losses.to_csv(train_loss_file, index=False)
        del tmp_df_losses
    
    # Cool the GPU. #
    if n_iter % cooling_step == 0:
        print("Cooling GPU for 2 minutes.")
        time.sleep(120)
        print("Resume Training.")
        print("-" * 50)

