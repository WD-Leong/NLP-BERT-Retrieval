import time
import numpy as np
import pandas as pd
import pickle as pkl

import tensorflow as tf
import tf_ver2_bert_keras as bert

# Function to sample the sequence. #
def prepare_input(
    token_input, word_2_idx, 
    seq_length, vocab_size, p_mask, 
    CLS_token, TRU_token, EOS_token, MSK_token):
    tmp_i_tok = [word_2_idx.get(
        x, UNK_token) for x in token_input]
    num_token = len(tmp_i_tok)

    # Truncate the sequence if it exceeds the maximum #
    # sequence length. Randomly select the review's   #
    # start and end index to be the positive example. #
    if num_token > seq_length:
        # For the anchor. #
        id_st = np.random.randint(
            0, num_token-seq_length)
        id_en = id_st + seq_length
        
        tmp_i_idx = [CLS_token]
        tmp_i_idx += tmp_i_tok[id_st:id_en]
        
        if id_en < num_token:
            # Add TRUNCATE token. #
            tmp_i_idx += [TRU_token]
        else:
            tmp_i_idx += [EOS_token]
        del id_st, id_en
    else:
        tmp_i_idx = [CLS_token] + tmp_i_tok
        tmp_i_idx += [EOS_token]
    n_input = len(tmp_i_idx)
    
    # Generate the masked sequence. #
    mask_seq  = [MSK_token] * n_input
    tmp_mask  = np.random.binomial(
        1, p_mask, size=n_input)
    
    tmp_noise = [CLS_token]
    tmp_noise += list(np.random.choice(
        vocab_size, size=n_input-2))
    tmp_noise += [tmp_i_idx[-1]]
    
    tmp_unif = np.random.uniform()
    if tmp_unif <= 0.8:
        # Replace with MASK token. #
        tmp_i_msk = [
            tmp_i_idx[x] if tmp_mask[x] == 0 else \
                mask_seq[x] for x in range(n_input)]
    elif tmp_unif <= 0.9:
        # Replace with random word. #
        tmp_i_msk = [
            tmp_i_idx[x] if tmp_mask[x] == 0 else \
                tmp_noise[x] for x in range(n_input)]
    else:
        # No replacement. #
        tmp_i_msk = tmp_i_idx
    return tmp_i_idx, tmp_i_msk, tmp_mask

# Define the weight update step for multiple sub-batches. #
def sub_batch_train_step(
    model, sub_batch_sz, 
    x_mask_anc, x_mask_pos, x_mask_neg, 
    x_in_anchor, x_in_positive, x_in_negative, 
    x_seq_anc, x_seq_pos, x_seq_neg, optimizer, 
    learning_rate=1.0e-3, grad_clip=1.0, alpha=5.0, beta=5.0):
    optimizer.lr.assign(learning_rate)
    
    batch_size = x_in_anchor.shape[0]
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
        
        tmp_in_anc = x_in_anchor[id_st:id_en, :]
        tmp_in_pos = x_in_positive[id_st:id_en, :]
        tmp_in_neg = x_in_negative[id_st:id_en, :]

        tmp_anc_msk = x_mask_anc[id_st:id_en, :]
        tmp_pos_msk = x_mask_pos[id_st:id_en, :]
        tmp_neg_msk = x_mask_neg[id_st:id_en, :]

        tmp_out_anc = x_seq_anc[id_st:id_en, :]
        tmp_out_pos = x_seq_pos[id_st:id_en, :]
        tmp_out_neg = x_seq_neg[id_st:id_en, :]
        
        with tf.GradientTape() as grad_tape:
            anc_outputs = model(tmp_in_anc, training=True)
            pos_outputs = model(tmp_in_pos, training=True)
            neg_outputs = model(tmp_in_neg, training=True)
            
            # Extract the CLS output embeddings. #
            embed_anc = anc_outputs[1][:, 0, :]
            embed_pos = pos_outputs[1][:, 0, :]
            embed_neg = neg_outputs[1][:, 0, :]

            # Extract the average embeddings. #
            anc_avg = tf.reduce_mean(
                anc_outputs[1][:, 1:, :], axis=1)
            pos_avg = tf.reduce_mean(
                pos_outputs[1][:, 1:, :], axis=1)
            neg_avg = tf.reduce_mean(
                neg_outputs[1][:, 1:, :], axis=1)

            # Extract the vocabulary logits. #
            anc_logits = anc_outputs[0]
            pos_logits = pos_outputs[0]
            neg_logits = neg_outputs[0]

            # Triplet loss. #
            tmp_pos_dist = tf.reduce_mean(
                tf.square(embed_anc - embed_pos), axis=1)
            tmp_neg_dist = tf.reduce_mean(
                tf.square(embed_anc - embed_neg), axis=1)
            triplet_loss = tf.maximum(
                0.0, tmp_pos_dist - tmp_neg_dist + alpha)
            
            # Masked Language Modeling loss. #
            msk_anc_xent = tf.multiply(
                tf.nn.sparse_softmax_cross_entropy_with_logits(
                    labels=tmp_out_anc, logits=anc_logits), tmp_anc_msk)
            
            msk_pos_xent = tf.multiply(
                tf.nn.sparse_softmax_cross_entropy_with_logits(
                    labels=tmp_out_pos, logits=pos_logits), tmp_anc_msk)
            
            msk_neg_xent = tf.multiply(
                tf.nn.sparse_softmax_cross_entropy_with_logits(
                    labels=tmp_out_neg, logits=neg_logits), tmp_anc_msk)
            
            num_anc_mask = tf.cast(
                tf.reduce_sum(tmp_anc_msk, axis=1), tf.float32)
            num_pos_mask = tf.cast(
                tf.reduce_sum(tmp_pos_msk, axis=1), tf.float32)
            num_neg_mask = tf.cast(
                tf.reduce_sum(tmp_neg_msk, axis=1), tf.float32)
            
            msk_anc_losses = tf.reduce_sum(tf.math.divide_no_nan(
                tf.reduce_sum(msk_anc_xent, axis=1), num_anc_mask))
            msk_pos_losses = tf.reduce_sum(tf.math.divide_no_nan(
                tf.reduce_sum(msk_pos_xent, axis=1), num_pos_mask))
            msk_neg_losses = tf.reduce_sum(tf.math.divide_no_nan(
                tf.reduce_sum(msk_neg_xent, axis=1), num_neg_mask))
            
            # Regularize the CLS embeddings to the average #
            # embeddings of the rest of the sequence.      #
            anc_emb_losses = tf.reduce_sum(tf.reduce_mean(
                tf.square(embed_anc - anc_avg), axis=1))
            pos_emb_losses = tf.reduce_sum(tf.reduce_mean(
                tf.square(embed_pos - pos_avg), axis=1))
            neg_emb_losses = tf.reduce_sum(tf.reduce_mean(
                tf.square(embed_neg - neg_avg), axis=1))
            
            # Total loss in this sub-batch. #
            msk_losses = tf.add(tf.add(
                msk_anc_losses, msk_pos_losses), msk_neg_losses)
            emb_losses = tf.add(tf.add(
                anc_emb_losses, pos_emb_losses), neg_emb_losses)
            pre_losses = msk_losses + emb_losses
            tmp_losses = tf.add(
                pre_losses, beta * tf.reduce_sum(triplet_loss))
        
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
    
    clipped_tuple = tf.clip_by_global_norm(
        acc_gradients, grad_clip)
    optimizer.apply_gradients(
        zip(clipped_tuple[0], model_params))
    return average_loss

# Model Parameters. #
batch_size = 256
sub_batch  = 64
seq_length = 25
num_heads  = 4
num_layers = 3

gradient_clip = 1.00
maximum_iter  = 3000
restore_flag  = True
save_step     = 100
warmup_steps  = 2000
display_step  = 50
anneal_step   = 2500
anneal_rate   = 0.75

prob_mask = 0.15
prob_keep = 0.90
hidden_size = 256
ffwd_size   = 4*hidden_size
warmup_flag = True
cooling_step = 100

model_ckpt_dir  = "TF_Models/bert_reddit"
train_loss_file = "train_loss_bert_reddit.csv"

# Load the data. #
tmp_pkl_file = "../../Data/reddit_jokes/reddit_jokes.pkl"
with open(tmp_pkl_file, "rb") as tmp_load_file:
    tmp_data = pkl.load(tmp_load_file)
    idx2word = pkl.load(tmp_load_file)
    word2idx = pkl.load(tmp_load_file)

vocab_size = len(word2idx)
print("Vocabulary Size:", str(vocab_size)+".")

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

# Build the Transformer. #
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
tmp_anc_in = np.zeros(
    [batch_size, seq_length+2], dtype=np.int32)
tmp_pos_in = np.zeros(
    [batch_size, seq_length+2], dtype=np.int32)
tmp_neg_in = np.zeros(
    [batch_size, seq_length+2], dtype=np.int32)

tmp_anc_out = np.zeros(
    [batch_size, seq_length+2], dtype=np.int32)
tmp_pos_out = np.zeros(
    [batch_size, seq_length+2], dtype=np.int32)
tmp_neg_out = np.zeros(
    [batch_size, seq_length+2], dtype=np.int32)

tmp_mask_anc = np.zeros(
    [batch_size, seq_length+2], dtype=np.int32)
tmp_mask_pos = np.zeros(
    [batch_size, seq_length+2], dtype=np.int32)
tmp_mask_neg = np.zeros(
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
    
    # Reset the batch input arrays. #
    tmp_mask_anc[:, :] = 1.0
    tmp_mask_pos[:, :] = 1.0
    tmp_mask_neg[:, :] = 1.0

    tmp_anc_in[:, :] = PAD_token
    tmp_pos_in[:, :] = PAD_token
    tmp_neg_in[:, :] = PAD_token
    
    tmp_anc_out[:, :] = PAD_token
    tmp_pos_out[:, :] = PAD_token
    tmp_neg_out[:, :] = PAD_token
    
    for n_index in range(batch_size):
        tmp_index1 = batch_sample[n_index]
        tmp_random = tmp_index1 + batch_add_int[n_index]
        tmp_index2 = tmp_random % num_data
        
        tmp_p_tok1 = tmp_data[tmp_index1].split(" ")
        tmp_p_tok2 = tmp_data[tmp_index2].split(" ")
        
        # Randomly sample the positive input #
        # to be used as the anchor.          #
        n_input1 = len(tmp_p_tok1)
        if n_input1 <= 2:
            n_sample = 1
        else:
            n_sample = np.random.randint(1, n_input1-1)

        tmp_sample = list(sorted(list(
            np.random.permutation(n_input1)[:n_sample])))
        tmp_p_anc  = [tmp_p_tok1[x] for x in tmp_sample]
        
        # tmp_i_idx, tmp_i_msk, tmp_mask
        tmp_anc_tuple = prepare_input(
            tmp_p_anc, word2idx, 
            seq_length, vocab_size, prob_mask, 
            CLS_token, TRU_token, EOS_token, MSK_token)
        
        tmp_pos_tuple = prepare_input(
            tmp_p_tok1, word2idx, 
            seq_length, vocab_size, prob_mask, 
            CLS_token, TRU_token, EOS_token, MSK_token)
        
        tmp_neg_tuple = prepare_input(
            tmp_p_tok2, word2idx, 
            seq_length, vocab_size, prob_mask, 
            CLS_token, TRU_token, EOS_token, MSK_token)
        
        n_anc = len(tmp_anc_tuple[0])
        n_pos = len(tmp_pos_tuple[0])
        n_neg = len(tmp_neg_tuple[0])
        del tmp_sample, n_sample
        
        tmp_anc_in[n_index, :n_anc] = tmp_anc_tuple[1]
        tmp_pos_in[n_index, :n_pos] = tmp_pos_tuple[1]
        tmp_neg_in[n_index, :n_neg] = tmp_neg_tuple[1]

        tmp_anc_out[n_index, :n_anc] = tmp_anc_tuple[0]
        tmp_pos_out[n_index, :n_pos] = tmp_pos_tuple[0]
        tmp_neg_out[n_index, :n_neg] = tmp_neg_tuple[0]

        tmp_mask_anc[n_index, :n_anc] = tmp_anc_tuple[2]
        tmp_mask_pos[n_index, :n_pos] = tmp_pos_tuple[2]
        tmp_mask_neg[n_index, :n_neg] = tmp_neg_tuple[2]
    
    tmp_loss = sub_batch_train_step(
        bert_model, sub_batch, 
        tmp_mask_anc, tmp_mask_pos, tmp_mask_neg, 
        tmp_anc_in, tmp_pos_in, tmp_neg_in, 
        tmp_anc_out, tmp_pos_out, tmp_neg_out, 
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
        
        tmp_test_anc[:, 0] = CLS_token
        tmp_test_pos[:, 0] = CLS_token
        tmp_test_neg[:, 0] = CLS_token
        
        sample_add = np.random.randint(
            1, num_data, size=1)
        sample_pos = np.random.choice(num_data, size=1)
        sample_neg = (sample_pos + sample_add) % num_data
        
        tmp_in_pos = tmp_data[sample_pos[0]]
        tmp_in_neg = tmp_data[sample_neg[0]]
        
        tmp_p_pos = tmp_in_pos.split(" ")
        tmp_p_tuple = prepare_input(
            tmp_p_pos, word2idx, 
            seq_length, vocab_size, prob_mask, 
            CLS_token, TRU_token, EOS_token, MSK_token)
        
        tmp_n_tuple = prepare_input(
            tmp_in_neg.split(" "), word2idx, 
            seq_length, vocab_size, prob_mask, 
            CLS_token, TRU_token, EOS_token, MSK_token)
        
        if len(tmp_p_pos) <= 2:
            n_sample = 1
        else:
            n_sample = np.random.randint(1, len(tmp_p_pos))
        
        tmp_sample = list(sorted(
            list(np.random.permutation(
                len(tmp_p_pos))[:n_sample])))
        
        tmp_in_anc = tmp_in_pos.split(" ")
        tmp_in_anc = " ".join(
            [tmp_in_anc[x] for x in tmp_sample])
        tmp_p_anc  = [x for x in tmp_in_anc.split(" ")]

        tmp_a_tuple = prepare_input(
            tmp_p_anc, word2idx, 
            seq_length, vocab_size, prob_mask, 
            CLS_token, TRU_token, EOS_token, MSK_token)
        
        tmp_i_anc  = tmp_a_tuple[0]
        tmp_i_pos  = tmp_p_tuple[0]
        tmp_i_neg  = tmp_n_tuple[0]

        n_anc_toks = len(tmp_i_anc)
        n_pos_toks = len(tmp_i_pos)
        n_neg_toks = len(tmp_i_neg)
        del tmp_sample, n_sample
        
        tmp_test_anc[0, :n_anc_toks] = tmp_i_anc
        tmp_test_pos[0, :n_pos_toks] = tmp_i_pos
        tmp_test_neg[0, :n_neg_toks] = tmp_i_neg
        
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

