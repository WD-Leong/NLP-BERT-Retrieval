
import time
import numpy as np
import pandas as pd
import pickle as pkl

import tensorflow as tf
import tensorflow_addons as tfa
import tf_ver2_bert_keras_v1 as bert

# Function to sample the sequence. #
def prepare_input(
    token_input, word_2_idx, 
    seq_length, vocab_size, p_mask, 
    CLS_token, TRU_token, EOS_token, MSK_token):
    tmp_i_tok = [word_2_idx.get(
        x, UNK_token) for x in token_input]
    num_token = len(tmp_i_tok)

    # Truncate the sequence if it exceeds the maximum #
    # sequence length. Randomly select the document's #
    # start and end index to be truncated.            #
    if num_token > seq_length:
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
    model, optimizer, sub_batch_sz, x_msk_data, 
    x_in_data, x_seq_anc, x_seq_data, alpha=10.0, beta=1.0, 
    gamma=1.0, learning_rate=1.0e-3, gradient_clip=1.0):
    optimizer.lr.assign(learning_rate)
    
    batch_size = x_in_data.shape[0]
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
    mlm_losses = 0.0
    for n_sub in range(sub_batch):
        id_st = n_sub*sub_batch_sz
        if n_sub != (sub_batch-1):
            id_en = (n_sub+1)*sub_batch_sz
        else:
            id_en = batch_size
        
        tmp_in_data  = x_in_data[id_st:id_en, :]
        tmp_data_msk = x_msk_data[id_st:id_en, :]
        tmp_out_anc  = x_seq_anc[id_st:id_en, :]
        tmp_out_data = x_seq_data[id_st:id_en, :]
        
        num_sub_batch = id_en - id_st
        with tf.GradientTape() as grad_tape:
            anc_outputs  = model(tmp_out_anc, training=True)
            data_outputs = model(tmp_out_data, training=True)

            # Extract the average embeddings. #
            anc_avg  = tf.expand_dims(
                tf.reduce_mean(anc_outputs[1], axis=1), axis=0)
            data_avg = tf.expand_dims(
                tf.reduce_mean(data_outputs[1], axis=1), axis=1)
            
            # Euclidean Distance Loss. #
            tmp_dist = tf.reduce_sum(
                tf.square(anc_avg - data_avg), axis=2)

            # Extract the vocabulary logits. #
            data_logits = model(tmp_in_data, training=True)[0]

            # Euclidean Distance Loss. #
            diag_mask = tf.linalg.band_part(tf.ones(
                [num_sub_batch, num_sub_batch]), 0, 0)
            diag_mask = tf.expand_dims(diag_mask, axis=0)
            
            non_diag_mask = 1.0 - diag_mask
            num_non_diag  = tf.reduce_sum(non_diag_mask)
            
            # Modification of the Triplet Loss. #
            pos_distance = diag_mask * tmp_dist
            neg_distance = tf.maximum(
                0.0, alpha - (non_diag_mask * tmp_dist))
            
            pos_distance = tf.reduce_sum(
                tf.divide(pos_distance, num_sub_batch))
            neg_distance = tf.reduce_sum(
                tf.divide(neg_distance, num_non_diag))
            
            # Masked Language Modeling loss. #
            msk_xent_loss = tf.multiply(
                tf.nn.sparse_softmax_cross_entropy_with_logits(
                    labels=tmp_out_data, logits=data_logits), tmp_data_msk)
            num_data_mask = tf.cast(
                tf.reduce_sum(tmp_data_msk, axis=1), tf.float32)
            msk_xent_loss = tf.reduce_sum(tf.math.divide_no_nan(
                tf.reduce_sum(msk_xent_loss, axis=1), num_data_mask))
            
            # Modeling loss of non-masked tokens. #
            tmp_non_msk  = 1.0 - tmp_data_msk
            non_msk_xent = tf.multiply(
                tf.nn.sparse_softmax_cross_entropy_with_logits(
                    labels=tmp_out_data, logits=data_logits), tmp_non_msk)
            num_non_mask = tf.cast(tf.reduce_sum(
                1.0-tmp_data_msk, axis=1), tf.float32)
            non_msk_xent = tf.reduce_sum(tf.math.divide_no_nan(
                tf.reduce_sum(non_msk_xent, axis=1), num_non_mask))
            
            # Compute the individual component losses. #
            tot_dist_loss = pos_distance + neg_distance
            tot_mask_loss = gamma * non_msk_xent + msk_xent_loss

            # Total loss in this sub-batch. #
            tmp_losses = tot_mask_loss + beta * tot_dist_loss
        
        # Accumulate the gradients. #
        tot_losses += tmp_losses
        mlm_losses += tot_mask_loss

        tmp_gradients = grad_tape.gradient(
            tmp_losses, model_params)
        acc_gradients = [tf.add(
            acc_grad, grad) for acc_grad, grad \
                in zip(acc_gradients, tmp_gradients)]
    
    # Update using the optimizer. #
    average_loss = tot_losses / batch_size
    avg_msk_loss = mlm_losses / batch_size

    acc_gradients = [tf.math.divide_no_nan(
        acc_grad, batch_size) for acc_grad in acc_gradients]
    
    clip_tuple = tf.clip_by_global_norm(
        acc_gradients, gradient_clip)
    optimizer.apply_gradients(
        zip(clip_tuple[0], model_params))
    return average_loss, avg_msk_loss

# Model Parameters. #
batch_size = 256
sub_batch  = 64
seq_length = 25
num_heads  = 4
num_layers = 3

gradient_clip = 1.00
maximum_iter  = 2000
restore_flag  = True
save_step     = 100
warmup_steps  = 3000
display_step  = 50
anneal_step   = 2500
anneal_rate   = 0.75
n_max_samples = 5

prob_mask = 0.15
prob_keep = 0.90
hidden_size = 256
ffwd_size   = 4*hidden_size
warmup_flag  = True
cooling_step = 100

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
print("Vocabulary Size:", str(vocab_size)+".")

# Define the special tokens. #
num_data  = len(jokes_data)
CLS_token = word_2_idx["[CLS]"]
EOS_token = word_2_idx["[EOS]"]
PAD_token = word_2_idx["[PAD]"]
UNK_token = word_2_idx["[UNK]"]
TRU_token = word_2_idx["[TRU]"]
MSK_token = word_2_idx["[MSK]"]
print("Total of", str(num_data), "rows loaded.")

# Set the number of threads to use. #
tf.config.threading.set_intra_op_parallelism_threads(1)
tf.config.threading.set_inter_op_parallelism_threads(1)

# Build the Transformer. #
print("Building the BERT Model.")
start_time = time.time()

bert_model = bert.BERT(
    num_layers, num_heads, hidden_size, ffwd_size, 
    vocab_size, seq_length+2, rate=1.0-prob_keep)
bert_optim = tfa.optimizers.AdamW(
    weight_decay=1.0e-3)

elapsed_time = (time.time()-start_time) / 60
print("BERT Model Built", "(" + str(elapsed_time), "mins).")

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
tmp_in_data  = np.zeros(
    [batch_size, seq_length+2], dtype=np.int32)
tmp_out_anc  = np.zeros(
    [batch_size, seq_length+2], dtype=np.int32)
tmp_out_data = np.zeros(
    [batch_size, seq_length+2], dtype=np.int32)

tmp_mask_data = np.zeros(
    [batch_size, seq_length+2], dtype=np.int32)
tmp_test_anc  = np.zeros(
    [1, seq_length+2], dtype=np.int32)
tmp_test_data = np.zeros(
    [2, seq_length+2], dtype=np.int32)

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
mlm_loss = 0.0
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
    tmp_mask_data[:, :] = 0.0
    
    tmp_in_data[:, :]  = PAD_token
    tmp_out_anc[:, :]  = PAD_token
    tmp_out_data[:, :] = PAD_token
    
    for n_index in range(batch_size):
        tmp_index1 = batch_sample[n_index]
        tmp_random = tmp_index1 + batch_add_int[n_index]
        tmp_index2 = tmp_random % num_data
        
        tmp_p_tok1 = jokes_data[tmp_index1].lower().split(" ")
        tmp_p_tok2 = jokes_data[tmp_index2].lower().split(" ")
        
        # Randomly sample the positive input #
        # to be used as the anchor.          #
        n_input1 = len(tmp_p_tok1)
        if n_input1 <= 2:
            n_sample = 1
        else:
            n_sample = np.random.randint(1, n_max_samples)
        n_sample = min(n_sample, n_input1)

        tmp_sample = list(sorted(list(
            np.random.permutation(n_input1)[:n_sample])))
        tmp_p_anc  = [tmp_p_tok1[x] for x in tmp_sample]
        
        tmp_anc_tuple = prepare_input(
            tmp_p_anc, word_2_idx, 
            seq_length, vocab_size, prob_mask, 
            CLS_token, TRU_token, EOS_token, MSK_token)
        
        tmp_pos_tuple = prepare_input(
            tmp_p_tok1, word_2_idx, 
            seq_length, vocab_size, prob_mask, 
            CLS_token, TRU_token, EOS_token, MSK_token)
        
        tmp_neg_tuple = prepare_input(
            tmp_p_tok2, word_2_idx, 
            seq_length, vocab_size, prob_mask, 
            CLS_token, TRU_token, EOS_token, MSK_token)
        
        n_anc = len(tmp_anc_tuple[0])
        n_pos = len(tmp_pos_tuple[0])
        n_neg = len(tmp_neg_tuple[0])
        del tmp_sample, n_sample
        
        tmp_in_data[n_index, :n_pos] = tmp_pos_tuple[1]
        tmp_out_anc[n_index, :n_anc] = tmp_anc_tuple[0]
        tmp_out_data[n_index, :n_pos]  = tmp_pos_tuple[0]
        tmp_mask_data[n_index, :n_pos] = tmp_pos_tuple[2]
    
    tmp_loss = sub_batch_train_step(
        bert_model, bert_optim, sub_batch, 
        tmp_mask_data, tmp_in_data, tmp_out_anc, 
        tmp_out_data, alpha=100.0, learning_rate=learning_rate)
    
    n_iter += 1
    ckpt.step.assign_add(1)
    
    tot_loss += tmp_loss[0].numpy()
    mlm_loss += tmp_loss[1].numpy()
    if n_iter % display_step == 0:
        end_tm = time.time()
        
        avg_tot_loss = tot_loss / display_step
        avg_mlm_loss = mlm_loss / display_step

        tot_loss = 0.0
        mlm_loss = 0.0
        elapsed_tm = (end_tm - start_tm) / 60
        
        tmp_test_anc[:, :]  = PAD_token
        tmp_test_data[:, :] = PAD_token
        
        sample_add = np.random.randint(1, num_data)
        sample_pos = np.random.choice(num_data)
        sample_neg = (sample_pos + sample_add) % num_data
        
        tmp_in_pos = jokes_data[sample_pos].lower()
        tmp_in_neg = jokes_data[sample_neg].lower()
        
        tmp_p_pos = tmp_in_pos.split(" ")
        tmp_p_neg = tmp_in_neg.split(" ")

        tmp_p_tuple = prepare_input(
            tmp_p_pos, word_2_idx, 
            seq_length, vocab_size, prob_mask, 
            CLS_token, TRU_token, EOS_token, MSK_token)
        
        tmp_n_tuple = prepare_input(
            tmp_p_neg, word_2_idx, 
            seq_length, vocab_size, prob_mask, 
            CLS_token, TRU_token, EOS_token, MSK_token)
        
        if len(tmp_p_pos) <= 2:
            n_sample = 1
        else:
            n_sample = np.random.randint(1, n_max_samples)
        n_sample = min(n_sample, len(tmp_p_pos))

        tmp_sample = list(sorted(
            list(np.random.permutation(
                len(tmp_p_pos))[:n_sample])))
        
        tmp_in_anc = tmp_in_pos.split(" ")
        tmp_in_anc = " ".join(
            [tmp_in_anc[x] for x in tmp_sample])
        tmp_p_anc  = [x for x in tmp_in_anc.split(" ")]

        tmp_a_tuple = prepare_input(
            tmp_p_anc, word_2_idx, 
            seq_length, vocab_size, prob_mask, 
            CLS_token, TRU_token, EOS_token, MSK_token)
        
        tmp_i_anc  = tmp_a_tuple[0]
        tmp_i_pos  = tmp_p_tuple[0]
        tmp_i_neg  = tmp_n_tuple[0]

        n_anc_toks = len(tmp_i_anc)
        n_pos_toks = len(tmp_i_pos)
        n_neg_toks = len(tmp_i_neg)
        del tmp_sample, n_sample
        
        tmp_test_anc[0, :n_anc_toks]  = tmp_i_anc
        tmp_test_data[0, :n_pos_toks] = tmp_i_pos
        tmp_test_data[1, :n_neg_toks] = tmp_i_neg
        
        tmp_anc_emb  = np.mean(bert_model(
            tmp_test_anc, training=False)[1], axis=1)
        tmp_data_emb = np.mean(bert_model(
            tmp_test_data, training=False)[1], axis=1)
        del sample_pos, sample_neg, sample_add
        del n_pos_toks, n_neg_toks, n_anc_toks
        
        pos_dist = tf.reduce_mean(tf.square(
            tmp_anc_emb - tmp_data_emb[0]), axis=1)[0]
        neg_dist = tf.reduce_mean(tf.square(
            tmp_anc_emb - tmp_data_emb[1]), axis=1)[0]
        
        print("Iteration", str(n_iter)+".")
        print("Elapsed Time:", str(elapsed_tm), "mins.")
        print("Gradient Clip:", str(gradient_clip) + ".")
        print("Learning Rate:", str(learning_rate) + ".")
        print("Average MLM Loss:  ", str(avg_mlm_loss) + ".")
        print("Average Total Loss:", str(avg_tot_loss) + ".")
        print("")
        
        print("Anchor:", tmp_in_anc)
        print("Positive", "(Distance:", 
              str(round(pos_dist.numpy(), 3)) + "):", tmp_in_pos)
        print("Negative", "(Distance:", 
              str(round(neg_dist.numpy(), 3)) + "):", tmp_in_neg)
        
        train_loss_list.append(
            (n_iter, avg_tot_loss, avg_mlm_loss))
        start_tm = time.time()
        print("-" * 50)
    
    # Save the model. #
    if n_iter % save_step == 0:
        # Save the model. #
        save_path = manager.save()
        print("Saved model to {}".format(save_path))
        
        tmp_df_columns = ["n_iter", "tot_loss", "mlm_loss"]
        tmp_df_losses  = pd.DataFrame(
            train_loss_list, columns=tmp_df_columns)
        tmp_df_losses.to_csv(train_loss_file, index=False)
        del tmp_df_losses
    
    # Cool the GPU. #
    if n_iter % cooling_step == 0:
        print("Cooling GPU for 2 minutes.")
        time.sleep(120)
        print("Resume Training.")
        print("-" * 50)
