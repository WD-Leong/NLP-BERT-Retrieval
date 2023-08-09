
import time
import numpy as np
import pandas as pd
import pickle as pkl

import tensorflow as tf
import tensorflow_addons as tfa
import tf_ver2_gpt_keras_v3 as tf_gpt

# Define the weight update step for multiple sub-batches. #
def sub_batch_train_step(
    model, optimizer, sub_batch_sz, 
    x_anchor, x_input, x_output, 
    alpha=1.0, beta=1.0, gamma=100.0, 
    learning_rate=1.0e-3, gradient_clip=1.00):
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
    llm_losses = 0.0
    for n_sub in range(sub_batch):
        id_st = n_sub*sub_batch_sz
        if n_sub != (sub_batch-1):
            id_en = (n_sub+1)*sub_batch_sz
        else:
            id_en = batch_size
        
        tmp_in_anc  = x_anchor[id_st:id_en, :]
        tmp_in_data = x_input[id_st:id_en, :]
        tmp_out_data = x_output[id_st:id_en, :]
        
        num_sub_batch = id_en - id_st
        with tf.GradientTape() as grad_tape:
            anc_outputs  = model(tmp_in_anc, training=True)
            data_outputs = model(tmp_in_data, training=True)
            
            # Extract the average embeddings. #
            anc_avg  = tf.expand_dims(
                tf.reduce_mean(anc_outputs[1], axis=1), axis=0)
            data_avg = tf.expand_dims(
                tf.reduce_mean(data_outputs[1], axis=1), axis=1)

            # Extract the vocabulary logits to train the word embeddings. #
            data_logits = data_outputs[0]

            # Euclidean Distance Loss. #
            tmp_dist = tf.reduce_sum(
                tf.square(anc_avg - data_avg), axis=2)
            
            # Language Modeling loss. #
            xent_loss = tf.reduce_sum(tf.reduce_mean(
                tf.nn.sparse_softmax_cross_entropy_with_logits(
                    labels=tmp_out_data, logits=data_logits), axis=1))
            
            # Euclidean Distance Loss. #
            diag_mask = tf.linalg.band_part(tf.ones(
                [num_sub_batch, num_sub_batch]), 0, 0)
            diag_mask = tf.expand_dims(diag_mask, axis=0)
            
            non_diag_mask = 1.0 - diag_mask
            num_non_diag  = tf.reduce_sum(non_diag_mask)
            
            # Set up the Triplet Loss margins. #
            pos_distance = diag_mask * tmp_dist
            neg_distance = tf.maximum(
                0.0, gamma - (non_diag_mask * tmp_dist))
            
            pos_distance = tf.reduce_sum(
                tf.divide(pos_distance, num_sub_batch))
            neg_distance = tf.reduce_sum(
                tf.divide(neg_distance, num_non_diag))
            
            # Modification of the Triplet Loss. #
            tot_dist_loss = pos_distance + neg_distance
            
            # Total loss in this sub-batch. #
            tmp_losses = tf.add(
                alpha * xent_loss, beta * tot_dist_loss)
        
        # Accumulate the gradients. #
        llm_losses += xent_loss
        tot_losses += tmp_losses
        
        tmp_gradients = grad_tape.gradient(
            tmp_losses, model_params)
        acc_gradients = [tf.add(
            acc_grad, grad) for acc_grad, grad \
                in zip(acc_gradients, tmp_gradients)]
    
    # Update using the optimizer. #
    average_loss  = tot_losses / batch_size
    avg_llm_loss  = llm_losses / batch_size
    acc_gradients = [tf.math.divide_no_nan(
        acc_grad, batch_size) for acc_grad in acc_gradients]
    
    clip_tuple = tf.clip_by_global_norm(
        acc_gradients, gradient_clip)
    optimizer.apply_gradients(
        zip(clip_tuple[0], model_params))
    return average_loss, avg_llm_loss

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
n_max_sample  = 5

prob_keep = 0.90
hidden_size = 256
ffwd_size   = 4*hidden_size
warmup_flag  = True
cooling_step = 100

model_ckpt_dir  = "../TF_Models/gpt_fraser_retrieval"
train_loss_file = "train_loss_gpt_fraser_retrieval.csv"

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
SOS_token = word_2_idx["[CLS]"]
EOS_token = word_2_idx["[EOS]"]
PAD_token = word_2_idx["[PAD]"]
UNK_token = word_2_idx["[UNK]"]
print("Total of", num_data, "rows loaded.")

# Set the number of threads to use. #
tf.config.threading.set_intra_op_parallelism_threads(1)
tf.config.threading.set_inter_op_parallelism_threads(1)

# Build the Transformer. #
print("Building the GPT Model.")
start_time = time.time()

gpt_model = tf_gpt.GPTDecoder(
    num_layers, num_heads, hidden_size, 
    ffwd_size, vocab_size, seq_length+1, 
    rate1=0.0, rate2=1.0-prob_keep)
gpt_optim = tfa.optimizers.AdamW(
    weight_decay=1.0e-3)

elapsed_time = (time.time()-start_time) / 60
print("GPT Model Built", "(" + str(elapsed_time), "mins).")

# Create the model checkpoint. #
ckpt = tf.train.Checkpoint(
    step=tf.Variable(0), 
    gpt_model=gpt_model, 
    gpt_optim=gpt_optim)

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

# Train the GPT model. #
tmp_anc_data = np.zeros(
    [batch_size, seq_length+1], dtype=np.int32)
tmp_seq_data = np.zeros(
    [batch_size, seq_length+2], dtype=np.int32)

tmp_test_anc  = np.zeros(
    [1, seq_length+1], dtype=np.int32)
tmp_test_data = np.zeros(
    [2, seq_length+1], dtype=np.int32)

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
print("Training the GPT Network", 
      "(" + str(n_iter) + " iterations).")
print(str(num_data), "training samples.")
print("-" * 50)

# Update the neural network's weights. #
tot_loss = 0.0
llm_loss = 0.0
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
    tmp_anc_data[:, :] = PAD_token
    tmp_seq_data[:, :] = PAD_token
    
    for n_index in range(batch_size):
        tmp_index = batch_sample[n_index]
        tmp_p_tok = jokes_data[tmp_index].split(" ")
        
        tmp_p_idx = [word_2_idx.get(
            x, UNK_token) for x in tmp_p_tok if x != ""]
        
        # Randomly sample the positive input #
        # to be used as the anchor.          #
        n_input = len(tmp_p_idx)
        if n_input <= 2:
            n_sample = 1
        else:
            n_sample = np.random.randint(1, n_max_sample)
        n_sample = min(n_sample, n_input)

        # Anchor does not need EOS token. #
        tmp_sample = list(sorted(list(
            np.random.permutation(n_input)[:n_sample])))
        tmp_p_anc = [SOS_token]
        tmp_p_anc += [tmp_p_idx[x] for x in tmp_sample]
        tmp_p_idx = [SOS_token] + tmp_p_idx + [EOS_token]

        l_anc  = len(tmp_p_anc)
        l_data = len(tmp_p_idx)
        del tmp_sample, n_sample
        
        tmp_anc_data[n_index, :l_anc]  = tmp_p_anc
        tmp_seq_data[n_index, :l_data] = tmp_p_idx
    
    tmp_seq_input  = tmp_seq_data[:, :-1]
    tmp_seq_output = tmp_seq_data[:, 1:]

    tmp_loss = sub_batch_train_step(
        gpt_model, gpt_optim, sub_batch, 
        tmp_anc_data, tmp_seq_input, tmp_seq_output, beta=1.0, 
        learning_rate=learning_rate, gradient_clip=gradient_clip)
    
    n_iter += 1
    ckpt.step.assign_add(1)
    
    tot_loss += tmp_loss[0].numpy()
    llm_loss += tmp_loss[1].numpy()
    if n_iter % display_step == 0:
        end_tm = time.time()
        
        avg_tot_loss = tot_loss / display_step
        avg_llm_loss = llm_loss / display_step

        tot_loss = 0.0
        llm_loss = 0.0
        run_time = (end_tm - start_tm) / 60
        
        tmp_test_anc[:, :]  = PAD_token
        tmp_test_data[:, :] = PAD_token
        
        sample_add = np.random.randint(1, num_data)
        sample_pos = np.random.choice(num_data)
        sample_neg = (sample_pos + sample_add) % num_data
        
        test_pos = jokes_data[sample_pos]
        test_neg = jokes_data[sample_neg]
        
        tmp_p_pos = [
            x for x in test_pos.split(" ") if x != ""]
        if len(tmp_p_pos) <= 2:
            n_sample = 1
        else:
            n_sample = np.random.randint(1, n_max_sample)
        n_sample = min(n_sample, n_input)

        tmp_sample = list(sorted(
            list(np.random.permutation(
                len(tmp_p_pos))[:n_sample])))
        
        tmp_in_anc = [
            x for x in test_pos.split(" ") if x != ""]
        tmp_in_anc = " ".join(
            [tmp_in_anc[x] for x in tmp_sample])
        
        # Note that the inputs do not have EOS tokens. #
        tmp_i_anc = [SOS_token]
        tmp_i_anc += [word_2_idx.get(
            x, UNK_token) for x in tmp_in_anc.split(" ")]
        
        tmp_i_pos = [SOS_token]
        tmp_i_pos += [word_2_idx.get(
            x, UNK_token) for x in test_pos.split(" ") if x != ""]
        
        tmp_i_neg = [SOS_token]
        tmp_i_neg = [word_2_idx.get(
            x, UNK_token) for x in test_neg.split(" ") if x != ""]

        n_anc_toks = len(tmp_i_anc)
        n_pos_toks = len(tmp_i_pos)
        n_neg_toks = len(tmp_i_neg)
        del tmp_sample, n_sample
        
        tmp_test_anc[0, :n_anc_toks]  = tmp_i_anc
        tmp_test_data[0, :n_pos_toks] = tmp_i_pos
        tmp_test_data[1, :n_neg_toks] = tmp_i_neg
        
        tmp_anc_emb  = tf.reduce_mean(gpt_model(
            tmp_test_anc, training=False)[1], axis=1)
        tmp_data_emb = tf.reduce_mean(gpt_model(
            tmp_test_data, training=False)[1], axis=1)
        del sample_pos, sample_neg, sample_add
        del n_pos_toks, n_neg_toks, n_anc_toks
        
        anc_dist = tf.reduce_sum(tf.square(
            tmp_anc_emb - tmp_data_emb), axis=1)
        pos_dist = anc_dist[0]
        neg_dist = anc_dist[1]
        
        print("Iteration", str(n_iter)+".")
        print("Elapsed Time:", str(run_time), "mins.")
        print("Gradient Clip:", str(gradient_clip) + ".")
        print("Learning Rate:", str(learning_rate) + ".")
        print("Average Tot Loss:", str(avg_tot_loss) + ".")
        print("Average LLM Loss:", str(avg_llm_loss) + ".")
        print("")
        
        print("Anchor:", tmp_in_anc)
        print("Positive", "(Distance:", 
              str(pos_dist.numpy()) + "):", test_pos)
        print("Negative", "(Distance:", 
              str(neg_dist.numpy()) + "):", test_neg)
        
        train_loss_list.append((n_iter, avg_tot_loss, avg_llm_loss))
        start_tm = time.time()
        print("-" * 50)
    
    # Save the model. #
    if n_iter % save_step == 0:
        # Save the model. #
        save_path = manager.save()
        print("Saved model to {}".format(save_path))
        
        tmp_df_columns = ["n_iter", "tot_loss", "llm_loss"]
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
