
import time
import pandas as pd
import pickle as pkl
from collections import Counter
from nltk.tokenize import wordpunct_tokenize

print("Loading the data.")
start_tm = time.time()

tmp_file = "../Data/jokes/short-jokes-train.parquet"
tmp_data = pd.read_parquet(tmp_file)
max_len  = 25
print("Total of", str(len(tmp_data)), "jokes loaded.")

# Extract the data. #
tmp_jokes = []
for n_row in range(len(tmp_data)):
    tmp_joke = tmp_data.iloc[n_row]["text"]
    tmp_joke = tmp_joke.replace("\"", "").replace("\n", " ")
    tmp_jokes.append(tmp_joke)

# Process the data. #
jokes_filtered = []

w_counter = Counter()
for tmp_joke in tmp_jokes:
    tmp_tokens = [
        x for x in wordpunct_tokenize(
            tmp_joke.lower()) if x != ""]
    
    if len(tmp_tokens) <= max_len:
        w_counter.update(tmp_tokens)
        jokes_filtered.append(tmp_joke)
    del tmp_tokens

print("Total of", len(jokes_filtered), "jokes filtered.")
del tmp_jokes

# Only use words which occur more than 3 times. #
min_count  = 3
word_vocab = ["[CLS]", "[EOS]", "[UNK]", "[PAD]", "[TRU]","[MSK]"]
word_vocab += list(sorted([
    x for x, y in w_counter.most_common() if y > min_count]))

word_2_idx = dict([
    (word_vocab[x], x) for x in range(len(word_vocab))])
idx_2_word = dict([
    (x, word_vocab[x]) for x in range(len(word_vocab))])
print("Vocabulary Size:", len(word_vocab), "words.")

elapsed_tm = (time.time() - start_tm) / 60
print("Total of", str(len(word_vocab)), "words.")
print("Elapsed Time:", str(elapsed_tm), "mins.")

# Save the data. #
print("Saving the file.")

tmp_pkl_file = "../Data/jokes/short_jokes_words.pkl"
with open(tmp_pkl_file, "wb") as tmp_file_save:
    pkl.dump(jokes_filtered, tmp_file_save)

    pkl.dump(word_vocab, tmp_file_save)
    pkl.dump(word_2_idx, tmp_file_save)
    pkl.dump(idx_2_word, tmp_file_save)

