# NLP-BERT-Retrieval

This repository contains an implementation of using [BERT](https://arxiv.org/abs/1810.04805) to train a document retrieval algorithm in an self-supervised manner. Inspired by this [paper](https://arxiv.org/abs/2007.15651), the original BERT model is modified to implement a triplet loss function by randomly sampling the word tokens in positive example to form the anchor in the triplet loss. The negative example is any other document in the corpus which is different from the positive example. 

(Model architecture to be added)

## Training the Model
To train the model, first download the data of interest - (i) [Reddit Jokes](https://github.com/taivop/joke-dataset) or (ii) [Yelp Polarity Review](https://course.fast.ai/datasets), followed by processing the data
```
python process_reddit_jokes.py
```
for Reddit Jokes or
```
python process_yelp_tokens.py
```
for the Yelp dataset. Then run
```
python train_reddit_jokes_tf_ver2_bert_triplet_loss.py
```
to train the model on the Reddit Jokes or
```
python train_yelp_reviews_tf_ver2_bert_triplet_loss.py
```
to train the model on the YELP dataset. Finally, run
```
python infer_reddit_jokes_tf_ver2_bert_triplet_loss.py
```
or 
```
python infer_yelp_reviews_tf_ver2_bert_triplet_loss.py
```
to perform inference on their respective datasets. The BERT model applied is a tiny model with 3 layers, 4 heads, 256 hidden units and 1024 feedforward units.

## Inference
Some results of the document retrieval of the Reddit dataset is presented below:
```
Enter phrase: what did say to
Top 1 match (distance = 4.833143):
bad_joke what did one organ say ...? what did one organ say to another organ ? gland to meet you .
==================================================
Top 2 match (distance = 4.962682):
bad_joke what did the policeman say to the jumper ? pullover
==================================================
Top 3 match (distance = 4.9796743):
ok_joke what did the kidney say to the other kidney when it failed ? urine trouble now .
==================================================
Top 4 match (distance = 5.002075):
bad_joke what did the rastafarian say to the hypnotist jamaican me sleepy
==================================================
Top 5 match (distance = 5.0185523):
bad_joke what did my wife say to me ? i want a divorce
--------------------------------------------------
```
Similarly, some results of the document retrieval for the YELP dataset is presented below:
```
Enter phrase: cheap price nice food
Top 1 match (distance = 3.4652174):
pretty good food , crab legs are succulent and tasty . nice service and fresh food . not bad for the price . sushi is awesome quality .
==================================================
Top 2 match (distance = 3.9526186):
cheap price for great food
==================================================
Top 3 match (distance = 4.4345365):
good food , decent price .
==================================================
Top 4 match (distance = 4.652419):
yum . good breakfast and cheap beer . staff was a little slow but pretty good food for cheap vegas food .
==================================================
Top 5 match (distance = 4.740176):
average food for super expensive prices .
--------------------------------------------------
```
## Masked Language Modelling
An extension has been added to include Masked Language Modelling (MLM) to try to enhance the retrieval results. The training of the model can be done via
```
python train_reddit_jokes_tf_ver2_bert_triplet_mlm.py
```
and inference can be done via
```
python infer_reddit_jokes_tf_ver2_bert_triplet_mlm.py
```
for Reddit Jokes dataset. This implementation uses the averaged BERT output embeddings to regularize the `CLS` token's embedding.

The code has been improved to modify both the language modelling loss as well as the distance loss. In particular, the triplet loss has been modified such that the distance from positive examples are minimised while the distance from negative examples are maximised separately. The negative examples' distance is subjected to a maximum distance `gamma` in order to prevent the total loss from going to negative infinity. In addition, the loss also minimises the non-masked tokens to try to let the model learn better embeddings.

This modified loss is trained using [RoBERT](https://arxiv.org/abs/1907.11692) architecture on the [Fraser jokes dataset](https://huggingface.co/datasets/Fraser/short-jokes). The `parquet` file of this dataset can be found [here](https://huggingface.co/datasets/Fraser/short-jokes/tree/refs%2Fconvert%2Fparquet/default](https://huggingface.co/datasets/Fraser/short-jokes/blob/refs%2Fconvert%2Fparquet/default/short-jokes-train.parquet)https://huggingface.co/datasets/Fraser/short-jokes/blob/refs%2Fconvert%2Fparquet/default/short-jokes-train.parquet). Before training the model, the data has to be first processed to extract the vocabulary. This can be done by running
```
python process_fraser_jokes_word.py
```
followed by
```
python train_fraser_jokes_tf_ver2_robert_mlm_retrieval.py
```
to train the model. To perform inference, run
```
python infer_fraser_jokes_tf_ver2_robert_retrieval.py
```
and enter a query to start retrieving relevant jokes from the document set.

After training, the word embeddings can be checked by running `check_vocab_fraser_jokes_tf_ver2_robert_mlm_retrieval.py` to verify that the model is satisfactorily trained:
```
==================================================
Enter word: donkey
Top match (Distance: 0.557)  --  meerkat
2nd match (Distance: 0.557)  --  crip
3rd match (Distance: 0.559)  --  totem
==================================================
Enter word: bar
Top match (Distance: 0.568)  --  seal
2nd match (Distance: 0.57)   --  future
3rd match (Distance: 0.572)  --  bartender
==================================================
Enter word: man
Top match (Distance: 0.539)  --  guy
2nd match (Distance: 0.56)   --  pig
3rd match (Distance: 0.562)  --  woman
==================================================
```
Examples of retrieved documents (jokes):
```
Enter phrase: man bar
Euclidean Distance:
Top 1 match (distance = 0.018480647):
a man walks into a bar Ouch! he said.
==================================================
Euclidean Distance:
Top 2 match (distance = 0.020611817):
A blind man walks into a bar ow.
==================================================
Euclidean Distance:
Top 3 match (distance = 0.020611817):
A blind man walks into a bar Ouch!
==================================================
Euclidean Distance:
Top 4 match (distance = 0.022706196):
A man walked into a bar Ouch!
==================================================
Euclidean Distance:
Top 5 match (distance = 0.022706196):
A man walked into a bar Ouch
--------------------------------------------------
Enter phrase: chicken road
Euclidean Distance:
Top 1 match (distance = 0.045933265):
Why did the chicken run out into traffic? To get to the other side.
==================================================
Euclidean Distance:
Top 2 match (distance = 0.048717245):
Why'd the chicken cross the road? To show a deer how it's done.
==================================================
Euclidean Distance:
Top 3 match (distance = 0.05173428):
Why did leeroy Jenkins cross the road To get the chicken
==================================================
Euclidean Distance:
Top 4 match (distance = 0.05557366):
Why'd the chicken cross the road Wtf Idk
==================================================
Euclidean Distance:
Top 5 match (distance = 0.055962004):
Why did the chicken cross the road? To show the deer how it's done.
--------------------------------------------------
```

