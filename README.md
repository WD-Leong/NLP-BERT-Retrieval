# NLP-BERT-Retrieval

This repository contains an implementation of using [BERT](https://arxiv.org/abs/1810.04805) to train a document retrieval algorithm. Inspired by this [paper](https://arxiv.org/abs/2007.15651), the original BERT model is modified to implement a triplet loss function by randomly sampling the word tokens in positive example to form the anchor in the triplet loss. The negative example is any other document in the corpus which is different from the positive example. 

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

