# install methods
# !pip install transformers

import torch
from transformers import BertTokenizer, BertModel, BertForSequenceClassification
import pandas as pd
import numpy as np
import time
import pprint
import sys
import json
import pathlib
import glob
import os

currentpath = pathlib.Path(__file__).parent.absolute()

path = str(currentpath) + "\\jsonfiles"

try:
    os.mkdir(path)
except OSError:
    print("Creation of the directory %s failed" % path)
else:
    print("Successfully created the directory %s " % path)

# Load pre-trained model tokenizer (vocabulary)
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
print("Imported libraries.")

# load data for creating word embeddings

i = 0
reviews = []
file = open(str(currentpath) + '\\restData20k.txt', 'r', encoding="utf-8")

review = file.read()
review = review.replace("\n", ' ')

# Reviews are split on ,|,
wrong_review = review.split(",|,")
for w in wrong_review:
    i += 1
    reviews.append(w)
# reviews = reviews[0:1000]
print("Number of reviews : ", len(reviews))

# Code for removing negation words in same sentence
i = 0
review = 0
sentC = 0
reviewsNew = []
neg = " not "
neg2 = " nothing "
neg3 = "never "
neg4 = " didn\'t"
neg5 = " wouldn\'t"
neg6 = " don\'t"
neg7 = " can\'t"
neg8 = " doesn\'t"
neg9 = " coudn't"
case = 0
for z in reviews:
    sent2New = []
    sentNew = ""
    sent = z.split('.')
    case = 0
    for j in sent:
        sent2 = j.split('!')
        for l in sent2:
            sentC += 1
            if (neg in l) or (neg2 in l) or (neg3 in l) or (neg4 in l) or (neg5 in l) or (neg6 in l) or (neg7 in l) or (
                    neg8 in l) or (neg9 in l):
                i += 1
                sent2.remove(l)
                if case == 0:
                    review += 1
                    case = 1
        sent2New.append("!".join(sent2))
    # print(sent2New)
    sentNew = (".".join(sent2New))
    reviewsNew.append(str(sentNew))

print("Number of sentences with negation word:", i)
print("Number of reviews with these negation word:", review)
print("Number of sentences in text: ", sentC)
print("Number of sentences in text after removing sentences: ", sentC - i)

# Load pre-trained model (weights)
model = BertModel.from_pretrained('bert-base-uncased', output_hidden_states=True, )  # pretrained
# model = BertModel.from_pretrained('POST2',output_hidden_states = True,)                          #posttrained
# model = BertModel.from_pretrained('finetune',output_hidden_states = True,)                       #finetuned

# Put the model in "evaluation" mode, meaning feed-forward operation.
model.eval()
print("BERT model is downloaded.")

currentfile = 0
# Create pretrained vectors
start = time.time()
intermediate_time = start
review_counter = 1
words = {}
j = 1
name_file = "output.txt"
# loops over the reviews
for rev in reviews:
    """
    if review_counter > 400:
        break;
    """
    if review_counter % 100 == 0:
        start_time = intermediate_time
        intermediate_time = time.time()
        ETA = (intermediate_time - start_time) * (1000 - review_counter / 100)
        print("Estimated time: " + str(ETA))
        print("Embedding review: " + str(review_counter))

        # print memory
        print("size of words:" + str(sys.getsizeof(words)))

        # save words to file
        with open(str(currentpath) + "/jsonfiles/f" + str(currentfile) + ".json", "w") as f:
            json.dump(words, f)
        currentfile = currentfile + 1

        del words
        words = {}

    result = []
    tokenized_text = tokenizer.tokenize(rev)
    # change to 512 or shorter
    if len(tokenized_text) >= 512:
        del tokenized_text[512:len(tokenized_text)]
    indexed_tokens = tokenizer.convert_tokens_to_ids(tokenized_text)
    segments_ids = [1] * len(tokenized_text)
    tokens_tensor = torch.tensor([indexed_tokens])
    segments_tensors = torch.tensor([segments_ids])
    with torch.no_grad():
        outputs = model(tokens_tensor, segments_tensors)
        hidden_states = outputs.hidden_states
    token_embeddings = torch.stack(hidden_states, dim=0)
    token_embeddings = torch.squeeze(token_embeddings, dim=1)
    token_embeddings = token_embeddings.permute(1, 0, 2)
    token_vecs_sum = []
    for token in token_embeddings:
        sum_vec = torch.sum(token[-4:], dim=0)
        token_vecs_sum.append(sum_vec)
    token_vecs = hidden_states[-2][0]
    sentence_embedding = torch.mean(token_vecs, dim=0)
    # get all results for one review and display
    # replace with whole vector!
    i = 0
    while i < len(tokenized_text):
        veccie = [round(vec, 4) for vec in token_vecs_sum[i].tolist()]
        result.append([tokenized_text[i], indexed_tokens[i], segments_ids[i], veccie, review_counter])
        i += 1

    for word in result:
        string1 = str(word[0])
        words[j] = {'word': string1,
                    'vector': word[3]
            , 'sentence id': word[4]
                    }
        j = j + 1

    review_counter += 1

    # FREE MEMORY
    del result
    del token_vecs_sum


def mangle(s):
    return s.strip()[1:-1]


print(str(currentpath) + "\\jsonfiles" + "\\*.json")
read_files = glob.glob(str(currentpath) + "\\jsonfiles" + "\\*.json")
print(read_files)


def cat_json(output_filename, input_filenames):
    print("here")
    with open(output_filename, "w") as outfile:
        first = True
        for infile_name in input_filenames:
            with open(infile_name) as infile:
                if first:
                    outfile.write('[')
                    first = False
                else:
                    outfile.write(',')
                outfile.write(mangle(infile.read()))
        outfile.write(']')


cat_json(str(currentpath) + "\\out.json", read_files);