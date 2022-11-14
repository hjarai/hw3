"""
converts CoNLL data (train and dev) into features of the parser configuration paired
with parser decisions. This should be human-readable, i.e. a text file of words/labels. The format
should be described in the README
"""
#%%
import itertools
from data_utils import Token, Sentence 
from tqdm import tqdm
import json
file = "train.orig.conll"

with open(file, 'r', encoding='utf-8') as file:
    data = file.read().splitlines()
    data_split = [line.strip().split() for line in data]
    by_sentences = [list(y) for x, y in itertools.groupby(data_split, lambda z: z == []) if not x]

vocab = {}
word_count = 0
labels = set()
def build_vocab(word):
    global word_count
    if word not in vocab:
        vocab[word] = word_count
        word_count+=1

[build_vocab(word) for word in ['<root>', '<null>', '<unk>', 'None', 'POSNone', 'DEPNone']]

sentence_features = []
cooounter = 0
for sentence in tqdm(by_sentences):
    tokens_list = []
    for word in sentence:
        tokens_list.append(Token(token_id=word[0], word=word[2], pos=word[4], head=word[6], dep=word[7]))
        [build_vocab(word) for word in [word[2], word[4], word[7]]]
    this_sentence = Sentence(tokens_list)
    if not this_sentence.is_projective():
        continue
    counter = 0
    cooounter+=1
    while not (this_sentence.buffer == [] and len(this_sentence.stack) == 1):
        features = this_sentence.get_features()
        curr_trans = this_sentence.get_trans()
        labels.add(curr_trans)
        sentence_features.append(features+[curr_trans])
        counter +=1
        this_sentence.update_state(curr_trans)  


with open('train.converted', 'w', encoding='utf-8') as out_file:
    for features in sentence_features:
        out_file.write(' '.join(features)+'\n')

with open('vocab.json', 'w') as out_file:
    vocab['LABEL_LIST'] = list(labels)
    json.dump(vocab, out_file)

print(len(vocab))
print(len(labels))
print(vocab['LABEL_LIST'])

