"""
fills CONLL data

"""
#%%
import itertools
from data_utils import Token, Sentence 
from tqdm import tqdm
import json
from mymodel import myModel
import torch
file = "dev.orig.conll"

# load file
with open(file, 'r', encoding='utf-8') as file:
    data = file.read().splitlines()
    data_split = [line.strip().split() for line in data]
    by_sentences = [list(y) for x, y in itertools.groupby(data_split, lambda z: z == []) if not x]

# load model
saved = torch.load("train.model")
model_params = saved['model_params']
state_dict = saved['state_dict']
model = myModel(model_params['input_dim'], model_params['hidden_dim'], model_params['output_dim'], len(model_params['vocab_dict']), model_params['embedding_dim'])
model.load_state_dict(state_dict)


sentence_features = []
for sentence in tqdm(by_sentences):
    tokens_list = []
    for word in sentence:
        tokens_list.append(Token(token_id=word[0], word=word[2], pos=word[4], head=word[6], dep=word[7]))
    this_sentence = Sentence(tokens_list)
    counter = 0
    while not (this_sentence.buffer == [] and len(this_sentence.stack) == 1):
        features = this_sentence.get_features()
        # predict arc here!
        
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

