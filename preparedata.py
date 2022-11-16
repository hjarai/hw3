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
for input in ['train', 'dev']:
    with open('{}.orig.conll'.format(input), 'r', encoding='utf-8') as file:
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
    cooounter, projective_count = 0, 0
    for sentence in tqdm(by_sentences):
        tokens_list = []
        for word in sentence:
            tokens_list.append(Token(token_id=word[0], word=word[2], pos=word[4], head=word[6], dep=word[7]))
            # [build_vocab(word) for word in [word[2], word[4], word[7]]]
            #only need pos and dep if we use glove embeddings
            [build_vocab(word) for word in [word[4], word[7]]]

        this_sentence = Sentence(tokens_list)
        if not this_sentence.is_projective():
            projective_count += 1
            continue
        cooounter+=1
        while not (this_sentence.buffer == [] and len(this_sentence.stack) == 1):
            features = this_sentence.get_features()
            curr_trans = this_sentence.get_trans()
            labels.add(curr_trans)
            sentence_features.append(features+[curr_trans])
            this_sentence.update_state(this_sentence.format_prediction(curr_trans)[0])  


    print("projective: {} out of total: {} for {}".format(projective_count, cooounter, projective_count/cooounter))
    with open('{}.converted'.format(input), 'w', encoding='utf-8') as out_file:
        for features in sentence_features:
            out_file.write(' '.join(features)+'\n')
    if input == 'train':
        with open('my_vocab.json', 'w') as out_file:
            vocab['LABEL_LIST'] = list(labels)
            json.dump(vocab, out_file)

