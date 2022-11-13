"""
converts CoNLL data (train and dev) into features of the parser configuration paired
with parser decisions. This should be human-readable, i.e. a text file of words/labels. The format
should be described in the README
"""
#%%
import itertools
from data_utils import Token, Sentence 
from tqdm import tqdm

file = "train.orig.conll"

with open(file, 'r', encoding='utf-8') as file:
    data = file.read().splitlines()
    data_split = [line.strip().split() for line in data]
    by_sentences = [list(y) for x, y in itertools.groupby(data_split, lambda z: z == []) if not x]


sentences_list = []
vocab = {}
word_count = 0
for sentence in tqdm(by_sentences):
    tokens_list = []
    for word in sentence:
        tokens_list.append(Token(token_id=word[0], word=word[2], pos=word[4], head=word[6], dep=word[7]))
        #build vocab
        if word[2] not in vocab:
            vocab[word[2]] = word_count
            word_count+=1
    this_sentence = Sentence(tokens_list)
    continue
    if not sentence.is_projective():
        continue
        # print(sentence)
    # sentences_list.append(sentence)

# termination buffer is empty and stack is just root
while not (this_sentence.buffer == [] and len(this_sentence.stack) == 1):
    curr_trans = this_sentence.get_trans()
    print(curr_trans)
    this_sentence.update_state(curr_trans)
    print([token.word for token in this_sentence.stack], [token.word for token in this_sentence.buffer])

# %%

# class parse_sent():
# class parse_sent():
#     def __init__(self, sentence): 
#         self.tokens = sentence
#         self.buffer = sentence
#         self.stack = []
#         pass
        
#     def one_step(self):

#         pass
#     def get_features(self):
#         #18 things?
#         # the first three words on the stack and the buffer (and their POS tags) (12 features)
# # • the words, POS tags, and arc labels of the first and second leftmost and rightmost children
# # of the first two words on the stack, (24 features)
# # • the words, POS tags, and arc labels of leftmost child of the leftmost child and rightmost child
# # of rightmost child of the first two words of the stack (12 features)
#         pass
#     def get_transitions(self):
#         #left/right/shift
#         pass
#     def write_out(self, out_path):
#         pass

# class token():
#     def __init__(word):



