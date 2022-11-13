import numpy as np


P_PREFIX = '<p>'
L_PREFIX = '<l>'
ROOT = '<root>'
NULL = '<null>'
UNK = '<unk>'


class Token:

    def __init__(self, token_id, word, pos, head, dep):
        self.token_id = int(token_id)
        self.word = word
        self.pos = pos
        self.head = int(head)
        self.dep = dep
        self.predicted_head = -1
        self.predicted_dep = '<null>'
        self.lc, self.rc = [], []

    def reset_states(self):
        self.predicted_head = -1
        self.predicted_dep = '<null>'
        self.lc, self.rc = [], []


ROOT_TOKEN = Token(token_id=0, word=ROOT, pos=ROOT, head=-1, dep=ROOT)
NULL_TOKEN = Token(token_id=-1, word=NULL, pos=NULL, head=-1, dep=NULL)
UNK_TOKEN = Token(token_id=-1, word=UNK, pos=UNK, head=-1, dep=UNK)


class Sentence:

    def __init__(self, tokens):
        self.root = ROOT_TOKEN
        self.tokens = tokens
        self.stack = [ROOT_TOKEN]
        self.buffer = tokens
        arcs = []
        for token in tokens:
            pair = [token.token_id, token.head]
            pair.sort()
            arcs.append(pair)
        arcs.sort(key=lambda x: x[0])
        self.arcs = arcs
        self.predicted_arcs = None

    def is_projective(self):
        """ determines if sentence is projective when ground truth given """
        for pair in self.arcs:
            for compare_pair in self.arcs:
                if pair != compare_pair:
                    #starts within the two and ends outide of the two (after the compare_pair[1])
                    if pair[0] > compare_pair[0] and pair[0]<compare_pair[1] and pair[1]>compare_pair[1]:
                        return False
        return True


    def get_trans(self):  # this function is only used for the ground truth
        """ decide transition operation from [shift, left_arc, or right_arc] """
        #there are enough things on the stack
        def no_dependents(buffer, parent_index):

            return not (parent_index in [word.head for word in buffer])
        stack = self.stack
        if len(stack)>=2:
            if stack[-2].head == stack[-1].token_id:
                return "LEFT" #+stack[-1].dep
            elif (stack[-1].head == stack[-2].token_id and no_dependents(self.buffer, stack[-1].token_id)):
                return "RIGHT"
            else:
                return "SHIFT"
        else:
            return "SHIFT"


    def check_trans(self, potential_trans):
        """ checks if transition can legally be performed"""
        pass
    
    
    def update_state(self, curr_trans, predicted_dep=None):
        """ updates the sentence according to the given transition (may or may not assume legality, you implement) """
        # shift, left, right
        if curr_trans == "LEFT":
            self.stack[-1].lc.append(self.stack[-2])
            del self.stack[-2]
        elif curr_trans =="RIGHT":
            self.stack[-2].rc.append(self.stack[-1])
            del self.stack[-1]
        else: #SHIFT

            self.stack.append(self.buffer.pop(0))

    def get_features(self):

        def get_word(token):
            return 'None' if token ==[] else token.word
        def get_pos(token):
            return 'None' if token ==[] else token.pos
        def get_arc(token):
            return 'None' if token ==[] else token.dep
            
        # words: word and pos for stack[-3:-1] buffer[0:2]  2*3*2
        words_stack = [get_tag(word) for word in self.stack[-3:] for get_tag in (get_word, get_pos)]
        words_buff = [get_tag(word) for word in self.buffer[:3] for get_tag in (get_word, get_pos)]
        words_feats = ['None']*(6-len(words_stack))+words_stack+words_buff+['None']*(6-len(words_buff))
        
        # children: word and pos and arc labels for lc[0:1] rc[-2:-1] for stack[-2:-1] 3*4*2
        children, children_feats, grandchildren, grandchildren_feats = [], [], [], []
        stack_head = self.stack[-2:]
        for word in stack_head: 
            children+=[word.lc[:1],word.lc[1:2],word.rc[-2:-1],word.rc[-1:]]
        padded_children = [[],[],[],[]]*(2-len(stack_head))+children
        for item in padded_children:
            if item == []:
                children_feats+=['None','None','None']
            else:
                children_feats+=[get_tags(item[0]) for get_tags in (get_word, get_pos, get_arc)]
        
        # grandchildren: word and pos and arc labels for lc[0].lc[0] rc[-1].rc[-1] for stack[-2:-1] 3*2*2
        stack_head = self.stack[-2:]
        for word in stack_head: 
            try:
                grandchildren.append(word.lc[0].lc[0])
            except:
                grandchildren.append([])
            try:
                grandchildren.append(word.rc[-1].rc[-1])
            except:
                grandchildren.append([])
        padded_grandchildren = [[],[],[],[]]*(2-len(stack_head))+grandchildren

        for item in padded_grandchildren:
            if item == []:
                grandchildren_feats+=['None','None','None']
            else:
                grandchildren_feats+=[get_tags(item) for get_tags in (get_word, get_pos, get_arc)]
        return words_feats + children_feats + grandchildren_feats
        
        # • the first three words on the stack and the buffer (and their POS tags) (12 features)
        # • the words, POS tags, and arc labels of the first and second leftmost and rightmost children
        #  of the first two words on the stack, (24 features)
        # • the words, POS tags, and arc labels of leftmost child of the leftmost child and rightmost child
        # of rightmost child of the first two words of the stack (12 features)


class FeatureGenerator:

    def __init__(self):

        pass

    def extract_features(self, sentence):
        """ returns the features for a sentence parse configuration """
        word_features = []
        pos_features = []
        dep_features = []

        return word_features, pos_features, dep_features
