import numpy as np


P_PREFIX = '<p>'
L_PREFIX = '<l>'
ROOT = '<root>'
NULL = '<null>'
UNK = '<unk>'


class Token:

    def __init__(self, token_id, word, pos, head, dep):
        self.token_id = token_id
        self.word = word
        self.pos = pos
        self.head = head
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
            pair = [int(token.token_id), int(token.head)]
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
        else:
            self.stack.append(self.buffer.pop(0))


class FeatureGenerator:

    def __init__(self):
        pass

    def extract_features(self, sentence):
        """ returns the features for a sentence parse configuration """
        word_features = []
        pos_features = []
        dep_features = []

        return word_features, pos_features, dep_features
