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
from collections import defaultdict
import pickle
import numpy as np
import argparse

def parse(model_file, input_file, output_file):
    print("called")
    # load file
    with open(input_file, 'r', encoding='utf-8') as file:
        data = file.read().splitlines()
        data_split = [line.strip().split() for line in data]
        by_sentences = [list(y) for x, y in itertools.groupby(data_split, lambda z: z == []) if not x]

    # load model
    with open(model_file, "rb") as file:
        saved = pickle.load(file)
    model_params = saved['model_params']
    state_dict = saved['state_dict']
    vocab_size = model_params['vocab_len'] if 'vocab_len' in model_params.keys() else len(model_params['vocab_dict'])
    model = myModel(model_params['input_dim'], model_params['hidden_dim'], model_params['output_dim'], vocab_size , model_params['embedding_dim'])
    model.load_state_dict(state_dict)
    vocab_dict = defaultdict(lambda:model_params['vocab_dict']['<unk>'],model_params['vocab_dict'])
    labelset = model_params['label_list']
    right_indeces = [i for (i,label) in enumerate(labelset) if 'RIGHT' in label]
    left_indeces = [i for (i,label) in enumerate(labelset) if 'LEFT' in label]
    shift_indeces = [i for (i,label) in enumerate(labelset) if 'SHIFT' in label]

    def mask_illegal(illegal_trans:list, logits): #numpy array
        if "RIGHT" in illegal_trans:
            logits[right_indeces] = np.NINF
        if "LEFT" in illegal_trans:
            logits[left_indeces] = np.NINF
        if "SHIFT" in illegal_trans:
            logits[shift_indeces] = np.NINF
        return logits


    all_updated_sentences = []
    word_count, correct_head, correct_dep = 0,0,0

    for sentence in tqdm(by_sentences):
        tokens_list = []
        for word in sentence:
            tokens_list.append(Token(token_id=word[0], word=word[2], pos=word[4], head=word[6], dep=word[7]))
        this_sentence = Sentence(tokens_list)
        while not (this_sentence.buffer == [] and len(this_sentence.stack) == 1):
            features = this_sentence.get_features()
            torched_features = torch.tensor([vocab_dict[feat] for feat in features])

            # predict arc here!
            model.eval()
            forward_output = model(torched_features)
            masked_forward_output = mask_illegal(this_sentence.check_trans(), forward_output.detach().numpy()[0])
            yhat_index = masked_forward_output.argmax()
            pred_label, pred_arc = this_sentence.format_prediction(labelset[yhat_index])
            this_sentence.update_state(pred_label, pred_arc)
        #it's parsed!
        predictions = this_sentence.predicted_arcs
        updated_sentence = []
        for word in sentence:
            token_id = int(word[0])
            word_copy = word.copy()
            word_copy[6] = str(predictions[token_id][0])
            word_copy[7] = predictions[token_id][1]
            updated_sentence.append(word_copy)
            word_count+=1
            correct_head += 1 if word[6]==word_copy[6] else 0
            correct_dep += (1 if word[7]==word_copy[7] else 0)
        

        all_updated_sentences.append(updated_sentence)
    print("Head accuracy is {} and dep accuracy is {}".format(correct_head/word_count, correct_dep/word_count))
    print(correct_dep, correct_head, word_count)

        # if counter == 4:
        #     break

    #%%

    with open(output_file, 'w', encoding='utf-8') as file:
        for sentence in all_updated_sentences:
            sentence_string = ''
            for word in sentence:
                sentence_string+='\t'.join(word)+'\n'
            file.write(sentence_string+'\n')



if __name__ == '__main__':
    #python parse.py -m [modelfile] -i [inputfile] -o [outputfile]
    parser = argparse.ArgumentParser(description='parsing file details')
    parser.add_argument('-m', type=str, help='model file path')
    parser.add_argument('-i', type=str, help='input file in CONLL format to be parsed')
    parser.add_argument('-o', type=str, help='output file path')

    args = parser.parse_args()

    parse(args.m, args.i, args.o)
