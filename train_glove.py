"""train.py trains a model given data preprocessed by preparedata.py and writes a model file.
"""
#%%
import torch
import torch.nn as nn
import json
import pandas as pd
from collections import defaultdict
from mymodel import myModel
import matplotlib.pyplot as plt
from tqdm import tqdm 
import pickle
import numpy as np
#TODO recreate dict with POS? no just add to glove if it's not in it
 
#use embeddings
with open('glove.6B.50d.txt', encoding='utf-8') as file:
    data = file.read().splitlines()
    data_split = [line.strip().split() for line in data]
# embedding_dict = {line[0]: line[1:] for line in data_split}
glove_stoi = {line[0]:i for i, line in enumerate(data_split)}
glove_table = torch.tensor([[float(value) for value in line[1:]] for line in data_split]) # torch.Size([400001, 50])
glove_length, embedding_dim = glove_table.shape
#%%
import json
#all of the pos and dep words
with open('my_vocab.json', 'r') as vocab_file:
    my_vocab_stoi = json.load(vocab_file)
label_list =  my_vocab_stoi.pop('LABEL_LIST')
# initialize a table of random floats between -0.01 and 0.01
my_vocab_table = (torch.rand(len(my_vocab_stoi), 50)-0.5)/100 #torch.Size([91, 50])

#%%

# combine tables and stoi for glove and my_vocab
#shift the indeces for my_vocab
my_vocab_stoi_shifted = {key:value+glove_length for key, value in my_vocab_stoi.items()}
glove_stoi.update(my_vocab_stoi)
orig_vocab_dict = glove_stoi
vocab_table = torch.cat((glove_table, my_vocab_table))
vocab_length, embedding_dim = vocab_table.shape

vocab_dict = defaultdict(lambda:orig_vocab_dict['<unk>'],orig_vocab_dict)
print("vocab loaded")
#%%
with open("train.converted", 'r', encoding='utf-8') as file:
    data = file.read().splitlines()
    data_split = [line.strip().split() for line in data]

print("dataset loaded")
pd_data = pd.DataFrame(data_split)
pd_labels = pd_data.pop(48)



#%%
#make into numbers --> tensors
numbers_data = pd_data.applymap(lambda x: vocab_dict[x])
torched_data = torch.from_numpy(numbers_data.values)

label_dict = defaultdict(lambda: 71,{label:i for i,label in enumerate(label_list)})
numbers_labels = pd_labels.apply(lambda x: label_dict[x])
torched_labels = torch.from_numpy(numbers_labels.values)

trainset = torch.utils.data.TensorDataset(torched_data, torched_labels)
trainloader = torch.utils.data.DataLoader(trainset, batch_size=100, shuffle=True, drop_last=True)


#DEV
with open("dev.converted", 'r', encoding='utf-8') as file:
    dev_data = file.read().splitlines()
    dev_data_split = [line.strip().split() for line in dev_data]
dev_pd_data = pd.DataFrame(dev_data_split)
dev_pd_labels = dev_pd_data.pop(48)

dev_data = torch.from_numpy(dev_pd_data.applymap(lambda x: vocab_dict[x]).values)
dev_labels = torch.from_numpy(dev_pd_labels.apply(lambda x: label_dict[x]).values)

#%%
# model parameters

embedding_dim = 50
input_dim = 48*embedding_dim
hidden_dim = 300
output_dim = 71
device = 'cpu'

model = myModel(input_dim, hidden_dim, output_dim, vocab_length, embedding_dim, vocab_table)
model.to(device)

# training parameters
loss_func = nn.CrossEntropyLoss(reduction='sum')
learning_rate = 0.01
optimizer = torch.optim.Adagrad(model.parameters(), lr=learning_rate)
num_epochs = 15



'''
TRAIN THE MODEL
'''
iter = 0
for_graphing = {'train_acc':[], 'dev_acc': [], 'iter':[]}
for epoch in range(num_epochs):
    for i,(data,labels) in enumerate(tqdm(trainloader)):
        data = data.to(device)
        labels = labels.to(device)
        
        optimizer.zero_grad()

        # Forward pass to get output/logits
        outputs = model(data)
        yhats=torch.argmax(outputs, dim=1)
        loss = loss_func(outputs, labels)

        # Getting gradients w.r.t. parameters
        loss.backward()

        # Updating parameters
        optimizer.step()

        iter += 1

        if iter % 3000 == 0:
            acc = (yhats==labels).sum()/labels.size(0)
            outputs = model(dev_data)

                # Get predictions from the maximum value
            _, predicted = torch.max(outputs.data, 1)

            dev_acc = (predicted == dev_labels).sum()/dev_labels.size(0)
            # print("for epoch {} trauin accuracy is {} and loss is {} and dev acc is {}".format(epoch,acc, loss.item(), dev_acc))
            for_graphing['dev_acc'].append(dev_acc)
            for_graphing['train_acc'].append(acc)
            for_graphing['iter'].append(iter)

plt.plot(for_graphing['iter'], for_graphing['train_acc'], label="Train acc", alpha=0.7)
plt.plot(for_graphing['iter'], for_graphing['dev_acc'], label="dev acc", alpha=0.7)

plt.title("Learning Curves")
plt.xlabel("Number of Iters")
plt.ylabel("accuracies")
plt.legend()
plt.tight_layout()
plt.savefig("graphs/curve_glove.png")
plt.show() 
  
model_params = {'embedding_dim': embedding_dim,
                'input_dim': input_dim,
                'hidden_dim': hidden_dim,
                'output_dim': output_dim,
                'label_list': label_list,
                'label_dict': {label:i for i,label in enumerate(label_list)},
                'vocab_dict': orig_vocab_dict,
                'vocab_len': vocab_table.shape[0]
                }


with open("train.glove.model", "wb") as file:
    pickle.dump({
    'model_params': model_params,
    'state_dict': model.state_dict()}, file)

