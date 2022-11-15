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

#%%
# loading vocab and training data
with open('vocab.json', 'r') as vocab_file:
    orig_vocab_dict = json.load(vocab_file)
vocab_size = len(orig_vocab_dict)
vocab_dict = defaultdict(lambda:orig_vocab_dict['<unk>'],orig_vocab_dict)
print("vocab loaded")
#%%
with open("train.converted", 'r', encoding='utf-8') as file:
    data = file.read().splitlines()
    data_split = [line.strip().split() for line in data]

print("dataset loaded")
pd_data = pd.DataFrame(data_split)
pd_labels = pd_data.pop(48)

#make into numbers --> tensors
numbers_data = pd_data.applymap(lambda x: vocab_dict[x])
torched_data = torch.from_numpy(numbers_data.values)

label_dict = defaultdict(lambda: 71,{label:i for i,label in enumerate(vocab_dict['LABEL_LIST'])})
numbers_labels = pd_labels.apply(lambda x: label_dict[x])
torched_labels = torch.from_numpy(numbers_labels.values)

trainset = torch.utils.data.TensorDataset(torched_data, torched_labels)
trainloader = torch.utils.data.DataLoader(trainset, batch_size=1000, shuffle=True, drop_last=True)


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

embedding_dim = 10
input_dim = 48*embedding_dim
hidden_dim = 50
output_dim = 71
label_list = vocab_dict['LABEL_LIST']

model = myModel(input_dim, hidden_dim, output_dim, vocab_size, embedding_dim)

# training parameters
loss_func = nn.CrossEntropyLoss(reduction='sum')
learning_rate = 0.001
optimizer = torch.optim.Adagrad(model.parameters(), lr=learning_rate)
num_epochs = 5



'''
TRAIN THE MODEL
'''
iter = 0
for_graphing = {'train_acc':[], 'dev_acc': [], 'iter':[]}
for epoch in range(num_epochs):
    for i,(data,labels) in enumerate(tqdm(trainloader)):
        
        optimizer.zero_grad()

        # Forward pass to get output/logits
        outputs = model(data)
        # yhat_labels=[label_list[thing] for thing in torch.argmax(outputs, dim=1)]
        yhats=torch.argmax(outputs, dim=1)
        loss = loss_func(outputs, labels)

        # Getting gradients w.r.t. parameters
        loss.backward()

        # Updating parameters
        optimizer.step()

        iter += 1

        if iter % 1000 == 0:
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
plt.savefig("graphs/curve.png")
plt.show() 
  
model_params = {'embedding_dim': embedding_dim,
                'input_dim': input_dim,
                'hidden_dim': hidden_dim,
                'output_dim': output_dim,
                'label_list': label_list,
                'label_dict': {label:i for i,label in enumerate(vocab_dict['LABEL_LIST'])},
                'vocab_dict': orig_vocab_dict}

torch.save({'model_params': model_params,
            'state_dict':model.state_dict()}, "train.model")



# if __name__ == '__main__':
#     parser = argparse.ArgumentParser(description='Neural net training arguments.')

#     parser.add_argument('-u', type=int, help='number of hidden units')
#     parser.add_argument('-l', type=float, help='learning rate')
#     parser.add_argument('-f', type=int, help='max sequence length')
#     parser.add_argument('-b', type=int, help='mini-batch size')
#     parser.add_argument('-e', type=int, help='number of epochs to train for')
#     parser.add_argument('-E', type=str, help='word embedding file')
#     parser.add_argument('-i', type=str, help='training file')
#     parser.add_argument('-o', type=str, help='model file to be written')
#     parser.add_argument('-g', type=bool, help='whether to output learning curve graph')

#     args = parser.parse_args()

#     model = train_torch(input_params = {'units': args.u, 
#                                         'learning_rate':args.l, 
#                                         'seq':args.f, 
#                                         'batch':args.b, 
#                                         'epochs': args.e, 
#                                         'embed':args.E,
#                                         'input': args.i,
#                                         'graph': args.g,
#                                         'output': args.o}) # probably want to pass some arguments here
        

#         #TODO: fix the batches
#         # write the test-torch
#         # write the README

# %%
