"""train.py trains a model given data preprocessed by preparedata.py and writes a model file.
"""
#%%
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import json
import numpy as np 
import pandas as pd
from collections import defaultdict

with open('vocab.json', 'r') as vocab_file:
    vocab_dict = json.load(vocab_file)
vocab_size = len(vocab_dict)
vocab_dict = defaultdict(lambda:vocab_dict['<unk>'],vocab_dict)

print("vocab loaded")
# embeds = nn.Embedding(len(vocab_dict), 10)  # ?? words in vocab, 10 dimensional embeddings
# lookup_tensor = torch.tensor([vocab_dict["hello"]], dtype=torch.long)
# hello_embed = embeds(lookup_tensor)
# print(hello_embed)



# import torchvision.transforms as transforms
# import torchvision.datasets as dsets

'''
STEP 1: LOADING DATASET


# train_dataset = dsets.MNIST(root='./data', 
#                             train=True, 
#                             transform=transforms.ToTensor(),
#                             download=True)

# test_dataset = dsets.MNIST(root='./data', 
#                            train=False, 
#                            transform=transforms.ToTensor())

'''
with open("train.converted", 'r', encoding='utf-8') as file:
    data = file.read().splitlines()
    data_split = [line.strip().split() for line in data]

print("dataset loaded")
data = pd.DataFrame(data_split)
labels = data.pop(48)

#%%
#make into numbers
numbers_data = data.applymap(lambda x: vocab_dict[x])
torched_data = torch.from_numpy(numbers_data.values)
chunked_data = torch.chunk(torched_data,10000, dim =0)

label_dict = {label:i for i,label in enumerate(vocab_dict['LABEL_LIST'])}
numbers_labels = labels.apply(lambda x: label_dict[x])
torched_labels = torch.from_numpy(numbers_labels.values)
chunked_labels = torch.chunk(torched_labels,10000, dim =0)
print("chunked")
#%%

'''
STEP 2: MAKING DATASET ITERABLE


# batch_size = 100
# n_iters = 3000
# num_epochs = n_iters / (len(train_dataset) / batch_size)
# num_epochs = int(num_epochs)

# train_loader = torch.utils.data.DataLoader(dataset=train_dataset, 
#                                            batch_size=batch_size, 
#                                            shuffle=True)

# test_loader = torch.utils.data.DataLoader(dataset=test_dataset, 
#                                           batch_size=batch_size, 
#                                           shuffle=False)
'''
'''
STEP 3: CREATE MODEL CLASS
'''
#%%
from statistics import mean
class myModel(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super(myModel, self).__init__()
        # embeddings
        self.embeddings = nn.Embedding(vocab_size, embedding_dim)
        # Linear function
        self.fc1 = nn.Linear(input_dim, hidden_dim) 
        # Non-linearity
        self.relu = nn.ReLU()
        # Linear function 
        self.fc2 = nn.Linear(hidden_dim, output_dim)  
        # I think i need softmax here?
        self.softmax = nn.Softmax(dim=None)

    def forward(self, x):
       
        embeds = self.embeddings(x).view((-1, 480))
        # Linear function
        out = self.fc1(embeds)
        # Non-linearity
        out = self.relu(out)
        # Linear function 
        out = self.fc2(out)
        #softmax
        out = self.softmax(out)
        return out
'''
STEP 4: INSTANTIATE MODEL CLASS
'''
embedding_dim = 10
input_dim = 48*embedding_dim
hidden_dim = 50
output_dim = 71
label_list = vocab_dict['LABEL_LIST']
model = myModel(input_dim, hidden_dim, output_dim)

'''
STEP 5: INSTANTIATE LOSS CLASS
'''
loss_func = nn.CrossEntropyLoss()


'''
STEP 6: INSTANTIATE OPTIMIZER CLASS
'''
learning_rate = 0.001

optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)

'''
STEP 7: TRAIN THE MODEL
'''
iter = 0
num_epochs = 5
for epoch in range(num_epochs):
    for i,batch in enumerate(chunked_data):
        
        optimizer.zero_grad()

        # Forward pass to get output/logits
        outputs = model(batch)
        # yhat_labels=[label_list[thing] for thing in torch.argmax(outputs, dim=1)]
        yhats=torch.argmax(outputs, dim=1)
        print(mean([1 if yhat == y else 0 for yhat, y in zip(yhats, chunked_labels[i])]))

        # Calculate Loss: softmax --> cross entropy loss
        loss = loss_func(outputs, chunked_labels[i])


        # Getting gradients w.r.t. parameters
        loss.backward()

        # Updating parameters
        optimizer.step()
        # if iter % 500 == 0:
            # print(loss.item())

        iter += 1

        # if iter % 500 == 0:
        #     # Calculate Accuracy         
        #     correct = 0
        #     total = 0
        #     # Iterate through test dataset
        #     for images, labels in test_loader:
        #         # Load images with gradient accumulation capabilities
        #         images = images.view(-1, 28*28).requires_grad_()

        #         # Forward pass only to get logits/output
        #         outputs = model(images)

        #         # Get predictions from the maximum value
        #         _, predicted = torch.max(outputs.data, 1)

        #         # Total number of labels
        #         total += labels.size(0)

        #         # Total correct predictions
        #         correct += (predicted == labels).sum()

        #     accuracy = 100 * correct / total

        #     # Print Loss
        #     print('Iteration: {}. Loss: {}. Accuracy: {}'.format(iter, loss.item(), accuracy))
#%%

#
































def train_torch(input_params):
    embedding_length={'g':50, 'u':100, 'f':300}[input_params['embed'][0]]
    big_x, labels, labelset = get_embeddings(input_params['input'], input_params['seq'], input_params['embed'], embedding_length)   
    big_x = torch.from_numpy(big_x) 
    num_sentences = big_x.shape[0]

    model = nn.Sequential(nn.Linear(embedding_length*input_params['seq'], input_params['units']),
                        nn.ReLU(),
                        nn.Linear(input_params['units'],len(labelset)),
                        nn.Softmax(dim=1)) 
    loss_function = nn.CrossEntropyLoss(reduction='sum')

    losses = []
    batch_size = input_params['batch']
    for epoch in tqdm(range(input_params['epochs'])):
        for batch in range(num_sentences//batch_size):
            start_index = int((epoch*batch) %num_sentences)
            end_index= int((start_index+batch_size % num_sentences)+1)

            if end_index<start_index:
                continue
            #batch will have a block of 
            x = big_x[start_index:end_index]

            y = torch.Tensor([labelset.index(label) for label in labels[start_index:end_index]]).long()
            y_onehot = nn.functional.one_hot(y,num_classes=len(labelset))

            forward_output = model(x.float())
            loss = loss_function(forward_output, y)
            
            if (epoch*batch_size+batch) % 100 == 99:
                losses.append(loss.item())

            model.zero_grad()
            loss.backward()
            with torch.no_grad():
                for param in model.parameters():
                    param -= input_params['learning_rate'] * param.grad
            # optimizer.step()
    
    import matplotlib.pyplot as plt
    plt.plot(losses)
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.show()
    plt.savefig("graphs/{}-curve.png".format("torch"))

    torch.save({'labelset': labelset,
                'input_params': input_params,
                'state_dict':model.state_dict()}, input_params['output'])
    return None


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Neural net training arguments.')

    parser.add_argument('-u', type=int, help='number of hidden units')
    parser.add_argument('-l', type=float, help='learning rate')
    parser.add_argument('-f', type=int, help='max sequence length')
    parser.add_argument('-b', type=int, help='mini-batch size')
    parser.add_argument('-e', type=int, help='number of epochs to train for')
    parser.add_argument('-E', type=str, help='word embedding file')
    parser.add_argument('-i', type=str, help='training file')
    parser.add_argument('-o', type=str, help='model file to be written')
    parser.add_argument('-g', type=bool, help='whether to output learning curve graph')

    args = parser.parse_args()

    model = train_torch(input_params = {'units': args.u, 
                                        'learning_rate':args.l, 
                                        'seq':args.f, 
                                        'batch':args.b, 
                                        'epochs': args.e, 
                                        'embed':args.E,
                                        'input': args.i,
                                        'graph': args.g,
                                        'output': args.o}) # probably want to pass some arguments here
        

        #TODO: fix the batches
        # write the test-torch
        # write the README
