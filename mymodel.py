import torch
import torch.nn as nn
class myModel(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, vocab_size, embedding_dim, pretrained = None):
        super(myModel, self).__init__()
        # embeddings
        if pretrained == None:
            self.embeddings = nn.Embedding(vocab_size, embedding_dim)
        else:
            self.embeddings= nn.Embedding.from_pretrained(pretrained)
        # Linear function
        self.fc1 = nn.Linear(input_dim, hidden_dim) 
        # Non-linearity
        self.relu = nn.ReLU()
        # Linear function 
        self.fc2 = nn.Linear(hidden_dim, output_dim)  

        self.dropout = nn.Dropout(0.10)
        self.input_dim = input_dim

    def forward(self, x):
       
        embeds = self.embeddings(x).view((-1, self.input_dim))
        # Linear function
        out = self.fc1(embeds)
        # Non-linearity
        out = self.relu(out)
        # Linear function 
        out = self.fc2(out)
        #softmax
        # out = self.softmax(out)
        return out