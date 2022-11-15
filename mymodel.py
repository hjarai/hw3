import torch
import torch.nn as nn
class myModel(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, vocab_size, embedding_dim):
        super(myModel, self).__init__()
        # embeddings
        self.embeddings = nn.Embedding(vocab_size, embedding_dim)
        # Linear function
        self.fc1 = nn.Linear(input_dim, hidden_dim) 
        # Non-linearity
        self.relu = nn.ReLU()
        # Linear function 
        self.fc2 = nn.Linear(hidden_dim, output_dim)  

        # self.dropout = nn.Dropout(0.25)

    def forward(self, x):
       
        embeds = self.embeddings(x).view((-1, 480))
        # Linear function
        out = self.fc1(embeds)
        # Non-linearity
        out = self.relu(out)
        # Linear function 
        out = self.fc2(out)
        #softmax
        # out = self.softmax(out)
        return out