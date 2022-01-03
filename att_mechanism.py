from _typeshed import Self
import torch
import torch.nn as nn

class SelfAttention(nn.Module):
    def __init__(self, embed_size, heads): # heads are the numbers of the splitted embeding
        super(SelfAttention).__init__()
        self.embed_size = embed_size
        self.heads = heads
        self.head_dim = embed_size // heads 

        assert (self.head_dim * heads == embed_size), 'Embed size must be div by the number of heads'

        self.values = nn.Linear(self.head_dim, self.head_dim , bias=False)
        self.values = nn.Linear(self.head_dim, self.head_dim , bias=False)
        self.queries = nn.Linear(self.head_dim, self.head_dim , bias=False)
        self.fc = nn.Linear(heads*self.head_dim, embed_size)
        

