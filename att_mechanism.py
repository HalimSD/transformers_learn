import torch
import torch.nn as nn

class SelfAttention(nn.Module):
    def __init__(self, embed_size, heads): # heads are the numbers of the splitted embeding
        super(SelfAttention, self).__init__()
        self.embed_size = embed_size
        self.heads = heads
        self.head_dim = embed_size // heads 

        assert (self.head_dim * heads == embed_size), 'Embed size must be div by the number of heads'

        self.values = nn.Linear(self.head_dim, self.head_dim , bias=False)
        self.values = nn.Linear(self.head_dim, self.head_dim , bias=False)
        self.queries = nn.Linear(self.head_dim, self.head_dim , bias=False)
        self.fc_out = nn.Linear(heads*self.head_dim, embed_size)

    def forward(self, keys, values, query, mask):
        N = query.shape[0] # number of training examples, How many examples we send in at the same time
        value_len, key_len, query_len = values.shape[1], keys.shape[1], query.shape[1]

        # split embedings into self.heads pieces
        values = values.reshape(N, value_len, self.heads, self.head_dim)
        keys = keys.reshape(N, key_len, self.heads, self.head_dim)
        queries = query.reshape(N, query_len, self.heads, self.head_dim)

        #multiply the queries by the keys
        ene = torch.einsum("nqhd,nkhd->nhqk", [queries, keys])
        # queries shape : (N, query_len, heads, head_dim)
        # keys shape : (N, key_len, heads, head_dim)
        # ene shape : (N, heads, query_len, key_len)

        if mask is not None:
            ene = ene.masked_fill(mask == 0 , float("-1e20"))

        attention = torch.softmax(ene/(self.embed_size ** (1/2)), dim=3)

        out = torch.einsum("nhql,nlhd->nqhd", [attention, values]).reshape(N, query_len, self.heads*self.head_dim)
        # attention shape : (N, heads, query_len, key_len)
        # values shape : (N, values_len, heads, head_dim)
        # out shape : after einsum (N, query_len, heads, head_dim) then flatten the last 2 dims

        out = self.fc_out(out)
        return out 

class TransformerBlock(nn.Module):
    def __init__(self, embed_size, heads, dropout, forward_expansion):
        super(TransformerBlock, self).__init__()
        self.attention= SelfAttention(embed_size, heads)
        self.norm1 = nn.LayerNorm(embed_size)
        self.norm2 = nn.LayerNorm(embed_size)

        self.feed_forward = nn.Sequential(
            nn.Linear(embed_size, forward_expansion*embed_size),
            nn.ReLU(),
            nn.Linear(forward_expansion*embed_size, embed_size)
        )

        self.dorpout = nn.Dropout(dropout)

    def forward(self, value, key, query, mask):
        attention = self.attention(value, key, query, mask)
        x = self.dorpout(self.norm1(attention + query))

        forward = self.feed_forward(x)

        out = self.dorpout(self.norm2(forward+x))
        return out

class Encoder(nn.Module):
    def __init__(self, src_vocab_size, embed_size, num_layers, heads, 
                    forward_expansion, dropout, max_lenght, device):
        super(Encoder, self).__init__()

        self.embed_size = embed_size
        self.device = device

        self.word_embedding = nn.Embedding(src_vocab_size, embed_size)
        self.position_embedding = nn.Embedding(max_lenght, embed_size)

        self.layers = nn.ModuleList([
            TransformerBlock(embed_size, heads, dropout,forward_expansion)
        ])

        self.dropout = nn.Dropout(dropout)

    def forward(self, x, mask):
        N, seq_len = x.shape
        positions = torch.arange(0, seq_len).expand(N, seq_len).to(self.device)

        out = self.dropout(self.word_embedding(x) + self.position_embedding(positions))

        for layer in self.layers:
            out = layer(out,out,out,mask)

        return out
        

