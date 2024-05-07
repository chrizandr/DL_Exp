import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


class MultiHeadSDPAttentionBlock(nn.Module):
    def __init__(self, heads, dk, dv):
        super().__init__()
        self.attention_blocks = [SDPAttentionBlock(dk, dv, heads) for i in range(heads)]

    def forward(self, keys, queries, values, masking=False):
        out = [x(keys, queries, values, masking) for x in self.attention_blocks]
        out = torch.concat(out, 1)
        return out


class SDPAttentionBlock(nn.Module):
    def __init__(self, dk, dv, heads=1):
        super().__init__()
        self.dk = dk
        self.WQ = torch.nn.Linear(dk, int(dk / heads), bias=False)
        self.WK = torch.nn.Linear(dk, int(dk / heads), bias=False)
        self.WV = torch.nn.Linear(dv, int(dv / heads), bias=False)

    def forward(self, keys, queries, values, masking=False):
        Q_ = self.WQ(queries)
        K_ = self.WK(keys)
        y = torch.matmul(Q_, K_.T) / np.sqrt(keys.shape[1])

        if masking:
            num_samples = keys.shape[0]
            mask = torch.tensor([[0 if i <= j else -1 * float("inf")
                                for i in range(num_samples)] for j in range(num_samples)],
                                dtype=torch.float32)
            y = mask + y

        z = F.softmax(y, 1)
        output = torch.matmul(z, self.WV(values))
        return output


class NNBlock(nn.Module):
    def __init__(self, dv, hidden_dim):
        super().__init__()
        self.l1 = nn.Linear(dv, hidden_dim, bias=True)
        self.l2 = nn.Linear(hidden_dim, dv, bias=True)

    def forward(self, values):
        z1 = F.relu(self.l1(values))
        z2 = self.l2(z1)
        return z2


def positional_encodings(dk, context_length=2048):
    i = np.array(list(range(dk)))
    even_mask = (i % 2 == 0).astype(int)
    encodings = []
    power = 10000 ** (2 * i / dk)

    for pos in range(context_length):
        angles = pos / power
        sins = np.sin(angles)
        coses = np.cos(angles)
        encoding = sins * even_mask + coses * (1-even_mask)
        encodings.append(encoding)

    return torch.tensor(np.array(encodings))


class EncoderBlock(nn.Module):
    def __init__(self, dk, dv, hidden_dim, heads):
        super().__init__()
        self.lnorm1 = nn.LayerNorm(dv)
        self.lnorm2 = nn.LayerNorm(dv)
        self.attention = MultiHeadSDPAttentionBlock(heads, dk, dv)
        self.ffn = NNBlock(dv, hidden_dim)

    def forward(self, keys, queries, values):
        z = self.attention(keys, queries, values)
        z = self.lnorm1(z + queries)
        y = self.ffn(z)
        y = self.lnorm2(y + z)
        return y


class Encoder(nn.Module):
    def __init__(self, num_blocks, dk, dv, hidden_dim, heads):
        super().__init__()
        self.blocks = [EncoderBlock(dk, dv, hidden_dim, heads) for _ in range(num_blocks)]

    def forward(self, keys, queries, values):
        for block in self.blocks:
            out = block(keys, queries, values)
            keys, queries, values = out, out, out
        return values


class DecoderBlock(nn.Module):
    def __init__(self, dk, dv, hidden_dim, heads):
        super().__init__()
        self.lnorm1 = nn.LayerNorm(dv)
        self.lnorm2 = nn.LayerNorm(dv)
        self.lnorm3 = nn.LayerNorm(dv)
        self.attention = MultiHeadSDPAttentionBlock(heads, dk, dv)
        self.cross_attention = MultiHeadSDPAttentionBlock(heads, dk, dv)
        self.ffn = NNBlock(dv, hidden_dim)

    def forward(self, keys, queries, values, encoder_values):
        z = self.attention(keys, queries, values, masking=True)
        z = self.lnorm1(z + queries)
        y = self.cross_attention(encoder_values, z, encoder_values)
        y = self.lnorm2(y + z)
        out = self.ffn(y)
        out = self.lnorm3(y + out)
        return out


class Decoder(nn.Module):
    def __init__(self, num_blocks, dk, dv, hidden_dim, heads):
        super().__init__()
        self.blocks = [DecoderBlock(dk, dv, hidden_dim, heads)
                       for _ in range(num_blocks)]

    def forward(self, keys, queries, values, encoder_values):
        for block in self.blocks:
            out = block(keys, queries, values, encoder_values)
            keys, queries, values = out, out, out
        return values


class Transformer(nn.Module):
    def __init__(self, context_length, dmodel, vocab_size):
        super().__init__()
        self.pos = positional_encodings(dmodel, context_length)
        self.embedding = nn.Embedding(vocab_size, dmodel)



if __name__ == "__main__":
    torch.manual_seed(0)
    dk = 128
    dv = 128
    heads = 4
    hidden_dim = 2048
    keys = torch.rand(10, dk)
    queries = torch.rand(10, dk)
    values = torch.rand(10, dv)

    block = EncoderBlock(dk, dv, hidden_dim, heads)
    # out = block(keys, queries, values)

    # block = NNBlock(dk, hidden_dim)
    out = block(keys, queries, values)
    out = positional_encodings(dk)
    breakpoint()