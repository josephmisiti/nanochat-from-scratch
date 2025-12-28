import torch
import torch.nn as nn
from torch.nn import functional as F
torch.manual_seed(1337)


with open('input.txt', 'r', encoding='utf-8') as f:
    text = f.read()


chars = sorted(list(set(text)))
vocab_size = len(chars)

stoi = { ch:i for i,ch in enumerate(chars) }
itos = { i:ch for i,ch in enumerate(chars) }
encode = lambda s: [stoi[c] for c in s] # encoder: take a string, output a list of integers
decode = lambda l: ''.join([itos[i] for i in l]) # decoder: take a list of integers, output a string


data = torch.tensor(encode(text), dtype=torch.long)


batch_size = 32

def get_batch(data):
    # Random starting indices (not the last position)
    ix = torch.randint(len(data) - 1, (batch_size,))
    xb = data[ix]       # (batch_size,)
    yb = data[ix + 1]   # (batch_size,)
    return xb, yb

xb, yb = get_batch(data)
print(xb.shape)  # torch.Size([32])
print(yb.shape)  # torch.Size([32])


class BigramLanguageModel(nn.Module):
    def __init__(self, vocab_size):
        super().__init__()
        self.token_embedding_table = nn.Embedding(vocab_size, vocab_size)

    
    def forward(self, x, targets=None):
        logits = self.token_embedding_table(x)  # (batch_size, vocab_size

        loss = F.cross_entropy(logits, targets)

        return logits, loss


m = BigramLanguageModel(vocab_size)


optimizer = torch.optim.Adam(m.parameters(), lr=1e-3)

for step in range(10000):
    # Get a batch
    xb, yb = get_batch(data)
    
    # Forward pass
    logits, loss = m(xb, yb)
    
    # Backward pass
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    
    if step % 1000 == 0:
        print(f"Step {step}, Loss: {loss.item():.4f}")