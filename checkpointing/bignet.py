import warnings
from time import time

import torch

warnings.filterwarnings('ignore')

class BigNet(torch.nn.Module):
    class Block(torch.nn.Module):
        def __init__(self, hidden_size, device="mps"):
            super().__init__()
            self.transformer_layer = torch.nn.TransformerEncoderLayer(
                d_model=hidden_size,
                nhead=8,
                dim_feedforward=4*hidden_size,
                dropout=0,
                batch_first=True,
                device=device,
            )

        def forward(self, x):
            from torch.utils.checkpoint import checkpoint
            return self.transformer_layer(x)
        
    def __init__(self, input_size=4096, hidden_size=2048, output_size=10, device="mps"):
        super().__init__()
        self.model = torch.nn.Sequential(
            torch.nn.Linear(input_size, hidden_size, device=device),
            *[self.Block(hidden_size, device=device) for _ in range(8)],
            torch.nn.ReLU(),
            torch.nn.Linear(hidden_size, output_size, device=device),
        )

    def forward(self, x):
        return self.model(x)
    
def test_forward(model, batch_size=1, seq_len=1024):
    # One forward pass
    print(f"Model weights           {torch.mps.current_allocated_memory() / 2**30:6.2f} GB")
    x = torch.randn(batch_size, seq_len, 4096, device="mps")
    print(f"Model weights + inputs  {torch.mps.current_allocated_memory() / 2**30:6.2f} GB")
    max_mem = float("-inf")
    with torch.no_grad():
        y = model(x).sum()
    # torch.mps.synchronize()

    # max_mem = max(max_mem, torch.mps.driver_allocated_memory())
    # print(f"Forward (peak)          {max_mem / 2**30:6.2f} GB")

def test_backward(model, batch_size=1, seq_len=1024):
    # One forward pass
    print(f"Model weights           {torch.mps.current_allocated_memory() / 2**30:6.2f} GB")
    x = torch.randn(batch_size, seq_len, 4096, device="mps")
    print(f"Model weights + inputs  {torch.mps.current_allocated_memory() / 2**30:6.2f} GB")
    # Sum for 'most efficient' loss function memory possible
    y = model(x).sum()
    y.backward()
    print(f"Backward (alloc)         {torch.mps.current_allocated_memory() / 2**30:6.2f} GB")

def train_step(model, batch_size=1, seq_len=1024):
    print(f"Model weights           {torch.mps.current_allocated_memory() / 2**30:6.2f} GB")
    optim = torch.optim.AdamW(model.parameters(), lr=1e-3)
    print(f"Model weights + opt     {torch.mps.current_allocated_memory() / 2**30:6.2f} GB")
    for _ in range(2):
        print(f"Step {_}")
        x = torch.randn(batch_size, seq_len, 4096, device="mps", requires_grad=True)
        print(f"        Model wgh + opt + inp {torch.mps.current_allocated_memory() / 2**30:6.2f} GB")
        t0 = time()
        y = model(x).sum()
        y.item()
        t1 = time()
        print(f"        Time:                                                          {t1-t0:0.2f} s")
        t0 = time()
        y.backward()
        if x.grad is not None:
            x.grad.view(-1)[0].item()
        t1 = time()
        print(f"        Backward (alloc)        {torch.mps.current_allocated_memory() / 2**30:6.2f} GB")
        print(f"        Time:                                                          {t1-t0:0.2f} s")
        optim.step()
        print(f"        Step (alloc)            {torch.mps.current_allocated_memory() / 2**30:6.2f} GB")
        optim.zero_grad()
        print(f"        zero_grad (alloc)       {torch.mps.current_allocated_memory() / 2**30:6.2f} GB")



if __name__ == "__main__":
    batch_size, seq_len = 4, 1024
    model = BigNet(device="mps")

    # test_forward(model, batch_size=batch_size, seq_len=seq_len)
    # test_backward(model, batch_size=batch_size, seq_len=seq_len)
    train_step(model, batch_size=batch_size, seq_len=seq_len)