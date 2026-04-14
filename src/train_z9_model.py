import torch
import numpy as np
import sys
import os

# Ensure the local src folder is in the path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from z9_swarm_optimizer import Z9SwarmOptimizer

class SkyrmatronMini(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.fc = torch.nn.Linear(36, 9)
    def forward(self, x):
        return self.fc(x.mean(dim=1))

if __name__ == '__main__':
    model = SkyrmatronMini()
    optimizer = Z9SwarmOptimizer(model.parameters(), lr=3e-4, pop_size=9)

    data = np.random.randn(10000, 512, 36).astype(np.float32)
    labels = np.random.randint(0, 9, 10000)

    for epoch in range(8):
        for i in range(0, len(data), 64):
            batch = torch.from_numpy(data[i:i+64])
            target = torch.from_numpy(labels[i:i+64])
            optimizer.zero_grad()
            loss = torch.nn.functional.cross_entropy(model(batch), target)
            loss.backward()
            optimizer.step()
        print(f'Epoch {epoch} loss: {loss.item():.4f}')

    # Save to the models folder
    save_path = '/home/badgoysclub/Desktop/GEMINI/PROJECTS/Z9_ORACLE/models/skyrmatron_trained.pth'
    torch.save(model.state_dict(), save_path)
    print(f'Training complete — model saved to {save_path}')
