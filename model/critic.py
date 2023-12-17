import torch
import torch.nn as nn
import os


class Critic(nn.Module):
    def __init__(self, input_dim, chkpt_dir="models/trpo"):
        super().__init__()
        self.checkpoint_file = os.path.join(chkpt_dir, "torch_critic")
        self.value = nn.Sequential(
            nn.Linear(input_dim, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Linear(512, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Linear(256, 64),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Linear(64, 1)
        )
        self.optimizer = torch.optim.Adam(self.parameters(), lr=1e-3)
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.to(self.device)

    def forward(self, state):
        value = self.value(state)
        return value

    def save_checkpoint(self):
        torch.save(self.state_dict(), self.checkpoint_file)

    def load_checkpoint(self):
        self.load_state_dict(torch.load(self.checkpoint_file, map_location=self.device))
