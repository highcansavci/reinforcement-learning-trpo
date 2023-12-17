import torch.nn as nn


class KL(nn.Module):
    def __init__(self, max_d_kl):
        super().__init__()
        self.max_d_kl = max_d_kl

    def forward(self, prob_p, prob_q):
        return (prob_p * (prob_p.log() - prob_q.log())).sum(-1).mean()
