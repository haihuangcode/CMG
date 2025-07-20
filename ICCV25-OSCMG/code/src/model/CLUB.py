import torch
import torch.nn as nn
import torch.nn.functional as F


class CLUBSample_group(nn.Module):  # Sampled version of the CLUB estimator
    def __init__(self, x_dim, y_dim, hidden_size):
        super(CLUBSample_group, self).__init__()
        self.x_dim = x_dim
        self.y_dim = y_dim
        self.hidden_size = hidden_size
        self.p_mu = nn.Sequential(nn.Linear(self.x_dim, self.hidden_size // 2),
                                  nn.ReLU(),
                                  nn.Linear(self.hidden_size // 2, self.hidden_size // 2),
                                  nn.ReLU(),
                                  nn.Linear(self.hidden_size // 2, self.hidden_size // 2),
                                  nn.ReLU(),
                                  nn.Linear(self.hidden_size // 2, self.y_dim))

        self.p_logvar = nn.Sequential(nn.Linear(self.x_dim, self.hidden_size // 2),
                                      nn.ReLU(),
                                      nn.Linear(self.hidden_size // 2, self.hidden_size // 2),
                                      nn.ReLU(),
                                      nn.Linear(self.hidden_size // 2, self.hidden_size // 2),
                                      nn.ReLU(),
                                      nn.Linear(self.hidden_size // 2, self.y_dim),
                                      nn.Tanh())

    def get_mu_logvar(self, x_samples):
        mu = self.p_mu(x_samples)
        logvar = self.p_logvar(x_samples)
        return mu, logvar

    def loglikeli(self, x_samples, y_samples):  # unnormalized loglikelihood
        mu, logvar = self.get_mu_logvar(x_samples)  # mu/logvar: (bs, y_dim)
        # mu = mu.unsqueeze(1).expand(-1, y_samples.shape[1], -1).reshape(-1, mu.shape[
        #     -1])  # (bs, y_dim) -> (bs, 1, y_dim) -> (bs, T, y_dim) -> (bs*T, y_dim)
        mu = mu.reshape(-1, mu.shape[-1])
        #logvar = logvar.unsqueeze(1).expand(-1, y_samples.shape[1], -1).reshape(-1, logvar.shape[-1])
        logvar = logvar.reshape(-1, logvar.shape[-1])
        y_samples = y_samples.reshape(-1, y_samples.shape[-1])  # (bs, T, y_dim) -> (bs*T, y_dim)
        return (-(mu - y_samples) ** 2 / logvar.exp() - logvar).sum(dim=1).mean(dim=0) / 2

    def mi_est(self, x_samples, y_samples):  # x_samples: (bs, x_dim); y_samples: (bs, T, y_dim)

        mu, logvar = self.get_mu_logvar(x_samples)

        sample_size = x_samples.shape[0]
        # random_index = torch.randint(sample_size, (sample_size,)).long()
        random_index = torch.randperm(sample_size).long()

        # log of conditional probability of positive sample pairs
        #mu_exp1 = mu.unsqueeze(1).expand(-1, y_samples.shape[1], -1)  # (bs, y_dim) -> (bs, T, y_dim)
        mu_exp1 = mu

        # logvar_exp1 = logvar.unqueeze(1).expand(-1, y_samples.shape[1], -1).reshape(-1, logvar.shape[-1])
        positive = - ((mu_exp1 - y_samples) ** 2).mean(dim=1) / logvar.mean(dim=1).exp()  # mean along T
        negative = - ((mu_exp1 - y_samples[random_index]) ** 2).mean(dim=1) / logvar.mean(dim=1).exp()  # mean along T

        return (positive.sum(dim=-1) - negative.sum(dim=-1)).mean() / 2
