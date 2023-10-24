import torch
import torch.nn as nn
import torch.nn.functional as F


# class Mine(nn.Module):
#     def __init__(self):
#         super(Mine, self).__init__()
#         self.fc1_x = nn.Linear(2048, 512)
#         self.fc1_y = nn.Linear(2048, 512)
#         self.fc2 = nn.Linear(512,1)
#     def forward(self, x,y):
#         h1 = F.leaky_relu(self.fc1_x(x)+self.fc1_y(y))
#         h2 = self.fc2(h1)
#         return h2
#
# Mine = Mine()
# def mi_estimator(x, y, y_):
#
#     joint, marginal = Mine(x, y), Mine(x, y_)
#     return torch.mean(joint) - torch.log(torch.mean(torch.exp(marginal)))

# x = torch.randn(32, 10, 2048)
# y = torch.randn(32, 10, 2048)
# y_ = torch.randn(32, 10, 2048)
# joint, marginal = Mine(x, y), Mine(x, y_)
# loss = torch.mean(joint) - torch.log(torch.mean(torch.exp(marginal)))
# print(loss)

# class Mine2(nn.Module):
#     def __init__(self, x_dim, y_dim, hidden_dim):
#         super(Mine2, self).__init__()

#
#
# class MINE(nn.Module):
#     def __init__(self, hidden_size=256):
#         super(MINE, self).__init__()
#         self.layers = nn.Sequential(nn.Linear(512, hidden_size),
#                                     nn.ReLU(),
#                                     nn.Linear(hidden_size, 1))
#
#     def forward(self, x, y):
#         batch_size = x.size(0)
#         tiled_x = torch.cat([x, x, ], dim=0)
#         print("tiled_x:",tiled_x.size())
#         idx = torch.randperm(batch_size)
#
#         shuffled_y = y[idx]
#         concat_y = torch.cat([y, shuffled_y], dim=0)
#         print("concat_y:", concat_y.size())
#
#
#         inputs = torch.cat([tiled_x, concat_y], dim=1)
#         print("inputs:",inputs.size())
#         logits = self.layers(inputs)
#
#         pred_xy = logits[:batch_size]
#         pred_x_y = logits[batch_size:]
#         loss = -(torch.mean(pred_xy)
#                  - torch.log(torch.mean(torch.exp(pred_x_y))))
#
#         return loss
# #


class MINE(nn.Module):
    def __init__(self, x_dim, y_dim, hidden_size):
        super(MINE, self).__init__()
        self.T_func = nn.Sequential(nn.Linear(x_dim + y_dim, hidden_size),
                                    nn.ReLU(),
                                    nn.Linear(hidden_size, 1))

    def forward(self, x_samples, y_samples):  # samples have shape [sample_size, dim]
        # shuffle and concatenate
        sample_size = y_samples.shape[0]
        random_index = torch.randint(sample_size, (sample_size,)).long()

        y_shuffle = y_samples[random_index]
        #print("y_shuffle", y_shuffle.size())

        # np默认返回float64类型。F.linear对精度傻了。所以加了个.to(torch.float32)
        T0 = self.T_func(torch.cat([x_samples, y_samples], dim=-1).to(torch.float32))
        #print("T0:",T0.size())
        T1 = self.T_func(torch.cat([x_samples, y_shuffle], dim=-1).to(torch.float32))
        #print("T1:", T1.size())

        lower_bound = T0.mean() - torch.log(T1.exp().mean())

        # compute the negative loss (maximise loss == minimise -loss)
        return lower_bound

    def learning_loss(self, x_samples, y_samples):
        return -self.forward(x_samples, y_samples)


class CLUBSample(nn.Module):  # Sampled version of the CLUB estimator
    def __init__(self, x_dim, y_dim, hidden_size):
        super(CLUBSample, self).__init__()
        self.p_mu = nn.Sequential(nn.Linear(x_dim, hidden_size // 2),
                                  nn.ReLU(),
                                  nn.Linear(hidden_size // 2, y_dim))

        self.p_logvar = nn.Sequential(nn.Linear(x_dim, hidden_size // 2),
                                      nn.ReLU(),
                                      nn.Linear(hidden_size // 2, y_dim),
                                      nn.Tanh())

    def get_mu_logvar(self, x_samples):
        mu = self.p_mu(x_samples)
        logvar = self.p_logvar(x_samples)
        return mu, logvar

    def loglikeli(self, x_samples, y_samples):
        mu, logvar = self.get_mu_logvar(x_samples)
        return (-(mu - y_samples) ** 2 / logvar.exp() - logvar).sum(dim=1).mean()

    def forward(self, x_samples, y_samples):
        mu, logvar = self.get_mu_logvar(x_samples)

        sample_size = x_samples.shape[0]
        # random_index = torch.randint(sample_size, (sample_size,)).long()
        random_index = torch.randperm(sample_size).long()

        positive = - (mu - y_samples) ** 2 / logvar.exp()
        negative = - (mu - y_samples[random_index]) ** 2 / logvar.exp()
        upper_bound = (positive.sum(dim=-1) - negative.sum(dim=-1)).mean()
        return upper_bound / 2.

    def learning_loss(self, x_samples, y_samples):
        return - self.loglikeli(x_samples, y_samples)

# x = torch.randn(32, 10, 512)
# y = torch.randn(32, 10, 2048)
#
# model = MINE(x_dim=512, y_dim=2048, hidden_size=256)
# loss = model.learning_loss(x, y)
# print(loss)