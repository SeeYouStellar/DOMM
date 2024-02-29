import torch
from model import Policy
import numpy as np
from replaybuffer import ReplayBuffer
import torch.nn.functional as F
import hyperparameter as hp
class DQN(object):
    def __init__(self, state_dim, action_dim):
        self.state_dim = state_dim
        self.action_dim = action_dim

        self.policy = Policy(state_dim, action_dim)
        self.target = Policy(state_dim, action_dim)

        self.optimizer = torch.optim.RMSprop(self.policy.parameters(), lr=hp.lr)
        self.replybuffer = ReplayBuffer(hp.MEMORY_CAPACITY)

    def choose_action(self, s, deterministic):

        s = torch.unsqueeze(torch.tensor(s, dtype=torch.float), 0)  # (1, 2*m+n+m*(n+1))
        prob_weights = self.policy(s).detach().numpy().flatten()  # probability distribution(numpy)

        if deterministic:  # We use the deterministic policy during the evaluating
            a = self.argmax_action(prob_weights)
            return a
        else:  # We use the stochastic policy during the training
            a = self.greedy_action(prob_weights)
            return a

    def greedy_action(self, prob_weights):
        a = []
        group = lambda lst, n: [lst[i:i + n] for i in range(0, len(lst), n)]
        prob_weights = group(prob_weights, 3)
        for prob_weight in prob_weights:
            prob_weight = F.softmax(torch.tensor(prob_weight), dim=0)
            a.append(np.random.choice(range(3), p=prob_weight))
        return np.array(a)  # (m*n,)
    def argmax_action(self, prob_weights):
        a = []
        group = lambda lst, n: [lst[i:i + n] for i in range(0, len(lst), n)]
        prob_weights = group(prob_weights, 3)
        for prob_weight in prob_weights:
            prob_weight = F.softmax(torch.tensor(prob_weight, dtype=float), dim=0)
            a.append(np.argmax(prob_weight))
        return np.array(a)  # (m*n,)

    def learn(self, bs, ba, br, bdone, bs_):
        # compute loss
        # qvals = self.policy(bs).gather(1, ba.unsqueeze(1)).squeeze()  # gather:按索引从输入张量中检索值
        qvals = self.policy(bs).squeeze()  # (batch_size, action_dim)
        qvals = torch.reshape(qvals, (qvals.size()[0], -1, 3))
        qvals = qvals[torch.arange(qvals.size(0)).unsqueeze(1), torch.arange(qvals.size(1)), ba]  # (batch_size, m*n)

        next_qvals = self.target(bs_).detach()
        next_qvals_values, _ = torch.reshape(next_qvals, (next_qvals.size()[0], -1, 3)).max(dim=2)  # (batch_size, m*n)
        # print(br)
        # print(next_qvals_values)
        # print(next_qvals.size())
        y = br.unsqueeze(1) + hp.GAMMA * next_qvals_values  # (batch_size, m*(n+1))

        loss = F.mse_loss(y, qvals)

        # loss backward
        loss.backward()
        self.optimizer.step()
        self.optimizer.zero_grad()

        return loss.detach().numpy()




