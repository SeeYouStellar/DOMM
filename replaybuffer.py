from dataclasses import dataclass, field
import numpy as np
import torch
@dataclass
class ReplayBuffer:
    maxsize: int
    size: int = 0
    state: list = field(default_factory=list)
    action: list = field(default_factory=list)
    next_state: list = field(default_factory=list)
    reward: list = field(default_factory=list)
    done: list = field(default_factory=list)

    def push(self, state, action, reward, done, next_state):
        if self.size < self.maxsize:
            self.state.append(state)
            self.action.append(action)
            self.reward.append(reward)
            self.done.append(done)
            self.next_state.append(next_state)
        else:
            # 溢出就覆盖
            position = self.size % self.maxsize
            self.state[position] = state
            self.action[position] = action
            self.reward[position] = reward
            self.done[position] = done
            self.next_state[position] = next_state
        self.size += 1

    def sample(self, n):
        total_number = self.size if self.size < self.maxsize else self.maxsize
        indices = np.random.randint(total_number, size=n)
        state = [self.state[i] for i in indices]
        action = [self.action[i] for i in indices]
        reward = [self.reward[i] for i in indices]
        done = [self.done[i] for i in indices]
        next_state = [self.next_state[i] for i in indices]
        return torch.tensor(state, dtype=torch.float), torch.tensor(action), torch.tensor(reward, dtype=torch.float), torch.tensor(done, dtype=torch.bool), torch.tensor(next_state, dtype=torch.float)
