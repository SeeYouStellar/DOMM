import numpy as np
import hyperparameter as hp
import math


class Env:
    def __init__(self):
        self.n = hp.N
        self.m = hp.M
        self.current_T = 0

        self.Cm = np.random.uniform(hp.cmin, hp.cmax, size=self.m)
        self.Dm = np.random.uniform(hp.dmin, hp.dmax, size=self.m)
        self.F0 = np.random.uniform(hp.f0min,  hp.f0max, size=self.m)
        self.Fn = np.random.uniform(hp.fnmin, hp.fnmax, size=self.n)
        self.Pm = np.random.uniform(hp.Pmin, hp.Pmax, size=self.m)

        # self.cm = np.random.uniform(2e9, 3e9, size=(hp.T + 1, m))
        # self.dm = np.random.uniform(2e8, 3e8, size=(hp.T + 1, m))
        # self.fn = np.random.uniform(5e9, 7e9, size=(hp.T + 1, n))
        # self.f0 = np.random.uniform(1.5e9, 2e9, size=self.m)

        self.current_state = None  # [dm, cm, fn, pai]
        self.r = hp.B * math.log2(1 + math.pow(10, hp.SNR / 10))  # 上下限不好确定，所以固定

        self.E_last = 0

    def env_change(self):
        # 每隔env_change_frequency个step对环境中的固定参数进行改变，模拟时变环境
        self.Cm = np.random.uniform(hp.cmin, hp.cmax, size=self.m)  # [m, 0]
        self.Dm = np.random.uniform(hp.dmin, hp.dmax, size=self.m)  # [m, 0]
        self.F0 = np.random.uniform(hp.f0min, hp.f0max, size=self.m)  # [m, 0]
        self.Fn = np.random.uniform(hp.fnmin, hp.fnmax, size=self.n)  # [n, 0]

    def compute_latency(self):
        percent_task = self.current_state[2 * self.m + self.n:]  # [m*(n+1),]
        percent_task = np.reshape(percent_task, [self.m, self.n + 1])  # [m, n+1]
        percent_task_locals = percent_task[:, 0]  # [m, 1]
        percent_task_caps = percent_task[:, 1:]  # [m, n]

        # Task offloading phase
        L1 = 0
        for i in range(1, self.m + 1):
            percent_task_i_m = percent_task_caps[i-1]  # [n,]
            lmi = percent_task_i_m * self.Dm[i-1] / self.r
            lmi = np.sum(lmi)
            L1 = max(L1, lmi)

        # Task computing phase
        l0 = np.max(percent_task_locals * self.Cm / self.F0)
        ln = 0
        for i in range(1, self.n + 1):
            percent_task_i_n = percent_task_caps[:, i-1].reshape(-1)  # [m,]
            lni = percent_task_i_n.dot(self.Cm.T) / self.Fn[i - 1]
            ln = max(ln, lni)
        L2 = max(l0, ln)

        L = L1 + L2
        return L
    def compute_energy(self):
        percent_task = self.current_state[2 * self.m + self.n:]  # [m*(n+1),]
        percent_task = np.reshape(percent_task, [self.m, self.n + 1])  # [m, n+1]
        percent_task_locals = percent_task[:, 0]  # [m, 1]
        percent_task_caps = percent_task[:, 1:]  # [m, n]

        # Task offloading phase
        E1 = 0
        for i in range(1, self.m + 1):
            percent_task_i_m = percent_task_caps[i - 1]  # [n,]
            emi = self.Pm[i-1] * percent_task_i_m * self.Dm[i - 1] / self.r
            emi = np.sum(emi)
            E1 += emi

        # Task computing phase
        e0 = np.sum(percent_task_locals * self.Cm * self.F0 * self.F0 * hp.zeta_user)
        en = 0
        for i in range(1, self.n + 1):
            percent_task_i_n = percent_task_caps[:, i-1].reshape(-1)  # [m,]
            eni = percent_task_i_n.dot(self.Cm.T) * self.Fn[i - 1] * self.Fn[i - 1] * hp.zeta_cap
            en += eni
        E2 = e0 + en

        E = E1 + E2

        return E

    def reset(self):
        self.Cm = np.random.uniform(hp.cmin, hp.cmax, size=self.m)  # [m, 0]
        self.Dm = np.random.uniform(hp.dmin, hp.dmax, size=self.m)  # [m, 0]
        self.F0 = np.random.uniform(hp.f0min, hp.f0max, size=self.m)  # [m, 0]
        self.Fn = np.random.uniform(hp.fnmin, hp.fnmax, size=self.n)  # [n, 0]
        self.Pm = np.random.uniform(hp.Pmin, hp.Pmax, size=self.m)

        pai = np.zeros((self.m, self.n + 1))
        pai[:, 0] = 1
        self.current_state = np.concatenate((self.Cm, self.Dm, self.Fn, pai.flatten()))

        return self.current_state

    def reward(self):
        L = self.compute_latency()
        E = self.compute_energy() / 100
        # reward in criterion 1
        reward1 = -(hp.lamdba * L + (1 - hp.lamdba) * E)

        # reward in criterion 2
        if L >= hp.Lth:
            reward2 = -hp.miu_1
        elif E < self.E_last:
            reward2 = hp.miu_2
        else:
            reward2 = -hp.miu_2

        self.E_last = E
        return reward1, reward2

    # def check(self):

    def step(self, action, step):

        # 时变环境
        if step % hp.env_change_frequency == 0:
            self.env_change()
        else:
            # get env from state
            self.Cm = self.current_state[:self.m]
            self.Dm = self.current_state[self.m:2*self.m]
            self.Fn = self.current_state[2*self.m:2*self.m+self.n]

        pai = self.current_state[2 * self.m + self.n:]
        pai = np.reshape(pai, [self.m, self.n+1])

        reward1, reward2 = self.reward()

        # if :
        #     done = True
        #     return self.current_state, 10, done

        # update pai
        def update_pai(pai, action, det):
            action = np.reshape(action, [hp.M, hp.N])
            updated_pai = np.copy(pai)  # Make a copy of pai to store updated values
            for i in range(action.shape[0]):
                for j in range(action.shape[1]):
                    if action[i, j] == 0:
                        if updated_pai[i, j+1] + det <= 1 and updated_pai[i, 0] - det >= 0:
                            updated_pai[i, j+1] += det
                            updated_pai[i, 0] -= det
                        else:
                            det_ = min(1 - updated_pai[i, j+1], updated_pai[i, 0])
                            updated_pai[i, j+1] += det_
                            updated_pai[i, 0] -= det_
                    elif action[i, j] == 2:
                        if updated_pai[i, 0] + det <= 1 and updated_pai[i, j+1] - det >= 0:
                            updated_pai[i, j+1] -= det
                            updated_pai[i, 0] += det
                        else:
                            det_ = min(1 - updated_pai[i, 0], updated_pai[i, j+1])
                            updated_pai[i, j+1] -= det_
                            updated_pai[i, 0] += det_
                    # If action[i, j] == 1, no change to updated_pai[i, j]
            return updated_pai

        updated_pai = update_pai(pai, action, hp.det)

        cm = self.Cm
        dm = self.Dm
        fn = self.Fn

        # update current_state
        self.current_state = np.concatenate([cm, dm, fn, updated_pai.flatten()])
        done = False
        return self.current_state, reward1, done
