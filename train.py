from env import Env
import torch
import numpy as np
from Agent import DQN
import hyperparameter as hp
import matplotlib.pyplot as plt


# if gpu is to be used
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def plot_reward(rewards):
    plt.plot(rewards)
    plt.xlabel('Training Iteration(*100)')
    plt.ylabel('Reward')
    plt.title('Reward during training')
    plt.show()

if __name__ == '__main__':

    env = Env()
    state_dim = hp.N + 2 * hp.M + hp.M * (hp.N + 1)  # 神经网络输入大小
    action_dim = hp.N * hp.M * 3  # 神经网络输出大小
    agent = DQN(state_dim, action_dim)

    step = 0
    episode = 0
    while episode < hp.max_episodes:
        s = env.reset()
        episode += 1
        rewards = []
        mean_rewards = []
        mean_losss = []
        losss = []
        # done = False
        # while done is not True:
        for step in range(1, 15000):
            a = agent.choose_action(s, deterministic=True)  #
            s_, r, done = env.step(a, step)
            step += 1
            rewards.append(r)
            agent.replybuffer.push(s, a, r, done, s_)
            s = s_
            if step > hp.MEMORY_CAPACITY:
                bs, ba, br, bd, bs_ = agent.replybuffer.sample(n=hp.BATCH_SIZE)
                loss = agent.learn(bs, ba, br, bd, bs_)
                losss.append(loss)
                if step % 20 == 0:
                    mean_loss = np.mean(losss)
                    mean_losss.append(mean_loss)
                    losss = []
                    print("episode:{} \t step:{} \t loss:{}".format(episode, step, mean_loss))
            if step % hp.target_update_frequency == 0:
                agent.target.load_state_dict(agent.policy.state_dict())
            # if step % 20 == 0:
                # mean_reward = np.mean(rewards)
                # mean_rewards.append(mean_reward)
                # rewards = []
                # print("episode:{} \t step:{} \t reward:{}".format(episode, step, mean_reward))
            if step % hp.plot_frequency == 0:
                # plot_reward(mean_rewards)
                plot_reward(mean_losss)
