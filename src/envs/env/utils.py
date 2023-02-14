import torch
import torch.nn as nn
import torch.nn.functional as F
from numpy import inf
from torch.utils.tensorboard import SummaryWriter
import gym
import numpy as np
import logging
import math
from math import cos, sin, pi, atan2, degrees
import os
import copy

import matplotlib.pyplot as plt

# global parameter
MAX_V = 3
MIN_V = 0.0
MAX_W = pi / 6
MIN_W = -pi / 6


def normalize_angle(angle):
    norm_angle = angle % (2 * math.pi)
    if norm_angle > math.pi:
        norm_angle -= 2 * math.pi
    return norm_angle


def plot_arrow(x, y, yaw, length=1.2, width=0.3):  # pragma: no cover
    plt.arrow(x, y, length * math.cos(yaw), length * math.sin(yaw),
              head_length=width, head_width=width)
    plt.plot(x, y)


def cal_samples(sample_region, basic_sample_reso, sample_reso_scale):
    '''
    :param sample_length:
    :param basic_sample_reso:
    :param sample_reso_scale:
    :return: number of samples obtained from each direction
    '''
    nums = []
    for sample_length in sample_region:
        a = math.log(sample_length / basic_sample_reso * (sample_reso_scale - 1) + 1)
        b = math.log(sample_reso_scale)
        nums.append(int(a / b))
    return nums


def get_perception_map(sample_map_width, sample_map_height, basic_sample_reso, sample_reso_scale, sample_map_center):
    percept_sample_map = []
    for i in range(sample_map_width):
        tmp_list_y = []
        for j in range(sample_map_height):
            x = basic_sample_reso * (1 - sample_reso_scale ** abs(i - sample_map_center[0])) \
                / (1 - sample_reso_scale)
            if i - sample_map_center[0] >= 0:
                x = x
            else:
                x = -x
            y = basic_sample_reso * (1 - sample_reso_scale ** abs(j - sample_map_center[1])) \
                / (1 - sample_reso_scale)
            if j - sample_map_center[1] >= 0:
                y = y
            else:
                y = -y
            tmp_list_y.append(np.array([x, y]))
        percept_sample_map.append(tmp_list_y)
    return percept_sample_map


# filter obstacle out of percept region
def filter_ob(obstacle_radius, ob_list, percept_region):
    percept_region_expanded = percept_region + np.array(
        [-obstacle_radius, obstacle_radius, -obstacle_radius, obstacle_radius])

    # [-4.3, 4.3, -2.3, 6.3]
    def is_in_percept_region(ob):
        check_ob_center_in = percept_region_expanded[0] <= ob[1] <= percept_region_expanded[1] and \
                             percept_region_expanded[2] <= ob[0] <= percept_region_expanded[3]

        return check_ob_center_in

    filtered_ob_list = list(
        filter(is_in_percept_region, ob_list))  # todo: sames if is_in_percept_region is empty can not filter
    return filtered_ob_list


# def load_model(path):
#     loaded_model = PPO.load(path)
#     # show the save hyperparameters
#     print("loaded:", "gamma =", loaded_model.gamma, "n_steps =", loaded_model.n_steps)
#
#     return loaded_model


# def record_video(eval_env, model, video_length=500, prefix='', video_folder='videos/'):
#     """
#   :param env_name
#   :param model: (RL model)
#   :param video_length: (int)
#   :param prefix: (str)
#   :param video_folder: (str)
#   """
#
#     # check_env(eval_env, warn=True)
#
#     # Start the video at step=0 and record 500 steps
#     eval_env = VecVideoRecorder(eval_env, video_folder=video_folder,
#                                 record_video_trigger=lambda step: step == 0, video_length=video_length,
#                                 name_prefix=prefix)
#
#     obs = eval_env.reset()
#     for _ in range(video_length):
#         action, _ = model.predict(obs)
#         obs, _, _, _ = eval_env.step(action)
#
#     # Close the video recorder
#     eval_env.close()


# def show_videos(video_path='', prefix=''):
#     """
#   Taken from https://github.com/eleurent/highway-env
#
#   :param video_path: (str) Path to the folder containing videos
#   :param prefix: (str) Filter the video, showing only the only starting with this prefix
#   """
#     html = []
#     for mp4 in Path(video_path).glob("{}*.mp4".format(prefix)):
#         video_b64 = base64.b64encode(mp4.read_bytes())
#         html.append('''<video alt="{}" autoplay
#                     loop controls style="height: 400px;">
#                     <source src="data:video/mp4;base64,{}" type="video/mp4" />
#                 </video>'''.format(mp4, video_b64.decode('ascii')))
#     ipythondisplay.display(ipythondisplay.HTML(data="<br>".join(html)))


def before_train_evaluate(model, num_episodes=100):
    """
    Evaluate a RL agent
    :param model: (BaseRLModel object) the RL Agent
    :param num_episodes: (int) number of episodes to evaluate it
    :return: (float) Mean reward for the last num_episodes
    """
    # This function will only work for a single Environment
    env = model.get_env()
    all_episode_rewards = []
    for i in range(num_episodes):
        episode_rewards = []
        done = False
        obs = env.reset()
        while not done:
            # _states are only useful when using LSTM policies
            action, _states = model.predict(obs)
            # here, action, rewards and dones are arrays
            # because we are using vectorized env
            obs, reward, done, info = env.step(action)
            episode_rewards.append(reward)

        all_episode_rewards.append(sum(episode_rewards))

    mean_episode_reward = np.mean(all_episode_rewards)
    print("Random Agent, before training: Mean reward:", mean_episode_reward, "Num episodes:", num_episodes)

    return mean_episode_reward


# class NormalizeActionWrapper(gym.Wrapper):
#     """
#     :param env: (gym.Env) Gym environment that will be wrapped
#     """
#
#     def __init__(self, env):
#         # Retrieve the action space
#         action_space = env.action_space
#         assert isinstance(action_space,
#                           gym.spaces.Box), "This wrapper only works with continuous action space (spaces.Box)"
#         # Retrieve the max/min values
#         self.low, self.high = action_space.low, action_space.high
#
#         # We modify the action space, so all actions will lie in [-1, 1]
#         env.action_space = gym.spaces.Box(low=np.array([-1, 1]), high=np.array([-1, 1]))
#
#         # Call the parent constructor, so we can access self.env later
#         super(NormalizeActionWrapper, self).__init__(env)
#
#     def rescale_action(self, scaled_action):
#         """
#         Rescale the action from [-1, 1] to [low, high]
#         (no need for symmetric action space)
#         :param scaled_action: (np.ndarray)
#         :return: (np.ndarray)
#         """
#         return self.low + (0.5 * (scaled_action + 1.0) * (self.high - self.low))
#
#     def reset(self):
#         """
#         Reset the environment
#         """
#         return self.env.reset()
#
#     def step(self, action):
#         """
#         :param action: ([float] or int) Action taken by the agent
#         :return: (np.ndarray, float, bool, dict) observation, reward, is the episode over?, additional informations
#         """
#         # Rescale action from [-1, 1] to original [low, high] interval
#         rescaled_action = self.rescale_action(action)
#         obs, reward, done, info = self.env.step(rescaled_action)
#         return obs, reward, done, info
#


class Actor(nn.Module):
    def __init__(self, state_dim, action_dim):
        super(Actor, self).__init__()

        self.layer_1 = nn.Linear(state_dim, 800)
        self.layer_2 = nn.Linear(800, 600)
        self.layer_3 = nn.Linear(600, action_dim)
        self.tanh = nn.Tanh()

    def forward(self, s):
        s = F.relu(self.layer_1(s))
        s = F.relu(self.layer_2(s))
        a = self.tanh(self.layer_3(s))
        return a


class Critic(nn.Module):
    def __init__(self, state_dim, action_dim):
        super(Critic, self).__init__()

        self.layer_1 = nn.Linear(state_dim, 800)
        self.layer_2_s = nn.Linear(800, 600)
        self.layer_2_a = nn.Linear(action_dim, 600)
        self.layer_3 = nn.Linear(600, 1)

        self.layer_4 = nn.Linear(state_dim, 800)
        self.layer_5_s = nn.Linear(800, 600)
        self.layer_5_a = nn.Linear(action_dim, 600)
        self.layer_6 = nn.Linear(600, 1)

    def forward(self, s, a):
        s1 = F.relu(self.layer_1(s))
        self.layer_2_s(s1)
        self.layer_2_a(a)
        s11 = torch.mm(s1, self.layer_2_s.weight.data.t())
        s12 = torch.mm(a, self.layer_2_a.weight.data.t())
        s1 = F.relu(s11 + s12 + self.layer_2_a.bias.data)
        q1 = self.layer_3(s1)

        s2 = F.relu(self.layer_4(s))
        self.layer_5_s(s2)
        self.layer_5_a(a)
        s21 = torch.mm(s2, self.layer_5_s.weight.data.t())
        s22 = torch.mm(a, self.layer_5_a.weight.data.t())
        s2 = F.relu(s21 + s22 + self.layer_5_a.bias.data)
        q2 = self.layer_6(s2)
        return q1, q2


class TD3(object):
    def __init__(self, state_dim, action_dim, max_action):
        # Initialize the Actor network
        self.actor = Actor(state_dim, action_dim).to(device)
        self.actor_target = Actor(state_dim, action_dim).to(device)
        self.actor_target.load_state_dict(self.actor.state_dict())
        self.actor_optimizer = torch.optim.Adam(self.actor.parameters())

        # Initialize the Critic networks
        self.critic = Critic(state_dim, action_dim).to(device)
        self.critic_target = Critic(state_dim, action_dim).to(device)
        self.critic_target.load_state_dict(self.critic.state_dict())
        self.critic_optimizer = torch.optim.Adam(self.critic.parameters())

        self.max_action = max_action
        self.writer = SummaryWriter()
        self.iter_count = 0

    def get_action(self, state):
        # Function to get the action from the actor
        state = torch.Tensor(state.reshape(1, -1)).to(device)
        return self.actor(state).cpu().data.numpy().flatten()

    # training cycle
    def train(
            self,
            replay_buffer,
            iterations,
            batch_size=100,
            discount=1,
            tau=0.005,
            policy_noise=0.2,  # discount=0.99
            noise_clip=0.5,
            policy_freq=2,
    ):
        av_Q = 0
        max_Q = -inf
        av_loss = 0
        for it in range(iterations):
            # sample a batch from the replay buffer
            (
                batch_states,
                batch_actions,
                batch_rewards,
                batch_dones,
                batch_next_states,
            ) = replay_buffer.sample_batch(batch_size)
            state = torch.Tensor(batch_states).to(device)
            next_state = torch.Tensor(batch_next_states).to(device)
            action = torch.Tensor(batch_actions).to(device)
            reward = torch.Tensor(batch_rewards).to(device)
            done = torch.Tensor(batch_dones).to(device)

            # Obtain the estimated action from the next state by using the actor-target
            next_action = self.actor_target(next_state)

            # Add noise to the action
            noise = torch.Tensor(batch_actions).data.normal_(0, policy_noise).to(device)
            noise = noise.clamp(-noise_clip, noise_clip)
            next_action = (next_action + noise).clamp(-self.max_action, self.max_action)

            # Calculate the Q values from the critic-target network for the next state-action pair
            target_Q1, target_Q2 = self.critic_target(next_state, next_action)

            # Select the minimal Q value from the 2 calculated values
            target_Q = torch.min(target_Q1, target_Q2)
            av_Q += torch.mean(target_Q)
            max_Q = max(max_Q, torch.max(target_Q))
            # Calculate the final Q value from the target network parameters by using Bellman equation
            target_Q = reward + ((1 - done) * discount * target_Q).detach()

            # Get the Q values of the basis networks with the current parameters
            current_Q1, current_Q2 = self.critic(state, action)

            # Calculate the loss between the current Q value and the target Q value
            loss = F.mse_loss(current_Q1, target_Q) + F.mse_loss(current_Q2, target_Q)

            # Perform the gradient descent
            self.critic_optimizer.zero_grad()
            loss.backward()
            self.critic_optimizer.step()

            if it % policy_freq == 0:
                # Maximize the actor output value by performing gradient descent on negative Q values
                # (essentially perform gradient ascent)
                actor_grad, _ = self.critic(state, self.actor(state))
                actor_grad = -actor_grad.mean()
                self.actor_optimizer.zero_grad()
                actor_grad.backward()
                self.actor_optimizer.step()

                # Use soft update to update the actor-target network parameters by
                # infusing small amount of current parameters
                for param, target_param in zip(
                        self.actor.parameters(), self.actor_target.parameters()
                ):
                    target_param.data.copy_(
                        tau * param.data + (1 - tau) * target_param.data
                    )
                # Use soft update to update the critic-target network parameters by infusing
                # small amount of current parameters
                for param, target_param in zip(
                        self.critic.parameters(), self.critic_target.parameters()
                ):
                    target_param.data.copy_(
                        tau * param.data + (1 - tau) * target_param.data
                    )

            av_loss += loss
        self.iter_count += 1
        # Write new values for tensorboard
        self.writer.add_scalar("loss", av_loss / iterations, self.iter_count)
        self.writer.add_scalar("Av. Q", av_Q / iterations, self.iter_count)
        self.writer.add_scalar("Max. Q", max_Q, self.iter_count)

    def save(self, filename, directory):
        torch.save(self.actor.state_dict(), "%s/%s_actor.pth" % (directory, filename))
        torch.save(self.critic.state_dict(), "%s/%s_critic.pth" % (directory, filename))

    def load(self, filename, directory):
        self.actor.load_state_dict(
            torch.load("%s/%s_actor.pth" % (directory, filename))
        )
        self.critic.load_state_dict(
            torch.load("%s/%s_critic.pth" % (directory, filename))
        )


# Set the parameters for the implementation
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")  # cuda or cpu
seed = 0  # Random seed number
eval_freq = 5e3  # After how many steps to perform the evaluation
max_ep = 500  # maximum number of steps per episode
eval_ep = 10  # number of episodes for evaluation
max_timesteps = 5e6  # Maximum number of steps to perform
expl_noise = 1  # Initial exploration noise starting value in range [expl_min ... 1]
expl_decay_steps = (500000)  # Number of steps over which the initial exploration noise will decay over

expl_min = 0.1  # Exploration noise after the decay in range [0...expl_noise]
batch_size = 40  # Size of the mini-batch
discount = 0.99999  # Discount factor to calculate the discounted future reward (should be close to 1)
tau = 0.005  # Soft target update variable (should be close to 0)
policy_noise = 0.2  # Added noise for exploration
noise_clip = 0.5  # Maximum clamping values of the noise
policy_freq = 2  # Frequency of Actor network updates
buffer_size = 1e6  # Maximum size of the buffer
file_name = "TD3_velodyne"  # name of the file to store the policy
save_model = True  # Weather to save the model or not
random_near_obstacle = True  # To take random actions near obstacles or not


def init(module, weight_init, bias_init, gain=1):
    weight_init(module.weight.data, gain=gain)
    bias_init(module.bias.data)
    return module


def get_clones(module, N):
    return torch.nn.ModuleList([copy.deepcopy(module) for i in range(N)])