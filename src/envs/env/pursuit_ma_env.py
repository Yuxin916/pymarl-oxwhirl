import random
import yaml
import numpy as np
from math import pi
import matplotlib.pyplot as plt

from gym import spaces
from gym.utils import seeding

import hydra
from omegaconf import DictConfig

from .agv_model import AGVAgent, RobotStatusIdx
from .pursuit_env_base import PursuitEnvBase
from .utils import plot_arrow


class PursuitMAEnv(PursuitEnvBase):
    def __init__(self, cfg):
        super().__init__(cfg)

        self.n_pursuer = cfg['agent']['pursuer']['n']
        self.n_evader = cfg['agent']['evader']['n']
        # Init pursuer and evader models
        assert self.n_pursuer >= 1, "n_pursuer must be greater than 1"
        assert self.n_evader >= 1, "n_evader must be greater than 1"

        self.shared_reward = True

        # Number of pursuer
        self.num_agents = cfg['agent']['pursuer']['n']

        # Init pursuer and evader models
        self.pursuer_agents = [AGVAgent(i, self.max_v, self.min_v, self.max_w, self.max_v_acc, self.max_w_acc,  # Limit
                                        self.init_x, self.init_y, self.init_yaw, 0.0, 0.0, self.pursuer_radius,  # Init
                                        True, self.laser_angle_max, self.laser_angle_min, self.laser_angle_step,
                                        self.laser_range)  # Laser
                               for i in range(self.n_pursuer)]

        self.evader_model = AGVAgent(0, self.max_v, self.min_v, self.max_w, self.max_v_acc, self.max_w_acc,  # Limit
                                     self.target_init_x, self.target_init_y, self.target_init_yaw, 0.0, 0.0,
                                     self.evader_radius,  # Init
                                     laser_on=False)  # Laser

        # Spaces # No agents index
        self.action_space = []
        self.observation_space = []
        self.share_observation_space = []

        share_obs_dim = 0  # TODO： ？

        for agents in self.pursuer_agents:
            self.action_space.append(spaces.Box(low=np.array([-1.0, -1.0], dtype=np.float32),
                                                high=np.array([1.0, 1.0], dtype=np.float32)))
            self.obs_num = agents.laser_map_size + 4  # 4 for reference angle (2) and distance (2)
            share_obs_dim += self.obs_num
            self.observation_space.append(spaces.Box(-np.inf, np.inf, shape=(self.obs_num,), dtype=float))
        self.share_observation_space = [spaces.Box(
            low=-np.inf, high=+np.inf, shape=(share_obs_dim,), dtype=np.float32)] * self.num_agents

        self.fig = None
        self.writer = None

    def seed(self, seed=None):
        self.np_RNG, seed_ = seeding.np_random(seed)
        print('The seed: ', seed_)
        return [seed_]

    def step(self, action_n):

        self.current_step += 1
        obs_n = []
        reward_n = []
        done_n = []
        truncate_n = []
        info_n = []
        goal_distances = []

        # set action for each agent
        for i, agent in enumerate(self.pursuer_agents):
            self._set_action(action_n[i, :], agent)

        # record observation for each agent
        for agent in self.pursuer_agents:
            obs_n.append(self._get_obs(agent))
            # TODO: move env check into world step and refine _get_obs for redundant
            agent.collide = self.obstacle_collision_check(agent) or self.boundary_collision_check(agent)
            agent.catch = self.catch_check(agent)
            if agent.catch:
                agent.caught = True
            agent.truncate = self.current_step >= self.limit_step
            # goal_distances.append(self._get_reward(agent)[0])
            reward_n.append([self._get_reward(agent)])
            info = {'individual_reward': self._get_reward(agent), 'Done': self._get_info(agent)}
            # print(info)
            info_n.append(info)
            if info['Done']:
                # print(str(agent.idx), info)
                pass
            done_n.append(self._get_done(agent))
            truncate_n.append(agent.truncate)
        both_catch = self._get_task_complete()
        if both_catch:
            reward_n = [[reward[0]+250.0] for reward in reward_n]
        done_n = [done or both_catch for done in done_n]

        # all agents get total reward in cooperative case, if shared reward, all agents have the same reward, and reward is sum
        reward = np.sum(reward_n)
        if self.shared_reward:
            reward_n = [[reward]] * self.num_agents

        return np.array(obs_n), np.array(reward_n), done_n, truncate_n, info_n

    def reset(self, **kwargs):
        self.reward_list = []

        # obstacle coordinates
        self.ob_list = self.cfg['obstacle']['o_coordinates']

        # initial setting for pursuer
        if self.pursuer_fixed:
            self.init_x = self.cfg['agent']['pursuer']['x']
            self.init_y = self.cfg['agent']['pursuer']['y']
            self.init_yaw = self.cfg['agent']['pursuer']['yaw']
            for agent in self.pursuer_agents:
                agent.set_init_state(self.init_x, self.init_y, self.init_yaw)
        else:
            for agent in self.pursuer_agents:
                init_x, init_y = self._random_spawn(agent)
                init_yaw = random.random() * pi * random.choice([1, -1])
                agent.set_init_state(init_x, init_y, init_yaw)

        self.pursuer_radius = self.cfg['agent']['pursuer']['radius']
        # initial condition for evader
        self.target_init_x = self.cfg['agent']['evader']['x']
        self.target_init_y = self.cfg['agent']['evader']['y']
        self.target_init_yaw = self.cfg['agent']['evader']['yaw']
        self.ob_radius = self.cfg['agent']['evader']['radius']

        self.evader_model.set_init_state(self.target_init_x, self.target_init_y, self.target_init_yaw)

        self.current_step = 0

        obs_n = []
        for i, agent in enumerate(self.pursuer_agents):
            # self._set_action([-1.0, 0.0], agent)
            obs_n.append(self._get_obs(agent))

        return np.array(obs_n), {}

    def render(self, mode='human'):
        plt.cla()

        # executed linear&angular velocity + set linear&angular velocity
        # plt.text(0, 4, "Exe: " + str(pursuer_state[-4:-2]), fontsize=10)
        # plt.text(0, 4.5, "Des: " + str(pursuer_state[-2:]), fontsize=10)

        # draw evader and pursuer
        for i, agent in enumerate(self.pursuer_agents):
            pursuer_state = agent.state
            plt.text(0, 4 - 0.6 * i, "The no.{} Action: ".format(i + 1) +
                     str([pursuer_state[RobotStatusIdx.LinearVelocityDes.value],
                          pursuer_state[RobotStatusIdx.AngularVelocityDes.value]]), fontsize=10)
            circle_pursuer = plt.Circle(
                (pursuer_state[RobotStatusIdx.XCoordinateID.value], pursuer_state[RobotStatusIdx.YCoordinateID.value]),
                agent.robot_radius, color="r")
            plt.gca().add_patch(circle_pursuer)

        evader_state = self.evader_model.state
        circle_evader = plt.Circle(
            (evader_state[RobotStatusIdx.XCoordinateID.value], evader_state[RobotStatusIdx.YCoordinateID.value]),
            self.evader_radius, color="blue")
        plt.gca().add_patch(circle_evader)

        ax = plt.gca()
        # draw obstacles
        if self.ob_list is not None:
            for o in self.ob_list:
                plt.gca().add_patch(plt.Circle(o, self.ob_radius, color="black"))
                # draw boundary wall
                rect_wall = plt.Rectangle((self.boundary_xy[0], self.boundary_xy[1]), self.boundary_wh[0],
                                          -self.boundary_wh[1],
                                          fill=False, color="red", linewidth=2)
                plt.gca().add_patch(rect_wall)

        # draw robot movement
        for agent in self.pursuer_agents:
            pursuer_state = agent.state
            plot_arrow(pursuer_state[0], pursuer_state[1], pursuer_state[2])
            # Render laser
            agent.laser.render(ax)

        plt.xlim([self.boundary_wall[0] - 1, self.boundary_wall[2] + 1])
        plt.ylim([self.boundary_wall[3] - 1, self.boundary_wall[1] + 1])
        plt.grid(True)

        if self.writer is not None:
            self.writer.grab_frame()
        plt.pause(0.05)

    def _get_reward(self, agent):
        goal_distance = np.linalg.norm(agent.state[:RobotStatusIdx.YawAngleID.value] -
                                       self.evader_model.state[:RobotStatusIdx.YawAngleID.value])

        reward = 1 - goal_distance * 0.1  # TODO

        if agent.collide:
            reward += -20
        # if agent.catch and not agent.caught:  # Only reward once
        #     reward += 20
        if agent.truncate:
            reward += -20

        # return goal_distance, reward - 0.5  # time
        return reward - 0.5  # time

    def _get_done(self, agent):
        if self.current_step >= self.limit_step or agent.collide:
            return True
        else:
            return False

    def _get_task_complete(self):
        for agent in self.pursuer_agents:
            if not agent.catch:
                return False
        return True

    def set_writer(self, writer):
        self.fig = plt.figure(figsize=(7, 7))
        self.writer = writer


def fixed_action_env_test(env):
    # show plot
    for _ in range(100):
        # print("The initial observation is {}".format(obs))
        env.reset()
        a = 0
        while a < 100:
            action = np.array([[1.0, 0.0], [0.0, 0.0]])
            obs, reward, done, truncate, info = env.step(action)
            env.render(mode="human")
            print(reward)
            done_flag = False
            for ele in done:
                if ele:
                    done_flag = True
                    break
            if done_flag:
                break
            for ele in truncate:
                if ele:
                    done_flag = True
                    break
            if done_flag:
                break
            # print("The new reward is {}".format(reward))
            a += 1


# @hydra.main(version_base=None, config_path="../config/environment", config_name="pursuit_ma_env")
# def main(cfg: DictConfig):
#     '''call env'''
#     env = PursuitMAEnv(cfg)
#
#     '''Test Env'''
#     fixed_action_env_test(env)
#
#
# def main_yaml():
#     with open('../config/environment/PE.yaml') as f:
#         cfg = yaml.load(f, Loader=yaml.FullLoader)
#     '''call env'''
#     env = PursuitMAEnv(cfg)
#
#     '''Test Env'''
#     fixed_action_env_test(env)
#
#
# if __name__ == '__main__':
#     main()
