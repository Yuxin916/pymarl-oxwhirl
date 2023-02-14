import random
import numpy as np
from math import pi
import matplotlib.pyplot as plt

from gym import spaces
from gym.utils import seeding

import hydra
from omegaconf import DictConfig

from env.agv_model import AGVAgent, RobotStatusIdx
from env.pursuit_env_base import PursuitEnvBase
from utils import plot_arrow


class PursuitEnv2(PursuitEnvBase):
    def __init__(self, cfg):
        super().__init__(cfg)
        # Init pursuer and evader models
        self.pursuer_model = AGVAgent(0, self.max_v, self.min_v, self.max_w, self.max_v_acc, self.max_w_acc,  # Limit
                                      self.init_x, self.init_y, self.init_yaw, 0.0, 0.0, self.pursuer_radius,  # Init
                                      True, self.laser_angle_max, self.laser_angle_min, self.laser_angle_step,
                                      self.laser_range)  # Laser

        self.evader_model = AGVAgent(0, self.max_v, self.min_v, self.max_w, self.max_v_acc, self.max_w_acc,  # Limit
                                     self.target_init_x, self.target_init_y, self.target_init_yaw, 0.0, 0.0,
                                     self.evader_radius,  # Init
                                     laser_on=False)  # Laser

        # action space
        self.action_space = spaces.Box(low=np.array([-1.0, -1.0], dtype=np.float32),
                                       high=np.array([1.0, 1.0], dtype=np.float32))

        # observations space
        self.obs_num = self.pursuer_model.laser_map_size + 4  # 4 for reference angle (2) and distance (2)
        self.observation_space = spaces.Box(-np.inf, np.inf, shape=(self.obs_num,), dtype=float)

        self.fig = None
        self.writer = None

    def seed(self, seed=None):
        self.np_RNG, seed_ = seeding.np_random(seed)
        print('The seed: ', seed_)
        return [seed_]

    def step(self, action: np.ndarray):
        # Counter ++
        self.current_step += 1

        # Set action
        self._set_action(action, self.pursuer_model)

        # TODO: move env check into world step and refine _get_obs
        # Record observation
        obs = self._get_obs(self.pursuer_model)
        self.pursuer_model.collide = self.obstacle_collision_check(self.pursuer_model) or self.boundary_collision_check(self.pursuer_model)
        self.pursuer_model.catch = self.catch_check(self.pursuer_model)
        self.pursuer_model.truncate = self.current_step >= self.limit_step

        # Record reward, done, truncate, info
        reward = self._get_reward(self.pursuer_model)
        info = {'individual_reward': reward, 'Done': self._get_info(self.pursuer_model)}
        if info['Done']:
            # print(info)
            pass
        done = self._get_done(self.pursuer_model)
        truncate = self.pursuer_model.truncate

        return obs, reward, done, truncate, info

    def reset(self, **kwargs):
        self.reward_list = []

        # obstacle coordinates
        self.ob_list = self.cfg['obstacle']['o_coordinates']
        self.ob_radius = self.cfg['agent']['evader']['radius']

        self.pursuer_model.robot_radius = self.cfg['agent']['pursuer']['radius']
        # initial setting for pursuer
        if self.pursuer_fixed:
            self.init_x = self.cfg['agent']['pursuer']['x']
            self.init_y = self.cfg['agent']['pursuer']['y']
            self.init_yaw = self.cfg['agent']['pursuer']['yaw']
        else:
            # Avoid randomly generated pursuer make a collision in the beginning
            # self.init_x = random.random() * self.max_init_distance * random.choice([1, -1])
            # self.init_y = (self.max_init_distance ** 2 - self.init_x ** 2) ** 0.5 * random.choice([1, -1])
            init_yaw = random.random() * pi * random.choice([1, -1])
            init_x, init_y = self._random_spawn(self.pursuer_model)
            self.pursuer_model.set_init_state(init_x, init_y, init_yaw)
        # initial condition for evader
        self.target_init_x = self.cfg['agent']['evader']['x']
        self.target_init_y = self.cfg['agent']['evader']['y']
        self.target_init_yaw = self.cfg['agent']['evader']['yaw']

        # model 2 robots
        self.pursuer_model.set_init_state(self.init_x, self.init_y, self.init_yaw)
        self.evader_model.set_init_state(self.target_init_x, self.target_init_y, self.target_init_yaw)

        self.current_step = 0

        self._set_action([-1.0, 0.0], self.pursuer_model)
        obs = self._get_obs(self.pursuer_model)

        return obs, {}

    def render(self, mode='human'):
        plt.cla()

        pursuer_state = self.pursuer_model.state
        evader_state = self.evader_model.state

        # executed linear&angular velocity + set linear&angular velocity
        plt.text(0, 4, "Exe: " + str(pursuer_state[-4:-2]), fontsize=10)
        plt.text(0, 4.5, "Des: " + str(pursuer_state[-2:]), fontsize=10)

        # draw evader and pursuer
        circle_pursuer = plt.Circle(
            (pursuer_state[RobotStatusIdx.XCoordinateID.value], pursuer_state[RobotStatusIdx.YCoordinateID.value]),
            self.pursuer_radius, color="r")
        circle_evader = plt.Circle(
            (evader_state[RobotStatusIdx.XCoordinateID.value], evader_state[RobotStatusIdx.YCoordinateID.value]),
            self.evader_radius, color="blue")
        plt.gca().add_patch(circle_pursuer)
        plt.gca().add_patch(circle_evader)

        ax = plt.gca()
        # draw obstacles
        if self.ob_list is not None:
            for o in self.ob_list:
                plt.gca().add_patch(plt.Circle(o, self.ob_radius, color="black"))
        # draw boundary wall
        rect_wall = plt.Rectangle((self.boundary_xy[0], self.boundary_xy[1]), self.boundary_wh[0], -self.boundary_wh[1],
                                  fill=False, color="red", linewidth=2)
        plt.gca().add_patch(rect_wall)

        # draw robot movement
        plot_arrow(pursuer_state[0], pursuer_state[1], pursuer_state[2])

        # Render laser
        self.pursuer_model.laser.render(ax)

        plt.xlim([self.boundary_wall[0] - 1, self.boundary_wall[2] + 1])
        plt.ylim([self.boundary_wall[3] - 1, self.boundary_wall[1] + 1])
        plt.grid(True)

        if self.writer is not None:
            self.writer.grab_frame()
        plt.pause(0.05)

    def _get_reward(self, agent):
        goal_distance = np.linalg.norm(agent.state[:RobotStatusIdx.YawAngleID.value] -
                                       self.evader_model.state[:RobotStatusIdx.YawAngleID.value])
        distance_penalty = 1 - goal_distance*0.1

        reward = distance_penalty

        if agent.collide:
            reward += -20
        if agent.catch:
            reward += self.limit_step
        # if agent.truncate:
        #     reward += -2
        return reward-1  # time

    def set_writer(self, writer):
        self.fig = plt.figure(figsize=(7, 7))
        self.writer = writer


def fixed_action_env_test(env):
    # show plot
    for _ in range(100):
        # print("The initial observation is {}".format(obs))
        env.reset()
        a = 0
        while a < 600:
            action = np.array([-0.0, -1.0])
            obs, reward, done, truncated, info = env.step(action)
            env.render()
            if done:
                break
            # print("The new reward is {}".format(reward))
            a += 1


@hydra.main(version_base=None, config_path="../config/environment", config_name="pursuit_env_v2")
def main(cfg: DictConfig):

    env = PursuitEnv2(cfg)

    '''Test Env'''
    fixed_action_env_test(env)


if __name__ == '__main__':
    main()
