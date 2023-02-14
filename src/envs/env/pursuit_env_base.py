import random

from gym.utils import seeding
import gym
from .agv_model import RobotStatusIdx
from .base.utils import Line
from utils import *


class PursuitEnvBase(gym.Env):
    def __init__(self, cfg):
        self.cfg = cfg

        # boundary condition
        self.max_v = cfg['constraints']['max_v']
        self.min_v = cfg['constraints']['min_v']
        self.max_v_acc = cfg['constraints']['max_v_acc']
        self.max_w = cfg['constraints']['max_w']
        self.min_w = -self.max_w
        self.max_w_acc = cfg['constraints']['max_w_acc']
        self.boundary_wall = cfg['constraints']['boundary_wall']
        self.boundary_xy = [self.boundary_wall[0], self.boundary_wall[1]]
        self.boundary_wh = [self.boundary_wall[2] - self.boundary_wall[0],
                            self.boundary_wall[1] - self.boundary_wall[3]]
        self.line_ob_list = [Line((self.boundary_wall[0], self.boundary_wall[1]), (self.boundary_wall[2], self.boundary_wall[1])),
                             Line((self.boundary_wall[2], self.boundary_wall[1]), (self.boundary_wall[2], self.boundary_wall[3])),
                             Line((self.boundary_wall[2], self.boundary_wall[3]), (self.boundary_wall[0], self.boundary_wall[3])),
                             Line((self.boundary_wall[0], self.boundary_wall[3]), (self.boundary_wall[0], self.boundary_wall[1]))]
        # initial condition for pursuer
        self.init_x = cfg['agent']['pursuer']['x']
        self.init_y = cfg['agent']['pursuer']['y']
        self.init_yaw = cfg['agent']['pursuer']['yaw']
        self.pursuer_radius = cfg['agent']['pursuer']['radius']
        self.pursuer_fixed = cfg['agent']['pursuer']['fixed']
        # initial condition for evader
        self.target_init_x = cfg['agent']['evader']['x']
        self.target_init_y = cfg['agent']['evader']['y']
        self.target_init_yaw = cfg['agent']['evader']['yaw']
        self.evader_radius = cfg['agent']['evader']['radius']
        self.evader_fixed = cfg['agent']['evader']['fixed']
        # obstacle
        self.ob_radius = cfg['obstacle']['radius']
        self.ob_list = cfg['obstacle']['o_coordinates']
        # time
        self.max_init_distance = cfg['max_init_distance']
        self.dt = cfg['time_step']
        # laser
        self.laser_angle_max = cfg['laser']['max_laser_angle']
        self.laser_angle_min = cfg['laser']['min_laser_angle']
        self.laser_angle_step = cfg['laser']['step_laser_angle']
        self.laser_range = cfg['laser']['laser_range']
        self.basic_sample_reso = cfg['laser']['laser_sample_resolution']
        self.sample_reso_scale = cfg['laser']['laser_sample_resolution_scale']

        self.limit_step = cfg['train']['episode_timelimit']

        # seed
        self.np_RNG = None
        self.seed()

        # time limit
        self.current_step = 0

        # record both pursuer's policy(action) and trajectory for visualization
        self.pursuit_strategy = None
        self.evader_strategy = None
        self.traj_pursuer = None
        self.traj_evader = None

        # self.target_u = np.random.random_sample(size=2)
        self.reward_list = []

        self.evader_model = None

    def seed(self, seed=None):
        self.np_RNG, seed_ = seeding.np_random(seed)
        print('The seed: ', seed_)
        return [seed_]

    def step(self, action_n: list):
        raise NotImplementedError

    def _set_action(self, action, agent):
        """
        Applies the given action to the agent with target idx.
        action: [linear vel, angular vel]
        """
        # Action de-normalize
        action[0] = agent.min_v + (action[0] + 1) * (agent.max_v - agent.min_v) / 2.0
        action[1] = agent.min_w + (action[1] + 1) * (agent.max_w - agent.min_w) / 2.0

        # Input action to pursuer and evader model
        agent.motion(action, self.dt, (self.ob_list, self.ob_radius), self.line_ob_list)

    def _get_obs(self, agent):
        """
        Returns the observation of target agent.
        """
        percept_map = agent.laser.raster_map
        percept_map = percept_map.flatten().tolist()

        # Get both states information of pursuer and evader
        # pursuer & evader state: [x, y, yaw, v_exe, w_exe, v_des, w_des]
        pursuer_state = agent.state
        evader_state = self.evader_model.state

        # evader's reference angle w.r.t pursuer's coordinate
        goal_angle = atan2(evader_state[1] - pursuer_state[RobotStatusIdx.YCoordinateID.value],
                           evader_state[0] - pursuer_state[RobotStatusIdx.XCoordinateID.value]) - pursuer_state[
                         RobotStatusIdx.YawAngleID.value]
        goal_angle = normalize_angle(goal_angle)

        # evader's sqrt distance w.r.t pursuer's coordinate
        goal_distance = np.linalg.norm(evader_state[:RobotStatusIdx.YawAngleID.value] -
                                       pursuer_state[:RobotStatusIdx.YawAngleID.value])

        observation = np.array(percept_map + [goal_angle, goal_distance,
                                              pursuer_state[RobotStatusIdx.LinearVelocityExe.value],
                                              pursuer_state[RobotStatusIdx.AngularVelocityExe.value]])

        return observation

    def _get_info(self, agent):
        if agent.collide:
            info = "Collision"
        elif agent.catch:
            info = "Caught"
        elif agent.truncate:
            info = "Time_limit_reached"
        else:
            info = ""
        return info

    def _get_reward(self, agent):
        raise NotImplementedError

    def _get_done(self, agent):  # TODO catch collision ...
        if self.current_step >= self.limit_step or \
                agent.collide or agent.catch:
            return True
        else:
            return False

    def reset(self, **kwargs):
        return NotImplementedError

    def render(self, mode='human'):
        return NotImplementedError

    def obstacle_collision_check(self, agent):
        if self.ob_list is None:
            return False
        for ob in self.ob_list:
            if np.linalg.norm(agent.state[:2] - ob) <= agent.robot_radius + self.ob_radius:
                return True
        return False

    def boundary_collision_check(self, agent):
        inner_boundary = [self.boundary_wall[0] + agent.robot_radius,
                          self.boundary_wall[1] - agent.robot_radius,
                          self.boundary_wall[2] - agent.robot_radius,
                          self.boundary_wall[3] + agent.robot_radius]
        if agent.state[RobotStatusIdx.XCoordinateID.value] < inner_boundary[0] or \
                agent.state[RobotStatusIdx.YCoordinateID.value] > inner_boundary[1] or \
                agent.state[RobotStatusIdx.XCoordinateID.value] > inner_boundary[2] or \
                agent.state[RobotStatusIdx.YCoordinateID.value] < inner_boundary[3]:
            return True
        else:
            return False

    def catch_check(self, agent):
        pursuer_state = agent.state
        evader_state = self.evader_model.state
        distance = np.linalg.norm(evader_state[:RobotStatusIdx.YawAngleID.value] -
                                  pursuer_state[:RobotStatusIdx.YawAngleID.value])

        if distance < self.pursuer_radius + self.evader_radius:
            return True
        else:
            return False

    def _random_spawn(self, agent, gap=0.0):
        valid = False
        while not valid:
            init_x = self.boundary_xy[0] + random.random() * self.boundary_wh[0]
            init_y = self.boundary_xy[1] - random.random() * self.boundary_wh[1]
            for ob in self.ob_list:
                if np.linalg.norm(np.array(
                        [init_x, init_y]) - ob) <= agent.robot_radius + self.ob_radius + gap:
                    continue
            valid = True
        return init_x, init_y


if __name__ == "__main__":
    ray_line = Line((-1, -1), (1, -1))
    print(ray_line.get_distance((0, 0)))
