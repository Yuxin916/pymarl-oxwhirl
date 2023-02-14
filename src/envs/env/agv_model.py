import numpy as np
from math import cos, sin, pi
from .utils import normalize_angle
from enum import Enum

from .laser_model import Laser


class RobotStatusIdx(Enum):
    XCoordinateID = 0  # Coordinates
    YCoordinateID = 1
    YawAngleID = 2  # Yaw Angle
    LinearVelocityExe = 3  # Executed Velocity
    AngularVelocityExe = 4
    LinearVelocityDes = 5  # Desired Velocity
    AngularVelocityDes = 6


class AGVModel:
    def __init__(self, max_v, min_v, max_w, max_acc_v, max_acc_w,
                 init_x, init_y, init_yaw, init_v, init_w, robot_radius,
                 laser_on, laser_max_angle=0.0, laser_min_angle=0.0, laser_angle_resolution=0.0, laser_range=0.0):

        # state = [x-coordinate, y-coordinate, angle, linear velocity (actual), set angular velocity (actual),
        #          linear velocity (set), angular velocity (set)]
        self.state = np.zeros(7)
        self.state[RobotStatusIdx.XCoordinateID.value] = init_x
        self.state[RobotStatusIdx.YCoordinateID.value] = init_y
        self.state[RobotStatusIdx.YawAngleID.value] = init_yaw
        self.state[RobotStatusIdx.LinearVelocityExe.value] = init_v
        self.state[RobotStatusIdx.AngularVelocityExe.value] = init_w
        self.state[RobotStatusIdx.LinearVelocityDes.value] = 0.0
        self.state[RobotStatusIdx.AngularVelocityDes.value] = 0.0

        # physical constrain
        self.max_v = max_v
        self.min_v = min_v
        self.max_w = max_w
        self.min_w = -max_w
        self.max_acc_v = max_acc_v
        self.max_acc_w = max_acc_w

        # Collision Volume
        self.robot_radius = robot_radius

        # Laser parameters
        self.laser_min_angle = laser_min_angle
        self.laser_max_angle = laser_max_angle
        self.laser_angle_resolution = laser_angle_resolution
        self.laser_range = laser_range

        # Whether load laser computation
        self.laser_on = laser_on
        if laser_on:
            self.laser = Laser(self.laser_max_angle, self.laser_min_angle, laser_angle_resolution, laser_range)
            self.laser_map_size = self.laser.raster_map.shape[0] * self.laser.raster_map.shape[1]
        else:
            self.laser_map_size = 0
        # self.evasion_route = np.array([
        #     [4.0, 3.5],
        #     [6.0, 2.7],
        #     [7.55, 6.8],
        #     [9.57, 11.2],
        #     [9.3, 16.5],
        #     [5.44, 16.0],
        #     [5.44, 13.23],
        #     [5.44, 9.24],
        #     [0.0, 4.5],
        #     [-2.2, -1.1],
        #     [0.0, -5.0]
        # ])

    def constrain_input_velocity(self, state, dt):
        v_pre_max = min(state[RobotStatusIdx.LinearVelocityExe.value] + self.max_acc_v * dt, self.max_v)
        v_pre_min = max(state[RobotStatusIdx.LinearVelocityExe.value] - self.max_acc_v * dt, self.min_v)
        w_pre_max = min(state[RobotStatusIdx.AngularVelocityExe.value] + self.max_acc_w * dt, self.max_w)
        w_pre_min = max(state[RobotStatusIdx.AngularVelocityExe.value] - self.max_acc_w * dt, -self.max_w)

        return [v_pre_min, v_pre_max, w_pre_min, w_pre_max]

    def motion(self, input_u, dt, obstacles: tuple, line_ob_list) -> None:
        """
        :param input_u: velocity. [0] - linear speed, [1] - angular speed
        :param dt: travel time
        :param obstacles: Tuple: (ob_list, ob_radius), for laser update only
        :return: new state
        """
        # constrain input velocity
        constrain = self.constrain_input_velocity(self.state, dt)
        u = np.array(input_u)
        u[0] = max(constrain[0], u[0])
        u[0] = min(constrain[1], u[0])

        u[1] = max(constrain[2], u[1])
        u[1] = min(constrain[3], u[1])

        # motion model, euler
        self.state[RobotStatusIdx.XCoordinateID.value] += u[RobotStatusIdx.XCoordinateID.value] * cos(
            self.state[RobotStatusIdx.YawAngleID.value]) * dt
        self.state[RobotStatusIdx.YCoordinateID.value] += u[RobotStatusIdx.XCoordinateID.value] * sin(
            self.state[RobotStatusIdx.YawAngleID.value]) * dt

        self.state[RobotStatusIdx.YawAngleID.value] += u[1] * dt
        self.state[RobotStatusIdx.YawAngleID.value] = normalize_angle(self.state[RobotStatusIdx.YawAngleID.value])

        self.state[RobotStatusIdx.LinearVelocityExe.value] = u[0]  # actual
        self.state[RobotStatusIdx.AngularVelocityExe.value] = u[1]  # actual
        self.state[RobotStatusIdx.LinearVelocityDes.value] = input_u[0]  # set
        self.state[RobotStatusIdx.AngularVelocityDes.value] = input_u[1]  # set

        if self.laser_on:
            self.laser.laser_points_update(np.array(obstacles[0]), obstacles[1], self.get_transform(), line_ob_list)

    def rot_to_angle(self, theta):
        norm_theta = normalize_angle(theta - self.state[2])
        dead_zone = pi / 8.0
        factor = self.max_w / dead_zone
        # angular_velocity = norm_theta * 7
        if norm_theta > dead_zone:
            angular_velocity = self.max_w
        elif norm_theta < -dead_zone:
            angular_velocity = -self.max_w
        else:
            angular_velocity = norm_theta * factor
        return angular_velocity

    def set_init_state(self, init_x, init_y, init_yaw, init_v=0.0, init_w=0.0):
        self.state[RobotStatusIdx.XCoordinateID.value] = init_x
        self.state[RobotStatusIdx.YCoordinateID.value] = init_y
        self.state[RobotStatusIdx.YawAngleID.value] = init_yaw
        self.state[RobotStatusIdx.LinearVelocityExe.value] = init_v
        self.state[RobotStatusIdx.AngularVelocityExe.value] = init_w

    def get_transform(self):
        """
        :return: transition matrix
        """
        theta = self.state[RobotStatusIdx.YawAngleID.value]  # yaw angle of pursuer
        position = self.state[:RobotStatusIdx.YawAngleID.value]  # x-y coordinates of pursuer
        transform = np.array([
            [cos(theta), -sin(theta), position[0]],
            [sin(theta), cos(theta), position[1]],
            [0, 0, 1]
        ])
        return transform


class AGVAgent(AGVModel):
    def __init__(self, idx, max_v, min_v, max_w, max_acc_v, max_acc_w, init_x, init_y, init_yaw, init_v, init_w,
                 robot_radius, laser_on, laser_max_angle=0.0, laser_min_angle=0.0, laser_angle_resolution=0.0, laser_range=0.0):
        super().__init__(max_v, min_v, max_w, max_acc_v, max_acc_w, init_x, init_y, init_yaw, init_v, init_w,
                         robot_radius, laser_on, laser_max_angle, laser_min_angle, laser_angle_resolution, laser_range)
        self.idx = idx
        self.collide = False
        self.catch = False
        self.caught = False
        self.truncate = False
