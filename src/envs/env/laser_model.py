import math

import numpy as np
import matplotlib.pyplot as plt


class Laser:
    def __init__(self, laser_angle_max, laser_angle_min, laser_angle_resolution, laser_distance_max,
                 map_resolution=0.3):

        self.laser_angle_max = laser_angle_max
        self.laser_angle_min = laser_angle_min
        self.laser_angle_resolution = laser_angle_resolution
        self.laser_distance_max = laser_distance_max
        self.map_resolution = map_resolution

        assert self.laser_angle_min < self.laser_angle_max, "The max angle should be larger than the min angle!"

        self.angle_list = np.arange(self.laser_angle_min, self.laser_angle_max,
                                    self.laser_angle_resolution)  # in radians

        # Assume square shape first
        self.map_size = np.arange(-laser_distance_max, laser_distance_max, map_resolution).shape[
            0]  # number of points in one direction
        self.raster_map = np.zeros((self.map_size + 1, self.map_size + 1))
        #
        self.points = np.zeros((self.angle_list.shape[0], 2))
        self.ray_vectors = np.zeros_like(self.points)
        self.ray_vectors[:, 0] = np.cos(self.angle_list)
        self.ray_vectors[:, 1] = np.sin(self.angle_list)

        self.transform = np.zeros((3, 3))
        self.x, self.y = 0, 0
        self.theta = 0

    def laser_points_update(self, ob_list: np.ndarray, ob_radius, transform, line_ob_list):
        # Update transform
        self.transform = transform
        self.x, self.y = transform[0, 2], transform[1, 2]

        ob_list_transformed = np.matmul(np.linalg.inv(self.transform),
                                        np.column_stack([ob_list, np.ones((ob_list.shape[0], 1))]).T)[:2, :].T
        ob_list_filtered = self._ob_list_filter(ob_list_transformed, ob_radius)
        points = self.ray_vectors * self.laser_distance_max
        # Mark the points at the max range as False while the points crossing with obstacles as True
        max_range_mask = np.zeros(points.shape[0], dtype=bool)
        for i in range(ob_list_filtered.shape[0]):
            cross_point, nan_mask = self._ray_circle_cross(self.ray_vectors, ob_list_filtered[i, :], ob_radius)
            points[~nan_mask, :] = cross_point[~nan_mask, :]
            max_range_mask[~nan_mask] = True

        points_ = []
        line_ob_list_filtered = self._line_ob_list_filter(line_ob_list)
        if len(line_ob_list_filtered) > 0:
            for line_ob in line_ob_list_filtered:
                points_transformed = line_ob.transformed(self.transform)
                v0 = -points_transformed[0, :]
                v1 = np.array([points_transformed[1, 0] - points_transformed[0, 0],
                               points_transformed[1, 1] - points_transformed[0, 1]])
                v1_length = np.linalg.norm(v1)
                v1 = v1 / v1_length
                l_o = np.dot(v1, v0)
                v2 = l_o * v1
                v_p = v2 - v0
                l_delta = math.sqrt(self.laser_distance_max ** 2 - np.linalg.norm(v_p, ord=2)**2)
                points_range = np.array([v1 * max(0, l_o - l_delta), v1 * min(v1_length, l_o + l_delta)]) - v0
                points_range += transform[:2, 2]
                points_.append(points_range)
        self.points = np.matmul(self.transform, np.column_stack([points, np.ones((points.shape[0], 1))]).T)[:2, :].T
        self._raster_map_update(max_range_mask, points_)

    def render(self, ax, point_size=2):
        ax.add_patch(plt.Circle((self.x, self.y), self.laser_distance_max, color="yellow", fill=False))
        ax.scatter(self.points[:, 0], self.points[:, 1], s=point_size)
        ax.imshow(self.raster_map, extent=(self.x - self.laser_distance_max, self.x + self.laser_distance_max,
                                           self.y - self.laser_distance_max, self.y + self.laser_distance_max))

    def _ob_list_filter(self, ob_list: np.ndarray, ob_radius) -> np.ndarray:
        rbt2ob_distance = np.linalg.norm(ob_list, axis=1)

        return ob_list[rbt2ob_distance <= (self.laser_distance_max + ob_radius), :]

    # Assume the coordinates are 0,0
    def _ray_circle_cross(self, ray_vector, circle_center, circle_radius):
        projection_connection2ray_vector = np.dot(ray_vector, circle_center)
        projection_connection2ray_vector[projection_connection2ray_vector < 0] = np.nan
        circle_center2ray_vector = np.sqrt(np.linalg.norm(circle_center) ** 2 - projection_connection2ray_vector ** 2)
        ray_length = projection_connection2ray_vector - np.sqrt(circle_radius ** 2 - circle_center2ray_vector ** 2)
        mask = np.isnan(ray_length)
        return ray_vector * np.expand_dims(ray_length, axis=1), mask

    def _raster_map_update(self, max_range_mask, line_intersection_points):
        self.raster_map[:] = 0
        valid_points = self.points[max_range_mask, :]
        if valid_points.shape[0] > 0:
            self._coordinates_to_array(valid_points)
        intersection_point_list = []
        if len(line_intersection_points) > 0:
            for p in line_intersection_points:
                vector = p[1, :] - p[0, :]
                length = np.linalg.norm(vector)
                vector /= length
                length_range = np.append(np.arange(0, length, self.map_resolution), length)
                intersection_point_list.append(p[0, :] + [vector*length_range[i] for i in range(length_range.shape[0])])
            self._coordinates_to_array(np.concatenate(intersection_point_list))

    def _coordinates_to_array(self, points):
        """
        Convert coordinates array that indicates occupied points
        :param points:
        :return:
        """
        points[:, 0] -= self.x
        points[:, 1] -= self.y
        # Resample to fill up map array
        points[:, 0] -= -self.laser_distance_max
        points[:, 1] = -(points[:, 1] - self.laser_distance_max)
        points = (points / self.map_resolution).astype(int)
        for i in range(points.shape[0]):
            self.raster_map[points[i, 1], points[i, 0]] = 1

    def _line_ob_list_filter(self, line_ob_list):
        """
        Fast filter the lines with no intersection
        :param line_ob_list:
        :return:
        """
        line_ob_list_filtered = []
        for line_ob in line_ob_list:
            distance = line_ob.get_distance((self.x, self.y))
            if distance <= self.laser_distance_max:
                line_ob_list_filtered.append(line_ob)
        return line_ob_list_filtered


def main():
    plt.clf()
    plt.xlim((-5, 5))
    plt.ylim((-5, 5))

    ax = plt.gca()
    ax.set_aspect(1)

    ob_list = np.array([[0.0, 1.0], [0.5, 0.0], [0.5, 0.5], [4.0, 4.0]])
    ob_radius = 0.2

    ax.scatter(ob_list[:, 0], ob_list[:, 1], color="black", s=ob_radius)
    for i in range(len(ob_list)):
        ax.add_patch(plt.Circle(ob_list[i, :], ob_radius, color="black"))

    transform = np.array([
        [np.cos(np.pi / 4), -np.sin(np.pi / 4), -1],
        [np.sin(np.pi / 4), np.cos(np.pi / 4), 1],
        [0, 0, 1]
    ])
    # transform = np.eye(3)

    laser = Laser(np.pi, -np.pi, np.pi / 180, 3.0, map_resolution=0.3)
    laser.laser_points_update(ob_list, ob_radius, transform)
    laser.render(ax)

    plt.show()


if __name__ == "__main__":
    main()
