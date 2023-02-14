import numpy as np
from math import sqrt, pi, atan2


class Point:
    def __init__(self, x, y):
        self.x = x
        self.y = y


class Line:
    def __init__(self, p1, p2):
        self.p1 = Point(*p1)
        self.p2 = Point(*p2)
        self.vector = np.array([self.p2.x-self.p1.x, self.p2.y-self.p1.y], dtype=float)
        self.length = np.linalg.norm(self.vector)
        self.vector /= self.length

    def get_distance(self, xy):
        # Get the minimized distance from point xy to the line
        v1 = (xy[0] - self.p1.x, xy[1] - self.p1.y)
        v2 = (xy[0] - self.p2.x, xy[1] - self.p2.y)
        s1 = v1[0] * self.vector[0] + v1[1] * self.vector[1]
        s2 = v2[0] * -self.vector[0] + v2[1] * -self.vector[1]
        if s1 >= 0 and s2 >= 0:
            # Both acute angle means the minimized distance is orthogonal distance
            return sqrt(v1[0] ** 2 + v1[1] ** 2 - s1 ** 2)
        else:
            # Else select the minimized distance between two points
            return min(sqrt(v1[0] ** 2 + v1[1] ** 2), sqrt(v2[0] ** 2 + v2[1] ** 2))

    def transformed(self, transform):
        t = np.repeat([transform[:2, 2]], 2, axis=0)
        points = np.array([[self.p1.x, self.p1.y],
                           [self.p2.x, self.p2.y]])
        return points - t
