

class Entity(object):
    def __init__(self, fixed: bool, visual: bool, collision: bool):
        self._fixed = fixed
        self._visual = visual
        self._collision = collision


class LineEntity(Entity):
    def __init__(self, start_point, vector, length,
                 fixed: bool, visual: bool, collision: bool):
        super().__init__(fixed, visual, collision)

        self._start_point = start_point
        self._vector = vector
        self._length = length


class CircleEntity(Entity):
    def __init__(self, xy,
                 fixed: bool, visual: bool, collision: bool):
        super().__init__(fixed, visual, collision)

        self._xy = xy
