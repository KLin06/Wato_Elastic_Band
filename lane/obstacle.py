"""
Obstacle model: a small axis-aligned box on the road (debris, a cone, ...).

Default: 1.0 m long x 0.5 m wide, in lane 2 at x = 25 m, against the RIGHT
side of the lane. Lane 2 spans y = 3.33 .. 6.67 m, so the obstacle covers
y = 3.38 .. 3.88 m and leaves 6.67 - 3.88 = 2.79 m of lane free. The car is
2 m wide, so it still fits inside lane 2, but has to shift left to get past.

(Centred in the lane, the obstacle would leave only 1.42 m on each side, and
the car couldn't pass without leaving the lane.)

How it enters the costmap: costs are for the car's CENTRE (the car is a
point), so the costmap makes lethal every centre position where the car's
outline would overlap the obstacle, i.e. the obstacle grown by half the car's
length and width. See LaneCostmap._stamp_obstacle().
"""
from dataclasses import dataclass

import numpy as np

OBSTACLE_LENGTH = 1.0   # m, along x
OBSTACLE_WIDTH = 0.5    # m, across y


@dataclass
class Obstacle:
    x: float = 25.0                 # m, centre
    y: float = 3.33 + 0.30          # m, centre: lane 2's right line + 0.3 m
    length: float = OBSTACLE_LENGTH
    width: float = OBSTACLE_WIDTH

    def footprint(self):
        """4 corners of the obstacle, for drawing."""
        hl, hw = self.length / 2, self.width / 2
        return np.array([[self.x + hl, self.y + hw], [self.x + hl, self.y - hw],
                         [self.x - hl, self.y - hw], [self.x - hl, self.y + hw]])

    def __str__(self):
        return (f"Obstacle at ({self.x:.2f}, {self.y:.2f}) m, "
                f"{self.length:g} m long x {self.width:g} m wide")
