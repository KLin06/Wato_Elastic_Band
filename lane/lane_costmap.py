"""
Lane costmap: a straight 3-lane road, 10 m wide and 30 m long, with no obstacles.

Cost scale (same as nav_msgs/OccupancyGrid):
    0         free
    1 .. 99   allowed but discouraged (higher = worse)
    100       lethal: never drive here

Each cell is a possible position of the CAR'S CENTRE (the car is modelled as
a point; see car.py). The car is car.width wide, so it only fits inside its
lane while its centre stays within

    allowed = lane_width / 2 - car.width / 2      (3.33 / 2 - 2 / 2 = 0.67 m)

of the lane's centre line. Further out, part of the car would be over a line.

Each cell's cost is the MAX of three layers:
  1. Lane centring  0 on a lane's centre line, rising to CENTER_COST at the
                    edge of the allowed band:  CENTER_COST * (lateral / allowed)^2
                    Every cell except the one closest to the centre costs at
                    least NEAR_CENTER_COST, so the exact centre is cheapest.
  2. Lines          the car's SIDE getting close to a line. Fades in over
                    LINE_INFLUENCE metres and peaks when the side touches it:
                    LANE_LINE_COST for dashed lines, EDGE_COST for road edges.
  3. Out of lane    OUT_OF_LANE_COST (high, but not lethal) wherever the car's
                    centre is more than `allowed` from its lane's centre, i.e.
                    part of the car is over a dashed lane line. Strongly
                    discouraged, but a planner can still cross if it must.
                    Part of the car over a ROAD EDGE stays lethal (100).

Coordinates: x runs ALONG the road (0 -> 30 m, the driving direction),
y runs ACROSS it (0 = right road edge, 10 = left road edge).
grid[row, col]: row = y, col = x, origin (0, 0) at the bottom-left.
"""
import numpy as np

from car import Car

# ---- Road ------------------------------------------------------------------
ROAD_LENGTH = 50.0          # m, along x
ROAD_WIDTH = 10.0           # m, along y
N_LANES = 3
LANE_WIDTH = ROAD_WIDTH / N_LANES     # 3.33 m
RESOLUTION = 0.1            # m per cell

# ---- Costs -----------------------------------------------------------------
LETHAL = 100
CENTER_COST = 20            # cost at the edge of the allowed band from the centring layer
LANE_LINE_COST = 50         # peak cost when the car's side touches a dashed lane line
EDGE_COST = LETHAL          # peak cost when the car's side touches a road edge
OUT_OF_LANE_COST = 60       # part of the car over a dashed lane line (high, not lethal)
NEAR_CENTER_COST = 1        # minimum cost just off the lane centre, so the centre itself is cheapest
LINE_INFLUENCE = 0.2        # m, distance from the car's side at which a line's cost starts
OBSTACLE_BUFFER_COST = 50   # peak cost when the car's outline is about to touch an obstacle
OBSTACLE_INFLUENCE = 0.3    # m, gap between car and obstacle at which that cost starts


class LaneCostmap:
    def __init__(self, length=ROAD_LENGTH, width=ROAD_WIDTH, n_lanes=N_LANES,
                 resolution=RESOLUTION, car=None, obstacles=None):
        self.length, self.width = length, width
        self.n_lanes = n_lanes
        self.lane_width = width / n_lanes
        self.resolution = resolution

        # The car the costs are built for. Only its width matters here.
        self.car = car if car is not None else Car()
        self.car_width = self.car.width

        # How far the car's centre may move from its lane's centre line
        # before its side reaches a line.
        self.allowed = self.lane_width / 2 - self.car_width / 2
        if self.allowed <= 0:
            raise ValueError(f"car ({self.car_width} m) doesn't fit in a "
                             f"{self.lane_width:.2f} m lane")

        # y position of every lane line, bottom to top: [0, 3.33, 6.67, 10]
        self.boundaries = np.linspace(0, width, n_lanes + 1)
        # y position of every lane's centre line: [1.67, 5, 8.33]
        self.centers = (self.boundaries[:-1] + self.boundaries[1:]) / 2

        self.rows = int(round(width / resolution))     # cells across (y)
        self.cols = int(round(length / resolution))    # cells along (x)
        self.obstacles = list(obstacles) if obstacles else []   # Obstacle objects (obstacle.py)
        self.grid = self._build()
        for obs in self.obstacles:
            self._stamp_obstacle(obs)

    def _build(self):
        # y of every row's cell centre. Every column is identical on a
        # straight road, so compute ONE column of costs and repeat it.
        y = (np.arange(self.rows) + 0.5) * self.resolution
        half_car = self.car_width / 2

        # 1) Lane centring: distance from the car's centre to the nearest lane
        #    centre line, scaled so CENTER_COST is reached at the allowed limit.
        lateral = np.min(np.abs(y[:, None] - self.centers[None, :]), axis=1)
        cost = CENTER_COST * np.clip(lateral / self.allowed, 0, 1) ** 2

        #    Near the centre the curve is so flat that several cells round to 0.
        #    Give every cell except the one(s) closest to a lane centre at least
        #    NEAR_CENTER_COST, so the exact centre is always the cheapest place.
        on_center = lateral <= self.resolution / 2 + 1e-9
        cost = np.where(on_center, 0.0, np.maximum(cost, NEAR_CENTER_COST))

        # 2) Lines: cost grows as the car's SIDE (centre +/- half the car width)
        #    gets within LINE_INFLUENCE of a line.
        for i, by in enumerate(self.boundaries):
            is_edge = i == 0 or i == len(self.boundaries) - 1
            peak = EDGE_COST if is_edge else LANE_LINE_COST
            side_gap = np.abs(y - by) - half_car       # car side -> line distance
            cost = np.maximum(cost, peak * np.clip(1 - side_gap / LINE_INFLUENCE, 0, 1))

        # 3) Out of lane: part of the car over a dashed line is a high cost
        #    (not lethal), so the car stays in its lane unless it really has to cross.
        cost[lateral > self.allowed] = np.maximum(cost[lateral > self.allowed],
                                                  OUT_OF_LANE_COST)

        #    Part of the car over a ROAD EDGE is still lethal: that's off the road.
        off_road = (y < half_car) | (y > self.width - half_car)
        cost[off_road] = LETHAL

        column = np.clip(np.rint(cost), 0, LETHAL).astype(np.int8)
        return np.repeat(column[:, None], self.cols, axis=1)

    def _stamp_obstacle(self, obs):
        """Add one obstacle to the grid (for the car's CENTRE):
          - lethal wherever the car's outline would overlap the obstacle, i.e.
            within obs.length/2 + car.length/2 along x AND
                   obs.width/2  + car.width/2  across y of the obstacle's centre
          - a buffer outside that: OBSTACLE_BUFFER_COST when the car is about
            to touch, fading to 0 over OBSTACLE_INFLUENCE metres of gap
        """
        x = (np.arange(self.cols) + 0.5) * self.resolution
        y = (np.arange(self.rows) + 0.5) * self.resolution
        dx = np.abs(x[None, :] - obs.x) - (obs.length + self.car.length) / 2
        dy = np.abs(y[:, None] - obs.y) - (obs.width + self.car.width) / 2
        gap = np.hypot(np.maximum(dx, 0), np.maximum(dy, 0))     # car-to-obstacle gap (m)

        cost = OBSTACLE_BUFFER_COST * np.clip(1 - gap / OBSTACLE_INFLUENCE, 0, 1)
        cost[(dx <= 0) & (dy <= 0)] = LETHAL                     # overlap
        self.grid = np.maximum(self.grid, np.rint(cost).astype(np.int8))

    # ---- Queries -----------------------------------------------------------
    def cost_at(self, x, y):
        """Cost at world point (x, y) in metres; LETHAL if off the map."""
        row, col = int(y / self.resolution), int(x / self.resolution)
        if 0 <= row < self.rows and 0 <= col < self.cols:
            return int(self.grid[row, col])
        return LETHAL

    def lane_of(self, y):
        """Lane number (1 = rightmost) for a y position, or None if off the road."""
        if not 0 <= y < self.width:
            return None
        return int(y // self.lane_width) + 1

    @property
    def extent(self):
        """(left, right, bottom, top) for plt.imshow(..., origin='lower')."""
        return (0, self.length, 0, self.width)

    # ---- Printing ----------------------------------------------------------
    def __str__(self):
        g = self.grid
        return (f"LaneCostmap: {self.length:g} m long x {self.width:g} m wide, "
                f"{self.n_lanes} lanes of {self.lane_width:.2f} m, "
                f"{self.cols}x{self.rows} cells @ {self.resolution} m\n"
                f"  car {self.car_width:g} m wide: centre must stay within "
                f"+/-{self.allowed:.2f} m of a lane centre\n"
                f"  lane lines at y = {np.round(self.boundaries, 2).tolist()}\n"
                f"  lane centres at y = {np.round(self.centers, 2).tolist()}\n"
                + "".join(f"  {o}\n" for o in self.obstacles) +
                f"  cost range {g.min()}..{g.max()}, "
                f"{(g == 0).mean() * 100:.1f}% free, {(g >= LETHAL).mean() * 100:.1f}% lethal")

    def print_cross_section(self, step=2):
        """Print the cost across the road (one column), top (left edge) first.
        Every column is the same on a straight road, so this shows the whole map."""
        col = self.grid[:, 0]
        for row in range(self.rows - 1, -1, -step):
            y = (row + 0.5) * self.resolution
            lane = self.lane_of(y)
            print(f"y {y:5.2f} m  lane {lane}  cost {col[row]:3d}  " + "#" * (col[row] // 4))


if __name__ == "__main__":
    import matplotlib.pyplot as plt

    cm = LaneCostmap()
    print(cm)
    print("\nCost across the road (every 2nd row, top = left road edge):")
    cm.print_cross_section(step=2)

    # ---- Plot --------------------------------------------------------------
    fig, ax = plt.subplots(figsize=(13, 5))
    im = ax.imshow(cm.grid, cmap="magma_r", origin="lower", extent=cm.extent,
                   vmin=0, vmax=LETHAL, interpolation="nearest")

    # Lane lines: solid black road edges, dashed white lane lines.
    for i, by in enumerate(cm.boundaries):
        is_edge = i == 0 or i == len(cm.boundaries) - 1
        ax.axhline(by, color="black" if is_edge else "white",
                   ls="-" if is_edge else "--", lw=2 if is_edge else 1.5)
        ax.text(cm.length + 0.3, by, "road edge" if is_edge else "lane line (dashed)",
                va="center", fontsize=8)

    # Lane labels and centre lines.
    for k, cy in enumerate(cm.centers):
        ax.axhline(cy, color="tab:cyan", ls=":", lw=1)
        ax.text(1.0, cy, f"Lane {k + 1}", va="center", fontsize=11, weight="bold",
                bbox=dict(facecolor="white", alpha=0.8, lw=0))

    # Driving direction.
    ax.annotate("", xy=(cm.length / 2 + 3, cm.width + 0.6),
                xytext=(cm.length / 2 - 3, cm.width + 0.6),
                arrowprops=dict(arrowstyle="->", lw=1.5), annotation_clip=False)
    ax.text(cm.length / 2, cm.width + 0.9, "driving direction", ha="center", fontsize=9)

    ax.set_xlabel("x: distance along the road (m)")
    ax.set_ylabel("y: across the road (m)")
    ax.set_title(f"Lane costmap: {cm.n_lanes} lanes, {cm.width:g} m x {cm.length:g} m, "
                 f"car {cm.car_width:g} m wide (cost of the car's CENTRE; dotted = lane centres)",
                 pad=40)
    ax.set_aspect("equal")
    fig.colorbar(im, ax=ax, label="cost (0 = free, 100 = lethal)", fraction=0.02, pad=0.12)
    plt.tight_layout()
    plt.savefig("lane_costmap_preview.png", dpi=100)
    plt.show()
