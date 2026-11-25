"""
Car model.

The car is modelled as a POINT: its position (x, y) is the centre of the car,
and that single point is what moves and what gets looked up in the costmap.
The car's size (width, length) isn't used for motion. It's stored so other
code can account for it, e.g. the lane costmap uses the width to decide how
far the point may drift before the car's side crosses a line.

The car also owns its planning:
  - global_planner  (planner.py)      cheapest path along the WHOLE road
  - band            (elastic_band.py) an elastic band over only the next
                                      BAND_LENGTH metres of that path, starting
                                      at the car, which smooths it and pushes
                                      it off obstacles

    global path:  start ─────────────────────────────────────────── end
    band:            car ═══════════ (BAND_LENGTH m ahead)

Typical use:
    car = Car()
    car.set_costmap(LaneCostmap(car=car))
    car.plan()           # global path; puts the car at its start
    car.update_band()    # elastic band over the next 15 m

Coordinates match lane_costmap.py: x along the road (driving direction),
y across it, metres. yaw is the heading in radians (0 = facing +x).
"""
from dataclasses import dataclass, field

import numpy as np

CAR_WIDTH = 2.0      # m
CAR_LENGTH = 5.0     # m
BAND_LENGTH = 15.0   # m of global path the elastic band covers, ahead of the car


@dataclass
class Car:
    x: float = 0.0                  # m, centre of the car
    y: float = 0.0                  # m, centre of the car
    yaw: float = 0.0                # rad, heading (0 = along +x)
    width: float = CAR_WIDTH        # m
    length: float = CAR_LENGTH      # m
    band_length: float = BAND_LENGTH

    # Set by set_costmap(), plan() and update_band(); not constructor arguments.
    costmap: object = field(default=None, init=False, repr=False)
    global_planner: object = field(default=None, init=False, repr=False)
    global_path: np.ndarray = field(default=None, init=False, repr=False)
    band_map: object = field(default=None, init=False, repr=False)
    band: object = field(default=None, init=False, repr=False)
    tail_idx: int = field(default=0, init=False, repr=False)   # last global-path index in the band
    trail: list = field(default_factory=list, init=False, repr=False)  # positions driven so far

    # ======================================================================
    # Geometry
    # ======================================================================
    @property
    def position(self):
        """The point the car is modelled as: (x, y)."""
        return np.array([self.x, self.y])

    def footprint(self):
        """The car's outline as 4 corners (front-left, front-right,
        rear-right, rear-left), rotated by yaw. For drawing only."""
        hl, hw = self.length / 2, self.width / 2
        local = np.array([[hl, hw], [hl, -hw], [-hl, -hw], [-hl, hw]])
        c, s = np.cos(self.yaw), np.sin(self.yaw)
        rot = np.array([[c, -s], [s, c]])
        return local @ rot.T + self.position

    # ======================================================================
    # Planning
    # ======================================================================
    def set_costmap(self, costmap, cost_weight=None):
        """Give the car a map to plan on. Creates the global planner and the
        obstacle view the elastic band uses."""
        # Imported here, not at the top: lane_costmap.py imports Car, so a
        # top-level import of planner.py (which imports lane_costmap) would be circular.
        from elastic_band import BandMap
        from planner import Planner

        self.costmap = costmap
        self.global_planner = (Planner(costmap) if cost_weight is None
                               else Planner(costmap, cost_weight=cost_weight))
        self.band_map = BandMap(costmap)

    def plan(self, start_y=None, end_y=None):
        """Run the global planner from x = 0 at start_y to the end of the road at
        end_y (both default to the centre of lane 2), and put the car at the start.
        Returns the global path as an (N, 2) array."""
        if self.global_planner is None:
            raise RuntimeError("call set_costmap() first")
        self.global_path, self.global_cost = self.global_planner.plan(start_y, end_y)
        self.x, self.y = self.global_path[0]
        d = self.global_path[1] - self.global_path[0]
        self.yaw = float(np.arctan2(d[1], d[0]))
        self.band = None
        self.trail = [self.position.copy()]
        return self.global_path

    def band_window(self):
        """The piece of the global path the band covers: from the car's position
        to `band_length` metres further along the path."""
        if self.global_path is None:
            raise RuntimeError("call plan() first")
        path = self.global_path

        # Start from the global-path point nearest the car.
        i0 = int(np.argmin(np.linalg.norm(path - self.position, axis=1)))
        window = [self.position.copy()]
        dist = 0.0
        self.tail_idx = i0
        for j in range(i0 + 1, len(path)):
            dist += float(np.linalg.norm(path[j] - window[-1]))
            window.append(path[j])
            self.tail_idx = j
            if dist >= self.band_length:
                break
        return np.array(window)

    def update_band(self, max_iter=300, **band_params):
        """Build the elastic band over the next `band_length` metres of the global
        path and let it settle. Both ends stay fixed: the car, and the point
        `band_length` metres ahead on the global path. Returns the band."""
        from elastic_band import ElasticBand

        self.band = ElasticBand(self.band_map, self.band_window(), **band_params)
        self.band.run(max_iter=max_iter)
        return self.band

    # ======================================================================
    # Driving
    # ======================================================================
    @property
    def done(self):
        """True once the band has reached the end of the global path and the
        car is on it."""
        return (self.global_path is not None
                and self.tail_idx == len(self.global_path) - 1
                and np.linalg.norm(self.position - self.global_path[-1]) < 0.05)

    def drive_step(self, distance=0.2, band_steps=3):
        """Drive `distance` metres along the band, then slide the band forward.

          1. ADVANCE  move the car along the band; cut off the band behind it
          2. EXTEND   add global-path points at the far end until the band is
                      `band_length` long again
          3. DEFORM   `band_steps` elastic band updates (the band keeps its
                      shape from the last tick, so a few steps are enough)

        Returns how far the car actually moved (0 if blocked or done).
        """
        if self.band is None:
            self.update_band()
        if self.done:
            return 0.0
        band, path = self.band, self.global_path
        pts = band.points

        # 1) Walk along the band's segments for `distance` metres.
        remaining, pos, i = distance, pts[0].copy(), 0
        while i < len(pts) - 1:
            seg = pts[i + 1] - pos
            seg_len = float(np.linalg.norm(seg))
            if seg_len >= remaining:
                pos = pos + seg / seg_len * remaining
                break
            remaining -= seg_len
            pos = pts[i + 1].copy()
            i += 1
        if self.band_map.rho(pos) < band.rho_min:       # blocked: stay put
            return 0.0

        moved = pos - self.position
        if np.linalg.norm(moved) > 1e-9:
            self.yaw = float(np.arctan2(moved[1], moved[0]))
        self.x, self.y = pos
        self.trail.append(pos.copy())
        new_pts = [pos] + list(pts[i + 1:])

        # 2) Extend the far end along the global path back up to band_length.
        length = float(np.sum(np.linalg.norm(np.diff(new_pts, axis=0), axis=1))) \
            if len(new_pts) > 1 else 0.0
        while length < self.band_length and self.tail_idx < len(path) - 1:
            self.tail_idx += 1
            length += float(np.linalg.norm(path[self.tail_idx] - new_pts[-1]))
            new_pts.append(path[self.tail_idx])

        # 3) Let the band relax a little.
        band.points = np.array(new_pts)
        band.fix_connectivity()
        for _ in range(band_steps):
            band.step()
        return float(np.linalg.norm(moved))

    def __str__(self):
        s = (f"Car at ({self.x:.2f}, {self.y:.2f}) m, yaw {np.degrees(self.yaw):.1f} deg, "
             f"{self.width:g} m wide x {self.length:g} m long")
        if self.global_path is not None:
            s += f"\n  global path: {len(self.global_path)} points, cost {self.global_cost:.2f}"
        if self.band is not None:
            s += f"\n  band ({self.band_length:g} m window): {self.band}"
        return s


if __name__ == "__main__":
    # python car.py                              lane 2 -> lane 2
    # python car.py --start-y 1.67 --end-y 8.33  lane change(s)
    import argparse

    import matplotlib.pyplot as plt
    from matplotlib.patches import Circle

    from lane_costmap import LETHAL, LaneCostmap

    ap = argparse.ArgumentParser(description="Car with a global planner and a 15 m elastic band.")
    ap.add_argument("--start-y", type=float, default=None, help="start y (default: lane 2 centre)")
    ap.add_argument("--end-y", type=float, default=None, help="end y (default: lane 2 centre)")
    args = ap.parse_args()

    car = Car()
    cm = LaneCostmap(car=car)
    car.set_costmap(cm)
    car.plan(args.start_y, args.end_y)
    car.update_band()
    print(car)

    # ---- Plot --------------------------------------------------------------
    fig, ax = plt.subplots(figsize=(13, 4.5))
    im = ax.imshow(cm.grid, cmap="magma_r", origin="lower", extent=cm.extent,
                   vmin=0, vmax=LETHAL, interpolation="nearest")
    for i, by in enumerate(cm.boundaries):
        is_edge = i == 0 or i == len(cm.boundaries) - 1
        ax.axhline(by, color="black" if is_edge else "white",
                   ls="-" if is_edge else "--", lw=2 if is_edge else 1.5)

    gp = car.global_path
    ax.plot(gp[:, 0], gp[:, 1], "--", color="gray", lw=1.5, label="global path")
    b = car.band
    for p, r in zip(b.points, b.radii()):
        ax.add_patch(Circle(p, r, color="tab:blue", alpha=0.15))
    ax.plot(b.points[:, 0], b.points[:, 1], "-o", color="tab:blue", ms=3, lw=2,
            label=f"elastic band ({car.band_length:g} m)")
    outline = car.footprint()
    ax.fill(outline[:, 0], outline[:, 1], fc="tab:green", alpha=0.5, ec="tab:green",
            label="car")

    ax.set_xlabel("x: distance along the road (m)")
    ax.set_ylabel("y: across the road (m)")
    ax.set_title("Car: global path (whole road) + elastic band (next 15 m)")
    ax.set_aspect("equal")
    ax.legend(loc="upper right", fontsize=8)
    fig.colorbar(im, ax=ax, label="cost", fraction=0.02)
    plt.tight_layout()
    plt.savefig("car_preview.png", dpi=100)
    plt.show()
