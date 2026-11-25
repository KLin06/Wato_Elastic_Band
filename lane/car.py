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
STOP_BACKOFF = 1.0   # m, the car stops this far before a point the corridor can't clear
DETOUR_OFFSET = 0.1  # m off the reference that counts as a detour (status)
LANE_SETTLE_MARGIN = 0.3  # m the car's sides must be inside the new lane before it becomes the boundary


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
    reference: np.ndarray = field(default=None, init=False, repr=False)  # global path shifted clear of obstacles
    band_map: object = field(default=None, init=False, repr=False)
    band: object = field(default=None, init=False, repr=False)
    tail_idx: int = field(default=0, init=False, repr=False)   # last global-path index in the band
    trail: list = field(default_factory=list, init=False, repr=False)  # positions driven so far
    # Allowed lanes as (x_from, x_to, (lowest_lane, highest_lane)) pieces; None = whole road.
    corridor: list = field(default=None, init=False, repr=False)
    lane_change_target: int = field(default=None, init=False, repr=False)  # lane being changed into
    blocked_idx: int = field(default=None, init=False, repr=False)  # first reference point no shift can clear

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
        self.band_map = BandMap(costmap)       # replaced once a corridor is set

    def plan(self, start_y=None, end_y=None):
        """Run the global planner from x = 0 at start_y to the end of the road at
        end_y (both default to the centre of lane 2), and put the car at the start.
        Returns the global path as an (N, 2) array."""
        if self.global_planner is None:
            raise RuntimeError("call set_costmap() first")
        self.global_path, self.global_cost = self.global_planner.plan(start_y, end_y)
        self._set_corridor(None)                    # search planner: whole road allowed
        return self._start_on_path()

    def plan_reference(self, lane=2, target_lane=None, change_x=15.0, blend_length=20.0):
        """Use a reference line (reference.py) instead of the search planner:
        the centreline of `lane`, optionally blending into `target_lane`
        starting at `change_x` over `blend_length` metres. The band is pulled
        toward this line. Puts the car at its start. Returns the path.

        Also sets the allowed-lane CORRIDOR (what the decision layer authorises):
          staying         : `lane` along the whole road
          changing lanes  : `lane` before change_x; every lane from `lane` to
                            `target_lane` from change_x on. Once the car and its
                            whole band are inside `target_lane`, the corridor
                            narrows to `target_lane` from the car onward, so the
                            new lane's lines become the hard boundaries
                            (see _finish_lane_change()).
        """
        from reference import lane_reference

        if self.costmap is None:
            raise RuntimeError("call set_costmap() first")
        self.global_path = lane_reference(self.costmap, lane, target_lane, change_x, blend_length)
        self.global_cost = float("nan")

        L = self.costmap.length
        if target_lane is None or target_lane == lane:
            corridor = [(0.0, L, (lane, lane))]
            self.lane_change_target = None
        else:
            lo, hi = min(lane, target_lane), max(lane, target_lane)
            corridor = [(0.0, change_x, (lane, lane)),
                        (change_x, L, (lo, hi))]
            self.lane_change_target = target_lane
        self._set_corridor(corridor)
        return self._start_on_path()

    def _in_lane(self, p, lane, margin=0.0):
        """True if a car centred at p is entirely inside `lane`, with its sides
        at least `margin` metres from the lane's lines."""
        centre = self.costmap.centers[lane - 1]
        allowed = self.costmap.lane_width / 2 - self.width / 2 - margin
        return abs(p[1] - centre) <= allowed + 1e-9

    def _finish_lane_change(self):
        """Once the car AND every point of its band are inside the target lane
        (sides at least LANE_SETTLE_MARGIN from its lines), narrow the corridor
        to the target lane from the car's position onward.

        Waiting for the whole band (not just the car) matters: if the corridor
        narrowed while the band ahead was still crossing, those band points
        would suddenly be inside a wall. The margin stops the new wall being
        placed right against the car. Returns True if the corridor changed."""
        target = self.lane_change_target
        if target is None or self.band is None:
            return False
        if not all(self._in_lane(p, target, LANE_SETTLE_MARGIN) for p in self.band.points):
            return False

        # Keep the corridor behind the car as it was; target lane only from here on.
        x = self.x
        corridor = [(x0, min(x1, x), lanes) for x0, x1, lanes in self.corridor if x0 < x]
        corridor.append((x, self.costmap.length, (target, target)))
        self._set_corridor(corridor)
        self.lane_change_target = None

        # Rebuild what depends on the corridor: the obstacle-clear reference
        # (shifts must now stay inside the target lane) and the band's map.
        self.reference = self._clear_reference(self.global_path)
        self.band.map = self.band_map
        self.band.set_reference(self.reference)
        return True

    # ======================================================================
    # Corridor (allowed lanes)
    # ======================================================================
    def _set_corridor(self, corridor):
        """Store the corridor and rebuild the band's obstacle map with it."""
        from elastic_band import BandMap

        self.corridor = corridor
        self.band_map = BandMap(self.costmap, inside=self.corridor_mask())

    def corridor_mask(self):
        """Bool grid (rows, cols): True where the car's CENTRE keeps the whole
        car inside the allowed lanes, i.e. between the lowest allowed lane's
        right line + half the car width and the highest allowed lane's left
        line - half the car width. None if the whole road is allowed."""
        if self.corridor is None:
            return None
        cm = self.costmap
        x = (np.arange(cm.cols) + 0.5) * cm.resolution
        y = (np.arange(cm.rows) + 0.5) * cm.resolution
        inside = np.zeros((cm.rows, cm.cols), dtype=bool)
        half_car = self.width / 2
        for x0, x1, (lo, hi) in self.corridor:
            cols = (x >= x0) & (x < x1) if x1 < cm.length else (x >= x0)
            y_min = cm.boundaries[lo - 1] + half_car
            y_max = cm.boundaries[hi] - half_car
            rows = (y >= y_min) & (y <= y_max)
            inside |= rows[:, None] & cols[None, :]
        return inside

    def _start_on_path(self):
        """Put the car at the start of the global path, facing along it, and
        build the obstacle-free reference the band uses."""
        self.reference = self._clear_reference(self.global_path)
        self.x, self.y = self.global_path[0]
        d = self.global_path[1] - self.global_path[0]
        self.yaw = float(np.arctan2(d[1], d[0]))
        self.band = None
        self.trail = [self.position.copy()]
        return self.global_path

    def _clear_shift(self, p, sign=None, max_shift=3.0, rho_min=0.05):
        """Smallest sideways (y) shift that moves `p` clear of obstacles, or None.
        sign=None tries both sides (left first on a tie); +1 / -1 forces a side."""
        if self.band_map.rho(p) >= rho_min:
            return 0.0
        res = self.costmap.resolution
        for k in range(1, int(max_shift / res) + 1):
            for s in ((1, -1) if sign is None else (sign,)):
                if self.band_map.rho(np.array([p[0], p[1] + s * k * res])) >= rho_min:
                    return s * k * res
        return None

    def _clear_reference(self, path, margin=5.0):
        """The global path, shifted sideways around obstacles. Used both to seed
        new band points and as what the band is pulled toward.

        Why: the reference line (lane centre) can run straight through an
        obstacle. The band can't start inside one (its bubbles would have zero
        radius), and pulling it toward a point inside one makes the pull fight
        the obstacle repulsion until the band breaks. So every point within
        `margin` metres before or after a blocked stretch gets that stretch's
        shift: the sideways jumps happen in free space and the band smooths them.

        Prototype simplification: shifts along y (the road is straight along x).
        A real implementation would shift along the path's normal."""
        path = np.asarray(path, float)
        shifts = np.zeros(len(path))
        sign = None
        self.blocked_idx = None
        for j, p in enumerate(path):                       # shift needed at each point
            s = self._clear_shift(p, sign)
            if s is None and self.blocked_idx is None:     # no shift inside the corridor clears it
                self.blocked_idx = j
            if s:                                          # blocked: keep one side per obstacle
                shifts[j], sign = s, (1 if s > 0 else -1)
            else:
                sign = None if s == 0.0 else sign

        # Spread each shift `margin` metres before and after (largest wins).
        ds = max(np.linalg.norm(path[1] - path[0]), 1e-6)
        n = int(margin / ds)
        spread = shifts.copy()
        for j in np.nonzero(shifts)[0]:
            lo, hi = max(j - n, 0), min(j + n + 1, len(path))
            window = spread[lo:hi]
            spread[lo:hi] = np.where(np.abs(window) >= abs(shifts[j]), window, shifts[j])

        clear = path.copy()
        clear[:, 1] += spread
        return clear

    @property
    def stop_idx(self):
        """Last global-path index the band may extend to: the end of the road,
        or STOP_BACKOFF metres before the first point the corridor can't clear."""
        if self.blocked_idx is None:
            return len(self.global_path) - 1
        ds = max(np.linalg.norm(self.global_path[1] - self.global_path[0]), 1e-6)
        return max(self.blocked_idx - int(STOP_BACKOFF / ds), 0)

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
        for j in range(i0 + 1, self.stop_idx + 1):
            dist += float(np.linalg.norm(path[j] - window[-1]))
            window.append(self.reference[j])
            self.tail_idx = j
            if dist >= self.band_length:
                break
        return np.array(window)

    def update_band(self, max_iter=300, **band_params):
        """Build the elastic band over the next `band_length` metres of the global
        path and let it settle. Both ends stay fixed: the car, and the point
        `band_length` metres ahead on the global path. Returns the band."""
        from elastic_band import ElasticBand

        # The band is pulled toward the obstacle-free reference.
        self.band = ElasticBand(self.band_map, self.band_window(),
                                reference=self.reference, **band_params)
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

    @property
    def status(self):
        """What the car is doing, for the decision layer:
          done       reached the end of the reference
          blocked    stopped: nothing inside the corridor clears the obstacle ahead
          detour     off the reference to pass an obstacle: the reference is
                     shifted here, or the band has pushed the car more than
                     DETOUR_OFFSET off it
          following  on the reference
        """
        if self.global_path is None:
            return "idle"
        if self.done:
            return "done"
        if (self.blocked_idx is not None and self.band is not None
                and self.tail_idx >= self.stop_idx
                and np.linalg.norm(self.position - self.band.points[-1]) < 0.05):
            return "blocked"
        i = int(np.argmin(np.linalg.norm(self.global_path - self.position, axis=1)))
        if (abs(self.reference[i, 1] - self.global_path[i, 1]) > 1e-6
                or np.linalg.norm(self.position - self.global_path[i]) > DETOUR_OFFSET):
            return "detour"
        return "following"

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
        while length < self.band_length and self.tail_idx < self.stop_idx:
            self.tail_idx += 1
            length += float(np.linalg.norm(path[self.tail_idx] - new_pts[-1]))
            new_pts.append(self.reference[self.tail_idx])

        # 3) Let the band relax a little.
        band.points = np.array(new_pts)
        band.fix_connectivity()
        for _ in range(band_steps):
            band.step()

        # 4) Lane change finished? Then the new lane becomes the hard boundary.
        self._finish_lane_change()
        return float(np.linalg.norm(moved))

    def __str__(self):
        s = (f"Car at ({self.x:.2f}, {self.y:.2f}) m, yaw {np.degrees(self.yaw):.1f} deg, "
             f"{self.width:g} m wide x {self.length:g} m long")
        if self.global_path is not None:
            s += f"\n  global path: {len(self.global_path)} points, cost {self.global_cost:.2f}"
        if self.band is not None:
            s += f"\n  band ({self.band_length:g} m window): {self.band}"
        if self.global_path is not None:
            s += f"\n  status: {self.status}"
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
