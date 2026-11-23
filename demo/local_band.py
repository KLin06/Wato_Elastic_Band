"""
Step 5: a LOCAL elastic band that moves with the robot.

What is different from band.py?
--------------------------------
band.py stretches one band over the WHOLE A* path, from start to goal.
A real robot doesn't need that: it only cares about the next few metres.
This version works like a moving window (how ROS's eband_local_planner works):

    global A* path:  S ─────────────────────────────────────────── G
    local band:            R ═══════════ T
                           ^ robot       ^ tail (a point on the A* path,
                                           about `lookahead` metres of band ahead)

Every tick:
  1. ADVANCE  the robot moves `speed` metres along the band, and the part of
              the band behind it is cut off. The robot is always bubble 0.
  2. EXTEND   while the band is shorter than `lookahead`, the next A* path
              points are added at the tail, so the window slides forward.
  3. DEFORM   the usual elastic band step(s) from band.py: contraction,
              repulsion, insert/delete bubbles. Robot (first bubble) and tail
              (last bubble) stay fixed.

Only the bubbles inside the window are updated, so the cost per tick depends
on `lookahead`, not on how long the whole route is.
"""
import numpy as np

from band import ElasticBand
from costmap import Costmap


class LocalElasticBand(ElasticBand):
    def __init__(self, costmap: Costmap, global_path,
                 lookahead=3.0,     # (m) how much band to keep ahead of the robot
                 speed=0.03,        # (m) how far the robot moves per tick
                 goal_tol=0.05,     # (m) "arrived" when the robot is this close to the goal
                 **band_params):    # k_c, k_r, rho0, ... passed on to ElasticBand
        self.global_path = np.array(global_path, dtype=float)
        self.lookahead, self.speed, self.goal_tol = lookahead, speed, goal_tol

        # tail_idx = index of the last global-path point already added to the band
        self.tail_idx = 0

        # Start as a one-bubble band at the robot, then grow it to the lookahead.
        super().__init__(costmap, self.global_path[:1], **band_params)
        self.extend()
        self.fix_connectivity()

    # ======================================================================
    # Robot state
    # ======================================================================
    @property
    def robot(self):
        """The robot is always the first bubble of the band."""
        return self.points[0]

    @property
    def goal(self):
        return self.global_path[-1]

    @property
    def done(self):
        """True once the window has reached the goal and the robot is on it."""
        return (self.tail_idx == len(self.global_path) - 1 and
                np.linalg.norm(self.robot - self.goal) < self.goal_tol)

    # ======================================================================
    # 1. Advance the robot along the band
    # ======================================================================
    def advance(self):
        """Move the robot `speed` metres along the band and drop the band behind it."""
        pts = self.points
        remaining = self.speed
        pos = pts[0].copy()
        i = 0                                   # index of the band point behind `pos`

        # Walk along the band's segments until we've used up `speed` metres.
        while i < len(pts) - 1:
            seg = pts[i + 1] - pos
            seg_len = np.linalg.norm(seg)
            if seg_len >= remaining:            # the new position is on this segment
                pos = pos + seg / seg_len * remaining
                break
            remaining -= seg_len                # skip the whole segment
            pos = pts[i + 1].copy()
            i += 1

        # Safety: if an obstacle has moved onto the next spot, wait instead.
        if self.cm.rho(pos) < self.rho_min:
            return

        # New band = robot position + every band point still ahead of it.
        self.points = np.vstack([pos, pts[i + 1:]])

    # ======================================================================
    # 2. Extend the tail along the global path
    # ======================================================================
    def extend(self):
        """Add global-path points at the tail until the band is `lookahead` long
        (or the goal has been added)."""
        pts = list(self.points)
        length = self.length()
        while length < self.lookahead and self.tail_idx < len(self.global_path) - 1:
            self.tail_idx += 1
            nxt = self.global_path[self.tail_idx]
            length += float(np.linalg.norm(nxt - pts[-1]))
            pts.append(nxt)
        self.points = np.array(pts)

    # ======================================================================
    # One tick = advance + extend + deform
    # ======================================================================
    def tick(self, band_steps=1):
        """Move the robot, slide the window forward, then relax the band.
        Returns how far the robot actually moved (0 if it had to wait)."""
        if self.done:
            return 0.0
        before = self.robot.copy()
        self.advance()
        self.extend()
        self.fix_connectivity()
        for _ in range(band_steps):
            self.step()
        return float(np.linalg.norm(self.robot - before))

    def remaining_distance(self):
        """Distance left to the goal: rest of this band + rest of the global path."""
        rest = self.global_path[self.tail_idx:]
        return self.length() + float(np.sum(np.linalg.norm(np.diff(rest, axis=0), axis=1)))

    def __str__(self):
        status = "DONE" if self.done else ("BROKEN" if self.broken else "ok")
        return (f"LocalElasticBand: robot ({self.robot[0]:.2f}, {self.robot[1]:.2f}) | "
                f"{len(self.points)} bubbles | window {self.length():.2f} m | "
                f"to goal {self.remaining_distance():.2f} m | {status}")


if __name__ == "__main__":
    # Run `python local_band.py`: drive the robot to the goal and print progress.
    from astar import astar
    from costmap import demo_map

    cm = demo_map()
    path = astar(cm, (1.0, 1.0), (9.0, 9.0))
    band = LocalElasticBand(cm, path, lookahead=3.0, speed=0.03)

    trail = [band.robot.copy()]
    tick = 0
    while not band.done and tick < 2000:
        band.tick()
        trail.append(band.robot.copy())
        tick += 1
        if tick % 100 == 0:
            print(f"tick {tick:4d}: {band}")

    trail = np.array(trail)
    driven = float(np.sum(np.linalg.norm(np.diff(trail, axis=0), axis=1)))
    min_clear = min(cm.rho(p) for p in trail)
    print(f"\nArrived: {band.done} after {tick} ticks | driven {driven:.2f} m"
          f" | min clearance along the way {min_clear:.2f} m")
