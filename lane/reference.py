"""
Reference line: where the car SHOULD drive, before obstacles are considered.

Replaces a lattice / search planner. The lane structure already says where
to drive (the lane centreline), so the reference is just:

  - the centreline of the current lane, and
  - for a lane change: a smooth blend from the current lane's centreline to
    the target lane's centreline, over `blend_length` metres.

The elastic band is then pulled toward this line and pushed off obstacles.

Blend shape: quintic smoothstep  y = y0 + (y1 - y0) * (10s^3 - 15s^4 + 6s^5),
s = 0..1 over the blend. Heading AND curvature are zero at both ends, so the
car enters and leaves the lane change without a jump in steering. This is
the standard lateral profile for lane changes in Frenet-frame planning
(Werling et al., ICRA 2010).

Peak curvature ~ 5.77 * lateral_shift / blend_length^2. With a 3.33 m shift
over 20 m: ~0.048 1/m (a ~21 m radius). In WATO, blend_length should grow
with speed (e.g. speed x a lane-change time of 3-5 s).
"""
import numpy as np


def smoothstep5(s):
    """Quintic smoothstep: 0 -> 1 with zero 1st and 2nd derivative at both ends."""
    s = np.clip(s, 0.0, 1.0)
    return s ** 3 * (10 - 15 * s + 6 * s ** 2)


def lane_reference(costmap, lane=2, target_lane=None, change_x=15.0, blend_length=20.0, ds=0.1):
    """Reference line along the whole road as an (N, 2) array of (x, y).

    lane         : lane to start in (1 = rightmost)
    target_lane  : lane to end in; None = stay in `lane`
    change_x     : x (m) where the lane change starts
    blend_length : distance (m) over which it happens
    """
    x = np.arange(ds / 2, costmap.length, ds)
    y0 = costmap.centers[lane - 1]
    y1 = costmap.centers[(target_lane or lane) - 1]
    y = y0 + (y1 - y0) * smoothstep5((x - change_x) / blend_length)
    return np.column_stack([x, y])


def max_curvature(points, spacing=0.5):
    """Largest curvature (1/m) along a polyline, after resampling it every
    `spacing` metres (so per-cell jitter doesn't count). Uses the circle
    through each 3 consecutive points (Menger curvature)."""
    pts = np.asarray(points, float)
    s = np.concatenate([[0], np.cumsum(np.linalg.norm(np.diff(pts, axis=0), axis=1))])
    if s[-1] < 2 * spacing:
        return 0.0
    t = np.arange(0, s[-1], spacing)
    p = np.column_stack([np.interp(t, s, pts[:, 0]), np.interp(t, s, pts[:, 1])])
    a, b, c = p[:-2], p[1:-1], p[2:]
    cross = np.abs((b[:, 0] - a[:, 0]) * (c[:, 1] - a[:, 1]) - (b[:, 1] - a[:, 1]) * (c[:, 0] - a[:, 0]))
    denom = (np.linalg.norm(b - a, axis=1) * np.linalg.norm(c - b, axis=1) * np.linalg.norm(c - a, axis=1))
    k = np.where(denom > 1e-12, 2 * cross / np.maximum(denom, 1e-12), 0.0)
    return float(k.max())
