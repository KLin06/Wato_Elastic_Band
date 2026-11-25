"""
Elastic band on the lane costmap (adapted from demo/band.py).

The band takes a piece of the global path and reshapes it: it TIGHTENS
(shorter, smoother) and is PUSHED AWAY from obstacles. It is made of
"bubbles": each band point has a radius = distance to the nearest obstacle,
and neighbouring bubbles must overlap so the band stays collision free.

What counts as an obstacle here: obstacle grid + corridor
----------------------------------------------------------
The band only needs to know where it may NOT go; that is the paper's model
(bubble radius = distance to the nearest obstacle). Two sources:

  1. Obstacle grid: costmap cells with cost >= OBSTACLE_COST (lethal):
     objects (grown by the car's size) and the car being over a road edge.
  2. Corridor: the lanes the decision layer allows. Wherever the car's
     centre would put part of the car outside them is also an obstacle
     (see Car.corridor_mask()). This keeps the car in its lane(s) with the
     same collision-free guarantee as for objects.

The costmap's graded lane costs (centring, lane lines, out-of-lane) are not
used: the reference pull centres the band and the corridor bounds it.

One update step
---------------
For each bubble except the two fixed ends:
  1. contraction  k_c * (unit(prev - b) + unit(next - b))        tightens the band
  2. repulsion    k_r * (rho0 - rho) * grad(rho)  if rho < rho0  pushes off obstacles
  3. reference    k_ref * (nearest reference point - b)          pulls back toward the
                                                                 lane (reference.py)
  4. drop the part of the force along the band
  5. move by alpha * rho * force, capped at half the gap to the nearest
     neighbour; halve the step if it would get within rho_min of an obstacle
Then insert bubbles where neighbours stop overlapping and delete redundant ones.

Not handled yet: the car's turning limit. Nothing stops the band bending
tighter than the car can steer.

Steps 1, 2, 4, 5 are Quinlan & Khatib (1993). Step 3 is the reference-path
term used for on-road bands: TEB's path following (Rosmann et al. 2012/2013)
and Autoware's elastic band smoother keep the band near a reference path the
same way. Without it, contraction alone straightens the band across lanes.
"""
import numpy as np
from scipy.ndimage import distance_transform_edt, map_coordinates
from scipy.spatial import cKDTree

from lane_costmap import LETHAL, LaneCostmap

OBSTACLE_COST = LETHAL     # cells at or above this are obstacles for the band


def unit(v):
    """v scaled to length 1 (zero vector stays zero)."""
    n = np.linalg.norm(v)
    return v / n if n > 1e-9 else np.zeros_like(v)


class BandMap:
    """Distance-to-obstacle view of a LaneCostmap, which is what the band needs.

    inside      : optional bool grid (rows, cols), True where the car's centre
                  is inside the allowed corridor; outside counts as obstacle
    rho(p)      : distance (m) from world point p to the nearest obstacle cell
    grad_rho(p) : direction that increases rho fastest (away from obstacles)
    """

    def __init__(self, costmap: LaneCostmap, obstacle_cost=OBSTACLE_COST, inside=None):
        self.cm = costmap
        self.resolution = costmap.resolution
        free = costmap.grid < obstacle_cost
        if inside is not None:
            free &= inside
        self.dist = distance_transform_edt(free) * self.resolution
        gy, gx = np.gradient(self.dist, self.resolution)   # rows = y, cols = x
        self.grad_x, self.grad_y = gx, gy

    def _sample(self, field, p):
        """Bilinear lookup of a grid field at world point p = (x, y)."""
        col = p[0] / self.resolution - 0.5
        row = p[1] / self.resolution - 0.5
        return float(map_coordinates(field, [[row], [col]], order=1, mode="nearest")[0])

    def rho(self, p):
        return self._sample(self.dist, p)

    def grad_rho(self, p):
        return np.array([self._sample(self.grad_x, p), self._sample(self.grad_y, p)])


class ElasticBand:
    def __init__(self, band_map: BandMap, path, reference=None,
                 k_c=1.0,              # contraction gain: how hard the band tightens
                 k_r=4.0,              # repulsion gain: how hard obstacles push
                 k_ref=1.0,            # reference gain: how hard the band is pulled to the reference
                 rho0=0.5,             # (m) repulsion acts only within this distance
                 rho_max=1.0,          # (m) cap on bubble radius
                 rho_min=0.05,         # (m) bubbles never get closer than this to an obstacle
                 alpha=0.3,            # step size as a fraction of the bubble radius
                 insert_overlap=0.9,   # insert if |b_i - b_i+1| > this * (r_i + r_i+1)
                 delete_overlap=0.6,   # delete b_i+1 if |b_i - b_i+2| < this * (r_i + r_i+2)
                 max_bubbles=500):
        self.map = band_map
        self.k_c, self.k_r, self.k_ref = k_c, k_r, k_ref

        self.set_reference(reference)
        self.rho0, self.rho_max, self.rho_min = rho0, rho_max, rho_min
        self.alpha = alpha
        self.insert_overlap, self.delete_overlap = insert_overlap, delete_overlap
        self.max_bubbles = max_bubbles

        self.points = np.array(path, dtype=float)
        self.broken = False      # True if the band couldn't stay connected / collision free
        self.iterations = 0
        self.fix_connectivity()

    def set_reference(self, reference):
        """Reference line (N x 2), e.g. from reference.py, that the band is
        pulled toward. A KD-tree finds the nearest reference point to each
        bubble quickly. None = no pull."""
        self.reference = None if reference is None else np.asarray(reference, float)
        self._ref_tree = None if reference is None else cKDTree(self.reference)

    # ---- Bubbles -----------------------------------------------------------
    def radius(self, p):
        return min(self.map.rho(p), self.rho_max)

    def radii(self):
        return np.array([self.radius(p) for p in self.points])

    # ---- Forces ------------------------------------------------------------
    def total_force(self, i):
        b = self.points
        f = self.k_c * (unit(b[i - 1] - b[i]) + unit(b[i + 1] - b[i]))    # contraction
        rho = self.map.rho(b[i])
        if rho < self.rho0:                                                # repulsion
            f = f + self.k_r * (self.rho0 - rho) * self.map.grad_rho(b[i])
        if self._ref_tree is not None:                                     # reference pull
            _, j = self._ref_tree.query(b[i])
            f = f + self.k_ref * (self.reference[j] - b[i])
        t = unit(b[i + 1] - b[i - 1])                                      # tangent
        return f - np.dot(f, t) * t                                        # sideways part only

    # ---- Update ------------------------------------------------------------
    def step(self):
        """Move every bubble once, then repair the chain. Returns the largest move (m)."""
        old = self.points
        new = old.copy()
        max_move = 0.0
        for i in range(1, len(old) - 1):                  # both ends stay fixed
            move = self.alpha * self.radius(old[i]) * self.total_force(i)

            # Never move more than half the distance to the nearest neighbour,
            # so a bubble can't jump past a neighbour and back.
            gap = min(np.linalg.norm(old[i] - old[i - 1]), np.linalg.norm(old[i + 1] - old[i]))
            n = np.linalg.norm(move)
            if n > 0.5 * gap:
                move *= 0.5 * gap / n

            # Never step into (or right next to) an obstacle: try full, half, quarter.
            for _ in range(3):
                if self.map.rho(old[i] + move) >= self.rho_min:
                    new[i] = old[i] + move
                    max_move = max(max_move, float(np.linalg.norm(move)))
                    break
                move = move / 2

        self.points = new
        self.fix_connectivity()
        self.iterations += 1
        return max_move

    def run(self, max_iter=300, tol=1e-3):
        """Step until bubbles stop moving (largest move < tol) or max_iter."""
        for _ in range(max_iter):
            if self.step() < tol:
                break
        return self

    def fix_connectivity(self):
        """Insert bubbles where neighbours don't overlap; delete redundant ones."""
        pts = list(self.points)
        i = 0
        while i < len(pts) - 1:
            a, c = pts[i], pts[i + 1]
            ra, rc = self.radius(a), self.radius(c)

            if np.linalg.norm(c - a) > self.insert_overlap * (ra + rc):     # gap: insert
                mid = (a + c) / 2
                if self.map.rho(mid) <= 0 or len(pts) >= self.max_bubbles:
                    self.broken = True
                    i += 1
                    continue
                pts.insert(i + 1, mid)
                continue

            if i + 2 < len(pts):                                            # redundant: delete
                d = pts[i + 2]
                if np.linalg.norm(d - a) < self.delete_overlap * (ra + self.radius(d)):
                    pts.pop(i + 1)
                    continue
            i += 1

        self.points = np.array(pts)
        if any(self.map.rho(p) <= 0 for p in self.points):
            self.broken = True

    # ---- Stats -------------------------------------------------------------
    def length(self):
        return float(np.sum(np.linalg.norm(np.diff(self.points, axis=0), axis=1)))

    def __str__(self):
        status = "BROKEN" if self.broken else "ok"
        return (f"ElasticBand: {len(self.points)} bubbles | length {self.length():.2f} m"
                f" | iterations {self.iterations} | {status}")
