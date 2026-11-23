"""
Step 3: the elastic band (Quinlan & Khatib, 1993).

What is this file for?
----------------------
A* gives a path that picks the route but is blocky and hugs walls. The elastic
band takes that path and reshapes it, as if it were a rubber band lying in the
free space:

  - it TIGHTENS (shorter and smoother), and
  - it is PUSHED OFF obstacles (keeps clearance).

If the map changes (an obstacle moves), the band keeps deforming in real time,
without running A* again.

Bubbles
-------
The band is a chain of points b[0], b[1], ..., b[N-1]. Around each point is a
"bubble": a circle of radius rho(b) = distance to the nearest obstacle. Inside
a bubble the robot cannot hit anything. If every pair of neighbouring bubbles
OVERLAPS, the whole band is guaranteed to be collision free.

One update step (step())
------------------------
For each bubble (except the fixed start and goal):

 1. Contraction force (internal), pulls the bubble toward its neighbours:
        f_c = k_c * ( unit(b[i-1] - b[i]) + unit(b[i+1] - b[i]) )
    Unit vectors, so the pull does not depend on how far apart bubbles are.
    On a straight band the two vectors cancel. At a bend they add up and pull
    the bend straight.

 2. Repulsion force (external), pushes the bubble away from obstacles:
        f_r = k_r * (rho0 - rho(b)) * grad_rho(b)     if rho(b) < rho0
            = 0                                       otherwise
    grad_rho points away from the nearest obstacle (see costmap.py).

 3. Remove the tangential part of f = f_c + f_r (the part along the band).
    Only sideways motion changes the band's shape. Sliding along the band just
    bunches bubbles together.

 4. Move the bubble by a step proportional to its own radius:
        b[i] += alpha * rho(b[i]) * f
    Big bubbles (open space) move fast, small bubbles (near walls) move
    carefully. If the new spot is too close to an obstacle, the step is halved.

Then repair the chain (fix_connectivity()):
  - If two neighbours no longer overlap, insert a bubble between them.
  - If a bubble's two neighbours already overlap each other, it is redundant:
    delete it.
"""
import numpy as np

from costmap import Costmap


def unit(v):
    """Vector v scaled to length 1 (or zero if v is zero)."""
    n = np.linalg.norm(v)
    return v / n if n > 1e-9 else np.zeros_like(v)


class ElasticBand:
    def __init__(self, costmap: Costmap, path,
                 k_c=1.0,          # contraction gain: how hard the band tightens
                 k_r=4.0,          # repulsion gain: how hard obstacles push
                 rho0=0.5,         # (m) repulsion acts only within this clearance
                 rho_max=1.0,      # (m) cap on bubble radius (keeps bubbles reasonable in open space)
                 rho_min=0.05,     # (m) a bubble may never get closer than this to an obstacle
                 alpha=0.3,        # step size as a fraction of the bubble radius
                 insert_overlap=0.9,   # insert if |b_i - b_i+1| > this * (rho_i + rho_i+1)
                 delete_overlap=0.6,   # delete b_i+1 if |b_i - b_i+2| < this * (rho_i + rho_i+2)
                 max_bubbles=500):
        self.cm = costmap
        self.k_c, self.k_r = k_c, k_r
        self.rho0, self.rho_max, self.rho_min = rho0, rho_max, rho_min
        self.alpha = alpha
        self.insert_overlap, self.delete_overlap = insert_overlap, delete_overlap
        self.max_bubbles = max_bubbles

        # The band starts as the A* path. Every grid cell becomes a bubble;
        # fix_connectivity() then deletes the redundant ones.
        self.points = np.array(path, dtype=float)
        self.broken = False        # True if the band could not stay connected
        self.iterations = 0
        self.fix_connectivity()

    # ======================================================================
    # Bubble radius
    # ======================================================================
    def radius(self, p):
        """Bubble radius at p: distance to the nearest obstacle, capped at rho_max."""
        return min(self.cm.rho(p), self.rho_max)

    def radii(self):
        """Radius of every bubble, as an array (handy for drawing)."""
        return np.array([self.radius(p) for p in self.points])

    # ======================================================================
    # Forces
    # ======================================================================
    def contraction_force(self, i):
        b = self.points
        return self.k_c * (unit(b[i - 1] - b[i]) + unit(b[i + 1] - b[i]))

    def repulsion_force(self, i):
        p = self.points[i]
        rho = self.cm.rho(p)
        if rho >= self.rho0:
            return np.zeros(2)                       # far enough: no push
        return self.k_r * (self.rho0 - rho) * self.cm.grad_rho(p)

    def total_force(self, i):
        """Contraction + repulsion, with the component along the band removed."""
        f = self.contraction_force(i) + self.repulsion_force(i)
        t = unit(self.points[i + 1] - self.points[i - 1])   # tangent direction
        return f - np.dot(f, t) * t

    # ======================================================================
    # One update
    # ======================================================================
    def step(self):
        """Move every bubble once, then repair the chain.
        Returns the largest distance any bubble moved (m), to check convergence."""
        # Forces are computed from the OLD positions for all bubbles, then
        # applied together, so the result doesn't depend on update order.
        old = self.points
        new = old.copy()
        max_move = 0.0

        for i in range(1, len(old) - 1):             # start and goal stay fixed
            f = self.total_force(i)
            move = self.alpha * self.radius(old[i]) * f

            # Don't move more than half the distance to the nearest neighbour.
            # Otherwise a bubble that sits very close to a neighbour (e.g. right
            # in front of the robot in local_band.py) can jump past it and back
            # on alternate steps, so the band jitters instead of settling.
            gap = min(np.linalg.norm(old[i] - old[i - 1]),
                      np.linalg.norm(old[i + 1] - old[i]))
            n = np.linalg.norm(move)
            if n > 0.5 * gap:
                move *= 0.5 * gap / n

            # Safety: never step into (or right next to) an obstacle.
            # Try the full step, then half, then a quarter; otherwise stay put.
            for _ in range(3):
                if self.cm.rho(old[i] + move) >= self.rho_min:
                    new[i] = old[i] + move
                    max_move = max(max_move, float(np.linalg.norm(move)))
                    break
                move = move / 2

        self.points = new
        self.fix_connectivity()
        self.iterations += 1
        return max_move

    def run(self, max_iter=500, tol=1e-3, callback=None):
        """Call step() until bubbles stop moving (max move < tol) or max_iter.
        callback(band) is called after every step (used for animation)."""
        for _ in range(max_iter):
            moved = self.step()
            if callback:
                callback(self)
            if moved < tol:
                break
        return self

    # ======================================================================
    # Keeping the chain connected
    # ======================================================================
    def fix_connectivity(self):
        """Insert bubbles where neighbours don't overlap, delete redundant ones.

        Inserting and deleting use different thresholds (0.9 vs 0.6). That
        gap stops a bubble being inserted and deleted again on alternate steps.
        """
        pts = list(self.points)
        i = 0
        while i < len(pts) - 1:
            a, c = pts[i], pts[i + 1]
            ra, rc = self.radius(a), self.radius(c)

            # 1) Gap between neighbours: insert a bubble halfway.
            if np.linalg.norm(c - a) > self.insert_overlap * (ra + rc):
                mid = (a + c) / 2
                if self.cm.rho(mid) <= 0 or len(pts) >= self.max_bubbles:
                    # The midpoint is inside an obstacle (or there are too
                    # many bubbles): the band is broken here. In a real system
                    # you would call the global planner (A*) again.
                    self.broken = True
                    i += 1
                    continue
                pts.insert(i + 1, mid)
                continue                 # re-check a against the new bubble

            # 2) Bubble i+1 is redundant if a and pts[i+2] overlap enough on their own.
            if i + 2 < len(pts):
                d = pts[i + 2]
                if np.linalg.norm(d - a) < self.delete_overlap * (ra + self.radius(d)):
                    pts.pop(i + 1)
                    continue             # re-check a against its new neighbour

            i += 1

        self.points = np.array(pts)

        # 3) A bubble whose centre ended up inside an obstacle (e.g. an
        #    obstacle moved onto it faster than it could escape) also breaks the band.
        if any(self.cm.rho(p) <= 0 for p in self.points):
            self.broken = True

    # ======================================================================
    # Stats / printing
    # ======================================================================
    def length(self):
        return float(np.sum(np.linalg.norm(np.diff(self.points, axis=0), axis=1)))

    def min_clearance(self):
        return float(min(self.cm.rho(p) for p in self.points))

    def __str__(self):
        status = "BROKEN" if self.broken else "ok"
        return (f"ElasticBand: {len(self.points)} bubbles | length {self.length():.2f} m"
                f" | min clearance {self.min_clearance():.2f} m"
                f" | iterations {self.iterations} | {status}")


if __name__ == "__main__":
    # Run `python band.py`: A* path -> elastic band, printed and plotted.
    import matplotlib.pyplot as plt
    from matplotlib.patches import Circle
    from astar import astar, path_stats
    from costmap import demo_map

    cm = demo_map()
    start, goal = (1.0, 1.0), (9.0, 9.0)
    path = astar(cm, start, goal)

    length, min_c, _ = path_stats(cm, path)
    print(f"A* path        : {len(path)} cells | length {length:.2f} m | min clearance {min_c:.2f} m")

    band = ElasticBand(cm, path)
    print(f"Band (initial) : {band}")
    band.run(max_iter=500)
    print(f"Band (final)   : {band}")

    print("\nFinal band drawn on the grid (every 2nd cell):")
    cm.print_grid(step=2, path=band.points)

    # ---- Plot -----------------------------------------------------------
    fig, ax = plt.subplots(figsize=(6, 6))
    ax.imshow(cm.grid, cmap="gray_r", origin="lower", extent=cm.extent)
    xs, ys = zip(*path)
    ax.plot(xs, ys, "--", color="gray", lw=1.5, label="A* path")
    for p, r in zip(band.points, band.radii()):
        ax.add_patch(Circle(p, r, color="tab:blue", alpha=0.12))
    ax.plot(band.points[:, 0], band.points[:, 1], "-o", color="tab:blue",
            ms=3, lw=2, label="elastic band")
    ax.plot(*start, "go", ms=9, label="start")
    ax.plot(*goal, "r*", ms=13, label="goal")
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 10)
    ax.set_aspect("equal")
    ax.set_title(f"Elastic band after {band.iterations} iterations")
    ax.legend(loc="lower right", fontsize=8)
    plt.tight_layout()
    plt.savefig("band_preview.png", dpi=100)
    plt.show()
