"""
Step 1: the costmap.

What is this file for?
----------------------
The elastic band algorithm (Quinlan & Khatib, 1993) represents a path as a
chain of "bubbles". A bubble is a circle of free space around a point on the
path. To build bubbles, the band needs to ask the map two questions:

  1. rho(p)       -> "How far is point p from the nearest obstacle?"
                     This is the bubble's radius. Inside that circle the robot
                     cannot hit anything.

  2. grad_rho(p)  -> "Which direction moves p AWAY from obstacles fastest?"
                     The band uses this to push bubbles off walls
                     (the paper's "external repulsive force").

This file builds a 2D occupancy grid and pre-computes the answers to both
questions for every cell, so the band can look them up quickly.

Coordinates
-----------
- The grid is a numpy array indexed as grid[row, col].
    row increases upward  (y direction)
    col increases right   (x direction)
- Everything the band sees is in WORLD units (metres), never cells.
  A world point is p = (x, y). We convert with `resolution` (metres per cell):
      col = x / resolution,   row = y / resolution
"""
import numpy as np
from scipy.ndimage import distance_transform_edt, map_coordinates

# Cell values in the occupancy grid
FREE = 0
OBSTACLE = 1


class Costmap:
    def __init__(self, width_cells=100, height_cells=100, resolution=0.1):
        """
        width_cells, height_cells : size of the grid in cells
        resolution                : metres per cell (0.1 -> each cell is 10 cm)

        Default = 100 x 100 cells at 0.1 m  ->  a 10 m x 10 m world.
        """
        self.resolution = resolution

        # The occupancy grid itself. Starts completely free (all zeros).
        self.grid = np.zeros((height_cells, width_cells), dtype=np.uint8)

        # Build the distance field for the empty map so queries work right away.
        self.update()

    # ======================================================================
    # Building the map
    # ======================================================================
    def add_rectangle(self, x0, y0, x1, y1):
        """Mark an axis-aligned box as obstacle. Corners are given in metres:
        (x0, y0) = bottom-left, (x1, y1) = top-right."""
        r = self.resolution
        # Convert metres -> cell indices, then fill that slice of the array.
        self.grid[int(y0 / r):int(y1 / r), int(x0 / r):int(x1 / r)] = OBSTACLE

    def add_circle(self, cx, cy, radius):
        """Mark a disc as obstacle. Centre (cx, cy) and radius are in metres."""
        h, w = self.grid.shape

        # mgrid gives the row and column index of every cell.
        ys, xs = np.mgrid[0:h, 0:w]

        # Convert each cell index to the world position of the cell's CENTRE
        # (+0.5 because cell 0 spans 0 -> 0.1 m, so its centre is 0.05 m).
        xs = (xs + 0.5) * self.resolution
        ys = (ys + 0.5) * self.resolution

        # Every cell whose centre is inside the circle becomes an obstacle.
        inside = (xs - cx) ** 2 + (ys - cy) ** 2 <= radius ** 2
        self.grid[inside] = OBSTACLE

    def add_border(self):
        """Put a one-cell-thick wall around the edge so nothing leaves the map."""
        self.grid[0, :] = OBSTACLE    # bottom row
        self.grid[-1, :] = OBSTACLE   # top row
        self.grid[:, 0] = OBSTACLE    # left column
        self.grid[:, -1] = OBSTACLE   # right column

    def update(self):
        """Recompute the distance field and its gradient from the grid.

        IMPORTANT: call this every time you change the grid (add or move an
        obstacle). Otherwise rho() and grad_rho() describe the old map.
        """
        # distance_transform_edt: for every non-zero cell, the Euclidean
        # distance (in cells) to the nearest zero cell. We pass (grid == FREE),
        # so free cells are non-zero and obstacles are zero. That gives
        # "distance from each free cell to the nearest obstacle".
        # Multiply by resolution to convert cells -> metres.
        self.dist = distance_transform_edt(self.grid == FREE) * self.resolution

        # np.gradient returns the derivative along each axis:
        #   first along rows (y), then along columns (x).
        # The gradient points toward INCREASING distance, i.e. away from
        # obstacles, which is exactly the direction we want to push bubbles.
        gy, gx = np.gradient(self.dist, self.resolution)
        self.grad_x, self.grad_y = gx, gy

    # ======================================================================
    # Queries used by the elastic band
    # ======================================================================
    def _sample(self, field, p):
        """Read the value of a grid-shaped array `field` at world point p = (x, y).

        Bubbles live at real-valued positions, not exactly on cell centres, so
        we blend the 4 surrounding cells (bilinear interpolation, order=1).
        That keeps the forces smooth, so bubbles don't jitter from cell to cell.
        """
        col = p[0] / self.resolution - 0.5   # -0.5: cell centres sit at +0.5
        row = p[1] / self.resolution - 0.5
        value = map_coordinates(field, [[row], [col]], order=1, mode="nearest")
        return float(value[0])

    def rho(self, p):
        """Distance (m) from world point p to the nearest obstacle.
        This is the radius of a bubble centred at p. 0 means p is inside an obstacle."""
        return self._sample(self.dist, p)

    def grad_rho(self, p):
        """Gradient of rho at p, as the vector (d rho/dx, d rho/dy).
        Points away from the nearest obstacle."""
        return np.array([self._sample(self.grad_x, p),
                         self._sample(self.grad_y, p)])

    def is_free(self, p):
        """True if world point p is not inside an obstacle."""
        return self.rho(p) > 0

    @property
    def extent(self):
        """World bounds (left, right, bottom, top) in metres. Pass it to
        plt.imshow(..., extent=costmap.extent, origin='lower') so the plot
        axes show metres instead of cell indices."""
        h, w = self.grid.shape
        return (0, w * self.resolution, 0, h * self.resolution)

    # ======================================================================
    # Printing / debugging
    # ======================================================================
    def __str__(self):
        """Called by print(costmap). Gives a one-paragraph summary."""
        h, w = self.grid.shape
        n_obs = int(self.grid.sum())
        return (f"Costmap {w}x{h} cells @ {self.resolution} m/cell "
                f"({w * self.resolution:.1f} m x {h * self.resolution:.1f} m)\n"
                f"  obstacle cells : {n_obs} ({100 * n_obs / self.grid.size:.1f}%)\n"
                f"  max clearance  : {self.dist.max():.2f} m")

    def print_grid(self, step=1, path=None):
        """Print the occupancy grid as text in the terminal.

        step : print every `step`-th cell, so a big map fits on screen
               (e.g. step=2 on a 100x100 map prints 50x50).
        path : optional list of world points (x, y) to draw as '*'.

        Legend:  '#' obstacle   '.' free   '*' path
        """
        # Downsample first: keep every `step`-th row and column, then turn the
        # result into a 2D array of characters.
        small = self.grid[::step, ::step]
        chars = np.where(small == OBSTACLE, "#", ".").astype("<U1")

        # Overlay the path points, if given. Divide by `step` so each point
        # lands in its downsampled cell (otherwise points on skipped rows or
        # columns would vanish and the path would look broken).
        if path is not None:
            for x, y in path:
                row = int(y / self.resolution) // step
                col = int(x / self.resolution) // step
                if 0 <= row < chars.shape[0] and 0 <= col < chars.shape[1]:
                    chars[row, col] = "*"

        # Print top row first so "up" on screen is +y, like the plots.
        for row in chars[::-1]:
            print(" ".join(row))

    def print_distances(self, step=5):
        """Print the distance field (metres to nearest obstacle) as a table of
        numbers, sampling every `step`-th cell. Useful to check rho by eye."""
        sub = self.dist[::-1][::step, ::step]   # flip so top of map prints first
        for row in sub:
            print(" ".join(f"{d:4.1f}" for d in row))


def demo_map():
    """A 10 m x 10 m map with a few obstacles, used by the demo."""
    cm = Costmap(100, 100, 0.1)
    cm.add_border()
    cm.add_rectangle(2.0, 0.0, 2.6, 6.0)    # wall rising from the bottom
    cm.add_rectangle(5.0, 4.0, 5.6, 10.0)   # wall hanging from the top
    cm.add_circle(7.8, 3.0, 1.0)            # round obstacle, bottom right
    cm.add_circle(3.8, 7.5, 0.8)            # round obstacle, top middle
    cm.update()                             # must call after editing the grid
    return cm


if __name__ == "__main__":
    # Run `python costmap.py` to see the map in the terminal and as a plot.
    import matplotlib.pyplot as plt

    cm = demo_map()

    # ---- Terminal output ------------------------------------------------
    print(cm)
    print("\nOccupancy grid (every 2nd cell):")
    cm.print_grid(step=2)
    print("\nDistance to nearest obstacle in metres (every 5th cell):")
    cm.print_distances(step=5)
    p = (1.0, 1.0)
    print(f"\nAt p={p}: rho = {cm.rho(p):.3f} m, grad_rho = {cm.grad_rho(p).round(2)}")

    # ---- Plot -----------------------------------------------------------
    fig, axes = plt.subplots(1, 2, figsize=(11, 5))

    # Left: the raw occupancy grid (black = obstacle).
    axes[0].imshow(cm.grid, cmap="gray_r", origin="lower", extent=cm.extent)
    axes[0].set_title("Occupancy grid")

    # Right: the distance field as colour, with gradient arrows on top.
    im = axes[1].imshow(cm.dist, cmap="viridis", origin="lower", extent=cm.extent)
    step = 6                                   # draw an arrow every 6 cells
    h, w = cm.grid.shape
    ys, xs = np.mgrid[0:h:step, 0:w:step]
    axes[1].quiver((xs + 0.5) * cm.resolution, (ys + 0.5) * cm.resolution,
                   cm.grad_x[::step, ::step], cm.grad_y[::step, ::step],
                   color="white", scale=40)
    axes[1].set_title("Distance to obstacle ρ  +  ∇ρ")
    fig.colorbar(im, ax=axes[1], label="metres")

    plt.tight_layout()
    plt.savefig("costmap_preview.png", dpi=100)
    plt.show()
