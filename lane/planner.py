"""
Planner: the cheapest path from one end of the road (x = 0) to the other
(x = road length), on the lane costmap.

The car is a point (car.py), so a path is a sequence of (x, y) positions of
the car's centre.

Moves
-----
The car only drives FORWARD. From a cell it can go to the next column
(x + one cell) in the same row, one row up, or one row down:

        (row+1) ↗
    cell ─────→ (row)
        (row-1) ↘

So y can change by at most one cell per cell forward (45 degrees).

Cost of a move
--------------
    move_cost = step_length * (1 + COST_WEIGHT * cell_cost / 100)

  - step_length : metres moved (resolution, or resolution * sqrt(2) diagonally)
  - cell_cost   : costmap value of the cell moved INTO (0..100)
  - lethal cells (100) can't be entered at all

So a path is charged for its length, plus extra for every metre spent on
expensive cells. COST_WEIGHT sets how much the costmap matters compared with
distance: 0 = shortest path only, higher = sticks to cheap cells more.

Why it's optimal
----------------
Because every move goes one column forward, the cells form layers (one per
column) and a path visits each layer once. So we can sweep left to right:
the cheapest way to reach each cell in column c is the cheapest of the three
cells it can come from in column c-1, plus the move cost. This dynamic
programming gives the exact cheapest path, without needing A*.
"""
import numpy as np

from lane_costmap import LETHAL, LaneCostmap

COST_WEIGHT = 10.0      # how much the costmap matters compared with distance


class Planner:
    def __init__(self, costmap: LaneCostmap, cost_weight=COST_WEIGHT):
        self.cm = costmap
        self.cost_weight = cost_weight

    # ---- Conversions -------------------------------------------------------
    def row_of(self, y):
        return int(np.clip(y / self.cm.resolution, 0, self.cm.rows - 1))

    def y_of(self, row):
        return (row + 0.5) * self.cm.resolution

    def x_of(self, col):
        return (col + 0.5) * self.cm.resolution

    # ---- Planning ----------------------------------------------------------
    def plan(self, start_y=None, end_y=None):
        """Cheapest path from (x = 0, start_y) to (x = road end, end_y).

        start_y, end_y : metres across the road. Default: centre of lane 2.
        Returns (path, total_cost): path is an (N, 2) array of (x, y) points.
        Raises ValueError if the start or end is lethal, or no path exists.
        """
        cm = self.cm
        default_y = cm.centers[1]                         # lane 2 centre
        start_y = default_y if start_y is None else start_y
        end_y = default_y if end_y is None else end_y
        r_start, r_end = self.row_of(start_y), self.row_of(end_y)

        grid = cm.grid.astype(float)
        for name, r, c in (("start", r_start, 0), ("end", r_end, cm.cols - 1)):
            if grid[r, c] >= LETHAL:
                raise ValueError(f"{name} y = {self.y_of(r):.2f} m is lethal (off the road)")

        res = cm.resolution
        # Extra factor for entering each cell; lethal cells can't be entered.
        factor = 1 + self.cost_weight * grid / LETHAL
        factor[grid >= LETHAL] = np.inf

        # best[r]    = cheapest cost to reach row r of the current column
        # came[c, r] = row in column c-1 that the cheapest path to (r, c) came from
        best = np.full(cm.rows, np.inf)
        best[r_start] = 0.0
        came = np.zeros((cm.cols, cm.rows), dtype=np.int64)

        rows = np.arange(cm.rows)
        for c in range(1, cm.cols):
            # Three candidates for each row r: come from r (straight), r-1 or r+1 (diagonal).
            straight = best + res * factor[:, c]
            from_below = np.full(cm.rows, np.inf)          # came from row r-1
            from_below[1:] = best[:-1] + res * np.sqrt(2) * factor[1:, c]
            from_above = np.full(cm.rows, np.inf)          # came from row r+1
            from_above[:-1] = best[1:] + res * np.sqrt(2) * factor[:-1, c]

            options = np.vstack([straight, from_below, from_above])
            pick = np.argmin(options, axis=0)              # 0 straight, 1 below, 2 above
            best = options[pick, rows]
            came[c] = rows - (pick == 1) + (pick == 2)

        total = best[r_end]
        if not np.isfinite(total):
            raise ValueError("no path from start to end")

        # Walk back from the end through `came` to recover the rows.
        path_rows = [r_end]
        for c in range(cm.cols - 1, 0, -1):
            path_rows.append(came[c, path_rows[-1]])
        path_rows.reverse()

        path = np.column_stack([[self.x_of(c) for c in range(cm.cols)],
                                [self.y_of(r) for r in path_rows]])
        return path, float(total)

    # ---- Stats -------------------------------------------------------------
    def path_stats(self, path):
        """Length (m), max cell cost along the path, and lanes it passes through."""
        length = float(np.sum(np.linalg.norm(np.diff(path, axis=0), axis=1)))
        costs = [self.cm.cost_at(x, y) for x, y in path]
        lanes = []
        for _, y in path:
            lane = self.cm.lane_of(y)
            if not lanes or lanes[-1] != lane:
                lanes.append(lane)
        return length, max(costs), lanes


if __name__ == "__main__":
    # python planner.py                      lane 2 centre -> lane 2 centre
    # python planner.py --start-y 1.67 --end-y 8.33
    import argparse

    import matplotlib.pyplot as plt

    from car import Car

    ap = argparse.ArgumentParser(description="Cheapest path along the lane costmap.")
    ap.add_argument("--start-y", type=float, default=None,
                    help="start y in metres at x = 0 (default: lane 2 centre)")
    ap.add_argument("--end-y", type=float, default=None,
                    help="end y in metres at the end of the road (default: lane 2 centre)")
    ap.add_argument("--cost-weight", type=float, default=COST_WEIGHT,
                    help=f"how much the costmap matters vs distance (default {COST_WEIGHT:g})")
    args = ap.parse_args()

    cm = LaneCostmap()
    planner = Planner(cm, cost_weight=args.cost_weight)
    path, total = planner.plan(args.start_y, args.end_y)
    length, max_cost, lanes = planner.path_stats(path)

    print(cm)
    print(f"\nPath: ({path[0, 0]:.2f}, {path[0, 1]:.2f}) -> ({path[-1, 0]:.2f}, {path[-1, 1]:.2f})")
    print(f"  length {length:.2f} m | total cost {total:.2f} | max cell cost {max_cost}"
          f" | lanes {' -> '.join(map(str, lanes))}")

    # ---- Plot --------------------------------------------------------------
    fig, ax = plt.subplots(figsize=(13, 4.5))
    im = ax.imshow(cm.grid, cmap="magma_r", origin="lower", extent=cm.extent,
                   vmin=0, vmax=LETHAL, interpolation="nearest")
    for i, by in enumerate(cm.boundaries):
        is_edge = i == 0 or i == len(cm.boundaries) - 1
        ax.axhline(by, color="black" if is_edge else "white",
                   ls="-" if is_edge else "--", lw=2 if is_edge else 1.5)
    for k, cy in enumerate(cm.centers):
        ax.text(1.0, cy, f"Lane {k + 1}", va="center", fontsize=10, weight="bold",
                bbox=dict(facecolor="white", alpha=0.8, lw=0))

    ax.plot(path[:, 0], path[:, 1], "-", color="tab:blue", lw=2.5, label="planned path")
    for (x, y), label in ((path[0], "start"), (path[-1], "end")):
        outline = Car(x=x, y=y).footprint()
        ax.fill(outline[:, 0], outline[:, 1], fc="none", ec="tab:green", lw=1.5)
        ax.plot(x, y, "o", color="tab:green", ms=7)
        ax.text(x, y + 1.3, label, ha="center", fontsize=9, color="tab:green", weight="bold")

    ax.set_xlabel("x: distance along the road (m)")
    ax.set_ylabel("y: across the road (m)")
    ax.set_title(f"Planned path: length {length:.2f} m, total cost {total:.1f} "
                 f"(green boxes = car outline at start and end)")
    ax.set_aspect("equal")
    ax.legend(loc="upper right", fontsize=8)
    fig.colorbar(im, ax=ax, label="cost", fraction=0.02)
    plt.tight_layout()
    plt.savefig("planner_preview.png", dpi=100)
    plt.show()
