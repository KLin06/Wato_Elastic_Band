"""
Step 2: the initial path (A*).

What is this file for?
----------------------
The elastic band does not search for a path. It only DEFORMS an existing one.
Forces can bend the band, but they can never make it jump to the other side of
an obstacle. So something has to choose the route first: which way around each
obstacle. A* on the grid does that.

This is plain shortest-path A*:

    move_cost = step_length      (1 cell straight, sqrt(2) cells diagonal)

The resulting path hugs walls and is blocky (straight and 45 degree segments).
That is fine: pushing it off walls and smoothing it is the elastic band's job.

Heuristic
---------
A* needs an estimate h(n) of the remaining cost that never OVERestimates it.
We use the "octile" distance: the shortest path on an 8-connected grid if
there were no obstacles. Obstacles only make the real path longer, so this
never overestimates, and A* returns the shortest path.
"""
import heapq
import math

import numpy as np

from costmap import Costmap, OBSTACLE

# The 8 neighbours of a cell, as (d_row, d_col, step_length_in_cells)
NEIGHBOURS = [
    (-1, 0, 1.0), (1, 0, 1.0), (0, -1, 1.0), (0, 1, 1.0),                      # straight
    (-1, -1, math.sqrt(2)), (-1, 1, math.sqrt(2)),
    (1, -1, math.sqrt(2)), (1, 1, math.sqrt(2)),                               # diagonal
]


def octile(a, b):
    """Shortest 8-connected grid distance from cell a to cell b with no obstacles:
    go diagonally while both rows and columns differ, then straight."""
    dr, dc = abs(a[0] - b[0]), abs(a[1] - b[1])
    return (dr + dc) + (math.sqrt(2) - 2) * min(dr, dc)


def astar(costmap: Costmap, start, goal):
    """Find the shortest path from `start` to `goal`, both world points (x, y) in metres.

    Returns a list of world points (cell centres) from start to goal,
    or None if no path exists.
    """
    grid = costmap.grid
    h, w = grid.shape
    res = costmap.resolution

    # World (x, y) -> grid cell (row, col)
    def to_cell(p):
        return (int(p[1] / res), int(p[0] / res))

    # Grid cell (row, col) -> world point at the cell centre
    def to_world(cell):
        return ((cell[1] + 0.5) * res, (cell[0] + 0.5) * res)

    s, g = to_cell(start), to_cell(goal)
    for name, c in (("start", s), ("goal", g)):
        if not (0 <= c[0] < h and 0 <= c[1] < w) or grid[c] == OBSTACLE:
            raise ValueError(f"{name} {to_world(c)} is outside the map or in an obstacle")

    # g_cost[cell] = shortest known distance from start to that cell
    g_cost = {s: 0.0}
    # came_from[cell] = the cell we reached it from (to rebuild the path at the end)
    came_from = {}
    # Priority queue of (f = g + h, cell). heapq always pops the smallest f.
    open_heap = [(octile(s, g), s)]
    closed = set()   # cells whose shortest distance is final

    while open_heap:
        _, cur = heapq.heappop(open_heap)
        if cur in closed:
            continue          # stale queue entry; we already found a shorter one
        if cur == g:
            break             # reached the goal
        closed.add(cur)

        for dr, dc, step in NEIGHBOURS:
            nr, nc = cur[0] + dr, cur[1] + dc
            if not (0 <= nr < h and 0 <= nc < w) or grid[nr, nc] == OBSTACLE:
                continue

            # Don't let a diagonal move squeeze between two obstacle cells
            # (corner cutting). Both straight cells beside the diagonal must be free.
            if dr and dc and (grid[cur[0] + dr, cur[1]] == OBSTACLE or
                              grid[cur[0], cur[1] + dc] == OBSTACLE):
                continue

            nxt = (nr, nc)
            new_g = g_cost[cur] + step

            if new_g < g_cost.get(nxt, math.inf):
                g_cost[nxt] = new_g
                came_from[nxt] = cur
                heapq.heappush(open_heap, (new_g + octile(nxt, g), nxt))
    else:
        return None           # queue emptied without reaching the goal

    # Walk back from the goal to the start through came_from.
    cells = [g]
    while cells[-1] != s:
        cells.append(came_from[cells[-1]])
    cells.reverse()
    return [to_world(c) for c in cells]


def path_stats(costmap: Costmap, path):
    """Length (m) and minimum / mean clearance (m) along a path."""
    pts = np.array(path)
    length = float(np.sum(np.linalg.norm(np.diff(pts, axis=0), axis=1)))
    clear = [costmap.rho(p) for p in path]
    return length, min(clear), float(np.mean(clear))


if __name__ == "__main__":
    # Run `python astar.py` to see the A* path in the terminal and as a plot.
    import matplotlib.pyplot as plt
    from costmap import demo_map

    cm = demo_map()
    start, goal = (1.0, 1.0), (9.0, 9.0)
    path = astar(cm, start, goal)

    length, min_c, mean_c = path_stats(cm, path)
    print(f"A* path: length {length:.2f} m | min clearance {min_c:.2f} m"
          f" | mean clearance {mean_c:.2f} m | {len(path)} cells")

    print("\nA* path drawn on the grid (every 2nd cell):")
    cm.print_grid(step=2, path=path)

    # ---- Plot -----------------------------------------------------------
    fig, ax = plt.subplots(figsize=(6, 6))
    ax.imshow(cm.grid, cmap="gray_r", origin="lower", extent=cm.extent)
    xs, ys = zip(*path)
    ax.plot(xs, ys, "-", lw=2, label="A* path")
    ax.plot(*start, "go", ms=9, label="start")
    ax.plot(*goal, "r*", ms=13, label="goal")
    ax.set_title("Plain A* (shortest path)")
    ax.legend(loc="lower right", fontsize=8)
    plt.tight_layout()
    plt.savefig("astar_preview.png", dpi=100)
    plt.show()
