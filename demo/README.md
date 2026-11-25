# Elastic Band Path Planning

A 2D Python demo of the elastic band algorithm from
[Quinlan & Khatib, *Elastic Bands: Connecting Path Planning and Control* (ICRA 1993)](https://khatib.stanford.edu/publications/pdfs/Quinlan_1993_ICRA.pdf).

A* picks a route through a grid costmap. The elastic band then reshapes that
route like a stretched rubber band: it pulls the path tight and pushes it away
from obstacles. When the map changes, the band bends in real time, and A* runs
again only if an obstacle cuts the band.

## Setup

Requires Python 3.10+. The virtual environment and `requirements.txt` live in
the `elastic_band/` folder (one level up) and are shared with `lane/`.

```powershell
# from elastic_band/
python -m venv .venv
.venv\Scripts\Activate.ps1          # macOS/Linux: source .venv/bin/activate
pip install -r requirements.txt
cd demo                             # run the commands below from demo/
```

## Running

```powershell
python main.py                                  # watch the band tighten; click to add obstacles
python main.py --moving                         # an obstacle slides across the band
python main.py --moving --save band.gif         # save a GIF instead of opening a window
python main.py --local                          # a robot drives; the band covers only the next 3 m
python main.py --local --moving                 # local band + moving obstacle
```

| Option | Default | Meaning |
|---|---|---|
| `--moving` | off | Add an obstacle that slides back and forth across the band |
| `--start X Y` | `1 1` | Start point in metres |
| `--goal X Y` | `9 9` | Goal point in metres |
| `--steps N` | `1` | Band update steps per animation frame (higher settles faster) |
| `--save FILE.gif` | | Write a GIF instead of showing a window |
| `--frames N` | `300` (`650` with `--local`) | Number of frames to save with `--save` |
| `--local` | off | Robot drives the route; the band is a sliding window ahead of it |
| `--lookahead M` | `3.0` | Length of the local band window, in metres |
| `--speed M` | `0.03` | Distance the robot moves per frame in `--local` mode, in metres |

**Controls** (interactive window):

| Input | Action |
|---|---|
| Left click | Add a round obstacle |
| Right click | Remove the clicked obstacle nearest the mouse |
| Space | Pause / resume |
| `r` | Clear clicked obstacles and run A* again |

Each step can also be run on its own:

```powershell
python costmap.py    # print the map and its distance field, and plot them
python astar.py      # print and plot the A* path
python band.py       # run the band to convergence, print stats, and plot
python local_band.py # drive the robot to the goal with the local band, print progress
```

## How it works

```
costmap.py ──▶ astar.py ──▶ band.py / local_band.py ──▶ visualize.py / main.py
 the map       the route     the shape                   the animation
```

### 1. Costmap ([costmap.py](costmap.py))

An occupancy grid (`0` = free, `1` = obstacle), 10 m × 10 m at 0.1 m per
cell. From the grid we compute, once per map change:

- **`rho(p)`**: distance from point `p` to the nearest obstacle
  (`scipy.ndimage.distance_transform_edt`). This is a bubble's radius.
- **`grad_rho(p)`**: the direction that increases `rho` fastest, i.e. directly
  away from the nearest obstacle. The band pushes bubbles along it.

Lookups blend the four nearest cells, so forces change smoothly as bubbles move.
Every public method works in metres.

### 2. A* ([astar.py](astar.py))

Plain shortest-path A* on an 8-connected grid, with no corner cutting. The band
can only *deform* a path: it can never jump to the other side of an obstacle.
So a global planner has to choose the route first. The A* path is blocky and
hugs walls, which is fine, because fixing that is the band's job.

### 3. Elastic band ([band.py](band.py))

The path becomes a chain of **bubbles**: circles of free space around each
point. If neighbouring bubbles overlap, the whole path is collision free.
Each update step, for every bubble except the fixed start and goal:

1. **Contraction** pulls the bubble toward its neighbours, which straightens bends:
   `f_c = k_c · ( unit(b[i-1] − b[i]) + unit(b[i+1] − b[i]) )`
2. **Repulsion** pushes it away from obstacles within `rho0`:
   `f_r = k_r · (rho0 − rho) · ∇rho`
3. **Only the sideways part counts.** The part of the force along the band is
   removed, so bubbles don't slide along the band and bunch up.
4. **Step size scales with the bubble's radius:** `b += alpha · rho · f`. The
   step is also capped at half the distance to the nearest neighbouring bubble,
   so a bubble can't jump past a close neighbour and back. If the new spot
   would be within `rho_min` of an obstacle, the step is halved.

Then the chain is repaired: a bubble is **inserted** where neighbours no longer
overlap, and **deleted** where a bubble's two neighbours already overlap
without it. If the chain can't stay connected, or a bubble ends up inside an
obstacle, the band is marked `broken` and A* runs again.

| Parameter | Default | Effect |
|---|---|---|
| `k_c` | 1.0 | How hard the band tightens |
| `k_r` | 4.0 | How hard obstacles push |
| `rho0` | 0.5 m | Repulsion acts only within this distance |
| `rho_max` | 1.0 m | Cap on bubble radius in open space |
| `rho_min` | 0.05 m | Bubbles may never get closer than this to an obstacle |
| `alpha` | 0.3 | Step size, as a fraction of the bubble radius |
| `insert_overlap` / `delete_overlap` | 0.9 / 0.6 | Thresholds for inserting / deleting bubbles (the gap prevents flip-flopping) |

### 4. Local band ([local_band.py](local_band.py))

`LocalElasticBand` is the moving-window version, closer to how a robot uses
the algorithm (and how ROS's `eband_local_planner` works). Instead of covering
the whole A* path, the band covers only `lookahead` metres ahead of the robot:

```
global A* path:  S ─────────────────────────────────────────── G
local band:            R ═══════════ T
                       robot         tail (a point on the A* path)
```

Every tick:

1. **Advance:** the robot moves `speed` metres along the band, and the part
   behind it is cut off. The robot is always the first bubble. If an
   obstacle covers the next spot, the robot waits.
2. **Extend:** the next A* points are added at the tail until the band is
   `lookahead` long again, so the window slides forward.
3. **Deform:** the normal elastic band step (from `band.py`), with the robot
   and the tail held fixed.

Each tick only updates the bubbles in the window, so the cost depends on
`lookahead`, not on the length of the route. If an obstacle cuts the window,
A* runs again **from the robot's current position**, and a new window starts
from there.

### 5. Animation ([visualize.py](visualize.py))

Each frame: rebuild the map (static obstacles + clicked obstacles + moving
obstacle) → `cm.update()` → `band.step()` → run A* again if the band broke →
redraw. The band only reads `rho` and `grad_rho`, so it reacts to any change
in the map without knowing what moved.

`LocalBandAnimator` does the same with a `LocalElasticBand`, and also draws
the robot (green dot) and the route it has driven (green line). Its plots
show the window length and the **robot's** clearance. The band's minimum
clearance would be misleading here, because the far end of the window sits
on fresh A* points that the band hasn't pushed off the walls yet.

## Results on the demo map

Start `(1, 1)`, goal `(9, 9)`:

| | Points | Length | Closest to an obstacle |
|---|---|---|---|
| A* path | 138 cells | 16.39 m | 0.10 m |
| Full elastic band (shape settles in ~20 steps) | 28 bubbles | 16.68 m | 0.35 m |
| Local band (3 m window): route the robot drove | 548 ticks | 16.44 m | 0.27 m |

The full band adds about 0.3 m of length in exchange for more than three times
the clearance, and smooths A*'s right-angle corners into curves. The local
band drives a similar route while only ever updating about 3 m of band.

## Files

| File | Contents |
|---|---|
| `costmap.py` | `Costmap` class (grid, `rho`, `grad_rho`, printing) and `demo_map()` |
| `astar.py` | `astar()` and `path_stats()` |
| `band.py` | `ElasticBand` class |
| `local_band.py` | `LocalElasticBand` class (sliding window that moves with the robot) |
| `visualize.py` | `BandAnimator`, `LocalBandAnimator` and `MovingObstacle` |
| `main.py` | Command-line entry point |
| `../requirements.txt` | Pinned `numpy`, `scipy`, `matplotlib` (shared, in `elastic_band/`) |
