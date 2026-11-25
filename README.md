# Elastic Band

Elastic band path planning ([Quinlan & Khatib, ICRA 1993](https://khatib.stanford.edu/publications/pdfs/Quinlan_1993_ICRA.pdf)) for WATO.

| Folder | Contents |
|---|---|
| [`demo/`](demo/) | Standalone Python demo: grid costmap, A*, full and local (sliding-window) elastic bands, and an interactive matplotlib animation. See [demo/README.md](demo/README.md). |
| [`lane/`](lane/) | Lane prototype: a car following a lane reference with a sliding elastic band, obstacle avoidance inside the allowed lanes, and lane changes. |

## Lane prototype (`lane/`)

The reference is set from the lane the car is told to use: that lane's
centreline, or a smooth S-curve into the target lane for a lane change,
shifted sideways inside the allowed lanes wherever an obstacle is in the way.
An elastic band covering the next 15 m then follows that reference, pushed
away from obstacles and lane walls and pulled back toward the reference,
sliding forward as the car drives.

```powershell
cd lane
..\.venv\Scripts\python drive.py                                # obstacle at the side of lane 2
..\.venv\Scripts\python drive.py --obstacle-y 5.0               # obstacle blocks lane 2: car stops
..\.venv\Scripts\python drive.py --target-lane 3 --change-x 25  # lane change 2 -> 3
```

| File | Contents |
|---|---|
| `reference.py` | Lane centreline / quintic lane-change reference |
| `car.py` | Point car (2 m × 5 m), allowed-lane corridor, obstacle-clear reference, sliding band, status (`following` / `detour` / `blocked` / `done`) |
| `elastic_band.py` | Elastic band: contraction, obstacle repulsion, pull toward the reference |
| `obstacle.py` | Small obstacle (default: side of lane 2, x = 10 m) |
| `lane_costmap.py` | 3-lane road costmap (obstacles grown by the car's size) |
| `drive.py` | Animation: lane walls, obstacle no-go zones, band, car, status |
| `planner.py` | Earlier search planner (`drive.py --planner search`) |

Not done yet: a turning limit (the band can bend tighter than a car can
steer), keeping a detour's passing side when the map changes, and leaving
`blocked` once the lane clears.

## Design: how the band should handle lanes

Two approaches:

- **A. Lanes as obstacles.** Lane lines are obstacles for the band. Bubbles
  and repulsion keep the car between them, and it settles where the pushes
  from both lines balance. This is the paper's model, unchanged.
- **B. Lanes in a costmap.** Lanes are graded costs (centring, line
  proximity, out-of-lane). Only true hazards (road edges, objects) are
  obstacles. The band follows the costs through an extra cost-gradient force.

| | A. Lanes as obstacles | B. Lanes in a costmap |
|---|---|---|
| Staying in lane | Guaranteed: the band can't cross an obstacle | Preferred, not guaranteed: depends on tuning |
| Lane changes | Need the decision layer to remove the line between the two lanes for the manoeuvre (the lane-change decision comes from there in both approaches) | Natural: crossing a dashed line costs more but is allowed |
| Expressiveness | Binary: lane or no lane | Graded: centre vs. edge, dashed vs. solid, obstacle buffers in one map |
| Narrow lanes | The car's centre has only ±0.67 m of room (3.33 m lane, 2 m car), so bubbles are small: many bubbles, small steps, slow to settle | Bubble size depends only on hard obstacles, so bubbles stay large |
| Sensor noise | A gap or jitter in a detected line can break the band | Noise only shifts costs slightly |
| Tuning | Few knobs (`rho0`, `k_r`) | `k_cost` and the blur trade straddling time against crossing angle (measured: see `costmap_modification` branch) |
| Fit with WATO | Needs a separate lane-to-obstacle conversion | Matches the existing `costmap` node (`OccupancyGrid`, 0–100 layers): lanes become one more layer |
| Paper fidelity | Exact | Adds a force the paper doesn't have |

**Implemented: A, with lane changes authorised by the decision layer.**
The lane walls are a corridor of allowed lanes; a lane change removes the
wall between the two lanes for the manoeuvre. This guarantees the car stays
in its allowed lanes, makes a blocked lane an explicit status, and keeps the
decision about crossing with the decision layer. A reference pull (instead of
graded lane costs) centres the band. Approach B is kept on the
`costmap_modification` branch.

Measurements for B (`costmap_modification`, 50 m drives):
- Without the cost force, the band ignores lanes below the obstacle
  threshold: 13.0 m spent straddling lines on a Lane 2 → 3 change.
- With it, that drops to 5.8 m, at the cost of a steeper crossing (25°).

Open issue for both approaches: the band has no vehicle kinematics, so
curvature or heading limits must be added separately.
