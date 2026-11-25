# Elastic Band

Elastic band path planning ([Quinlan & Khatib, ICRA 1993](https://khatib.stanford.edu/publications/pdfs/Quinlan_1993_ICRA.pdf)) for WATO.

| Folder | Contents |
|---|---|
| [`demo/`](demo/) | Standalone Python demo: grid costmap, A*, full and local (sliding-window) elastic bands, and an interactive matplotlib animation. See [demo/README.md](demo/README.md). |

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
| Lane changes | Impossible: the band can't move across an obstacle, so it needs a replan *and* removing the line obstacles during the manoeuvre | Natural: crossing a dashed line costs more but is allowed |
| Expressiveness | Binary: lane or no lane | Graded: centre vs. edge, dashed vs. solid, obstacle buffers in one map |
| Narrow lanes | The car's centre has only ±0.67 m of room (3.33 m lane, 2 m car), so bubbles are small: many bubbles, small steps, slow to settle | Bubble size depends only on hard obstacles, so bubbles stay large |
| Sensor noise | A gap or jitter in a detected line can break the band | Noise only shifts costs slightly |
| Tuning | Few knobs (`rho0`, `k_r`) | `k_cost` and the blur trade straddling time against crossing angle (measured: see `costmap_modification` branch) |
| Fit with WATO | Needs a separate lane-to-obstacle conversion | Matches the existing `costmap` node (`OccupancyGrid`, 0–100 layers): lanes become one more layer |
| Paper fidelity | Exact | Adds a force the paper doesn't have |

**Choice: B, with hard constraints only where crossing is illegal.**
Lane changes are a core manoeuvre, and A rules them out structurally. B
reuses WATO's costmap and can still be strict where it matters. Road edges
are lethal today; `SOLID_SOLID` lane boundaries (from `lanelet_msgs/Lanelet`)
should be too.

What the measurements show (`lane/`, 50 m drives):
- Without the cost force, B ignores lanes below the obstacle threshold:
  13.0 m spent straddling lines on a Lane 2 → 3 change.
- With it (`costmap_modification`), that drops to 5.8 m, at the cost of a
  steeper crossing (25°).
- Passing a 0.5 m obstacle inside Lane 2: the car shifts 0.35 m, keeps a
  0.46 m gap, and never leaves the lane.

Open issue for both approaches: the band has no vehicle kinematics, so
curvature or heading limits must be added separately.
