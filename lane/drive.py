"""
Animate the car driving along the lane.

Each frame the car calls drive_step(): it moves `--speed` metres along its
elastic band, the band slides forward along the global path to stay
BAND_LENGTH (15 m) long, and the band relaxes for a few steps.

Default: REFERENCE LINE mode (reference.py): the global path is the lane
centreline (optionally blending into another lane), and the band is pulled
toward it while being pushed off obstacles.

  python drive.py                                   lane 2, obstacle at the side of lane 2
  python drive.py --obstacle-y 5.0                  obstacle in the middle of lane 2
  python drive.py --target-lane 3 --change-x 15     lane change 2 -> 3
  python drive.py --planner search --end-y 8.33     old search planner instead
  python drive.py --save drive.gif                  save a GIF instead of a window

The car may only use the lanes the corridor allows (its lane, plus the target
lane during a lane change). If nothing inside the corridor clears an
obstacle, the car stops before it and reports BLOCKED.

On screen (plain road, no cost colours): grey dashed = global path, cyan
dotted = reference the band is pulled toward (global path shifted clear of
obstacles), orange lines = lane walls for the car centre (1 m inside the
lane lines the car may not cross), pink dashed = walls removed for a lane
change, blue = elastic band (circles =
bubbles), green line = where the car has driven, green box = the car, red
box = the obstacle, orange box around it = its no-go zone for the car centre
(a wall: where the car would touch it), red dashed line = blocked point.
"""
import argparse
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.animation import FuncAnimation, PillowWriter
from matplotlib.collections import LineCollection, PatchCollection
from matplotlib.patches import Circle, Polygon

from car import Car
from lane_costmap import LaneCostmap
from obstacle import Obstacle


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--planner", choices=["reference", "search"], default="reference",
                    help="reference line (default) or the search planner (planner.py)")
    ap.add_argument("--lane", type=int, default=2, help="reference mode: start lane (default 2)")
    ap.add_argument("--target-lane", type=int, default=None,
                    help="reference mode: lane to change into (default: stay)")
    ap.add_argument("--change-x", type=float, default=15.0,
                    help="reference mode: x where the lane change starts (default 15)")
    ap.add_argument("--blend", type=float, default=20.0,
                    help="reference mode: lane-change length in m (default 20)")
    ap.add_argument("--start-y", type=float, default=None, help="search mode: start y (default: lane 2 centre)")
    ap.add_argument("--end-y", type=float, default=None, help="search mode: end y (default: lane 2 centre)")
    ap.add_argument("--obstacle-x", type=float, default=None, help="obstacle x (default 10)")
    ap.add_argument("--obstacle-y", type=float, default=None,
                    help="obstacle y (default 3.63: right side of lane 2)")
    ap.add_argument("--speed", type=float, default=0.2, help="metres driven per frame (default 0.2)")
    ap.add_argument("--band-steps", type=int, default=3,
                    help="band updates per frame (default 3)")
    ap.add_argument("--save", metavar="FILE.gif", help="save a GIF instead of showing a window")
    ap.add_argument("--no-obstacle", action="store_true",
                    help="drive on an empty road (default: one obstacle in lane 2)")
    args = ap.parse_args()

    car = Car()
    obs = Obstacle()
    if args.obstacle_x is not None:
        obs.x = args.obstacle_x
    if args.obstacle_y is not None:
        obs.y = args.obstacle_y
    obstacles = [] if args.no_obstacle else [obs]
    cm = LaneCostmap(car=car, obstacles=obstacles)
    car.set_costmap(cm)
    if args.planner == "reference":
        car.plan_reference(args.lane, args.target_lane, args.change_x, args.blend)
    else:
        car.plan(args.start_y, args.end_y)
    car.update_band()

    # ---- Static background: plain road, lane markings, obstacles -------------
    fig, ax = plt.subplots(figsize=(13, 4.5))
    ax.set_facecolor("#b0b0b0")                       # road surface
    for i, by in enumerate(cm.boundaries):
        is_edge = i == 0 or i == len(cm.boundaries) - 1
        ax.axhline(by, color="black" if is_edge else "white",
                   ls="-" if is_edge else "--", lw=2 if is_edge else 1.5)
    for k, cy in enumerate(cm.centers):
        ax.text(0.5, cy, f"Lane {k + 1}", va="center", fontsize=9, weight="bold",
                bbox=dict(facecolor="white", alpha=0.8, lw=0))
    # Obstacles: the red box is the obstacle itself. The orange wall around it
    # is its NO-GO ZONE for the car's centre: where the car would touch it,
    # i.e. the obstacle grown by half the car's length (2.5 m) along the road
    # and half its width (1 m) across it. This is what the band stays outside.
    for i, obs in enumerate(obstacles):
        ax.add_patch(Polygon(obs.footprint(), closed=True, fc="tab:red", ec="black",
                             zorder=6, label="obstacle" if i == 0 else None))
        zone = Obstacle(obs.x, obs.y, obs.length + car.length, obs.width + car.width)
        ax.add_patch(Polygon(zone.footprint(), closed=True, fill=False, ec="tab:orange",
                             lw=4, joinstyle="miter", zorder=4,
                             label="obstacle no-go zone (wall, car centre)" if i == 0 else None))
    gp = car.global_path
    ax.plot(gp[:, 0], gp[:, 1], "--", color="gray", lw=1.5, label="global path")
    ref = car.reference
    if not np.allclose(ref, gp):
        ax.plot(ref[:, 0], ref[:, 1], ":", color="tab:cyan", lw=1.5, label="reference (clear)")

    # Corridor as WALLS for the car's CENTRE (same as the obstacle no-go
    # zones): half a car width (1 m) inside each lane line bounding the
    # allowed lanes, i.e. where the car's side would touch that line (orange).
    # During a lane change the walls on both sides of the line between the
    # source and target lane are removed (pink dashed), which is what lets
    # the car cross it.
    # Animated: up to the band's far end the real corridor is drawn; beyond
    # it, the corridor at the far end is carried forward. So the wall visibly
    # disappears when the band reaches the lane-change point, which is when
    # the band starts using the extra lane.
    def corridor_lanes_at(x):
        """(lowest, highest) allowed lane at road position x."""
        for x0, x1, lanes in car.corridor:
            if x0 <= x < x1:
                return lanes
        return car.corridor[-1][2]

    def visible_pieces():
        """Corridor pieces (x_from, x_to, (lo, hi)) as currently drawn."""
        tail_x = car.band.points[-1, 0]
        pieces = [(x0, min(x1, tail_x), lanes) for x0, x1, lanes in car.corridor if x0 < tail_x]
        pieces.append((tail_x, cm.length, corridor_lanes_at(tail_x)))
        return pieces

    half_car = car.width / 2

    def wall_segments():
        walls, removed = [], []
        for x0, x1, (lo, hi) in visible_pieces():
            # Outer lines: wall 1 m inside each (above the lowest, below the highest).
            for y in (cm.boundaries[lo - 1] + half_car, cm.boundaries[hi] - half_car):
                walls.append([(x0, y), (x1, y)])
            # Lines between allowed lanes: the walls on both sides are removed.
            for k in range(lo, hi):
                for y in (cm.boundaries[k] - half_car, cm.boundaries[k] + half_car):
                    removed.append([(x0, y), (x1, y)])
        return walls, removed

    wall_lc = removed_lc = None
    if car.corridor is not None:
        walls, removed = wall_segments()
        wall_lc = LineCollection(walls, colors="tab:orange", linewidths=5, alpha=0.85,
                                 zorder=3, label="lane wall (car centre)")
        removed_lc = LineCollection(removed, colors="tab:pink", linewidths=4, linestyles="--",
                                    zorder=3, label="wall removed (crossing allowed)")
        ax.add_collection(wall_lc)
        ax.add_collection(removed_lc)
    if car.blocked_idx is not None:
        bx = car.global_path[car.blocked_idx, 0]
        ax.axvline(bx, color="tab:red", ls="--", lw=1.5, label="blocked")

    # ---- Moving parts --------------------------------------------------------
    (trail_line,) = ax.plot([], [], "-", color="tab:green", lw=2, label="driven")
    (band_line,) = ax.plot([], [], "-o", color="tab:blue", ms=3, lw=2,
                           label=f"elastic band ({car.band_length:g} m)")
    car_patch = Polygon(car.footprint(), closed=True, fc="tab:green", ec="black",
                        alpha=0.7, zorder=5)
    ax.add_patch(car_patch)
    bubbles = [None]
    info = ax.text(0.99, 1.02, "", transform=ax.transAxes, ha="right", fontsize=9,
                   family="monospace")

    ax.set_xlim(cm.extent[0], cm.extent[1])
    ax.set_ylim(cm.extent[2], cm.extent[3])
    ax.set_aspect("equal")
    ax.set_xlabel("x: distance along the road (m)")
    ax.set_ylabel("y: across the road (m)")
    ax.set_title("Car driving: global path + sliding 15 m elastic band", loc="left")
    ax.legend(loc="upper right", fontsize=8)
    plt.tight_layout()

    def draw():
        b = car.band
        tr = np.array(car.trail)
        trail_line.set_data(tr[:, 0], tr[:, 1])
        band_line.set_data(b.points[:, 0], b.points[:, 1])
        car_patch.set_xy(car.footprint())
        if bubbles[0] is not None:
            bubbles[0].remove()
        bubbles[0] = PatchCollection([Circle(p, r) for p, r in zip(b.points, b.radii())],
                                     fc=(0.12, 0.47, 0.71, 0.12), ec=(0.12, 0.47, 0.71, 0.8),
                                     lw=0.8, zorder=3)
        ax.add_collection(bubbles[0])
        corridor_text = ""
        if wall_lc is not None:
            walls, removed = wall_segments()
            wall_lc.set_segments(walls)
            removed_lc.set_segments(removed)
            lo, hi = corridor_lanes_at(b.points[-1, 0])
            corridor_text = f"  corridor ahead: lane {lo}" + (f"-{hi}" if hi != lo else "")
        lane = cm.lane_of(car.y)
        info.set_text(f"x {car.x:5.1f} m  y {car.y:4.2f} m  lane {lane}  "
                      f"status {car.status.upper()}" + corridor_text)

    def update(_):
        if not car.done:
            car.drive_step(args.speed, band_steps=args.band_steps)
        draw()

    draw()
    # Enough frames to reach the end, plus a short pause at the end.
    frames = int(np.ceil(cm.length * 1.2 / args.speed)) + 20
    anim = FuncAnimation(fig, update, frames=frames, interval=30, repeat=False,
                         cache_frame_data=False)
    if args.save:
        print(f"Saving {frames} frames to {args.save} ...")
        anim.save(args.save, writer=PillowWriter(fps=30))
        print("done")
    else:
        plt.show()


if __name__ == "__main__":
    main()
