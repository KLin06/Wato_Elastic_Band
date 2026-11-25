"""
Animate the car driving along the lane.

Each frame the car calls drive_step(): it moves `--speed` metres along its
elastic band, the band slides forward along the global path to stay
BAND_LENGTH (15 m) long, and the band relaxes for a few steps.

  python drive.py                               lane 2 -> lane 2
  python drive.py --start-y 1.67 --end-y 8.33   lane changes
  python drive.py --end-y 8.33 --save drive.gif save a GIF instead of a window

On screen: grey dashed = global path, blue = elastic band (circles = bubbles),
green line = where the car has driven, green box = the car.
"""
import argparse

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.animation import FuncAnimation, PillowWriter
from matplotlib.collections import PatchCollection
from matplotlib.patches import Circle, Polygon

from car import Car
from lane_costmap import LETHAL, LaneCostmap


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--start-y", type=float, default=None, help="start y (default: lane 2 centre)")
    ap.add_argument("--end-y", type=float, default=None, help="end y (default: lane 2 centre)")
    ap.add_argument("--speed", type=float, default=0.2, help="metres driven per frame (default 0.2)")
    ap.add_argument("--band-steps", type=int, default=3,
                    help="band updates per frame (default 3)")
    ap.add_argument("--save", metavar="FILE.gif", help="save a GIF instead of showing a window")
    args = ap.parse_args()

    car = Car()
    cm = LaneCostmap(car=car)
    car.set_costmap(cm)
    car.plan(args.start_y, args.end_y)
    car.update_band()

    # ---- Static background ---------------------------------------------------
    fig, ax = plt.subplots(figsize=(13, 4.5))
    ax.imshow(cm.grid, cmap="magma_r", origin="lower", extent=cm.extent,
              vmin=0, vmax=LETHAL, interpolation="nearest")
    for i, by in enumerate(cm.boundaries):
        is_edge = i == 0 or i == len(cm.boundaries) - 1
        ax.axhline(by, color="black" if is_edge else "white",
                   ls="-" if is_edge else "--", lw=2 if is_edge else 1.5)
    for k, cy in enumerate(cm.centers):
        ax.text(0.5, cy, f"Lane {k + 1}", va="center", fontsize=9, weight="bold",
                bbox=dict(facecolor="white", alpha=0.8, lw=0))
    gp = car.global_path
    ax.plot(gp[:, 0], gp[:, 1], "--", color="gray", lw=1.5, label="global path")

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
                                     fc="tab:blue", alpha=0.15, ec="none")
        ax.add_collection(bubbles[0])
        lane = cm.lane_of(car.y)
        info.set_text(f"x {car.x:5.1f} m  y {car.y:4.2f} m  lane {lane}  "
                      f"yaw {np.degrees(car.yaw):5.1f} deg  "
                      f"cost here {cm.cost_at(car.x, car.y):3d}"
                      + ("   ARRIVED" if car.done else ""))

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
