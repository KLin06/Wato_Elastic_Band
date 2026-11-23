"""
Elastic band demo (Quinlan & Khatib, 1993).

  python main.py                  watch the band tighten; click to add obstacles
  python main.py --moving         a round obstacle slides across the band
  python main.py --moving --save band.gif     write a GIF instead of opening a window
  python main.py --local          a robot drives the route; the band covers only
                                  the next few metres and slides forward with it
  python main.py --local --moving --lookahead 2.5 --speed 0.04

Pipeline:  costmap.py -> astar.py (route) -> band.py / local_band.py (shape)
           -> visualize.py (animation)
"""
import argparse

import matplotlib.pyplot as plt

from costmap import demo_map
from visualize import BandAnimator, LocalBandAnimator, MovingObstacle


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--moving", action="store_true",
                    help="add an obstacle that slides back and forth across the band")
    ap.add_argument("--start", type=float, nargs=2, default=(1.0, 1.0), metavar=("X", "Y"))
    ap.add_argument("--goal", type=float, nargs=2, default=(9.0, 9.0), metavar=("X", "Y"))
    ap.add_argument("--steps", type=int, default=1,
                    help="band update steps per animation frame (default 1)")
    ap.add_argument("--save", metavar="FILE.gif",
                    help="save a GIF instead of showing a window")
    ap.add_argument("--frames", type=int, default=None,
                    help="frames to save with --save (default 300, or 650 with --local)")
    ap.add_argument("--local", action="store_true",
                    help="robot drives along; band covers only --lookahead metres ahead")
    ap.add_argument("--lookahead", type=float, default=3.0,
                    help="(m) length of the local band window (default 3.0)")
    ap.add_argument("--speed", type=float, default=0.03,
                    help="(m) robot distance per frame in --local mode (default 0.03)")
    args = ap.parse_args()
    if args.frames is None:
        args.frames = 650 if args.local else 300

    cm = demo_map()

    moving = None
    if args.moving:
        # Slides horizontally through the gap right of the tall wall, crossing
        # the band's path between the wall and the goal.
        moving = MovingObstacle(p0=(9.2, 6.5), p1=(6.3, 6.5), radius=0.5, period=300)

    if args.local:
        viz = LocalBandAnimator(cm, tuple(args.start), tuple(args.goal),
                                lookahead=args.lookahead, speed=args.speed,
                                moving=moving, steps_per_frame=args.steps)
    else:
        viz = BandAnimator(cm, tuple(args.start), tuple(args.goal),
                           moving=moving, steps_per_frame=args.steps)

    if args.save:
        print(f"Saving {args.frames} frames to {args.save} ...")
        viz.save_gif(args.save, frames=args.frames)
        print("done")
    else:
        viz.animate()
        plt.show()


if __name__ == "__main__":
    main()
