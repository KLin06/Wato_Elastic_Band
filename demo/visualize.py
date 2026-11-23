"""
Step 4: animating the elastic band.

What you see
------------
Left  : the map. Grey dashed = A* path, blue = elastic band, light-blue
        circles = bubbles (free space around each band point).
Right : band length and minimum clearance after every update step.

Each animation frame:
  1. rebuild the costmap if anything changed (moving obstacle, mouse click)
  2. run `steps_per_frame` band updates (ElasticBand.step)
  3. if the band broke (an obstacle cut through it), run A* again and
     start a new band from that path: the "global replan" fallback
  4. redraw

Controls (interactive window only)
----------------------------------
  left click   add a round obstacle where you click
  right click  remove the clicked obstacle nearest to the mouse
  space        pause / resume
  r            remove all clicked obstacles and replan from scratch
"""
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, PillowWriter
from matplotlib.collections import PatchCollection
from matplotlib.patches import Circle

from astar import astar
from band import ElasticBand
from costmap import Costmap
from local_band import LocalElasticBand


class MovingObstacle:
    """A round obstacle that slides back and forth between p0 and p1.

    period : frames for one full trip p0 -> p1 -> p0
    """

    def __init__(self, p0, p1, radius=0.6, period=300):
        self.p0, self.p1 = np.array(p0, float), np.array(p1, float)
        self.radius, self.period = radius, period

    def position(self, frame):
        # Triangle wave from 0 to 1 and back, so the motion ping-pongs.
        phase = (frame % self.period) / self.period          # 0 .. 1
        s = 1 - abs(2 * phase - 1)                           # 0 -> 1 -> 0
        return self.p0 + s * (self.p1 - self.p0)


class BandAnimator:
    def __init__(self, costmap: Costmap, start, goal, moving=None,
                 steps_per_frame=1, band_kwargs=None, click_radius=0.4):
        self.cm = costmap
        self.base_grid = costmap.grid.copy()     # the map without dynamic obstacles
        self.start, self.goal = start, goal
        self.moving = moving                     # MovingObstacle or None
        self.steps_per_frame = steps_per_frame
        self.band_kwargs = band_kwargs or {}
        self.click_radius = click_radius
        self.clicked = []                        # obstacles added by clicking: (x, y, r)

        self.frame = 0
        self.paused = False
        self.map_dirty = False                   # True when a click changed the map
        self.replans = 0
        self.status = ""
        self.history = {"iter": [], "length": [], "clearance": []}

        self.rebuild_map()
        self.replan()
        self._setup_figure()

    # ======================================================================
    # Map and planning
    # ======================================================================
    def rebuild_map(self):
        """Static map + clicked obstacles + moving obstacle, then refresh rho."""
        self.cm.grid[:] = self.base_grid
        for x, y, r in self.clicked:
            self.cm.add_circle(x, y, r)
        if self.moving:
            cx, cy = self.moving.position(self.frame)
            self.cm.add_circle(cx, cy, self.moving.radius)
        self.cm.update()

    def replan(self):
        """Run A* on the current map and start a fresh band from its path."""
        try:
            path = astar(self.cm, self.start, self.goal)
        except ValueError as e:                  # start or goal is covered
            path, self.status = None, str(e)
        if path is None:
            self.status = self.status or "no path: keeping the old band"
            return
        self.path = path
        self.band = ElasticBand(self.cm, path, **self.band_kwargs)
        self.replans += 1
        self.status = f"A* (re)plan #{self.replans}"

    # ======================================================================
    # Figure
    # ======================================================================
    def _setup_figure(self):
        self.fig = plt.figure(figsize=(11, 6))
        gs = self.fig.add_gridspec(2, 2, width_ratios=[1.6, 1])
        self.ax = self.fig.add_subplot(gs[:, 0])
        self.ax_len = self.fig.add_subplot(gs[0, 1])
        self.ax_clr = self.fig.add_subplot(gs[1, 1], sharex=self.ax_len)

        # --- map panel ---
        self.img = self.ax.imshow(self.cm.grid, cmap="gray_r", origin="lower",
                                  extent=self.cm.extent, vmin=0, vmax=1)
        self.bubbles = None
        (self.astar_line,) = self.ax.plot([], [], "--", color="gray", lw=1.5, label="A* path")
        (self.band_line,) = self.ax.plot([], [], "-o", color="tab:blue", ms=3, lw=2,
                                         label="elastic band")
        self.ax.plot(*self.start, "go", ms=9, label="start")
        self.ax.plot(*self.goal, "r*", ms=13, label="goal")
        self.info = self.ax.text(0.02, 0.98, "", transform=self.ax.transAxes, va="top",
                                 fontsize=8, family="monospace",
                                 bbox=dict(facecolor="white", alpha=0.8, lw=0))
        self.ax.set_xlim(self.cm.extent[0], self.cm.extent[1])
        self.ax.set_ylim(self.cm.extent[2], self.cm.extent[3])
        self.ax.set_aspect("equal")
        self.ax.legend(loc="lower right", fontsize=8)
        self.ax.set_title("Elastic band   (click: add obstacle, space: pause)")

        # --- stats panels ---
        (self.len_line,) = self.ax_len.plot([], [], color="tab:blue")
        self.ax_len.set_ylabel("band length (m)")
        self.ax_len.grid(alpha=0.3)
        (self.clr_line,) = self.ax_clr.plot([], [], color="tab:orange")
        self.ax_clr.axhline(self.band.rho0, ls=":", color="gray", lw=1)
        self.ax_clr.set_ylabel("min clearance (m)")
        self.ax_clr.set_xlabel("frame")
        self.ax_clr.grid(alpha=0.3)
        self.fig.tight_layout()

        # --- interaction ---
        self.fig.canvas.mpl_connect("button_press_event", self.on_click)
        self.fig.canvas.mpl_connect("key_press_event", self.on_key)

    def draw(self):
        b = self.band
        self.img.set_data(self.cm.grid)
        self.astar_line.set_data(*zip(*self.path))
        self.band_line.set_data(b.points[:, 0], b.points[:, 1])

        # Bubbles change in number and size every frame, so rebuild them.
        if self.bubbles is not None:
            self.bubbles.remove()
        circles = [Circle(p, r) for p, r in zip(b.points, b.radii())]
        self.bubbles = PatchCollection(circles, facecolor="tab:blue", alpha=0.12,
                                       edgecolor="tab:blue", linewidth=0.3)
        self.ax.add_collection(self.bubbles)

        self.info.set_text(
            f"frame {self.frame:4d} | {len(b.points):3d} bubbles\n"
            f"length {b.length():5.2f} m | min clearance {b.min_clearance():.2f} m\n"
            f"{self.status}{'   [PAUSED]' if self.paused else ''}")

        h = self.history
        self.len_line.set_data(h["iter"], h["length"])
        self.clr_line.set_data(h["iter"], h["clearance"])
        for ax in (self.ax_len, self.ax_clr):
            ax.relim()
            ax.autoscale_view()

    # ======================================================================
    # Animation loop
    # ======================================================================
    def update(self, _frame_number):
        if not self.paused:
            self.frame += 1
            if self.moving or self.map_dirty:
                self.rebuild_map()
                self.map_dirty = False

            for _ in range(self.steps_per_frame):
                self.band.step()

            if self.band.broken:       # obstacle cut through the band
                self.replan()

            self.history["iter"].append(self.frame)
            self.history["length"].append(self.band.length())
            self.history["clearance"].append(self.band.min_clearance())
        self.draw()

    def animate(self, frames=None, interval=30):
        """frames=None runs forever (interactive). Returns the FuncAnimation."""
        self.anim = FuncAnimation(self.fig, self.update, frames=frames,
                                  interval=interval, blit=False, cache_frame_data=False)
        return self.anim

    def save_gif(self, filename, frames=300, fps=30):
        anim = self.animate(frames=frames)
        anim.save(filename, writer=PillowWriter(fps=fps))

    # ======================================================================
    # Mouse and keyboard
    # ======================================================================
    def on_click(self, event):
        if event.inaxes is not self.ax or event.xdata is None:
            return
        if event.button == 1:                                    # left: add
            self.clicked.append((event.xdata, event.ydata, self.click_radius))
            self.status = "obstacle added"
        elif event.button == 3 and self.clicked:                 # right: remove nearest
            d = [np.hypot(x - event.xdata, y - event.ydata) for x, y, _ in self.clicked]
            self.clicked.pop(int(np.argmin(d)))
            self.status = "obstacle removed"
        self.map_dirty = True

    def on_key(self, event):
        if event.key == " ":
            self.paused = not self.paused
        elif event.key == "r":
            self.clicked.clear()
            self.rebuild_map()
            self.replan()


class LocalBandAnimator(BandAnimator):
    """Animates LocalElasticBand: a robot drives along the route while a short
    band (`lookahead` metres) slides ahead of it.

    Extra on screen: the robot (green dot) and the route it has driven (green line).
    The grey dashed line is still the global A* path. Only the blue window
    ahead of the robot is an elastic band.
    """

    def __init__(self, costmap: Costmap, start, goal, lookahead=3.0, speed=0.03, **kwargs):
        self.lookahead, self.speed = lookahead, speed
        self.trail = [np.array(start, float)]
        super().__init__(costmap, start, goal, **kwargs)

    def replan(self):
        """A* from the robot's CURRENT position, then a fresh local band."""
        here = tuple(self.band.robot) if hasattr(self, "band") else self.start
        try:
            path = astar(self.cm, here, self.goal)
        except ValueError as e:                  # robot or goal is covered
            path, self.status = None, str(e)
        if path is None:
            self.status = self.status or "no path: waiting"
            return
        # A* returns cell centres; start exactly at the robot so it doesn't jump.
        path = [here] + path[1:]
        self.path = path
        self.band = LocalElasticBand(self.cm, path, lookahead=self.lookahead,
                                     speed=self.speed, **self.band_kwargs)
        self.replans += 1
        self.status = f"A* (re)plan #{self.replans}"

    def _setup_figure(self):
        super()._setup_figure()
        (self.trail_line,) = self.ax.plot([], [], "-", color="tab:green", lw=2,
                                          label="driven")
        (self.robot_dot,) = self.ax.plot([], [], "o", color="tab:green", ms=11,
                                         mec="black", zorder=5)
        self.ax.legend(loc="lower right", fontsize=8)
        self.ax.set_title(f"Local elastic band ({self.lookahead:g} m window)   "
                          "click: add obstacle, space: pause")
        self.ax_len.set_ylabel("window length (m)")
        self.ax_clr.set_ylabel("robot clearance (m)")

    def update(self, _frame_number):
        if not self.paused and not self.band.done:
            self.frame += 1
            if self.moving or self.map_dirty:
                self.rebuild_map()
                self.map_dirty = False

            self.band.tick(band_steps=self.steps_per_frame)

            if self.band.broken:
                self.replan()

            self.trail.append(self.band.robot.copy())
            self.history["iter"].append(self.frame)
            self.history["length"].append(self.band.length())
            # Plot the ROBOT's clearance, not the band's minimum: the far end of
            # the window sits on fresh, wall-hugging A* points that the band
            # hasn't pushed off yet, so the band minimum would mislead.
            self.history["clearance"].append(self.cm.rho(self.band.robot))
        if self.band.done:
            self.status = "goal reached"
        self.draw()

    def draw(self):
        super().draw()
        tr = np.array(self.trail)
        self.trail_line.set_data(tr[:, 0], tr[:, 1])
        self.robot_dot.set_data([self.band.robot[0]], [self.band.robot[1]])
        self.info.set_text(self.info.get_text() +
                           f"\nrobot clearance {self.cm.rho(self.band.robot):.2f} m"
                           f" | to goal {self.band.remaining_distance():5.2f} m")
