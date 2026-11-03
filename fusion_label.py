#!/usr/bin/env python3
"""
STEP 2 of camera + lidar fusion: give every lidar dot a LABEL (rock or ground)
using what the CAMERA sees at that spot.  This is the first real "fusion":
the lidar says WHERE, the camera says WHAT.

What it does, in plain English:
  1-4. Same as before: turn each lidar beam into a pixel in the camera image.
  5.   Look at a small square of camera pixels around that dot.
  6.   Decide: is that patch ROCK-coloured or GROUND-coloured?
  7.   Draw the dot in a colour that shows the label (not the distance any more):
         RED    = ROCK    (grey-brown, dull colour)
         BLUE   = GROUND  (vivid rusty-orange terrain)
         YELLOW = UNSURE  (grey but bright, probably sky = dot is off the object)

!! The colour test only works in THIS simulation (grey rocks, orange sand).
   It is a stand-in so we can build and test the whole pipeline now.
   Later a real detector replaces the single function classify() and
   nothing else has to change.

How to run:
  python3 fusion_label.py --ros-args -p use_sim_time:=true
  Then in rqt_image_view pick  /fusion_debug/labeled/compressed
  (or set SHOW_WINDOW = True below to get a pop-up window instead)
"""

import math

import cv2
import numpy as np
import rclpy
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data
import message_filters
from sensor_msgs.msg import CompressedImage, LaserScan

# ---------------------------------------------------------------------------
# NUMBERS FROM YOUR WORLD FILE (same as step 1)
# ---------------------------------------------------------------------------
CAM_X = 0.3325    # camera is 0.33 m in front of the lidar
CAM_Y = -0.1175   # negative = camera is to the RIGHT of the lidar
CAM_Z = -0.035    # camera is 3.5 cm below the lidar
FX = 462.2663     # focal length in pixels
FY = 462.2663
CX = 320.5        # image centre column
CY = 240.5        # image centre row
IMG_W = 640
IMG_H = 480
NEAR_CLIP = 0.25  # camera cannot see closer than this

# ---------------------------------------------------------------------------
# THE "IS IT A ROCK?" TEST  (sim-only colour trick, tune these if needed)
# ---------------------------------------------------------------------------
PATCH = 5            # looks at an 11x11 pixel square around each dot
SAT_ROCK_MAX = 100   # saturation 0-255. Rocks are dull (low), the sand is vivid (high)
VAL_SKY_MIN = 140    # brightness 0-255. Dull AND bright = sky, not a rock

ROCK, GROUND, UNSURE = 'ROCK', 'GROUND', 'UNSURE'
COLORS = {                       # OpenCV colours are (Blue, Green, Red)
    ROCK: (0, 0, 255),           # red
    GROUND: (255, 120, 0),       # blue
    UNSURE: (0, 255, 255),       # yellow
}

# How to look at the result
SHOW_WINDOW = True         # True = pop-up window from this node
PUBLISH_COMPRESSED = True   # True = publish /fusion_debug/labeled/compressed


def project_beam(r, angle):
    """Lidar beam (range r, angle) -> pixel (u, v) in the camera image.
    Returns None if the point is not visible to the camera."""
    # point the beam hit, in the lidar frame (x forward, y left, z up)
    px = r * math.cos(angle)
    py = r * math.sin(angle)
    pz = 0.0                      # a 2D lidar only sees one flat slice

    # re-measure it from the CAMERA's position
    dx = px - CAM_X
    dy = py - CAM_Y
    dz = pz - CAM_Z

    # camera "optical" axes: x = right, y = down, z = forward
    xc = -dy
    yc = -dz
    zc = dx
    if zc < NEAR_CLIP:            # behind or too close to the camera
        return None

    # pinhole camera formula: 3D point -> pixel
    u = int(round(FX * xc / zc + CX))
    v = int(round(FY * yc / zc + CY))
    if not (0 <= u < IMG_W and 0 <= v < IMG_H):
        return None
    return u, v


def classify(hsv, u, v):
    """Look at the camera pixels around (u, v) and decide ROCK / GROUND / UNSURE.
    hsv is the camera image converted to Hue-Saturation-Value.
    Returns (label, (h, s, val)) - the numbers are handy for tuning."""
    y0, y1 = max(v - PATCH, 0), min(v + PATCH + 1, IMG_H)
    x0, x1 = max(u - PATCH, 0), min(u + PATCH + 1, IMG_W)
    patch = hsv[y0:y1, x0:x1].reshape(-1, 3)
    h, s, val = np.median(patch, axis=0)   # median = ignores a few odd pixels

    if s >= SAT_ROCK_MAX:
        label = GROUND            # vivid orange sand
    elif val >= VAL_SKY_MIN:
        label = UNSURE            # dull but bright = sky (dot missed the object)
    else:
        label = ROCK              # dull and dark = rock
    return label, (int(h), int(s), int(val))


class FusionLabel(Node):
    def __init__(self):
        super().__init__('fusion_label')

        scan_sub = message_filters.Subscriber(
            self, LaserScan, '/scan', qos_profile=qos_profile_sensor_data)
        img_sub = message_filters.Subscriber(
            self, CompressedImage, '/camera/image_raw/compressed',
            qos_profile=qos_profile_sensor_data)

        # pairs each 5 Hz scan with the closest 15 Hz camera frame
        self.sync = message_filters.ApproximateTimeSynchronizer(
            [scan_sub, img_sub], queue_size=10, slop=0.1)
        self.sync.registerCallback(self.on_pair)

        self.pub = None
        if PUBLISH_COMPRESSED:
            self.pub = self.create_publisher(
                CompressedImage, '/fusion_debug/labeled/compressed', 1)

        # Remembers the label of every beam from the latest scan:
        # {beam number: 'ROCK' / 'GROUND' / 'UNSURE'}.
        # The next step (filtering the scan) will use this.
        self.last_labels = {}
        self.get_logger().info('fusion_label ready, waiting for /scan + camera...')

    def on_pair(self, scan: LaserScan, img_msg: CompressedImage):
        # decode the JPEG from the camera
        frame = cv2.imdecode(
            np.frombuffer(img_msg.data, np.uint8), cv2.IMREAD_COLOR)
        if frame is None:
            self.get_logger().warn('could not decode camera frame')
            return

        # Convert to HSV BEFORE drawing, so our dots do not pollute the colours
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

        counts = {ROCK: 0, GROUND: 0, UNSURE: 0}
        labels = {}
        tune_text = ''
        best_gap = 10 ** 9   # used to find the dot closest to the image centre

        for i, r in enumerate(scan.ranges):
            # skip beams that hit nothing or are out of range
            if not math.isfinite(r) or r < scan.range_min or r > scan.range_max:
                continue

            angle = scan.angle_min + i * scan.angle_increment
            pixel = project_beam(r, angle)
            if pixel is None:
                continue                       # outside the camera's view
            u, v = pixel

            label, (h, s, val) = classify(hsv, u, v)
            labels[i] = label
            counts[label] += 1

            # draw: colour = label
            color = COLORS[label]
            cv2.circle(frame, (u, v), 5, color, -1)
            cv2.line(frame, (u, v - 15), (u, v + 15), color, 1)

            # remember the HSV numbers of the dot nearest the image centre
            gap = abs(u - CX)
            if gap < best_gap:
                best_gap = gap
                tune_text = f'centre dot  H={h} S={s} V={val}  r={r:.1f} m'

        self.last_labels = labels

        # Status text (each line is in its own label colour, so it is also the legend)
        y = 25
        for name in (ROCK, GROUND, UNSURE):
            cv2.putText(frame, f'{name}: {counts[name]}', (10, y),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, COLORS[name], 2)
            y += 28
        if tune_text:
            cv2.putText(frame, tune_text, (10, IMG_H - 12),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1)

        if SHOW_WINDOW:
            cv2.imshow('fusion_label', frame)
            cv2.waitKey(1)

        if self.pub is not None:
            ok, buf = cv2.imencode('.jpg', frame, [cv2.IMWRITE_JPEG_QUALITY, 70])
            if ok:
                out = CompressedImage()
                out.header = img_msg.header
                out.format = 'jpeg'
                out.data = buf.tobytes()
                self.pub.publish(out)


def main():
    rclpy.init()
    node = FusionLabel()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    node.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()
