"""
Draw the A* path on a camera frame (pure numpy + OpenCV, no ROS imports -> testable anywhere).

World frame: x,y ground (m), z up. Rover pose comes from /odom; ground height from the terrain map.
Camera model: pinhole, mounted `cam_forward` m ahead of / `cam_height` m above the rover ground point.
Rover pitch is ESTIMATED from the terrain slope along the heading (roll is ignored).
"""
import math
import numpy as np
import cv2

NEAR = 0.3          # m, clip plane in front of the camera


def intrinsics(w, h, hfov=1.085595, K=None):
    """K = (fx, fy, cx, cy) from CameraInfo if available, else derived from the horizontal FOV."""
    if K is not None:
        return K
    fx = (w / 2.0) / math.tan(hfov / 2.0)
    return (fx, fx, w / 2.0, h / 2.0)


def _to_cam(P, rx, ry, rz, yaw, pitch, cam_forward, cam_height):
    """world (N,3) -> camera-frame (forward, left, up), with pitch compensation."""
    dx, dy = P[:, 0] - rx, P[:, 1] - ry
    f = dx * math.cos(yaw) + dy * math.sin(yaw) - cam_forward
    l = -dx * math.sin(yaw) + dy * math.cos(yaw)
    u = P[:, 2] - (rz + cam_height)
    cp, sp = math.cos(pitch), math.sin(pitch)          # nose-up pitch
    return f * cp + u * sp, l, u * cp - f * sp


def _pix(f, l, u, K):
    fx, fy, cx, cy = K
    return (cx - fx * l / f, cy - fy * u / f)


def _outlined_text(img, text, org, scale=0.5, color=(255, 255, 255)):
    cv2.putText(img, text, org, cv2.FONT_HERSHEY_SIMPLEX, scale, (0, 0, 0), 3, cv2.LINE_AA)
    cv2.putText(img, text, org, cv2.FONT_HERSHEY_SIMPLEX, scale, color, 1, cv2.LINE_AA)


def draw_path_overlay(img, path_xyz, cur_idx, rover_x, rover_y, rover_yaw, height_fn,
                      K, cam_forward=0.064, cam_height=0.11, cam_pitch=0.0,
                      max_range=25.0, hud=None):
    """
    img       : BGR uint8 frame (modified in place and returned)
    path_xyz  : (N,3) array, world x, y, ground z for every waypoint
    cur_idx   : index of the waypoint the rover is currently driving to
    height_fn : callable (x, y) -> ground height (vectorised); used for rover z and pitch
    hud       : list of strings printed top-left
    """
    rz = float(height_fn(rover_x, rover_y))
    d = 0.4
    hf = float(height_fn(rover_x + d * math.cos(rover_yaw), rover_y + d * math.sin(rover_yaw)))
    hb = float(height_fn(rover_x - d * math.cos(rover_yaw), rover_y - d * math.sin(rover_yaw)))
    pitch = math.atan2(hf - hb, 2 * d) + cam_pitch

    # rover's own ground point first, so the line visibly starts at the rover
    start = max(0, cur_idx - 1)
    P = np.vstack([[rover_x, rover_y, rz], path_xyz[start:]])
    reach = np.hypot(P[:, 0] - rover_x, P[:, 1] - rover_y) <= max_range
    if not reach.all():
        P = P[: int(np.argmin(reach))]                  # stop at first point beyond range
    if len(P) >= 2:
        f, l, u = _to_cam(P, rover_x, rover_y, rz, rover_yaw, pitch, cam_forward, cam_height)
        for i in range(len(P) - 1):
            a = np.array([f[i], l[i], u[i]]); b = np.array([f[i + 1], l[i + 1], u[i + 1]])
            if a[0] < NEAR and b[0] < NEAR:
                continue
            if a[0] < NEAR:                              # clip segment at the near plane
                t = (NEAR - a[0]) / (b[0] - a[0]); a = a + t * (b - a)
            elif b[0] < NEAR:
                t = (NEAR - b[0]) / (a[0] - b[0]); b = b + t * (a - b)
            pa, pb = _pix(*a, K), _pix(*b, K)
            thick = int(np.clip(40.0 / max(a[0], 0.5), 2, 10))
            cv2.line(img, (int(pa[0]), int(pa[1])), (int(pb[0]), int(pb[1])),
                     (255, 0, 255), thick, cv2.LINE_AA)
        # waypoint dots (only those in front of the camera)
        for i in range(1, len(P)):
            if f[i] > NEAR:
                px, py = _pix(f[i], l[i], u[i], K)
                cv2.circle(img, (int(px), int(py)), int(np.clip(30.0 / f[i], 2, 7)), (255, 255, 0), -1)

    # current target waypoint (yellow ring) and goal (red)
    for idx, col, label in ((cur_idx, (0, 255, 255), "WP"), (len(path_xyz) - 1, (0, 0, 255), "GOAL")):
        if idx >= len(path_xyz):
            continue
        f1, l1, u1 = _to_cam(path_xyz[idx:idx + 1], rover_x, rover_y, rz, rover_yaw, pitch,
                             cam_forward, cam_height)
        if f1[0] > NEAR and f1[0] <= max_range:
            px, py = _pix(f1[0], l1[0], u1[0], K)
            r = int(np.clip(60.0 / f1[0], 6, 28))
            cv2.circle(img, (int(px), int(py)), r, col, 3, cv2.LINE_AA)
            _outlined_text(img, f"{label} {f1[0]:.1f}m", (int(px) + r + 4, int(py)), 0.5, col)

    lines = list(hud or [])
    lines.append(f"est. pitch {math.degrees(pitch):+.1f} deg  ground z {rz:+.2f} m")
    for i, t in enumerate(lines):
        _outlined_text(img, t, (8, 20 + 20 * i))
    return img
