#!/usr/bin/env python3
import os
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import LaserScan
from nav_msgs.msg import Odometry
from geometry_msgs.msg import Twist
import numpy as np
import cv2
import csv
import matplotlib.pyplot as plt
import heapq
import math
from camera_overlay import draw_path_overlay, intrinsics
import time
from sensor_msgs.msg import Image, CameraInfo
import tf2_ros
from rclpy.time import Time
 
from dtm_planner import load_dtm, build_costmap, plan, CostMap, plot
 
class LidarDetector(Node):
    @property
    def state(self):
        """Current controller state. The exact value is also used in the HUD and trail CSV."""
        return self._state

    @state.setter
    def state(self, value):
        """Record every real state transition, including transitions added in future code."""
        new_state = str(value)
        old_state = getattr(self, '_state', None)
        self._state = new_state

        # During __init__, the CSV writer does not exist yet. The first odometry
        # callback will write the initial state. Once open, every state change is
        # written immediately, even if the rover has not moved 0.5 m.
        if old_state is None or old_state == new_state:
            return
        if not hasattr(self, 'trail_writer') or not hasattr(self, 'trail_file'):
            return
        if self.trail_file.closed:
            return

        self._write_trail_row(state=new_state, flush=True)
        self.get_logger().info(f'STATE CHANGE: {old_state} -> {new_state}')

    def _write_trail_row(self, state=None, flush=False):
        """Write one pose/state sample without changing the distance sampling anchor."""
        trail_file = getattr(self, 'trail_file', None)
        trail_writer = getattr(self, 'trail_writer', None)
        if trail_file is None or trail_writer is None or trail_file.closed:
            return

        # Use the exact same state string that camera_callback displays.
        state_value = self.state if state is None else str(state)
        trail_writer.writerow([
            round(float(self.rover_x), 2),
            round(float(self.rover_y), 2),
            state_value,
        ])
        self._trail_n = getattr(self, '_trail_n', 0) + 1
        if flush or self._trail_n % 20 == 0:
            trail_file.flush()

    def __init__(self):
        super().__init__('lidar_detector')
        
        self.rover_x = 0.0
        self.rover_y = 0.0
        self.rover_yaw = 0.0
        
        # ================= robhar_2 config (override with --ros-args -p name:=value) =================
        P = lambda name, default: self.declare_parameter(name, default).value
        odom_topic = P('odom_topic', '/odom')
        scan_topic = P('scan_topic', '/scan')        # LEVEL lidar topic - check `ros2 topic list`
        cmd_topic = P('cmd_vel_topic', '/cmd_vel')
        self.base_frame = P('base_frame', 'base_footprint')
        self.camera_frame_param = P('camera_frame', '')    # '' -> use frame_id of the incoming image
        self.use_tf_camera_mount = P('use_tf_camera_mount', True)

        # where the rover spawned in the world (odom starts at 0,0,0 at the spawn point)
        self.spawn_x = P('spawn_x', 0.0)
        self.spawn_y = P('spawn_y', 0.0)
        self.spawn_yaw = P('spawn_yaw', 0.0)

        # chassis outline, measured FROM THE LEVEL-LIDAR AXIS to the outer edge of body + wheels (m)
        self.body_front = P('body_front', 0.385)
        self.body_back = P('body_back', 0.385)
        self.body_hw = P('body_half_width', 0.33)
        self.SELF_MARGIN = P('self_margin', 0.03)          # returns inside outline + this = rover's own body
        robot_radius = P('robot_radius', 0.0)              # 0 -> auto (circumscribed circle + 5 cm)

        # drive limits (4-wheel skid steer) - starting guesses, tune in sim
        self.MIN_SPEED = P('min_speed', 0.25)
        self.MAX_SPEED = P('max_speed', 0.50)
        self.MAX_ANGULAR = P('max_angular', 0.75)
        self.STEER_GAIN = P('steer_gain', 1.0)
        self.VFH_BIN_DEG = P('vfh_bin_deg', 5.0)           # histogram bin width, independent of beam count

        self.cam_frame = None
        self.cam_size = None
        self._cam_mount = None
        self._fp = None
        self._fp_key = None
        self._min_gap = float('inf')
        self._min_rear_gap = float('inf')
        self.tf_buffer = tf2_ros.Buffer()
        self.tf_listener = tf2_ros.TransformListener(self.tf_buffer, self)
        # ===========================================================================================
        
        self.ROBOT_RADIUS = robot_radius if robot_radius > 0 else \
            math.hypot(max(self.body_front, self.body_back), self.body_hw) + 0.05   # inflate obstacles by this
        
        self.odom_sub = self.create_subscription(
            Odometry, odom_topic, self.odom_callback, 10)
        self.lidar_sub = self.create_subscription(
            LaserScan, scan_topic, self.lidar_callback, 10)
        
        self.pub = self.create_publisher(Twist, cmd_topic, 10)
 
        # ---- camera window with the A* path drawn on it ----
        self.declare_parameter('camera_topic', '/camera/image_raw')
        self.declare_parameter('camera_info_topic', '/camera/camera_info')
        self.declare_parameter('cam_hfov', 1.085595)     # rad, used if no CameraInfo arrives
        self.declare_parameter('cam_forward', 0.064)     # m ahead of base
        self.declare_parameter('cam_height', 0.11)       # m above ground
        self.declare_parameter('cam_pitch', 0.0)         # rad, fixed tilt (+ = looking up)
        self.cam_K = None
        self._ov_path = None
        self._ov_xyz = None
        self.create_subscription(Image, self.get_parameter('camera_topic').value, self.camera_callback, 1)
        self.create_subscription(CameraInfo, self.get_parameter('camera_info_topic').value, self.camera_info_callback, 1)
        cv2.namedWindow('Camera + A* Path', cv2.WINDOW_NORMAL)
        cv2.resizeWindow('Camera + A* Path', 640, 480)
        
        # Declare parameters for DTM file and planning thresholds
        self.declare_parameter('dtm_path', 'mars.tif')  # Path to your .tif, .npz, or .obj file
        self.declare_parameter('world_file', '')        # Optional .world file for .obj meshes
        self.declare_parameter('start_x', 0.0)
        self.declare_parameter('start_y', 0.0)
        self.declare_parameter('goal_x', -20.0)
        self.declare_parameter('goal_y', 40.0)
        self.declare_parameter('costmap_res', 0.5)
        
        dtm_file = self.get_parameter('dtm_path').value
        world_file = self.get_parameter('world_file').value or None
        
        # 1. Load DTM and build/load CostMap

        self.get_logger().info(f"Loading DTM from {dtm_file}...")
        xs, ys, H = load_dtm(dtm_file, world=world_file)
        res = self.get_parameter('costmap_res').value
        self.cm = build_costmap(xs, ys, H, res=res, rover_r=self.ROBOT_RADIUS)

        start = (self.get_parameter('start_x').value, self.get_parameter('start_y').value)
        goal = (self.get_parameter('goal_x').value, self.get_parameter('goal_y').value)

        # 2. Plan global path
        self.path = plan(self.cm, start, goal)
        if self.path is None:
            raise RuntimeError('No path found from start to goal using dtm_planner!')
        self.path = self._densify(self.path)
        
        self.get_logger().info(f"Global path planned successfully with {len(self.path)} waypoints.")
        
        self.replan_retry_count = 0
        self.MAX_REPLAN_RETRIES = 10
        self.waypoint_idx = 0
        self._p_xy = (0.0, 0.0)
        self._p_t = time.monotonic()
        self.STUCK_WINDOW_S = 6.0     # seconds with no real movement
        self.STUCK_MIN_MOVE = 0.15    # meters that count as "moving"

        self.emergency_stop = False
        self._unsafe_count = 0
        self.SAFETY_SCANS = 3      # consecutive scans before a stop is confirmed
 
        self.CRITICAL_DISTANCE = 0.4   # m of clearance between the CHASSIS OUTLINE and a return (not the lidar axis)

        # VFH tunables (starting guesses - tune in sim)
        self.VFH_MIN_RANGE = 0.35     # was a hard-coded 0.35 -> dead band between 0.1 and 0.35 m

        self.FRONT_ANGLE = np.radians(90)
 
        # rear cone, used only while backtracking so we don't reverse into something
        self.REAR_CRITICAL_DISTANCE = 0.2
        self.REAR_ANGLE = np.radians(90)

        # ---------------- motion-aware safety zones ----------------
        # all gaps are metres from the CHASSIS OUTLINE (not from the lidar axis).
        # CRITICAL_DISTANCE = FRONT stop gap, REAR_CRITICAL_DISTANCE = REAR stop gap (defined above).
        self.ZONE_LAT_MARGIN = P('zone_lat_margin', 0.10)     # front/rear corridor half-width = body_half_width + this
        self.SIDE_GAP = P('side_gap', 0.25)                   # side wall closer than this -> don't turn toward it
        _spin = P('spin_clear_radius', 0.0)                   # 0 -> auto: swept circle + 0.15 m
        self.SPIN_CLEAR_RADIUS = _spin if _spin > 0 else math.hypot(max(self.body_front, self.body_back), self.body_hw) + 0.15
        _vr = P('vfh_radius', 0.0)                            # 0 -> auto: half width + lateral margin (NOT the spin circle)
        self.VFH_RADIUS = _vr if _vr > 0 else self.body_hw + self.ZONE_LAT_MARGIN
        self.SPIN_V_EPS = 0.03                                # |v| below this + turning = "spinning in place"
        self.ESCAPE_SPEED = P('escape_speed', 0.10)           # m/s, creep speed when we can't pirouette
        self.ESCAPE_ANGULAR = P('escape_angular', 0.30)       # rad/s, arc rate while creeping away
        self.MAX_STOP_CYCLES = 6                              # stops near the same spot before the run is declared FAILED
        self.STOP_PROGRESS_M = 1.0                            # moving farther than this from the anchor resets the count
        self.PIVOT_ENTER = math.radians(P('pivot_enter_deg', 40.0))   # heading error above this -> rotate in place first
        self.PIVOT_EXIT = math.radians(P('pivot_exit_deg', 20.0))     # ...until it drops below this (hysteresis)
        self.VETO_SCANS = int(P('veto_scans', 10))                    # scans of "wanted to move, nothing legal" before BACKTRACK
        self._pivoting = False
        self._cmd_vetoed = False
        self._veto_count = 0
        self._smoothed_angular = 0.0
        self._stop_anchor = None
        self._stop_cycles = 0
        self._blind_checked = False
        self._zone = {'front': float('inf'), 'rear': float('inf'), 'left': float('inf'),
                      'right': float('inf'), 'spin_r': float('inf'), 'spin_x': 0.0, 'spin_y': 0.0}
 
        # backtrack tuning
        # Backtracking configuration
        self.BACKTRACK_WAYPOINTS = 5
        self.BACKTRACK_SPEED = 0.10          # m/s

        self.BACKTRACK_MIN_TIME = 2.0        # minimum duration of manoeuvre
        self.BACKTRACK_MAX_TIME = 15.0      # hard timeout, in seconds

        self.BACKTRACK_WP_TOLERANCE = 0.4    # waypoint capture radius, metres
        self.BACKTRACK_TIMEOUT_FACTOR = 1.5
        self.BACKTRACK_TIMEOUT_MARGIN = 5.0

        # Abort backtracking if the rover cannot make progress
        self.BACKTRACK_STUCK_TIMEOUT = 6.0
        self.BACKTRACK_STUCK_MOVE = 0.10    # metres

        self.backtrack_target_idx = None
        self.backtrack_cursor_idx = None

        self.backtrack_start_time = None
        self.backtrack_start_x = None
        self.backtrack_start_y = None

        self.backtrack_timeout = 0.0
        self.backtrack_last_progress_xy = None
        self.backtrack_last_progress_time = None

        self.replan_attempted = False
        
        self.state = "NORMAL"
        
        # ---- trail recording: planned path + where the rover actually went ----
        with open('planned_path.csv', 'w', newline='') as f:
            w = csv.writer(f); w.writerow(['x', 'y']); w.writerows(self.path)
        self.trail_file = open('rover_trail.csv', 'w', newline='')
        self.trail_writer = csv.writer(self.trail_file)
        self.trail_writer.writerow(['x', 'y', 'state'])
        self._trail_last = None
        self._trail_n = 0
        
        cv2.namedWindow('Lidar View', cv2.WINDOW_NORMAL)
        stamp = time.strftime('%m%d_%H%M%S')
        cv2.resizeWindow('Lidar View', 400, 400)
        self.csv_file = open(
            'vfh_data.csv',
            'w',
            newline=''
        )
 
        self.csv_writer = csv.writer(self.csv_file)
 
        self.csv_writer.writerow([
                'scan_id',
                'rover_x',
                'rover_y',
                'rover_yaw_deg',
                'goal_angle_deg',
                'chosen_angle_deg',
                'center_density',
                'max_density_in_valley',
                'valley_start',
                'valley_width'
            ])
 
        # Number of complete LiDAR histograms to record
        self.max_scans = 5000
 
        self.scan_id = 0
 
        self.recording = True
 
        self.get_logger().info(
            'VFH CSV recording started: vfh_data.csv'
        )
    def _reset_progress(self):
        self._p_xy = (self.rover_x, self.rover_y)
        self._p_t = time.monotonic()

    def _is_stuck(self):
        if math.hypot(self.rover_x - self._p_xy[0], self.rover_y - self._p_xy[1]) > self.STUCK_MIN_MOVE:
            self._reset_progress()
            return False
        return time.monotonic() - self._p_t > self.STUCK_WINDOW_S
        
    def lidar_callback(self, msg):
        # blank black 400x400 image 
        self.preprocess_scan(msg)   # NaN/inf out, rover's own body hits out, clearance measured
        self.vfh_auto(msg)
        img = np.zeros((400, 400, 3), dtype=np.uint8)
        for i, r in enumerate(msg.ranges):
            if r < msg.range_min or r > msg.range_max:
                continue  
            angle = msg.angle_min + i * msg.angle_increment
            x = r * np.cos(angle)
            y = r * np.sin(angle)
            px = 200 + int(x * 40)
            py = 200 - int(y * 40)
            if 0 <= px < 400 and 0 <= py < 400:
                cv2.circle(img, (px, py), 2, (0, 255, 0), -1)
        cv2.rectangle(img, (200 - int(self.body_back * 40), 200 - int(self.body_hw * 40)),
                      (200 + int(self.body_front * 40), 200 + int(self.body_hw * 40)), (255, 128, 0), 1)
        cv2.imshow('Lidar View', img)   
        cv2.waitKey(1)
    
    def camera_info_callback(self, msg):
        self.cam_K = (msg.k[0], msg.k[4], msg.k[2], msg.k[5])
 
    @staticmethod
    def _image_to_bgr(msg):
        ch = {'rgb8': 3, 'bgr8': 3, 'rgba8': 4, 'bgra8': 4, 'mono8': 1}.get(msg.encoding)
        if ch is None:
            return None
        a = np.frombuffer(msg.data, np.uint8).reshape(msg.height, msg.step)[:, :msg.width * ch]
        a = a.reshape(msg.height, msg.width, ch)
        if msg.encoding == 'rgb8':
            a = a[:, :, ::-1]
        elif msg.encoding == 'rgba8':
            a = a[:, :, 2::-1]
        elif msg.encoding == 'bgra8':
            a = a[:, :, :3]
        elif msg.encoding == 'mono8':
            a = cv2.cvtColor(a, cv2.COLOR_GRAY2BGR)
        return np.ascontiguousarray(a)
 
    def camera_callback(self, msg):
        self.cam_size = (msg.width, msg.height)
        self.cam_frame = self.camera_frame_param or msg.header.frame_id
        img = self._image_to_bgr(msg)
        if img is None:
            self.get_logger().warn(f'Unsupported camera encoding: {msg.encoding}', throttle_duration_sec=5.0)
            return
        path = getattr(self, 'path', None)
        if path:
            if self._ov_path is not path:               # path changed (replan) -> rebuild cache
                xy = np.asarray(path, dtype=float)
                
                heights = []
                for x, y in xy:
                    r, c = self.cm.w2g(x, y)
                    r = min(max(r, 0), self.cm.Z.shape[0] - 1)
                    c = min(max(c, 0), self.cm.Z.shape[1] - 1)
                    heights.append(float(self.cm.Z[r, c]))
                
                self._ov_xyz = np.column_stack([xy, np.array(heights)])
                self._ov_path = path
                
            idx = min(self.waypoint_idx, len(path) - 1)
            gx, gy = self._ov_xyz[-1, :2]
            wx, wy = self._ov_xyz[idx, :2]
            to_wp = math.hypot(wx - self.rover_x, wy - self.rover_y)
            to_goal = math.hypot(gx - self.rover_x, gy - self.rover_y)
            bearing = (math.atan2(wy - self.rover_y, wx - self.rover_x) - self.rover_yaw + math.pi) % (2 * math.pi) - math.pi
            # This is the canonical state string also written to rover_trail.csv.
            state_text = self.state
            hud = [f'{state_text}  WP {idx}/{len(path) - 1}',
                   f'rover ({self.rover_x:+.2f}, {self.rover_y:+.2f}) yaw {math.degrees(self.rover_yaw):+.0f} deg',
                   f'to WP {to_wp:.2f} m  bearing {math.degrees(bearing):+.0f} deg  to goal {to_goal:.1f} m']
            mount = self._get_camera_mount()
            K = intrinsics(msg.width, msg.height, self.get_parameter('cam_hfov').value, self.cam_K)
            
            # Robust height lookup supporting both scalar and vector inputs
            def height_fn(xx, yy):
                if isinstance(xx, (np.ndarray, list, tuple)):
                    res = []
                    for px, py in zip(xx, yy):
                        r, c = self.cm.w2g(px, py)
                        r = min(max(r, 0), self.cm.Z.shape[0] - 1)
                        c = min(max(c, 0), self.cm.Z.shape[1] - 1)
                        res.append(float(self.cm.Z[r, c]))
                    return np.array(res)
                else:
                    r, c = self.cm.w2g(xx, yy)
                    r = min(max(r, 0), self.cm.Z.shape[0] - 1)
                    c = min(max(c, 0), self.cm.Z.shape[1] - 1)
                    return float(self.cm.Z[r, c])

            draw_path_overlay(img, self._ov_xyz, idx, self.rover_x, self.rover_y, self.rover_yaw,
                              height_fn,
                              K,
                              cam_forward=mount[0],
                              cam_height=mount[1],
                              cam_pitch=mount[2], hud=hud)
        cv2.imshow('Camera + A* Path', img)
        cv2.waitKey(1)
 
    def odom_callback(self, msg):
        ox = msg.pose.pose.position.x
        oy = msg.pose.pose.position.y
        q = msg.pose.pose.orientation
        oyaw = np.arctan2(
            2.0 * (q.w * q.z + q.x * q.y),
            1.0 - 2.0 * (q.y * q.y + q.z * q.z)
        )
        # diff-drive odom starts at (0,0,0) wherever the rover spawned -> shift into the world frame
        c, s_ = math.cos(self.spawn_yaw), math.sin(self.spawn_yaw)
        self.rover_x = self.spawn_x + c * ox - s_ * oy
        self.rover_y = self.spawn_y + s_ * ox + c * oy
        self.rover_yaw = (oyaw + self.spawn_yaw + math.pi) % (2 * math.pi) - math.pi
        if self._trail_last is None or math.hypot(
                self.rover_x - self._trail_last[0],
                self.rover_y - self._trail_last[1]) >= 0.5:
            self._trail_last = (self.rover_x, self.rover_y)
            # Keep spatial trail samples every 0.5 m; state transitions are
            # recorded separately and immediately by the state property setter.
            self._write_trail_row()
        if not getattr(self, '_odom_seen', False):
            self._odom_seen = True
            self._reset_progress()

    # ---------------- robhar_2 helpers ----------------
    def _tf_matrix(self, target, source):
        """4x4 matrix taking points expressed in `source` into `target` (latest TF), or None."""
        try:
            t = self.tf_buffer.lookup_transform(target, source, Time())
        except Exception:
            return None
        q = t.transform.rotation
        x, y, z, w = q.x, q.y, q.z, q.w
        T = np.eye(4)
        T[:3, :3] = [[1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
                     [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
                     [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)]]
        T[:3, 3] = [t.transform.translation.x, t.transform.translation.y, t.transform.translation.z]
        return T

    def _get_camera_mount(self):
        """(forward, height, pitch) of the camera w.r.t. base_footprint, straight from the URDF via TF.
        Falls back to the cam_forward / cam_height / cam_pitch parameters until TF is available."""
        if self._cam_mount is None and self.use_tf_camera_mount and self.cam_frame:
            T = self._tf_matrix(self.base_frame, self.cam_frame)
            if T is not None:
                self._cam_mount = (float(T[0, 3]), float(T[2, 3]),
                                   float(np.arcsin(np.clip(T[2, 0], -1.0, 1.0))))   # + = looking up
                self.get_logger().info(
                    f'Camera mount from TF ({self.base_frame} -> {self.cam_frame}): '
                    f'forward {self._cam_mount[0]:.3f} m, height {self._cam_mount[1]:.3f} m, '
                    f'pitch {math.degrees(self._cam_mount[2]):.1f} deg')
        if self._cam_mount is None:
            return (self.get_parameter('cam_forward').value,
                    self.get_parameter('cam_height').value,
                    self.get_parameter('cam_pitch').value)
        return self._cam_mount

    def _footprint_dist(self, angles):
        """Distance from the lidar axis to the chassis outline (a rectangle) along each beam."""
        c, s_ = np.cos(angles), np.sin(angles)
        with np.errstate(divide='ignore', invalid='ignore'):
            dx = np.where(c > 1e-9, self.body_front / c, np.where(c < -1e-9, self.body_back / -c, np.inf))
            dy = np.where(np.abs(s_) > 1e-9, self.body_hw / np.abs(s_), np.inf)
        return np.minimum(dx, dy)
        
    @staticmethod
    
    def _densify(path, step=1.5):
        out = [path[0]]
        for (x0, y0), (x1, y1) in zip(path[:-1], path[1:]):
            n = max(1, int(math.hypot(x1 - x0, y1 - y0) / step))
            for k in range(1, n + 1):
                out.append((x0 + (x1 - x0) * k / n, y0 + (y1 - y0) * k / n))
        return out

    def preprocess_scan(self, msg):
        r = np.asarray(msg.ranges, dtype=np.float64)
        ang = msg.angle_min + np.arange(len(r)) * msg.angle_increment
        ang = (ang + np.pi) % (2 * np.pi) - np.pi
        key = (len(r), msg.angle_min, msg.angle_increment)
        if self._fp_key != key:
            self._fp = self._footprint_dist(ang)
            self._fp_key = key
        with np.errstate(invalid='ignore'):
            gap = r - self._fp
            bad = (~np.isfinite(r)) | (r < msg.range_min) | (r > msg.range_max) | (gap < self.SELF_MARGIN)
        valid = ~bad
        rear = np.abs(np.abs(ang) - np.pi) <= self.REAR_ANGLE
        self._min_gap = float(gap[valid].min()) if valid.any() else float('inf')
        self._min_rear_gap = float(gap[valid & rear].min()) if (valid & rear).any() else float('inf')
        if valid.any():
            k = np.where(valid)[0][np.argmin(gap[valid])]
            self._worst = (math.degrees(ang[k]), float(r[k]), float(self._fp[k]))
        
        self._update_zones(msg, r, ang, valid)

        r[bad] = msg.range_max + 1.0
        
        # FIX: Convert the numpy array to a standard Python list of floats
        msg.ranges = r.astype(np.float32).tolist()

    def lidar2pixel(self, msg):
        """Project the level-lidar points into the camera image using TF + CameraInfo
        (replaces the hard-coded waffle offsets/intrinsics)."""
        if self.cam_frame is None or self.cam_size is None:
            return []
        T = self._tf_matrix(self.cam_frame, msg.header.frame_id)
        if T is None:
            return []
        if self.cam_K:
            fx, fy, cx, cy = self.cam_K
        else:
            w, h = self.cam_size
            fx = fy = (w / 2.0) / math.tan(self.get_parameter('cam_hfov').value / 2.0)
            cx, cy = w / 2.0, h / 2.0
        r = np.asarray(msg.ranges, dtype=np.float64)
        ang = msg.angle_min + np.arange(len(r)) * msg.angle_increment
        ok = np.isfinite(r) & (r >= msg.range_min) & (r <= msg.range_max)
        if not ok.any():
            return []
        pts = np.stack([r * np.cos(ang), r * np.sin(ang), np.zeros_like(r), np.ones_like(r)])[:, ok]
        p = (T @ pts)[:3]                      # camera link frame: x fwd, y left, z up
        X, Y, Z = -p[1], -p[2], p[0]           # -> optical frame: x right, y down, z fwd
        f = Z > 0
        u = fx * X[f] / Z[f] + cx
        v = fy * Y[f] / Z[f] + cy
        return [(int(a), int(b), float(d)) for a, b, d in zip(u, v, r[ok][f])]

    def get_window(self, ranges, center, half_width, range_min, range_max):
        n = len(ranges)
        indices = [(center + i) % n for i in range(-half_width, half_width)]
        valid = [ranges[i] for i in indices if range_min <= ranges[i] <= range_max]
        return valid if valid else [float('inf')]  # no valid readings = treat as clear    
    
    def stop_rover(self):
 
        twist = Twist()
        twist.linear.x = 0.0
        twist.angular.z = 0.0
 
        self.pub.publish(twist)
    
    def _update_zones(self, msg, r, ang, valid):
        """Cleaned scan -> clearance per zone (m, measured from the chassis OUTLINE), in the lidar/base frame.
        FRONT / REAR: rectangular corridor, |y| <= half_width + margin, beyond the front/back face.
        LEFT / RIGHT: beside the chassis (-body_back < x < body_front), beyond the half-width.
        spin_r: range of the nearest valid return in ANY direction (for the pirouette check)."""
        INF = float('inf')

        if not self._blind_checked:       # one-time self-check: a trigger inside the lidar's blind zone can never fire
            self._blind_checked = True
            checks = {'FRONT stop': self.body_front + self.CRITICAL_DISTANCE,
                      'REAR stop': self.body_back + self.REAR_CRITICAL_DISTANCE,
                      'SIDE limit': self.body_hw + self.SIDE_GAP,
                      'SPIN check': self.SPIN_CLEAR_RADIUS}
            for name, trig in checks.items():
                if trig <= msg.range_min + 0.05:
                    self.get_logger().error(
                        f"{name} fires at r={trig:.2f} m from the lidar axis, inside its blind zone "
                        f"(range_min={msg.range_min:.2f} m) -> it can NEVER trigger. Increase its gap.")

        z = {'front': INF, 'rear': INF, 'left': INF, 'right': INF, 'spin_r': INF, 'spin_x': 0.0, 'spin_y': 0.0}
        if valid.any():
            rv, av = r[valid], ang[valid]
            x, y = rv * np.cos(av), rv * np.sin(av)
            cw = self.body_hw + self.ZONE_LAT_MARGIN
            fr = (x >= self.body_front) & (np.abs(y) <= cw)
            rr = (x <= -self.body_back) & (np.abs(y) <= cw)
            beside = (x > -self.body_back) & (x < self.body_front)
            lf = beside & (y > self.body_hw)
            rt = beside & (y < -self.body_hw)
            if fr.any(): z['front'] = float((x[fr] - self.body_front).min())
            if rr.any(): z['rear'] = float((-x[rr] - self.body_back).min())
            if lf.any(): z['left'] = float((y[lf] - self.body_hw).min())
            if rt.any(): z['right'] = float((-y[rt] - self.body_hw).min())
            k = int(np.argmin(rv))
            z['spin_r'], z['spin_x'], z['spin_y'] = float(rv[k]), float(x[k]), float(y[k])
        self._zone = z

    def _apply_zone_limits(self, twist):
        """Last gate before a Twist is published. A blocked zone VETOES THAT DIRECTION of motion, nothing more:
          - front corridor blocked  -> forward translation vetoed; rotation / reverse still allowed
          - rear  corridor blocked  -> reverse vetoed
          - pure rotation           -> only if the swept circle is clear, else creep away in an arc
          - moving arc              -> never steer toward a close side wall
        Sets self._cmd_vetoed = True when a real command ended up as 'do nothing' (feeds the BACKTRACK decision)."""
        z = self._zone
        v0, w0 = twist.linear.x, twist.angular.z
        wanted = abs(v0) > self.SPIN_V_EPS or abs(w0) > 1e-3
        fwd_ok = z['front'] > self.CRITICAL_DISTANCE
        rev_ok = z['rear'] > self.REAR_CRITICAL_DISTANCE
        v, w = v0, w0

        if v > 0.0 and not fwd_ok:                                          # ---- direction vetoes
            v = 0.0
            self.get_logger().warn(f"FORWARD VETOED: front gap {z['front']:.2f} m <= {self.CRITICAL_DISTANCE:.2f}",
                                   throttle_duration_sec=2.0)
        elif v < 0.0 and not rev_ok:
            v = 0.0
            self.get_logger().warn(f"REVERSE VETOED: rear gap {z['rear']:.2f} m <= {self.REAR_CRITICAL_DISTANCE:.2f}",
                                   throttle_duration_sec=2.0)

        if abs(v) < self.SPIN_V_EPS and abs(w) > 1e-3:                      # ---- pirouette
            if z['spin_r'] >= self.SPIN_CLEAR_RADIUS:
                v = 0.0                                                      # circle clear -> pure rotation allowed
            else:                                                            # blocked: translate away + arc away
                v, w = 0.0, 0.0
                for sgn in ((-1.0, 1.0) if z['spin_x'] > 0.15 else (1.0, -1.0)):   # nearest thing ahead -> back off first
                    if (sgn > 0 and fwd_ok) or (sgn < 0 and rev_ok):
                        away = -math.copysign(1.0, z['spin_y'] if z['spin_y'] != 0.0 else 1.0) * sgn
                        v, w = sgn * self.ESCAPE_SPEED, away * self.ESCAPE_ANGULAR
                        self.get_logger().warn(
                            f"SPIN BLOCKED (nearest return {z['spin_r']:.2f} m < {self.SPIN_CLEAR_RADIUS:.2f}) "
                            f"-> creeping {'forward' if sgn > 0 else 'reverse'} and arcing away",
                            throttle_duration_sec=2.0)
                        break
                else:
                    self.get_logger().warn('BOXED IN: spin, forward and reverse all blocked -> holding still',
                                           throttle_duration_sec=2.0)
        elif abs(v) >= self.SPIN_V_EPS:                                      # ---- moving arc: don't steer into a wall
            sgn = 1.0 if v > 0.0 else -1.0
            if (w * sgn > 0.0 and z['left'] <= self.SIDE_GAP) or (w * sgn < 0.0 and z['right'] <= self.SIDE_GAP):
                w = 0.0

        twist.linear.x, twist.angular.z = v, w
        self._cmd_vetoed = wanted and abs(v) < self.SPIN_V_EPS and abs(w) < 1e-3
        return twist

    def _note_stop_cycle(self):
        """Displacement watchdog: many front-stops near the same spot = livelock. Returns True when the run is declared FAILED."""
        xy = (self.rover_x, self.rover_y)
        if self._stop_anchor is None or math.hypot(xy[0] - self._stop_anchor[0], xy[1] - self._stop_anchor[1]) > self.STOP_PROGRESS_M:
            self._stop_anchor, self._stop_cycles = xy, 1
        else:
            self._stop_cycles += 1
        if self._stop_cycles > self.MAX_STOP_CYCLES:
            self.get_logger().error(
                f"FAILED: {self._stop_cycles} safety stops within {self.STOP_PROGRESS_M:.1f} m of ({self._stop_anchor[0]:.1f}, {self._stop_anchor[1]:.1f}) - livelock, giving up.")
            self.stop_rover()
            self.state = "FAILED"
            return True
        return False

    def safety_override(self, msg):
        """(no longer called from NORMAL) FRONT corridor stop check - kept for diagnostics / manual use."""
        z = self._zone
        self._unsafe_count = self._unsafe_count + 1 if z['front'] <= self.CRITICAL_DISTANCE else 0
        if self._unsafe_count >= self.SAFETY_SCANS:
            if not self.emergency_stop:
                self.get_logger().warn(
                    f"FRONT SAFETY STOP: {z['front']:.2f} m ahead of the chassis (limit {self.CRITICAL_DISTANCE:.2f}) | "
                    f"side L {z['left']:.2f} R {z['right']:.2f} | rear {z['rear']:.2f} | nearest any-direction {z['spin_r']:.2f} m")
            self.emergency_stop = True
            self.stop_rover()
            return True
        self.emergency_stop = False
        return False

    def reverse_safety_override(self, msg):
        """REAR corridor only. Used while backtracking so we don't back into something."""
        z = self._zone
        if z['rear'] <= self.REAR_CRITICAL_DISTANCE:
            self.get_logger().warn(
                f"BACKTRACK BLOCKED: rear obstacle {z['rear']:.2f} m from the chassis (limit {self.REAR_CRITICAL_DISTANCE:.2f})")
            self.stop_rover()
            return True
        return False

    def backtrack(self, msg):

        # 1. Never reverse into an obstacle detected behind the rover.
        if self.reverse_safety_override(msg):
            self.state = "REPLANNING"
            self.replan_attempted = False
            return

        now = time.monotonic()

        # 2. Initialise a new backtracking manoeuvre.
        if self.backtrack_start_time is None:

            if self.backtrack_target_idx is None:
                self.backtrack_target_idx = max(
                    0, self.waypoint_idx - self.BACKTRACK_WAYPOINTS
                )

            self.backtrack_target_idx = min(
                max(0, int(self.backtrack_target_idx)),
                len(self.path) - 1
            )

            # Start at the previous waypoint, not the current forward target.
            self.backtrack_cursor_idx = max(
                self.backtrack_target_idx,
                min(self.waypoint_idx - 1, len(self.path) - 1)
            )

            self.backtrack_start_time = now
            self.backtrack_start_x = self.rover_x
            self.backtrack_start_y = self.rover_y

            self.backtrack_last_progress_xy = (
                self.rover_x, self.rover_y
            )
            self.backtrack_last_progress_time = now

            # Estimate the route length from the rover to the backtrack target.
            cursor = self.backtrack_cursor_idx
            target = self.backtrack_target_idx

            route_length = math.hypot(
                self.rover_x - self.path[cursor][0],
                self.rover_y - self.path[cursor][1]
            )

            for i in range(cursor, target, -1):
                x1, y1 = self.path[i]
                x0, y0 = self.path[i - 1]
                route_length += math.hypot(x1 - x0, y1 - y0)

            # Allow enough time for the estimated route, with a hard upper limit.
            estimated_timeout = (
                self.BACKTRACK_TIMEOUT_MARGIN
                + self.BACKTRACK_TIMEOUT_FACTOR
                * route_length / max(self.BACKTRACK_SPEED, 0.01)
            )

            self.backtrack_timeout = min(
                self.BACKTRACK_MAX_TIME,
                max(self.BACKTRACK_MIN_TIME + 1.0, estimated_timeout)
            )

            self.get_logger().warn(
                f"BACKTRACK START: WP {self.waypoint_idx} -> "
                f"WP {self.backtrack_target_idx}, "
                f"estimated distance {route_length:.2f} m, "
                f"timeout {self.backtrack_timeout:.1f} s"
            )

        elapsed = now - self.backtrack_start_time

        # 3. Monitor actual movement.
        progress = math.hypot(
            self.rover_x - self.backtrack_last_progress_xy[0],
            self.rover_y - self.backtrack_last_progress_xy[1]
        )

        if progress >= self.BACKTRACK_STUCK_MOVE:
            self.backtrack_last_progress_xy = (
                self.rover_x, self.rover_y
            )
            self.backtrack_last_progress_time = now

        # 4. Walk backward through the previous A* waypoints.
        # Advance the reverse target only after reaching the current waypoint.
        while self.backtrack_cursor_idx > self.backtrack_target_idx:

            tx, ty = self.path[self.backtrack_cursor_idx]

            distance = math.hypot(
                tx - self.rover_x,
                ty - self.rover_y
            )

            if distance <= self.BACKTRACK_WP_TOLERANCE:
                self.backtrack_cursor_idx -= 1
            else:
                break

        # Current waypoint the rover must reach while reversing.
        tx, ty = self.path[self.backtrack_cursor_idx]

        distance_to_wp = math.hypot(
            tx - self.rover_x,
            ty - self.rover_y
        )

        target_reached = (
            self.backtrack_cursor_idx == self.backtrack_target_idx
            and distance_to_wp <= self.BACKTRACK_WP_TOLERANCE
        )

        # 5. Reached the requested previous waypoint.
        if target_reached:

            self.stop_rover()

            # Do not finish the manoeuvre before the minimum duration.
            # If the target is reached early, wait stopped rather than
            # reversing past the target.
            if elapsed < self.BACKTRACK_MIN_TIME:
                return

            self.get_logger().info(
                f"BACKTRACK COMPLETE: reached WP "
                f"{self.backtrack_target_idx} in {elapsed:.1f} s"
            )

            self.waypoint_idx = self.backtrack_target_idx
            self._reset_progress()

            self.state = "REPLANNING"
            self.replan_attempted = False

            self.backtrack_start_time = None
            self.backtrack_cursor_idx = None
            self.backtrack_target_idx = None

            return

        # 6. Abort if the rover is not making progress.
        if (
            now - self.backtrack_last_progress_time
            > self.BACKTRACK_STUCK_TIMEOUT
        ):

            self.stop_rover()

            self.get_logger().warn(
                "BACKTRACK ABORTED: insufficient movement. "
                "Attempting to replan."
            )

            self.state = "REPLANNING"
            self.replan_attempted = False

            self.backtrack_start_time = None
            self.backtrack_cursor_idx = None

            return

        # 7. Hard timeout: do not pretend the waypoint was reached.
        if elapsed >= self.backtrack_timeout:

            self.stop_rover()

            self.get_logger().warn(
                f"BACKTRACK TIMEOUT after {elapsed:.1f} s. "
                "Target waypoint was not reached; attempting to replan."
            )

            self.state = "REPLANNING"
            self.replan_attempted = False

            self.backtrack_start_time = None
            self.backtrack_cursor_idx = None

            return

        # 8. Steer toward the previous waypoint while driving backwards.
        bearing = math.atan2(
            ty - self.rover_y,
            tx - self.rover_x
        )

        # When reversing, the rover's nose points opposite its direction
        # of travel.
        desired_yaw = math.atan2(
            math.sin(bearing + math.pi),
            math.cos(bearing + math.pi)
        )

        heading_error = math.atan2(
            math.sin(desired_yaw - self.rover_yaw),
            math.cos(desired_yaw - self.rover_yaw)
        )

        twist = Twist()

        # Reduce reverse speed when alignment is poor.
        twist.linear.x = (
            -self.BACKTRACK_SPEED
            * max(0.0, math.cos(heading_error))
        )

        twist.angular.z = float(np.clip(
            heading_error,
            -self.MAX_ANGULAR,
            self.MAX_ANGULAR
        ))

        twist = self._apply_zone_limits(twist)
        self.pub.publish(twist)
        
    def handle_fusion_and_replan(self, msg):
        self.get_logger().info(
            "Obstacle detected. Starting sensor fusion..."
        )
        # For now, use LiDAR-only obstacle points
        # as a temporary placeholder.
        self.replan_around_obstacle(msg)
        self.state = "NORMAL"
        
    def vfh_select(self, msg, goal_angle):
        """Scan -> polar histogram -> free valleys -> steering angle toward the goal.
        Returns a dict, or None if there is no safe valley."""
        BIN = max(1, int(round(np.radians(self.VFH_BIN_DEG) / msg.angle_increment)))   # beams per bin
        THRESHOLD = 5          # density >= this -> sector blocked
        SAFE_MARGIN = 3.5      # steering target must be calmer than this
        MIN_VALLEY_WIDTH = 4   # bins
        min_r = self.VFH_MIN_RANGE

        # 1) histogram: (angle, density, nearest range) per 5-beam bin
        bins = []
        for i in range(0, len(msg.ranges), BIN):
            vals = [k for k in msg.ranges[i:i + BIN] if min_r < k < msg.range_max]
            density = (sum(1.0 / v for v in vals) * (5.0 / BIN)) if vals else 0.0   # scaled so THRESHOLD means what it did for 5 beams
            nearest = min(vals) if vals else float('inf')
            a = msg.angle_min + (i + BIN // 2) * msg.angle_increment
            a = (a + np.pi) % (2 * np.pi) - np.pi
            bins.append((a, density, nearest))
        bins.sort(key=lambda b: b[0])
        n = len(bins)
        angles = [b[0] for b in bins]
        dens = [b[1] for b in bins]
        bin_w = BIN * msg.angle_increment

        # 2) block sectors, then inflate each blocked sector by the robot's angular half-width
        blocked = [d >= THRESHOLD for d in dens]
        grown = blocked[:]
        for k in range(n):
            if blocked[k]:
                d = max(bins[k][2], min_r)
                half = int(np.ceil(np.arcsin(min(self.VFH_RADIUS / d, 1.0)) / bin_w))
                for j in range(max(0, k - half), min(n, k + half + 1)):
                    grown[j] = True

        # 3) free runs = valleys (sentinel closes a run that reaches the last bin)
        free = [not g for g in grown] + [False]
        valleys, start = [], None
        for k, f in enumerate(free):
            if f and start is None:
                start = k
            elif not f and start is not None:
                valleys.append((start, k - start))
                start = None
        cands = [(s, l) for s, l in valleys if l >= MIN_VALLEY_WIDTH]
        if not cands:
            return None

        # 4) pick the valley nearest the goal; steer AT the goal if it lies inside the valley,
        #    otherwise at the nearer valley edge (one bin of margin)
        best = None
        for s, l in cands:
            lo, hi = angles[s], angles[s + l - 1]
            m = min(bin_w, (hi - lo) / 2.0)
            if lo + m <= goal_angle <= hi - m:
                target = goal_angle
            elif goal_angle < lo + m:
                target = lo + m
            else:
                target = hi - m
            cost = abs((target - goal_angle + np.pi) % (2 * np.pi) - np.pi)
            idx = min(range(s, s + l), key=lambda j: abs(angles[j] - target))
            if dens[idx] > SAFE_MARGIN:
                continue
            if best is None or (cost, -l) < (best[0], -best[2]):
                best = (cost, s, l, target, idx)
        if best is None:
            return None
        _, s, l, target, idx = best
        return {'angle_best': target, 'start': s, 'width': l,
                'target_density': dens[idx], 'max_density': max(dens[s:s + l]),
                'n_valleys': len(cands), 'n_bins': n}

    def vfh_auto(self, msg):
 
            if self.state in ("FAILED", "MANUAL_OVERRIDE_NEEDED"):   # terminal: previously MANUAL_OVERRIDE_NEEDED fell through to NORMAL
                self.stop_rover()
                return

            if self.state != "NORMAL":                       # fresh start of NORMAL: no stale steering / pivot memory
                self._smoothed_angular = 0.0
                self._pivoting = False
                self._veto_count = 0

            if self.state == "SAFETY_STOP":
                self.stop_rover()
                # obstacle right in front - back away first, then replan.
                # (going straight to replan here just re-stops next
                # scan since the rover hasn't actually moved)
                self.backtrack_target_idx = max(0, self.waypoint_idx - self.BACKTRACK_WAYPOINTS)
                self.backtrack_start_time = None
                self.backtrack_start_x = None
                self.backtrack_start_y = None
                self.state = "BACKTRACK"
                return
 
            if self.state == "BACKTRACK":
                self.backtrack(msg)
                return
 
            if self.state == "REPLANNING":
                self.stop_rover()
 
                if not self.replan_attempted:
                    self.replan_attempted = True
                    self.get_logger().info("Starting A* replanning...")
                    success = self.replan_around_obstacle(msg)
 
                    if success:
                        self.state = "NORMAL"
                        self.replan_retry_count = 0
                        self._reset_progress()
                    else:
                        self.replan_retry_count += 1
                        if self.replan_retry_count > self.MAX_REPLAN_RETRIES:
                            self.get_logger().error("Replanning exhausted retries. Manual intervention needed.")
                            self.stop_rover()
                            self.state = "MANUAL_OVERRIDE_NEEDED"
                            self.stop_rover()
                            return
                        
                        self.get_logger().warn(f"Replan failed (attempt {self.replan_retry_count}), backtracking further.")
                        self.backtrack_target_idx = max(0, self.waypoint_idx - self.BACKTRACK_WAYPOINTS * self.replan_retry_count)
                        self.backtrack_start_time = None
                        self.backtrack_start_x = None
                        self.backtrack_start_y = None
                        self.replan_attempted = False
                        self.state = "BACKTRACK"
                return
 
            # NORMAL state
            # NOTE: the front limit is NOT a trigger any more. It vetoes forward motion inside _apply_zone_limits(),
            # so the rover can still pivot / reverse / slide. BACKTRACK is entered only when no legal motion is left.
            WAYPOINT_REACHED_THRESHOLD = 0.6
            last = len(self.path) - 1
            wx, wy = self.path[self.waypoint_idx]
            distance = math.hypot(wx - self.rover_x, wy - self.rover_y)

            # advance if we're inside the capture radius OR the next waypoint is already closer (overshoot)
            while self.waypoint_idx < last:
                nx, ny = self.path[self.waypoint_idx + 1]
                d_next = math.hypot(nx - self.rover_x, ny - self.rover_y)
                if distance < WAYPOINT_REACHED_THRESHOLD or d_next < distance:
                    self.waypoint_idx += 1
                    wx, wy, distance = nx, ny, d_next
                else:
                    break

            if self.waypoint_idx == last and distance < WAYPOINT_REACHED_THRESHOLD:
                self.stop_rover()
                self.get_logger().info('Goal reached — stopping.', throttle_duration_sec=5.0)
                return

            goal_angle = math.atan2(wy - self.rover_y, wx - self.rover_x) - self.rover_yaw
            goal_angle = (goal_angle + math.pi) % (2 * math.pi) - math.pi

            if self._is_stuck():
                self.get_logger().info('No progress, changing route!')
                if self._note_stop_cycle():
                    return
                self.backtrack_target_idx = max(0, self.waypoint_idx - self.BACKTRACK_WAYPOINTS)
                self.backtrack_start_time = None
                self.state = "BACKTRACK"
                return
            
            #print("rover:", self.rover_x, self.rover_y, "waypoint:", waypoint_x, waypoint_y, "goal_angle_deg:", np.degrees(goal_angle))
            
            if self.waypoint_idx == len(self.path) - 1 and distance < WAYPOINT_REACHED_THRESHOLD:
                twist = Twist()
                twist.linear.x = 0.0
                twist.angular.z = 0.0
                self.pub.publish(twist)
                self.get_logger().info('Goal reached — stopping.')
                return
            
            sel = self.vfh_select(msg, goal_angle)

            if sel is None:
                # no safe valley: stop. (Old code reversed blindly at -0.1 here.)
                # If the rover stays stuck, stuck_counter -> BACKTRACK (rear-checked) -> replan.
                self.get_logger().warn('VFH: no safe valley, stopping', throttle_duration_sec=2.0)
                self.stop_rover()
                return

            s, l = sel['start'], sel['width']
            angle_best = sel['angle_best']
            if abs(angle_best) < 0.05:
                angle_best = 0.0

            twist = Twist()
            min_length = 0
            max_length = sel['n_bins']
            min_speed = self.MIN_SPEED
            max_speed = self.MAX_SPEED
            ratio = (max_speed - min_speed) / (max_length - min_length)
            final_speed = min_speed + l * ratio
            final_speed *= max(0.0, math.cos(angle_best))   # facing away -> slow down
            pa = abs(angle_best)                            # pivot-first with hysteresis: large heading error -> v = 0
            self._pivoting = (pa > self.PIVOT_EXIT) if self._pivoting else (pa > self.PIVOT_ENTER)
            if self._pivoting:
                final_speed = 0.0

            # Smooth steering to eliminate high-frequency wobbling
            target_angular = np.clip(angle_best * self.STEER_GAIN, -self.MAX_ANGULAR, self.MAX_ANGULAR)
            prev_angular = getattr(self, '_smoothed_angular', 0.0)
            smoothed_angular = 0.6 * prev_angular + 0.4 * target_angular  # Low-pass filter
            self._smoothed_angular = smoothed_angular
            
###############################################################################            
                                                                          #####
            twist.linear.x = final_speed                                  #####
            twist.angular.z = smoothed_angular                            #####
                                                                          #####
###############################################################################            
            
            if self.recording:
                self.csv_writer.writerow([
                    self.scan_id,
                    round(self.rover_x, 3),
                    round(self.rover_y, 3),
                    round(np.degrees(self.rover_yaw), 2),
                    round(np.degrees(goal_angle), 2),
                    round(np.degrees(angle_best), 2),
                    round(sel['target_density'], 3),
                    round(sel['max_density'], 3),
                    s,
                    l
                ])
                self.scan_id += 1
                self.csv_file.flush()
 
                if self.scan_id >= self.max_scans:
                    self.recording = False
                    self.csv_file.flush()
                    self.csv_file.close()
                    self.get_logger().info(
                        'CSV recording complete: '
                        f'{self.max_scans} scans saved to vfh_data.csv'
                    )
 
            twist = self._apply_zone_limits(twist)
            self.pub.publish(twist)

            # wanted to move but every direction is vetoed for ~1 s -> genuinely stuck -> back off and replan
            self._veto_count = self._veto_count + 1 if self._cmd_vetoed else 0
            if self._veto_count >= self.VETO_SCANS:
                self.get_logger().warn(f"No legal motion for {self._veto_count} scans "
                                       f"(front {self._zone['front']:.2f} rear {self._zone['rear']:.2f} "
                                       f"L {self._zone['left']:.2f} R {self._zone['right']:.2f}) -> SAFETY_STOP")
                self._veto_count = 0
                if self._note_stop_cycle():
                    return
                self.state = "SAFETY_STOP"
 
    def destroy_node(self):
        if self.recording:
            self.csv_file.close()
        self.trail_file.close()
        cv2.destroyAllWindows()
        super().destroy_node()
        
    def replan_around_obstacle(self, msg):
        # Create a working copy of the cost matrix to inject dynamic LiDAR obstacles
        live_cost = self.cm.cost.copy()
        
        DYNAMIC_OBSTACLE_RADIUS = 0.35  # meters

        for i, r in enumerate(msg.ranges):
            if r < msg.range_min or r > msg.range_max:
                continue
            angle = msg.angle_min + i * msg.angle_increment
            world_angle = angle + self.rover_yaw
            obstacle_x = self.rover_x + r * np.cos(world_angle)
            obstacle_y = self.rover_y + r * np.sin(world_angle)
            
            # Convert world coordinate to costmap grid cells using cm.w2g
            obstacle_row, obstacle_col = self.cm.w2g(obstacle_x, obstacle_y)

            inflation_cells = int(DYNAMIC_OBSTACLE_RADIUS / self.cm.res)

            for dr in range(-inflation_cells, inflation_cells + 1):
                for dc in range(-inflation_cells, inflation_cells + 1):
                    row = obstacle_row + dr
                    col = obstacle_col + dc

                    if not (0 <= row < live_cost.shape[0] and 0 <= col < live_cost.shape[1]):
                        continue

                    distance = np.hypot(dr, dc) * self.cm.res
                    if distance <= DYNAMIC_OBSTACLE_RADIUS:
                        # Mark as blocked (inf cost)
                        live_cost[row, col] = np.inf

        current_pos = self.cm.w2g(self.rover_x, self.rover_y)

        goal_pos = self.cm.w2g(*self.path[-1])
        self.get_logger().info(f"Replanning around dynamic obstacles...")
        
        # Call dtm_planner's internal astar function
        from dtm_planner import astar, smooth
        path_indices, gv = astar(live_cost, current_pos, goal_pos)
        
        if path_indices is not None:
            # Smooth the newly found path
            smoothed_indices = smooth(live_cost, path_indices, gv)
            # Convert grid indices back to world coordinates
            self.path = self._densify([self.cm.g2w(r, c) for r, c in smoothed_indices])

            self.waypoint_idx = 0
            self.last_waypoint_idx = -1
            self.stuck_counter = 0
            self.get_logger().info(f"New dynamic path found: {len(self.path)} waypoints")
            return True
        else:
            self.get_logger().warn("No alternate path found around dynamic obstacle!")
            return False
    
 
def main():
    rclpy.init()
    node = LidarDetector()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()
 
if __name__ == '__main__':
    main()
 
"""
 
30deg zone BINARY obs avoidance code
 
angle_min = msg.angle_min 
        angle_increment = msg.angle_increment
        front_index = int(round(-angle_min/angle_increment))
        window_rad = 30 * (np.pi/180)
        window_indices = int(window_rad/angle_increment) 
        
        window = self.get_window(msg.ranges, front_index, window_indices, msg.range_min, msg.range_max)
        print(front_index, window_indices, len(msg.ranges))
        min_dist = min(window)
        
        if min_dist <= 0.5:
            left_window = self.get_window(msg.ranges, front_index + window_indices*2, window_indices, msg.range_min, msg.range_max)
            right_window = self.get_window(msg.ranges, front_index - window_indices*2, window_indices, msg.range_min, msg.range_max)
            left_min = min(left_window)
            right_min = min(right_window)
 
            twist = Twist()
            twist.linear.x = 0.0   # stop forward motion, obstacle ahead
 
            if left_min > right_min:
                twist.angular.z = 0.3   # turn left
            else:
                twist.angular.z = -0.3  # turn right
 
            self.pub.publish(twist)
        else:
            twist = Twist()
            twist.linear.x = 0.2    # clear ahead, go straight
            twist.angular.z = 0.0
            self.pub.publish(twist)
"""
