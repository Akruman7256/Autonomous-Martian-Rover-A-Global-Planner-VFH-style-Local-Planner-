DTM-based global path planning + LiDAR-based local obstacle avoidance on a Mars terrain, in ROS 2 Humble + Gazebo Classic.

> Final-year mini project, Mechatronics Engineering, MIT Manipal.
A simulated 4-wheel rover that drives itself across a ~1 km Mars terrain:

1. **Plan globally**: turn the terrain elevation map (DTM) into a slope-based cost map and run A* to get a path.
2. **Drive locally**: follow the path while a LiDAR-based Vector Field Histogram (VFH) steers around obstacles.
3. **Recover**: if the rover gets blocked or stuck, it backs up and replans using live LiDAR obstacles.

**Scope:** simulation only (no hardware). Navigation uses a 2D LiDAR; the camera is used for visualisation/fusion experiments, not for driving decisions (see [Limitations](#-known-limitations)).


 Mars mesh (.obj) ──► DTM raster ──► slope costmap ──► A* global path ──┐
                                                                         ▼
 2D LiDAR (/scan) ──► self-body filter ──► VFH (steering + speed) ──► /cmd_vel ──► rover
        │                                         ▲
        └──► obstacle points ──► replan (A* on live costmap)
 Odometry (/odom) ──► pose in world frame ────────┘
 Camera + TF ──► fusion overlay (debug)
```

**State machine inside the navigation node:**

`NORMAL` → `SAFETY_STOP` → `BACKTRACK` → `REPLANNING` → `NORMAL`
(falls to `MANUAL_OVERRIDE_NEEDED` after too many failed replans)

---

Repo contents

| File | What it does |
|---|---|
| `mars_1km_robhar.world` | Gazebo world: Mars terrain mesh, gravity 3.72 m/s², physics, and the `robhar_2` rover |
| `dtm_planner.py` | DTM → traversability map → cost grid → A* path |
| `lidar_detector_terrain_robhar.py` | Main ROS 2 node: global plan + VFH + safety + backtrack + replan |
| `terrain_to_grid.py` | Applies the world file's mesh scale/rotation to the `.obj` so the planner sees the *same* terrain Gazebo does |
| `fusion_overlay.py` | LiDAR ↔ camera fusion debug node |
| `robhar_2/` | Rover URDF, meshes and ROS 2 package (SolidWorks export) |

---
Code overview

### `lidar_detector_terrain_robhar.py` (the main node)
This is the brain of the rover. On start-up it loads the terrain, builds the cost map, plans a global path from start to goal, and densifies it into waypoints (~1.5 m apart). Every LiDAR scan then goes through: (1) **pre-processing**, which removes the rover's own body from the scan using the chassis outline; (2) a **360° safety check**, which stops the rover if anything gets too close to the chassis; and (3) **VFH**, which builds a polar histogram of obstacle density, inflates blocked sectors by the rover's size, finds free "valleys" and steers into the one closest to the next waypoint (speed scales with valley width, and the rover pivots in place when facing away). If the rover is blocked or makes no progress for a few seconds, it reverses a short distance and **replans**, injecting the current LiDAR hits into the cost map as blocked cells. It also logs `planned_path.csv`, `rover_trail.csv` and `vfh_data.csv` for later analysis.

### `dtm_planner.py` (global planner)
Loads a terrain elevation map (`.tif`, `.npz`, or the Gazebo `.obj` mesh), smooths it at rover-footprint scale, computes slope, and classifies every cell as **FREE**, **DIFFICULT** (slope > 8°, passable but costly) or **BLOCKED** (slope > 15°, unknown data, or map edge). Blocked areas are inflated by the rover radius and given a soft safety halo, then an 8-connected **A\*** (no corner cutting) finds the cheapest path. A line-of-sight smoother removes unnecessary zig-zags. If start/goal sit in disconnected regions, it tells you why and snaps to the main traversable region instead of failing silently. Can be run standalone with `--plot` to inspect the map and path.

### `fusion_overlay.py` (LiDAR + camera fusion)
Projects LiDAR returns into the RGB camera image using the TF transform between the LiDAR and camera frames plus the camera intrinsics, and publishes the result as a debug image (`/fusion_debug/image`, plus a labelled compressed stream). It was built to verify sensor alignment (LiDAR dots land on the rocks they belong to) and as groundwork for semantic labelling. **It does not currently feed the planner or VFH.**

---

Terrain (DTM) and what we changed

The terrain is a NASA JPL-style Mars mesh (`mars_terrain.obj`) loaded as a **static** model with friction set high (`mu = 20`) so the wheels grip. The OBJ is Y-up, so the world file rolls it 1.57 rad about X to make it Z-up, and the **mesh scale's Y component is the vertical (height) exaggeration**.

Changes made along the way:

- **Vertical exaggeration `1 5 1` → `1 2 1`.** At 5× the hills were intentionally steep so the rover faced non-flat terrain, but it also produced many **ghost LiDAR detections** (steep slopes returning as walls) that confused VFH. Dropping to 2× cut them down a lot while keeping the terrain non-trivial.
- **Planner reads the same transformed terrain as Gazebo.** `terrain_to_grid.py` applies the world file's scale and rotation to the mesh vertices, so the cost map matches what the rover physically drives on.
- **Slope-based 3-class traversability** instead of a simple binary grid, so the planner prefers gentle ground but can still cross moderate slopes when it has to.
- **Costmap resolution of 0.5 m** at node level to keep planning fast on a 1 km map.
- **Rocks as static models.** Generated boulders (0.1 to 15 m tall) were first dynamic and slowly sank through the terrain; making them static fixed that.

---

How the simulation was done

- **Simulator:** Gazebo Classic with ROS 2 Humble, ODE physics, Mars gravity (3.72 m/s²).
- **Rover plugins:** `libgazebo_ros_diff_drive` (4-wheel skid steer, odometry), `libgazebo_ros_ray_sensor` (LiDAR), `libgazebo_ros_camera`, `libgazebo_ros_imu_sensor`, joint state publisher.
- **Test method:** spawn the rover near the origin, give the planner a goal in world coordinates, and let the node drive autonomously. Obstacle tests: place a wall or rocks in the path and check that the rover avoids or replans around them.
- **Logging:** path, trail and VFH decisions are written to CSV on every run, so runs can be compared and plotted afterwards.

Benchmarks

<!-- Verify these against your final runs before publishing. -->

| Goal (x, y) in m | Time | Notes |
|---|---|---|
| (-50, -30) | 2 m 28 s | |
| (-67, 24) | 3 m 42 s | |
| (200, 200) | 17 m 53 s | |
| (200, 300) | 14 m 22 s | from the project report |
| (300, 0) | TODO | |
| (81, -122) | TODO | |

---

Rover design (`robhar_2`)

Designed from scratch in **SolidWorks** after downloaded CAD and a Fusion export caused version and mesh problems.

- **Layout:** TurtleBot3-style chassis with two base plates separated by four pillars.
- **Wheels:** four Ø250 mm wheels, 0.5 m track, skid-steer drive.
- **Mass / size:** ~8.3 kg, roughly 0.6 × 0.4 m body.
- **Sensors:** level 2D LiDAR, an IMU, and left/right camera mounts (the RGB camera is the one active in sim).
- **Pipeline:** SolidWorks → URDF export → ROS 2 package `robhar_2`, checked in RViz (all four wheels rolling forward) before being spawned in Gazebo.

The CAD is deliberately simple. It's a platform for the navigation stack, not a mechanical contribution.

---

Setup and run

**Requirements:** Ubuntu 22.04, ROS 2 Humble, Gazebo Classic 11 + `gazebo_ros_pkgs`, Python 3: `numpy`, `scipy`, `opencv-python`, `matplotlib`, `rasterio` (only for `.tif`).

```bash
# 1. build the rover package
cd ~/ros2_ws && colcon build --packages-select robhar_2 && source install/setup.bash

# 2. edit the mesh path in mars_1km_robhar.world (it is absolute: file:///workspace/rover_project/...)

# 3. launch Gazebo with the world + rover  (TODO: replace with your launch command)
ros2 launch robhar_2 <your_launch_file>.launch.py

# 4. run the navigation node
python3 lidar_detector_terrain_robhar.py --ros-args \
  -p dtm_path:=/path/to/mars_terrain.obj \
  -p world_file:=/path/to/mars_1km_robhar.world \
  -p goal_x:=200.0 -p goal_y:=300.0
```

Inspect the planner on its own:

```bash
python3 dtm_planner.py --dtm mars_terrain.obj --world mars_1km_robhar.world \
  --start 0 0 --goal 200 400 --plot
```

---

Limitations Stated plainly, because they're the next things to fix:

- 🔸 The **2D LiDAR can't see obstacles below its mounting height** (low rocks, holes).
- 🔸 **The camera is not used for navigation.** Fusion currently stops at visualisation.
- 🔸 Replanning can occasionally route through an obstacle; obstacle inflation during replan needs tuning.
- 🔸 Ghost LiDAR detections from steep terrain still appear in places.
- 🔸 Simulation only; the rover model is simple (no suspension, no rocker-bogie).

 Roadmap

- 3D LiDAR + ground removal, with a **2D vs 3D benchmark** to quantify the below-lidar blind spot
- Camera segmentation model (rocks / sand / valleys) feeding terrain cost
- IMU/EKF fusion, higher-resolution terrain, better chassis

---

Credits

- Terrain mesh: NASA JPL (check the original license before redistributing).
- TurtleBot3 simulation packages (ROBOTIS) were used for early testing.
- References: Jeddisaravi et al. (2016) on multi-objective A*; MarsSeg (2024) and SegMarsViT (2022) for the segmentation direction.

License

MIT (for the code in this repo). Third-party assets keep their own licenses.
