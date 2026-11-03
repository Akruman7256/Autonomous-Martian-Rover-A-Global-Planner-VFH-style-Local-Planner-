# Starts Gazebo (with your world), publishes the robot model, and drops the rover in.
import os
from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, IncludeLaunchDescription, SetEnvironmentVariable
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node


def generate_launch_description():
    pkg_share = get_package_share_directory('robhar_2')
    urdf_path = os.path.join(pkg_share, 'urdf', 'robhar_2_gazebo.urdf')
    with open(urdf_path, 'r') as f:
        robot_description = f.read()

    # Lets Gazebo find the package://robhar_2/meshes/... files
    # (it looks for a folder called robhar_2 inside this path)
    model_path = os.path.dirname(pkg_share) + os.pathsep + os.environ.get('GAZEBO_MODEL_PATH', '')

    return LaunchDescription([
        # world file: pass your own with  world:=/path/to/mars_1km.world
        DeclareLaunchArgument('world', default_value=''),
        # where the rover is dropped (z is small because the origin is at ground level)
        DeclareLaunchArgument('x', default_value='0.0'),
        DeclareLaunchArgument('y', default_value='0.0'),
        DeclareLaunchArgument('z', default_value='0.3'),

        SetEnvironmentVariable('GAZEBO_MODEL_PATH', model_path),

        # Gazebo itself
        IncludeLaunchDescription(
            PythonLaunchDescriptionSource(
                os.path.join(get_package_share_directory('gazebo_ros'), 'launch', 'gazebo.launch.py')),
            launch_arguments={'world': LaunchConfiguration('world')}.items(),
        ),

        # publishes the robot's TF tree from the URDF
        Node(
            package='robot_state_publisher',
            executable='robot_state_publisher',
            parameters=[{'robot_description': robot_description, 'use_sim_time': True}],
            output='screen',
        ),

        # puts the robot into Gazebo
        Node(
            package='gazebo_ros',
            executable='spawn_entity.py',
            arguments=['-topic', 'robot_description', '-entity', 'robhar_2',
                       '-x', LaunchConfiguration('x'),
                       '-y', LaunchConfiguration('y'),
                       '-z', LaunchConfiguration('z')],
            output='screen',
        ),
    ])
