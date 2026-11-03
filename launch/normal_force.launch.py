import os

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node


def generate_launch_description():
    share = get_package_share_directory('visual_normal_force')
    return LaunchDescription([
        DeclareLaunchArgument('params_file', default_value=os.path.join(share, 'config', 'normal_force.yaml')),
        Node(package='visual_normal_force', executable='normal_force_node', name='normal_force',
             output='screen', parameters=[LaunchConfiguration('params_file')]),
    ])
