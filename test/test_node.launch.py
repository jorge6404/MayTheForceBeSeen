"""De punta a punta: una CompressedImage real entra por input_topic y la fuerza sale por output_topic.

Usa la imagen de test/data (rosbag_fin_5/000000.jpg, −7,51 N con el ONNX) y el motor CPU.
"""
import os
import time
import unittest

import launch
import launch_ros.actions
import launch_testing.actions
import pytest
import rclpy
from sensor_msgs.msg import CompressedImage
from std_msgs.msg import Float64

IMAGEN = os.path.join(os.path.dirname(__file__), 'data', 'rosbag_fin_5_000000.jpg')
ENTRADA, SALIDA = '/test_vnf/image/compressed', '/test_vnf/force'


@pytest.mark.launch_test
def generate_test_description():
    nodo = launch_ros.actions.Node(
        package='visual_normal_force', executable='normal_force_node', output='screen',
        parameters=[{'input_topic': ENTRADA, 'output_topic': SALIDA, 'engine': 'cpu', 'rate_hz': 5.0}])
    return launch.LaunchDescription([nodo, launch_testing.actions.ReadyToTest()])


class TestNodo(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        rclpy.init()

    @classmethod
    def tearDownClass(cls):
        rclpy.shutdown()

    def test_publica_fuerza(self):
        node = rclpy.create_node('test_vnf')
        recibidas = []
        node.create_subscription(Float64, SALIDA, lambda m: recibidas.append(m.data), 10)
        pub = node.create_publisher(CompressedImage, ENTRADA, 10)
        img = CompressedImage(format='jpeg', data=open(IMAGEN, 'rb').read())
        fin = time.monotonic() + 60.0  # incluye el arranque del nodo y la carga del modelo
        try:
            while len(recibidas) < 3 and time.monotonic() < fin:
                pub.publish(img)
                rclpy.spin_once(node, timeout_sec=0.1)
        finally:
            node.destroy_node()
        self.assertGreaterEqual(len(recibidas), 3, 'el nodo no publicó la fuerza')
        for f in recibidas:
            self.assertAlmostEqual(f, -7.51, delta=0.1)
