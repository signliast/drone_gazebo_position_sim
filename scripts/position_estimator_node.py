import json

import rclpy
from rclpy.node import Node

from geometry_msgs.msg import PoseStamped
from std_msgs.msg import String


class PositionEstimatorNode(Node):

    def __init__(self):
        super().__init__("position_estimator_node")

        # 월드와 드론 무버에 사용한 동일한 출발 위치
        self.origin_world_x = -13.5
        self.origin_world_y = 7.5
        self.origin_world_z = 0.73

        self.grid_size = 3.0

        self.pose_sub = self.create_subscription(
            PoseStamped,
            "/sim_drone/pose",
            self.pose_callback,
            10
        )

        self.position_pub = self.create_publisher(
            String,
            "/position/current",
            10
        )

        self.get_logger().info(
            "============================================"
        )
        self.get_logger().info(
            "Gazebo 위치 추정 노드 시작"
        )
        self.get_logger().info(
            f"로컬 원점 World=("
            f"{self.origin_world_x}, "
            f"{self.origin_world_y}, "
            f"{self.origin_world_z})"
        )
        self.get_logger().info(
            "============================================"
        )

    def pose_callback(self, msg):
        world_x = float(msg.pose.position.x)
        world_y = float(msg.pose.position.y)
        world_z = float(msg.pose.position.z)

        # 헬리패드 출발점을 기준으로 하는 상대 좌표
        local_x = world_x - self.origin_world_x
        local_y = world_y - self.origin_world_y
        local_z = world_z - self.origin_world_z

        # 월드 중심 (0,0) 기준의 격자 인덱스
        grid_x = int(round(world_x / self.grid_size))
        grid_y = int(round(world_y / self.grid_size))

        data = {
            "origin_set": True,

            "local_x": local_x,
            "local_y": local_y,
            "local_z": local_z,

            "world_x": world_x,
            "world_y": world_y,
            "world_z": world_z,

            "grid_x": grid_x,
            "grid_y": grid_y,

            "altitude_from_vertiport": local_z
        }

        output = String()
        output.data = json.dumps(data)

        self.position_pub.publish(output)

        self.get_logger().info(
            f"World=({world_x:+.2f}, "
            f"{world_y:+.2f}, "
            f"{world_z:+.2f}) | "
            f"Local=({local_x:+.2f}, "
            f"{local_y:+.2f}, "
            f"{local_z:+.2f}) | "
            f"Grid=({grid_x:+d}, {grid_y:+d})",
            throttle_duration_sec=0.5
        )


def main(args=None):
    rclpy.init(args=args)

    node = PositionEstimatorNode()

    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()

        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    main()