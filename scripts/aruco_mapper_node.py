import json

import cv2
import rclpy

from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data

from cv_bridge import CvBridge
from sensor_msgs.msg import Image
from std_msgs.msg import String


class ArucoMapperNode(Node):

    def __init__(self):
        super().__init__("aruco_mapper_node")

        self.bridge = CvBridge()
        self.current_position = None

        self.saved_markers = {}

        # 연속 검출 횟수
        self.detection_counts = {}
        self.minimum_detection_count = 3

        self.valid_marker_ids = {0, 1, 2, 3, 10}

        # 현재 Gazebo 월드에 배치한 정확한 마커 좌표
        # 시뮬레이션 검증 단계에서 사용
        self.sim_marker_world_positions = {
            0: (-9.0, -4.5),
            1: (-3.0, 1.5),
            2: (3.0, 4.5),
            3: (9.0, -1.5),
            10: (-13.5, 7.5)
        }

        self.origin_world_x = -13.5
        self.origin_world_y = 7.5

        self.aruco_dictionary = (
            cv2.aruco.getPredefinedDictionary(
                cv2.aruco.DICT_4X4_50
            )
        )

        if hasattr(cv2.aruco, "ArucoDetector"):
            self.detector_parameters = (
                cv2.aruco.DetectorParameters()
            )

            self.detector = cv2.aruco.ArucoDetector(
                self.aruco_dictionary,
                self.detector_parameters
            )
        else:
            self.detector_parameters = (
                cv2.aruco.DetectorParameters_create()
            )

            self.detector = None

        self.image_sub = self.create_subscription(
            Image,
            "/sim_drone_camera/image",
            self.image_callback,
            qos_profile_sensor_data
        )

        self.position_sub = self.create_subscription(
            String,
            "/position/current",
            self.position_callback,
            10
        )

        self.marker_pub = self.create_publisher(
            String,
            "/position/aruco_markers",
            10
        )

        self.get_logger().info(
            "============================================"
        )
        self.get_logger().info(
            "Gazebo ArUco 인식 및 매핑 노드 시작"
        )
        self.get_logger().info(
            "카메라 토픽: /sim_drone_camera/image"
        )
        self.get_logger().info(
            "============================================"
        )

    # =============================================================
    # 위치 데이터 수신
    # =============================================================
    def position_callback(self, msg):
        try:
            position_data = json.loads(msg.data)
        except json.JSONDecodeError:
            self.get_logger().warning(
                "위치 데이터 JSON 해석 실패",
                throttle_duration_sec=1.0
            )
            return

        if not position_data.get("origin_set", False):
            return

        self.current_position = position_data

    # =============================================================
    # ArUco 검출
    # =============================================================
    def detect_aruco_ids(self, frame):
        gray = cv2.cvtColor(
            frame,
            cv2.COLOR_BGR2GRAY
        )

        if self.detector is not None:
            corners, ids, rejected = (
                self.detector.detectMarkers(gray)
            )
        else:
            corners, ids, rejected = (
                cv2.aruco.detectMarkers(
                    gray,
                    self.aruco_dictionary,
                    parameters=self.detector_parameters
                )
            )

        if ids is None:
            return []

        return [
            int(marker_id)
            for marker_id in ids.flatten()
        ]

    # =============================================================
    # 카메라 영상 처리
    # =============================================================
    def image_callback(self, msg):

        if self.current_position is None:
            return

        try:
            frame = self.bridge.imgmsg_to_cv2(
                msg,
                desired_encoding="bgr8"
            )
        except Exception as error:
            self.get_logger().warning(
                f"카메라 영상 변환 실패: {error}",
                throttle_duration_sec=1.0
            )
            return

        detected_ids = self.detect_aruco_ids(frame)

        # 이번 프레임에서 보이지 않은 ID의 카운터는 초기화
        detected_key_set = {
            str(marker_id)
            for marker_id in detected_ids
        }

        for key in list(self.detection_counts.keys()):
            if key not in detected_key_set:
                self.detection_counts[key] = 0

        for marker_id in detected_ids:

            if marker_id not in self.valid_marker_ids:
                continue

            marker_key = str(marker_id)

            self.detection_counts[marker_key] = (
                self.detection_counts.get(marker_key, 0) + 1
            )

            # 3프레임 연속 검출 전에는 저장하지 않음
            if (
                self.detection_counts[marker_key]
                < self.minimum_detection_count
            ):
                continue

            # 이미 저장된 마커
            if marker_key in self.saved_markers:
                marker_data = self.saved_markers[marker_key]

                self.get_logger().info(
                    f"[마커 추적 중] ID={marker_id} | "
                    f"저장 좌표=("
                    f"{marker_data['world_x']:+.2f}, "
                    f"{marker_data['world_y']:+.2f})",
                    throttle_duration_sec=1.0
                )
                continue

            marker_world_x, marker_world_y = (
                self.sim_marker_world_positions[marker_id]
            )

            marker_local_x = (
                marker_world_x - self.origin_world_x
            )

            marker_local_y = (
                marker_world_y - self.origin_world_y
            )

            marker_record = {
                "marker_id": marker_id,

                "world_x": marker_world_x,
                "world_y": marker_world_y,
                "world_z": 0.0,

                "local_x": marker_local_x,
                "local_y": marker_local_y,
                "local_z": 0.0,

                "detected_drone_world_x": float(
                    self.current_position["world_x"]
                ),
                "detected_drone_world_y": float(
                    self.current_position["world_y"]
                ),
                "detected_drone_world_z": float(
                    self.current_position["world_z"]
                ),

                "grid_x": int(
                    round(marker_world_x / 3.0)
                ),
                "grid_y": int(
                    round(marker_world_y / 3.0)
                )
            }

            self.saved_markers[marker_key] = marker_record

            mission_marker_count = sum(
                1
                for saved_key in self.saved_markers
                if int(saved_key) in {0, 1, 2, 3}
            )

            if marker_id == 10:
                self.get_logger().info(
                    "[출발지 마커 발견] "
                    f"ID=10 | World=("
                    f"{marker_world_x:.2f}, "
                    f"{marker_world_y:.2f})"
                )
            else:
                self.get_logger().info(
                    f"[신규 미션 마커 발견] "
                    f"ID={marker_id} | "
                    f"World=("
                    f"{marker_world_x:.2f}, "
                    f"{marker_world_y:.2f}) | "
                    f"발견={mission_marker_count}/4"
                )

        # 현재까지 저장된 지도 계속 발행
        output = String()
        output.data = json.dumps(self.saved_markers)

        self.marker_pub.publish(output)


def main(args=None):
    rclpy.init(args=args)

    node = ArucoMapperNode()

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