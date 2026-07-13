import json

import cv2
import numpy as np

import rclpy
from rclpy.node import Node

from sensor_msgs.msg import Image
from std_msgs.msg import Float32, Bool, String
from cv_bridge import CvBridge


class LineFollowerNode(Node):
    def __init__(self):
        super().__init__("line_follower_node")

        self.bridge = CvBridge()

        # 최종 월드의 카메라 토픽
        self.image_sub = self.create_subscription(
            Image,
            "/sim_drone_camera/image",
            self.image_callback,
            10
        )

        # 라인 중심 오차 출력
        self.error_x_pub = self.create_publisher(
            Float32,
            "/line_follower/error_x",
            10
        )

        self.error_y_pub = self.create_publisher(
            Float32,
            "/line_follower/error_y",
            10
        )

        # 교차점 감지 출력
        self.intersection_pub = self.create_publisher(
            Bool,
            "/line_follower/intersection",
            10
        )

        # 전체 상태 JSON 출력
        self.status_pub = self.create_publisher(
            String,
            "/line_follower/status",
            10
        )

        # 디버그 이미지 출력
        self.debug_image_pub = self.create_publisher(
            Image,
            "/line_follower/debug_image",
            10
        )

        # 검은색 라인 threshold
        # 라인이 너무 안 잡히면 90~120 사이로 올리고,
        # 바닥까지 검게 잡히면 50~70으로 낮추면 됨.
        self.black_threshold = 80

        # 교차점 판정 기준
        self.intersection_black_ratio_threshold = 0.18

        self.get_logger().info("Line follower node started.")
        self.get_logger().info("Subscribing: /sim_drone_camera/image")
        self.get_logger().info("Publishing: /line_follower/error_x, /line_follower/error_y, /line_follower/intersection")

    def image_callback(self, msg):
        try:
            frame = self.bridge.imgmsg_to_cv2(msg, desired_encoding="bgr8")
        except Exception as e:
            self.get_logger().warn(f"cv_bridge conversion failed: {e}")
            return

        h, w, _ = frame.shape

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # 검은색 라인만 추출
        _, binary = cv2.threshold(
            gray,
            self.black_threshold,
            255,
            cv2.THRESH_BINARY_INV
        )

        # 노이즈 제거
        kernel = np.ones((5, 5), np.uint8)
        binary = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel)
        binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel)

        # 하향 카메라 기준 중앙 영역 사용
        # 너무 넓게 보면 헬리패드/아루코까지 잡힐 수 있어서 중앙 ROI만 봄
        roi_x1 = int(w * 0.20)
        roi_x2 = int(w * 0.80)
        roi_y1 = int(h * 0.20)
        roi_y2 = int(h * 0.80)

        roi = binary[roi_y1:roi_y2, roi_x1:roi_x2]
        roi_h, roi_w = roi.shape

        moments = cv2.moments(roi)

        line_found = False
        error_x = 0.0
        error_y = 0.0
        cx_full = w // 2
        cy_full = h // 2

        if moments["m00"] > 0:
            line_found = True

            cx_roi = int(moments["m10"] / moments["m00"])
            cy_roi = int(moments["m01"] / moments["m00"])

            cx_full = roi_x1 + cx_roi
            cy_full = roi_y1 + cy_roi

            # 이미지 중심 기준 오차
            # error_x > 0 : 검은 라인이 화면 오른쪽
            # error_y > 0 : 검은 라인이 화면 아래쪽
            error_x = float(cx_full - (w / 2.0))
            error_y = float(cy_full - (h / 2.0))

        black_ratio = np.count_nonzero(roi) / float(roi.size)

        # 교차점은 검은 영역 비율이 평소보다 크게 늘어나는 것으로 판정
        intersection_detected = bool(
            line_found and black_ratio > self.intersection_black_ratio_threshold
        )

        # publish error
        error_x_msg = Float32()
        error_x_msg.data = error_x
        self.error_x_pub.publish(error_x_msg)

        error_y_msg = Float32()
        error_y_msg.data = error_y
        self.error_y_pub.publish(error_y_msg)

        inter_msg = Bool()
        inter_msg.data = intersection_detected
        self.intersection_pub.publish(inter_msg)

        status = {
            "line_found": line_found,
            "error_x": error_x,
            "error_y": error_y,
            "intersection": intersection_detected,
            "black_ratio": black_ratio,
            "threshold": self.black_threshold
        }

        status_msg = String()
        status_msg.data = json.dumps(status)
        self.status_pub.publish(status_msg)

        # debug image
        debug = frame.copy()

        # ROI 표시
        cv2.rectangle(
            debug,
            (roi_x1, roi_y1),
            (roi_x2, roi_y2),
            (255, 0, 0),
            2
        )

        # 이미지 중심
        cv2.circle(debug, (w // 2, h // 2), 6, (0, 255, 0), -1)

        # 라인 중심
        if line_found:
            cv2.circle(debug, (cx_full, cy_full), 7, (0, 0, 255), -1)
            cv2.line(debug, (w // 2, h // 2), (cx_full, cy_full), (0, 0, 255), 2)

        text = f"found={line_found} ex={error_x:.1f} ey={error_y:.1f} inter={intersection_detected} ratio={black_ratio:.2f}"
        cv2.putText(
            debug,
            text,
            (20, 35),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (0, 0, 255),
            2
        )

        try:
            debug_msg = self.bridge.cv2_to_imgmsg(debug, encoding="bgr8")
            debug_msg.header = msg.header
            self.debug_image_pub.publish(debug_msg)
        except Exception:
            pass


def main():
    rclpy.init()
    node = LineFollowerNode()

    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass

    node.destroy_node()
    rclpy.shutdown()


if __name__ == "__main__":
    main()
