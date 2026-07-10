
import json
import cv2
import rclpy
from rclpy.node import Node
from cv_bridge import CvBridge
from sensor_msgs.msg import Image
from std_msgs.msg import String

class ArucoMapperNode(Node):
    def __init__(self):
        super().__init__("aruco_mapper_node")
        self.bridge = CvBridge()
        self.current_position = None
        
        # 실시간 지도 데이터 저장소
        self.saved_markers = {}
        
        # 미션 타겟 마커(0~3)와 안전 귀환용 출발지 버티포트 마커(10) 목록 반영
        self.valid_marker_ids = {0, 1, 2, 3, 10} 
        
        # OpenCV ArUco 검출기 사전 로드 정의
        self.aruco_dict = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_50)
        if hasattr(cv2.aruco, "ArucoDetector"):
            self.parameters = cv2.aruco.DetectorParameters()
            self.detector = cv2.aruco.ArucoDetector(self.aruco_dict, self.parameters)
        else:
            self.parameters = cv2.aruco.DetectorParameters_create()
            self.detector = None
            
        # ROS2 토픽 구독 및 발행 설정
        self.image_sub = self.create_subscription(Image, "/sim_drone_camera/image", self.image_callback, 10)
        self.position_sub = self.create_subscription(String, "/position/current", self.position_callback, 10)
        self.marker_pub = self.create_publisher(String, "/position/aruco_markers", 10)
        
        self.get_logger().info("Yaw-Lock 자율비행 대응형 ArUco 실시간 매핑 노드 가동 완료")

    def position_callback(self, msg):
        try:
            self.current_position = json.loads(msg.data)
        except json.JSONDecodeError:
            pass

    def detect_aruco_ids(self, frame):
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        if self.detector is not None:
            corners, ids, rejected = self.detector.detectMarkers(gray)
        else:
            corners, ids, rejected = cv2.aruco.detectMarkers(gray, self.aruco_dict, parameters=self.parameters)
        
        if ids is None: 
            return []
        return [int(marker_id) for marker_id in ids.flatten()]

    def image_callback(self, msg):
        # 위치 추정 데이터가 아직 들어오지 않았다면 영상 처리 스킵
        if self.current_position is None: 
            return
            
        # 위치 추정 노드가 이륙을 완료하여 정식 원점 세팅을 완료하기 전에는 매핑 프로세스 차단
        # 이륙 도중 불완전한 오도메트리 좌표가 마커 위치로 등록되는 현상을 원천 방지합니다.
        if "local_z" not in self.current_position:
            return

        try:
            frame = self.bridge.imgmsg_to_cv2(msg, desired_encoding="bgr8")
        except Exception:
            return
            
        detected_ids = self.detect_aruco_ids(frame)
        
        for marker_id in detected_ids:
            if marker_id not in self.valid_marker_ids: 
                continue
                
            key = str(marker_id)
            
            # Case A: 이미 맵에 등록되어 알고 있는 마커인 경우 (지속 트래킹 로그 출력)
            if key in self.saved_markers:
                orig = self.saved_markers[key]
                # 실시간 트래킹 로그 출력 주기를 조절하여 터미널 오버플로우 방지
                self.get_logger().info(
                    f"🎯 [마커 추적 중] ID: {marker_id:02d} | "
                    f"맵핑된 고정위치: grid({orig['grid_x']:.0f}, {orig['grid_y']:.0f}) | "
                    f"현재 실시간 좌표: local({self.current_position['local_x']:+.1f}, {self.current_position['local_y']:+.1f})",
                    throttle_duration_sec=1.0 # 1초에 한 번만 출력되도록 캡슐화
                )
                continue
                
            # Case B: 비행 중 처음으로 시야에 포착된 신규 마커인 경우 (메모리 지도에 즉시 등록)
            self.saved_markers[key] = {
                "marker_id": marker_id,
                "local_x": self.current_position["local_x"], 
                "local_y": self.current_position["local_y"],
                "local_z": self.current_position["local_z"], 
                "grid_x": self.current_position["grid_x"],
                "grid_y": self.current_position["grid_y"]
            }
            
            self.get_logger().info(
                f"🔥 [NEW 마커 발견 및 저장] ID: {marker_id} "
                f"-> 실시간 오도메트리 기준 맵에 기록 완료! "
                f"grid_pos=({self.current_position['grid_x']}, {self.current_position['grid_y']})"
            )
            
        # 실시간 수집된 마커 지도 데이터베이스 브로드캐스팅 발행
        out = String()
        out.data = json.dumps(self.saved_markers)
        self.marker_pub.publish(out)

def main():
    rclpy.init()
    node = ArucoMapperNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    node.destroy_node()
    rclpy.shutdown()

if __name__ == "__main__":
    main()
