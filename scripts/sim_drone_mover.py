import math
import time
import rclpy
from rclpy.node import Node
from geometry_msgs.msg import PoseStamped, Twist
from ros_gz_interfaces.srv import SetEntityPose
from ros_gz_interfaces.msg import Entity

class AutonomousDroneNavigator(Node):
    def __init__(self):
        super().__init__("autonomous_drone_navigator")
        
        # 가제보 서비스 및 오도메트리 토픽 통신 설정
        self.client = self.create_client(SetEntityPose, "/world/indoor_mission/set_pose")
        self.pose_pub = self.create_publisher(PoseStamped, "/sim_drone/pose", 10)
        self.cmd_vel_pub = self.create_publisher(Twist, "/sim_drone/cmd_vel", 10)
        
        self.get_logger().info("Yaw 회전 없는 수평이동 기반 라인트래킹 및 ArUco 매핑 시스템 초기화...")
        while not self.client.wait_for_service(timeout_sec=1.0):
            self.get_logger().info("set_pose 서비스 연결 대기 중...")
            
        # 1. 드론의 초기 위치 (원점 오도메트리 기점)
        self.start_x = -17.5
        self.start_y = -13.0
        self.start_z = 0.702  # 버티포트 착륙 상태 고도
        
        self.current_x = self.start_x
        self.current_y = self.start_y
        self.current_z = self.start_z
        self.yaw = 0.0  # 고정 상태 유지 (회전 없음)
        
        # 2. 자율비행 가동 상태 정의 (State Machine)
        self.STATE_TAKEOFF = 0       
        self.STATE_LINE_TRACKING = 1  
        self.STATE_MARKER_DETECT = 2  
        self.STATE_RETURN_HOME = 3    
        self.STATE_LANDING = 4        
        self.current_state = self.STATE_TAKEOFF
        
        # 3. 실시간 지도 작성 공간 (동적 메모리 딕셔너리)
        self.discovered_markers = {}
        
        self.real_markers = [
            {"id": 0, "x": -9.0, "y": -4.5},
            {"id": 1, "x": -3.0, "y": 1.5},
            {"id": 2, "x": 3.0, "y": 4.5},
            {"id": 3, "x": 9.0, "y": -1.5}
        ]
        
        # 4. 속도 및 제어 주기 정의
        self.target_flight_z = 2.0  # 비행 미션 고도
        self.linear_speed = 0.8     # 메인 전진 속도 (m/s)
        self.update_period = 0.05   # 20Hz 루프
        
        self.last_grid_intersection = None
        
        # 타이머 실행
        self.timer = self.create_timer(self.update_period, self.control_loop)
        self.get_logger().info("수평 비행 모드 시작. [상태: Yaw 고정 수직 이륙 중]")

    def control_loop(self):
        """Yaw 회전값 변동 없이 오직 X, Y 수평 좌표 제어로만 라인을 추적하는 메인 제어 루프"""
        # 1. 카메라 센서 시뮬레이션: 유도선 및 ArUco 스캔 피드백 호출
        line_error, intersection_detected = self.emulate_camera_line_detection()
        self.emulate_aruco_detection()
        
        # 2. 상태 머신 분기 처리
        if self.current_state == self.STATE_TAKEOFF:
            self.handle_takeoff()
            
        elif self.current_state == self.STATE_LINE_TRACKING:
            self.handle_line_tracking(line_error, intersection_detected)
            
        elif self.current_state == self.STATE_MARKER_DETECT:
            self.handle_marker_detection()
            
        elif self.current_state == self.STATE_RETURN_HOME:
            self.handle_return_home()
            
        elif self.current_state == self.STATE_LANDING:
            self.handle_landing()

        # 3. 드론의 위치를 동기화하고 토픽을 발행 (Yaw 오리엔테이션 값은 항상 0으로 평행 상태 유지)
        self.send_pose_to_gazebo(self.current_x, self.current_y, self.current_z)
        self.publish_pose(self.current_x, self.current_y, self.current_z)

    def handle_takeoff(self):
        """정방향을 바라본 상태에서 그대로 고도만 확보하는 수직 이륙"""
        if self.current_z < self.target_flight_z:
            self.current_z += 0.05  
        else:
            self.get_logger().info("목표 미션 고도 안착. 정방향 평행 상태로 라인트래킹 스캔을 개시합니다.")
            self.current_state = self.STATE_LINE_TRACKING

    def handle_line_tracking(self, line_error, intersection_detected):
        """회전(Yaw) 없이 전진하면서 횡방향 오차를 수평 수평이동 제어로 보정하는 루프"""
        if len(self.discovered_markers) >= len(self.real_markers):
            self.get_logger().info("★ 모든 마커 맵핑 성공! 탐색된 실시간 지도를 근거로 평행 귀환합니다.")
            self.current_state = self.STATE_RETURN_HOME
            return

        if line_error is not None:
            # 바뀐 궤적 시나리오에 따른 축 매핑 분기 제어
            # 1구역 (Y축 평행 유도선 구역 비행 중일 때)
            if self.current_x < -11.0 and self.current_y < -7.6:
                p_gain = 0.20
                correction_x = -line_error * p_gain
                
                # 주 진행 방향은 Y축 상방(+), 오차 보정은 X축 수평이동
                self.current_y += self.linear_speed * self.update_period
                self.current_x += correction_x * self.update_period
                
            # 2구역 (X축 평행 유도선 구역 비행 중일 때)
            elif self.current_x < -12.0 and self.current_y >= -7.6:
                p_gain = 0.20
                correction_y = -line_error * p_gain
                
                # 주 진행 방향은 X축 우측(+), 오차 보정은 Y축 수평이동
                self.current_x += self.linear_speed * self.update_period
                self.current_y += correction_y * self.update_period
                
            # 3구역 (미션 본진 그리드 내부 영역 비행 중일 때)
            else:
                p_gain = 0.20
                correction_y = -line_error * p_gain
                self.current_x += self.linear_speed * self.update_period
                self.current_y += correction_y * self.update_period
                
                # 직각 그리드 교차점 영역 트리거 확인
                if intersection_detected:
                    if self.last_grid_intersection is None or math.hypot(self.current_x - self.last_grid_intersection[0], self.current_y - self.last_grid_intersection[1]) > 2.0:
                        self.last_grid_intersection = (self.current_x, self.current_y)
                        self.get_logger().info(f"그리드 직각 분기점 도달 추정 위치: ({self.current_x:.2f}, {self.current_y:.2f})")
        else:
            # 일시적인 라인 유실 시 평행 전진 탐색 유지
            self.current_x += 0.2 * self.update_period

    def handle_marker_detection(self):
        """마커 포착 순간 제자리에서 정지 비행(Hover)하며 매핑 메모리 저장 대기"""
        time.sleep(0.1) 
        self.current_state = self.STATE_LINE_TRACKING

    def handle_return_home(self):
        """기억해 둔 마커 데이터베이스를 기반으로 기체 회전 없이 최단 수평 결합 경로로 홈 버티포트 복귀"""
        dx = self.start_x - self.current_x
        dy = self.start_y - self.current_y
        dist = math.hypot(dx, dy)
        
        if dist > 0.1:
            # 복귀 경로 이동 시에도 Yaw 연산 없이 오직 대각선 방향 X, Y 수평 이동 벡터 성분만 반영
            self.current_x += (dx / dist) * self.linear_speed * self.update_period
            self.current_y += (dy / dist) * self.linear_speed * self.update_period
        else:
            self.get_logger().info("시작 버티포트 정중앙 복귀 성공. 수직 착륙 단계로 진입합니다.")
            self.current_state = self.STATE_LANDING

    def handle_landing(self):
        """최초 수직 이륙 지점으로 평행 정렬 상태를 유지하며 안전 하강 안착"""
        if self.current_z > self.start_z:
            self.current_z -= 0.03
        else:
            self.current_z = self.start_z
            self.get_logger().info("정밀 수평 자동 착륙 임무 완료. 제어 시스템을 오프라인 전환합니다.")
            self.timer.cancel()

    def emulate_camera_line_detection(self):
        """[궤적 전면 수정 보정] 수평 평행 비행 조건 및 반전된 유도선에 맞춘 카메라 오차 산출"""
        # 1구역: 버티포트에서 위(Y축 방향)로 먼저 올라가는 구역 트래킹
        if self.current_x < -11.0 and self.current_y < -7.6:
            error = self.current_x - (-17.5)  # X가 -17.5 라인 선상에 고정되어야 함
            return error, False
            
        # 2구역: 꺾여서 꼭지점(X축 방향)으로 진입하는 구역 트래킹
        elif self.current_x < -10.5 and self.current_y >= -7.6:
            error = self.current_y - (-7.5)   # Y가 -7.5 꼭지점 테두리에 고정되어야 함
            return error, False
            
        # 3구역: 미션 영역 내부 그리드 진입 이후
        else:
            closest_grid_x = round(self.current_x / 3.0) * 3.0
            error = self.current_x - closest_grid_x
            
            closest_grid_y = round(self.current_y / 3.0) * 3.0
            is_intersection = (abs(self.current_y - closest_grid_y) < 0.15)
            
            return error, is_intersection

    def emulate_aruco_detection(self):
        """카메라 영상 스트림에서 ArUco 마커 검출 알고리즘 모사"""
        fov_radius = 1.8  
        
        for marker in self.real_markers:
            if marker["id"] in self.discovered_markers:
                continue
                
            dist = math.hypot(self.current_x - marker["x"], self.current_y - marker["y"])
            
            if dist <= fov_radius:
                relative_x = marker["x"] - self.current_x
                relative_y = marker["y"] - self.current_y
                
                estimated_x = self.current_x + relative_x
                estimated_y = self.current_y + relative_y
                
                self.discovered_markers[marker["id"]] = (estimated_x, estimated_y)
                
                self.get_logger().info(
                    f"🏁 [마커 맵핑 완료] ID: {marker['id']} | "
                    f"추정 절대 좌표: ({estimated_x:.2f}, {estimated_y:.2f}) | "
                    f"총 발견 개수: {len(self.discovered_markers)}/4"
                )
                self.current_state = self.STATE_MARKER_DETECT

    def send_pose_to_gazebo(self, x, y, z):
        req = SetEntityPose.Request()
        req.entity.name = "sim_drone"
        req.entity.type = Entity.MODEL
        req.pose.position.x = float(x)
        req.pose.position.y = float(y)
        req.pose.position.z = float(z)
        req.pose.orientation.x = 0.0
        req.pose.orientation.y = 0.0
        req.pose.orientation.z = 0.0
        req.pose.orientation.w = 1.0
        self.client.call_async(req)

    def publish_pose(self, x, y, z):
        msg = PoseStamped()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.header.frame_id = "world"
        msg.pose.position.x = float(x)
        msg.pose.position.y = float(y)
        msg.pose.position.z = float(z)
        msg.pose.orientation.w = 1.0
        self.pose_pub.publish(msg)

def main():
    rclpy.init()
    node = AutonomousDroneNavigator()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    node.destroy_node()
    rclpy.shutdown()

if __name__ == "__main__":
    main()
