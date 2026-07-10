
import json
import rclpy
from rclpy.node import Node
from geometry_msgs.msg import PoseStamped
from std_msgs.msg import String

class PositionEstimatorNode(Node):
    def __init__(self):
        super().__init__("position_estimator_node")
        self.grid_size = 3.0
        
        # 원점 세팅 관련 플래그 및 변수
        self.origin_set = False
        self.origin_x = 0.0
        self.origin_y = 0.0
        self.origin_z = 0.0
        
        # [추가] 안정적인 이륙 감지를 위한 목표 임무 고도 설정
        self.target_flight_z = 2.0 
        
        self.sub = self.create_subscription(PoseStamped, "/sim_drone/pose", self.pose_callback, 10)
        self.pub = self.create_publisher(String, "/position/current", 10)
        
        self.get_logger().info("Yaw-Lock 수평 자율비행용 정밀 위치 추정 노드 가동")
        self.get_logger().info(f"격자 그리드 크기 설정: {self.grid_size}m")

    def pose_callback(self, msg):
        world_x = msg.pose.position.x
        world_y = msg.pose.position.y
        world_z = msg.pose.position.z
        
        # [수정] 드론이 버티포트에서 이륙하여 목표 미션 고도(2.0m) 근처에 도달했을 때 원점을 확정합니다.
        # 이렇게 해야 지면과의 오차 및 초기 도약 노이즈가 오도메트리 원점에 반영되지 않습니다.
        if not self.origin_set:
            if world_z >= (self.target_flight_z - 0.05):
                self.origin_x = world_x
                self.origin_y = world_y
                self.origin_z = world_z  # 비행 순회 고도(2.0m)가 Local Z의 0점 기점이 됨
                self.origin_set = True
                self.get_logger().info("==========================================")
                self.get_logger().info(f"★ 비행 원점 확정 (이륙 완료): X={world_x:.2f}, Y={world_y:.2f}, Z={world_z:.2f}")
                self.get_logger().info("==========================================")
            else:
                # 아직 이륙 중일 때는 대기하며 현재 월드 좌표만 단순 로깅
                self.get_logger().info(f"드론 수직 이륙 중... 현재 고도: {world_z:.2f}m", throttle_duration_sec=1.0)
                return
            
        # 원점 기준 상대 좌표(Local Odometry) 연산
        local_x = world_x - self.origin_x
        local_y = world_y - self.origin_y
        local_z = world_z - self.origin_z  # 미션 고도 유지 시 0.0 근처에서 제어됨
        
        # [수정] 월드 좌표계의 격자선 배치 규칙과 부합하도록 정밀 그리드 인덱스 매핑 계산
        # 미션 영역 바닥 그리드는 월드 중심(0,0)을 기준으로 3m 간격 레이아웃이므로 
        # world_x, world_y를 직접 grid_size로 나누어 정사영 인덱스를 추적하는 것이 맵핑 아키텍처상 매끄럽습니다.
        grid_x = int(round(world_x / self.grid_size))
        grid_y = int(round(world_y / self.grid_size))
        
        # 통합 데이터 구조 패킹
        data = {
            "local_x": local_x, "local_y": local_y, "local_z": local_z,
            "grid_x": grid_x, "grid_y": grid_y,
            "world_x": world_x, "world_y": world_y, "world_z": world_z
        }
        
        msg_out = String()
        msg_out.data = json.dumps(data)
        self.pub.publish(msg_out)
        
        # 터미널 모니터링 가독성 향상
        self.get_logger().info(
            f"상대위치=({local_x:+.2f}, {local_y:+.2f}, {local_z:+.2f}) | "
            f"추정그리드=({grid_x:+.0f}, {grid_y:+.0f})"
        )

def main():
    rclpy.init()
    node = PositionEstimatorNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    node.destroy_node()
    rclpy.shutdown()

if __name__ == "__main__":
    main()
