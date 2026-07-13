import math
import time
import cv2
import numpy as np
import rclpy

from rclpy.node import Node
from sensor_msgs.msg import Image
from std_msgs.msg import String
from geometry_msgs.msg import PoseStamped
from cv_bridge import CvBridge

from ros_gz_interfaces.srv import SetEntityPose
from ros_gz_interfaces.msg import Entity


CAMERA_TOPIC = "/sim_drone_camera/image"
SET_POSE_SERVICE = "/world/indoor_mission/set_pose"
POSE_TOPIC = "/sim_drone/pose"


# =========================
# set_pose 이동용 시작 좌표
# =========================

START_X = -13.5
START_Y = 7.5
START_Z = 0.73

FLIGHT_Z = 2.0


# =========================
# 이륙 후 앞으로 이동
# =========================

FORWARD_UNIT_X = 1.0
FORWARD_UNIT_Y = 0.0

MOVE_FORWARD_DISTANCE = 1.5


# =========================
# 진입선 코너 설정
# =========================

ENTRY_BEFORE_CORNER_AXIS = "Y"
ENTRY_BEFORE_CORNER_DIR = 1
ENTRY_BEFORE_CORNER_TRACK = "V"

ENTRY_AFTER_CORNER_AXIS = "X"
ENTRY_AFTER_CORNER_DIR = 1
ENTRY_AFTER_CORNER_TRACK = "V"

ENTRY_CORNER_EXIT_SPEED = 0.14


# =========================
# 격자 탐색 설정
# =========================

POINTS_PER_SWEEP = 9
TOTAL_SWEEPS = 6
TOTAL_GRID_POINTS = POINTS_PER_SWEEP * TOTAL_SWEEPS

GRID_X_TRACK = "V"
GRID_Y_TRACK = "H"

MISSION_MARKER_IDS = {0, 1, 2, 3}
START_MARKER_ID = 10

REVISIT_SEQUENCE = [3, 2, 1, 0]


# =========================
# 격자점 / 코너 감지 설정
# =========================

INTERSECTION_ROI_HALF = 65

INTERSECTION_COOLDOWN_TICKS = 2
INTERSECTION_LOST_REQUIRED_TICKS = 6

FIRST_LINE_REQUIRED_TICKS = 5

BRANCH_CENTER_HALF = 12
BRANCH_GAP = 10
BRANCH_LENGTH = 55
BRANCH_THICK = 16
BRANCH_RATIO_MIN = 0.08
CENTER_PIXEL_MIN = 20


# =========================
# 라인트레이싱 ROI 설정
# =========================

LINE_TRACK_ROI_HALF = 65
LINE_TRACK_LENGTH_HALF = 150

LINE_PROJECTION_MIN_PIXELS = 8
LINE_PEAK_WINDOW = 14
LINE_SMOOTH_KERNEL = 9

LINE_SEARCH_SPEED = 0.08
LINE_LOST_STOP_TICKS = 4
LINE_SEARCH_SWITCH_TICKS = 25


# =========================
# 속도 설정
# =========================

TAKEOFF_SPEED_Z = 0.55
MOVE_FORWARD_SPEED = 0.60

ENTRY_LINE_SPEED = 0.20
FIRST_THREE_WAY_APPROACH_SPEED = 0.18

FORWARD_SPEED_MAX = 0.18
FORWARD_SPEED_MIN = 0.06

RETURN_START_SPEED = 0.45

MAX_XY_SPEED = 0.8
MAX_Z_SPEED = 0.8

LAND_SPEED_Z = -0.25


# =========================
# ArUco 설정
# =========================

ARUCO_SAVE_CENTER_TOL_PX = 90
ARUCO_PENDING_MAX_TICKS = 12

ARUCO_DETECT_HOVER_SECONDS = 3.0

ARUCO_CENTER_TOL_PX = 18
ARUCO_CENTER_HOLD_TICKS = 35

REVISIT_MARKER_CONFIRM_TICKS = 8


# =========================
# 로그 설정
# =========================

NAV_DEBUG_LOG_SEC = 0.4


class GridDirectionSetPoseTracer(Node):
    def __init__(self):
        super().__init__("grid_direction_setpose_tracer")

        self.bridge = CvBridge()
        self.current_frame = None

        self.image_sub = self.create_subscription(
            Image,
            CAMERA_TOPIC,
            self.image_callback,
            10
        )

        self.grid_pub = self.create_publisher(
            String,
            "/grid/current",
            10
        )

        self.pose_pub = self.create_publisher(
            PoseStamped,
            POSE_TOPIC,
            10
        )

        self.pose_client = self.create_client(
            SetEntityPose,
            SET_POSE_SERVICE
        )

        self.dt = 0.05

        self.S_WAIT_CAMERA = "WAIT_CAMERA"
        self.S_TAKEOFF = "TAKEOFF"
        self.S_MOVE_FORWARD_3M = "MOVE_FORWARD_3M"

        self.S_LINE_TRACE_TO_ENTRY_CORNER = "LINE_TRACE_TO_ENTRY_CORNER"
        self.S_EXIT_ENTRY_CORNER = "EXIT_ENTRY_CORNER"
        self.S_LINE_TRACE_TO_FIRST_3WAY = "LINE_TRACE_TO_FIRST_3WAY"

        self.S_SCAN_SWEEP = "SCAN_SWEEP"
        self.S_SCAN_SHIFT_NEXT_ROW = "SCAN_SHIFT_NEXT_ROW"

        self.S_REVISIT_PREPARE = "REVISIT_PREPARE"
        self.S_REVISIT_MOVE = "REVISIT_MOVE"
        self.S_REVISIT_CENTER = "REVISIT_CENTER"
        self.S_REVISIT_SEARCH_MOVE = "REVISIT_SEARCH_MOVE"
        self.S_REVISIT_SEARCH_CHECK = "REVISIT_SEARCH_CHECK"

        self.S_RETURN_GRID_HOME = "RETURN_GRID_HOME"
        self.S_RETURN_START = "RETURN_START"
        self.S_HOME_CENTER = "HOME_CENTER"
        self.S_LAND = "LAND"
        self.S_DONE = "DONE"

        self.state = self.S_WAIT_CAMERA

        self.current_x = START_X
        self.current_y = START_Y
        self.current_z = START_Z
        self.current_yaw = 0.0

        self.initial_pose_sent = False
        self.pending_pose_future = None
        self.forward_moved = 0.0

        self.grid_x = 0
        self.grid_y = 0
        self.grid_origin_set = False

        self.grid_count_enabled = False

        self.entry_corner_lost_ticks = 0

        self.first_line_seen_ticks = 0
        self.first_three_way_ready = False

        self.sweep_dir_x = 1
        self.shift_dir_y = -1

        self.points_in_current_sweep = 0
        self.current_sweep_index = 0
        self.completed_sweeps = 0

        self.intersection_was_visible = False
        self.intersection_cooldown = 0
        self.wait_until_intersection_lost = False
        self.intersection_lost_ticks = 0

        self.visited_grid_points = []
        self.visited_grid_set = set()

        self.marker_grid_map = {}

        self.pending_aruco_id = None
        self.pending_aruco_ticks = 0

        self.aruco_hover_until = 0.0
        self.aruco_hover_marker_id = None

        self.revisit_index = 0
        self.revisit_target_id = None
        self.revisit_target_grid = None

        self.search_points = []
        self.search_point_index = 0
        self.search_check_ticks = 0
        self.search_check_max_ticks = 25

        self.nav_last_axis = None
        self.nav_active_target = None
        self.nav_phase = "X"

        self.revisit_lost_ticks = 0
        self.revisit_center_hold = 0

        self.home_center_hold = 0
        self.home_lost_ticks = 0

        self.Kp = 0.0032
        self.Ki = 0.00001
        self.Kd = 0.0012

        self.integral = 0.0
        self.prev_error = 0.0

        self.err_filt = 0.0
        self.err_alpha = 0.30
        self.jump_limit = 35.0

        self.speed_max = FORWARD_SPEED_MAX
        self.speed_min = FORWARD_SPEED_MIN
        self.err_slowdown_px = 100.0
        self.lateral_clamp = 0.14

        self.no_line_ticks = 0
        self.line_lost_ticks = 0
        self.line_search_dir = 1

        self.sign_y_lateral = -1.0
        self.sign_x_lateral = -1.0

        self.sign_align_vx = -1.0
        self.sign_align_vy = -1.0

        self.align_ex_f = 0.0
        self.align_ey_f = 0.0
        self.align_lpf_alpha = 0.40
        self.align_gain = 0.0014
        self.align_deadzone = 12
        self.align_clamp = 0.18

        self.last_log_time = {}

        if not hasattr(cv2, "aruco"):
            raise RuntimeError(
                "OpenCV aruco 모듈이 없습니다. opencv-contrib-python 설치가 필요합니다."
            )

        if hasattr(cv2.aruco, "getPredefinedDictionary"):
            self.aruco_dict = cv2.aruco.getPredefinedDictionary(
                cv2.aruco.DICT_4X4_50
            )
        else:
            self.aruco_dict = cv2.aruco.Dictionary_get(
                cv2.aruco.DICT_4X4_50
            )

        if hasattr(cv2.aruco, "DetectorParameters"):
            self.aruco_params = cv2.aruco.DetectorParameters()
        else:
            self.aruco_params = cv2.aruco.DetectorParameters_create()

        if hasattr(cv2.aruco, "ArucoDetector"):
            self.aruco_detector = cv2.aruco.ArucoDetector(
                self.aruco_dict,
                self.aruco_params
            )
        else:
            self.aruco_detector = None

        self.timer = self.create_timer(self.dt, self.loop)

        self.get_logger().info("set_pose 이동 + 격자점/방향 판단 노드 시작")
        self.get_logger().info("ArUco는 후보 감지 후 다음 격자점 카운트 순간 저장")
        self.get_logger().info("복귀 이동은 X축 먼저 보정 후 Y축 보정")
        self.get_logger().info("복귀 때 ArUco 중앙정렬 이동 제거, 마커 확인 후 3초 호버링")
        self.get_logger().info("호버링은 time.perf_counter() 기준 실제 시간 3초")
        self.get_logger().info("이동축과 카메라 선 방향 분리 적용")
        self.get_logger().info("코너 이후 X 이동 + 카메라 세로선 추적")
        self.get_logger().info("라인트레이싱: 중앙에 가까운 선 선택")
        self.get_logger().info("초록 안전선 제거 포함")

    def image_callback(self, msg):
        try:
            self.current_frame = self.bridge.imgmsg_to_cv2(
                msg,
                desired_encoding="bgr8"
            )
        except Exception as e:
            self.get_logger().error(f"이미지 변환 실패: {e}")

    def loop(self):
        if not self.pose_client.service_is_ready():
            if not self.pose_client.wait_for_service(timeout_sec=0.01):
                self.log_every("service", "Gazebo set_pose 서비스 대기 중...", 1.0)
                return

        if not self.initial_pose_sent:
            self.send_pose()
            self.publish_pose()
            self.initial_pose_sent = True
            self.get_logger().info(
                f"초기 위치 set_pose 완료: "
                f"({self.current_x:.2f}, {self.current_y:.2f}, {self.current_z:.2f})"
            )

        if self.aruco_hover_until > 0.0:
            now = time.perf_counter()

            if now < self.aruco_hover_until:
                self.hold_pose()
                return

            marker_id = self.aruco_hover_marker_id

            self.aruco_hover_until = 0.0
            self.aruco_hover_marker_id = None

            self.reset_line_pid()
            self.reset_intersection_detector(wait_until_lost=True)

            self.get_logger().info(
                f"ArUco {marker_id} 인식 후 실제 시간 3초 호버링 종료 → 임무 재개"
            )

            return

        if self.state == self.S_WAIT_CAMERA:
            self.do_wait_camera()
        elif self.state == self.S_TAKEOFF:
            self.do_takeoff()
        elif self.state == self.S_MOVE_FORWARD_3M:
            self.do_move_forward_3m()
        elif self.state == self.S_LINE_TRACE_TO_ENTRY_CORNER:
            self.do_line_trace_to_entry_corner()
        elif self.state == self.S_EXIT_ENTRY_CORNER:
            self.do_exit_entry_corner()
        elif self.state == self.S_LINE_TRACE_TO_FIRST_3WAY:
            self.do_line_trace_to_first_3way()
        elif self.state == self.S_SCAN_SWEEP:
            self.do_scan_sweep()
        elif self.state == self.S_SCAN_SHIFT_NEXT_ROW:
            self.do_scan_shift_next_row()
        elif self.state == self.S_REVISIT_PREPARE:
            self.do_revisit_prepare()
        elif self.state == self.S_REVISIT_MOVE:
            self.do_revisit_move()
        elif self.state == self.S_REVISIT_CENTER:
            self.do_revisit_center()
        elif self.state == self.S_REVISIT_SEARCH_MOVE:
            self.do_revisit_search_move()
        elif self.state == self.S_REVISIT_SEARCH_CHECK:
            self.do_revisit_search_check()
        elif self.state == self.S_RETURN_GRID_HOME:
            self.do_return_grid_home()
        elif self.state == self.S_RETURN_START:
            self.do_return_start()
        elif self.state == self.S_HOME_CENTER:
            self.do_home_center()
        elif self.state == self.S_LAND:
            self.do_land()
        elif self.state == self.S_DONE:
            self.hold_pose()

    def do_wait_camera(self):
        self.grid_count_enabled = False
        self.reset_intersection_detector(wait_until_lost=False)
        self.hold_pose()

        if self.current_frame is None:
            self.log_every("camera", "카메라 프레임 대기 중...", 1.0)
            return

        self.state = self.S_TAKEOFF
        self.get_logger().info("카메라 수신 완료 → set_pose 이륙 시작")

    def do_takeoff(self):
        self.grid_count_enabled = False
        self.reset_intersection_detector(wait_until_lost=False)

        if self.current_z < FLIGHT_Z:
            self.apply_velocity(0.0, 0.0, TAKEOFF_SPEED_Z)
            return

        self.current_z = FLIGHT_Z
        self.send_pose()
        self.publish_pose()

        self.forward_moved = 0.0
        self.state = self.S_MOVE_FORWARD_3M
        self.get_logger().info("이륙 완료 → 미션 영역 좌상단까지 +X 1.5m 이동 시작")

    def do_move_forward_3m(self):
        """
        헬리패드 중심 (-13.5, 7.5)에서 미션 영역 좌상단 격자점
        (-12.0, 7.5)까지 +X 방향으로 1.5m 이동합니다.

        현재 월드에는 별도의 진입 연결선이 없으므로,
        도착 즉시 이 지점을 grid=(0, 0)으로 설정하고
        첫 번째 행을 +X 방향으로 탐색합니다.
        """
        self.grid_count_enabled = False
        self.reset_intersection_detector(wait_until_lost=False)

        if self.forward_moved < MOVE_FORWARD_DISTANCE:
            remaining = MOVE_FORWARD_DISTANCE - self.forward_moved
            step = min(MOVE_FORWARD_SPEED * self.dt, remaining)

            self.current_x += FORWARD_UNIT_X * step
            self.current_y += FORWARD_UNIT_Y * step
            self.forward_moved += step

            self.send_pose()
            self.publish_pose()
            return

        # 좌표 누적 오차를 제거하고 정확히 미션 영역 좌상단에 맞춤
        self.current_x = -12.0
        self.current_y = 7.5
        self.current_z = FLIGHT_Z
        self.send_pose()
        self.publish_pose()

        # 좌상단 격자점을 탐색 원점으로 사용
        self.grid_origin_set = True
        self.grid_count_enabled = True

        self.grid_x = 0
        self.grid_y = 0

        self.points_in_current_sweep = 1
        self.current_sweep_index = 0
        self.completed_sweeps = 0

        self.sweep_dir_x = 1
        self.shift_dir_y = -1

        self.visited_grid_points = []
        self.visited_grid_set = set()
        self.marker_grid_map = {}

        self.pending_aruco_id = None
        self.pending_aruco_ticks = 0

        self.nav_active_target = None
        self.nav_phase = "X"
        self.nav_last_axis = None

        self.record_scan_grid_point()

        self.reset_line_pid()
        self.reset_intersection_detector(wait_until_lost=True)

        self.state = self.S_SCAN_SWEEP

        self.get_logger().info(
            "미션 영역 좌상단 (-12.0, 7.5) 도착 "
            "→ grid=(0,0) 설정 "
            "→ 첫 번째 행 +X 방향 탐색 시작"
        )

    def do_line_trace_to_entry_corner(self):
        self.grid_count_enabled = False

        vx, vy = self.line_follow_velocity(
            move_axis=ENTRY_BEFORE_CORNER_AXIS,
            direction=ENTRY_BEFORE_CORNER_DIR,
            speed_limit=ENTRY_LINE_SPEED,
            track_orientation=ENTRY_BEFORE_CORNER_TRACK
        )

        self.apply_velocity(vx, vy, 0.0)

        if self.detect_entry_corner():
            self.state = self.S_EXIT_ENTRY_CORNER
            self.entry_corner_lost_ticks = 0

            self.reset_line_pid()
            self.reset_intersection_detector(wait_until_lost=False)

            self.get_logger().info(
                "격자 진입 전 L자 코너 감지 → +X 방향으로 코너 탈출 시작"
            )

    def do_exit_entry_corner(self):
        self.grid_count_enabled = False

        vx, vy = self.line_follow_velocity(
            move_axis=ENTRY_AFTER_CORNER_AXIS,
            direction=ENTRY_AFTER_CORNER_DIR,
            speed_limit=ENTRY_CORNER_EXIT_SPEED,
            track_orientation=ENTRY_AFTER_CORNER_TRACK
        )

        self.apply_velocity(vx, vy, 0.0)

        visible = self.detect_entry_corner()

        if visible:
            self.entry_corner_lost_ticks = 0
            return

        self.entry_corner_lost_ticks += 1

        if self.entry_corner_lost_ticks >= INTERSECTION_LOST_REQUIRED_TICKS:
            self.first_line_seen_ticks = 0
            self.first_three_way_ready = False

            self.reset_line_pid()
            self.reset_intersection_detector(wait_until_lost=False)

            self.state = self.S_LINE_TRACE_TO_FIRST_3WAY

            self.get_logger().info(
                "진입선 L자 코너 탈출 완료 → 첫 3-way 격자점 탐색 시작"
            )

    def do_line_trace_to_first_3way(self):
        self.grid_count_enabled = True

        vx, vy = self.line_follow_velocity(
            move_axis=ENTRY_AFTER_CORNER_AXIS,
            direction=ENTRY_AFTER_CORNER_DIR,
            speed_limit=FIRST_THREE_WAY_APPROACH_SPEED,
            track_orientation=ENTRY_AFTER_CORNER_TRACK
        )

        self.apply_velocity(vx, vy, 0.0)

        line_ok = self.has_line(
            move_axis=ENTRY_AFTER_CORNER_AXIS,
            track_orientation=ENTRY_AFTER_CORNER_TRACK
        )

        if line_ok:
            self.first_line_seen_ticks += 1
        else:
            self.first_line_seen_ticks = 0

        if self.first_line_seen_ticks >= FIRST_LINE_REQUIRED_TICKS:
            self.first_three_way_ready = True

        if not self.first_three_way_ready:
            return

        if self.detect_new_first_three_way():
            self.grid_origin_set = True

            self.grid_x = 0
            self.grid_y = 0

            self.points_in_current_sweep = 1
            self.current_sweep_index = 0
            self.completed_sweeps = 0

            self.sweep_dir_x = ENTRY_AFTER_CORNER_DIR
            self.shift_dir_y = -1

            self.visited_grid_points = []
            self.visited_grid_set = set()
            self.marker_grid_map = {}

            self.pending_aruco_id = None
            self.pending_aruco_ticks = 0

            self.nav_active_target = None
            self.nav_phase = "X"
            self.nav_last_axis = None

            self.record_scan_grid_point()

            self.state = self.S_SCAN_SWEEP

            self.reset_line_pid()
            self.reset_intersection_detector(wait_until_lost=True)

            self.get_logger().info(
                "첫 3-way 교차점 감지 완료 → grid=(0,0) 설정 → 격자 탐색 시작"
            )

    def do_scan_sweep(self):
        self.grid_count_enabled = True

        vx, vy = self.line_follow_velocity(
            move_axis="X",
            direction=self.sweep_dir_x,
            track_orientation=GRID_X_TRACK
        )

        self.apply_velocity(vx, vy, 0.0)

        self.update_aruco_candidate()

        if self.detect_new_countable_intersection():
            self.grid_x += self.sweep_dir_x
            self.points_in_current_sweep += 1

            self.record_scan_grid_point()

            if self.points_in_current_sweep >= POINTS_PER_SWEEP:
                self.completed_sweeps += 1

                self.get_logger().info(
                    f"{self.completed_sweeps}/{TOTAL_SWEEPS}번째 줄 완료"
                )

                if self.completed_sweeps >= TOTAL_SWEEPS:
                    self.state = self.S_REVISIT_PREPARE
                    self.revisit_index = 0
                    self.nav_active_target = None
                    self.nav_phase = "X"
                    self.nav_last_axis = None

                    self.reset_line_pid()
                    self.reset_intersection_detector(wait_until_lost=True)

                    self.print_scan_summary()

                    self.get_logger().info(
                        "54개 격자점 탐색 완료 → ArUco 3→2→1→0 역순 재방문 시작"
                    )
                    return

                self.state = self.S_SCAN_SHIFT_NEXT_ROW
                self.reset_line_pid()
                self.reset_intersection_detector(wait_until_lost=True)

                self.get_logger().info(
                    "현재 줄 완료 → 다음 줄로 Y방향 한 칸 이동"
                )

    def do_scan_shift_next_row(self):
        self.grid_count_enabled = True

        vx, vy = self.line_follow_velocity(
            move_axis="Y",
            direction=self.shift_dir_y,
            track_orientation=GRID_Y_TRACK
        )

        self.apply_velocity(vx, vy, 0.0)

        self.update_aruco_candidate()

        if self.detect_new_countable_intersection():
            self.grid_y += self.shift_dir_y

            self.points_in_current_sweep = 1
            self.current_sweep_index += 1

            self.record_scan_grid_point()

            self.sweep_dir_x *= -1

            self.state = self.S_SCAN_SWEEP
            self.reset_line_pid()
            self.reset_intersection_detector(wait_until_lost=True)

            direction_text = "+X" if self.sweep_dir_x > 0 else "-X"

            self.get_logger().info(
                f"다음 줄 도착: grid=({self.grid_x}, {self.grid_y}) "
                f"| 다음 진행 방향 {direction_text}"
            )

    def do_revisit_prepare(self):
        self.grid_count_enabled = True

        if self.revisit_index >= len(REVISIT_SEQUENCE):
            self.state = self.S_RETURN_GRID_HOME
            self.nav_active_target = None
            self.nav_phase = "X"
            self.nav_last_axis = None
            self.reset_line_pid()
            self.reset_intersection_detector(wait_until_lost=True)

            self.get_logger().info(
                "ArUco 3→2→1→0 재방문 완료 → grid=(0,0) 복귀 시작"
            )
            return

        target_id = REVISIT_SEQUENCE[self.revisit_index]
        self.revisit_target_id = target_id

        if target_id not in self.marker_grid_map:
            self.get_logger().warn(
                f"ArUco {target_id}는 탐색 중 저장되지 않음 → 다음 마커로 넘어감"
            )
            self.revisit_index += 1
            return

        target_grid = self.marker_grid_map[target_id]
        self.revisit_target_grid = target_grid

        self.nav_active_target = None
        self.nav_phase = "X"
        self.nav_last_axis = None

        self.reset_line_pid()
        self.reset_intersection_detector(wait_until_lost=True)

        self.get_logger().info(
            f"[REVISIT TARGET] "
            f"ArUco={target_id} "
            f"| current_grid=({self.grid_x},{self.grid_y}) "
            f"| target_grid=({target_grid['grid_x']},{target_grid['grid_y']}) "
            f"| dx={target_grid['grid_x'] - self.grid_x} "
            f"| dy={target_grid['grid_y'] - self.grid_y} "
            f"| plan=X first, then Y"
        )

        self.state = self.S_REVISIT_MOVE

    def do_revisit_move(self):
        self.grid_count_enabled = True

        target_x = self.revisit_target_grid["grid_x"]
        target_y = self.revisit_target_grid["grid_y"]

        arrived = self.navigate_to_grid(target_x, target_y)

        if not arrived:
            return

        self.state = self.S_REVISIT_CENTER
        self.revisit_lost_ticks = 0
        self.revisit_center_hold = 0
        self.grid_count_enabled = False

        self.get_logger().info(
            f"ArUco {self.revisit_target_id} 저장 grid 도착 "
            f"→ grid=({self.grid_x}, {self.grid_y}) "
            f"→ 마커 확인 시작"
        )

    def do_revisit_center(self):
        self.grid_count_enabled = False

        marker = self.detect_aruco(
            exclude_ids={START_MARKER_ID},
            target_id=self.revisit_target_id
        )

        if marker is None:
            self.revisit_lost_ticks += 1
            self.hold_pose()

            if self.revisit_lost_ticks > 35:
                self.prepare_revisit_search()
                self.get_logger().warn(
                    f"ArUco {self.revisit_target_id}가 저장 grid에서 안 보임 "
                    f"→ 주변 격자 탐색 시작"
                )
            return

        self.revisit_lost_ticks = 0
        self.hold_pose()
        self.revisit_center_hold += 1

        if self.revisit_center_hold >= REVISIT_MARKER_CONFIRM_TICKS:
            marker_id = self.revisit_target_id

            self.get_logger().info(
                f"ArUco {marker_id} 재방문 확인 완료 "
                f"→ grid=({self.grid_x}, {self.grid_y})"
            )

            self.start_aruco_hover(marker_id)

            self.revisit_index += 1
            self.state = self.S_REVISIT_PREPARE
            self.revisit_target_id = None
            self.revisit_target_grid = None
            self.revisit_center_hold = 0

            self.nav_active_target = None
            self.nav_phase = "X"
            self.nav_last_axis = None

            self.reset_line_pid()
            self.reset_intersection_detector(wait_until_lost=True)

    def prepare_revisit_search(self):
        self.grid_count_enabled = True

        if self.revisit_target_grid is None:
            self.revisit_index += 1
            self.state = self.S_REVISIT_PREPARE
            return

        center_x = self.revisit_target_grid["grid_x"]
        center_y = self.revisit_target_grid["grid_y"]

        self.search_points = self.make_search_points_around(center_x, center_y)
        self.search_point_index = 0
        self.search_check_ticks = 0

        self.state = self.S_REVISIT_SEARCH_MOVE

        self.nav_active_target = None
        self.nav_phase = "X"
        self.nav_last_axis = None

        self.reset_line_pid()
        self.reset_intersection_detector(wait_until_lost=True)

    def do_revisit_search_move(self):
        self.grid_count_enabled = True

        marker = self.detect_aruco(
            exclude_ids={START_MARKER_ID},
            target_id=self.revisit_target_id
        )

        if marker is not None:
            self.state = self.S_REVISIT_CENTER
            self.revisit_lost_ticks = 0
            self.revisit_center_hold = 0
            self.get_logger().info(
                f"주변 탐색 중 ArUco {self.revisit_target_id} 발견 → 마커 확인"
            )
            return

        if self.search_point_index >= len(self.search_points):
            self.get_logger().warn(
                f"ArUco {self.revisit_target_id} 주변 탐색 실패 → 다음 마커로 넘어감"
            )
            self.revisit_index += 1
            self.state = self.S_REVISIT_PREPARE
            self.hold_pose()
            return

        target_x, target_y = self.search_points[self.search_point_index]

        arrived = self.navigate_to_grid(target_x, target_y)

        if arrived:
            self.state = self.S_REVISIT_SEARCH_CHECK
            self.search_check_ticks = 0

            self.get_logger().info(
                f"주변 탐색 위치 도착: grid=({target_x}, {target_y}) "
                f"| {self.search_point_index + 1}/{len(self.search_points)}"
            )

    def do_revisit_search_check(self):
        self.grid_count_enabled = False

        marker = self.detect_aruco(
            exclude_ids={START_MARKER_ID},
            target_id=self.revisit_target_id
        )

        if marker is not None:
            self.state = self.S_REVISIT_CENTER
            self.revisit_lost_ticks = 0
            self.revisit_center_hold = 0
            self.get_logger().info(
                f"ArUco {self.revisit_target_id} 발견 → 마커 확인"
            )
            return

        self.hold_pose()

        self.search_check_ticks += 1

        if self.search_check_ticks >= self.search_check_max_ticks:
            self.search_point_index += 1
            self.state = self.S_REVISIT_SEARCH_MOVE

            self.nav_active_target = None
            self.nav_phase = "X"
            self.nav_last_axis = None

            self.reset_line_pid()
            self.reset_intersection_detector(wait_until_lost=True)

    def do_return_grid_home(self):
        self.grid_count_enabled = True

        arrived = self.navigate_to_grid(0, 0)

        if arrived:
            self.grid_count_enabled = False
            self.state = self.S_RETURN_START
            self.get_logger().info(
                "grid=(0,0) 복귀 완료 → 시작 지점으로 부드럽게 복귀 시작"
            )

    def do_return_start(self):
        self.grid_count_enabled = False
        self.reset_intersection_detector(wait_until_lost=False)

        arrived = self.move_towards_xy(
            START_X,
            START_Y,
            RETURN_START_SPEED
        )

        if not arrived:
            return

        start_marker = self.detect_aruco(
            exclude_ids=set(),
            target_id=START_MARKER_ID
        )

        if start_marker is not None:
            self.state = self.S_HOME_CENTER
            self.reset_align()
            self.home_center_hold = 0
            self.home_lost_ticks = 0
            self.get_logger().info("시작 ArUco 10 감지 → 중앙정렬 시작")
        else:
            self.state = self.S_LAND
            self.get_logger().warn(
                "시작 위치 도착, ArUco 10 미검출 → 착륙 시작"
            )

    def do_home_center(self):
        self.grid_count_enabled = False

        marker = self.detect_aruco(
            exclude_ids=set(),
            target_id=START_MARKER_ID
        )

        if marker is None:
            self.home_lost_ticks += 1
            self.hold_pose()

            if self.home_lost_ticks > 40:
                self.state = self.S_LAND
                self.get_logger().warn("시작 ArUco 10 정렬 실패 → 착륙 시작")
            return

        self.home_lost_ticks = 0

        h, w = self.current_frame.shape[:2]
        ex = marker["px_x"] - w // 2
        ey = marker["px_y"] - h // 2

        vx, vy, centered = self.align_control(ex, ey)

        if centered:
            self.hold_pose()
            self.home_center_hold += 1
        else:
            self.home_center_hold = 0
            self.apply_velocity(vx, vy, 0.0)

        if self.home_center_hold >= ARUCO_CENTER_HOLD_TICKS:
            self.state = self.S_LAND
            self.get_logger().info("시작 ArUco 10 중앙정렬 완료 → 착륙 시작")

    def do_land(self):
        self.grid_count_enabled = False
        self.reset_intersection_detector(wait_until_lost=False)

        if self.current_z > START_Z:
            self.apply_velocity(0.0, 0.0, LAND_SPEED_Z)
            return

        self.current_z = START_Z
        self.send_pose()
        self.publish_pose()

        self.state = self.S_DONE
        self.get_logger().info("임무 완료: 착륙 종료")

    def navigate_to_grid(self, target_x, target_y):
        target_key = (target_x, target_y)

        if self.grid_x == target_x and self.grid_y == target_y:
            self.hold_pose()

            self.nav_active_target = None
            self.nav_phase = "X"
            self.nav_last_axis = None

            return True

        if self.nav_active_target != target_key:
            self.nav_active_target = target_key
            self.nav_phase = "X"
            self.nav_last_axis = None

            self.reset_line_pid()
            self.reset_intersection_detector(wait_until_lost=True)

            self.get_logger().info(
                f"[NAV PLAN START] "
                f"current=({self.grid_x},{self.grid_y}) "
                f"target=({target_x},{target_y}) "
                f"| phase=X first"
            )

        if self.nav_phase == "X":
            if self.grid_x == target_x:
                self.nav_phase = "Y"
                self.nav_last_axis = None

                self.reset_line_pid()
                self.reset_intersection_detector(wait_until_lost=True)

                self.get_logger().info(
                    f"[NAV PHASE CHANGE] "
                    f"X축 보정 완료 → Y축 보정 시작 "
                    f"| current=({self.grid_x},{self.grid_y}) "
                    f"| target=({target_x},{target_y})"
                )
            else:
                move_axis = "X"
                direction = 1 if target_x > self.grid_x else -1
                track_orientation = GRID_X_TRACK

                return self.navigate_one_axis(
                    target_x=target_x,
                    target_y=target_y,
                    move_axis=move_axis,
                    direction=direction,
                    track_orientation=track_orientation
                )

        if self.nav_phase == "Y":
            if self.grid_y == target_y:
                self.hold_pose()

                self.get_logger().info(
                    f"[NAV ARRIVED] "
                    f"target=({target_x},{target_y}) 도착"
                )

                self.nav_active_target = None
                self.nav_phase = "X"
                self.nav_last_axis = None

                return True

            move_axis = "Y"
            direction = 1 if target_y > self.grid_y else -1
            track_orientation = GRID_Y_TRACK

            return self.navigate_one_axis(
                target_x=target_x,
                target_y=target_y,
                move_axis=move_axis,
                direction=direction,
                track_orientation=track_orientation
            )

        return False

    def navigate_one_axis(
        self,
        target_x,
        target_y,
        move_axis,
        direction,
        track_orientation
    ):
        if self.nav_last_axis != move_axis:
            self.nav_last_axis = move_axis

            self.reset_line_pid()
            self.reset_intersection_detector(wait_until_lost=True)

            if move_axis == "X":
                axis_text = "+X" if direction > 0 else "-X"
            else:
                axis_text = "+Y" if direction > 0 else "-Y"

            self.get_logger().info(
                f"[NAV AXIS START] "
                f"phase={self.nav_phase} "
                f"| selected={axis_text} "
                f"| current=({self.grid_x},{self.grid_y}) "
                f"| target=({target_x},{target_y}) "
                f"| track={track_orientation}"
            )

        vx, vy = self.line_follow_velocity(
            move_axis=move_axis,
            direction=direction,
            track_orientation=track_orientation
        )

        self.log_every(
            "nav_xy_correction",
            (
                f"[NAV XY] "
                f"phase={self.nav_phase} "
                f"| current=({self.grid_x},{self.grid_y}) "
                f"| target=({target_x},{target_y}) "
                f"| axis={move_axis} "
                f"| direction={direction} "
                f"| track={track_orientation} "
                f"| vx={vx:.3f} vy={vy:.3f} "
                f"| wait_lost={self.wait_until_intersection_lost} "
                f"| lost_ticks={self.intersection_lost_ticks}"
            ),
            NAV_DEBUG_LOG_SEC
        )

        self.apply_velocity(vx, vy, 0.0)

        if self.detect_new_countable_intersection():
            before_x = self.grid_x
            before_y = self.grid_y

            if move_axis == "X":
                self.grid_x += direction
            else:
                self.grid_y += direction

            self.get_logger().info(
                f"[NAV COUNT] "
                f"phase={self.nav_phase} "
                f"| axis={move_axis} "
                f"| direction={direction} "
                f"| before=({before_x},{before_y}) "
                f"→ after=({self.grid_x},{self.grid_y}) "
                f"| target=({target_x},{target_y})"
            )

            self.publish_grid_status(
                prefix="NAV",
                extra=f"target=({target_x},{target_y})"
            )

            self.reset_line_pid()
            self.reset_intersection_detector(wait_until_lost=True)

        return False

    def is_grid_count_state(self):
        return self.state in [
            self.S_LINE_TRACE_TO_FIRST_3WAY,
            self.S_SCAN_SWEEP,
            self.S_SCAN_SHIFT_NEXT_ROW,
            self.S_REVISIT_MOVE,
            self.S_REVISIT_SEARCH_MOVE,
            self.S_RETURN_GRID_HOME,
        ]

    def detect_new_first_three_way(self):
        if not self.grid_count_enabled:
            self.reset_intersection_detector(wait_until_lost=False)
            return False

        if self.state != self.S_LINE_TRACE_TO_FIRST_3WAY:
            self.reset_intersection_detector(wait_until_lost=False)
            return False

        visible = self.detect_three_way_intersection()

        return self.handle_intersection_edge(visible)

    def detect_new_countable_intersection(self):
        if not self.grid_count_enabled:
            self.reset_intersection_detector(wait_until_lost=False)
            return False

        if not self.is_grid_count_state():
            self.reset_intersection_detector(wait_until_lost=False)
            return False

        visible = self.detect_grid_intersection()

        return self.handle_intersection_edge(visible)

    def handle_intersection_edge(self, visible):
        if self.wait_until_intersection_lost:
            if visible:
                self.intersection_lost_ticks = 0
                return False

            self.intersection_lost_ticks += 1

            if self.intersection_lost_ticks >= INTERSECTION_LOST_REQUIRED_TICKS:
                self.wait_until_intersection_lost = False
                self.intersection_was_visible = False
                self.intersection_lost_ticks = 0

            return False

        if self.intersection_cooldown > 0:
            self.intersection_cooldown -= 1

        is_new = (
            visible
            and not self.intersection_was_visible
            and self.intersection_cooldown <= 0
        )

        self.intersection_was_visible = visible

        if is_new:
            self.intersection_cooldown = INTERSECTION_COOLDOWN_TICKS
            self.wait_until_intersection_lost = True
            self.intersection_lost_ticks = 0

        return is_new

    def make_search_points_around(self, center_x, center_y):
        points = []
        used = set()

        def add_point(x, y):
            key = (x, y)

            if key in used:
                return

            if key not in self.visited_grid_set:
                return

            used.add(key)
            points.append(key)

        add_point(center_x, center_y)

        for radius in [1, 2]:
            for dx in range(-radius, radius + 1):
                add_point(center_x + dx, center_y + radius)

            for dy in range(radius - 1, -radius - 1, -1):
                add_point(center_x + radius, center_y + dy)

            for dx in range(radius - 1, -radius - 1, -1):
                add_point(center_x + dx, center_y - radius)

            for dy in range(-radius + 1, radius):
                add_point(center_x - radius, center_y + dy)

        if not points:
            points.append((center_x, center_y))

        self.get_logger().info(
            f"ArUco 주변 격자 탐색 후보 {len(points)}개 생성"
        )

        return points

    def record_scan_grid_point(self):
        index = len(self.visited_grid_points) + 1

        item = {
            "index": index,
            "grid_x": self.grid_x,
            "grid_y": self.grid_y,
            "sweep": self.current_sweep_index,
            "point_in_sweep": self.points_in_current_sweep,
            "world_x": self.current_x,
            "world_y": self.current_y
        }

        self.visited_grid_points.append(item)
        self.visited_grid_set.add((self.grid_x, self.grid_y))

        text = (
            f"[SCAN {index}/{TOTAL_GRID_POINTS}] "
            f"grid=({self.grid_x}, {self.grid_y}) "
            f"| sweep={self.current_sweep_index + 1}/{TOTAL_SWEEPS} "
            f"| point={self.points_in_current_sweep}/{POINTS_PER_SWEEP} "
            f"| pose=({self.current_x:.2f}, {self.current_y:.2f})"
        )

        self.get_logger().info(text)

        msg = String()
        msg.data = text
        self.grid_pub.publish(msg)

        self.save_aruco_candidate_at_current_grid()

    def publish_grid_status(self, prefix="GRID", extra=""):
        text = (
            f"[{prefix}] current_grid=({self.grid_x}, {self.grid_y}) "
            f"| pose=({self.current_x:.2f}, {self.current_y:.2f})"
        )

        if extra:
            text += f" | {extra}"

        self.get_logger().info(text)

        msg = String()
        msg.data = text
        self.grid_pub.publish(msg)

    def print_scan_summary(self):
        self.get_logger().info("========== 최종 격자점 방문 기록 ==========")

        for item in self.visited_grid_points:
            self.get_logger().info(
                f"[{item['index']}/{TOTAL_GRID_POINTS}] "
                f"grid=({item['grid_x']}, {item['grid_y']}) "
                f"| sweep={item['sweep'] + 1} "
                f"| point={item['point_in_sweep']} "
                f"| pose=({item['world_x']:.2f}, {item['world_y']:.2f})"
            )

        self.get_logger().info(
            f"총 방문 격자점 수: "
            f"{len(self.visited_grid_points)}/{TOTAL_GRID_POINTS}"
        )

        self.get_logger().info("========== 저장된 ArUco 격자 좌표 ==========")

        for marker_id in sorted(self.marker_grid_map.keys()):
            grid = self.marker_grid_map[marker_id]
            self.get_logger().info(
                f"ArUco {marker_id}: "
                f"grid=({grid['grid_x']}, {grid['grid_y']})"
            )

        missing = sorted(
            list(MISSION_MARKER_IDS - set(self.marker_grid_map.keys()))
        )

        if missing:
            self.get_logger().warn(f"탐색 중 저장하지 못한 ArUco: {missing}")

    def get_centered_mission_aruco_id(self):
        marker = self.detect_aruco(exclude_ids={START_MARKER_ID})

        if marker is None:
            return None

        marker_id = marker["id"]

        if marker_id not in MISSION_MARKER_IDS:
            return None

        if marker_id in self.marker_grid_map:
            return None

        if self.current_frame is None:
            return None

        h, w = self.current_frame.shape[:2]

        ex = abs(marker["px_x"] - w // 2)
        ey = abs(marker["px_y"] - h // 2)

        if ex > ARUCO_SAVE_CENTER_TOL_PX:
            return None

        if ey > ARUCO_SAVE_CENTER_TOL_PX:
            return None

        return marker_id

    def update_aruco_candidate(self):
        if self.pending_aruco_id is not None:
            self.pending_aruco_ticks += 1

            if self.pending_aruco_ticks > ARUCO_PENDING_MAX_TICKS:
                self.pending_aruco_id = None
                self.pending_aruco_ticks = 0

        marker_id = self.get_centered_mission_aruco_id()

        if marker_id is None:
            return

        if self.pending_aruco_id != marker_id:
            self.get_logger().info(
                f"ArUco {marker_id} 후보 감지 → 다음 격자점 카운트 시 저장 예정"
            )

        self.pending_aruco_id = marker_id
        self.pending_aruco_ticks = 0

    def save_aruco_candidate_at_current_grid(self):
        marker_id = self.get_centered_mission_aruco_id()

        if marker_id is None:
            marker_id = self.pending_aruco_id

        if marker_id is None:
            return False

        if marker_id in self.marker_grid_map:
            self.pending_aruco_id = None
            self.pending_aruco_ticks = 0
            return False

        self.marker_grid_map[marker_id] = {
            "grid_x": self.grid_x,
            "grid_y": self.grid_y
        }

        self.pending_aruco_id = None
        self.pending_aruco_ticks = 0

        self.get_logger().info(
            f"ArUco {marker_id} 저장 완료 "
            f"→ grid=({self.grid_x}, {self.grid_y})"
        )

        self.start_aruco_hover(marker_id)

        return True

    def start_aruco_hover(self, marker_id):
        if self.aruco_hover_until > 0.0:
            return

        self.aruco_hover_marker_id = marker_id
        self.aruco_hover_until = time.perf_counter() + ARUCO_DETECT_HOVER_SECONDS

        self.get_logger().info(
            f"ArUco {marker_id} 인식 → 실제 시간 기준 "
            f"{ARUCO_DETECT_HOVER_SECONDS:.1f}초 호버링"
        )

    def line_follow_velocity(
        self,
        move_axis,
        direction,
        speed_limit=None,
        track_orientation=None
    ):
        raw, found = self.line_error(
            move_axis=move_axis,
            track_orientation=track_orientation
        )

        if not found:
            self.line_lost_ticks += 1

            if self.line_lost_ticks % LINE_SEARCH_SWITCH_TICKS == 0:
                self.line_search_dir *= -1

            if self.line_lost_ticks <= LINE_LOST_STOP_TICKS:
                forward_speed = FORWARD_SPEED_MIN * 0.5
            else:
                forward_speed = 0.0

            search = LINE_SEARCH_SPEED * self.line_search_dir

            if move_axis == "Y":
                vx = search
                vy = forward_speed * float(direction)
                return vx, vy

            vx = forward_speed * float(direction)
            vy = search
            return vx, vy

        self.line_lost_ticks = 0

        err = self.filter_error(raw, found)
        steer = self.pid(err)
        speed = self.adaptive_speed()

        if speed_limit is not None:
            speed = min(speed, speed_limit)

        if move_axis == "Y":
            vx = self.sign_y_lateral * steer
            vy = speed * float(direction)
            return vx, vy

        vx = speed * float(direction)
        vy = self.sign_x_lateral * steer
        return vx, vy

    def has_line(self, move_axis, track_orientation=None):
        _, found = self.line_error(
            move_axis=move_axis,
            track_orientation=track_orientation
        )
        return found

    def preprocess_line_image(self):
        if self.current_frame is None:
            return None

        frame = self.current_frame
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        thresh = cv2.adaptiveThreshold(
            gray,
            255,
            cv2.ADAPTIVE_THRESH_MEAN_C,
            cv2.THRESH_BINARY_INV,
            blockSize=31,
            C=8
        )

        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

        green_mask = cv2.inRange(
            hsv,
            np.array([30, 35, 30]),
            np.array([95, 255, 255])
        )

        kernel_g = cv2.getStructuringElement(
            cv2.MORPH_ELLIPSE,
            (7, 7)
        )

        green_mask = cv2.dilate(
            green_mask,
            kernel_g,
            iterations=2
        )

        thresh[green_mask > 0] = 0

        kernel = cv2.getStructuringElement(
            cv2.MORPH_ELLIPSE,
            (5, 5)
        )

        thresh = cv2.morphologyEx(
            thresh,
            cv2.MORPH_CLOSE,
            kernel
        )

        return thresh

    def line_error(self, move_axis, track_orientation=None):
        thresh = self.preprocess_line_image()

        if thresh is None:
            return 0.0, False

        h, w = thresh.shape
        cx = w // 2
        cy = h // 2

        roi_half = LINE_TRACK_ROI_HALF
        length_half = LINE_TRACK_LENGTH_HALF

        if track_orientation is None:
            if move_axis == "Y":
                track_orientation = "V"
            else:
                track_orientation = "H"

        if track_orientation == "V":
            x0 = max(0, cx - roi_half)
            x1 = min(w, cx + roi_half)

            y0 = max(0, cy - length_half)
            y1 = min(h, cy + length_half)

            roi = thresh[y0:y1, x0:x1]

            projection = np.count_nonzero(roi, axis=0)

            line_center_local, found = self.find_center_closest_peak(
                projection
            )

            if not found:
                return 0.0, False

            line_center_x = x0 + line_center_local
            error = line_center_x - cx

            return float(error), True

        y0 = max(0, cy - roi_half)
        y1 = min(h, cy + roi_half)

        x0 = max(0, cx - length_half)
        x1 = min(w, cx + length_half)

        roi = thresh[y0:y1, x0:x1]

        projection = np.count_nonzero(roi, axis=1)

        line_center_local, found = self.find_center_closest_peak(
            projection
        )

        if not found:
            return 0.0, False

        line_center_y = y0 + line_center_local
        error = line_center_y - cy

        return float(error), True

    def find_center_closest_peak(self, projection):
        if projection.size == 0:
            return 0.0, False

        projection = projection.astype(np.float32)

        kernel_size = LINE_SMOOTH_KERNEL
        kernel = np.ones(kernel_size, dtype=np.float32) / float(kernel_size)
        smooth = np.convolve(projection, kernel, mode="same")

        candidates = np.where(smooth >= LINE_PROJECTION_MIN_PIXELS)[0]

        if len(candidates) == 0:
            return 0.0, False

        center_index = len(smooth) // 2

        segments = []
        start = int(candidates[0])
        prev = int(candidates[0])

        for value in candidates[1:]:
            value = int(value)

            if value == prev + 1:
                prev = value
            else:
                segments.append((start, prev))
                start = value
                prev = value

        segments.append((start, prev))

        best_segment = None
        best_distance = 999999.0

        for start, end in segments:
            seg_center = 0.5 * (start + end)
            distance = abs(seg_center - center_index)

            if distance < best_distance:
                best_distance = distance
                best_segment = (start, end)

        if best_segment is None:
            return 0.0, False

        start, end = best_segment

        peak_local = start + int(np.argmax(smooth[start:end + 1]))

        lo = max(0, peak_local - LINE_PEAK_WINDOW)
        hi = min(len(smooth), peak_local + LINE_PEAK_WINDOW + 1)

        weights = smooth[lo:hi]
        coords = np.arange(lo, hi, dtype=np.float32)

        if np.sum(weights) <= 0:
            return float(peak_local), True

        center = np.sum(coords * weights) / np.sum(weights)

        return float(center), True

    def detect_entry_corner(self):
        thresh = self.preprocess_line_image()

        if thresh is None:
            return False

        branches = self.detect_intersection_branches(thresh)

        if not branches["center"]:
            return False

        vertical = branches["up"] or branches["down"]
        horizontal = branches["left"] or branches["right"]

        if not vertical:
            return False

        if not horizontal:
            return False

        return True

    def detect_three_way_intersection(self):
        thresh = self.preprocess_line_image()

        if thresh is None:
            return False

        branches = self.detect_intersection_branches(thresh)

        if not branches["center"]:
            return False

        branch_count = (
            int(branches["left"])
            + int(branches["right"])
            + int(branches["up"])
            + int(branches["down"])
        )

        if branch_count < 3:
            return False

        return True

    def detect_grid_intersection(self):
        thresh = self.preprocess_line_image()

        if thresh is None:
            return False

        branches = self.detect_intersection_branches(thresh)

        if not branches["center"]:
            return False

        horizontal = branches["left"] or branches["right"]
        vertical = branches["up"] or branches["down"]

        if not horizontal or not vertical:
            return False

        return True

    def detect_intersection_branches(self, thresh):
        h, w = thresh.shape

        cx = w // 2
        cy = h // 2

        roi_half = INTERSECTION_ROI_HALF

        x_min = max(0, cx - roi_half)
        x_max = min(w, cx + roi_half)

        y_min = max(0, cy - roi_half)
        y_max = min(h, cy + roi_half)

        def crop_count(x0, x1, y0, y1):
            x0 = max(x_min, min(x_max, x0))
            x1 = max(x_min, min(x_max, x1))
            y0 = max(y_min, min(y_max, y0))
            y1 = max(y_min, min(y_max, y1))

            if x1 <= x0 or y1 <= y0:
                return 0.0, 0

            crop = thresh[y0:y1, x0:x1]
            count = cv2.countNonZero(crop)
            ratio = count / float(crop.size)

            return ratio, count

        center_ratio, center_count = crop_count(
            cx - BRANCH_CENTER_HALF,
            cx + BRANCH_CENTER_HALF,
            cy - BRANCH_CENTER_HALF,
            cy + BRANCH_CENTER_HALF
        )

        left_ratio, _ = crop_count(
            cx - BRANCH_GAP - BRANCH_LENGTH,
            cx - BRANCH_GAP,
            cy - BRANCH_THICK,
            cy + BRANCH_THICK
        )

        right_ratio, _ = crop_count(
            cx + BRANCH_GAP,
            cx + BRANCH_GAP + BRANCH_LENGTH,
            cy - BRANCH_THICK,
            cy + BRANCH_THICK
        )

        up_ratio, _ = crop_count(
            cx - BRANCH_THICK,
            cx + BRANCH_THICK,
            cy - BRANCH_GAP - BRANCH_LENGTH,
            cy - BRANCH_GAP
        )

        down_ratio, _ = crop_count(
            cx - BRANCH_THICK,
            cx + BRANCH_THICK,
            cy + BRANCH_GAP,
            cy + BRANCH_GAP + BRANCH_LENGTH
        )

        return {
            "center": center_count >= CENTER_PIXEL_MIN,
            "left": left_ratio >= BRANCH_RATIO_MIN,
            "right": right_ratio >= BRANCH_RATIO_MIN,
            "up": up_ratio >= BRANCH_RATIO_MIN,
            "down": down_ratio >= BRANCH_RATIO_MIN,
            "left_ratio": left_ratio,
            "right_ratio": right_ratio,
            "up_ratio": up_ratio,
            "down_ratio": down_ratio,
            "center_count": center_count
        }

    def reset_intersection_detector(self, wait_until_lost=False):
        self.intersection_was_visible = False
        self.intersection_cooldown = 0
        self.wait_until_intersection_lost = wait_until_lost
        self.intersection_lost_ticks = 0

    def detect_aruco(self, exclude_ids=None, target_id=None):
        if exclude_ids is None:
            exclude_ids = set()

        if self.current_frame is None:
            return None

        gray = cv2.cvtColor(self.current_frame, cv2.COLOR_BGR2GRAY)

        if self.aruco_detector is not None:
            corners, ids, _ = self.aruco_detector.detectMarkers(gray)
        else:
            corners, ids, _ = cv2.aruco.detectMarkers(
                gray,
                self.aruco_dict,
                parameters=self.aruco_params
            )

        if ids is None:
            return None

        detected = []

        for i, marker_id in enumerate(ids.flatten()):
            marker_id = int(marker_id)

            if marker_id in exclude_ids:
                continue

            if target_id is not None and marker_id != target_id:
                continue

            c = corners[i][0]

            px_x = int(c[:, 0].mean())
            px_y = int(c[:, 1].mean())
            area = cv2.contourArea(c.astype(np.float32))

            detected.append({
                "id": marker_id,
                "px_x": px_x,
                "px_y": px_y,
                "area": area
            })

        if not detected:
            return None

        detected.sort(key=lambda item: item["area"], reverse=True)

        return detected[0]

    def reset_align(self):
        self.align_ex_f = 0.0
        self.align_ey_f = 0.0

    def align_control(self, ex, ey):
        a = self.align_lpf_alpha

        self.align_ex_f = a * ex + (1.0 - a) * self.align_ex_f
        self.align_ey_f = a * ey + (1.0 - a) * self.align_ey_f

        fx = self.align_ex_f
        fy = self.align_ey_f

        ux = 0.0 if abs(fx) < self.align_deadzone else fx
        uy = 0.0 if abs(fy) < self.align_deadzone else fy

        if abs(ux) >= abs(uy):
            gx = self.align_gain
            gy = self.align_gain * 0.5
        else:
            gx = self.align_gain * 0.5
            gy = self.align_gain

        vy = self.sign_align_vy * ux * gx
        vx = self.sign_align_vx * uy * gy

        vx = self.clamp(vx, -self.align_clamp, self.align_clamp)
        vy = self.clamp(vy, -self.align_clamp, self.align_clamp)

        centered = (
            abs(fx) < ARUCO_CENTER_TOL_PX
            and abs(fy) < ARUCO_CENTER_TOL_PX
        )

        return vx, vy, centered

    def reset_line_pid(self):
        self.integral = 0.0
        self.prev_error = 0.0
        self.err_filt = 0.0
        self.no_line_ticks = 0
        self.line_lost_ticks = 0

    def filter_error(self, raw_error, found):
        if found:
            jump = raw_error - self.err_filt

            if abs(jump) > self.jump_limit:
                raw_error = self.err_filt + math.copysign(
                    self.jump_limit,
                    jump
                )

            self.err_filt = (
                self.err_alpha * raw_error
                + (1.0 - self.err_alpha) * self.err_filt
            )

            self.no_line_ticks = 0
        else:
            self.no_line_ticks += 1
            self.err_filt *= 0.90

        return self.err_filt

    def pid(self, error):
        self.integral += error * self.dt
        self.integral = self.clamp(self.integral, -20.0, 20.0)

        derivative = (error - self.prev_error) / self.dt
        self.prev_error = error

        out = (
            self.Kp * error
            + self.Ki * self.integral
            + self.Kd * derivative
        )

        return self.clamp(out, -self.lateral_clamp, self.lateral_clamp)

    def adaptive_speed(self):
        mag = abs(self.err_filt)
        ratio = min(mag / self.err_slowdown_px, 1.0)

        speed = self.speed_max - (self.speed_max - self.speed_min) * ratio

        if mag > 70:
            speed *= 0.65

        if mag > 110:
            speed *= 0.45

        if self.no_line_ticks > 3:
            speed *= 0.5

        if self.no_line_ticks > 8:
            speed = 0.0

        return speed

    def apply_velocity(self, vx, vy, vz):
        vx = self.clamp(vx, -MAX_XY_SPEED, MAX_XY_SPEED)
        vy = self.clamp(vy, -MAX_XY_SPEED, MAX_XY_SPEED)
        vz = self.clamp(vz, -MAX_Z_SPEED, MAX_Z_SPEED)

        self.current_x += vx * self.dt
        self.current_y += vy * self.dt
        self.current_z += vz * self.dt

        if self.current_z < START_Z:
            self.current_z = START_Z

        if self.current_z > FLIGHT_Z:
            self.current_z = FLIGHT_Z

        self.send_pose()
        self.publish_pose()

    def move_towards_xy(self, target_x, target_y, speed):
        dx = target_x - self.current_x
        dy = target_y - self.current_y

        dist = math.hypot(dx, dy)

        if dist < 0.12:
            self.current_x = target_x
            self.current_y = target_y
            self.send_pose()
            self.publish_pose()
            return True

        ux = dx / dist
        uy = dy / dist

        step = speed * self.dt

        if step >= dist:
            self.current_x = target_x
            self.current_y = target_y
            self.send_pose()
            self.publish_pose()
            return True

        self.apply_velocity(ux * speed, uy * speed, 0.0)

        return False

    def hold_pose(self):
        self.send_pose()
        self.publish_pose()

    def send_pose(self):
        # 이전 비동기 요청이 끝나기 전에 새 요청을 무한히 쌓지 않음
        if (
            self.pending_pose_future is not None
            and not self.pending_pose_future.done()
        ):
            return

        req = SetEntityPose.Request()

        req.entity.name = "sim_drone"
        req.entity.type = Entity.MODEL

        req.pose.position.x = float(self.current_x)
        req.pose.position.y = float(self.current_y)
        req.pose.position.z = float(self.current_z)

        qx, qy, qz, qw = self.yaw_to_quaternion(self.current_yaw)

        req.pose.orientation.x = qx
        req.pose.orientation.y = qy
        req.pose.orientation.z = qz
        req.pose.orientation.w = qw

        self.pending_pose_future = self.pose_client.call_async(req)

    def publish_pose(self):
        msg = PoseStamped()

        msg.header.stamp = self.get_clock().now().to_msg()
        msg.header.frame_id = "world"

        msg.pose.position.x = float(self.current_x)
        msg.pose.position.y = float(self.current_y)
        msg.pose.position.z = float(self.current_z)

        qx, qy, qz, qw = self.yaw_to_quaternion(self.current_yaw)

        msg.pose.orientation.x = qx
        msg.pose.orientation.y = qy
        msg.pose.orientation.z = qz
        msg.pose.orientation.w = qw

        self.pose_pub.publish(msg)

    def yaw_to_quaternion(self, yaw):
        half = yaw * 0.5

        qx = 0.0
        qy = 0.0
        qz = math.sin(half)
        qw = math.cos(half)

        return qx, qy, qz, qw

    def clamp(self, value, min_value, max_value):
        return max(min(value, max_value), min_value)

    def log_every(self, key, message, period_sec):
        now = self.get_clock().now().nanoseconds / 1e9
        last = self.last_log_time.get(key, -999.0)

        if now - last >= period_sec:
            self.get_logger().info(message)
            self.last_log_time[key] = now


def main():
    rclpy.init()

    node = GridDirectionSetPoseTracer()

    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass

    node.destroy_node()
    rclpy.shutdown()


if __name__ == "__main__":
    main()