from pathlib import Path
import math
import cv2
import numpy as np

BASE = Path(__file__).resolve().parents[1]
WORLD_PATH = BASE / "worlds" / "indoor_mission.world.sdf"

SAFE_LENGTH = 32.0
SAFE_WIDTH = 23.0
MISSION_LENGTH = 24.0
MISSION_WIDTH = 15.0
GRID_SPACING = 3.0
LINE_WIDTH = 0.10
LINE_HEIGHT = 0.012
VERTIPORT_DIAMETER = 3.0
VERTIPORT_RADIUS = VERTIPORT_DIAMETER / 2.0
VERTIPORT_HEIGHT = 0.7

# New Position: Attached exactly to the top-left corner of the Mission Area
START_X = -13.5  # 미션 구역 경계(-12.0)에서 헬리패드 반지름(1.5)만큼만 왼쪽으로 배치
START_Y = 7.5
START_Z = VERTIPORT_HEIGHT + 0.002

MARKER_SIZE = 0.70 
MARKER_BOARD_SIZE = 0.90

MARKERS = [
    {"id": 0, "x": -9.0, "y": -4.5},
    {"id": 1, "x": -3.0, "y": 1.5},
    {"id": 2, "x": 3.0, "y": 4.5},
    {"id": 3, "x": 9.0, "y": -1.5}
]

def box_model(name, pose, size, color, collision=False):
    collision_block = ""
    if collision:
        collision_block = f"""
        <collision name="collision">
            <geometry><box><size>{size}</size></box></geometry>
        </collision>
        """
    return f"""
    <model name="{name}">
        <static>true</static>
        <pose>{pose}</pose>
        <link name="link">
            <visual name="visual">
                <geometry><box><size>{size}</size></box></geometry>
                <material><ambient>{color}</ambient><diffuse>{color}</diffuse></material>
            </visual>
            {collision_block}
        </link>
    </model>
    """

def cylinder_model(name, pose, radius, length, color, collision=False):
    collision_block = ""
    if collision:
        collision_block = f"""
        <collision name="collision">
            <geometry><cylinder><radius>{radius}</radius><length>{length}</length></cylinder></geometry>
        </collision>
        """
    return f"""
    <model name="{name}">
        <static>true</static>
        <pose>{pose}</pose>
        <link name="link">
            <visual name="visual">
                <geometry><cylinder><radius>{radius}</radius><length>{length}</length></cylinder></geometry>
                <material><ambient>{color}</ambient><diffuse>{color}</diffuse></material>
            </visual>
            {collision_block}
        </link>
    </model>
    """

def ring_boxes(name_prefix, center_x, center_y, z, radius, thickness, color, count=96):
    sdf = ""
    segment_length = (2.0 * math.pi * radius) / count
    for i in range(count):
        theta = i * (2.0 * math.pi / count)
        x = center_x + radius * math.cos(theta)
        y = center_y + radius * math.sin(theta)
        yaw = theta + math.pi / 2.0
        sdf += box_model(f"{name_prefix}_{i}", f"{x} {y} {z} 0 0 {yaw}", f"{segment_length} {thickness} 0.015", color, False)
    return sdf

def make_aruco_cells(marker_id):
    aruco_dict = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_50)
    marker_px = 600
    img = np.zeros((marker_px, marker_px), dtype=np.uint8)
    if hasattr(cv2.aruco, "generateImageMarker"):
        img = cv2.aruco.generateImageMarker(aruco_dict, marker_id, marker_px)
    else:
        cv2.aruco.drawMarker(aruco_dict, marker_id, marker_px, img, 1)
    
    cells = []
    cell_count = 6
    cell_px = marker_px // cell_count
    for r in range(cell_count):
        row = []
        for c in range(cell_count):
            x = int((c + 0.5) * cell_px)
            y = int((r + 0.5) * cell_px)
            value = img[y, x]
            row.append(1 if value > 128 else 0)
        cells.append(row)
    return cells

def aruco_marker_sdf(marker_id, center_x, center_y):
    sdf = ""
    sdf += box_model(f"aruco_{marker_id}_white_board", f"{center_x} {center_y} 0.015 0 0 0", f"{MARKER_BOARD_SIZE} {MARKER_BOARD_SIZE} 0.004", "1 1 1 1", False)
    cells = make_aruco_cells(marker_id)
    cell_count = 6
    cell_size = MARKER_SIZE / cell_count
    start_x = center_x - MARKER_SIZE / 2.0 + cell_size / 2.0
    start_y = center_y + MARKER_SIZE / 2.0 - cell_size / 2.0
    for r in range(cell_count):
        for c in range(cell_count):
            color = "1 1 1 1" if cells[r][c] == 1 else "0 0 0 1"
            x = start_x + c * cell_size
            y = start_y - r * cell_size
            sdf += box_model(f"aruco_{marker_id}_cell_{r}_{c}", f"{x} {y} 0.018 0 0 0", f"{cell_size} {cell_size} 0.002", color, False)
    return sdf

def aruco_marker_on_vertiport(marker_id, center_x, center_y):
    z_base = VERTIPORT_HEIGHT + 0.002
    sdf = ""
    small_marker_size = 0.50
    small_board_size = 0.62
    sdf += box_model("start_vertiport_center_marker_white_board", f"{center_x} {center_y} {z_base} 0 0 0", f"{small_board_size} {small_board_size} 0.002", "1 1 1 1", False)
    cells = make_aruco_cells(marker_id)
    cell_count = 6
    cell_size = small_marker_size / cell_count
    start_x = center_x - small_marker_size / 2.0 + cell_size / 2.0
    start_y = center_y + small_marker_size / 2.0 - cell_size / 2.0
    for r in range(cell_count):
        for c in range(cell_count):
            color = "1 1 1 1" if cells[r][c] == 1 else "0 0 0 1"
            x = start_x + c * cell_size
            y = start_y - r * cell_size
            sdf += box_model(f"start_vertiport_marker_cell_{r}_{c}", f"{x} {y} {z_base + 0.002} 0 0 0", f"{cell_size} {cell_size} 0.003", color, False)
    return sdf

def vertiport_sdf():
    sdf = ""
    sdf += cylinder_model("start_vertiport_base", f"{START_X} {START_Y} {VERTIPORT_HEIGHT / 2.0} 0 0 0", VERTIPORT_RADIUS, VERTIPORT_HEIGHT, "1 1 1 1", True)
    top_z = VERTIPORT_HEIGHT + 0.001
    sdf += ring_boxes("start_vertiport_black_outer_ring", START_X, START_Y, top_z, 1.35, 0.08, "0 0 0 1", 96)
    sdf += ring_boxes("start_vertiport_red_safety_ring", START_X, START_Y, top_z + 0.002, 1.00, 0.06, "1 0 0 1", 96)
    sdf += box_model("start_vertiport_v_left", f"{START_X - 0.05} {START_Y - 0.32} {top_z + 0.005} 0 0 1.2208", "0.12 1.45 0.005", "0 0 0 1", False)
    sdf += box_model(
        "start_vertiport_v_right", 
        f"{START_X - 0.05} {START_Y + 0.32} {top_z + 0.005} 0 0 1.9208", 
        "0.12 1.45 0.005", 
        "0 0 0 1", 
        False
    )    
    sdf += aruco_marker_on_vertiport(10, START_X, START_Y)
    return sdf

def safe_area_boundary_sdf():
    z = 0.005
    thickness = 0.14
    sdf = ""
    sdf += box_model("safe_boundary_top", f"0 {SAFE_WIDTH / 2.0} {z} 0 0 0", f"{SAFE_LENGTH} {thickness} 0.01", "0.0 0.6 0.0 1", False)
    sdf += box_model("safe_bosundary_bottom", f"0 {-SAFE_WIDTH / 2.0} {z} 0 0 0", f"{SAFE_LENGTH} {thickness} 0.01", "0.0 0.6 0.0 1", False)
    sdf += box_model("safe_boundary_left", f"{-SAFE_LENGTH / 2.0} 0 {z} 0 0 0", f"{thickness} {SAFE_WIDTH} 0.01", "0.0 0.6 0.0 1", False)
    sdf += box_model("safe_boundary_right", f"{SAFE_LENGTH / 2.0} 0 {z} 0 0 0", f"{thickness} {SAFE_WIDTH} 0.01", "0.0 0.6 0.0 1", False)
    return sdf

def mission_boundary_sdf():
    z = 0.006
    thickness = LINE_WIDTH
    sdf = ""
    sdf += box_model("mission_boundary_top", f"0 {MISSION_WIDTH / 2.0} {z} 0 0 0", f"{MISSION_LENGTH} {thickness} {LINE_HEIGHT}", "0 0 0 1", False)
    sdf += box_model("mission_boundary_bottom", f"0 {-MISSION_WIDTH / 2.0} {z} 0 0 0", f"{MISSION_LENGTH} {thickness} {LINE_HEIGHT}", "0 0 0 1", False)
    sdf += box_model("mission_boundary_left", f"{-MISSION_LENGTH / 2.0} 0 {z} 0 0 0", f"{thickness} {MISSION_WIDTH} {LINE_HEIGHT}", "0 0 0 1", False)
    sdf += box_model("mission_boundary_right", f"{MISSION_LENGTH / 2.0} 0 {z} 0 0 0", f"{thickness} {MISSION_WIDTH} {LINE_HEIGHT}", "0 0 0 1", False)
    return sdf

def mission_grid_sdf():
    sdf = ""
    x = -MISSION_LENGTH / 2.0 + GRID_SPACING
    idx = 0
    while x < MISSION_LENGTH / 2.0 - 1e-6:
        sdf += box_model(f"grid_vertical_{idx}", f"{x} 0 0.006 0 0 0", f"{LINE_WIDTH} {MISSION_WIDTH} {LINE_HEIGHT}", "0 0 0 1", False)
        x += GRID_SPACING
        idx += 1
    y = -MISSION_WIDTH / 2.0 + GRID_SPACING
    idx = 0
    while y < MISSION_WIDTH / 2.0 - 1e-6:
        sdf += box_model(f"grid_horizontal_{idx}", f"0 {y} 0.006 0 0 0", f"{MISSION_LENGTH} {LINE_WIDTH} {LINE_HEIGHT}", "0 0 0 1", False)
        y += GRID_SPACING
        idx += 1
    return sdf

def connect_line_sdf():
    """
    No connection line needed as the helipad is attached to the mission area.
    """
    sdf = ""
    
    # 연결선을 생성하는 코드를 삭제하여 아무것도 그리지 않음
    
    return sdf

def sim_drone_camera_model():
    arm_length = 0.32  
    arm_thick = 0.03
    arm_height = 0.02
    cos45 = 0.7071
    offset = arm_length * cos45
    z_offset = 0.015 

    return f"""
    <model name="sim_drone">
        <static>true</static>
        <pose>{START_X} {START_Y} {START_Z} 0 0 0</pose>
        <link name="link">
            <visual name="f450_center_hub">
                <pose>0 0 {z_offset} 0 0 0</pose>
                <geometry><box><size>0.15 0.15 0.03</size></box></geometry>
                <material><ambient>0.1 0.1 0.1 1</ambient><diffuse>0.1 0.1 0.1 1</diffuse></material>
            </visual>
            <visual name="arm_front_right">
                <pose>{offset/2} {-offset/2} {z_offset} 0 0 -0.7854</pose>
                <geometry><box><size>{arm_length} {arm_thick} {arm_height}</size></box></geometry>
                <material><ambient>1 0 0 1</ambient><diffuse>1 0 0 1</diffuse></material>
            </visual>
            <visual name="arm_front_left">
                <pose>{offset/2} {offset/2} {z_offset} 0 0 0.7854</pose>
                <geometry><box><size>{arm_length} {arm_thick} {arm_height}</size></box></geometry>
                <material><ambient>1 0 0 1</ambient><diffuse>1 0 0 1</diffuse></material>
            </visual>
            <visual name="arm_rear_right">
                <pose>{-offset/2} {-offset/2} {z_offset} 0 0 0.7854</pose>
                <geometry><box><size>{arm_length} {arm_thick} {arm_height}</size></box></geometry>
                <material><ambient>1 1 1 1</ambient><diffuse>1 1 1 1</diffuse></material>
            </visual>
            <visual name="arm_rear_left">
                <pose>{-offset/2} {offset/2} {z_offset} 0 0 -0.7854</pose>
                <geometry><box><size>{arm_length} {arm_thick} {arm_height}</size></box></geometry>
                <material><ambient>1 1 1 1</ambient><diffuse>1 1 1 1</diffuse></material>
            </visual>
            <visual name="rotor_front_right">
                <pose>{offset} {-offset} {z_offset + 0.01} 0 0 0</pose>
                <geometry><cylinder><radius>0.12</radius><length>0.01</length></cylinder></geometry>
                <material><ambient>0.2 0.2 0.2 0.6</ambient><diffuse>0.2 0.2 0.2 0.6</diffuse></material>
            </visual>
            <visual name="rotor_front_left">
                <pose>{offset} {offset} {z_offset + 0.01} 0 0 0</pose>
                <geometry><cylinder><radius>0.12</radius><length>0.01</length></cylinder></geometry>
                <material><ambient>0.2 0.2 0.2 0.6</ambient><diffuse>0.2 0.2 0.2 0.6</diffuse></material>
            </visual>
            <visual name="rotor_rear_right">
                <pose>{-offset} {-offset} {z_offset + 0.01} 0 0 0</pose>
                <geometry><cylinder><radius>0.12</radius><length>0.01</length></cylinder></geometry>
                <material><ambient>0.2 0.2 0.2 0.6</ambient><diffuse>0.2 0.2 0.2 0.6</diffuse></material>
            </visual>
            <visual name="rotor_rear_left">
                <pose>{-offset} {offset} {z_offset + 0.01} 0 0 0</pose>
                <geometry><cylinder><radius>0.12</radius><length>0.01</length></cylinder></geometry>
                <material><ambient>0.2 0.2 0.2 0.6</ambient><diffuse>0.2 0.2 0.2 0.6</diffuse></material>
            </visual>
            <sensor name="sim_drone_camera" type="camera">
                <always_on>true</always_on><update_rate>30</update_rate>
                <topic>/sim_drone_camera/image</topic>
                <pose>0 0 0 0 1.5708 0</pose>
                <camera>
                    <horizontal_fov>1.5708</horizontal_fov>
                    <image><width>640</width><height>480</height><format>R8G8B8</format></image>
                    <clip><near>0.05</near><far>40.0</far></clip>
                </camera>
            </sensor>
        </link>
    </model>
    """

def main():
    WORLD_PATH.parent.mkdir(parents=True, exist_ok=True)
    world = """<?xml version="1.0"?>
    <sdf version="1.9">
    <world name="indoor_mission">
        <scene>
            <ambient>0.95 0.95 0.95 1</ambient><background>0.95 0.95 0.95 1</background>
            <shadows>false</shadows><grid>false</grid><origin_visual>false</origin_visual>
        </scene>
        
        <light type="directional" name="sun">
            <cast_shadows>false</cast_shadows>
            <pose>0 0 20 0 0 0</pose>
            <diffuse>1.0 1.0 1.0 1</diffuse>
            <specular>0.3 0.3 0.3 1</specular>
            <direction>0.0 0.0 -1.0</direction>
        </light>
        
        <plugin filename="ignition-gazebo-physics-system" name="ignition::gazebo::systems::Physics"/>
        <plugin filename="ignition-gazebo-user-commands-system" name="ignition::gazebo::systems::UserCommands"/>
        <plugin filename="ignition-gazebo-scene-broadcaster-system" name="ignition::gazebo::systems::SceneBroadcaster"/>
        <plugin filename="ignition-gazebo-sensors-system" name="ignition::gazebo::systems::Sensors"><render_engine>ogre</render_engine></plugin>
    """
    
    world += box_model("safe_area_floor", "0 0 -0.01 0 0 0", f"{SAFE_LENGTH} {SAFE_WIDTH} 0.02", "0.88 0.96 0.88 1", True)
    world += box_model("mission_area_floor", "0 0 0 0 0 0", f"{MISSION_LENGTH} {MISSION_WIDTH} 0.002", "0.86 0.80 0.65 1", True)
    world += safe_area_boundary_sdf()
    world += mission_boundary_sdf()
    world += mission_grid_sdf()
    world += connect_line_sdf()  
    for marker in MARKERS:
        world += aruco_marker_sdf(marker["id"], marker["x"], marker["y"])
    world += vertiport_sdf()
    world += sim_drone_camera_model()
    world += "</world></sdf>"
    WORLD_PATH.write_text(world, encoding="utf-8")
    print(f"Perfect layer world saved: {WORLD_PATH}")

if __name__ == "__main__":
    main()