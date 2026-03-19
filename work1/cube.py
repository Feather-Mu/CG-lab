import taichi as ti
import math

ti.init(arch=ti.cpu)

# 立方体 8 个顶点
vertices = ti.Vector.field(3, dtype=ti.f32, shape=8)
screen_coords = ti.Vector.field(2, dtype=ti.f32, shape=8)

@ti.func
def get_model_matrix(angle_x: ti.f32, angle_y: ti.f32):
    """绕 X 轴和 Y 轴旋转的模型矩阵"""
    rx = angle_x * math.pi / 180.0
    ry = angle_y * math.pi / 180.0

    cx, sx = ti.cos(rx), ti.sin(rx)
    cy, sy = ti.cos(ry), ti.sin(ry)

    # 绕 X 轴旋转矩阵
    Rx = ti.Matrix([
        [1.0,  0.0, 0.0, 0.0],
        [0.0,   cx, -sx, 0.0],
        [0.0,   sx,  cx, 0.0],
        [0.0,  0.0, 0.0, 1.0]
    ])

    # 绕 Y 轴旋转矩阵
    Ry = ti.Matrix([
        [ cy,  0.0,  sy, 0.0],
        [0.0,  1.0, 0.0, 0.0],
        [-sy,  0.0,  cy, 0.0],
        [0.0,  0.0, 0.0, 1.0]
    ])

    return Ry @ Rx

@ti.func
def get_view_matrix(eye_pos):
    return ti.Matrix([
        [1.0, 0.0, 0.0, -eye_pos[0]],
        [0.0, 1.0, 0.0, -eye_pos[1]],
        [0.0, 0.0, 1.0, -eye_pos[2]],
        [0.0, 0.0, 0.0, 1.0]
    ])

@ti.func
def get_projection_matrix(eye_fov: ti.f32, aspect_ratio: ti.f32, zNear: ti.f32, zFar: ti.f32):
    n = -zNear
    f = -zFar
    fov_rad = eye_fov * math.pi / 180.0
    t = ti.tan(fov_rad / 2.0) * ti.abs(n)
    b = -t
    r = aspect_ratio * t
    l = -r

    M_p2o = ti.Matrix([
        [n,   0.0, 0.0,    0.0],
        [0.0,   n, 0.0,    0.0],
        [0.0, 0.0, n + f, -n * f],
        [0.0, 0.0, 1.0,    0.0]
    ])

    M_ortho_scale = ti.Matrix([
        [2.0 / (r - l), 0.0,           0.0,           0.0],
        [0.0,           2.0 / (t - b), 0.0,           0.0],
        [0.0,           0.0,           2.0 / (n - f), 0.0],
        [0.0,           0.0,           0.0,           1.0]
    ])

    M_ortho_trans = ti.Matrix([
        [1.0, 0.0, 0.0, -(r + l) / 2.0],
        [0.0, 1.0, 0.0, -(t + b) / 2.0],
        [0.0, 0.0, 1.0, -(n + f) / 2.0],
        [0.0, 0.0, 0.0, 1.0]
    ])

    return (M_ortho_scale @ M_ortho_trans) @ M_p2o

@ti.kernel
def compute_transform(angle_x: ti.f32, angle_y: ti.f32):
    eye_pos = ti.Vector([0.0, 0.0, 5.0])
    model = get_model_matrix(angle_x, angle_y)
    view  = get_view_matrix(eye_pos)
    proj  = get_projection_matrix(45.0, 1.0, 0.1, 50.0)
    mvp   = proj @ view @ model

    for i in range(8):
        v  = vertices[i]
        v4 = ti.Vector([v[0], v[1], v[2], 1.0])
        v_clip = mvp @ v4
        v_ndc  = v_clip / v_clip[3]
        screen_coords[i][0] = (v_ndc[0] + 1.0) / 2.0
        screen_coords[i][1] = (v_ndc[1] + 1.0) / 2.0

def main():
    # 边长为 2、中心在原点的立方体顶点（±1）
    #      6 --- 7
    #     /|    /|
    #    4 --- 5 |
    #    | 2 --| 3
    #    |/    |/
    #    0 --- 1
    cube_verts = [
        [-1, -1, -1],  # 0
        [ 1, -1, -1],  # 1
        [-1,  1, -1],  # 2
        [ 1,  1, -1],  # 3
        [-1, -1,  1],  # 4
        [ 1, -1,  1],  # 5
        [-1,  1,  1],  # 6
        [ 1,  1,  1],  # 7
    ]
    for i, v in enumerate(cube_verts):
        vertices[i] = v

    # 立方体的 12 条边（顶点索引对）
    edges = [
        (0, 1), (1, 3), (3, 2), (2, 0),  # 后面
        (4, 5), (5, 7), (7, 6), (6, 4),  # 前面
        (0, 4), (1, 5), (2, 6), (3, 7),  # 连接边
    ]

    # 每条边的颜色
    edge_colors = [
        0xFF4444, 0xFF4444, 0xFF4444, 0xFF4444,  # 后面：红
        0x44FF44, 0x44FF44, 0x44FF44, 0x44FF44,  # 前面：绿
        0x4488FF, 0x4488FF, 0x4488FF, 0x4488FF,  # 连接：蓝
    ]

    gui = ti.GUI("3D Cube (Taichi) — A/D: Y轴旋转  W/S: X轴旋转", res=(700, 700))

    angle_x = 20.0
    angle_y = 30.0

    while gui.running:
        # 键盘事件
        for e in gui.get_events(ti.GUI.PRESS):
            if e.key == 'a':
                angle_y += 10.0
            elif e.key == 'd':
                angle_y -= 10.0
            elif e.key == 'w':
                angle_x += 10.0
            elif e.key == 's':
                angle_x -= 10.0
            elif e.key == ti.GUI.ESCAPE:
                gui.running = False

        compute_transform(angle_x, angle_y)

        # 读取屏幕坐标
        coords = [screen_coords[i] for i in range(8)]

        # 绘制各条边
        for (i, j), color in zip(edges, edge_colors):
            gui.line(coords[i], coords[j], radius=2, color=color)

        gui.show()

if __name__ == '__main__':
    main()