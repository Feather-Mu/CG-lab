import taichi as ti
import math

ti.init(arch=ti.cpu)

# ─── 几何数据 ───────────────────────────────────────────────
vertices    = ti.Vector.field(3, dtype=ti.f32, shape=8)
screen_coords = ti.Vector.field(2, dtype=ti.f32, shape=8)

# 两个姿态的四元数（由主程序写入）
quat_a = ti.Vector.field(4, dtype=ti.f32, shape=())  # 姿态 A
quat_b = ti.Vector.field(4, dtype=ti.f32, shape=())  # 姿态 B
t_field = ti.field(dtype=ti.f32, shape=())            # 插值进度 [0, 1]

# ─── 工具函数 ──────────────────────────────────────────────
@ti.func
def euler_to_quat(ax: ti.f32, ay: ti.f32, az: ti.f32) -> ti.Vector:
    """
    欧拉角（度）→ 四元数 (x, y, z, w)
    旋转顺序：先 X，再 Y，最后 Z
    """
    hx = ax * math.pi / 360.0
    hy = ay * math.pi / 360.0
    hz = az * math.pi / 360.0

    cx, sx = ti.cos(hx), ti.sin(hx)
    cy, sy = ti.cos(hy), ti.sin(hy)
    cz, sz = ti.cos(hz), ti.sin(hz)

    return ti.Vector([
        sx*cy*cz - cx*sy*sz,   # x
        cx*sy*cz + sx*cy*sz,   # y
        cx*cy*sz - sx*sy*cz,   # z
        cx*cy*cz + sx*sy*sz,   # w
    ])

@ti.func
def quat_normalize(q: ti.template()) -> ti.Vector:
    return q / q.norm()

@ti.func
def slerp(qa: ti.template(), qb: ti.template(), t: ti.f32) -> ti.Vector:
    a = quat_normalize(qa)
    b = quat_normalize(qb)

    dot = a.dot(b)

    if dot < 0.0:
        b = -b
        dot = -dot

    # 用一个变量收集结果，避免分支内 return
    result = ti.Vector([0.0, 0.0, 0.0, 0.0])

    if dot > 0.9995:
        # 接近平行时线性插值
        result = quat_normalize(a + t * (b - a))
    else:
        theta0 = ti.acos(dot)
        theta  = theta0 * t
        sin0   = ti.sin(theta0)
        sin_t  = ti.sin(theta)
        sin_r  = ti.sin(theta0 - theta)
        result = (sin_r / sin0) * a + (sin_t / sin0) * b

    return result

@ti.func
def quat_to_matrix(q: ti.template()) -> ti.Matrix:
    """四元数 → 4×4 旋转矩阵"""
    x, y, z, w = q[0], q[1], q[2], q[3]
    return ti.Matrix([
        [1-2*(y*y+z*z),   2*(x*y-w*z),   2*(x*z+w*y), 0.0],
        [  2*(x*y+w*z), 1-2*(x*x+z*z),   2*(y*z-w*x), 0.0],
        [  2*(x*z-w*y),   2*(y*z+w*x), 1-2*(x*x+y*y), 0.0],
        [          0.0,           0.0,           0.0,  1.0],
    ])

@ti.func
def get_view_matrix(eye_pos: ti.template()) -> ti.Matrix:
    return ti.Matrix([
        [1.0, 0.0, 0.0, -eye_pos[0]],
        [0.0, 1.0, 0.0, -eye_pos[1]],
        [0.0, 0.0, 1.0, -eye_pos[2]],
        [0.0, 0.0, 0.0,          1.0],
    ])

@ti.func
def get_projection_matrix(fov: ti.f32, aspect: ti.f32,
                           zNear: ti.f32, zFar: ti.f32) -> ti.Matrix:
    n = -zNear
    f = -zFar
    fov_rad = fov * math.pi / 180.0
    t_val = ti.tan(fov_rad / 2.0) * ti.abs(n)
    b_val = -t_val
    r_val =  aspect * t_val
    l_val = -r_val

    M_p2o = ti.Matrix([
        [n,   0.0, 0.0,      0.0],
        [0.0,   n, 0.0,      0.0],
        [0.0, 0.0, n+f,   -n*f ],
        [0.0, 0.0, 1.0,      0.0],
    ])
    M_scale = ti.Matrix([
        [2.0/(r_val-l_val), 0.0,               0.0,          0.0],
        [0.0,               2.0/(t_val-b_val),  0.0,          0.0],
        [0.0,               0.0,                2.0/(n-f),    0.0],
        [0.0,               0.0,                0.0,          1.0],
    ])
    M_trans = ti.Matrix([
        [1.0, 0.0, 0.0, -(r_val+l_val)/2.0],
        [0.0, 1.0, 0.0, -(t_val+b_val)/2.0],
        [0.0, 0.0, 1.0, -(n+f)/2.0        ],
        [0.0, 0.0, 0.0,  1.0              ],
    ])
    return (M_scale @ M_trans) @ M_p2o

# ─── 主 Kernel ─────────────────────────────────────────────
@ti.kernel
def compute_transform():
    """
    用 Slerp 插值当前四元数，计算各顶点的屏幕坐标
    """
    t = t_field[None]
    qa = quat_a[None]
    qb = quat_b[None]

    q_interp = slerp(qa, qb, t)
    model    = quat_to_matrix(q_interp)

    eye_pos  = ti.Vector([0.0, 0.0, 6.0])
    view     = get_view_matrix(eye_pos)
    proj     = get_projection_matrix(45.0, 1.0, 0.1, 50.0)
    mvp      = proj @ view @ model

    for i in range(8):
        v  = vertices[i]
        v4 = ti.Vector([v[0], v[1], v[2], 1.0])
        vc = mvp @ v4
        vn = vc / vc[3]
        screen_coords[i][0] = (vn[0] + 1.0) / 2.0
        screen_coords[i][1] = (vn[1] + 1.0) / 2.0

# ─── 辅助：用 Python 计算四元数（用于初始化 field）──────────
def py_euler_to_quat(ax, ay, az):
    """欧拉角（度）→ 四元数 [x, y, z, w]，Python 端计算"""
    hx, hy, hz = math.radians(ax/2), math.radians(ay/2), math.radians(az/2)
    cx, sx = math.cos(hx), math.sin(hx)
    cy, sy = math.cos(hy), math.sin(hy)
    cz, sz = math.cos(hz), math.sin(hz)
    return [
        sx*cy*cz - cx*sy*sz,
        cx*sy*cz + sx*cy*sz,
        cx*cy*sz - sx*sy*cz,
        cx*cy*cz + sx*sy*sz,
    ]

# ─── 主程序 ────────────────────────────────────────────────
def main():
    # 立方体顶点（边长 2，中心原点）
    cube_verts = [
        [-1,-1,-1],[1,-1,-1],[-1,1,-1],[1,1,-1],
        [-1,-1, 1],[1,-1, 1],[-1,1, 1],[1,1, 1],
    ]
    for i, v in enumerate(cube_verts):
        vertices[i] = v

    # 12 条边及颜色
    edges = [
        (0,1),(1,3),(3,2),(2,0),   # 后面
        (4,5),(5,7),(7,6),(6,4),   # 前面
        (0,4),(1,5),(2,6),(3,7),   # 连接
    ]
    edge_colors = [
        0xFF4444,0xFF4444,0xFF4444,0xFF4444,
        0x44FF44,0x44FF44,0x44FF44,0x44FF44,
        0x4488FF,0x4488FF,0x4488FF,0x4488FF,
    ]

    # ── 定义两个姿态（欧拉角，度）──────────────────────────
    POSE_A = (20.0,  30.0,   0.0)   # (X轴, Y轴, Z轴)
    # 姿态 B：大幅旋转，呈现另一个面
    POSE_B = (50.0, 135.0,  45.0)

    qa = py_euler_to_quat(*POSE_A)
    qb = py_euler_to_quat(*POSE_B)

    quat_a[None] = qa
    quat_b[None] = qb
    t_field[None] = 0.0

    # ── 动画状态 ────────────────────────────────────────────
    t        = 0.0       # 当前插值进度 [0, 1]
    speed    = 0.008     # 每帧步进量（可按 +/- 调节）
    playing  = True      # 是否自动播放
    forward  = True      # 播放方向

    gui = ti.GUI(
        "Quaternion Slerp — SPACE:播放/暂停  R:重置  ←→:手动  +/-:速度",
        res=(700, 700)
    )

    while gui.running:
        # ── 事件处理 ────────────────────────────────────────
        for e in gui.get_events(ti.GUI.PRESS):
            if e.key == ti.GUI.ESCAPE:
                gui.running = False
            elif e.key == ' ':
                playing = not playing
            elif e.key == 'r':
                t = 0.0
                forward = True
                playing = False
            elif e.key == ti.GUI.RIGHT:
                t = min(1.0, t + 0.05)
                playing = False
            elif e.key == ti.GUI.LEFT:
                t = max(0.0, t - 0.05)
                playing = False
            elif e.key == '=':
                speed = min(0.05, speed + 0.002)
            elif e.key == '-':
                speed = max(0.001, speed - 0.002)

        # ── 自动播放（乒乓循环）──────────────────────────────
        if playing:
            if forward:
                t += speed
                if t >= 1.0:
                    t = 1.0
                    forward = False
            else:
                t -= speed
                if t <= 0.0:
                    t = 0.0
                    forward = True

        # ── 写入 field 并计算 ────────────────────────────────
        t_field[None] = t
        compute_transform()

        coords = [screen_coords[i] for i in range(8)]

        # ── 绘制立方体 ───────────────────────────────────────
        for (i, j), color in zip(edges, edge_colors):
            gui.line(coords[i], coords[j], radius=2, color=color)

if __name__ == '__main__':
    main()