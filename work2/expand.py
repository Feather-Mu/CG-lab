import taichi as ti
import numpy as np

ti.init(arch=ti.gpu)

WIDTH = 800
HEIGHT = 800
MAX_CONTROL_POINTS = 100

# 反走样：3×3 超采样，每像素细分为 9 个子样本
AA_GRID = 3
# B 样条每段采样点数
SAMPLES_PER_SEGMENT = 200
# 曲线总点数上限（最多 97 段，每段 200 点）
MAX_CURVE_POINTS = (MAX_CONTROL_POINTS - 3) * SAMPLES_PER_SEGMENT + 1

# ---------- GPU 缓冲区 ----------
# 累加缓冲：存每个像素被子样本命中的次数（float，方便归一化）
accum = ti.field(dtype=ti.f32, shape=(WIDTH, HEIGHT))

# 最终 RGB 输出
pixels = ti.Vector.field(3, dtype=ti.f32, shape=(WIDTH, HEIGHT))

# 曲线采样点（归一化坐标）
curve_points_field = ti.Vector.field(2, dtype=ti.f32, shape=MAX_CURVE_POINTS)

# GUI 用：控制点显示
gui_points = ti.Vector.field(2, dtype=ti.f32, shape=MAX_CONTROL_POINTS)
gui_indices = ti.field(dtype=ti.i32, shape=MAX_CONTROL_POINTS * 2)


# ============================================================
# CPU：均匀三次 B 样条求值
#   给定 4 个控制点 p0..p3，参数 t∈[0,1]，返回曲线上的点。
#   使用标准均匀三次 B 样条混合矩阵：
#       M = (1/6) * [[-1, 3,-3, 1],
#                    [ 3,-6, 3, 0],
#                    [-3, 0, 3, 0],
#                    [ 1, 4, 1, 0]]
# ============================================================
def bspline_segment(p0, p1, p2, p3, t):
    """均匀三次 B 样条单段求值，t ∈ [0, 1]"""
    t2 = t * t
    t3 = t2 * t
    # 混合系数
    b0 = (-t3 + 3*t2 - 3*t + 1) / 6.0
    b1 = ( 3*t3 - 6*t2         + 4) / 6.0
    b2 = (-3*t3 + 3*t2 + 3*t  + 1) / 6.0
    b3 = ( t3                      ) / 6.0
    x = b0*p0[0] + b1*p1[0] + b2*p2[0] + b3*p3[0]
    y = b0*p0[1] + b1*p1[1] + b2*p2[1] + b3*p3[1]
    return (x, y)


def compute_bspline_points(control_points):
    """
    对全部分段均匀采样，返回 numpy 数组 (N, 2)。
    n 个控制点 → n-3 段（每段需要 4 个连续点）。
    """
    n = len(control_points)
    num_segments = n - 3          # 至少需要 4 个控制点才能有 1 段
    if num_segments <= 0:
        return np.zeros((0, 2), dtype=np.float32)

    pts = []
    for seg in range(num_segments):
        p0 = control_points[seg]
        p1 = control_points[seg + 1]
        p2 = control_points[seg + 2]
        p3 = control_points[seg + 3]
        # 最后一段包含 t=1，其余段不含（避免重复端点）
        end = SAMPLES_PER_SEGMENT + 1 if seg == num_segments - 1 else SAMPLES_PER_SEGMENT
        for k in range(end):
            t = k / SAMPLES_PER_SEGMENT
            pts.append(bspline_segment(p0, p1, p2, p3, t))

    return np.array(pts, dtype=np.float32)


# ============================================================
# GPU Kernels
# ============================================================

@ti.kernel
def clear_buffers():
    """清空累加缓冲与像素缓冲"""
    for i, j in accum:
        accum[i, j] = 0.0
    for i, j in pixels:
        pixels[i, j] = ti.Vector([0.05, 0.05, 0.10])   # 深色背景


@ti.kernel
def draw_curve_aa(n: ti.i32):
    """
    3×3 超采样反走样：
    对曲线的每个采样点，在其归一化坐标对应的 3×3 子像素格中投票。
    每个子格偏移量为 (dx, dy) ∈ {-1/3, 0, 1/3} * (1/WIDTH or 1/HEIGHT)。
    """
    inv_aa = 1.0 / AA_GRID  # 子格间距（归一化）
    half = (AA_GRID - 1) / 2.0  # = 1.0，使偏移居中

    for i in range(n):
        cx = curve_points_field[i][0]
        cy = curve_points_field[i][1]

        # 遍历 3×3 子格
        for di in range(AA_GRID):
            for dj in range(AA_GRID):
                # 子格中心偏移（归一化坐标空间）
                ox = (ti.cast(di, ti.f32) - half) * inv_aa / WIDTH
                oy = (ti.cast(dj, ti.f32) - half) * inv_aa / HEIGHT

                sx = cx + ox
                sy = cy + oy

                px = ti.cast(sx * WIDTH,  ti.i32)
                py = ti.cast(sy * HEIGHT, ti.i32)

                if 0 <= px < WIDTH and 0 <= py < HEIGHT:
                    # 原子加法，避免 GPU 并发写冲突
                    ti.atomic_add(accum[px, py], 1.0)


@ti.kernel
def resolve_pixels():
    """
    将累加缓冲归一化写入 pixels：
    最大命中数 = AA_GRID² = 9（曲线完全覆盖该像素时）。
    归一化后作为曲线颜色的 alpha 混合因子。
    """
    max_hits = ti.cast(AA_GRID * AA_GRID, ti.f32)  # = 9.0
    curve_color = ti.Vector([0.20, 1.0, 0.55])      # 亮绿青色
    bg_color    = ti.Vector([0.05, 0.05, 0.10])      # 深色背景

    for i, j in pixels:
        hits = accum[i, j]
        if hits > 0.0:
            alpha = ti.min(hits / max_hits, 1.0)
            pixels[i, j] = alpha * curve_color + (1.0 - alpha) * bg_color


# ============================================================
# 主循环
# ============================================================
def main():
    window = ti.ui.Window("Uniform Cubic B-Spline  |  3×3 AA  |  Click to add points  |  C = clear",
                          (WIDTH, HEIGHT))
    canvas = window.get_canvas()
    control_points = []

    while window.running:
        # 事件处理
        for e in window.get_events(ti.ui.PRESS):
            if e.key == ti.ui.LMB:
                if len(control_points) < MAX_CONTROL_POINTS:
                    pos = window.get_cursor_pos()
                    control_points.append(list(pos))
                    print(f"[+] Control point #{len(control_points)}: ({pos[0]:.4f}, {pos[1]:.4f})")
            elif e.key == 'c':
                control_points = []
                print("[C] Canvas cleared.")

        # --- 清空 ---
        clear_buffers()

        current_count = len(control_points)

        # --- 需要至少 4 个控制点才能绘制 B 样条 ---
        if current_count >= 4:
            # CPU：计算所有采样点
            curve_np = compute_bspline_points(control_points)
            total_pts = len(curve_np)

            if total_pts > 0 and total_pts <= MAX_CURVE_POINTS:
                # 上传到 GPU（1 次 DMA）
                curve_points_field.from_numpy(
                    np.pad(curve_np,
                           ((0, MAX_CURVE_POINTS - total_pts), (0, 0)),
                           mode='constant')
                )
                # GPU 并行：3×3 超采样投票
                draw_curve_aa(total_pts)
                # GPU 并行：归一化写入像素
                resolve_pixels()

        # --- 输出帧 ---
        canvas.set_image(pixels)

        # --- 绘制控制点（红点）---
        if current_count > 0:
            np_pts = np.full((MAX_CONTROL_POINTS, 2), -10.0, dtype=np.float32)
            np_pts[:current_count] = np.array(control_points, dtype=np.float32)
            gui_points.from_numpy(np_pts)
            canvas.circles(gui_points, radius=0.005, color=(1.0, 0.3, 0.3))

            # --- 绘制控制多边形（灰色虚线感骨架）---
            if current_count >= 2:
                idx_list = []
                for i in range(current_count - 1):
                    idx_list.extend([i, i + 1])
                np_idx = np.zeros(MAX_CONTROL_POINTS * 2, dtype=np.int32)
                np_idx[:len(idx_list)] = idx_list
                gui_indices.from_numpy(np_idx)
                canvas.lines(gui_points, width=0.0015,
                             indices=gui_indices, color=(0.4, 0.4, 0.5))

            # --- 提示：控制点不足时给出提示 ---
            if current_count < 4:
                remaining = 4 - current_count
                print(f"\r  需要再添加 {remaining} 个控制点才能绘制曲线…", end="", flush=True)

        window.show()


if __name__ == '__main__':
    main()