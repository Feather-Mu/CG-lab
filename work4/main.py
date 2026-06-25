import taichi as ti

# 初始化 Taichi
ti.init(arch=ti.gpu)

# 窗口分辨率
res_x, res_y = 800, 600
pixels = ti.Vector.field(3, dtype=ti.f32, shape=(res_x, res_y))

# 定义全局交互参数
light_pos = ti.Vector.field(3, dtype=ti.f32, shape=())
max_bounces = ti.field(ti.i32, shape=())

# 初始化参数
light_pos[None] = [2.0, 3.0, 4.0]
max_bounces[None] = 3

@ti.func
def normalize(v):
    return v / v.norm(1e-5)

@ti.func
def reflect(I, N):
    return I - 2.0 * I.dot(N) * N

# 材质类型
MAT_DIFFUSE = 0
MAT_MIRROR = 1

@ti.func
def intersect_plane(ro, rd, y_plane):
    """测试光线与无限大平面相交（平面法线沿Y轴）"""
    t = -1.0
    normal = ti.Vector([0.0, 0.0, 0.0])
    mat_id = MAT_DIFFUSE
    
    if ti.abs(rd.y) > 1e-5:
        t = (y_plane - ro.y) / rd.y
        if t > 0:
            normal = ti.Vector([0.0, 1.0, 0.0])
    
    return t, normal, mat_id

@ti.func
def intersect_sphere(ro, rd, center, radius, mat_id_input):
    """测试光线与球体相交"""
    t = -1.0
    normal = ti.Vector([0.0, 0.0, 0.0])
    mat_id = mat_id_input
    
    oc = ro - center
    b = 2.0 * oc.dot(rd)
    c = oc.dot(oc) - radius * radius
    delta = b * b - 4.0 * c
    
    if delta > 0:
        t1 = (-b - ti.sqrt(delta)) / 2.0
        if t1 > 0:
            t = t1
            p = ro + rd * t
            normal = normalize(p - center)
    
    return t, normal, mat_id

@ti.func
def in_shadow(p, normal, light_pos):
    """检查点 p 是否在阴影中"""
    shadow_ray_dir = normalize(light_pos - p)
    shadow_ray_origin = p + normal * 1e-4  # 偏移避免自相交
    light_dist = (light_pos - p).norm()
    
    in_shadow_result = False
    
    # 检查与平面的阴影遮挡（平面在下方，不会遮挡上方的光）
    # 检查与红色球的阴影遮挡
    t_sph_red, _, _ = intersect_sphere(shadow_ray_origin, shadow_ray_dir, ti.Vector([-1.5, 0.0, 0.0]), 1.0, MAT_DIFFUSE)
    occluded_by_red = (0 < t_sph_red < light_dist)
    in_shadow_result = in_shadow_result or occluded_by_red
    
    # 检查与银色球的阴影遮挡
    t_sph_silver, _, _ = intersect_sphere(shadow_ray_origin, shadow_ray_dir, ti.Vector([1.5, 0.0, 0.0]), 1.0, MAT_MIRROR)
    occluded_by_silver = (0 < t_sph_silver < light_dist)
    in_shadow_result = in_shadow_result or occluded_by_silver
    
    return in_shadow_result

@ti.func
def get_plane_color(p):
    """获取平面的棋盘格颜色"""
    checker_size = 1.0
    x = ti.floor(p.x / checker_size)
    z = ti.floor(p.z / checker_size)
    is_white = ((x + z) % 2) == 0
    color = ti.Vector([0.9, 0.9, 0.9]) if is_white else ti.Vector([0.1, 0.1, 0.1])
    return color

@ti.kernel
def render():
    for i, j in pixels:
        u = (i - res_x / 2.0) / res_y * 2.0
        v = (j - res_y / 2.0) / res_y * 2.0
        
        ro = ti.Vector([0.0, 0.0, 5.0])
        rd = normalize(ti.Vector([u, v, -1.0]))
        
        final_color = ti.Vector([0.0, 0.0, 0.0])
        throughput = 1.0
        bounce_count = 0
        min_t = 1e10  # 在循环外初始化
        
        while bounce_count < max_bounces[None]:
            # 用于记录光线击中的最近物体
            min_t = 1e10  # 每次循环重置
            hit_normal = ti.Vector([0.0, 0.0, 0.0])
            hit_mat_id = MAT_DIFFUSE
            hit_pos = ti.Vector([0.0, 0.0, 0.0])
            hit_color = ti.Vector([0.0, 0.0, 0.0])
            
            # 1. 测试与地面平面的相交
            t_plane, n_plane, mat_plane = intersect_plane(ro, rd, -1.0)
            if t_plane > 0 and t_plane < min_t:
                min_t = t_plane
                hit_normal = n_plane
                hit_mat_id = mat_plane
                hit_pos = ro + rd * t_plane
                hit_color = get_plane_color(hit_pos)
            
            # 2. 测试与红色漫反射球的相交
            t_red, n_red, mat_red = intersect_sphere(ro, rd, ti.Vector([-1.5, 0.0, 0.0]), 1.0, MAT_DIFFUSE)
            if t_red > 0 and t_red < min_t:
                min_t = t_red
                hit_normal = n_red
                hit_mat_id = mat_red
                hit_pos = ro + rd * t_red
                hit_color = ti.Vector([0.8, 0.1, 0.1])
            
            # 3. 测试与银色镜面球的相交
            t_silver, n_silver, mat_silver = intersect_sphere(ro, rd, ti.Vector([1.5, 0.0, 0.0]), 1.0, MAT_MIRROR)
            if t_silver > 0 and t_silver < min_t:
                min_t = t_silver
                hit_normal = n_silver
                hit_mat_id = mat_silver
                hit_pos = ro + rd * t_silver
                hit_color = ti.Vector([0.8, 0.8, 0.8])
            
            # 如果没有击中任何物体，跳出循环
            if min_t >= 1e9:
                break
            
            # 根据材质类型处理
            if hit_mat_id == MAT_DIFFUSE:
                # 漫反射材质：计算光照
                L = normalize(light_pos[None] - hit_pos)
                N = hit_normal
                
                # 环境光
                ambient = 0.1 * hit_color
                
                # 检查阴影
                shadow = in_shadow(hit_pos, N, light_pos[None])
                
                if shadow:
                    # 在阴影中，只计算环境光
                    final_color += throughput * ambient
                else:
                    # 漫反射
                    diff = ti.max(0.0, N.dot(L))
                    diffuse = 0.9 * diff * hit_color
                    final_color += throughput * (ambient + diffuse)
                
                # 终止弹射
                break
            elif hit_mat_id == MAT_MIRROR:
                # 镜面反射材质：更新光线
                ro = hit_pos + hit_normal * 1e-4  # 偏移避免自相交
                rd = reflect(rd, hit_normal)
                throughput *= 0.8  # 反射衰减
                bounce_count += 1
            else:
                # 未知材质，终止弹射
                break
        
        # 设置背景色
        if bounce_count >= max_bounces[None] or min_t >= 1e9:
            # 如果没有击中任何物体，添加背景色
            if final_color.norm() < 1e-5:
                final_color = ti.Vector([0.05, 0.1, 0.15])
        
        pixels[i, j] = ti.math.clamp(final_color, 0.0, 1.0)

def main():
    window = ti.ui.Window("Ray Tracing with Reflection", (res_x, res_y))
    canvas = window.get_canvas()
    gui = window.get_gui()
    
    while window.running:
        # 执行并行渲染
        render()
        
        # 将渲染结果绘制到画布
        canvas.set_image(pixels)
        
        # 绘制交互面板
        with gui.sub_window("Light Position", 0.7, 0.05, 0.28, 0.18):
            light_pos[None][0] = gui.slider_float('Light X', light_pos[None][0], -5.0, 5.0)
            light_pos[None][1] = gui.slider_float('Light Y', light_pos[None][1], 0.5, 6.0)
            light_pos[None][2] = gui.slider_float('Light Z', light_pos[None][2], -5.0, 5.0)
        
        with gui.sub_window("Render Settings", 0.7, 0.25, 0.28, 0.12):
            max_bounces[None] = gui.slider_int('Max Bounces', max_bounces[None], 1, 5)
        
        # 显示窗口
        window.show()

if __name__ == '__main__':
    main()
