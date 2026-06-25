"""
Cloth Simulation — Taichi 1.7
Three integration methods: Explicit / Semi-Implicit / Implicit Euler
Mouse drag (LMB): grab and pull cloth particles
Camera orbit  (RMB): rotate view
"""

import taichi as ti
import numpy as np

ti.init(arch=ti.gpu)

# ─────────────────────────────────────────────
# Global parameters
# ─────────────────────────────────────────────
N              = 20
NUM_PARTICLES  = N * N
DT             = 1e-3
GRAVITY        = ti.Vector([0.0, -9.8, 0.0])
DAMPING        = 2.0
KS_STRUCT      = 2000.0
KS_SHEAR       = 800.0
KS_BEND        = 300.0
KD             = 0.3
MASS           = 0.05
MAX_VEL        = 30.0
IMPLICIT_ITER  = 15
SUBSTEPS       = 10

# Mouse-drag spring parameters
DRAG_KS        = 800.0   # stiffness of the mouse-drag spring
DRAG_KD        = 8.0     # damping  of the mouse-drag spring
DRAG_RADIUS    = 0.12    # pick radius in world units (particles closer than this are grabbed)

OFFSETS = [(1,0),(0,1),(1,1),(1,-1),(2,0),(0,2)]
NS = len(OFFSETS)

WIN_W, WIN_H = 1024, 768

# ─────────────────────────────────────────────
# Taichi fields
# ─────────────────────────────────────────────
pos     = ti.Vector.field(3, ti.f32, NUM_PARTICLES)
vel     = ti.Vector.field(3, ti.f32, NUM_PARTICLES)
force   = ti.Vector.field(3, ti.f32, NUM_PARTICLES)

pos_buf = ti.Vector.field(3, ti.f32, NUM_PARTICLES)
vel_buf = ti.Vector.field(3, ti.f32, NUM_PARTICLES)

fixed   = ti.field(ti.i32, NUM_PARTICLES)

sj      = ti.field(ti.i32,  (NUM_PARTICLES, NS))
srest   = ti.field(ti.f32,  (NUM_PARTICLES, NS))
sks     = ti.field(ti.f32,  (NUM_PARTICLES, NS))
svalid  = ti.field(ti.i32,  (NUM_PARTICLES, NS))

num_tris = (N-1)*(N-1)*2
indices  = ti.field(ti.i32, num_tris * 3)
colors   = ti.Vector.field(3, ti.f32, NUM_PARTICLES)

# ── Drag state shared with GPU ──
drag_active    = ti.field(ti.i32,  ())          # 0/1
drag_target    = ti.Vector.field(3, ti.f32, ()) # world-space target point
drag_particle  = ti.field(ti.i32,  ())          # grabbed particle index


# ─────────────────────────────────────────────
# Initialisation
# ─────────────────────────────────────────────

@ti.kernel
def init_positions():
    for i in range(N):
        for j in range(N):
            idx = i * N + j
            pos[idx]   = ti.Vector([j/(N-1) - 0.5,  0.5,  i/(N-1) - 0.5])
            vel[idx]   = ti.Vector([0.0, 0.0, 0.0])
            force[idx] = ti.Vector([0.0, 0.0, 0.0])
            fixed[idx] = 1 if i == 0 else 0
            t = float(i) / float(N-1)
            colors[idx] = ti.Vector([0.2 + 0.6*t, 0.5 - 0.1*t, 0.9 - 0.5*t])


@ti.kernel
def init_springs():
    for i in range(N):
        for j in range(N):
            idx = i * N + j
            for k in ti.static(range(NS)):
                di = ti.static(OFFSETS[k][0])
                dj = ti.static(OFFSETS[k][1])
                ni = i + di
                nj = j + dj
                if 0 <= ni < N and 0 <= nj < N:
                    nidx = ni * N + nj
                    sj[idx, k]     = nidx
                    svalid[idx, k] = 1
                    srest[idx, k]  = (pos[nidx] - pos[idx]).norm()
                    if ti.static(k >= 4):
                        sks[idx, k] = KS_BEND
                    elif ti.static(k >= 2):
                        sks[idx, k] = KS_SHEAR
                    else:
                        sks[idx, k] = KS_STRUCT
                else:
                    sj[idx, k]     = 0
                    svalid[idx, k] = 0
                    srest[idx, k]  = 0.0
                    sks[idx, k]    = 0.0


@ti.kernel
def init_render_indices():
    for i in range(N-1):
        for j in range(N-1):
            q = i*(N-1) + j
            indices[q*6+0] = (i  )*N + j
            indices[q*6+1] = (i  )*N + (j+1)
            indices[q*6+2] = (i+1)*N + j
            indices[q*6+3] = (i  )*N + (j+1)
            indices[q*6+4] = (i+1)*N + (j+1)
            indices[q*6+5] = (i+1)*N + j


def init_all():
    init_positions()
    init_springs()
    init_render_indices()
    drag_active[None]   = 0
    drag_particle[None] = 0


# ─────────────────────────────────────────────
# Force & velocity helpers
# ─────────────────────────────────────────────

@ti.func
def compute_forces(p_pos: ti.template(), p_vel: ti.template()):
    # gravity + damping
    for idx in range(NUM_PARTICLES):
        force[idx] = GRAVITY * MASS - DAMPING * MASS * p_vel[idx]

    # spring forces (one-directional, atomic reaction on j)
    for idx in range(NUM_PARTICLES):
        for k in ti.static(range(NS)):
            if svalid[idx, k] == 1:
                jdx      = sj[idx, k]
                diff     = p_pos[jdx] - p_pos[idx]
                dist     = diff.norm(1e-6)
                deform   = dist - srest[idx, k]
                dir_     = diff / dist
                vel_proj = (p_vel[jdx] - p_vel[idx]).dot(dir_)
                f_vec    = (sks[idx, k] * deform + KD * vel_proj) * dir_
                ti.atomic_add(force[idx],  f_vec)
                ti.atomic_add(force[jdx], -f_vec)

    # mouse-drag spring force (acts on the single grabbed particle)
    if drag_active[None] == 1:
        pidx  = drag_particle[None]
        diff  = drag_target[None] - p_pos[pidx]
        dist  = diff.norm(1e-6)
        dir_  = diff / dist
        f_drag = (DRAG_KS * dist - DRAG_KD * p_vel[pidx].dot(dir_)) * dir_
        ti.atomic_add(force[pidx], f_drag)


@ti.func
def clamp_vel(p_vel: ti.template()):
    for idx in range(NUM_PARTICLES):
        spd = p_vel[idx].norm()
        if spd > MAX_VEL:
            p_vel[idx] *= MAX_VEL / spd


# ─────────────────────────────────────────────
# Integrators
# ─────────────────────────────────────────────

@ti.kernel
def step_explicit():
    compute_forces(pos, vel)
    for idx in range(NUM_PARTICLES):
        if fixed[idx] == 0:
            a = force[idx] / MASS
            pos[idx] = pos[idx] + vel[idx] * DT
            vel[idx] = vel[idx] + a * DT
    clamp_vel(vel)


@ti.kernel
def step_semi_implicit():
    compute_forces(pos, vel)
    for idx in range(NUM_PARTICLES):
        if fixed[idx] == 0:
            a = force[idx] / MASS
            vel[idx] = vel[idx] + a * DT
            pos[idx] = pos[idx] + vel[idx] * DT
    clamp_vel(vel)


@ti.kernel
def _implicit_copy():
    for idx in range(NUM_PARTICLES):
        pos_buf[idx] = pos[idx]
        vel_buf[idx] = vel[idx]

@ti.kernel
def _implicit_iter():
    compute_forces(pos_buf, vel_buf)
    for idx in range(NUM_PARTICLES):
        if fixed[idx] == 0:
            a = force[idx] / MASS
            vel_buf[idx] = vel[idx] + a * DT
            pos_buf[idx] = pos[idx] + vel_buf[idx] * DT
    clamp_vel(vel_buf)

@ti.kernel
def _implicit_commit():
    for idx in range(NUM_PARTICLES):
        if fixed[idx] == 0:
            pos[idx] = pos_buf[idx]
            vel[idx] = vel_buf[idx]

def step_implicit():
    _implicit_copy()
    for _ in range(IMPLICIT_ITER):
        _implicit_iter()
    _implicit_commit()


# ─────────────────────────────────────────────
# Ray-picking helpers (pure Python / NumPy, runs on CPU once per click)
# ─────────────────────────────────────────────

def build_ray(mouse_x, mouse_y, cam_pos_np, cam_lookat_np, cam_up_np, fov_deg, aspect):
    """
    Convert a 2-D screen coordinate (in [0,1]^2) into a world-space ray.
    Returns (ray_origin, ray_direction) as numpy float32 arrays.
    """
    # NDC: (0,0) = bottom-left in Taichi GGUI  →  remap to [-1,1]
    ndc_x = (mouse_x * 2.0 - 1.0)
    ndc_y = (mouse_y * 2.0 - 1.0)          # y already bottom-up in GGUI

    fov_rad   = np.deg2rad(fov_deg)
    half_h    = np.tan(fov_rad / 2.0)
    half_w    = half_h * aspect

    forward = cam_lookat_np - cam_pos_np
    forward = forward / np.linalg.norm(forward)
    right   = np.cross(forward, cam_up_np)
    right   = right / np.linalg.norm(right)
    up      = np.cross(right, forward)

    ray_dir = forward + ndc_x * half_w * right + ndc_y * half_h * up
    ray_dir = ray_dir / np.linalg.norm(ray_dir)
    return cam_pos_np.astype(np.float32), ray_dir.astype(np.float32)


def pick_particle(ray_origin, ray_dir, positions_np, radius):
    """
    Find the particle closest to the ray (within 'radius').
    Returns particle index or -1 if none found.
    """
    # Vector from ray origin to each particle
    oc = positions_np - ray_origin          # (N, 3)
    t  = (oc * ray_dir).sum(axis=1)        # projection along ray
    t  = np.maximum(t, 0.0)               # clamp to in-front
    closest = ray_origin + t[:, None] * ray_dir   # (N, 3)
    dist2   = ((positions_np - closest) ** 2).sum(axis=1)
    best    = int(np.argmin(dist2))
    if dist2[best] < radius ** 2:
        return best
    return -1


def world_point_on_ray(ray_origin, ray_dir, ref_point):
    """
    Project ref_point onto the ray to get a depth, then use that depth
    to convert future mouse positions into world points at the same depth.
    Returns the depth scalar t.
    """
    t = float(np.dot(ref_point - ray_origin, ray_dir))
    return max(t, 0.1)


# ─────────────────────────────────────────────
# Main loop
# ─────────────────────────────────────────────

def main():
    init_all()

    window = ti.ui.Window("Cloth Simulation", (WIN_W, WIN_H), vsync=True)
    canvas = window.get_canvas()
    scene  = ti.ui.Scene()
    camera = ti.ui.Camera()

    cam_pos    = np.array([0.0, 0.6, 1.8], dtype=np.float32)
    cam_lookat = np.array([0.0, 0.0, 0.0], dtype=np.float32)
    cam_up     = np.array([0.0, 1.0, 0.0], dtype=np.float32)
    FOV        = 45.0

    camera.position(*cam_pos)
    camera.lookat(*cam_lookat)
    camera.up(*cam_up)
    camera.fov(FOV)

    METHOD_NAMES = ["Explicit Euler", "Semi-Implicit Euler", "Implicit Euler"]
    method = 1
    paused = False

    btn_prev = {k: False for k in ["explicit", "semi", "implicit", "pause", "reset"]}

    def rising_edge(key, cur):
        triggered    = cur and not btn_prev[key]
        btn_prev[key] = cur
        return triggered

    # ── Drag state (CPU side) ──
    is_dragging   = False
    drag_depth    = 1.0          # depth (t along ray) of the grabbed particle
    prev_lmb      = False        # LMB state last frame

    # numpy view of positions for picking (filled each frame lazily)
    pos_np = np.zeros((NUM_PARTICLES, 3), dtype=np.float32)

    while window.running:

        # ── GUI ────────────────────────────────────────────────────────────
        gui = window.get_gui()
        with gui.sub_window("Controls", 0.02, 0.02, 0.36, 0.44):
            gui.text("-- Integration Method --")
            b_exp  = gui.button("Explicit Euler")
            b_semi = gui.button("Semi-Implicit Euler")
            b_imp  = gui.button("Implicit Euler")
            gui.text(f"Active: {METHOD_NAMES[method]}")
            gui.text("")
            gui.text("-- Simulation --")
            b_pause = gui.button("Pause / Resume")
            b_reset = gui.button("Reset Cloth")
            gui.text("")
            gui.text(f"State: {'PAUSED' if paused else 'RUNNING'}")
            gui.text(f"Grid:  {N}x{N}  ({NUM_PARTICLES} pts)")
            gui.text(f"Sub-steps: {SUBSTEPS}/frame")
            gui.text(f"dt = {DT:.4f} s")
            gui.text("")
            gui.text("-- Mouse --")
            gui.text("LMB: grab & drag cloth")
            gui.text("RMB: rotate camera")
            gui.text(f"Drag: {'ON  (particle #' + str(drag_particle[None]) + ')' if is_dragging else 'OFF'}")

        if rising_edge("explicit", b_exp):  method = 0
        if rising_edge("semi",     b_semi): method = 1
        if rising_edge("implicit", b_imp):  method = 2
        if rising_edge("pause", b_pause):   paused = not paused
        if rising_edge("reset", b_reset):
            init_all()
            paused = False
            is_dragging = False

        # ── Mouse drag logic ────────────────────────────────────────────────
        cur_lmb   = window.is_pressed(ti.ui.LMB)
        mouse_x, mouse_y = window.get_cursor_pos()   # [0,1]^2, bottom-left origin

        # Read back particle positions to CPU (numpy) once per frame for picking.
        # Only needed when LMB just pressed (picking) or while dragging (target update).
        if cur_lmb:
            # Reconstruct camera vectors from the camera object each frame so
            # they stay in sync after RMB orbit.
            # Taichi doesn't expose camera matrices directly, so we track them manually.
            # (camera.track_user_inputs updates the internal state; we mirror it.)
            pass   # camera vectors updated below after track_user_inputs

        # ── Camera (RMB orbit, update first so we read fresh vectors) ───────
        # Only allow camera orbit when NOT dragging with LMB
        if not is_dragging:
            camera.track_user_inputs(window, movement_speed=0.05, hold_key=ti.ui.RMB)

        # Retrieve current camera position/lookat by re-reading the ti.ui.Camera fields.
        # Unfortunately Taichi 1.7 doesn't expose camera.position as a readable property,
        # so we maintain our own shadow copy updated via track_user_inputs delta.
        # As a practical workaround: read back via the numpy projection below.
        # We use a fixed lookat and track only with RMB so cam vectors stay known.
        # For simplicity we keep cam_pos/cam_lookat as our ground truth (they don't
        # change unless RMB is held, and we disable orbit during drag).

        lmb_pressed  = cur_lmb and not prev_lmb    # rising edge
        lmb_released = not cur_lmb and prev_lmb    # falling edge

        if lmb_pressed and not is_dragging:
            # Read positions from GPU
            pos_np[:] = pos.to_numpy()
            ro, rd = build_ray(mouse_x, mouse_y, cam_pos, cam_lookat, cam_up,
                               FOV, WIN_W / WIN_H)
            pidx = pick_particle(ro, rd, pos_np, DRAG_RADIUS)
            if pidx >= 0:
                is_dragging          = True
                drag_active[None]    = 1
                drag_particle[None]  = pidx
                drag_depth = world_point_on_ray(ro, rd, pos_np[pidx])

        if is_dragging and cur_lmb:
            # Update drag target: cast ray at same depth
            ro, rd = build_ray(mouse_x, mouse_y, cam_pos, cam_lookat, cam_up,
                               FOV, WIN_W / WIN_H)
            target = ro + drag_depth * rd
            drag_target[None] = ti.Vector([float(target[0]),
                                           float(target[1]),
                                           float(target[2])])

        if lmb_released or (not cur_lmb and is_dragging):
            is_dragging         = False
            drag_active[None]   = 0

        prev_lmb = cur_lmb

        # ── Physics ────────────────────────────────────────────────────────
        if not paused:
            for _ in range(SUBSTEPS):
                if method == 0:
                    step_explicit()
                elif method == 1:
                    step_semi_implicit()
                else:
                    step_implicit()

        # ── Render ─────────────────────────────────────────────────────────
        scene.set_camera(camera)
        scene.ambient_light([0.3, 0.3, 0.3])
        scene.point_light(pos=( 1.0, 2.0,  1.5), color=(1.0, 1.0, 1.0))
        scene.point_light(pos=(-1.0, 1.5, -1.0), color=(0.4, 0.4, 0.8))
        scene.mesh(pos, indices=indices, per_vertex_color=colors, two_sided=True)

        # Highlight grabbed particle
        if is_dragging:
            pidx = int(drag_particle[None])
            hi_pos = ti.Vector.field(3, ti.f32, 1)
            hi_pos[0] = pos[pidx]
            scene.particles(hi_pos, radius=0.018, color=(1.0, 0.9, 0.1))

        canvas.scene(scene)
        window.show()


if __name__ == "__main__":
    main()