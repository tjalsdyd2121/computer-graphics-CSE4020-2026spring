from OpenGL.GL import *
from glfw.GLFW import *
import glm
import numpy as np
import ctypes, os

BVH_SCALE          = 0.013

PITCHER_POS        = glm.vec3(0, 0, 0)
PITCHER_ROT_Y      = glm.radians(40)
BATTER_POS         = glm.vec3(0, 0, 18.4)
BATTER_ROT_Y       = glm.radians(230)

BATTER_DELAY       = 2.4
FRAME_DT           = 0.00833333
BATTER_READY_FRAME = 20

BALL_RELEASE_FRAME = 415
BALL_RELEASE_T     = BALL_RELEASE_FRAME * FRAME_DT
BALL_SPEED         = 11.5
BALL_HAND_OFFSET   = glm.vec3(0, 0.15, 0)
BATTER_MAX_FRAME   = 520
LOOP_DURATION      = 8.0

KILLCAM_SPEED      = 0.30
KILLCAM_ORBIT_W    = 0.55

LIGHT_POS          = glm.vec3(5, 20, 9)
JOINT_R            = 0.060
BONE_R             = 0.028
BALL_R             = 0.037

BAT_LENGTH         = 65.0
BAT_RADIUS         = 2.0
BAT_ROT_X          = glm.radians(90.0)

SKIP_JOINTS = {
    'leftEye', 'rightEye',
    'rThumb1','rThumb2','rIndex1','rIndex2',
    'rMid1','rMid2','rRing1','rRing2','rPinky1','rPinky2',
    'lThumb1','lThumb2','lIndex1','lIndex2',
    'lMid1','lMid2','lRing1','lRing2','lPinky1','lPinky2',
}

g_vertex_shader_src_color_attribute = '''
#version 330 core
layout (location = 0) in vec3 vin_pos;
layout (location = 1) in vec3 vin_color;
out vec4 vout_color;
uniform mat4 MVP;
void main() {
    gl_Position = MVP * vec4(vin_pos.xyz, 1.0);
    vout_color = vec4(vin_color, 1.);
}
'''

g_fragment_shader_src_color = '''
#version 330 core
in vec4 vout_color;
out vec4 FragColor;
void main() { FragColor = vout_color; }
'''

g_vertex_shader_src_light = '''
#version 330 core
layout (location = 0) in vec3 vin_pos;
layout (location = 1) in vec3 vin_normal;
out vec3 vout_surface_pos;
out vec3 vout_normal;
uniform mat4 MVP;
uniform mat4 M;
void main() {
    gl_Position = MVP * vec4(vin_pos.xyz, 1.0);
    vout_surface_pos = vec3(M * vec4(vin_pos, 1));
    vout_normal = normalize(mat3(inverse(transpose(M))) * vin_normal);
}
'''

g_fragment_shader_src_light = '''
#version 330 core
in vec3 vout_surface_pos;
in vec3 vout_normal;
out vec4 FragColor;
uniform vec3 material_color;
uniform vec3 light_pos;
uniform vec3 view_pos;
void main() {
    vec3 light_color = vec3(1.0, 1.0, 1.0);
    vec3 light_ambient  = 0.25 * light_color;
    vec3 light_diffuse  = light_color;
    vec3 light_specular = light_color;
    vec3 material_ambient  = material_color;
    vec3 material_diffuse  = material_color;
    vec3 material_specular = vec3(1.0, 1.0, 1.0);
    float material_shininess = 32.0;
    vec3 ambient = light_ambient * material_ambient;
    vec3 normal    = normalize(vout_normal);
    vec3 light_dir = normalize(light_pos - vout_surface_pos);
    float diff = max(dot(normal, light_dir), 0.0);
    vec3 diffuse = diff * light_diffuse * material_diffuse;
    vec3 view_dir    = normalize(view_pos - vout_surface_pos);
    vec3 reflect_dir = reflect(-light_dir, normal);
    float spec = pow(max(dot(view_dir, reflect_dir), 0.0), material_shininess);
    vec3 specular = 0.3 * spec * light_specular * material_specular;
    FragColor = vec4(ambient + diffuse + specular, 1.0);
}
'''


class BVHJoint:
    def __init__(self, name, parent=None):
        self.name     = name
        self.parent   = parent
        self.children = []
        self.offset   = glm.vec3(0)
        self.channels = []
        self.ch_start = 0
        self.is_end   = False
        self.link_transform_from_parent = glm.mat4(1.0)
        self.joint_transform            = glm.mat4(1.0)
        self.global_transform           = glm.mat4(1.0)

    def set_link_transform_from_parent(self):
        self.link_transform_from_parent = glm.translate(self.offset)

    def set_joint_transform(self, frame_data):
        if self.is_end:
            self.joint_transform = glm.mat4(1.0)
            return
        tx = ty = tz = 0.0
        R = glm.mat4(1.0)
        for k, ch in enumerate(self.channels):
            v = frame_data[self.ch_start + k]
            if   ch == 'Xposition': tx = v
            elif ch == 'Yposition': ty = v
            elif ch == 'Zposition': tz = v
            elif ch == 'Xrotation':
                R = R * glm.rotate(glm.radians(v), glm.vec3(1, 0, 0))
            elif ch == 'Yrotation':
                R = R * glm.rotate(glm.radians(v), glm.vec3(0, 1, 0))
            elif ch == 'Zrotation':
                R = R * glm.rotate(glm.radians(v), glm.vec3(0, 0, 1))
        if tx or ty or tz:
            self.joint_transform = glm.translate(glm.vec3(tx, ty, tz)) * R
        else:
            self.joint_transform = R

    def update_tree_global_transform(self):
        if self.parent is not None:
            self.global_transform = (self.parent.global_transform *
                                     self.link_transform_from_parent *
                                     self.joint_transform)
        else:
            self.global_transform = self.link_transform_from_parent * self.joint_transform
        for child in self.children:
            child.update_tree_global_transform()


class BVHMotion:
    def __init__(self):
        self.root       = None
        self.joints     = []
        self.n_frames   = 0
        self.frame_time = FRAME_DT
        self.frames     = []

    @staticmethod
    def parse(filepath):
        m = BVHMotion()
        with open(filepath) as f:
            lines = f.readlines()
        stack, ch_count, i = [], 0, 0
        while i < len(lines):
            tok = lines[i].split()
            if not tok:
                i += 1; continue
            kw = tok[0]
            if kw in ('ROOT', 'JOINT'):
                j = BVHJoint(tok[1], stack[-1] if stack else None)
                if j.parent:
                    j.parent.children.append(j)
                else:
                    m.root = j
                m.joints.append(j)
                stack.append(j)
            elif kw == 'End':
                j = BVHJoint('_end_' + stack[-1].name, stack[-1])
                j.is_end = True
                stack[-1].children.append(j)
                m.joints.append(j)
                stack.append(j)
            elif kw == 'OFFSET':
                if stack:
                    stack[-1].offset = glm.vec3(float(tok[1]),
                                                float(tok[2]),
                                                float(tok[3]))
                    stack[-1].set_link_transform_from_parent()
            elif kw == 'CHANNELS':
                n = int(tok[1])
                stack[-1].channels = tok[2:2+n]
                stack[-1].ch_start = ch_count
                ch_count += n
            elif kw == '}':
                if stack: stack.pop()
            elif kw == 'MOTION':
                i += 1
                m.n_frames   = int(lines[i].split()[1])
                i += 1
                m.frame_time = float(lines[i].split()[2])
                i += 1
                for _ in range(m.n_frames):
                    m.frames.append(list(map(float, lines[i].split())))
                    i += 1
                break
            i += 1
        return m

    def frame_at(self, t):
        return min(int(t / self.frame_time), self.n_frames - 1)

    def apply_frame(self, frame_idx):
        frame_idx = max(0, min(frame_idx, self.n_frames - 1))
        data = self.frames[frame_idx]
        for j in self.joints:
            j.set_joint_transform(data)
        self.root.update_tree_global_transform()

    def joint_by_name(self, name):
        for j in self.joints:
            if j.name == name:
                return j
        return None


def load_shaders(vertex_shader_source, fragment_shader_source):
    vertex_shader = glCreateShader(GL_VERTEX_SHADER)
    glShaderSource(vertex_shader, vertex_shader_source)
    glCompileShader(vertex_shader)
    if not glGetShaderiv(vertex_shader, GL_COMPILE_STATUS):
        infoLog = glGetShaderInfoLog(vertex_shader)
        print(infoLog)
    fragment_shader = glCreateShader(GL_FRAGMENT_SHADER)
    glShaderSource(fragment_shader, fragment_shader_source)
    glCompileShader(fragment_shader)
    if not glGetShaderiv(fragment_shader, GL_COMPILE_STATUS):
        infoLog = glGetShaderInfoLog(fragment_shader)
        print(infoLog)
    shader_program = glCreateProgram()
    glAttachShader(shader_program, vertex_shader)
    glAttachShader(shader_program, fragment_shader)
    glLinkProgram(shader_program)
    if not glGetProgramiv(shader_program, GL_LINK_STATUS):
        infoLog = glGetProgramInfoLog(shader_program)
        print(infoLog)
    glDeleteShader(vertex_shader)
    glDeleteShader(fragment_shader)
    return shader_program


def prepare_vao_phong(verts):
    arr = np.array(verts, dtype=np.float32)
    VAO = glGenVertexArrays(1)
    VBO = glGenBuffers(1)
    glBindVertexArray(VAO)
    glBindBuffer(GL_ARRAY_BUFFER, VBO)
    glBufferData(GL_ARRAY_BUFFER, arr.nbytes, arr, GL_STATIC_DRAW)
    stride = 6 * 4
    glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, stride, None)
    glEnableVertexAttribArray(0)
    glVertexAttribPointer(1, 3, GL_FLOAT, GL_FALSE, stride, ctypes.c_void_p(12))
    glEnableVertexAttribArray(1)
    glBindVertexArray(0)
    return VAO, len(verts) // 6


def prepare_vao_sphere(stacks=8, slices=12):
    verts = []
    def pt(phi, th):
        x = glm.cos(phi) * glm.cos(th)
        y = glm.sin(phi)
        z = glm.cos(phi) * glm.sin(th)
        return [x, y, z, x, y, z]
    for i in range(stacks):
        phi0 = glm.pi()/2 - i     * glm.pi() / stacks
        phi1 = glm.pi()/2 - (i+1) * glm.pi() / stacks
        for j in range(slices):
            th0 = j     * 2*glm.pi() / slices
            th1 = (j+1) * 2*glm.pi() / slices
            p00=pt(phi0,th0); p01=pt(phi0,th1)
            p10=pt(phi1,th0); p11=pt(phi1,th1)
            verts += p00+p10+p01 + p01+p10+p11
    return prepare_vao_phong(verts)


def prepare_vao_cylinder(segments=12):
    verts = []
    for i in range(segments):
        th0 = i     * 2*glm.pi() / segments
        th1 = (i+1) * 2*glm.pi() / segments
        x0,z0 = glm.cos(th0), glm.sin(th0)
        x1,z1 = glm.cos(th1), glm.sin(th1)
        verts += [x0,-1,z0, x0,0,z0,  x1,-1,z1, x1,0,z1,  x0, 1,z0, x0,0,z0]
        verts += [x1,-1,z1, x1,0,z1,  x1, 1,z1, x1,0,z1,  x0, 1,z0, x0,0,z0]
        verts += [0,-1,0, 0,-1,0,  x1,-1,z1, 0,-1,0,  x0,-1,z0, 0,-1,0]
        verts += [0, 1,0, 0, 1,0,  x0, 1,z0, 0, 1,0,  x1, 1,z1, 0, 1,0]
    return prepare_vao_phong(verts)


def prepare_vao_grid(half=30, step=1):
    verts = []
    col = [0.35, 0.35, 0.35]
    for i in range(-half, half+1, step):
        verts += [i,0,-half]+col + [i,0, half]+col
        verts += [-half,0,i]+col + [ half,0,i]+col
    arr = np.array(verts, dtype=np.float32)
    VAO = glGenVertexArrays(1)
    VBO = glGenBuffers(1)
    glBindVertexArray(VAO)
    glBindBuffer(GL_ARRAY_BUFFER, VBO)
    glBufferData(GL_ARRAY_BUFFER, arr.nbytes, arr, GL_STATIC_DRAW)
    stride = 6 * 4
    glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, stride, None)
    glEnableVertexAttribArray(0)
    glVertexAttribPointer(1, 3, GL_FLOAT, GL_FALSE, stride, ctypes.c_void_p(12))
    glEnableVertexAttribArray(1)
    glBindVertexArray(0)
    return VAO, len(verts) // 6


def draw_grid(vao, vertex_count, VP, shader_program, loc_MVP):
    glUseProgram(shader_program)
    glUniformMatrix4fv(loc_MVP, 1, GL_FALSE, glm.value_ptr(VP))
    glBindVertexArray(vao)
    glDrawArrays(GL_LINES, 0, vertex_count)


def draw_node(vao, vertex_count, M, VP, view_pos,
              material_color, shader_program,
              loc_MVP, loc_M, loc_material_color, loc_light_pos, loc_view_pos):
    MVP = VP * M
    glUniformMatrix4fv(loc_MVP,            1, GL_FALSE, glm.value_ptr(MVP))
    glUniformMatrix4fv(loc_M,              1, GL_FALSE, glm.value_ptr(M))
    glUniform3fv(loc_material_color,       1, glm.value_ptr(material_color))
    glUniform3fv(loc_light_pos,            1, glm.value_ptr(LIGHT_POS))
    glUniform3fv(loc_view_pos,             1, glm.value_ptr(view_pos))
    glBindVertexArray(vao)
    glDrawArrays(GL_TRIANGLES, 0, vertex_count)


def bone_matrix(p1, p2, radius):
    d = p2 - p1
    length = glm.length(d)
    if length < 1e-4:
        return None
    mid  = (p1 + p2) * 0.5
    d_n  = d / length
    y_ax = glm.vec3(0, 1, 0)
    cr   = glm.cross(y_ax, d_n)
    cr_l = glm.length(cr)
    if cr_l < 1e-6:
        R = glm.mat4(1.0) if d_n.y > 0 else glm.rotate(glm.pi(), glm.vec3(1,0,0))
    else:
        angle = glm.acos(max(-1.0, min(1.0, float(glm.dot(y_ax, d_n)))))
        R = glm.rotate(angle, cr / cr_l)
    return glm.translate(mid) * R * glm.scale(glm.vec3(radius, length*0.5, radius))


def get_world_pos(joint, world_mat):
    return glm.vec3(world_mat * joint.global_transform * glm.vec4(0, 0, 0, 1))


def draw_skeleton(motion, world_mat,
                  vao_sp, vcnt_sp, vao_cy, vcnt_cy,
                  VP, view_pos, color_joint, color_bone,
                  shader_program,
                  loc_MVP, loc_M, loc_material_color, loc_light_pos, loc_view_pos):
    wp = {}
    for j in motion.joints:
        skip = j.name in SKIP_JOINTS
        skip = skip or (j.is_end and j.parent and j.parent.name in SKIP_JOINTS)
        if not skip:
            wp[j.name] = get_world_pos(j, world_mat)
    glUseProgram(shader_program)
    for j in motion.joints:
        if j.name not in wp:
            continue
        pos = wp[j.name]
        if not j.is_end:
            M = glm.translate(pos) * glm.scale(glm.vec3(JOINT_R))
            draw_node(vao_sp, vcnt_sp, M, VP, view_pos, color_joint,
                      shader_program, loc_MVP, loc_M,
                      loc_material_color, loc_light_pos, loc_view_pos)
        if j.parent and j.parent.name in wp:
            Mb = bone_matrix(wp[j.parent.name], pos, BONE_R)
            if Mb is not None:
                draw_node(vao_cy, vcnt_cy, Mb, VP, view_pos, color_bone,
                          shader_program, loc_MVP, loc_M,
                          loc_material_color, loc_light_pos, loc_view_pos)


def make_world_mat(pos, rot_y):
    return glm.translate(pos) * glm.rotate(rot_y, glm.vec3(0,1,0)) * glm.scale(glm.vec3(BVH_SCALE))


g_cam_mode   = 0
g_cam_r      = 22.0
g_cam_theta  = glm.radians(0)
g_cam_phi    = glm.radians(20)
g_cam_center = glm.vec3(0, 1, 9)
g_P          = glm.mat4()

g_mouse_is_dragged = False
g_mouse_x_pos      = 0.0
g_mouse_y_pos      = 0.0
g_z_is_pressed     = False
g_x_is_pressed     = False

g_anim_t        = 0.0
g_prev_real     = 0.0
g_killcam_mode  = 0
g_killcam_orbit = 0.0


def key_callback(window, key, scancode, action, mods):
    global g_cam_mode, g_killcam_mode, g_killcam_orbit
    global g_anim_t, g_prev_real, g_z_is_pressed, g_x_is_pressed

    if action == GLFW_PRESS:
        if key == GLFW_KEY_ESCAPE:
            glfwSetWindowShouldClose(window, True)
        elif key == GLFW_KEY_R:
            g_anim_t = 0.0
            g_prev_real = glfwGetTime()
            g_killcam_mode  = 0
            g_killcam_orbit = 0.0
        elif key == GLFW_KEY_F:
            g_cam_mode = 0
        elif key == GLFW_KEY_1:
            g_cam_mode = 1
        elif key == GLFW_KEY_2:
            g_cam_mode = 2
        elif key == GLFW_KEY_3:
            g_cam_mode = 3
        elif key == GLFW_KEY_4:
            g_cam_mode = 4
        elif key == GLFW_KEY_K:
            g_killcam_mode  = (g_killcam_mode + 1) % 3
            g_killcam_orbit = 0.0
        elif key == GLFW_KEY_Z:
            g_z_is_pressed = True
        elif key == GLFW_KEY_X:
            g_x_is_pressed = True
    elif action == GLFW_RELEASE:
        if key == GLFW_KEY_Z:
            g_z_is_pressed = False
        elif key == GLFW_KEY_X:
            g_x_is_pressed = False


def button_callback(window, button, action, mods):
    global g_mouse_is_dragged, g_mouse_x_pos, g_mouse_y_pos
    if button == GLFW_MOUSE_BUTTON_LEFT:
        if action == GLFW_PRESS:
            g_mouse_is_dragged = True
            g_mouse_x_pos, g_mouse_y_pos = glfwGetCursorPos(window)
        else:
            g_mouse_is_dragged = False


def cursor_callback(window, xpos, ypos):
    global g_cam_r, g_cam_theta, g_cam_phi, g_cam_center
    global g_mouse_x_pos, g_mouse_y_pos
    if not g_mouse_is_dragged or g_cam_mode in (2, 4):
        g_mouse_x_pos, g_mouse_y_pos = xpos, ypos
        return
    dx = xpos - g_mouse_x_pos
    dy = ypos - g_mouse_y_pos
    if g_z_is_pressed:
        g_cam_r = max(0.5, g_cam_r + dy * 0.05)
    elif g_x_is_pressed:
        fwd   = glm.vec3(glm.sin(g_cam_theta), 0, glm.cos(g_cam_theta))
        right = glm.vec3(glm.sin(g_cam_theta - glm.pi()/2), 0, glm.cos(g_cam_theta - glm.pi()/2))
        g_cam_center -= (fwd * dy - right * dx) * 0.02
    else:
        g_cam_theta -= dx * 0.005
        g_cam_phi    = max(-glm.pi()/2+0.01, min(glm.pi()/2-0.01,
                           g_cam_phi + dy * 0.005))
    g_mouse_x_pos, g_mouse_y_pos = xpos, ypos


def scroll_callback(window, xoff, yoff):
    global g_cam_r
    g_cam_r = max(0.5, g_cam_r - yoff * 0.5)


def framebuffer_size_callback(window, width, height):
    glViewport(0, 0, width, height)


def get_ball_pos(t, rhand_now, release_pos, ball_dir):
    if t < BALL_RELEASE_T:
        return rhand_now
    dt = t - BALL_RELEASE_T
    return glm.vec3(
        release_pos.x + ball_dir.x * BALL_SPEED * dt,
        release_pos.y,
        release_pos.z + ball_dir.z * BALL_SPEED * dt,
    )


def main():
    global g_P, g_prev_real, g_anim_t
    global g_cam_r, g_cam_theta, g_cam_phi, g_cam_center, g_cam_mode
    global g_killcam_mode, g_killcam_orbit

    if not glfwInit():
        return
    glfwWindowHint(GLFW_CONTEXT_VERSION_MAJOR, 3)
    glfwWindowHint(GLFW_CONTEXT_VERSION_MINOR, 3)
    glfwWindowHint(GLFW_OPENGL_PROFILE, GLFW_OPENGL_CORE_PROFILE)
    glfwWindowHint(GLFW_OPENGL_FORWARD_COMPAT, GL_TRUE)

    window = glfwCreateWindow(1280, 720, 'project3', None, None)
    if not window:
        glfwTerminate(); return
    glfwMakeContextCurrent(window)
    glfwSetKeyCallback(window,             key_callback)
    glfwSetMouseButtonCallback(window,     button_callback)
    glfwSetCursorPosCallback(window,       cursor_callback)
    glfwSetScrollCallback(window,          scroll_callback)
    glfwSetFramebufferSizeCallback(window, framebuffer_size_callback)

    shader_program_color = load_shaders(g_vertex_shader_src_color_attribute,
                                        g_fragment_shader_src_color)
    shader_program_light = load_shaders(g_vertex_shader_src_light,
                                        g_fragment_shader_src_light)

    loc_MVP_color        = glGetUniformLocation(shader_program_color, 'MVP')
    loc_MVP_light        = glGetUniformLocation(shader_program_light, 'MVP')
    loc_M_light          = glGetUniformLocation(shader_program_light, 'M')
    loc_material_color   = glGetUniformLocation(shader_program_light, 'material_color')
    loc_light_pos        = glGetUniformLocation(shader_program_light, 'light_pos')
    loc_view_pos         = glGetUniformLocation(shader_program_light, 'view_pos')

    vao_grid, vcnt_grid = prepare_vao_grid(half=25, step=1)
    vao_sp,   vcnt_sp   = prepare_vao_sphere(stacks=8, slices=12)
    vao_cy,   vcnt_cy   = prepare_vao_cylinder(segments=12)

    bvh_dir = os.path.dirname(os.path.abspath(__file__))
    pitcher_motion = BVHMotion.parse(os.path.join(bvh_dir, '124_01.bvh'))
    batter_motion  = BVHMotion.parse(os.path.join(bvh_dir, '124_07.bvh'))

    pitcher_wmat = make_world_mat(PITCHER_POS, PITCHER_ROT_Y)
    batter_wmat  = make_world_mat(BATTER_POS,  BATTER_ROT_Y)

    pitcher_motion.apply_frame(BALL_RELEASE_FRAME)
    rhand_joint = pitcher_motion.joint_by_name('rHand')
    release_pos = (get_world_pos(rhand_joint, pitcher_wmat)
                   if rhand_joint else glm.vec3(0, 1, 0)) + BALL_HAND_OFFSET
    strike_zone = BATTER_POS + glm.vec3(0, 1.1, 0)
    ball_dir    = glm.normalize(strike_zone - release_pos)

    color_pitcher_joint = glm.vec3(0.9, 0.4, 0.2)
    color_pitcher_bone  = glm.vec3(0.7, 0.3, 0.15)
    color_batter_joint  = glm.vec3(0.2, 0.5, 0.9)
    color_batter_bone   = glm.vec3(0.1, 0.35, 0.7)
    color_ball          = glm.vec3(0.95, 0.95, 0.85)
    color_bat           = glm.vec3(0.55, 0.32, 0.10)

    bat_local = (glm.rotate(BAT_ROT_X, glm.vec3(1, 0, 0)) *
                 glm.translate(glm.vec3(0, BAT_LENGTH * 0.5, 0)) *
                 glm.scale(glm.vec3(BAT_RADIUS, BAT_LENGTH * 0.5, BAT_RADIUS)))

    g_prev_real = glfwGetTime()

    glEnable(GL_DEPTH_TEST)
    glClearColor(0.08, 0.08, 0.12, 1.0)

    while not glfwWindowShouldClose(window):
        now = glfwGetTime()
        real_dt = now - g_prev_real
        g_prev_real = now

        speed = KILLCAM_SPEED if g_killcam_mode > 0 else 1.0
        g_anim_t += real_dt * speed
        if g_anim_t >= LOOP_DURATION:
            g_anim_t = 0.0
        if g_killcam_mode > 0:
            g_killcam_orbit += real_dt * KILLCAM_ORBIT_W

        p_frame = pitcher_motion.frame_at(g_anim_t)
        if g_anim_t < BATTER_DELAY:
            b_frame = BATTER_READY_FRAME
        else:
            batter_t = g_anim_t - BATTER_DELAY
            b_frame  = min(batter_motion.frame_at(batter_t), BATTER_MAX_FRAME)

        pitcher_motion.apply_frame(p_frame)
        batter_motion.apply_frame(b_frame)

        width, height = glfwGetWindowSize(window)
        g_P = glm.perspective(glm.radians(45.0), width / max(height, 1), 0.05, 200.0)

        p_hip_j = pitcher_motion.joint_by_name('hip')
        b_hip_j = batter_motion.joint_by_name('hip')
        p_hip = get_world_pos(p_hip_j, pitcher_wmat) if p_hip_j else PITCHER_POS
        b_hip = get_world_pos(b_hip_j, batter_wmat)  if b_hip_j else BATTER_POS

        if g_cam_mode == 0:
            eye_x = g_cam_center.x + g_cam_r * glm.sin(g_cam_theta) * glm.cos(g_cam_phi)
            eye_y = g_cam_center.y + g_cam_r * glm.sin(g_cam_phi)
            eye_z = g_cam_center.z + g_cam_r * glm.cos(g_cam_phi) * glm.cos(g_cam_theta)
            eye = glm.vec3(eye_x, eye_y, eye_z)
            V   = glm.lookAt(eye, g_cam_center, glm.vec3(0, 1, 0))
        elif g_cam_mode == 1:
            eye_x = p_hip.x + g_cam_r * glm.sin(g_cam_theta) * glm.cos(g_cam_phi)
            eye_y = p_hip.y + g_cam_r * glm.sin(g_cam_phi)
            eye_z = p_hip.z + g_cam_r * glm.cos(g_cam_phi) * glm.cos(g_cam_theta)
            eye = glm.vec3(eye_x, eye_y, eye_z)
            V   = glm.lookAt(eye, p_hip, glm.vec3(0, 1, 0))
        elif g_cam_mode == 2:
            p_head_j = pitcher_motion.joint_by_name('head')
            p_head   = get_world_pos(p_head_j, pitcher_wmat) if p_head_j else p_hip
            eye      = p_head + glm.vec3(0, 0.1, 0)
            V        = glm.lookAt(eye, BATTER_POS + glm.vec3(0, 1.5, 0), glm.vec3(0, 1, 0))
        elif g_cam_mode == 3:
            eye_x = b_hip.x + g_cam_r * glm.sin(g_cam_theta) * glm.cos(g_cam_phi)
            eye_y = b_hip.y + g_cam_r * glm.sin(g_cam_phi)
            eye_z = b_hip.z + g_cam_r * glm.cos(g_cam_phi) * glm.cos(g_cam_theta)
            eye = glm.vec3(eye_x, eye_y, eye_z)
            V   = glm.lookAt(eye, b_hip, glm.vec3(0, 1, 0))
        else:
            b_head_j = batter_motion.joint_by_name('head')
            b_head   = get_world_pos(b_head_j, batter_wmat) if b_head_j else b_hip
            eye      = b_head + glm.vec3(0, 0.1, 0)
            V        = glm.lookAt(eye, PITCHER_POS + glm.vec3(0, 1.5, 0), glm.vec3(0, 1, 0))

        if g_killcam_mode == 1:
            eye = glm.vec3(p_hip.x + 4.0 * glm.cos(g_killcam_orbit),
                           p_hip.y + 2.0,
                           p_hip.z + 4.0 * glm.sin(g_killcam_orbit))
            V   = glm.lookAt(eye, p_hip, glm.vec3(0, 1, 0))
        elif g_killcam_mode == 2:
            eye = glm.vec3(b_hip.x + 4.0 * glm.cos(g_killcam_orbit),
                           b_hip.y + 2.0,
                           b_hip.z + 4.0 * glm.sin(g_killcam_orbit))
            V   = glm.lookAt(eye, b_hip, glm.vec3(0, 1, 0))

        VP = g_P * V

        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)

        draw_grid(vao_grid, vcnt_grid, VP, shader_program_color, loc_MVP_color)

        draw_skeleton(pitcher_motion, pitcher_wmat,
                      vao_sp, vcnt_sp, vao_cy, vcnt_cy,
                      VP, eye, color_pitcher_joint, color_pitcher_bone,
                      shader_program_light,
                      loc_MVP_light, loc_M_light,
                      loc_material_color, loc_light_pos, loc_view_pos)

        draw_skeleton(batter_motion, batter_wmat,
                      vao_sp, vcnt_sp, vao_cy, vcnt_cy,
                      VP, eye, color_batter_joint, color_batter_bone,
                      shader_program_light,
                      loc_MVP_light, loc_M_light,
                      loc_material_color, loc_light_pos, loc_view_pos)

        rhand_j   = pitcher_motion.joint_by_name('rHand')
        rhand_now = (get_world_pos(rhand_j, pitcher_wmat)
                     if rhand_j else release_pos) + BALL_HAND_OFFSET
        ball_pos = get_ball_pos(g_anim_t, rhand_now, release_pos, ball_dir)
        M_ball   = glm.translate(ball_pos) * glm.scale(glm.vec3(BALL_R))
        glUseProgram(shader_program_light)
        draw_node(vao_sp, vcnt_sp, M_ball, VP, eye, color_ball,
                  shader_program_light,
                  loc_MVP_light, loc_M_light,
                  loc_material_color, loc_light_pos, loc_view_pos)

        rhand_b = batter_motion.joint_by_name('rHand')
        if rhand_b:
            M_bat = batter_wmat * rhand_b.global_transform * bat_local
            draw_node(vao_cy, vcnt_cy, M_bat, VP, eye, color_bat,
                      shader_program_light,
                      loc_MVP_light, loc_M_light,
                      loc_material_color, loc_light_pos, loc_view_pos)

        glfwSwapBuffers(window)
        glfwPollEvents()

    glfwTerminate()


if __name__ == '__main__':
    main()
