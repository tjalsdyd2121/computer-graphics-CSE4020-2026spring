from OpenGL.GL import *
from glfw.GLFW import *
import glm
import numpy as np

WINDOW_WIDTH = 800
WINDOW_HEIGHT = 800

g_control_points = [
    glm.vec3(250, 350, 0),
    glm.vec3(350, 450, 0),
    glm.vec3(450, 450, 0),
    glm.vec3(550, 350, 0),
]
g_moving_index = None
g_animate = False

# ── NEW : P1의 회전 중심을 고정으로 저장 ─────────────────────────────────
g_anim_center = glm.vec2(350, 450)
g_anim_radius = 80.0
# ─────────────────────────────────────────────────────────────────────────

g_vao_control_points = None
g_vbo_control_points = None
g_vao_curve_points   = None
g_vbo_curve_points   = None

g_vertex_shader_src = '''
#version 330 core
layout (location = 0) in vec3 vin_pos;
uniform mat4 MVP;
void main() { gl_Position = MVP * vec4(vin_pos, 1.0); }
'''

g_fragment_shader_src = '''
#version 330 core
out vec4 FragColor;
uniform vec3 color;
void main() { FragColor = vec4(color, 1.0); }
'''

def load_shaders(vertex_shader_source, fragment_shader_source):
    vertex_shader = glCreateShader(GL_VERTEX_SHADER)
    glShaderSource(vertex_shader, vertex_shader_source)
    glCompileShader(vertex_shader)
    success = glGetShaderiv(vertex_shader, GL_COMPILE_STATUS)
    if not success:
        print("ERROR::VERTEX::\n" + glGetShaderInfoLog(vertex_shader).decode())

    fragment_shader = glCreateShader(GL_FRAGMENT_SHADER)
    glShaderSource(fragment_shader, fragment_shader_source)
    glCompileShader(fragment_shader)
    success = glGetShaderiv(fragment_shader, GL_COMPILE_STATUS)
    if not success:
        print("ERROR::FRAGMENT::\n" + glGetShaderInfoLog(fragment_shader).decode())

    shader_program = glCreateProgram()
    glAttachShader(shader_program, vertex_shader)
    glAttachShader(shader_program, fragment_shader)
    glLinkProgram(shader_program)
    success = glGetProgramiv(shader_program, GL_LINK_STATUS)
    if not success:
        print("ERROR::PROGRAM::\n" + glGetProgramInfoLog(shader_program).decode())

    glDeleteShader(vertex_shader)
    glDeleteShader(fragment_shader)
    return shader_program


def key_callback(window, key, scancode, action, mods):
    global g_animate, g_anim_center  
    if key == GLFW_KEY_ESCAPE and action == GLFW_PRESS:
        glfwSetWindowShouldClose(window, GLFW_TRUE)
    if key == GLFW_KEY_A:
        if action == GLFW_PRESS:
            g_anim_center = glm.vec2(g_control_points[1].x, g_control_points[1].y)
            g_animate = True
        elif action == GLFW_RELEASE:
            g_animate = False

def hittest(x, y, control_point):
    return glm.abs(x - control_point.x) < 10 and glm.abs(y - control_point.y) < 10

def button_callback(window, button, action, mod):
    global g_control_points, g_moving_index
    if button == GLFW_MOUSE_BUTTON_LEFT:
        x, y = glfwGetCursorPos(window)
        y = WINDOW_HEIGHT - y
        if action == GLFW_PRESS:
            g_moving_index = None
            for i in range(len(g_control_points)):
                if hittest(x, y, g_control_points[i]):
                    g_moving_index = i
                    break
        elif action == GLFW_RELEASE:
            g_moving_index = None

def cursor_callback(window, xpos, ypos):
    global g_control_points, g_moving_index
    global g_vbo_control_points, g_vbo_curve_points
    ypos = WINDOW_HEIGHT - ypos
    if g_moving_index is not None:
        g_control_points[g_moving_index].x = xpos
        g_control_points[g_moving_index].y = ypos
        copy_points_data(g_control_points, g_vbo_control_points)
        curve_points = generate_curve_points(g_control_points)
        copy_points_data(curve_points, g_vbo_curve_points)

def initialize_vao_for_points(points):
    VAO = glGenVertexArrays(1)
    glBindVertexArray(VAO)
    VBO = glGenBuffers(1)
    glBindBuffer(GL_ARRAY_BUFFER, VBO)
    vertices = glm.array(points)
    glBufferData(GL_ARRAY_BUFFER, vertices.nbytes, None, GL_DYNAMIC_DRAW)
    glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, 3 * glm.sizeof(glm.float32), None)
    glEnableVertexAttribArray(0)
    return VAO, VBO

def copy_points_data(points, vbo):
    glBindBuffer(GL_ARRAY_BUFFER, vbo)
    vertices = glm.array(points)
    glBufferSubData(GL_ARRAY_BUFFER, 0, vertices.nbytes, vertices.ptr)

def generate_curve_points(control_points):
    curve_points = []
    for t in np.linspace(0, 1, 101):
        T = np.array([t**3, t**2, t, 1])
        M = np.array([[-1, 3, -3, 1],
                      [ 3,-6,  3, 0],
                      [-3, 3,  0, 0],
                      [ 1, 0,  0, 0]], float)
        P = np.array(control_points)
        p = T @ M @ P
        curve_points.append(glm.vec3(p))
    return curve_points


def main():
    global g_vao_control_points, g_vao_curve_points
    global g_vbo_control_points, g_vbo_curve_points

    if not glfwInit():
        return
    glfwWindowHint(GLFW_CONTEXT_VERSION_MAJOR, 3)
    glfwWindowHint(GLFW_CONTEXT_VERSION_MINOR, 3)
    glfwWindowHint(GLFW_OPENGL_PROFILE, GLFW_OPENGL_CORE_PROFILE)
    glfwWindowHint(GLFW_OPENGL_FORWARD_COMPAT, GL_TRUE)

    window = glfwCreateWindow(WINDOW_WIDTH, WINDOW_HEIGHT,
                              "2022056562-lab-check-11",
                              None, None)
    if not window:
        glfwTerminate(); return
    glfwMakeContextCurrent(window)

    glfwSetKeyCallback(window, key_callback)
    glfwSetMouseButtonCallback(window, button_callback)
    glfwSetCursorPosCallback(window, cursor_callback)

    shader_program = load_shaders(g_vertex_shader_src, g_fragment_shader_src)
    unif_locs = {name: glGetUniformLocation(shader_program, name)
                 for name in ('color', 'MVP')}

    g_vao_control_points, g_vbo_control_points = initialize_vao_for_points(g_control_points)
    copy_points_data(g_control_points, g_vbo_control_points)

    curve_points = generate_curve_points(g_control_points)
    g_vao_curve_points, g_vbo_curve_points = initialize_vao_for_points(curve_points)
    copy_points_data(curve_points, g_vbo_curve_points)

    glPointSize(20)

    while not glfwWindowShouldClose(window):
        glClear(GL_COLOR_BUFFER_BIT)
        glUseProgram(shader_program)

        t = glfwGetTime()

        if g_animate:
            g_control_points[1].x = g_anim_center.x + g_anim_radius * np.cos(t)
            g_control_points[1].y = g_anim_center.y + g_anim_radius * np.sin(t)

            copy_points_data(g_control_points, g_vbo_control_points)
            curve_points = generate_curve_points(g_control_points)
            copy_points_data(curve_points, g_vbo_curve_points)

        P = glm.ortho(0, WINDOW_WIDTH, 0, WINDOW_HEIGHT, -1, 1)
        glUniformMatrix4fv(unif_locs['MVP'], 1, GL_FALSE, glm.value_ptr(P))

        # draw control polygon (green)
        glUniform3f(unif_locs['color'], 0, 1, 0)
        glBindVertexArray(g_vao_control_points)
        glDrawArrays(GL_LINE_LOOP, 0, len(g_control_points))
        glDrawArrays(GL_POINTS,    0, len(g_control_points))

        # highlight animated P1 (orange)
        if g_animate:
            glUniform3f(unif_locs['color'], 1, 0.5, 0)
            glDrawArrays(GL_POINTS, 1, 1)

        # draw curve (white)
        glUniform3f(unif_locs['color'], 1, 1, 1)
        glBindVertexArray(g_vao_curve_points)
        glDrawArrays(GL_LINE_STRIP, 0, len(curve_points))

        glfwSwapBuffers(window)
        glfwPollEvents()

    glfwTerminate()

if __name__ == "__main__":
    main()