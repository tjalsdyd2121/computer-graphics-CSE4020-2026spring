from OpenGL.GL import *
from glfw.GLFW import *
import glm
import ctypes
import numpy as np
from PIL import Image

g_cam_ang = 0.
g_cam_height = .1

# Added uniform vec2 uv_offset to animate texture coordinates each frame.
# The offset is passed from the CPU every frame using glfwGetTime().
g_vertex_shader_src = '''
#version 330 core

layout (location = 0) in vec3 vin_pos;
layout (location = 1) in vec3 vin_normal;
layout (location = 2) in vec2 vin_uv;

out vec3 vout_surface_pos;
out vec3 vout_normal;
out vec2 vout_uv;

uniform mat4 MVP;
uniform mat4 M;

void main()
{
    vec4 p3D_in_hcoord = vec4(vin_pos.xyz, 1.0);
    gl_Position = MVP * p3D_in_hcoord;

    vout_surface_pos = vec3(M * vec4(vin_pos, 1));
    vout_normal = normalize( mat3(inverse(transpose(M)) ) * vin_normal);
    vout_uv = vin_uv;
}
'''

g_fragment_shader_src = '''
#version 330 core

in vec3 vout_surface_pos;
in vec3 vout_normal;
in vec2 vout_uv;

out vec4 FragColor;

uniform vec3 view_pos;
uniform sampler2D texture_diffuse;
uniform sampler2D texture_specular;
uniform vec2 uv_offset;

void main()
{
    vec3 light_pos = vec3(3,2,4);
    vec3 light_color = vec3(1,1,1);

    // Apply animated uv_offset to both textures.
    // Because GL_MIRRORED_REPEAT is set, coordinates outside [0,1]
    // are mirrored back, creating a periodic back-and-forth motion.
    vec2 animated_uv = vout_uv + uv_offset;

    vec3 material_color = vec3(texture(texture_diffuse, animated_uv));
    float material_shininess = 32.0;

    vec3 light_ambient  = 0.1 * light_color;
    vec3 light_diffuse  = light_color;
    vec3 light_specular = light_color;

    vec3 material_ambient  = material_color;
    vec3 material_diffuse  = material_color;
    vec3 material_specular = vec3(texture(texture_specular, animated_uv));

    vec3 ambient = light_ambient * material_ambient;

    vec3 normal    = normalize(vout_normal);
    vec3 light_dir = normalize(light_pos - vout_surface_pos);

    float diff = max(dot(normal, light_dir), 0);
    vec3 diffuse = diff * light_diffuse * material_diffuse;

    vec3 view_dir    = normalize(view_pos - vout_surface_pos);
    vec3 reflect_dir = reflect(-light_dir, normal);
    float spec = pow(max(dot(view_dir, reflect_dir), 0.0), material_shininess);
    vec3 specular = spec * light_specular * material_specular;

    FragColor = vec4(ambient + diffuse + specular, 1.);
}
'''

def load_shaders(vertex_shader_source, fragment_shader_source):
    vertex_shader = glCreateShader(GL_VERTEX_SHADER)
    glShaderSource(vertex_shader, vertex_shader_source)
    glCompileShader(vertex_shader)
    success = glGetShaderiv(vertex_shader, GL_COMPILE_STATUS)
    if (not success):
        infoLog = glGetShaderInfoLog(vertex_shader)
        print("ERROR::SHADER::VERTEX::COMPILATION_FAILED\n" + infoLog.decode())

    fragment_shader = glCreateShader(GL_FRAGMENT_SHADER)
    glShaderSource(fragment_shader, fragment_shader_source)
    glCompileShader(fragment_shader)
    success = glGetShaderiv(fragment_shader, GL_COMPILE_STATUS)
    if (not success):
        infoLog = glGetShaderInfoLog(fragment_shader)
        print("ERROR::SHADER::FRAGMENT::COMPILATION_FAILED\n" + infoLog.decode())

    shader_program = glCreateProgram()
    glAttachShader(shader_program, vertex_shader)
    glAttachShader(shader_program, fragment_shader)
    glLinkProgram(shader_program)
    success = glGetProgramiv(shader_program, GL_LINK_STATUS)
    if (not success):
        infoLog = glGetProgramInfoLog(shader_program)
        print("ERROR::SHADER::PROGRAM::LINKING_FAILED\n" + infoLog.decode())

    glDeleteShader(vertex_shader)
    glDeleteShader(fragment_shader)
    return shader_program


def key_callback(window, key, scancode, action, mods):
    global g_cam_ang, g_cam_height
    if key==GLFW_KEY_ESCAPE and action==GLFW_PRESS:
        glfwSetWindowShouldClose(window, GLFW_TRUE)
    else:
        if action==GLFW_PRESS or action==GLFW_REPEAT:
            if key==GLFW_KEY_1:
                g_cam_ang += np.radians(-10)
            elif key==GLFW_KEY_3:
                g_cam_ang += np.radians(10)
            elif key==GLFW_KEY_2:
                g_cam_height += .1
            elif key==GLFW_KEY_W:
                g_cam_height += -.1

def prepare_vao_cube():
    vertices = glm.array(glm.float32,
        # position     # normal  # texture coordinates
        -1 ,  1 ,  1 ,  0, 0, 1,  0.0, 1.0,
         1 , -1 ,  1 ,  0, 0, 1,  1.0, 0.0,
         1 ,  1 ,  1 ,  0, 0, 1,  1.0, 1.0,

        -1 ,  1 ,  1 ,  0, 0, 1,  0.0, 1.0,
        -1 , -1 ,  1 ,  0, 0, 1,  0.0, 0.0,
         1 , -1 ,  1 ,  0, 0, 1,  1.0, 0.0,

        -1 ,  1 , -1 ,  0, 0,-1,  0.0, 1.0,
         1 ,  1 , -1 ,  0, 0,-1,  1.0, 1.0,
         1 , -1 , -1 ,  0, 0,-1,  1.0, 0.0,

        -1 ,  1 , -1 ,  0, 0,-1,  0.0, 1.0,
         1 , -1 , -1 ,  0, 0,-1,  1.0, 0.0,
        -1 , -1 , -1 ,  0, 0,-1,  0.0, 0.0,

        -1 ,  1 ,  1 ,  0, 1, 0,  0.0, 1.0,
         1 ,  1 ,  1 ,  0, 1, 0,  1.0, 1.0,
         1 ,  1 , -1 ,  0, 1, 0,  1.0, 0.0,

        -1 ,  1 ,  1 ,  0, 1, 0,  0.0, 1.0,
         1 ,  1 , -1 ,  0, 1, 0,  1.0, 0.0,
        -1 ,  1 , -1 ,  0, 1, 0,  0.0, 0.0,

        -1 , -1 ,  1 ,  0,-1, 0,  0.0, 1.0,
         1 , -1 , -1 ,  0,-1, 0,  1.0, 0.0,
         1 , -1 ,  1 ,  0,-1, 0,  1.0, 1.0,

        -1 , -1 ,  1 ,  0,-1, 0,  0.0, 1.0,
        -1 , -1 , -1 ,  0,-1, 0,  0.0, 0.0,
         1 , -1 , -1 ,  0,-1, 0,  1.0, 0.0,

         1 ,  1 ,  1 ,  1, 0, 0,  1.0, 1.0,
         1 , -1 ,  1 ,  1, 0, 0,  0.0, 1.0,
         1 , -1 , -1 ,  1, 0, 0,  0.0, 0.0,

         1 ,  1 ,  1 ,  1, 0, 0,  1.0, 1.0,
         1 , -1 , -1 ,  1, 0, 0,  0.0, 0.0,
         1 ,  1 , -1 ,  1, 0, 0,  1.0, 0.0,

        -1 ,  1 ,  1 , -1, 0, 0,  1.0, 1.0,
        -1 , -1 , -1 , -1, 0, 0,  0.0, 0.0,
        -1 , -1 ,  1 , -1, 0, 0,  0.0, 1.0,

        -1 ,  1 ,  1 , -1, 0, 0,  1.0, 1.0,
        -1 ,  1 , -1 , -1, 0, 0,  1.0, 0.0,
        -1 , -1 , -1 , -1, 0, 0,  0.0, 0.0,
    )

    VAO = glGenVertexArrays(1)
    glBindVertexArray(VAO)
    VBO = glGenBuffers(1)
    glBindBuffer(GL_ARRAY_BUFFER, VBO)
    glBufferData(GL_ARRAY_BUFFER, vertices.nbytes, vertices.ptr, GL_STATIC_DRAW)

    glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, 8 * glm.sizeof(glm.float32), None)
    glEnableVertexAttribArray(0)
    glVertexAttribPointer(1, 3, GL_FLOAT, GL_FALSE, 8 * glm.sizeof(glm.float32), ctypes.c_void_p(3*glm.sizeof(glm.float32)))
    glEnableVertexAttribArray(1)
    glVertexAttribPointer(2, 2, GL_FLOAT, GL_FALSE, 8 * glm.sizeof(glm.float32), ctypes.c_void_p(6*glm.sizeof(glm.float32)))
    glEnableVertexAttribArray(2)
    return VAO


def load_texture(path, wrap_mode):
    tex = glGenTextures(1)
    glBindTexture(GL_TEXTURE_2D, tex)

    # GL_MIRRORED_REPEAT: when UV goes outside [0,1], the texture is
    # mirrored and repeated, creating a periodic back-and-forth pattern.
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, wrap_mode)
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, wrap_mode)
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST_MIPMAP_LINEAR)
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR)

    try:
        img = Image.open(path)
        img = img.transpose(Image.FLIP_TOP_BOTTOM)
        glTexImage2D(GL_TEXTURE_2D, 0, GL_RGB, img.width, img.height, 0,
                     GL_RGB, GL_UNSIGNED_BYTE, img.tobytes())
        glGenerateMipmap(GL_TEXTURE_2D)
        img.close()
    except:
        print(f"Failed to load texture: {path}")
    return tex


def main():
    if not glfwInit():
        return
    glfwWindowHint(GLFW_CONTEXT_VERSION_MAJOR, 3)
    glfwWindowHint(GLFW_CONTEXT_VERSION_MINOR, 3)
    glfwWindowHint(GLFW_OPENGL_PROFILE, GLFW_OPENGL_CORE_PROFILE)
    glfwWindowHint(GLFW_OPENGL_FORWARD_COMPAT, GL_TRUE)

    window = glfwCreateWindow(800, 800, 'lab_check - animated UV', None, None)
    if not window:
        glfwTerminate()
        return
    glfwMakeContextCurrent(window)
    glfwSetKeyCallback(window, key_callback)

    shader_program = load_shaders(g_vertex_shader_src, g_fragment_shader_src)

    loc_MVP       = glGetUniformLocation(shader_program, 'MVP')
    loc_M         = glGetUniformLocation(shader_program, 'M')
    loc_view_pos  = glGetUniformLocation(shader_program, 'view_pos')
    loc_uv_offset = glGetUniformLocation(shader_program, 'uv_offset')

    vao_cube = prepare_vao_cube()

    glUseProgram(shader_program)

    import os
    base = os.path.dirname(os.path.abspath(__file__))

    # diffuse: earth texture with GL_MIRRORED_REPEAT
    tex_diffuse  = load_texture(
        os.path.join(base, '320px-Solarsystemscope_texture_8k_earth_daymap.jpg'),
        GL_MIRRORED_REPEAT)

    # specular: checkerboard texture with GL_MIRRORED_REPEAT
    tex_specular = load_texture(
        os.path.join(base, 'plain-checkerboard.jpg'),
        GL_MIRRORED_REPEAT)

    glUniform1i(glGetUniformLocation(shader_program, 'texture_diffuse'),  0)
    glActiveTexture(GL_TEXTURE0)
    glBindTexture(GL_TEXTURE_2D, tex_diffuse)

    glUniform1i(glGetUniformLocation(shader_program, 'texture_specular'), 1)
    glActiveTexture(GL_TEXTURE1)
    glBindTexture(GL_TEXTURE_2D, tex_specular)

    while not glfwWindowShouldClose(window):
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)
        glEnable(GL_DEPTH_TEST)

        t = glfwGetTime()

        # === Texture coordinate animation ===
        # Horizontal scrolling: u increases linearly with time.
        # Vertical oscillation: v oscillates with sin(), creating a
        # periodic up-down mirroring effect when combined with GL_MIRRORED_REPEAT.
        uv_u = t * 0.2
        uv_v = np.sin(t * 0.8) * 0.4
        glUseProgram(shader_program)
        glUniform2f(loc_uv_offset, uv_u, uv_v)
        # ====================================

        P = glm.perspective(45, 1, 1, 20)
        view_pos = glm.vec3(5*np.sin(g_cam_ang), g_cam_height, 5*np.cos(g_cam_ang))
        V = glm.lookAt(view_pos, glm.vec3(0,0,0), glm.vec3(0,1,0))
        M = glm.mat4()

        glUniformMatrix4fv(loc_MVP,      1, GL_FALSE, glm.value_ptr(P*V*M))
        glUniformMatrix4fv(loc_M,        1, GL_FALSE, glm.value_ptr(M))
        glUniform3f(loc_view_pos, view_pos.x, view_pos.y, view_pos.z)

        glBindVertexArray(vao_cube)
        glDrawArrays(GL_TRIANGLES, 0, 36)

        glfwSwapBuffers(window)
        glfwPollEvents()

    glfwTerminate()

if __name__ == "__main__":
    main()
