#include <assert.h>
#include <stdarg.h>
#include <stdio.h>
#include <time.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>

#include <GL/glew.h>
#include <GLFW/glfw3.h>

#include "render/cpu_raytracer.h"
#include "render/threaded_raytracer.h"
#include "render/openmp_raytracer.h"
#include "render/cuda_raytracer.cuh"

#include "render/gl_render.h"

#include "math/math.h"
#include "debug/debug.h"

static double previous_time;
static int frame_count;

static GLuint vao;
static GLuint vbo;
static GLuint tex;
static GLuint shader;

static raytracer r;

//
//TEMP
//

cuda_rt *crt;

void gl_realtime_render(raytracer rt)
{
	r = rt;
	GLFWwindow *window = gl_init(rt.config);

	if(!window) {
		log_msg(ERROR, "Failed to init gl realtime rendering\n");
		return;
	}
	
	log_msg(INFO, "Starting main game loop\n");
	while(!glfwWindowShouldClose(window)) {
		gl_input(window);
		gl_update(window);
		gl_render(window);
	}
	gl_cleanup(window);
}

GLFWwindow *gl_init(config c)
{
	log_msg(INFO, "Initializing OpenGL rendering setup\n");
	// start GL context and O/S window using the GLFW helper library
	log_msg(INFO, "Starting GLFW: %s\n", glfwGetVersionString());
	// register the error call-back function that we wrote, above
	glfwSetErrorCallback(gl_glfw_error_callback);
	if (!glfwInit()) {
		log_msg(ERROR, "Could not start GLFW3\n");
		return NULL;
	}
	
	glfwWindowHint(GLFW_CONTEXT_VERSION_MAJOR, 4);
	glfwWindowHint(GLFW_CONTEXT_VERSION_MINOR, 6);
	glfwWindowHint(GLFW_OPENGL_FORWARD_COMPAT, GL_TRUE);
	glfwWindowHint(GLFW_OPENGL_PROFILE, GLFW_OPENGL_CORE_PROFILE);
	glfwWindowHint(GLFW_DOUBLEBUFFER, GL_TRUE);
	glfwWindowHint(GLFW_RESIZABLE , GL_FALSE);
	glfwWindowHint(GLFW_OPENGL_DEBUG_CONTEXT, GL_TRUE);
	
       	GLFWwindow *window = NULL;
	if(c.fullscreen) {
		log_msg(INFO, "Using fullscreen mode\n");
		GLFWmonitor* mon = glfwGetPrimaryMonitor ();
		const GLFWvidmode* vmode = glfwGetVideoMode (mon);
		window = glfwCreateWindow (vmode->width, vmode->height,
					   "OpenGL - Raytracer", mon, NULL);

	}
	else {
		log_msg(INFO, "Using windowed mode\n");
		window = glfwCreateWindow(c.width, c.height,
					  "OpenGL - Raytracer", NULL, NULL);
	}
	
	if (!window) {
		log_msg(ERROR, "Could not open window with GLFW3\n");
		glfwTerminate();
		return NULL;
	}
	glfwMakeContextCurrent(window);
	glfwSetWindowSizeCallback(window, gl_window_resize_callback);
	glfwSetKeyCallback(window, gl_key_callback);
	glfwSetCursorPosCallback(window, gl_mouse_callback);
	glfwSetScrollCallback(window, gl_scroll_callback);

	glfwSetInputMode(window, GLFW_CURSOR, GLFW_CURSOR_DISABLED);
	glfwSetInputMode(window, GLFW_STICKY_KEYS, GLFW_TRUE);
	
	// start GLEW extension handler
	log_msg(INFO, "Starting GLEW\n");
	glewExperimental = GL_TRUE;
	glewInit();

	//start debugging
	glEnable(GL_DEBUG_OUTPUT);
	glDebugMessageCallback(opengl_debug, 0);
	
	//get version info
	log_msg(INFO, "Vendor: %s\n" \
		"\tDevice: %s\n" \
		"\tOpengl version: %s\n",
		glGetString(GL_VENDOR),
		glGetString(GL_RENDERER),
		glGetString(GL_VERSION));

	//set up global state
	previous_time = 0;
	frame_count = 0;
	
	glfwGetCursorPos(window, &r.state.last_x, &r.state.last_y);
	
	//setup rendering
	glBindFramebuffer(GL_FRAMEBUFFER, 0);
	glViewport(0, 0, c.width, c.height);
	log_msg(INFO, "Initializing shader program\n");
	shader = create_program("./render/quad.vert", "./render/quad.frag");
	log_msg(INFO, "Initializing fullscreen quad VAO and VBO\n");
	init_quad();
	log_msg(INFO, "Initializing fullscreen texture\n");
	init_texture(c.width, c.height);
	
	//setup the raytracer
	log_msg(INFO, "Starting raytracer\n");

	//
	//TEMP
	//
	
	crt = cuda_init(r);
	
	return window;
}

void gl_render(GLFWwindow *window)
{
	//get next from from raytracing renderer
	//cpu_render(r);
	//threaded_render(r);
	cuda_render(r, crt);
	//openmp_render(r);
	
	glBindTexture(GL_TEXTURE_2D, tex);
	glTexSubImage2D(GL_TEXTURE_2D, 0, 0, 0, r.canvas.width, r.canvas.height, GL_RGBA,
			GL_FLOAT, r.canvas.screen);

	//clear frame, draw tex to screen aligned quad
	glClearColor(0.0, 0.0, 0.0, 1.0);
	glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
	
	glBindVertexArray(vao);
	glBindTexture(GL_TEXTURE_2D, tex);
	glDrawArrays(GL_TRIANGLE_STRIP, 0, 6);
	glfwSwapBuffers(window);
}

void gl_input(GLFWwindow *window)
{
	glfwPollEvents();
}

void gl_update(GLFWwindow *window)
{
	gl_update_fps_counter(window);
	
	if(r.state.forward) {
		camera_forward(&r.camera, r.state.speed);
	}
	if(r.state.left) {
		camera_left(&r.camera, r.state.speed);
	}
	if(r.state.right) {
		camera_right(&r.camera, r.state.speed);
	}
	if(r.state.back) {
		camera_backward(&r.camera, r.state.speed);
	}
	if(r.state.up) {
		camera_up(&r.camera, r.state.speed);	
	}
	if(r.state.down) {
		camera_down(&r.camera, r.state.speed);	
	}
	log_msg(DEBUG, "position: %f, %f, %f\n direction: %f, %f, %f\n up: %f, %f, %f\n right: %f, %f, %f\n",
		r.camera.position.x,
		r.camera.position.y,
		r.camera.position.z,
		r.camera.direction.x,
		r.camera.direction.y,
		r.camera.direction.z,
		r.camera.up.x,
		r.camera.up.y,
		r.camera.up.z,
		r.camera.right.x,
		r.camera.right.y,
		r.camera.right.z);
	log_msg(DEBUG, "f:%d, b:%d, l:%d. r:%d, u:%d, d:%d\n",
		r.state.forward, r.state.back, r.state.left, r.state.right, r.state.up, r.state.down);
	
}

void gl_cleanup(GLFWwindow *window)
{
	log_msg(INFO, "Terminating OpenGL Rendering setup\n");
	log_msg(INFO, "Freeing fullscreen texture\n");
	glTextureSubImage2D(tex, 0, 0, 0, 0, 0, GL_RGBA,
			    GL_FLOAT, NULL);
	glBindTexture(GL_TEXTURE_2D, 0);
	glDeleteTextures(1, &tex);
	log_msg(INFO, "Freeing fullscreen quad VAO and VBO\n");
	glDeleteBuffers(1, &vao);
	glDeleteBuffers(1, &vbo);
	log_msg(INFO, "Freeing shader program\n");
	glDeleteProgram(shader);
	
	log_msg(INFO, "Terminating glfw window\n");
	glfwDestroyWindow(window);
	log_msg(INFO, "Terminating glfw\n");
       	glfwTerminate();
}

void gl_glfw_error_callback(int error, const char *description)
{
	log_msg(ERROR, "GLFW ERROR: code %i msg: %s\n", error, description);
}

void check_gl_error(const char *place)
{
	GLenum err;
	while((err = glGetError()) != GL_NO_ERROR)
	{
		log_msg(ERROR, "opengl error in %s: %X\n", place, err);
	}
}

void opengl_debug(GLenum source, GLenum type, GLuint id, GLenum severity,
		     GLsizei length, const GLchar* message, const void* userParam)
{
	log_msg(ERROR, "OpenGL Error: %s id:%i source: 0x%x type: 0x%x, severity: 0x%x, message = %s\n",
		(type == GL_DEBUG_TYPE_ERROR ? "** GL ERROR **" : ""),
		id,
		source,
		type,
		severity,
		message);
}

void gl_window_resize_callback(GLFWwindow *w, int width, int height)
{
	glfwWaitEvents();
        glfwSetWindowSize(w, width, height);
	glDeleteTextures(1, &tex);
	init_texture(width, height);
	canvas_term(r.canvas);
	r.canvas = canvas_init(width, height);
	r.camera = camera_init(r.camera.position, r.camera.direction, r.camera.up, r.camera.right,
			       width, height, r.camera.fov);
	glViewport(0, 0, width, height);
	log_msg(INFO, "Resize - width: %i height: %i\n", width, height);
}

void gl_key_callback(GLFWwindow* window, int key, int scancode, int action, int mods)
{
	if (key == GLFW_KEY_ESCAPE ) {
		glfwSetWindowShouldClose(window, GL_TRUE);
	}

	if(action == GLFW_PRESS) {
		if(key == GLFW_KEY_W) {
			r.state.forward = true;
		}
		if(key == GLFW_KEY_A) {
			r.state.left = true;
		}
		if(key == GLFW_KEY_S) {
			r.state.back = true;
		}
		if(key == GLFW_KEY_D) {
			r.state.right = true;
		}
		if(key == GLFW_KEY_Q) {
			r.state.up = true;
		}
		if(key == GLFW_KEY_E) {
			r.state.down = true;
		}
	}
	else if(action == GLFW_RELEASE) {
		if(key == GLFW_KEY_W) {
			r.state.forward = false;
		}
		if(key == GLFW_KEY_A) {
			r.state.left = false;
		}
		if(key == GLFW_KEY_S) {
			r.state.back = false;
		}
		if(key == GLFW_KEY_D) {
			r.state.right = false;
		}
		if(key == GLFW_KEY_Q) {
			r.state.up = false;
		}
		if(key == GLFW_KEY_E) {
			r.state.down = false;
		}	
	}
	
}
void gl_mouse_callback(GLFWwindow* window, double xpos, double ypos)
{
	float xoffset = xpos - r.state.last_x;
	float yoffset = r.state.last_y - ypos;
	
	r.state.last_x = xpos;
	r.state.last_y = ypos;

	xoffset *= r.state.sensitivity;
	yoffset *= r.state.sensitivity;
	
        r.state.yaw   += xoffset;
        r.state.pitch += yoffset;
	
	if (r.state.pitch > 89.9f) {
		r.state.pitch = 89.9f;
	}
	if (r.state.pitch < -89.9f) {
	        r.state.pitch = -89.9f;
	}
	
	log_msg(DEBUG, "mouse pitch: %f, mouse yaw: %f\n", r.state.pitch, r.state.yaw);
	camera_rotate(&r.camera, r.state.pitch, r.state.yaw);
}

void gl_scroll_callback(GLFWwindow* window, double xoffset, double yoffset)
{
	
}

GLuint load_shader(const char *filename, GLenum shadertype)
{
	FILE *fp = fopen(filename, "r");
	log_msg(INFO, "Reading in %s in\n", filename);
	fseek(fp, 0L, SEEK_END);
	size_t size = ftell(fp);
	rewind(fp);

	char *buffer = calloc(size+1, sizeof(char));
	size_t read_size = fread(buffer, size, 1, fp);
	if(size != read_size){
		log_msg(WARN, "Could not read all of config file %s\n", filename);
	}
	
	fclose(fp);
	GLuint shader_prog = glCreateShader(shadertype);
	glShaderSource(shader_prog, 1, (const GLchar * const *)&buffer, NULL);
	log_msg(INFO, "Compiling %s\n", filename);
	glCompileShader(shader_prog);

	GLint success;
	glGetShaderiv(shader_prog, GL_COMPILE_STATUS, &success);
	if(!success) {
		log_msg(ERROR, "Shader program could not be compiled, printing debug info\n");
		int max_length = 2048;
		int actual_length = 0;
		char log[2048];
		glGetShaderInfoLog(shader_prog, max_length, &actual_length, log);
		log_msg(ERROR, "%s\n", log);
	}
	
	free(buffer);
	return shader_prog;
}

GLuint create_program(const char *vert_path, const char *frag_path)
{
	GLuint vert = load_shader(vert_path, GL_VERTEX_SHADER);
	GLuint frag = load_shader(frag_path, GL_FRAGMENT_SHADER);

	//Attach the above shader to a program
	log_msg(INFO, "Creating shader program\n");
	GLuint program = glCreateProgram();
	log_msg(INFO, "Attaching vertex and fragment shader programs\n");
	glAttachShader(program, vert);
	glAttachShader(program, frag);
	
	//Flag the shaders for deletion
	glDeleteShader(vert);
	glDeleteShader(frag);
		
	// Link and use the program
	glLinkProgram(program);	
	glUseProgram(program);
	
	return program;
}

void init_quad(void)
{
	GLfloat vertices[] = {
		//  X      Y     Z      U     V
		-1.0f,  1.0f, 0.0f,  0.0f, 1.0f,
		-1.0f, -1.0f, 0.0f,  0.0f, 0.0f,
		 1.0f, -1.0f, 0.0f,  1.0f, 0.0f,

		-1.0f,  1.0f, 0.0f,  0.0f, 1.0f,
		 1.0f, -1.0f, 0.0f,  1.0f, 0.0f,
		 1.0f,  1.0f, 0.0f,  1.0f, 1.0f
	};

	log_msg(INFO, "Creating Vertex Buffer Object\n");
	glGenBuffers(1, &vbo);
	glBindBuffer(GL_ARRAY_BUFFER, vbo);
	glBufferData(GL_ARRAY_BUFFER, sizeof(vertices), vertices, GL_STATIC_DRAW);

	log_msg(INFO, "Creating Vertex Array Object\n");
	glGenVertexArrays(1, &vao);
	glBindVertexArray(vao);
	glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, 5 * sizeof(GLfloat), (void*)0);
	glEnableVertexAttribArray(0);
	glVertexAttribPointer(1, 2, GL_FLOAT, GL_FALSE, 5 * sizeof(GLfloat), (void*)(3 * sizeof(float)));
	glEnableVertexAttribArray(1);
}

void init_texture(int window_width, int window_height)
{
	glActiveTexture(GL_TEXTURE0);
	log_msg(INFO, "Creating texture object\n");
	glGenTextures(1, &tex);
	glBindTexture(GL_TEXTURE_2D, tex);
	glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA32F, window_width, window_height,
		     0, GL_RGBA, GL_FLOAT, 0);
	glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST);
	glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST);
}

void gl_update_fps_counter(GLFWwindow *w)
{
	char tmp[64];

	double current_time = glfwGetTime();
	double elapsed_time = current_time - previous_time;
	frame_count++;
	if(elapsed_time > 0.1f) {
		double fps = (double)frame_count/elapsed_time;
		sprintf(tmp, "OpenGL - Raytracer @ fps: %.2f", fps);
		glfwSetWindowTitle(w, tmp);

		previous_time = current_time;
		frame_count = 0;
	}
}

