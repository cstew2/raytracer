#include <assert.h>
#include <stdarg.h>
#include <stdio.h>
#include <time.h>
#include <stdlib.h>
#include <string.h>

#include <GL/glew.h>
#include <GLFW/glfw3.h>

#include "render/gl_render.h"
#include "debug/debug.h"

static double previous_seconds;
static int frame_count;

GLuint vao;
GLuint vbo;
GLuint tex;
GLuint shader;

bool forward = false;
bool left = false;
bool right = false;
bool back = false;
bool up = false;
bool down = false;

raytracer r;

void gl_realtime_render(raytracer rt)
{
	r = rt;
	GLFWwindow *window = gl_init(rt.config);

	while(!glfwWindowShouldClose(window)) {
		gl_input(window);
		gl_update(window);
		gl_render();
	}
	gl_cleanup(window);
}

GLFWwindow *gl_init(config c)
{
	// start GL context and O/S window using the GLFW helper library
	log_msg(INFO, "Starting GLFW: %s\n", glfwGetVersionString());
	// register the error call-back function that we wrote, above
	glfwSetErrorCallback(glfw_error_callback);
	if (!glfwInit()) {
		log_msg(ERROR, "Could not start GLFW3\n");
		return NULL;
	}
	
	glfwWindowHint(GLFW_CONTEXT_VERSION_MAJOR, 4);
	glfwWindowHint(GLFW_CONTEXT_VERSION_MINOR, 2);
	glfwWindowHint(GLFW_OPENGL_FORWARD_COMPAT, GL_TRUE);
	glfwWindowHint(GLFW_OPENGL_PROFILE, GLFW_OPENGL_CORE_PROFILE);
	glfwWindowHint(GLFW_DOUBLEBUFFER, GL_TRUE);
	glfwWindowHint(GLFW_RESIZABLE , GL_TRUE);
	glfwWindowHint(GLFW_OPENGL_DEBUG_CONTEXT, GL_TRUE);
	
	GLFWwindow *window = NULL;
	if(c.fullscreen) {
		log_msg(INFO, "Using fullscreen mode\n");
		GLFWmonitor* mon = glfwGetPrimaryMonitor ();
		const GLFWvidmode* vmode = glfwGetVideoMode (mon);
		window = glfwCreateWindow (vmode->width, vmode->height,
					   "OpenGL - Voxel Raytracer", mon, NULL);

	}
	else {
		log_msg(INFO, "Using windowed mode\n");
		window = glfwCreateWindow(c.width, c.height,
					  "OpenGL - Voxel Raytracer", NULL, NULL);
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

	previous_seconds = 0;
	frame_count = 0;

	//disable things not needed
	glDisable(GL_DEPTH_TEST);
	glDisable(GL_COLOR_LOGIC_OP);
	glDisable(GL_CULL_FACE);
	glDisable(GL_BLEND);
	glDisable(GL_DITHER);
	glDisable(GL_MULTISAMPLE);
	glDisable(GL_SCISSOR_TEST);
	glDisable(GL_STENCIL_TEST);
	
	//setup rendering
	glBindFramebuffer(GL_FRAMEBUFFER, 0);
	glViewport(0, 0, c.width, c.height);
	shader = create_program("./render/quad.vert", "./render/quad.frag");
	init_quad();
	init_texture(c.width, c.height);
	
	//setup the raytracer
	log_msg(INFO, "Starting raytracer\n");
	
	return window;
}

void gl_render()
{
	//get next from from raytracing renderer
	render(r);
	glTexSubImage2D(GL_TEXTURE_2D, 0, 0, 0, r.canvas.width, r.canvas.height, GL_RGBA,
			GL_UNSIGNED_BYTE, r.canvas.screen);

	//clear frame, draw tex to screen aligned quad
	glClearColor(0.5, 0.1, 0.9, 1.0);
	glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
	glBindVertexArray(vao);
	glBindTexture(GL_TEXTURE_2D, tex);
	glDrawArrays(GL_TRIANGLE_STRIP, 0, 6);
}

void gl_input(GLFWwindow *window)
{
	glfwPollEvents();
	if (GLFW_PRESS == glfwGetKey(window, GLFW_KEY_ESCAPE)) {
		glfwSetWindowShouldClose(window, 1);
	}
	else if(GLFW_PRESS == glfwGetKey(window, GLFW_KEY_W)) {
		//m.position = vec3_add(cam.position, vec3_new(1.0, 0.0, 0.0)); 
	}
	else if(GLFW_PRESS == glfwGetKey(window, GLFW_KEY_A)) {
		//m.position = vec3_add(cam.position, vec3_new(0.0, 1.0, 0.0)); 
	}
	else if(GLFW_PRESS == glfwGetKey(window, GLFW_KEY_S)) {
		//m.position = vec3_sub(cam.position, vec3_new(0.0, 1.0, 0.0)); 
	}
	else if(GLFW_PRESS == glfwGetKey(window, GLFW_KEY_D)) {
		//m.position = vec3_sub(cam.position, vec3_new(1.0, 0.0, 0.0)); 
	}
	else if(GLFW_PRESS == glfwGetKey(window, GLFW_KEY_Q)) {
		
	}
	else if(GLFW_PRESS == glfwGetKey(window, GLFW_KEY_E)) {
		
	}
			
}

void gl_update(GLFWwindow *window)
{
	previous_seconds = glfwGetTime();
	update_fps_counter(window);
	/*
	float cam_y = cam_pos.y;
	
	if(forward) {
		cam_pos += speed * cam_dir;
		cam_pos.y = cam_y;
	}
	if(left) {
		cam_pos -= glm::normalize(glm::cross(cam_dir, cam_up)) * speed;
		cam_pos.y = cam_y;
	}
	if(right) {
		cam_pos += glm::normalize(glm::cross(cam_dir, cam_up)) * speed;
		cam_pos.y = cam_y;
	}
	if(back) {
		cam_pos -= speed * cam_dir;
		cam_pos.y = cam_y;
	}
	if(up) {
		cam_pos -= glm::vec3(0, 1, 0) * speed; 	
	}
	if(down) {
		cam_pos += glm::vec3(0, 1, 0) * speed;
	}
	*/
}

void gl_cleanup(GLFWwindow *window)
{
	glDeleteTextures(1, &tex);
	glDeleteBuffers(1, &vao);
	glDeleteBuffers(1, &vbo);
	
	glfwDestroyWindow(window);
       	glfwTerminate();
}

void glfw_error_callback(int error, const char *description)
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
			forward = true;
		}
		if(key == GLFW_KEY_A) {
			left = true;
		}
		if(key == GLFW_KEY_S) {
			back = true;
		}
		if(key == GLFW_KEY_D) {
			right = true;
		}
		if(key == GLFW_KEY_Q) {
			up = true;
		}
		if(key == GLFW_KEY_E) {
			down = true;
		}
	}
	else if(action == GLFW_RELEASE) {
		if(key == GLFW_KEY_W) {
			forward = false;
		}
		if(key == GLFW_KEY_A) {
			left = false;
		}
		if(key == GLFW_KEY_S) {
			back = false;
		}
		if(key == GLFW_KEY_D) {
			right = false;
		}
		if(key == GLFW_KEY_Q) {
			up = false;
		}
		if(key == GLFW_KEY_E) {
			down = false;
		}	
	}
	
}
void gl_mouse_callback(GLFWwindow* window, double xpos, double ypos)
{
	/*
	if(first_mouse)
	{
		last_x = xpos;
		last_y = ypos;
		first_mouse = false;
	}
	
	float xoffset = xpos - last_x;
	float yoffset = last_y - ypos; 
	last_x = xpos;
	last_y = ypos;

	float sensitivity = 0.05;
	xoffset *= sensitivity;
	yoffset *= sensitivity;

	yaw   += xoffset;
	pitch += yoffset;

	if(pitch > 89.0f)
		pitch = 89.0f;
	if(pitch < -89.0f)
		pitch = -89.0f;

	glm::vec3 front;
	front.x = cos(glm::radians(yaw)) * cos(glm::radians(pitch));
	front.y = sin(glm::radians(pitch));
	front.z = sin(glm::radians(yaw)) * cos(glm::radians(pitch));
	cam_dir = glm::normalize(front);
	cam_up = glm::vec3(0.0, 1.0, 0.0);
	*/
}

void gl_scroll_callback(GLFWwindow* window, double xoffset, double yoffset)
{
        /*
	if(fov >= 1.0f && fov <= 90.0f)
		fov -= yoffset;
	if(fov <= 1.0f)
		fov = 1.0f;
	if(fov >= 90.0f)
		fov = 90.0f;
	*/
}

GLuint load_shader(const char *filename, GLenum shadertype)
{
	FILE *fp = fopen(filename, "r");

	fseek(fp, 0L, SEEK_END);
	size_t size = ftell(fp);
	rewind(fp);

	char *buffer = calloc(size+1, sizeof(char));
	fread(buffer, size, 1, fp);

	fclose(fp);
	GLuint shader_prog = glCreateShader(shadertype);
	glShaderSource(shader_prog, 1, (const GLchar * const *)&buffer, NULL);
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
	GLuint program = glCreateProgram();
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

	glGenVertexArrays(1, &vao);
	glBindVertexArray(vao);
		
	glGenBuffers(1, &vbo);
	glBindBuffer(GL_ARRAY_BUFFER, vbo);
	glBufferData(GL_ARRAY_BUFFER, sizeof(vertices), vertices, GL_STATIC_DRAW);
		
	glEnableVertexAttribArray(glGetAttribLocation(shader, "in_position"));
	glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, 5 * sizeof(float), (void*)0);
	glEnableVertexAttribArray(glGetAttribLocation(shader, "in_tex_coords"));
	glVertexAttribPointer(1, 2, GL_FLOAT, GL_FALSE, 5 * sizeof(float), (void*)(3 * sizeof(float)));
}

void init_texture(int window_width, int window_height)
{
	glActiveTexture(GL_TEXTURE0);	
	glGenTextures(1, &tex);
	glBindTexture(GL_TEXTURE_2D, tex);
	glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA, window_width, window_height, 0, GL_RGBA, GL_UNSIGNED_BYTE, 0);
	glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
	glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
}

void update_fps_counter(GLFWwindow *w)
{
	char tmp[64];

	double current_seconds = glfwGetTime();
	double elapsed_seconds = current_seconds - previous_seconds;
	if(elapsed_seconds > 0.0000001) {
		previous_seconds = current_seconds;

		double fps = (double)frame_count / elapsed_seconds;
		sprintf(tmp, "OpenGL - Raytracer @ fps: %.2f", fps);
		glfwSetWindowTitle(w, tmp);
		frame_count = 0;
	}
	frame_count++;
}

