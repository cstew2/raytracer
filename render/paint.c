#include <stdlib.h>
#include <stdio.h>
#include <stdarg.h>
#include <assert.h>
#include <time.h>

#include <GL/glew.h>
#include <GLFW/glfw3.h>

#include "render/paint.h"

const char *GL_LOG_FILE = "raytracer_gl.log";

GLuint vbo = 0;
GLuint vao = 0;
GLuint shader_programme = 0;
GLFWwindow *window = NULL;

int window_width = 640;
int window_height = 480;

canvas new_canvas(int width, int height, colour *c)
{
	canvas cv;
	cv.width = width;
	cv.height = height;
	if(c) {
		cv.screen = c;
	}
	else {
		cv.screen = malloc(sizeof(colour) * width * height);
	}
	return cv;
}

void gl_init(void)
{	assert(restart_gl_log());
        gl_log("starting GLFW\n%s\n", glfwGetVersionString());
	// register the error call-back function that we wrote, above
	glfwSetErrorCallback(glfw_error_callback);
	if(!glfwInit()) {
		fprintf(stderr, "ERROR: could not start GLFW3\n");
		return;
	}

	/*
	GLFWmonitor* mon = glfwGetPrimaryMonitor();
	const GLFWvidmode* vmode = glfwGetVideoMode(mon);
	window = glfwCreateWindow(vmode->width, vmode->height, "raytracer", mon, NULL);
	*/

	window = glfwCreateWindow(window_width, window_height, "raytracer", NULL, NULL);
	glfwSetWindowSizeCallback(window, glfw_window_size_callback);
	

	gl_log("Created Window\n");
	if (!window) {
		fprintf(stderr, "ERROR: could not open window with GLFW3\n");
		glfwTerminate();
		return;
	}
	glfwMakeContextCurrent(window);
                                  
	// start GLEW extension handler
	glewExperimental = GL_TRUE;
	glewInit();
	gl_log("Starting GLEW\n");
	
	// get version info
	const GLubyte* renderer = glGetString(GL_RENDERER); // get renderer string
	const GLubyte* version = glGetString(GL_VERSION); // version as a string
	gl_log("Renderer: %s\n", renderer);
	gl_log("OpenGL version supported %s\n", version);

	init();
	gl_log("Creating opengl structures\n");

	glClearColor(0.6f, 0.6f, 0.6f, 1.0f);

	gl_log("rendering\n");
	while(!glfwWindowShouldClose(window)) {
		render();
	}
	
	// close GL context and any other GLFW resources
	glfwTerminate();
	return;
}

void init(void)
{
	GLfloat point_values[] = {
		0.0f, 0.0f, 0.0f,
		1.0f, 0.0f, 0.0f,
		0.0f, 1.0f, 0.0f,
		
		1.0f, 0.0f, 0.0f,
		0.0f, 1.0f, 0.0f,
		1.0f, 1.0f, 0.0f
	};


	glGenBuffers(1, &vbo);
	glBindBuffer(GL_ARRAY_BUFFER, vbo);
	glBufferData(GL_ARRAY_BUFFER, 18 * sizeof(float), point_values, GL_STATIC_DRAW);

	glGenVertexArrays(1, &vao);
	glBindVertexArray(vao);
	glEnableVertexAttribArray(0);
	glBindBuffer(GL_ARRAY_BUFFER, vbo);
	glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, 0, NULL);

	const char *vertex_shader =
		"#version 150\n"
		"in vec3 vp;"
		"void main() {"
		"  gl_Position = vec4(vp, 1.0);"
		"}";

	const char *fragment_shader =
		"#version 150\n"
		"out vec4 frag_colour;"
		"void main() {"
		"  frag_colour = vec4(1.0, 0.0, 1.0, 1.0);"
		"}";

	GLuint vs = glCreateShader(GL_VERTEX_SHADER);
	glShaderSource(vs, 1, &vertex_shader, NULL);
	glCompileShader(vs);
	
	GLuint fs = glCreateShader(GL_FRAGMENT_SHADER);
	glShaderSource(fs, 1, &fragment_shader, NULL);
	glCompileShader(fs);

	shader_programme = glCreateProgram();
	glAttachShader(shader_programme, fs);
	glAttachShader(shader_programme, vs);
	glLinkProgram(shader_programme);
}

void render(void)
{
	// wipe the drawing surface clear
	glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
	glViewport(0, 0, window_width, window_height);
	
	glUseProgram(shader_programme);
	glBindVertexArray(vao);
	// draw points 0-3 from the currently bound VAO with current in-use shader
	glDrawArrays(GL_TRIANGLES, 0, 6);
	// update other events like input handling 
	glfwPollEvents();
	// put the stuff we've been drawing onto the display
	glfwSwapBuffers(window);
	
	if (GLFW_PRESS == glfwGetKey(window, GLFW_KEY_ESCAPE)) {
		glfwSetWindowShouldClose(window, 1);
	}
	return;
}

void glfw_error_callback(int error, const char* description)
{
	gl_log_err("GLFW ERROR: code %i msg: %s\n", error, description);
}

int gl_log_err(const char* message, ...) {
	va_list argptr;
	FILE* file = fopen(GL_LOG_FILE, "a");
	if(!file) {
		fprintf(stderr,
			"ERROR: could not open GL_LOG_FILE %s file for appending\n",
			GL_LOG_FILE);
		return 0;
	}
	va_start(argptr, message);
	vfprintf(file, message, argptr);
	va_end(argptr);
	va_start(argptr, message);
	vfprintf(stderr, message, argptr);
	va_end(argptr);
	fclose(file);
	return 1;
}


int restart_gl_log(void) {
	FILE* file = fopen(GL_LOG_FILE, "w");
	if(!file) {
		fprintf(stderr,
			"ERROR: could not open GL_LOG_FILE log file %s for writing\n",
			GL_LOG_FILE);
		return 0;
	}
	time_t now = time(NULL);
	char* date = ctime(&now);
	fprintf(file, "GL_LOG_FILE log. local time %s\n", date);
	fclose(file);
	return 1;
}

int gl_log(const char* message, ...) {
	va_list argptr;
	FILE* file = fopen(GL_LOG_FILE, "a");
	if(!file) {
		fprintf(
			stderr,
			"ERROR: could not open GL_LOG_FILE %s file for appending\n",
			GL_LOG_FILE
			);
		return 0;
	}
	va_start(argptr, message);
	vfprintf(file, message, argptr);
	va_end(argptr);
	fclose(file);
	return 1;
}

void glfw_window_size_callback(GLFWwindow* w, int width, int height)
{
        window_width = width;
	window_height = height;
	glfwSetWindowSize(w, width, height);
	/* update any perspective matrices used here */
}
