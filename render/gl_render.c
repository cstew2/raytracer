#include <GL/glew.h>
#include <GLFW/glfw3.h> 
#include <assert.h>
#include <stdarg.h>
#include <stdio.h>
#include <time.h>
#include <stdlib.h>

#include "render/gl_render.h"
#include "debug/debug.h"

double previous_seconds;
const GLubyte *renderer;
const GLubyte *version;

GLuint vbo;
GLuint vao;

GLuint shader_programme;

GLuint vs;
GLuint fs;


/* we will use this function to update the window title with a frame rate */
void update_fps_counter(GLFWwindow *w)
{
	char tmp[64];

	static int frame_count;

	double current_seconds = glfwGetTime();
	double elapsed_seconds = current_seconds - previous_seconds;
	if(elapsed_seconds > 0.25) {
		previous_seconds = current_seconds;

		double fps = (double)frame_count / elapsed_seconds;
		sprintf(tmp, "OpenGL - Raytracer @ fps: %.2f", fps);
		glfwSetWindowTitle(w, tmp);
		frame_count = 0;
	}
	frame_count++;
}

int gl_init(void)
{
	const GLfloat points[] = {1.0f, 1.0f, 0.0f,
				  1.0f, -1.0f, 0.0f,
				  -1.0f, 1.0f, 0.0f,
				  1.0f, -1.0f, 0.0f,
				  -1.0f, 1.0f, 0.0f,
				  -1.0f, -1.0f, 0.0f};
	
	GLchar const *vertex_shader;
	GLchar const *fragment_shader;

	int shader_version = 130;
	const char *shader_version_string = (char *)glGetString(GL_SHADING_LANGUAGE_VERSION);
	if(shader_version_string) {
		shader_version = atoi(shader_version_string);
	}
	
	if(shader_version == 130) {
		vertex_shader = load_shader("./render/shaders/raytracer_130.vert");
		fragment_shader = load_shader("./render/shaders/raytracer_130.frag");
	}
	
	else if(shader_version == 450 ) {
		vertex_shader = load_shader("./render/shaders/raytracer_450.vert");
		fragment_shader = load_shader("./render/shaders/raytracer_450.frag");
	}
	
	// start GL context and O/S window using the GLFW helper library
	log_msg(INFO, "Starting GLFW: %s\n", glfwGetVersionString());
	// register the error call-back function that we wrote, above
	glfwSetErrorCallback(glfw_error_callback);
	if (!glfwInit()) {
		log_msg(ERROR, "Could not start GLFW3\n");
		return 1;
	}

	/* we can run a full-screen window here */

	/*GLFWmonitor* mon = glfwGetPrimaryMonitor ();
	  const GLFWvidmode* vmode = glfwGetVideoMode (mon);
	  GLFWwindow* window = glfwCreateWindow (
	  vmode->width, vmode->height, "Extended GL Init", mon, NULL
	  );*/

	window = glfwCreateWindow(WINDOW_WIDTH, WINDOW_HEIGHT, "OpenGL - Voxel Raytracer", NULL, NULL);
	if (!window) {
		log_msg(ERROR, "Could not open window with GLFW3\n");
		glfwTerminate();
		return 1;
	}
	glfwSetWindowSizeCallback(window, glfw_window_size_callback);
	glfwMakeContextCurrent(window);

	// start GLEW extension handler
	glewExperimental = GL_TRUE;
	glewInit();

	// get version info
	renderer = glGetString(GL_RENDERER); // get renderer string
	version = glGetString(GL_VERSION);	 // version as a string
	log_msg(INFO, "Renderer: %s\n", renderer);
	log_msg(INFO, "OpenGL version supported %s\n", version);
	log_msg(INFO, "Renderer: %s version: %s\n", renderer, version);
	log_gl_params();
	// tell GL to only draw onto a pixel if the shape is closer to the viewer
	//glEnable(GL_DEPTH_TEST); // enable depth-testing
	//glDepthFunc(GL_LESS);		 // depth-testing interprets a smaller value as "closer"

	glGenBuffers(1, &vbo);
	glBindBuffer(GL_ARRAY_BUFFER, vbo);
	glBufferData(GL_ARRAY_BUFFER, 18 * sizeof(GLfloat), points, GL_STATIC_DRAW);

	glGenVertexArrays(1, &vao);
	glBindVertexArray(vao);
	glEnableVertexAttribArray(0);
	glBindBuffer(GL_ARRAY_BUFFER, vbo);
	glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, 0, NULL);

	vs = glCreateShader(GL_VERTEX_SHADER);
	glShaderSource(vs, 1, &vertex_shader, NULL);
	glCompileShader(vs);
	fs = glCreateShader(GL_FRAGMENT_SHADER);
	glShaderSource(fs, 1, &fragment_shader, NULL);
	glCompileShader(fs);
	shader_programme = glCreateProgram();
	glAttachShader(shader_programme, fs);
	glAttachShader(shader_programme, vs);
	glLinkProgram(shader_programme);

	free((GLchar *)vertex_shader);
	free((GLchar *)fragment_shader);
	
	return 0;
}

int gl_render(void)
{
	previous_seconds = glfwGetTime();
	update_fps_counter(window);
	glClearColor(0.6f, 0.6f, 0.6f, 1.0f);
	glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
	glViewport(0, 0, WINDOW_WIDTH, WINDOW_HEIGHT);
	glUseProgram(shader_programme);
	glBindVertexArray(vao);
	glDrawArrays(GL_TRIANGLES, 0, 6);
	glfwSwapBuffers(window);

	return 0;
}

void gl_input(void)
{
	glfwPollEvents();
	if (GLFW_PRESS == glfwGetKey( window, GLFW_KEY_ESCAPE)) {
		glfwSetWindowShouldClose( window, 1);
	}
}

void gl_cleanup(void)
{
	glfwTerminate();
}

int load_texture(canvas *c, GLuint *tex)
{
	glGenTextures(1, tex);
	glActiveTexture(GL_TEXTURE0);
	glBindTexture(GL_TEXTURE_2D, *tex);
	glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA, c->width, c->height,
		     0, GL_RGBA_INTEGER, GL_UNSIGNED_BYTE, c->screen);
	glGenerateMipmap(GL_TEXTURE_2D);
	glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE);
	glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE);
	glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
	glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR_MIPMAP_LINEAR);
	GLfloat max_aniso = 0.0f;
	glGetFloatv(GL_MAX_TEXTURE_MAX_ANISOTROPY_EXT, &max_aniso);
	// set the maximum!
	glTexParameterf(GL_TEXTURE_2D, GL_TEXTURE_MAX_ANISOTROPY_EXT, max_aniso);

	return 0;
}


void log_gl_params(void)
{
	GLenum params[] = {
		GL_MAX_COMBINED_TEXTURE_IMAGE_UNITS,
		GL_MAX_CUBE_MAP_TEXTURE_SIZE,
		GL_MAX_DRAW_BUFFERS,
		GL_MAX_FRAGMENT_UNIFORM_COMPONENTS,
		GL_MAX_TEXTURE_IMAGE_UNITS,
		GL_MAX_TEXTURE_SIZE,
		GL_MAX_VARYING_FLOATS,
		GL_MAX_VERTEX_ATTRIBS,
		GL_MAX_VERTEX_TEXTURE_IMAGE_UNITS,
		GL_MAX_VERTEX_UNIFORM_COMPONENTS,
		GL_MAX_VIEWPORT_DIMS,
		GL_STEREO,
	};
	const char *names[] = {
		"GL_MAX_COMBINED_TEXTURE_IMAGE_UNITS",
		"GL_MAX_CUBE_MAP_TEXTURE_SIZE",
		"GL_MAX_DRAW_BUFFERS",
		"GL_MAX_FRAGMENT_UNIFORM_COMPONENTS",
		"GL_MAX_TEXTURE_IMAGE_UNITS",
		"GL_MAX_TEXTURE_SIZE",
		"GL_MAX_VARYING_FLOATS",
		"GL_MAX_VERTEX_ATTRIBS",
		"GL_MAX_VERTEX_TEXTURE_IMAGE_UNITS",
		"GL_MAX_VERTEX_UNIFORM_COMPONENTS",
		"GL_MAX_VIEWPORT_DIMS",
		"GL_STEREO",
	};
	log_msg(INFO, "GL Context Params:\n");
	// integers - only works if the order is 0-10 integer return types
	for (int i = 0; i < 10; i++) {
		int v = 0;
		glGetIntegerv(params[i], &v);
		log_msg(INFO, "%s %i\n", names[i], v);
	}
	// others
	int v[2];
	v[0] = v[1] = 0;
	glGetIntegerv(params[10], v);
	log_msg(INFO, "%s %i %i\n", names[10], v[0], v[1]);
	unsigned char s = 0;
	glGetBooleanv( params[11], &s);
	log_msg(INFO, "%s %i\n", names[11], (unsigned int)s);
	log_msg(INFO, "-----------------------------\n");
}

GLchar const *load_shader(const char *filename)
{
	if(filename == NULL) {
		log_msg(WARN, "no file given to load\n");
		return NULL;
	}
	FILE *f = NULL;
	size_t size;
	f = fopen(filename, "r");
	if(f == NULL) {
		log_msg(WARN, "Improper path given to load file\n");
		return NULL;
	}
	
	fseek(f, 0, SEEK_END);
	size = ftell(f);
	fseek(f, 0, SEEK_SET);
	
	GLchar *file_buffer = calloc(size, sizeof(GLchar));
	fread(file_buffer, 1, size, f);
	fclose(f);

	return file_buffer;	
}
