#version 460 core

layout (location = 0) in vec3 in_position;
layout (location = 1) in vec2 in_tex_coords;

out vec2 v_tex_coords;

void main()
{
	gl_Position = vec4(in_position, 1.0);
	v_tex_coords = in_tex_coords;
}
