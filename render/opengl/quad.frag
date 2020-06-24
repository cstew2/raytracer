#version 460 core

in vec2 v_tex_coords;

layout (location = 0) out vec4 out_vec4;

uniform sampler2D tex_sample;

void main()
{
	out_vec4 = texture(tex_sample, v_tex_coords);
}
