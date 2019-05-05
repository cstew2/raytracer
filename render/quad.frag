#version 420 core

in vec2 v_tex_coords;

out vec4 out_colour;

uniform sampler2D tex_sample;

void main()
{
	out_colour = texture(tex_sample, v_tex_coords);
	//out_colour = vec4(0.5, 0.1, 0.8, 1.0);
}
