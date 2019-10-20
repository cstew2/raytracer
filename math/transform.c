#include <math.h>

#include "transform.h"

mat3 mat3_rotate_x(float theta)
{
	mat3 n = mat3_new_p(1, 0,            0,
			    0, cosf(theta), -sinf(theta),
			    0, sinf(theta),  cosf(theta));
	return n;
}

mat3 mat3_rotate_y(float theta)
{
	mat3 n = mat3_new_p( cosf(theta), 0, sinf(theta),
			     0,           1, 0,
			    -sinf(theta), 0, cosf(theta));
	return n;	
}

mat3 mat3_rotate_z(float theta)
{
        mat3 n = mat3_new_p(cosf(theta), -sinf(theta), 0,
			    sinf(theta),  cosf(theta), 0,
			    0,            0,          1);
	return n;	
}

mat4 mat4_rotate_x(float theta)
{
	mat4 n = mat4_new_p(1, 0,            0,           0,
			    0, cosf(theta), -sinf(theta), 0,
			    0, sinf(theta),  cosf(theta), 0,
			    0, 0,            0,           1);
	return n;
}

mat4 mat4_rotate_y(float theta)
{
	mat4 n = mat4_new_p( cosf(theta), 0, sinf(theta), 0,
			     0,           1, 0,          0,
			    -sinf(theta), 0, cosf(theta), 0,
			     0,           0, 0,          1);
	return n;	
}

mat4 mat4_rotate_z(float theta)
{
        mat4 n = mat4_new_p(cosf(theta), -sinf(theta), 0, 0,
			    sinf(theta),  cosf(theta), 0, 0,
			    0,            0,           1, 0,
			    0,            0,           0, 1);
	return n;	
}

mat4 mat4_scale(trans_t factors[3])
{
	mat4 n = mat4_new_p(factors[0], 0,          0,          0,
			    0,          factors[1], 0,          0,
			    0,          0,          factors[2], 0,
			    0,          0,          0,          1);
	return n;	
}

mat4 mat4_translate(trans_t delta[3])
{
	mat4 n = mat4_new_p(0, 0, 0, delta[0],
			    0, 0, 0, delta[1],
			    0, 0, 0, delta[2],
			    0, 0, 0, 1);
	return n;
}
