#ifndef __TRANSFORM_H__
#define __TRANSFORM_H__

#include "matrix.h"

typedef float trans_t;

mat3 mat3_rotate_x(trans_t theta);
mat3 mat3_rotate_y(trans_t theta);
mat3 mat3_rotate_z(trans_t theta);

mat4 mat4_rotate_x(trans_t theta);
mat4 mat4_rotate_y(trans_t theta);
mat4 mat4_rotate_z(trans_t theta);

mat4 mat4_scale(trans_t factors[3]);
mat4 mat4_translate(trans_t delta[3]);

#endif
