#ifndef _MATRIX_H_
#define _MATRIX_H_

#include "vector.h"

typedef struct {
	vec3 r1;
	vec3 r2;
	vec3 r3;
}mat3;

typedef struct {
	vec4 r1;
	vec4 r2;
	vec4 r3;
	vec4 r4;
}mat4;

mat3 mat3_new_p(const float x1, const float y1, const float z1,
		const float x2, const float y2, const float z2,
		const float x3, const float y3, const float z3);

mat3 mat3_new_r(const vec3 r1, const vec3 r2, const vec3 r3);
mat3 mat3_new_c(const vec3 c1, const vec3 c2, const vec3 c3);

mat3 mat3_transpose(const mat3 m);
mat3 mat3_multiply_vector(const mat3 m, const vec3 v);


mat4 mat4_new_p(const float x1, const float y1, const float z1, const float w1,
		const float x2, const float y2, const float z2, const float w2,
		const float x3, const float y3, const float z3, const float w3,
		const float x4, const float y4, const float z4, const float w4);
mat4 mat4_new_r(const vec4 r1, const vec4 r2, const vec4 r3, const vec4 r4);
mat4 mat4_new_c(const vec4 r1, const vec4 r2, const vec4 r3, const vec4 r4);

mat4 mat4_transpose(const mat4 m);


#endif
