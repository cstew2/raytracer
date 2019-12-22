#ifndef _MATRIX_H_
#define _MATRIX_H_

#include "vector.h"

typedef float mat3_t;
typedef struct {
	mat3_t e[9];
}mat3;

typedef float mat4_t;
typedef struct {
	mat4_t e[16];
}mat4;

mat3 mat3_new_p(const mat3_t x1, const mat3_t y1, const mat3_t z1,
		const mat3_t x2, const mat3_t y2, const mat3_t z2,
		const mat3_t x3, const mat3_t y3, const mat3_t z3);
mat3 mat3_new_r(const vec3 r1, const vec3 r2, const vec3 r3);
mat3 mat3_new_c(const vec3 c1, const vec3 c2, const vec3 c3);

mat3 mat3_transpose(const mat3 m);
mat3 mat3_multiply(const mat3 x, const mat3 y);
vec3 mat3_multiply_vec(const mat3 x, const vec3 a);


mat4 mat4_new_p(const mat4_t x1, const mat4_t y1, const mat4_t z1, const mat4_t w1,
		const mat4_t x2, const mat4_t y2, const mat4_t z2, const mat4_t w2,
		const mat4_t x3, const mat4_t y3, const mat4_t z3, const mat4_t w3,
		const mat4_t x4, const mat4_t y4, const mat4_t z4, const mat4_t w4);
mat4 mat4_new_r(const vec4 r1, const vec4 r2, const vec4 r3, const vec4 r4);
mat4 mat4_new_c(const vec4 r1, const vec4 r2, const vec4 r3, const vec4 r4);

mat4 mat4_transpose(const mat4 m);
mat4 mat4_multiply(const mat4 x, const mat4 y);
vec4 mat4_multiply_vec(const mat4 x, const vec4 a);


#endif
