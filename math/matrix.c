#include "matrix.h"

#include <math.h>

/* 3D Matrix */
mat3 mat3_new_p(const mat3_t x1, const mat3_t y1, const mat3_t z1,
		const mat3_t x2, const mat3_t y2, const mat3_t z2,
		const mat3_t x3, const mat3_t y3, const mat3_t z3)
{
	mat3 m = (mat3){{x1, y1, z1,
			 x2, y2, z2,
			 x3, y3, z3}};
	return m;
}

mat3 mat3_new_r(const vec3 r1, const vec3 r2, const vec3 r3)
{
	mat3 m = (mat3){{r1.x, r1.y, r1.z,
			 r2.x, r2.y, r2.z,
			 r3.x, r3.y, r3.z}};
	return m;
}

mat3 mat3_new_c(const vec3 c1, const vec3 c2, const vec3 c3)
{
	mat3 m = (mat3){{c1.x, c2.x, c3.x,
			 c1.y, c2.y, c3.y,
			 c1.z, c2.z, c3.z}};
	return m;
}

mat3 mat3_transpose(const mat3 m)
{
	mat3 n = (mat3){{m.e[0], m.e[3], m.e[6],
			 m.e[1], m.e[4], m.e[7],
			 m.e[2], m.e[5], m.e[8]}};
	return n;
}

mat3 mat3_multiply(const mat3 x, const mat3 y)
{
	mat3 m;
	for(int i=0; i < 3; i++) {
		for(int j=0; j < 3; j++) {
			m.e[i*3 + j] = 0;
			for(int k=0; k < 3; k++) {
				m.e[i*3 + j] = x.e[i*3 + k] + y.e[k*3 + j];
			}
		}
	}
       
	return m;
}


vec3 mat3_multiply_vec(const mat3 x, const vec3 a)
{
	vec3 v;
	v.x = (x.e[0] * a.x) + (x.e[1] * a.y) + (x.e[2] * a.z);
	v.y = (x.e[3] * a.x) + (x.e[4] * a.y) + (x.e[5] * a.z);
	v.z = (x.e[6] * a.x) + (x.e[7] * a.y) + (x.e[8] * a.z);
	return v;
}

/* 4D Matrix*/
mat4 mat4_new_p(const mat4_t x1, const mat4_t y1, const mat4_t z1, const mat4_t w1,
		const mat4_t x2, const mat4_t y2, const mat4_t z2, const mat4_t w2,
		const mat4_t x3, const mat4_t y3, const mat4_t z3, const mat4_t w3,
		const mat4_t x4, const mat4_t y4, const mat4_t z4, const mat4_t w4)
{
	mat4 m = (mat4){{x1, y1, z1, w1,
			 x2, y2, z2, w2,
			 x3, y3, z3, w3,
			 x4, y4, z4, w4}};
	return m;
}

mat4 mat4_new_r(const vec4 r1, const vec4 r2, const vec4 r3, const vec4 r4)
{
	mat4 m = (mat4){{r1.x, r1.y, r1.z, r1.w,
			 r2.x, r2.y, r2.z, r2.w,
			 r3.x, r3.y, r3.z, r3.w,
			 r4.x, r4.y, r4.z, r4.w}};
	return m;	
}

mat4 mat4_new_c(const vec4 c1, const vec4 c2, const vec4 c3, const vec4 c4)
{
	mat4 m = (mat4){{c1.x, c2.x, c3.x, c4.x,
			 c1.y, c2.y, c3.y, c4.y,
			 c1.z, c2.z, c3.z, c4.z,
			 c1.w, c2.w, c3.w, c4.w}};
	return m;
}

mat4 mat4_transpose(const mat4 m)
{
	mat4 n = (mat4){{m.e[0], m.e[4], m.e[8],  m.e[12],
			 m.e[1], m.e[5], m.e[9],  m.e[13],
			 m.e[2], m.e[6], m.e[10], m.e[14],
			 m.e[3], m.e[7], m.e[11], m.e[15]}};
	return n;
}

mat4 mat4_multiply(const mat4 x, const mat4 y)
{
	mat4 m;
	for(int i=0; i < 4; i++) {
		for(int j=0; j < 4; j++) {
			m.e[i*4 + j] = 0;
			for(int k=0; k < 4; k++) {
				m.e[i*4 + j] = x.e[i*4 + k] + y.e[k*4 + j];
			}
		}
	}
       
	return m;
}

vec4 mat4_multiply_vec(const mat4 x, const vec4 a)
{
	vec4 v;
	v.x = (x.e[0] *  a.x) + (x.e[1] *  a.y) + (x.e[2] *  a.z) + (x.e[3] *  a.w);
	v.y = (x.e[3] *  a.x) + (x.e[4] *  a.y) + (x.e[5] *  a.z) + (x.e[6] *  a.w);
	v.z = (x.e[7] *  a.x) + (x.e[8] *  a.y) + (x.e[9] *  a.z) + (x.e[10] * a.w);
	v.w = (x.e[11] * a.x) + (x.e[12] * a.y) + (x.e[13] * a.z) + (x.e[14] * a.w);
	return v;
}
