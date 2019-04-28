#include "matrix.h"

/* 3D Matrix */
mat3 mat3_new_p(const float x1, const float y1, const float z1,
		const float x2, const float y2, const float z2,
		const float x3, const float y3, const float z3)
{
	mat3 m;
	m.r1 = (vec3){x1, y1, z1};
	m.r1 = (vec3){x2, y2, z2};
	m.r1 = (vec3){x3, y3, z3};
	return m;
}

mat3 mat3_new_r(const vec3 r1, const vec3 r2, const vec3 r3)
{
	mat3 m = (mat3){r1, r2, r3};
	return m;
}

mat3 mat3_new_c(const vec3 c1, const vec3 c2, const vec3 c3)
{
	mat3 m = mat3_transpose((mat3){c1, c2, c3});
	return m;
}

mat3 mat3_transpose(const mat3 m)
{
	mat3 n = mat3_new_p(m.r1.x, m.r2.x, m.r2.x,
			    m.r1.y, m.r2.y, m.r2.y,
			    m.r1.z, m.r2.z, m.r2.z);
	return n;
}
	
/* 4D Matrix*/
mat4 mat4_new_p(const float x1, const float y1, const float z1, const float w1,
		const float x2, const float y2, const float z2, const float w2,
		const float x3, const float y3, const float z3, const float w3,
		const float x4, const float y4, const float z4, const float w4)
{
	mat4 m;
	m.r1 = (vec4){x1, y1, z1, w1};
	m.r1 = (vec4){x2, y2, z2, w2};
	m.r1 = (vec4){x3, y3, z3, w3};
	m.r1 = (vec4){x4, y4, z4, w4};
	return m;
}

mat4 mat4_new_r(const vec4 r1, const vec4 r2, const vec4 r3, const vec4 r4)
{
	mat4 m = (mat4){r1, r2, r3, r4};
	return m;
}

mat4 mat4_new_c(const vec4 c1, const vec4 c2, const vec4 c3, const vec4 c4)
{
	mat4 m = mat4_transpose((mat4){c1, c2, c3, c4});
	return m;
}

mat4 mat4_transpose(const mat4 m)
{
	mat4 n = mat4_new_p(m.r1.x, m.r2.x, m.r2.x, m.r2.x,
			    m.r1.y, m.r2.y, m.r2.y, m.r2.y,
			    m.r1.z, m.r2.z, m.r2.z, m.r2.z,
			    m.r1.w, m.r2.w, m.r2.w, m.r2.w);
	return n;
}

