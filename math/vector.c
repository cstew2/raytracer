#include <math.h>

#include "vector.h"

const vec3 vec3_origin = {0.0, 0.0, 0.0};
const vec4 vec4_origin = {0.0, 0.0, 0.0, 0.0};

vec3 vec3_add(const vec3 u, const vec3 v)
{
	vec3 w;
	w.x = u.x + v.x;
	w.y = u.y + v.y;
	w.z = u.z + v.z;
	return w;
}

vec3 vec3_sub(const vec3 u, const vec3 v)
{
	vec3 w;
	w.x = u.x - v.x;
	w.y = u.y - v.y;
	w.z = u.z - v.z;
	return w;
}

vec3 vec3_scale(const vec3 u, const float a)
{
	vec3 w;
	w.x = u.x * a;
	w.y = u.y * a;
	w.z = u.z * a;
	return w;
}

float vec3_dot(const vec3 u, const vec3 v)
{
	return (u.x * v.x) + (u.y * v.y) + (u.z * v.z);
}

vec3 vec3_cross(const vec3 u, const vec3 v)
{
	vec3 w;
	w.x = u.y*v.z - u.z*v.y;
	w.x = u.z*v.x - u.x*v.z;
	w.z = u.x*v.y - u.y*v.x;
	return w;
}

float vec3_length(const vec3 u)
{
	return sqrt((u.x*u.x)+(u.y*u.y)+(u.z*u.z)); 
}

vec3 vec3_normalise(const vec3 u)
{
	float length = vec3_length(u);
	vec3 v = u;
	if(length) {
		float n = 1/length;
		v.x *= n;
		v.y *= n;
		v.z *= n;
	}
	return v;
}

vec3 new_vec3(const float x, const float y, const float z)
{
	return (vec3){x, y, z};
}

vec4 vec4_add(const vec4 u, const vec4 v)
{
	vec4 w;
	w.w = u.w + v.w;
	w.x = u.x + v.x;
	w.y = u.y + v.y;
	w.z = u.z + v.z;
	return w;
}

vec4 vec4_sub(const vec4 u, const vec4 v)
{
	vec4 w;
	w.w = u.w - v.w;
	w.x = u.x - v.x;
	w.y = u.y - v.y;
	w.z = u.z - v.z;
	return w;
}

vec4 vec4_scale(const vec4 u, const float a)
{
	vec4 w;
	w.w = u.w * a;
	w.x = u.x * a;
	w.y = u.y * a;
	w.z = u.z * a;
	return w;
}

float vec4_dot(const vec4 u, const vec4 v)
{
	return (u.w * v.w) + (u.x * v.x) + (u.y * v.y) + (u.z * v.z);
}

vec4 vec4_cross(const vec4 u, const vec4 v)
{
	vec4 w;
	w.x = u.y*v.z - u.z*v.y;
	w.x = u.z*v.x - u.x*v.z;
	w.z = u.x*v.y - u.y*v.x;
	return w;
}

float vec4_length(const vec4 u)
{
	return sqrt((u.w*u.w)+(u.x*u.x)+(u.y*u.y)+(u.z*u.z)); 
}

vec4 vec4_normalise(const vec4 u)
{
	float length = vec4_length(u);
	vec4 v = u;
	if(length) {
		float n = 1/length;
		v.w *= n;
		v.x *= n;
		v.y *= n;
		v.z *= n;
	}
	return v;
}

vec4 new_vec4(const float w, const float x, const float y, const float z)
{
	return (vec4){w, x, y, z};
}
