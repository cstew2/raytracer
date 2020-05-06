#include <math.h>

#include "math/vector.h"

const vec2 vec2_origin = {0.0, 0.0};
const vec3 vec3_origin = {0.0, 0.0, 0.0};
const vec4 vec4_origin = {0.0, 0.0, 0.0, 0.0};

//vec2 functions
vec2 vec2_new(const vec2_t x, const vec2_t y)
{
	vec2 v = (vec2){x, y};
	return v;
}

//vec3 functions
vec3 vec3_new(const vec3_t x, const vec3_t y, const vec3_t z)
{
	vec3 v = (vec3){x, y, z};
	return v;
}


vec3 vec3_add(const vec3 u, const vec3 v)
{
	vec3 w = (vec3){u.x + v.x, u.y + v.y, u.z + v.z};
	return w;
}

vec3 vec3_sub(const vec3 u, const vec3 v)
{
	vec3 w = (vec3){u.x - v.x, u.y - v.y, u.z - v.z};
	return w;
}

vec3 vec3_scale(const vec3 u, const vec3_t a)
{
	vec3 w = (vec3){u.x * a, u.y * a, u.z * a};
	return w;	
}

vec3_t vec3_dot(const vec3 u, const vec3 v)
{
	vec3_t f = u.x * v.x + u.y * v.y + u.z * v.z;
	return f;
}

vec3 vec3_cross(const vec3 u, const vec3 v)
{
	vec3 w = (vec3){u.y * v.z - u.z * v.y,
			u.z * v.x - u.x * v.z,
			u.x * v.y - u.y * v.x};
	return w;
}


vec3_t vec3_length(const vec3 u)
{
	vec3_t f = sqrt(powf(u.x, 2) + powf(u.y, 2) + powf(u.z, 2));
	return f;
}

vec3 vec3_normalize(const vec3 u)
{
	vec3_t f = sqrt(powf(u.x, 2) + powf(u.y, 2) + powf(u.z, 2));
	vec3 w = (vec3){u.x/f, u.y/f, u.z/f};
	return w;
}

bool vec3_compare(const vec3 u, const vec3 v)
{
	if(u.x != v.x || u.y != v.y || u.z != v.z){
		return false;
	}
	return true;
}


vec3 vec3_rotation(const vec3 u, const vec3 about, const vec3_t theta)
{
	//v = u * cos(theta)+(about x u) * sin(theta) + k(about.u)(1 - cos(theta))
	vec3 v;

	vec3_t s = sinf(theta);
	vec3 cross = vec3_cross(u, about);
	cross = vec3_scale(cross, s);

	vec3_t c = cosf(theta);
	vec3_t dot = vec3_dot(u, about);
	vec3 d = vec3_scale(about, dot * (1 - c));

	v = u;
	v = vec3_scale(v, c);
	v = vec3_add(v, cross);
	v = vec3_add(v, d);

	return v;
}


//vec4 functions
vec4 vec4_new(const vec4_t w, const vec4_t x, const vec4_t y, const vec4_t z)
{
	vec4 v = (vec4){x, y, z, w};
	return v;
}


vec4 vec4_add(const vec4 u, const vec4 v)
{
	vec4 w = (vec4){u.x + v.x, u.y + v.y, u.z + v.z, u.w + v.w};
	return w;	
}

vec4 vec4_sub(const vec4 u, const vec4 v)
{
	vec4 w = (vec4){u.x - v.x, u.y - v.y, u.z - v.z, u.w - v.w};
	return w;
}

vec4 vec4_scale(const vec4 u, const vec4_t a)
{
	vec4 w = (vec4){u.x * a, u.y * a, u.z * a, u.w * a};
	return w;	
}

vec4_t vec4_dot(const vec4 u, const vec4 v)
{
	vec4_t f = u.x * v.x + u.y * v.y + u.z * v.z + u.w * v.w;
	return f;
}


vec4_t vec4_length(const vec4 u)
{
	vec4_t f = sqrt(powf(u.x, 2) + powf(u.y, 2) + powf(u.z, 2) + powf(u.w, 2));
	return f;
}

vec4 vec4_normalize(const vec4 u)
{
	vec4_t f = sqrt(powf(u.x, 2) + powf(u.y, 2) + powf(u.z, 2) + powf(u.w, 2));
	vec4 w = (vec4){u.x/f, u.y/f, u.z/f, u.w/f};
	return w;
}

bool vec4_compare(const vec4 u, const vec4 v)
{
	if(u.x != v.x || u.y != v.y || u.z != v.z || u.w != v.w){
		return false;
	}
	return true;
}
