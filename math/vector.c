#include <math.h>

#include "vector.h"

const v3 v3_origin = {0.0, 0.0, 0.0};
const v4 v4_origin = {0.0, 0.0, 0.0, 0.0};

v3 v3_add(const v3 u, const v3 v)
{
	v3 w;
	w.x = u.x + v.x;
	w.y = u.y + v.y;
	w.z = u.z + v.z;
	return w;
}

v3 v3_sub(const v3 u, const v3 v)
{
	v3 w;
	w.x = u.x - v.x;
	w.y = u.y - v.y;
	w.z = u.z - v.z;
	return w;
}

v3 v3_scale(const v3 u, const float a)
{
	v3 w;
	w.x = u.x * a;
	w.y = u.y * a;
	w.z = u.z * a;
	return w;
}

float v3_dot(const v3 u, const v3 v)
{
	return (u.x * v.x) + (u.y * v.y) + (u.z * v.z);
}

v3 v3_cross(const v3 u, const v3 v)
{
	v3 w;
	w.x = u.y*v.z - u.z*v.y;
	w.x = u.z*v.x - u.x*v.z;
	w.z = u.x*v.y - u.y*v.x;
	return w;
}

float v3_length(const v3 u)
{
	return sqrt((u.x*u.x)+(u.y*u.y)+(u.z*u.z)); 
}

v3 v3_normalise(const v3 u)
{
	float length = v3_length(u);
	v3 v = u;
	if(length) {
		float n = 1/length;
		v.x *= n;
		v.y *= n;
		v.z *= n;
	}
	return v;
}

v3 new_v3(const float x, const float y, const float z)
{
	return (v3){x, y, z};
}

v4 v4_add(const v4 u, const v4 v)
{
	v4 w;
	w.w = u.w + v.w;
	w.x = u.x + v.x;
	w.y = u.y + v.y;
	w.z = u.z + v.z;
	return w;
}

v4 v4_sub(const v4 u, const v4 v)
{
	v4 w;
	w.w = u.w - v.w;
	w.x = u.x - v.x;
	w.y = u.y - v.y;
	w.z = u.z - v.z;
	return w;
}

v4 v4_scale(const v4 u, const float a)
{
	v4 w;
	w.w = u.w * a;
	w.x = u.x * a;
	w.y = u.y * a;
	w.z = u.z * a;
	return w;
}

float v4_dot(const v4 u, const v4 v)
{
	return (u.w * v.w) + (u.x * v.x) + (u.y * v.y) + (u.z * v.z);
}

v4 v4_cross(const v4 u, const v4 v)
{
	v4 w;
	w.x = u.y*v.z - u.z*v.y;
	w.x = u.z*v.x - u.x*v.z;
	w.z = u.x*v.y - u.y*v.x;
	return w;
}

float v4_length(const v4 u)
{
	return sqrt((u.w*u.w)+(u.x*u.x)+(u.y*u.y)+(u.z*u.z)); 
}

v4 v4_normalise(const v4 u)
{
	float length = v4_length(u);
	v4 v = u;
	if(length) {
		float n = 1/length;
		v.w *= n;
		v.x *= n;
		v.y *= n;
		v.z *= n;
	}
	return v;
}

v4 new_v4(const float w, const float x, const float y, const float z)
{
	return (v4){w, x, y, z};
}
