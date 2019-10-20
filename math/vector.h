#ifndef __VECTOR_H__
#define __VECTOR_H__

#include <stdbool.h>

typedef float vec3_t;
typedef struct {
	vec3_t x;
	vec3_t y;
	vec3_t z;
}vec3;

typedef float vec4_t;
typedef struct {
	vec4_t x;
	vec4_t y;
	vec4_t z;
	vec4_t w;
}vec4;

//constants
extern const vec3 vec3_origin;
extern const vec4 vec4_origin;

//vec3 functions
vec3 vec3_new(const vec3_t x, const vec3_t y, const vec3_t z);

vec3 vec3_add(const vec3 u, const vec3 v);
vec3 vec3_sub(const vec3 u, const vec3 v);
vec3 vec3_scale(const vec3 u, const vec3_t a);
vec3_t vec3_dot(const vec3 u, const vec3 v);
vec3 vec3_cross(const vec3 u, const vec3 v);

vec3_t vec3_length(const vec3 u);
vec3 vec3_normalise(const vec3 u);
bool vec3_compare(const vec3 u, const vec3 v);

vec3 vec3_rotation(const vec3 u, const vec3 about, const vec3_t theta);


//vec4 functions
vec4 vec4_new(const vec4_t w, const vec4_t x, const vec4_t y, const vec4_t z);

vec4 vec4_add(const vec4 u, const vec4 v);
vec4 vec4_sub(const vec4 u, const vec4 v);
vec4 vec4_scale(const vec4 u, const vec4_t a);
vec4_t vec4_dot(const vec4 u, const vec4 v);

vec4_t vec4_length(const vec4 u);
vec4 vec4_normalise(const vec4 u);
bool vec4_compare(const vec4 u, const vec4 v);

#endif
