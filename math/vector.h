#ifndef _VECTOR_H_
#define _VECTOR_H_

typedef struct {
	float x;
	float y;
	float z;
}vec3;


typedef struct {
	float w;
	float x;
	float y;
	float z;
}vec4;


extern const vec3 vec3_origin;
extern const vec4 vec4_origin;


vec3 vec3_new(const float x, const float y, const float z);

vec3 vec3_add(const vec3 u, const vec3 v);
vec3 vec3_sub(const vec3 u, const vec3 v);
vec3 vec3_scale(const vec3 u, const float a);
float vec3_dot(const vec3 u, const vec3 v);
vec3 vec3_cross(const vec3 u, const vec3 v);

float vec3_length(const vec3 u);
vec3 vec3_normalise(const vec3 u);
int vec3_compare(const vec3 u, const vec3 v);

vec3 vec3_rotation(const vec3 u, const vec3 about, const float theta);

vec4 vec4_new(const float w, const float x, const float y, const float z);

vec4 vec4_add(const vec4 u, const vec4 v);
vec4 vec4_sub(const vec4 u, const vec4 v);
vec4 vec4_scale(const vec4 u, const float a);
float vec4_dot(const vec4 u, const vec4 v);
vec4 vec4_cross(const vec4 u, const vec4 v);

float vec4_length(const vec4 u);
vec4 vec4_normalise(const vec4 u);
int vec4_compare(const vec4 u, const vec4 v);

vec4 vec4_rotation(const vec4 u, const vec4 about);


#endif
