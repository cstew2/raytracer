#ifndef _VECTOR_H_
#define _VECTOR_H_

typedef struct {
	float x;
	float y;
	float z;
}v3;


typedef struct {
	float w;
	float x;
	float y;
	float z;
}v4;


extern const v3 v3_origin;
extern const v4 v4_origin;

v3 v3_add(const v3 u, const v3 v);
v3 v3_sub(const v3 u, const v3 v);
v3 v3_scale(const v3 u, const float a);
float v3_dot(const v3 u, const v3 v);
v3 v3_cross(const v3 u, const v3 v);

float v3_length(const v3 u);
v3 v3_normalise(const v3 u);
int v3_compare(const v3 u, const v3 v);

v3 new_v3(const float x, const float y, const float z);


v4 v4_add(const v4 u, const v4 v);
v4 v4_sub(const v4 u, const v4 v);
v4 v4_scale(const v4 u, const float a);
float v4_dot(const v4 u, const v4 v);
v4 v4_cross(const v4 u, const v4 v);

float v4_length(const v4 u);
v4 v4_normalise(const v4 u);
int v4_compare(const v4 u, const v4 v);

v4 new_v4(const float w, const float x, const float y, const float z);

#endif
