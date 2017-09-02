#ifndef _CAMERA_H_
#define _CAMERA_H_

#include "math/vector.h"

typedef struct {
	vec3 position;
	vec3 lookat;
	vec3 up;
}camera;

camera new_camera(vec3 p, vec3 l, vec3 u);

#endif
