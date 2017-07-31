#ifndef _CAMERA_H_
#define _CAMERA_H_

#include "math/vector.h"

typedef struct {
	v3 position;
	v3 lookat;
	v3 up;
}camera;

camera new_camera(v3 p, v3 l, v3 u);

#endif
