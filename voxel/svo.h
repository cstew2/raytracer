#ifndef _SVO_H_
#define _SVO_H_

#include "voxel/voxel.h"

typedef struct {
	voxel root;
}svo;

svo svo_new(void);
//void svo_add_voxel(const svo *s, const voxel *v);

#endif
