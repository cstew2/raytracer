#ifndef _VOXEL_H_
#define _VOXEL_H_

#include <stdint.h>

#include "math/vector.h"

typedef struct voxel_t{
	vec4 c;
	uint8_t children_mask;
	struct voxel_t *parent;
	struct voxel_t *children;
}voxel;

voxel voxel_new(const vec4 c, voxel *parent);
void voxel_set_children_num(voxel *v, const int children_num);
void voxel_add_child(voxel *v, const voxel *u);
int voxel_count_children(const voxel *v);

#endif
