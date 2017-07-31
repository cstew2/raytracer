#ifndef _VOXEL_H_
#define _VOXEL_H_

#include <stdint.h>

#include "world/colour.h"

typedef struct voxel_t{
	colour c;
	uint8_t children_mask;
	struct voxel_t *parent;
	struct voxel_t *children;
}voxel;

voxel voxel_new(const colour c, voxel *parent);
void voxel_set_children_num(voxel *v, const int children_num);
void voxel_add_child(voxel *v, const voxel *u);
int voxel_count_children(const voxel *v);

#endif
