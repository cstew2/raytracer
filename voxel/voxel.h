#ifndef _VOXEL_H_
#define _VOXEL_H_

typedef struct {
	color c;
	unsigned char children_mask;
	struct voxel *children[8];
}voxel;

voxel new_voxel(color c);
void add_child(voxel *v);

#endif
