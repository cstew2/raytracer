#include <stdlib.h>

#include "voxel/voxel.h"
#include "math/bits.h"

voxel voxel_new(const vec4 c, voxel *parent)
{
	voxel v;
	v.c = c;
	v.children_mask = 0x00;
	v.children = NULL;
	v.parent = parent;
	return v;
}

void voxel_set_children_num(voxel *v, const int children_num)
{
	v->children_mask = get_linear_bits(children_num);
	v->children = malloc(sizeof(voxel) * children_num);
}

void voxel_add_child(voxel *v, const voxel *u)
{
	int child_num = voxel_count_children(v) + 1;
	v->children = realloc(v->children, child_num);
	v->children[child_num] = *u;
	v->children_mask <<= 1;
	v->children_mask |= 1;
}

int voxel_count_children(const voxel *v)
{
	return count_linear_bits(v->children_mask);
}
