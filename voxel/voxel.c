#include <stdlib.h>

#include "voxel.h"

voxel new_voxel(color c)
{
	voxel v;
	v.c = c;
	v.children_mask = 0x00;
	return v;
}
