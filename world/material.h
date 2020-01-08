#ifndef __MATERIAL_H__
#define __MATERIAL_H__

#include <stdbool.h>

typedef struct {
	bool opaque;
	float ambient;
	float specular;
	float diffuse;
	float reflection;
	float refraction;
}material;

extern const material glass;
extern const material mirror;
extern const material matte;

#endif
