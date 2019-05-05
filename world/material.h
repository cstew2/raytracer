#ifndef __MATERIAL_H__
#define __MATERIAL_H__

#include <stdbool.h>

typedef struct {
	bool diffuse;
	bool reflection;
	bool refraction;
	float albedo;
	float reflective;
	float refractive;
}material;

extern const material glass;
extern const material matte;

#endif
