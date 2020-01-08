#include "world/material.h"

const material glass = {
	.opaque = false,
	.ambient = 0.0,
	.specular = 0.0,
	.diffuse = 0.0,
	.reflection = 0.2,
	.refraction = 1.0
};

const material mirror = {
	.opaque = false,
	.ambient = 0.0,
	.specular = 0.0,
	.diffuse = 0.0,
	.reflection = 1.0,
	.refraction = 0.0
};


const material matte = {
	.opaque = true,
	.ambient = 0.8,
	.specular = 0.0,
	.diffuse = 0.1,
	.reflection = 0.0,
	.refraction = 0.0
};

