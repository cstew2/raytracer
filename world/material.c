#include "world/material.h"

const material glass = {
	.opacity = 0.0,
	.ambient = 0.0,
	.specular = 0.0,
	.diffuse = 0.0,
	.reflection = 0.2,
	.refraction = 1.0
};

const material mirror = {
	.opacity = 0.0,
	.ambient = 0.0,
	.specular = 0.0,
	.diffuse = 0.0,
	.reflection = 1.0,
	.refraction = 0.0
};


const material matte = {
	.opacity = 0.0,
	.ambient = 0.8,
	.specular = 0.0,
	.diffuse = 0.1,
	.reflection = 0.0,
	.refraction = 0.0
};

