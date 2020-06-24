#ifndef __OPENMP_RAYTRACER_H__
#define __OPENMP_RAYTRACER_H__

#include "render/raytracer.h"

int openmp_render(const raytracer rt, void *cuda_rt);

#endif
