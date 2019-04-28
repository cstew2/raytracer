#ifndef _MATH_H_
#define _MATH_H_

#include "constants.h"

float deg2rad(float deg);
float rad2deg(float rad);

float fast_inv_sqrt(float x);
float fast_sqrt(float x);

float clamp(float x, float min, float max);

float maxf(float x, float y);
float minf(float x, float y);

int max(int x, int y);
int min(int x, int y);

#endif
