#include <math.h>

#include "math.h"
#include "constants.h"

float deg2rad(float deg)
{
	return deg * DEG2RAD;
}

float rad2deg(float rad)
{
	return rad * RAD2DEG;
}

float fast_inv_sqrt(float x)
{
	float xhalf = 0.5f*x;
	int i = *(int*)&x;
	i = 0x5f3759df - (i>>1);
	float y = *(float*)&i;
	return y*(1.5f - xhalf*y*y);
}

float fast_sqrt(float x)
{
	int val = *(int*)&x;
	val -= 1 << 23;
	val >>= 1;
	val += 1 << 29;
	return *(float*)&val;
}

float clamp(float x, float min, float max)
{
        if(x < min) {
		return min;
	}
	if(x > max) {
		return max;
	}
	return x;
}

float maxf(float x, float y)
{
	return x > y ? x : y;
}

float minf(float x, float y)
{
	return x < y ? x : y;
}

int max(int x, int y)
{
	return x > y ? x : y;
}

int min(int x, int y)
{
	return x < y ? x : y;
}

