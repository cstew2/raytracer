#include "light.h"

ambient_light new_ambient_light(colour c, float intensity)
{
	ambient_light l;
	l.c = c;
	l.intensity = intensity;
	return l;
}
