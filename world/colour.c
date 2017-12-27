#include "colour.h"

colour new_colour(uint8_t r, uint8_t g, uint8_t b, uint8_t a)
{
	colour c;
	c.rgba = r | (g << 8) | (b << 16) | (a << 24);
	return c;
}
