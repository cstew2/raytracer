#include "colour.h"

colour new_colour(uint8_t r, uint8_t g, uint8_t b, uint8_t a)
{
	colour c;
        c.rgba = r;
	c.rgba |= (g << 8);
	c.rgba |= (b << 16);
	c.rgba |= (a << 24);
	return c;
}
