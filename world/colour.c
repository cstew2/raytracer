#include "colour.h"

colour new_colour(uint8_t r, uint8_t g, uint8_t b, uint8_t a)
{
	colour c;
        c.rgba = r;
	c.rgba >> 8 |= g;
	c.rgba >> 16 |= b;
	c.rgba >> 24 |= a;
	return c;
}
