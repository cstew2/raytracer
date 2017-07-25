#include "color.h"

colour new_colour(uint_fast8_t r, uint_fast8_t g, uint_fast8_t b, uint_fast8_t a);
{
	colour c;
        c->c &= r;
	c->c &= g << 8;
	c->c &= b << 16;
	c->c &= a << 32;
	return c;
}
