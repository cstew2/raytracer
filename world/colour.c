#include "colour.h"

colour colour_new(uint8_t r, uint8_t g, uint8_t b)
{
        colour c;
	c = 0xFF << 24 | b << 16 | g << 8 | r;
	return c;
}

uint8_t get_channel(colour c, channel ch)
{
	switch(ch) {
	default:
	case RED:
		return (c >> 24);
	case GREEN:
		return (c >> 16);
	case BLUE:
		return (c >> 8);
	case ALPHA:
		return c & 0xFFFFFF00;
	}
}

