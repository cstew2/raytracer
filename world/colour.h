#ifndef __COLOUR_H__
#define __COLOUR_H__

#include <stdint.h>

typedef uint32_t colour;
typedef enum {RED, GREEN, BLUE, ALPHA} channel;

colour colour_new(uint8_t r, uint8_t g, uint8_t b);
uint8_t get_channel(colour c, channel ch);

#endif
