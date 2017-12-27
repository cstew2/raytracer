#ifndef _COLOR_H_
#define _COLOR_H_

#include <stdint.h>

typedef struct {
        uint32_t rgba;
}colour;

colour new_colour(uint8_t r, uint8_t g, uint8_t b, uint8_t a);

#endif
