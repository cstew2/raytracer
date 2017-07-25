#ifndef _COLOR_H_
#define _COLOR_H_

#include <stdint.h>

typedef struct {
        uint_fast32_t c;
}colour;

colour new_colour(uint_fast8_t r, uint_fast8_t g, uint_fast8_t b, uint_fast8_t a);

#endif
