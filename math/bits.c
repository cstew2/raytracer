#include "bits.h"

#include  <stdint.h>

int count_linear_bits(const uint8_t x)
{
	switch(x) {
	case 0x01: return 1; break;
	case 0x02: return 2; break;
	case 0x07: return 3; break;
	case 0x0F: return 4; break;
	case 0x1F: return 5; break;
	case 0x3F: return 6; break;
	case 0x7F: return 7; break;
	case 0xFF: return 8; break;
	default: return -1; break;
	}
	return -1;
}

uint8_t get_linear_bits(const int x)
{
	switch(x) {
	case 1: return 0x01; break;
	case 2: return 0x02; break;
	case 3: return 0x07; break;
	case 4: return 0x0F; break;
	case 5: return 0x1F; break;
	case 6: return 0x3F; break;
	case 7: return 0x7F; break;
	case 8: return 0xFF; break;	
	default: return -1; break;
	}
}
