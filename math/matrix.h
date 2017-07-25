#ifndef _MATRIX_H_
#define _MATRIX_H_

#include "vector.h"

typedef struct {
	v4 r1;
	v4 r2;
	v4 r3;
	v4 r4;
}m4;

m4 new_matrix_p(const float x1, const float y1, const float z1, const float w1,
		const float x2, const float y2, const float z2, const float w2,
		const float x3, const float y3, const float z3, const float w3,
		const float x4, const float y4, const float z4, const float w4);
m4 new_matrix_r(const v4 r1, const v4 r2, const v4 r3, const v4 r4);
m4 transpose(const m4 m);

#endif
