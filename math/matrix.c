#include "matrix.h"

m4 new_matrix_p(const float x1, const float y1, const float z1, const float w1,
		const float x2, const float y2, const float z2, const float w2,
		const float x3, const float y3, const float z3, const float w3,
		const float x4, const float y4, const float z4, const float w4)
{
	m4 m;
	m.r1 = (v4){x1, y1, z1, w1};
	m.r1 = (v4){x2, y2, z2, w2};
	m.r1 = (v4){x3, y3, z3, w3};
	m.r1 = (v4){x4, y4, z4, w4};
	return m;
}

m4 new_matrix_r(const v4 r1, const v4 r2, const v4 r3, const v4 r4)
{
	m4 m = (m4){r1, r2, r3, r4};
	return m;
}

m4 transpose(const m4 m)
{
	m4 n = new_matrix_p(m.r1.x, m.r2.x, m.r2.x, m.r2.x,
			    m.r1.y, m.r2.y, m.r2.y, m.r2.y,
			    m.r1.z, m.r2.z, m.r2.z, m.r2.z,
			    m.r1.w, m.r2.w, m.r2.w, m.r2.w);
	return n;
}

