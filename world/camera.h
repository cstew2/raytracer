#ifndef 
#define 

typedef struct {
	vect3 position;
	vect3 lookat;
	vect3 up;
}camera;

camera new_camera(vect3 p, vect3 l, vect3 u);

#endif
