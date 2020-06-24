SRC		+= compute/cpu/cpu_raytracer.c
INC		+= compute/cpu/cpu_raytracer.h

CCSRC		+= compute/common_raytracer.cc
CCSRC		+= compute/common_raytracer.hh

CUSRC		+= compute/cuda/cuda_raytracer.cu
CUINC		+= compute/cuda/cuda_raytracer.cuh

MPSRC		+= compute/openmp/openmp_raytracer.c
MPINC		+= compute/openmp/openmp_raytracer.h

THSRC		+= compute/pthread/pthread_raytracer.c
THINC		+= compute/pthread/pthread_raytracer.h
