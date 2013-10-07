CPPFLAGS= -I./ -arch=sm_20 -O3
LDFLAGS= -Xcompiler -fopenmp -use_fast_math
LDLIBS= -lcufft

sl1m: sl1m.o radioL1cuda.o readvises.o
	nvcc $(LDFLAGS) -o sl1m sl1m.o radioL1cuda.o readvises.o $(LDLIBS) 

sl1m.o: sl1m.cu radioL1cuda.h config.h readvises.h
	nvcc $(CPPFLAGS) -c sl1m.cu

radioL1cuda.o: radioL1cuda.cu radioL1cuda.h config.h
	nvcc $(CPPFLAGS) -c radioL1cuda.cu

readvises.o: readvises.cpp readvises.h
	nvcc $(CPPFLAGS) -c readvises.cpp
