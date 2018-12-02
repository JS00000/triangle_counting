main: main.cu
	nvcc -ccbin g++ -O3 -m64 -Xcompiler -fopenmp -gencode arch=compute_60,code=sm_60 -gencode arch=compute_60,code=compute_60 -o main main.cu
