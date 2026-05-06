# Worked in a pair: Nathaniel Black and Marcos Diaz Vazquez
CC=nvcc
#CFLAGS=-g
# Default build: B=64 (full convergence quality, grading-friendly), TF32 on,
# mega-graph on (captures the entire epoch into one CUDA graph; inner loop is
# 1 graph launch per epoch instead of ~4685 host calls).
# Override on the command line, e.g.
#   make CFLAGS="-std=c++17 -DBATCH_OVERRIDE=128 -DEPOCHS_OVERRIDE=10"
CFLAGS=-std=c++17 -DBATCH_OVERRIDE=64

nnp:loader.o nnp.o main.o kernels.o
	$(CC) $(CFLAGS) kernels.o nnp.o loader.o main.o -o nnp -lm -lcublas
loader.o: loader.h config.h loader.cc
	$(CC) -c loader.cc $(CFLAGS) -o loader.o

nnp.o: nnp.cu loader.h config.h nnp.h kernels.h
	$(CC) -c nnp.cu $(CFLAGS) -o nnp.o
main.o: main.cc config.h nnp.h
	$(CC) -c main.cc $(CFLAGS) -o main.o
kernels.o: kernels.cu kernels.h
	$(CC) -c kernels.cu $(CFLAGS) -o kernels.o
clean: prod
	rm -f nnp 
prod:
	rm -f *.o model.bin
