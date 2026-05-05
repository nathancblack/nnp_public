CC=nvcc
#CFLAGS=-g
CFLAGS=

nnp:loader.o nnp.o main.o kernels.o
	$(CC) $(CFLAGS) kernels.o nnp.o loader.o main.o -o nnp -lm
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
