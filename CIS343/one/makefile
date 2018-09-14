all: file_utils.h file_utils.c reverse.c
	gcc -o reverse file_utils.c reverse.c -g

test:
	./reverse sample.txt sample2.txt
	./reverse sample2.txt sample3.txt
