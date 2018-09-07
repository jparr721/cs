all: functions.h functions.o p1.o
	gcc -o p functions.o p1.o

p1.o: functions.h p1.c
	gcc -c p1.c

functions.o: functions.h functions.c
	gcc -c functions.c

enc:
	./p e FEATHER infile outfile

dec:
	./p d FEATHER outfile infile

test:
	./p e FEATHER infile outfile
	./p d FEATHER outfile newfile
	diff -s infile newfile

clean:
	rm -f *.o p1.o infile outfile
