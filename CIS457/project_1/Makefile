all: client/client.c server/server.c
	gcc -Wall -g server/server.c -o s
	gcc -Wall -g client/client.c -o c
	mv s server/
	mv c client/

clean:
	rm -rf client/c server/s
