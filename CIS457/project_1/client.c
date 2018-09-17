#include <arpa/inet.h>
#include <errno.h>
#include <netinet/in.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/socket.h>
#include <unistd.h>

# define MAXDATASIZE 4096

int main(char* argc, char** argv) {
  if (argc != 3) {
    fprintf(stderr, "usage: client ip port");
    return EXIT_FAILURE;
  }
}
