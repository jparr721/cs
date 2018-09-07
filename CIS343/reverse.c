#include "file_utils.c"
#include <errno.h>
#include <stdio.h>

int reverse(char* file_1, char* file_2) {
  char* buffer = malloc(sizeof(char)*2048);
  char* new_file = malloc(sizeof(char)*2048);

  if (read_file(file_1, &buffer) != 0) {
    fprinf(stderr, "BOYYYY WE CAN'T READ INTO THIS BUFFER");
    return errno;
  }

  if (write_file(file_2, new_file, sizeof(buffer))) {

  }
}

int main(int argc, char** argv) {
  char* file_1 = argv[1];
  char* file_2 = argv[2];
}
