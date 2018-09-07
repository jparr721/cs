#include "file_utils.h"
#include <errno.h>
#include <stdio.h>
#include <unistd.h>

int reverse(const char* file_1, const char* file_2) {
  char* buffer = (char*)malloc(4096 * sizeof(char));

  if (buffer == 0) {
    fprintf(stderr, "BOYYYY WE HAD AN ERROR ALLOCATING THIS MEMORY");
    return errno;
  }

  ssize_t read = 0;

  if ((read = read_file(file_1, &buffer)) != 0) {
    fprintf(stderr, "BOYYYY WE CAN'T READ INTO THIS BUFFER");
    return errno;
  }

  if ((read = write_file(file_2, buffer, sizeof(buffer)) != 0)) {
    fprintf(stderr, "BOYYY WE CAN'T WRITE TO THIS BUFFER");
    return errno;
  }

  free(buffer);

  return 0;
}

int main(int argc, char** argv) {
  if (argc != 3) {
    printf("Invalid number of arguments provided, exiting");
    return EXIT_FAILURE;
  }

  char* file_1 = argv[1];
  char* file_2 = argv[2];

  reverse(file_1, file_2);

  return EXIT_SUCCESS;
}
