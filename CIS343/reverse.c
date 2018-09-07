#include "file_utils.h"
#include <errno.h>
#include <stdio.h>
#include <stdlib.h>
#include <sys/stat.h>
#include <sys/types.h>
#include <unistd.h>

off_t fsize(const char* filename) {
  FILE * fp;
  fp = fopen(filename, "r");
  fseek(fp, 0, SEEK_END);
  long bytes = ftell(fp);
  rewind(fp);
  fclose(fp);
  return bytes;
}

int reverse(const char* file_1, const char* file_2) {
  off_t file_size = fsize(file_1);

  char* buffer = (char*)malloc(sizeof(char) * file_size);
  char* reverse = (char*)malloc(sizeof(char) * file_size);


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

  printf("%s\n", buffer);

  for (int i = 0; i < sizeof(buffer); i++) {
    reverse[i] = buffer[sizeof(buffer) - i - 1];
  }

  free(buffer);
  free(reverse);

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
