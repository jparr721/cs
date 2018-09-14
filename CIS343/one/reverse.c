#include "file_utils.h"
#include <errno.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/stat.h>
#include <sys/types.h>
#include <unistd.h>

/**
 * This helper function gets the size of a
 * file and returns it in off_t data type.
 *
 * @param {char*} filename - The name of the file to read
 */
off_t fsize(const char* filename) {
  FILE * fp;
  fp = fopen(filename, "r");
  fseek(fp, 0, SEEK_END);

  // get the size in bytes
  long bytes = ftell(fp);
  rewind(fp);
  fclose(fp);
  return bytes;
}

/**
 * This function reverses the file.
 *
 * @param {char*} - file_1
 * @param {char*} - file_2
 */
int reverse(const char* file_1, const char* file_2) {
  // The size of the file to be worked with
  // This is for proper malloc-ing
  off_t file_size = fsize(file_1);

  // Malloc a buffer to the size if the file
  char* buffer = (char*)malloc(sizeof(char) * file_size);
  char* reverse = (char*)malloc(sizeof(char) * file_size);

  if (buffer == 0) {
    fprintf(stderr, "BOYYYY WE HAD AN ERROR ALLOCATING THIS MEMORY");
    return errno;
  }

  // Reads the result of the read and write file functions
  ssize_t read = 0;

  if ((read = read_file(file_1, &buffer)) != 0) {
    fprintf(stderr, "BOYYYY WE CAN'T READ INTO THIS BUFFER");
    return errno;
  }

  for (int i = file_size; i > 0; --i) {
    reverse[i] = buffer[file_size - i - 1];
  }

  if ((read = write_file(file_2, reverse, file_size) != 0)) {
    fprintf(stderr, "BOYYY WE CAN'T WRITE TO THIS BUFFER");
    return errno;
  }

  // Free the memory
  free(buffer);
  free(reverse);

  return EXIT_SUCCESS;
}


int main(int argc, char** argv) {
  if (argc != 3) {
    printf("Invalid number of arguments provided, exiting");
    return EXIT_FAILURE;
  }

  // Read in the file names
  char* file_1 = argv[1];
  char* file_2 = argv[2];

  // Call the functon and reverse into the buffers
  reverse(file_1, file_2);

  return EXIT_SUCCESS;
}
