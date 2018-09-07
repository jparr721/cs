#include "file_utils.h"
#include <errno.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>

int read_file(const char* filename, char** buffer) {
  FILE* fp;

  fp = fopen(filename, "r+e");

  if (fp == NULL) {
    fprintf(stderr, "BOYYYY THAT AINT A FILE, YA'LL");

    return errno;
  }

  char c;

  while ((c = getc(fp)) != EOF) {
    printf("%c", c);
    strncpy(*buffer, &c, sizeof(buffer));
  }

  fclose(fp);

  return 0;
}

int write_file(const char* filename, char* buffer, size_t size) {
  return 0;
}
