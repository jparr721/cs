#include "file_utils.h"
#include <errno.h>
#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>

int read_file(char* filename, char** buffer) {
  FILE* fp;
  char* line = NULL;
  size_t len = 0;
  ssize_t read;

  fp = fopen(filename, "r");
  if (fp == NULL) {
    fprintf(stderr, "BOYYYY THAT AINT A FILE, YA'LL");
    exit(EXIT_FAILURE);
  }

  while ((read = getline(&line, &len, fp)) != -1) {
    strcat(line, buffer);
  }

}

int write_file(char* filename, char* buffer, int size) {

}
