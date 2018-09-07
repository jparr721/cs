#ifndef FILE_UTILS_H
#define FILE_UTILS_H

#include <stdlib.h>

int read_file(const char* filename, char **buffer);
int write_file (const char* filename, char *buffer, size_t size);

#endif
