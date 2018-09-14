#ifndef FILE_UTILS_H
#define FILE_UTILS_H

#include <stdlib.h>

/** Reads a file into a bufer by name **/
int read_file(const char* filename, char **buffer);

/** Writes a file from a buffer by name **/
int write_file (const char* filename, char *buffer, size_t size);

#endif
