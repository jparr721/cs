# Lab 8
Jarred Parr Alexander Fountain

1. This is a buffer overrun issue. Because the proper memory was not properly allocated, the length of the string overflows into other parts of memory that may not have been originally designated to the program. As a result, this causes problems. On a more high-scale project, this could be catastrophic and open the code to injection issues that would potentially cause harm to the target computer or exploit the software as a whole. The error in this file begins with the malloc of size 16. The problem here is that when you put in `notarealusername` the length is too large, and on the `scanf` line you see the data get placed into that memory incorrectly as a result.

**Corrected (Elegant) Code**
```C
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define SIZE 16

char* infinite_length_alloc(FILE* input_stream, size_t input_size) {
  char* new_string;
  int ch;
  size_t len = 0;
  new_string = malloc(input_size);

  while ((ch = fgetc(input_stream)) != EOF && ch != '\n') {
    new_string[len++] = ch;
    if (len == input_size) {
      // Double the size of the input string
      new_string = realloc(new_string, input_size * 2);
    }
  }

  // Add our nullbyte
  new_string[len++] = '\0';

  return new_string;
}

int main() {
    char *data1;

    data1 = malloc (SIZE);
    printf ("Please input username: ");
    data1 = infinite_length_alloc(stdin, SIZE);
    printf ("you entered: %s\n", data1);
    free (data1);
    return 0;
}
```
