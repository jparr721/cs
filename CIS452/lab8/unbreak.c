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
