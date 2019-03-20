#include <stdio.h>
#include <stdlib.h>

// Unitialized variable stored in uninitialized data segment
int global;

// Initialized data in the initiailzed segment
int global2 = 10;

int main(int argc, char** argv) {
  // Uninitialized variable stored in uninit segment
  static int i;
  // Stored in initialized data segment
  static int j = 100;

  // Dynamically made data which resides on the heap
  char* dyno = malloc(16);

  // Statically made data which resides on the stack
  int stati[2] = {1, 2};

  // Memory address of dyno
  char** addr = &dyno;

  // Free the dynamic memory
  free(dyno);

  return 0;
}
