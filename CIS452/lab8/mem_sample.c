#include <stdio.h>
#include <stdlib.h>

// Unitialized variable stored in uninitialized data segment
int global;

// Initialized data in the initiailzed segment
int global2 = 10;

int the_other_value(int val) {
  int val_array[6] = {1, 2, 3, 4, 5, val*97657};

  return val_array[5];
}

int main(int argc, char** argv) {
  // Uninitialized variable stored in uninit segment
  static int i;
  // Stored in initialized data segment
  static int j = 100;
  printf("%p\n", &j);

  // Dynamically made data which resides on the heap
  char* dyno = malloc(16);
  printf("%p\n", &dyno);

  // Statically made data which resides on the stack
  int stati[2] = {1, 2};
  printf("stati\n");
  printf("%p\n", &stati);

  // Memory address of dyno
  char** addr = &dyno;
  printf("%p\n", addr);

  printf("global\n");
  printf("%p\n", &global2);
  printf("%p\n", &global);

  printf("func\n");
  printf("%p\n", &the_other_value);

  int not_an_r_value = the_other_value(87);
  printf("func_output\n");
  printf("%p\n", &not_an_r_value);

  // Free the dynamic memory
  free(dyno);

  return 0;
}
