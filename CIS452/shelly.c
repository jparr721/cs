#include <stdio.h>
#include <stdlib.h>
#include <sys/types.h>
#include <sys/wait.h>
#include <unistd.h>
#include <string.h>
#include <sys/time.h>
#include <sys/resource.h>

int main(int argc, char** argv) {
  char instruction[4096];
  struct rusage resources_used;

  for (;;) {
    printf("Shellz0r> ");
    fgets(instruction, 4096, stdin);
  }
}
