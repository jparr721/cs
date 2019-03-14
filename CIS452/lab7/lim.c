#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>
#include <sys/sysinfo.h>

int main(int argc, char** argv) {
  struct sysinfo si;

  int ret = 0;
  if ((ret = sysinfo(&si)) < 0) {
    perror("wtf");
    return -1;
  }

  printf("max shared mem: %lu", si.sharedram);

  return 0;
}
