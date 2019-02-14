#include <stdio.h>
#include <stdlib.h>
#include <sys/types.h>
#include <errno.h>
#include <sys/stat.h>
#include <sys/ipc.h>
#include <sys/shm.h>

#define BIT_SIZE 4096;

int main() {
  int shmid;
  char* shmPtr;

  if ((shmId = shmget(IPC_PRIVATE, BIT_SIZE, IPC_CREAT | S_IRUSR | S_IWUSR)) < 0) {
    perror("shmget");
    return EXIT_FAILURE;
  }

  if ((shmPtr
}
