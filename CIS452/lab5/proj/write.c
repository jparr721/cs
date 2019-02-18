#include <signal.h>
#include <stdio.h>
#include <stdlib.h>
#include <sys/ipc.h>
#include <sys/shm.h>
#include <sys/stat.h>
#include <sys/types.h>
#define BIT_SIZE 4096

void sig_handler(int);

int shmid;
char* shm_ptr;
key_t key;

int main() {
  signal(SIGINT, sig_handler);

  key = ftok("f", 3);

  if ((shmid = shmget(key, BIT_SIZE, 0666|IPC_CREAT)) < 0) {
    perror("Failed to make shared memory");
    exit(EXIT_FAILURE);
  }

  if ((shm_ptr = shmat(shmid, 0, 0)) == (void*) -1) {
    perror("Failed to attach");
    exit(EXIT_FAILURE);
  }

  for (;;) {
    if (*shm_ptr == '#') {
      printf("Whatcha wanna say? ");
      scanf("%s", shm_ptr);
      printf("\n");
    }
  }

  return EXIT_SUCCESS;
}

void sig_handler(int i) {
  printf("Interrupt called");

  if (i == SIGINT) {
    if (shmdt(shm_ptr) < 0) {
      perror("Failed to let go\n");
    }
  }

  exit(EXIT_SUCCESS);
}

