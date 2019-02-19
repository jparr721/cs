#include <stdio.h>
#include <stdlib.h>
#include <sys/ipc.h>
#include <sys/shm.h>
#include <sys/stat.h>
#include <signal.h>
#include <string.h>
#define BIT_SIZE 4096

void sig_handler(int);

int shmid;
char* shm_ptr;
key_t key;

int main() {
  signal(SIGINT, sig_handler);

  key = ftok("f", 3);

  if ((shmid = shmget (key, BIT_SIZE, IPC_CREAT|S_IRUSR|S_IWUSR)) < 0) {
    perror("Failed to make shared memory");
    return EXIT_FAILURE;
  }

  if ((shm_ptr = shmat(shmid, 0, 0)) == (void*) -1) {
    perror("Can't attach");
    return EXIT_FAILURE;
  }

  for (;;) {
    if (*shm_ptr != '#') {
      printf("%s\n", shm_ptr);
      *shm_ptr = '#';
    }
  }

  return EXIT_SUCCESS;
}

void sig_handler(int i) {
  printf("Interrupted");

  if (shmctl(shmid, IPC_RMID, NULL) < 0) {
    perror("What??? We can't deallocate?!?! RUN, RUN NOW!!!");
    exit(EXIT_FAILURE);
  }
  exit(EXIT_SUCCESS);
}
