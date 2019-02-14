#include <iostream>
#include <sys/ipc.h>
#include <sys/shm.h>
#include <sys/stat.h>
#include <csignal>
#include <cstring>

const int BIT_SIZE = 4096;
int shmid;
char* str;
int interrupt = 0;

void sig_handler(int) {
  std::cout << "Interrupted" << std::endl;
  interrupt = 1;

  shmdt(str);

  shmctl(shmid, IPC_RMID, nullptr);

}

int main() {
  struct sigaction action;
  action.sa_handler = &sig_handler;
  action.sa_flags = 0;

  key_t key = ftok("shmfile", 65);

  sigaction(SIGINT, &action, nullptr);

  if ((shmid = shmget(key, BIT_SIZE, 0666|IPC_CREAT)) < 0) {
    std::cerr << "Failed to make shared memory, sorry mate" << std::endl;
    return EXIT_FAILURE;
  }

  char* last;
  while (!interrupt) {
    /* if ( == reinterpret_cast<void*>(-1)) { */
    /* } */
    str = reinterpret_cast<char*>(shmat(shmid, nullptr, 0));

    if (strcmp(last, str) != 0) {
      std::cout << "Data from shared memory: " << str << std::endl;
      last = str;
    } else {
      continue;
    }
  }

  return EXIT_SUCCESS;
}
