#include <iostream>
#include <sys/ipc.h>
#include <sys/shm.h>
#include <sys/stat.h>
#include <csignal>
#include <cstring>

void sig_hanlder(int);

const int BIT_SIZE = 4096;
int shmid;
char* shmPtr;
char* str;
int interrupt = 0;

void sig_handler(int) {
  std::cout << "Interrupted" << std::endl;
  interrupt = 1;

  shmdt(str);

  if (shmctl(shmid, IPC_RMID, nullptr) < 0) {
    std::cerr << "What??? We can't deallocate?!?! RUN, RUN NOW!!!" << std::endl;
    exit(EXIT_FAILURE);
  }

}

int main() {
  struct sigaction action;
  action.sa_handler = &sig_handler;
  action.sa_flags = 0;

  key_t key = ftok("shmfile", 65);

  sigaction(SIGINT, &action, nullptr);

  if ((shmid = shmget(key, BIT_SIZE, IPC_CREAT)) < 0) {
    std::cerr << "Failed to make shared memory, sorry mate" << std::endl;
    return EXIT_FAILURE;
  }

   if ((shmPtr = reinterpret_cast<char*>(shmat(shmid, nullptr, 0))) == (void*) -1) {
      std::cerr<< "can't attach" << std::endl;
      return EXIT_FAILURE;
   }

  for (;;) {
    if(*shmPtr != '#') {
      std::cout << shmPtr << std::endl;
      *shmPtr = '#';
    }
  }

  return EXIT_SUCCESS;
}
