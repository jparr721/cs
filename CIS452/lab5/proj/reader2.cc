#include <iostream>
#include <sys/ipc.h>
#include <sys/shm.h>
#include <sys/stat.h>
#include <csignal>
#include <cstring>

void sig_hanlder(int);

struct data {
  char* input;
  int reader_one;
  int reader_two;
};

const int BIT_SIZE = 4096;
int shmid;
char* str;
int interrupt = 0;
struct data* info;

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

  if ((shmid = shmget(key, BIT_SIZE, IPC_CREAT)) < 0) {
    std::cerr << "Failed to make shared memory, sorry mate" << std::endl;
    return EXIT_FAILURE;
  }

  if ((info = reinterpret_cast<struct data*>(shmat(shmid, nullptr, 0))) == reinterpret_cast<void*>(-1)) {
    std::cerr << "Failed to attach" << std::endl;
    return EXIT_FAILURE;
  }

  char* last;
  do {
    while(info->reader_two);
    std::cout << info->input << std::endl;

    info->reader_two = 1;
  } while(!interrupt);

  return EXIT_SUCCESS;
}
