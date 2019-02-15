#include <iostream>
#include <sys/types.h>
#include <sys/stat.h>
#include <sys/ipc.h>
#include <sys/shm.h>
#include <csignal>

struct data {
  char* input;
  int reader_one;
  int reader_two;
};

char* str;
int interrupt = 0;
struct data* info;

void sig_handler(int) {
  std::cout << "Interrupted" << std::endl;
  interrupt = 1;

  shmdt(str);
}

int runner() {
  struct sigaction action;
  action.sa_handler = &sig_handler;
  action.sa_flags = 0;

  const int BIT_SIZE = 4096;
  int shmid;
  key_t key = ftok("shmfile", 65);

  sigaction(SIGINT, &action, nullptr);

  if ((shmid = shmget(key, BIT_SIZE, IPC_CREAT|S_IRUSR|S_IWUSR)) < 0) {
    std::cerr << "Failed to make shared memory, sorry mate" << std::endl;
    return EXIT_FAILURE;
  }

  if ((info = reinterpret_cast<struct data*>(shmat(shmid, nullptr, 0))) == reinterpret_cast<void*>(-1)) {
    std::cerr << "Failed to attach" << std::endl;
    return EXIT_FAILURE;
  }

  info->reader_one = 1;
  info->reader_two = 1;

  do {
    std::cout << "What do you want to put on the buffer?";
    std::cin >> str;
    std::cout << std::endl;

    std::cout << "Data written to memory..." << std::endl;

  } while (!interrupt);

  return EXIT_SUCCESS;
}

int main() {
  return runner();
}
