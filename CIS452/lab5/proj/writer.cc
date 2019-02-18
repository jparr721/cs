#include <csignal>
#include <iostream>
#include <sys/types.h>
#include <sys/stat.h>
#include <sys/ipc.h>
#include <sys/shm.h>
#include <string>

int shmid;
char* info;
int interrupt = 0;

void sig_handler(int) {
  std::cout << "Interrupted" << std::endl;
  interrupt = 1;

  shmdt(info);

  if (shmctl(shmid, IPC_RMID, nullptr) < 0) {
    std::cerr << "What??? We can't deallocate?!?! RUN, RUN NOW!!!" << std::endl;
    exit(EXIT_FAILURE);
  }
}

int runner() {
  struct sigaction action;
  action.sa_handler = &sig_handler;
  action.sa_flags = 0;

  const int BIT_SIZE = 4096;
  key_t key = ftok("shmfile", 65);

  sigaction(SIGINT, &action, nullptr);

  if ((shmid = shmget(key, BIT_SIZE, IPC_CREAT|S_IRUSR|S_IWUSR)) < 0) {
    std::cerr << "Failed to make shared memory, sorry mate" << std::endl;
    return EXIT_FAILURE;
  }

  if ((info = reinterpret_cast<char*>(shmat(shmid, nullptr, 0))) == reinterpret_cast<void*>(-1)) {
    std::cerr << "Failed to attach" << std::endl;
    return EXIT_FAILURE;
  }

  for (;;) {
    std::string input;
    if (*info == '#') {
      std::cout << "Whatcha wanna say? ";
      std::getline(std::cin, input);
      info = const_cast<char*>(input.c_str());
      std::cout << std::endl;
    }
  }


  return EXIT_SUCCESS;
}

int main() {
  return runner();
}
