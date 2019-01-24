#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>
#include <sys/types.h>
#include <string.h>
#include <signal.h>
#include <time.h>
#include <wait.h>

#define READ 0
#define WRITE 1
#define MAX 1024

void child_process();
void sig_handler(int signum);

void child_process(pid_t pid) {
  srand(time(NULL));
  // Our random 5 second wait time
  for (;;) {
    int random_time = rand() % 6;
    int choice = rand() % 2;
    printf("waiting...");

    sleep(random_time);
    if (choice == 1)
      kill(pid, SIGUSR1);
    else
      kill(pid, SIGUSR2);
  }
}

void sig_handler(int signum) {
  if (signum == SIGUSR1) {
    printf("Received a SIGUSR1 signal\n");
  } else if (signum == SIGUSR2) {
    printf("Received a SIGUSR2 signal\n");
  } else if (signum == SIGINT) {
    printf("Oh we got a wise guy here eh? Shutting it down!\n");
    exit(EXIT_SUCCESS);
  } else {
    printf("What do dat one do");
  }
}

int main(int argc, char** argv) {
  pid_t pid;

  struct sigaction sa;
  sa.sa_handler = sig_handler;
  sigemptyset(&sa.sa_mask);
  sa.sa_flags = SA_RESTART;

  sigaction(SIGINT, &sa, NULL);
  sigaction(SIGUSR1, &sa, NULL);
  sigaction(SIGUSR2, &sa, NULL);
  /** spawn child **/

  if ((pid = fork()) < 0) {
    perror("Fork machine broke");
    return EXIT_FAILURE;
  } else if (pid == 0) {
    child_process(pid);
  } else {
    printf("Spawned child pid# %d\n", pid);
    wait(&pid);
  }
}
