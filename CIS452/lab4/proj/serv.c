#include <stdio.h>
#include <errno.h>
#include <stdlib.h>
#include <signal.h>
#include <pthread.h>
#include <unistd.h>
#include <string.h>

size_t interrupt = 0;
size_t requests = 0;
size_t runtime = 0;
pthread_mutex_t req_mutex = PTHREAD_MUTEX_INITIALIZER;
pthread_mutex_t time_mutex = PTHREAD_MUTEX_INITIALIZER;

void sig_handler(int num) {
  printf("Interrupt received");

  interrupt = 1;

  // Unlock our mutextes
  pthread_mutex_unlock(&req_mutex);
  pthread_mutex_unlock(&time_mutex);
}

void* worker(void* arg) {
    pthread_mutex_lock(&req_mutex);
    pthread_mutex_lock(&time_mutex);

    char* filename = (char*) arg;

    size_t sleep_duration;

    if ((rand() % 10) < 8) {
      sleep_duration = 1;
    } else {
      sleep_duration = (rand() % 3) + 7;
    }

    sleep(sleep_duration);

    printf("\nFound the file: %s\n", filename);

    runtime += sleep_duration;
    requests++;

    pthread_mutex_unlock(&req_mutex);
    pthread_mutex_unlock(&time_mutex);

    pthread_exit(NULL);
}

int server() {
  struct sigaction action;
  action.sa_handler = &sig_handler;
  action.sa_flags = 0;

  pthread_t thread;
  int status = 0;
  char input[1024];

  if (pthread_mutex_init(&req_mutex, NULL) != 0) {
    perror("Failed to make requests mutex");
    return EXIT_FAILURE;
  }

  if (pthread_mutex_init(&time_mutex, NULL) != 0) {
    perror("Failed to make time mutex");
    return EXIT_FAILURE;
  }

  if (sigaction(SIGINT, &action, NULL) < 0) {
    perror("Failed to catch SIGINT");
    return EXIT_FAILURE;
  }

  while (!interrupt) {
    printf("What file?: ");
    fgets(input, 1024, stdin);

    input[strcspn(input, "\n")] = 0;

    if (strlen(input) == 0) {
      printf("\nOh, we got a wise guy eh? Enter a file name like a real man!\n");
      continue;
    }

    if ((status = pthread_create(&thread, NULL, worker, input)) != 0) {
      perror("Thread create error");
      return EXIT_FAILURE;
    }
  }


  float proc = (float)runtime / requests;
  printf("Processing time %ld\n", runtime);
  printf("Average proc time %f\n", proc);
  printf("Total requests %ld\n", requests);

  pthread_mutex_destroy(&req_mutex);
  pthread_mutex_destroy(&time_mutex);
  return EXIT_SUCCESS;
}

int main() {
  server();
}
