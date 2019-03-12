#include <stdlib.h>
#include <unistd.h>
#include <stdio.h>
#include <sys/types.h>
#include <sys/wait.h>
#include <sys/ipc.h>
#include <sys/stat.h>
#include <time.h>
#include "lib.h"

#define LOCK -1
#define UNLOCK 1
#define NO_CHANGE -4
#define CHANGE 4

void wow() {
  printf("WOWZERS");
}

sem_t* sem_t_create(key_t key, struct sembuf* buf, int flags, int nsems) {
  sem_t sem;

  if ((sem.sem_id = semget(key, nsems, flags)) < 0) {
    perror("Error, failed to initialize semaphore!");
  }

  sem.buf = *buf;

  return &sem;
}

void sem_t_initialize(sem_t* sem, int sem_op, int sem_flg) {
  sem->buf.sem_num = sem->sem_id;
  sem->buf.sem_op = sem_op;
  sem->buf.sem_flg = sem_flg;
}

int sem_t_timed_wait(sem_t* sem, const struct timespec* timeout) {
  if (sizeof(sem->buf) == 0) {
    goto not_initialized;
  }
  /* The sem buffer must be initialized here*/
  if (semtimedop(sem->sem_id, sem->buf, 1, timeout) < 0) {
    return -1;
  }

  return 0;

not_initialized:
  perror("semaphore not initialized!");
  return EXIT_FAILURE;
}

void sem_t_lock(sem_t* sem) {
  sem->waiting = 1;
  sem->buf.sem_flg = LOCK;
}

void sem_t_unlock(sem_t* sem) {
  sem->waiting = 0;
  sem->buf.sem_flg = UNLOCK;
}

int sem_t_signal(sem_t* sem) {
  /* A process can signal if the semaphore is waiting */
  if (sem->waiting) {
    sem_t_unlock(sem);
    return CHANGE;
  }

  /* Otherwise, do nothing */
  return NO_CHANGE;
}

void destroy(sem_t* sem) {
  semctl(sem->sem_id, 0, IPC_RMID);
}
