#ifndef LIB_H_
#define LIB_H_

#include <sys/sem.h>

typedef struct {
  struct sembuf buf;
  int sem_id;
  int waiting;
} sem_t;

sem_t* sem_t_create(key_t key, struct sembuf* buf, int flags, int nsems);
void sem_t_initialize(sem_t* sem, int sem_t_op, int sem_t_flg);
int sem_t_timed_wait(sem_t* sem, const struct timespec* timeout);
void sem_t_lock(sem_t* sem);
void sem_t_unlock(sem_t* sem);
int sem_t_signal(sem_t* sem);
void sem_t_destroy(sem_t* sem);

#endif // LIB_H_
