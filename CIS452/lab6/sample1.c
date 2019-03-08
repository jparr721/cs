#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>
#include <sys/types.h>
#include <sys/wait.h>
#include <sys/ipc.h>
#include <sys/shm.h>
#include <sys/stat.h>
#include <sys/sem.h>

#define SIZE 16

int main (int argc, char** argv)
{
    if (argc != 2) {
      perror("Yo, enter number of loops, thanks");
      return EXIT_FAILURE;
    }
    int status;
    long int i, loop, temp, *shmPtr;
    int shmId, sem_id;
    pid_t pid;
    struct sembuf semaphores[2];

   loop = atol(argv[1]);

   if ((shmId = shmget (IPC_PRIVATE, SIZE, IPC_CREAT|S_IRUSR|S_IWUSR)) < 0) {
      perror ("i can't get no..\n");
      exit (1);
   }
   if ((shmPtr = shmat (shmId, 0, 0)) == (void*) -1) {
      perror ("can't attach\n");
      exit (1);
   }

   key_t key;
   key = ftok("Secret", 7);

   if ((sem_id = semget(key, 1, IPC_CREAT | 0666)) == -1) {
      perror("Failed to make the semaphore");
      return EXIT_FAILURE;
   }
   semctl(sem_id, 0, SETVAL, 1);
   semaphores[0].sem_num = 0;
   semaphores[0].sem_op = -1;
   semaphores[0].sem_flg = IPC_NOWAIT;
   semaphores[1].sem_num = 1;
   semaphores[1].sem_op = 1;
   semaphores[1].sem_flg = IPC_NOWAIT;

   shmPtr[0] = 0;
   shmPtr[1] = 1;

   if (!(pid = fork())) {
      for (i=0; i<loop; i++) {
        semop(sem_id, &semaphores[0], 1);
        temp = shmPtr[0];
        shmPtr[0] = shmPtr[1];
        shmPtr[1] = temp;
        semop(sem_id, &semaphores[1], 1);
      }
      if (shmdt (shmPtr) < 0) {
         perror ("just can't let go\n");
         exit (1);
      }
      exit(0);
   }
   else {
      for (i=0; i<loop; i++) {
        semop(sem_id, &semaphores[1], 1);
        temp = shmPtr[1];
        shmPtr[1] = shmPtr[0];
        shmPtr[0] = temp;
        semop(sem_id, &semaphores[0], 1);
      }
   }

   wait (&status);
   printf ("values: %li\t%li\n", shmPtr[0], shmPtr[1]);

   if (shmdt (shmPtr) < 0) {
      perror ("just can't let go\n");
      exit (1);
   }
   if (shmctl (shmId, IPC_RMID, 0) < 0) {
      perror ("can't deallocate\n");
      exit(1);
   }

   return 0;
}
