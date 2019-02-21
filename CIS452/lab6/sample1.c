#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>
#include <sys/types.h>
#include <sys/wait.h>
#include <sys/ipc.h>
#include <sys/shm.h>
#include <sys/stat.h>
#include <semaphore.h>

#define SIZE 16

int main (int argc, char** argv)
{
    int status;
    long int i, loop, temp, *shmPtr;
    int shmId;
    pid_t pid;

    sem_t mutex;

   if (sem_init(&mutex, 1, 1) < 0) {
      perror("Failed to make semaphore");
      return EXIT_FAILURE;
   }

   loop = atol(argv[1]);

   if ((shmId = shmget (IPC_PRIVATE, SIZE, IPC_CREAT|S_IRUSR|S_IWUSR)) < 0) {
      perror ("i can't get no..\n");
      exit (1);
   }
   if ((shmPtr = shmat (shmId, 0, 0)) == (void*) -1) {
      perror ("can't attach\n");
      exit (1);
   }

   shmPtr[0] = 0;
   shmPtr[1] = 1;

   if (!(pid = fork())) {
      for (i=0; i<loop; i++) {
        sem_wait(&mutex);
        printf("Child entered the critical section\n");
        temp = shmPtr[0];
        shmPtr[0] = shmPtr[1];
        shmPtr[1] = temp;
        printf("Child exiting critical section\n");
        sem_post(&mutex);
      }
      if (shmdt (shmPtr) < 0) {
         perror ("just can't let go\n");
         exit (1);
      }
      exit(0);
   }
   else {
      for (i=0; i<loop; i++) {
        sem_wait(&mutex);
        printf("Parent entered critical section\n");
        temp = shmPtr[1];
        shmPtr[1] = shmPtr[0];
        shmPtr[0] = temp;
        printf("Parent exiting critical section\n");
        sem_post(&mutex);
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

   sem_destroy(&mutex);
   return 0;
}
