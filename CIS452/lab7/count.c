#include <stdio.h>
#include <stdlib.h>
#include <sys/ipc.h>
#include <sys/sem.h>


int main ()
{
	long int count = 1;
	//Initialize Semaphore here
	int sem = 0;
	key_t key = ftok("!", 3);
	
	if ((sem = semget(key, 1, IPC_CREAT | 0666)) == -1) {
		perror("broke");
		return EXIT_FAILURE;
	}

	for(;;) {
		//Increment semaphore count here		
		count = semctl(sem, 0, GETVAL);

		count++;

		printf("Count: %ld\n", count);
	}

}
