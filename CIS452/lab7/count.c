#include <stdio.h>
#include <stdlib.h>
#include <sys/ipc.h>
#include <sys/sem.h>


int main ()
{
	long int count = 1;
	//Initialize Semaphore here
	int sem = semget(IPC_PRIVATE, 1, 00600);

	for(;;) {
		//Increment semaphore count here		

		if(semctl(sem, 0, SETVAL, count) == -1){
			perror("Darn\n");
			exit(1);
		}

		count++;

		printf("Count: %ld\n", count);
	}

}
