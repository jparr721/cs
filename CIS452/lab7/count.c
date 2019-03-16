#include <stdio.h>
#include <stdlib.h>
#include <semaphore.h>

int pshared;
int ret;
int value;
sem_t sem;

int sem_post(sem_t *sem);

long int count = 0;

int sem_init(sem_t *sem, int pshared, unsigned int value); 


int main ()
{
	pshared = 0;
	value = 1;

	for(;;) {
		ret = sem_post(&sem);
		count++;

		printf("Count: %ld\n", count);
	}

}
