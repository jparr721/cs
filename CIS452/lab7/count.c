#include <stdio.h>
#include <stdlib.h>
#include <semaphore.h>

int ret;
sem_t sem_name;

long int count = 1;

int sem_init(sem_t *sem, int pshared, unsigned int value); 


int main ()
{
	for(;;) {
		sem_post(s&sem_name);
		count++;

		printf("Count: %ld\n", count);
	}

}