#include <pthread.h>
#include <stdio.h>
#include <errno.h>
#include <unistd.h>
#include <stdlib.h>
#include <string.h>

void* do_greeting3 (void* arg);
// arguments :  arg is an untyped pointer pointing to a character
// returns :       a pointer to NULL
// side effects:  prints a greeting

// global (shared and specific) data
int sharedData = 5; char val[2] = {'a','b'};

int main()
{
    pthread_t thread1, thread2;
    void *result1, *result2;
    int status;

// create and start two threads executing the "do_greeting3" function
// pass each thread a pointer to its respective argument
    if ((status = pthread_create (&thread1, NULL,  do_greeting3, &val[0])) != 0) {
        fprintf (stderr, "thread create error %d: %s\n", status, strerror(status));
        exit (1);
    }
    if ((status = pthread_create (&thread2, NULL,  do_greeting3, &val[1])) != 0) {
        fprintf (stderr, "thread create error %d: %s\n", status, strerror(status));
        exit (1);
    }
    printf ("Parent sees %d\n", sharedData);
    sharedData++;

// join with the threads (wait for them to terminate);  get their return vals
    if ((status = pthread_join (thread1, &result1)) != 0) {
        fprintf (stderr, "join error %d: %s\n", status, strerror(status));
        exit (1);
    }
    if ((status = pthread_join (thread2, &result2)) != 0) {
        fprintf (stderr, "join error %d: %s\n", status, strerror(status));
        exit (1);
    }
    printf ("Parent sees %d\n", sharedData);
    return 0;
}

void* do_greeting3 (void* arg)
{
// note the cast of the void pointer to the desired data type
    char val_ptr = (char ) arg;

// print out a message
    printf ("Child receiving %c initially sees %d\n", val_ptr, sharedData);
    sleep(1);
    sharedData++;
    printf ("Child receiving %c now sees %d\n", val_ptr, sharedData);
    return NULL;
}
