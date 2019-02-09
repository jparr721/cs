#include <pthread.h>
#include <iostream>
#include <cstdlib>
#include <cerrno>
#include <cstring>
#include <unistd.h>

using namespace std;

void* do_greeting2 (void *arg);
// arguments :  arg is an untyped pointer
// returns :       a pointer to whatever was passed in to arg
// side effects:  prints a greeting

int main()
{
    pthread_t thread1, thread2;  // thread ID's
    void* result1, *result2;     // return values
    int status;

// create and start two threads; both executing the "do_greeting2" function
// pass the threads a pointer to NULL as their argument
    if ((status = pthread_create (&thread1, NULL,  do_greeting2, NULL)) != 0) {
        cerr << "thread create error: " << strerror(status) << endl;
        exit (1);
    }
    if ((status = pthread_create (&thread2, NULL,  do_greeting2, NULL)) != 0) {
        cerr << "thread create error: " << strerror(status) << endl;
        exit (1);
    }

// join with the threads (wait for them to terminate);  get their return vals
    if ((status = pthread_join (thread1, &result1)) != 0) {
        cerr << "join error: " << strerror(status) << endl;
        exit (1);
    }
    if ((status = pthread_join (thread2, &result2)) != 0) {
        cerr << "join error: " << strerror(status) << endl;
        exit (1);
    }

// threads return what they were passed (i.e. NULL)
    if (result1 != NULL || result2 != NULL) {
        cerr << "bad result"  << endl;
        exit (1);
    }
    return 0;
}

void* do_greeting2 (void *arg) {
    int val = rand() % 2;

// print out message based on val
    for (int loop = 0;  loop < 10;  loop++) {
        sleep(1);
        if (!val)
            cout << "Hello ";
        else
            cout << "World\n ";
    }
    return arg;
}
