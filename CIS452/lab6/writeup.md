# Lab 6
By Jarred Parr and Alexander Fountain

1. The program assigns the `shmPtr` values at the two indexes with the values 0 and 1. From there, the parent and also the child process begin to loop for some arbitrarily assigned amount of time. Each parent and child begins to swap the values at 0 and 1 back and forth. The program attempts to demonstrate how there can be collisions in the read and write. Since the parent and child are both using the same piece of shared memory they both manipulate the same value at the same time. The parent is explicitly responsible for taking the value in index 1 of the `shmPtr` value and swapping it with the value in index 0 of the `shmPtr` value. Inside of the child, it is responsible for taking the value in index 0 and swapping it with the value in index 1.

2. The programs expected output, if properly managed for critical sections, would be that for an odd number of runs the values would be opposite of how they were in the beginning (i.e. `shmPtr[0] = 1` and `shmPtr[1] = 0`). For an even number of runs the values would be as they were in the beginning.

3. As the loop values increase, we can observe that there exists a race condition issue from how the shared memory value is read. The value in shared memory may not have time to have been updated before the next read and as a result leads to both of the number ending up as the same value upon completion.

4. As was stated in problem 3, there is an issue in read and write times for the shared memory. With both the parent and child process attemtping to gain access to the piece of shared memory between them, at some point there is a mismatch about what value is stored in memory which leads both of them to take on the same value. There is no blocking operation for reading and writing from the shared memory space so the two processes clobber over one-another.
