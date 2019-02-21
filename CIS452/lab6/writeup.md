# Lab 6
By Jarred Parr and Alexander Fountain

1. The program assigns the `shmPtr` values at the two indexes with the values 0 and 1. From there, the parent and also the child process begin to loop for some arbitrarily assigned amount of time. Each parent and child begins to swap the values at 0 and 1 back and forth. The program attempts to demonstrate how there can be collisions in the read and write. Since the parent and child are both using the same piece of shared memory they both manipulate the same value at the same time. The parent is explicitly responsible for taking the value in index 1 of the `shmPtr` value and swapping it with the value in index 0 of the `shmPtr` value. Inside of the child, it is responsible for taking the value in index 0 and swapping it with the value in index 1.

2. The programs expected output, if properly managed for critical sections, would be that for an odd number of runs the values would be opposite of how they were in the beginning (i.e. `shmPtr[0] = 1` and `shmPtr[1] = 0`). For an even number of runs the values would be as they were in the beginning.

3. As the loop values increase, we can observe that there exists a race condition issue from how the shared memory value is read. The value in shared memory may not have time to have been updated before the next read and as a result leads to both of the number ending up as the same value upon completion.

4. As was stated in problem 3, there is an issue in read and write times for the shared memory. With both the parent and child process attemtping to gain access to the piece of shared memory between them, at some point there is a mismatch about what value is stored in memory which leads both of them to take on the same value. There is no blocking operation for reading and writing from the shared memory space so the two processes clobber over one-another.

5. The 3 options for `struct sembuf` are `sem_num`, `sem_op`, and `sem_flg`. These 3 options are specified as the following:
  - `sem_num` - The semaphore number. This specifies the semaphore that this particular buffer is associated with.
  - `sem_op` - The semaphore operation. This specifies the operation that the semaphore will be performing
  - `sem_flg` - The semaphore flag. You can initialize some of these values with certain flags that may affect creation constraints or runtime aspects within the semaphore. Flags faciliate that feature of the semop.

  6. `SEM_UNDO` undoes the specified operation after the program running the semaphore has terminated. This allows the kernel to back out of whatever the operation is running if the program exits unexpectedly. This gives the ability of cleaning up shared namespaces and memory, at the cost of all of the calls needing to now be able to access things in the kernel space if you're running it in that level of privilge. This is most useful in situations like this, because when running in the kernel space, a user could send a SIGKILL to a process and it might not effectively clean up the semaphore since it was not explicitly flagged. As a result, in rare cases, this causes the program to lock completely. In the case of critical programs, this could be a big no-no for system stability.
