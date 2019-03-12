# Lab 6
By Jarred Parr and Alexander Fountain

1. The program uses the child to swap two values in shared memory and then the parent swaps those values back.

2. The output, in an ideal case, will always be `values: 0 1`

3. There is a race condition present where the value may be double read and can cause the output to be `0 0` or `1 1` as a result.

4. As was stated in problem 3, there is an issue in read and write times for the shared memory. Because of the context switch in an undefined location in the loop, we can see that `temp` is assingned a value, say 0, then the context switches and the other loop replaces `temp` with the value, say 1, and then swaps the pointers. Because this loop finishes, the program context switches back to the first process, but temp now has a value of 1 instead of zero, this causes us to see that both values would end up at 1 in each index of the shared memory.

5. The 3 options for `struct sembuf` are `sem_num`, `sem_op`, and `sem_flg`. These 3 options are specified as the following:
  - `sem_num` - The semaphore number. This specifies the semaphore that the operation will be performed on.
  - `sem_op` - The semaphore operation. This specifies the operation that the semaphore will be performing.
  - `sem_flg` - The semaphore flag. You can initialize some of these values with certain flags that may affect creation constraints or runtime aspects within the semaphore. Flags faciliate that feature of the semop. The options `IPC_NOWAIT` and `SEM_UNDO` allow the program to run and undo the opertation when the process is done respectively. These options allow the field to control how semaphores work without needing to be extremely strict.

6. `SEM_UNDO` undoes the specified operation after the program running has terminated. This allows the kernel to back out of whatever the operation is running if the program exits unexpectedly. This gives the ability of cleaning up shared namespaces and memory, at the cost of all of the calls needing to now be able to access things in the kernel space if you're running it in that level of privilge. This is most useful in situations like this, because when running in the kernel space, a user could send a SIGKILL to a process and it might not effectively clean up the semaphore since it was not explicitly flagged. As a result, in rare cases, this causes the program to lock completely. In the case of critical programs, this could be a big no-no for system stability.

