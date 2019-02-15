# Lab 5
Jarred Parr, Alexander Fountain

1. The program first gets the shared memory id by allocating 4 bytes of space. Then, it creates a pointer by attaching to the newly created shared memory by id. Afterwards, it prints out the shared memory pointer at its originally allocated location and then the location plus an additional 4 bytes (the size of `FOO`). The additional 4 bytes point to the end location of the memory in this allocation.

2. The shmget function call takes 3 aguments, the key, the size, and the flag. The key keeps track of the memory location of the data and points to it. This can be shared to different variables. The size argument takes the size in bytes for the memory segment and allocates for it. The flags do different things. In the context of the code segment provided, a private flag means it won't be able to have its previously shared memory segment available. The `IPC_CREAT` will create a new memory segment. `S_IRUSR` specifies the access level permission of this shared data, same with `S_IWUSR`.

3. The shmctl function performs commands onto the existing shared memory allocation defined by shmget. Two examples of this would be the `IPC_RMID` like we see in the example program. This command will specify that we are going to remove the shared memory identifier fro the system and deallocate the shared memory. Another use is the `IPC_INFO` flag. This retuns information about the system wide memory limits into a struct. With this we can determine shared memory limits and parameters, this gives the programmer more knowledge when it comes to finally performing the allocation, or just working with shared memory in general.

4.  When running sample1part4.c we can see that the shared memory is 4096 bytes.  
