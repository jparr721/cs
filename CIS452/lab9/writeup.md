1. Working off of eos05.
  - There is currently 16283404kb of physical memory on the system.
  - There is currently 13183648kb of free memory on the system.

2.

   a.  Estimated memory demand: ~15,564kb. This was calculated after recording the available memory before code execution, during, and after via the `free -h` command. Then the during value was substracted from the before value.

   b. The approximate memory demand is: ~15632 when recorded with `vmstat 1 15` and once again, the before and after values were compared. Using free during the runtime of the program was what allowed me to have similar results for part a and b.

   c. The observed difference, however slight it may be, appears to be the result of the accuracy of vmstate being able to run continuously and check the memory at a reliable timestep. Because of this bit of error in manually reading calculations and watching the `free` command run.

3.

   a. The amount of free memory begins to drastically reduce as the malloc slowly takes up more and more space in main memory on the device.
   b. Other fields that changes are the amount of virtual memory. Virtual memory is used in this case because the amount of RAM currently being used is not enough to maintain the system. As a result, it must got to the hard disk for more space. This also caused a significant slowdown on the system.
   c. The amount of memory used as cache surprisingly went down. so and si also went way up to show how much memory was swapping in and out of disk as a result of the use of virtual memory. Pretty much everything from blocks, to interrrupts, to even time stolen from the virtual machine all saw a sharp spike as a result of the increase in load on the system imposed by the program.

4.
    a. The page size is 4096 bytes.  It takes 9.094 seconds to run.
    b. 
