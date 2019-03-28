1. Working off of eos05.
  - There is currently 16283404kb of physical memory on the system.
  - There is currently 13183648kb of free memory on the system.

2.

   a.  Estimated memory demand: ~15,564kb. This was calculated after recording the available memory before code execution, during, and after via the `free -h` command. Then the during value was substracted from the before value.

   b. The approximate memory demand is: ~15632 when recorded with `vmstat 1 15` and once again, the before and after values were compared. Using free during the runtime of the program was what allowed me to have similar results for part a and b.

   c. The observed difference, however slight it may be, appears to be the result of the accuracy of vmstate being able to run continuously and check the memory at a reliable timestep. Because of this bit of error in manually reading calculations and watching the `free` command run.

3.

   a. The `COEFFICIENT` parameter of `64` was chosen by manually calculating the memory overhead of using the system given the amount of free memory available. However, even though this coefficient represented the output we needed, it did not **FULLY** take up just enough memory on the system as the calculations suggested, however, the program profiled the projected memory use before it ran and would not allow it to start as it knew that it did not have enough memory for any values above a certain threshold. This value was the best option to not overload the system.
   b. Other fields that changed are the amount of virtual memory. Virtual memory is used in this case because the amount of RAM currently being used is not enough to maintain the system. As a result, it must got to the hard disk for more space. This also caused a significant slowdown on the system.
   c. The amount of memory used as cache surprisingly went down. so and si also went way up to show how much memory was swapping in and out of disk as a result of the use of virtual memory. Pretty much everything from blocks, to interrrupts, to even time stolen from the virtual machine all saw a sharp spike as a result of the increase in load on the system imposed by the program.

4.

   a. The page size is 4096 bytes.  It takes 9.094 seconds to run.
   b. By swapping the i and j values we now access memory column by column instead of row by row.
   c. The execution time is increased to 9.881 seconds, a 0.787 second difference.
   d. By swapping the i and j values we now have invalid TLB hits so it takes longer.  (See Diagram Below)

5.

   a. After manually calculating, it is clear that the value of `76` for the `COEFFICIENT` parameter is the best option here. Any value larger incurs a segmentation fault by the system which signifies that it has reached it max for a malloc. After running the program, it appears that the physical system memory is almost entirely used besides the small sections of RAM which are thought to be reserved for system critical processes.
  b.
  c. Similar to what was observed in question 3, the system used physical hard drive storage to get more pages of memory to run the intensive program in memory. The command `ps -eo min_flt,maj_flt,cmd` is used to see how many page faults occured during the run of the process. This number showed that > 11000 faults occured which were minor and > 100 major page faults. This is pretty significant, but definitely not nearly as much as an x2go session might incur on a system, which is in the millions. This shows that even when artificially taxing a system, nothing beats a truly intensive program
