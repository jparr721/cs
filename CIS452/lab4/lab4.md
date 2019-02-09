# Lab 4

Alexander Fountain, Jarred Parr

1. This program executes a thread which does some non-blocking operation. Because of the artificial overhead of the `sleep(1)` call added to the thread, there must be a sleep call added to the main function to allow for the program to observe the thread behavior. Otherwise the parent will have exited, cleaned up, and competed before the thread could even run its code.

2. It initially prints a jumbled looking output of "Hello"'s and "World"'s. It appears that these values are fighting for access to stdout. Since there isn't a blocking call on the print, the output get mangled when placed on the stdout stream, which is why we see the stream of Word following the first Hello, then all of the Hello's.

3. This improves upon question two. The sleep allowed for the threads to sync a bit better and produce a more consistent output, but it still had some errors.

4. Linux uses Many-to-one. This is because, when we observe the blocking call on a thread, we see that is does not hinder the other threads from competing for access to the standard output stream. When observing the process via the ps and top tools

5. In this run we see that the parent sees 5 as the shared data value.  The first child spawned off looks at the shared data value and also sees it as 5 because the parent hasn't increased the value to 6 yet.  The second child looks at the shared data value and also sees 5 because neither the parent or the other child has updated the value yet.  The first child hasn't updated it's value yet becasue of the sleep(1) before the increment.  The first child looks at the shared data value again and now sees 7 because the parent and the first child have increased it by one each.  The second child looks at the value and sees 8 because it increased the value by one more.  Finally, the parent prints 8 as the final value.  This output shows that there is time between when the shared value is updated vs when it is read even though it looks like it should update immediately and give the output, 5, 6, 7, 8, 8.  

6.  The thread specific data is passed via a pointer (void* arg) that is then used to specify the data type of char val_ptr.
