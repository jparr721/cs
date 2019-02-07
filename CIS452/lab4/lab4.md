# Lab 4

Alexander Fountain, Jarred Parr

1. This program executes a thread which does some non-blocking operation. Because of the artificial overhead of the `sleep(1)` call added to the thread, there must be a sleep call added to the main function to allow for the program to observe the thread behavior. Otherwise the parent will have exited, cleaned up, and competed before the thread could even run its code.

2. It initially prints a jumbled looking output of "Hello"'s and "World"'s. It appears that these values are fighting for access to stdout. Since there isn't a blocking call on the print, the output get mangled when placed on the stdout stream, which is why we see the stream of Word following the first Hello, then all of the Hello's.
3. This improves upon question two. The sleep allowed for the threads to sync a bit better and produce a more consistent output, but it still had some errors.
4. Linux uses Many-to-one. This is because, when we observe the blocking call on a thread, we see that is does not hinder the other threads from competing for access to the standard output stream. When observing the process via the ps and top tools