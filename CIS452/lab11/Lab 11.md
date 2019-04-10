# Lab 11

Jarred Parr and Alexander Fountain

1. a. `stat(1)` is a command line program and `stat(3)` is a library function which invokes that same functionality but inside of a running executable.

   b. This program reads a from `stdin` the name of a file and then uses `stat` to report the mode and type of file that it is.

   c.

   ```C
   #include <stdio.h>
   #include <stdlib.h>
   #include <sys/stat.h>
   #include <sys/types.h>
   #include <errno.h>
   
   int main(int argc, char *argv[])
   {
      struct stat statBuf;
   
      if (argc < 2) {
         printf ("Usage: filename required\n");
         exit(1);
      }
   
      if (stat (argv[1], &statBuf) < 0) {
         perror ("huh?  there is ");
         exit(1);
      }
   
      if ((statBuf.st_mode & S_IFMT) == S_IFDIR) {
         printf("%s is a directory\n", argv[1]);
      } else {
         printf("%s is not a directory\n", argv[1]);
      }
      return 0;
   }
   ```

   ![](/home/ghost/Screenshots/2019-04-10_19-25.png)