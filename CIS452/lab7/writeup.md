# Lab 7

Jarred Parr and Alexander Fountain

| System Object                      | Method    | Value                                                        | Details                                                      |
| ---------------------------------- | --------- | ------------------------------------------------------------ | ------------------------------------------------------------ |
| Max semaphores per process         | Static    | 256, 32,000                                                  | Found via cat /usr/include/bits/posix_lim.h and semget() man pages. First value is pre 3.19 linux and the other is post |
| Max value of a counting semaphore  | Static    | 32767                                                        | Found via the man pages of semctl                            |
| Max value of a counting semaphore  | Empirical | 65535                                                        | Found via count.c shown below. It loops until the value overflows then terminated and read. |
| Maximum shared memory segment size | Empirical | 61632512 bytes                                               | Found in the sysinfo struct in the file lim.c shown below. Value was read from the system given its current state |
| Page Size                          | Dynamic   | 4096 bytes                                                   | Found via the getconf PAGE_SIZE command                      |
| Physical Pages in System           | Dynamic   | 4194304                                                      | sysconf(_SC_PHYS_PAGES) to find the physical page number     |
| Maximum Processes Per User         | Dynamic   | 20 for student, 20 (soft) for faculty, 50 (hard) for faculty | Found via the cat /etc/security/limits.conf file where they are able to be set by the sysadmin |
| Max File Size                      | Dynamic   | Unlimited                                                    | Found via ulimit -a. This means there is theoretically no guard against a file taking up available system space. |
| Max Open Files (hard)              | Dynamic   | 1624448                                                      | Found via cat /proc/sys/fs/file-max.                         |
| Max Open Files (soft)              | Dynamic   | 1023                                                         | Found via file_lim.c program included in the appendix of this document. This program opens files until the os stops me |
| Clock Resolution                   | Dynamic   | 3820.789                                                     | Found via lscpu \| grep MHz. This value gives the average    |


### count.c
```C
#include <stdio.h>
#include <stdlib.h>
#include <sys/ipc.h>
#include <sys/sem.h>


int main ()
{
	long int count = 1;
	//Initialize Semaphore here
	int sem = semget(IPC_PRIVATE, 1, 00600);

	for(;;) {
		//Increment semaphore count here

		if(semctl(sem, 0, SETVAL, count) == -1){
			perror("Darn\n");
			exit(1);
		}

		count++;

		printf("Count: %ld\n", count);
	}

}
```

### file_lim.c
```C
# include <assert.h>
# include <stdio.h>
# include <stdlib.h>
# include <unistd.h>
# include <sys/types.h>
# include <sys/stat.h>
# include <sys/wait.h>
# include <string.h>
# include <fcntl.h>

int main(){
  int t;

  for(;;){
    t = open("test", O_RDONLY);
    if (t < 0){
      perror("open");
      exit(1);
    }
    printf("%d: ok\n", t);
  }
}
```

### lim.c
```C
#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>
#include <sys/sysinfo.h>

int main(int argc, char** argv) {
  struct sysinfo si;

  int ret = 0;
  if ((ret = sysinfo(&si)) < 0) {
    perror("wtf");
    return -1;
  }

  printf("max shared mem: %lu", si.sharedram);

  return 0;
}
```
