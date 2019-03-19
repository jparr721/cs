1. Max Page Size: 4096 bytes. This was done via `getconf PAGE_SIZE`
2. Max filesize in bytes: 32 or 64 depending on where in the system. This was done via `getconf FILESIZEBYTES /` and also `getconf FILESIZEBYTES ~/Desktop`  This is 2^32 bits, which is then divided by 8 to convert to bytes, gives us 536,870,912 bytes.  This is roughly half a gig.
3. Maximum number of processes per user(student): 20. Found via `cat /etc/security/limits.conf`
4. Maximum number of processes per user(faculty): 20(soft) 50(hard). Found via `cat /etc/security/limits.conf`
5. Maximum shared memory size: 61632512 bytes. Found via the `sysinfo` struct after calling `int sysinfo()`, code is in appendix of this document.
6. Maximum number of open files(soft): 1023. Found via the `file_lim.c` program which is included in the appendix
7. Maximum number of open files(hard): 1624448. Found via `cat /proc/sys/fs/file-max`. This is likely much larger due to the fact that it is a **system** limit and not just a per-user limit like we see with the soft limit.
8. Maximum semaphores per process: 256. Found in `/usr/include/bits/posix_lim.h`
9. Maximum semaphore value(static): 32767
10. Maximum number of physical pages in system: 3,981,566. Found via `cat /sys/meminfo` dividing the total paged phsyical memory by the page size.
11. Clock Resolution (msec): 3820.789. Found via `lscpu | grep MHz`
