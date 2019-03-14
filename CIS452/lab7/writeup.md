1. Max Page Size: 4096 bytes. This was done via `getconf PAGE_SIZE`
2. Max filesize in bytes: 32 or 64 depending on where in the system. This was done via `getconf FILESIZEBYTES /` and also `getconf FILESIZEBYTES ~/Desktop`
3. Maximum number of processes per user(student): 20. Found via `cat /etc/security/limits.conf`
4. Maximum number of processes per user(faculty): 20(soft) 50(hard). Found via `cat /etc/security/limits.conf`
5. Maximum shared memory size: 61632512 bytes. Found via the `sysinfo` struct after calling `int sysinfo()`, code is in appendix of this document.
6. Maximum number of open files(soft): 1023. Found via the `file_lim.c` program which is included in the appendix
7. Maximum number of open files(hard): 1624448. Found via `cat /proc/sys/fs/file-max`. This is likely much larger due to the fact that it is a **system** limit and not just a per-user limit like we see with the soft limit.
8.
