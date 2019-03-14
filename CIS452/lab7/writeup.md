1. Max Page Size: 4096 bytes. This was done via `getconf PAGE_SIZE`
2. Max filesize in bytes: 32 or 64 depending on where in the system. This was done via `getconf FILESIZEBYTES /` and also `getconf FILESIZEBYTES ~/Desktop`
3. Maximum number of processes per user(student): 20. Found via `cat /etc/security/limits.conf`
4. Maximum number of processes per user(faculty): 20(soft) 50(hard). Found via `cat /etc/security/limits.conf`
5. Maximum shared memory size: 61632512 bytes. Found via the `sysinfo` struct after calling `int sysinfo()`, codeis in appendix of this document.
6.
7.
8.
