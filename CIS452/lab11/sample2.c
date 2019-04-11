#include <stdio.h>
#include <dirent.h>
#include <sys/stat.h>
#include <sys/types.h>
#include <errno.h>

int main()
{
	DIR *dirPtr;
	struct dirent *entryPtr;

	dirPtr = opendir (".");

	while ((entryPtr = readdir (dirPtr))){
		printf ("%-20s\n", entryPtr->d_name);
		struct stat st;
		stat((entryPtr->d_name), &st);
		printf ("Size: %d bytes\n", st.st_size);
	}
	closedir (dirPtr);
	return 0;
}
