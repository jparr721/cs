#include <stdio.h>
#include <stdlib.h>
#include <dirent.h>
#include <string.h>
#include <sys/stat.h>
#include <sys/types.h>
#include <unistd.h>
#include <errno.h>

#define DLIST_LEN 1024
#define MIN(x, y) ((x) < (y)) ? (x) : (y)

int comp(char* n1, char* n2) {
  size_t l1 = strlen(n1);
  size_t l2 = strlen(n2);
  size_t min = MIN(l1, l2);

  for (int i = 0; i < min; ++i) {
    if (n1[i] != n2[i]) {
      char ch1 = n1[i] - 'a';
      char ch2 = n2[i] - 'a';
      if (ch1 < ch2) return 1;
    }
  }

  return l1 == min ? 1 : 0;
}

void merge(char* dlist[], int l, int m, int r) {
  int s1 = (m - 1) - l;
  int s2 = r - m;

  char *L[s1], *R[s2];
  for (int i = 0; i < s1; ++i) {
    L[i] = dlist[i];
  }

  for (int i = m; i < s2; ++i) {
    R[i] = dlist[i];
  }

  int i, j, k;
  for (i = 0, j = 0, k = l; i < s1 && j < s2; ++k) {
    if (comp(L[i], R[i])) {
      dlist[k] = L[i];
      ++i;
    }

    if (!comp(L[i], R[i])) {
      dlist[k] = R[j];
      ++j;
    }
  }

  while (i < s1) {
    dlist[k] = L[i];
    ++i;
    ++k;
  }

  while (j < s1) {
    dlist[k] = L[j];
    ++j;
    ++k;
  }
}

void sort(char* dlist[], int l, int r) {
  int m = l + (r - l) / 2;

  sort(dlist, l, m);
  sort(dlist, m, r);

  merge(dlist, l, m, r);
}

int main(int argc, char** argv) {
  DIR *dir_ptr;
  struct dirent *entry_ptr;
  struct stat stat_buf;

  int opt;

  char* directory = argv[2];
  if (stat(directory, &stat_buf) < 0) {
    perror("Invalid input supplied");
    return -1;
  }

  while ((opt = getopt(argc, argv, "n:i")) != -1) {
    switch(opt) {
      case 'n':
        dir_ptr = opendir(directory);
        while ((entry_ptr = readdir(dir_ptr))) {
          struct stat st;
          stat((entry_ptr->d_name), &st);
          printf("%-20s uid: %d gid: %d\n", entry_ptr->d_name, st.st_uid, st.st_gid);
        }
        break;
      case 'i':
        dir_ptr = opendir(directory);
        while ((entry_ptr = readdir(dir_ptr))) {
          struct stat st;
          stat((entry_ptr->d_name), &st);
          printf("%-20s inode: %lu\n", entry_ptr->d_name, st.st_ino);
        }
        break;
      case '?':
        printf("invalid option specified");
        break;
      default:
        dir_ptr = opendir("./");
        while ((entry_ptr = readdir(dir_ptr))) {
          struct stat st;
          stat((entry_ptr->d_name), &st);
          printf("%-20s\n", entry_ptr->d_name);
        }
    }
  }
}
