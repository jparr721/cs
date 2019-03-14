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
