#include <errno.h>
#include <netinet/in.h>
#include <stdio.h>
#include <string.h>
#include <sys/socket.h>

int main(void) {
  char* port, host;

  printf("Please enter the port to connect to: ");
  fgets(port, 4, stdin);

  printf("Please enter the host to connect to: ");
  fgets(host, 20, stdin);

  int sockfd = socket(AF_INET, SOCK_STREAM, 0);

  if (sockfd < 0) {
    fprintf(stderr, "There was an error creating the socket\n");
    return errno;
  }

  struct sockaddr_in server;
  server.sin_family = AF_INET;
  server.sin_port = htons(8080);
  server.sin_addr.s_addr = inet_addr("127.0.0.1");

  int e = connect(sockfd, (struct sockaddr*) &server, sizeof(server));

  if (e != 0) {
    fprintf(stderr, "There was an error connecting to the server!!");
    return errno;
  }

  return 0;
}
