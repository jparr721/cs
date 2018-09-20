#include <arpa/inet.h>
#include <netdb.h>
#include <netinet/in.h>
#include <signal.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/socket.h>
#include <sys/stat.h>
#include <sys/types.h>
#include <sys/wait.h>
#include <unistd.h>

#define MAXLINE 1024

void* get_in_addr(struct sockaddr* sa) {
  if (sa->sa_family == AF_INET) {
    return &(((struct sockaddr_in*)sa)->sin_addr);
  }

  return &(((struct sockaddr_in6*)sa)->sin6_addr);
}

int main(void) {
  char port[6];
  int sockfd;
  char buffer[MAXLINE];
  struct sockaddr_storage client;
  struct sockaddr_in server;

  printf("Please enter the port to bind to: ");
  fgets(port, sizeof(port), stdin);

  if ((sockfd = socket(AF_INET, SOCK_DGRAM, 0)) < 0) {
    fprintf(stderr, "Failed to create datagram socket");
    return EXIT_FAILURE;
  }

  memset(&server, 0, sizeof(server));
  memset(&client, 0, sizeof(client));

  server.sin_family = AF_INET;
  server.sin_addr.s_addr = INADDR_ANY;
  server.sin_port = htons(atoi(port));

  if (bind(sockfd, (struct sockaddr*) &server, sizeof(server)) == -1)  {
    fprintf(stderr, "Error, could not bind to specified port and address");
    return EXIT_FAILURE;
  }

  while(1) {
    socklen_t len = sizeof client;
    int n = recvfrom(sockfd, (char*) buffer, MAXLINE, MSG_WAITALL, (struct sockaddr*) &client, &len);

    buffer[n] = '\0';
    printf("Client said: %s", buffer);

    sendto(sockfd, buffer, strlen(buffer), MSG_CONFIRM, (struct sockaddr*) &client, len);
  }

  return EXIT_SUCCESS;
}
