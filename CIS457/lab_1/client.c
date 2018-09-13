#include <arpa/inet.h>
#include <errno.h>
#include <netinet/in.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/socket.h>
#include <unistd.h>

#define MAXDATASIZE 4096

int main(void) {
  int sockfd = socket(AF_INET, SOCK_STREAM, 0);
  int numbytes;
  char buffer[MAXDATASIZE];
  char port[6];
  char host[21];
  char filename[100];

  printf("Please enter the port to connect to: ");
  fgets(port, 6, stdin);

  printf("Please enter the host to connect to: ");
  fgets(host, 21, stdin);

  printf("Please enter the name of the file you want: ");
  fgets(filename, 100, stdin);

  if (sockfd < 0) {
    fprintf(stderr, "There was an error creating the socket\n");
    return errno;
  }

  struct sockaddr_in server;
  server.sin_family = AF_INET;
  server.sin_port = htons(atoi(port));
  server.sin_addr.s_addr = inet_addr(host);

  int e = connect(sockfd, (struct sockaddr*) &server, sizeof(server));

  if (e == -1) {
    fprintf(stderr, "There was an error connecting to the server!!");
    return errno;
  }

  if(send(sockfd, filename, strlen(filename), 0) == -1) {
    perror("Error sending data my boy");
  };

  if ((numbytes = recv(sockfd, buffer, MAXDATASIZE - 1, 0)) == -1) {
    perror("Error receiving file data");
  }

  printf("Client: Received some data: %s\n", buffer);

  FILE* fp = fopen("copy.txt", "w");

  fwrite(buffer, sizeof(char), strlen(buffer), fp);

  return 0;
}
