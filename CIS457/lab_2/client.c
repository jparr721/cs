
#include <arpa/inet.h>
#include <errno.h>
#include <netinet/in.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/socket.h>
#include <unistd.h>

#define MAXDATASIZE 1024

int main(void) {
  int numbytes;
  int sockfd = socket(AF_INET, SOCK_DGRAM, 0);
  char message[MAXDATASIZE];
  char buffer[MAXDATASIZE];
  char port[6];
  char ip[21];

  printf("Please enter the port to connect to: ");
  fgets(port, 6, stdin);

  printf("Please enter the host to connect to: ");
  fgets(ip, 21, stdin);

  printf("What would you like to say?: ");
  fgets(message, MAXDATASIZE, stdin);

  if (sockfd < 0) {
    fprintf(stderr, "There was an error creating the socket\n");
    return EXIT_FAILURE;
  }

  struct sockaddr_in server;
  memset(&server, 0, sizeof(server));

  server.sin_family = AF_INET;
  server.sin_port = htons(atoi(port));
  server.sin_addr.s_addr = inet_addr(ip);

  int e = connect(sockfd, (struct sockaddr*) &server, sizeof(server));

  if (e == -1) {
    fprintf(stderr, "There was an error conncecting to the server");
    return EXIT_FAILURE;
  }

  if (send(sockfd, message, strlen(message), 0) == -1) {
    fprintf(stderr, "Error sending the message");
  }

  if ((numbytes = recv(sockfd, buffer, MAXDATASIZE - 1, 0)) == -1) {
    fprintf(stderr, "error receiving message");
  }

  buffer[numbytes] = '\0';

  printf("\nGot a reply from the server: %s\n", buffer);

  close(sockfd);
  return EXIT_SUCCESS;
}
