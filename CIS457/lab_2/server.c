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

int main(int argc, char** argv) {
  if (argc != 3) {
    fprintf(stderr, "usage: ./server port message");
    return EXIT_FAILURE;
  }

  char* port = argv[1];
  char* message = argv[2];
  char client_ip[INET6_ADDRSTRLEN];
  int sockfd;
  char buffer[MAXLINE];
  struct sockaddr_storage client;
  struct sockaddr_in server;

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

  listen(sockfd, 10);

  while(1) {
    socklen_t sin_size = sizeof client;
    int client_socket = accept(sockfd, (struct sockaddr*) &client, &sin_size);

    if (client_socket == -1) {
      fprintf(stderr, "Error accepting connection from device\n");
    }

    inet_ntop(client.ss_family, get_in_addr((struct sockaddr*) &client), client_ip, sizeof(client_ip));

    printf("Server: New connection from: %s\n", client_ip);

    // Send data over the socket
    send(client_socket, message, strlen(message), 0);

    char* data;
    recv(client_socket, data, 100, 0);

    printf("Got the goods, boss: %s\n", data);
    close(client_socket);
  }

  return EXIT_SUCCESS;
}
