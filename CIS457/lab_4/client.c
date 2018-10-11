#include <arpa/inet.h>
#include <errno.h>
#include <netinet/in.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/select.h>
#include <sys/socket.h>
#include <unistd.h>

#define MAXDATASIZE 4096

int get_port();
in_addr_t get_host();
char* input_handler();

int get_port() {
  printf("Enter the port number to run on: ");
  char port[10] = "3000";
  fgets(port, 10, stdin);

  return atoi(port);
}

in_addr_t get_host() {
  printf("Enter the host to connect to: ");
  char ip[100] = "127.0.0.1";
  fgets(ip, 100, stdin);
  return inet_addr(ip);
}

char* input_handler() {
  char* message = (char*) malloc(sizeof(char));
  printf("> ");
  fgets(message, MAXDATASIZE, stdin);

  return message;
}

int main(int argc, char** argv) {
  int sockfd = socket(AF_INET, SOCK_STREAM, 0);
  if (sockfd < 0) {
    fprintf(stderr, "There was an error creating the socket!\n");
    return EXIT_FAILURE;
  }

  int port = get_port();
  in_addr_t host = get_host();

  struct sockaddr_in server;
  server.sin_family = AF_INET;
  server.sin_port = htons(port);
  server.sin_addr.s_addr = host;

  int c = connect(sockfd, (struct sockaddr*) &server, sizeof(server));

  if (c == -1) {
    fprintf(stderr, "There was an error connecting to the server!\n");
    return EXIT_FAILURE;
  }

  struct timeval timeout;
  timeout.tv_sec = 0;
  timeout.tv_usec = 0;

  char response[MAXDATASIZE];
  char* input;

  while(1) {
    fd_set read_fd;
    FD_ZERO(&read_fd);

    int fdmax = sockfd;
    FD_SET(sockfd, &read_fd);
    FD_SET(STDIN_FILENO, &read_fd);

    if(select(fdmax + 1, &read_fd, NULL, NULL, &timeout) == -1) {
      fprintf(stderr, "Unable to modify the file descriptor\n");
    }

    if (FD_ISSET(sockfd, &read_fd)) {
      /* int r = recv(sockfd, response, MAXDATASIZE, 0); */
      /* if (r < 0) { */
      /*   fprintf(stderr, "Failed to receive response, dawg"); */
      /* } else { */
      /*   if (strcmp("Quit", response) == 0) { */
      /*     printf("Server has ended the connection"); */
      /*     break; */
      /*   } */
      /*   printf("< %s\n", response); */
      /* } */
      recv(sockfd, response, MAXDATASIZE, 0);

      if (strcmp("Quit", response) == 0) {
        printf("Server has ended the connection");
        break;
      } else {
        printf("< %s\n", response);
      }
    }

    if (FD_ISSET(STDIN_FILENO, &read_fd)) {
      input = input_handler();
      int s = send(sockfd, input, strlen(input) + 1, 0);
      if (s < 0) {
        fprintf(stderr, "Failed to send message, sorry bruh");
      }

      if (strcmp("Quit", input) == 0) {
        printf("Ended the connection");
        break;
      }
    }
  }

  close(sockfd);

  return EXIT_SUCCESS;
}
