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

#define MAXLINE 4096

char* fzie(char* filename) {
  FILE* fp;
  filename[strlen(filename) - 1] = 0;

  fp = fopen(filename, "r");
  if (!fp) {
    fprintf(stderr, "Error, could not find file");
  }

  fseek(fp, 0, SEEK_END);
  long bytes = ftell(fp);
  fseek(fp, 0, SEEK_SET);

  char* buffer = (char*) malloc(bytes * sizeof(char));
  fread(buffer, sizeof(char), bytes, fp);

  fclose(fp);
  return buffer;
}

void *get_in_addr(struct sockaddr *sa) {
  if (sa->sa_family == AF_INET) {
    return &(((struct sockaddr_in*)sa)->sin_addr);
  }

  return &(((struct sockaddr_in6*)sa)->sin6_addr);
}

int main(int argc, char** argv) {
  if (argc != 2) {
    fprintf(stderr, "usage: ./server port");
    return EXIT_FAILURE;
  }

  char* port = argv[1];
  int sockfd;
  char buffer[MAXLINE];
  struct sockaddr_in server, client;

  if ((sockfd = socket(AF_INET, SOCK_DGRAM, 0)) < 0) {
    perror("Failed to create udp socket");
    return EXIT_FAILURE;
  }

  memset(&server, 0, sizeof(server));
  memset(&client, 0, sizeof(client));

  server.sin_family = AF_INET;
  server.sin_addr.s_addr = INADDR_ANY;
  server.sin_port = htons(atoi(port));

  if (bind(sockfd, (struct sockaddr*) &server, sizeof(server)) == -1) {
    fprintf(stderr, "Error, could not bind to specified port and address");
    return EXIT_FAILURE;
  }


  listen(sockfd, 10);

  while(1) {
    socklen_t len = sizeof client;
    // Accept the file name from the client
    int n = recvfrom(sockfd, (char*) buffer, MAXLINE, MSG_WAITALL, (struct sockaddr*) &client, &len);

    if (n == -1) {
      fprintf(stderr, "Error, could not receive data from client");
    }

    char* file_size = fsize(buffer);
  }
}
