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

off_t fsize(char* filename) {
  FILE * fp;
  fp = fopen(strtok(filename, "\n"), "r");
  fseek(fp, 0, SEEK_END);

  long bytes = ftell(fp);
  rewind(fp);
  fclose(fp);

  return bytes;
}

void* get_in_addr(struct sockaddr* sa) {
  if (sa->sa_family == AF_INET) {
    return &(((struct sockaddr_in*)sa)->sin_addr);
  }

  return &(((struct sockaddr_in6*)sa)->sin6_addr);
}

int main(int argc, char** argv) {
  char client_ip[INET6_ADDRSTRLEN];

  char port[5];

  printf("Please enter the port to connect to: ");
  fgets(port, 5, stdin);

  // Create our new ipv4 socket
  int sockfd = socket(AF_INET, SOCK_STREAM, 0);

  struct sockaddr_storage client;
  struct sockaddr_in server;
  server.sin_family = AF_INET;
  server.sin_port = htons(atoi(port));
  server.sin_addr.s_addr = INADDR_ANY; // Use my IP

  // Attempt to bind to the target port and addr
  if (bind(sockfd, (struct sockaddr*) &server, sizeof(server)) == -1) {
    fprintf(stderr, "Error, could not bind to specified port and address");
    return EXIT_FAILURE;
  }

  printf("Server now accepting connections...");
  listen(sockfd, 10);

  // Server loop
  while(1) {
    socklen_t sin_size = sizeof client;
    int client_socket = accept(sockfd, (struct sockaddr*) &client, &sin_size);

    if (client_socket == -1) {
      fprintf(stderr, "Error accepting connection from device\n");
    }

    inet_ntop(client.ss_family, get_in_addr((struct sockaddr *) &client), client_ip, sizeof(client_ip));

    printf("Server: New connection from: %s\n", client_ip);

    char file_name[100];
    // Get the requested file name
    recv(client_socket, file_name, 100, 0);

    off_t file_size = fsize(file_name);
    char* buffer = (char*)malloc(sizeof(char) * file_size);
    FILE* f = fopen(file_name, "r");

    if (f) {
      printf("file was found!\n");
      fread(&buffer, sizeof(char), sizeof(buffer)/sizeof(char), f);
      printf("%s", buffer);
      fclose(f);
    }

    printf("%s", buffer);
    send(client_socket, buffer, sizeof(buffer) * sizeof(char), 0);

    printf("Data sent to client successfully");
    close(client_socket);
  }

  return 0;
}
