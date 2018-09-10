#include <arpa/inet.h>
#include <netinet/in.h>
#include <signal.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/socket.h>
#include <sys/stat.h>
#include <sys/types.h>
#include <unistd.h>

char* file_exists(char* filename) {
  char* buffer;
  long bytes;
  FILE* f = fopen(filename, "r");

  if (f) {
    fseek(f, 0, SEEK_END);
    bytes = ftell(f);
    fseek(f, 0, SEEK_SET);

    buffer = malloc(bytes);

    fread(&buffer, sizeof(char), sizeof(buffer)/sizeof(char), f);
    fclose(f);

    return buffer;
  }

  return "Unable to find that file, boss";
}

int main(void) {
  char client_ip[INET6_ADDRSTRLEN];

  // Create our new ipv4 socket
  int sockfd = socket(AF_INET, SOCK_STREAM, 0);

  struct sockaddr_storage client;
  struct sockaddr_in server;
  server.sin_family = AF_INET;
  server.sin_port = htons(8080);
  server.sin_addr.s_addr = INADDR_ANY;

  // Attempt to bind to the target port and addr
  if (bind(sockfd, (struct sockaddr*) &server, sizeof(server)) != 0) {
    fprintf(stderr, "Error, could not bind to specified port and address");
    return EXIT_FAILURE;
  }

  listen(sockfd, 10);
  printf("Server now accepting connections...");

  // Server loop
  while(1) {
    int client_socket = accept(sockfd, (struct sockaddr*) &client, sizeof(client));
    if (client_socket != 0) {
      fprintf(stderr, "Error accepting connection from device");
    }

    inet_ntop(client.ss_family, get_in_addr((struct sockaddr *) &client), client_ip, sizeof(client_ip));

    printf("Server: New connection from: %s\n", client_ip);

    char file_name[100];
    // Get the requested file name
    recv(client_socket, file_name, 100, 0);

    char* message = "Checking on that for you...";
    send(client_socket, message, sizeof(message), 0);

    char* file_data = file_exists(file_name);
    send(client_socket, file_data, sizeof(file_data), 0);

    printf("Data sent to client successfully");
    close(client_socket);
  }

  return 0;
}
