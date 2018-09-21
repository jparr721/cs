#include <arpa/inet.h>
#include <errno.h>
#include <netdb.h>
#include <signal.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>

#define PACKET_SIZE 1024

off_t file_size(FILE** f);
FILE* init_fstream(char* location);
char* read_fild(long bytes, FILE** f);

FILE* init_fstream(char* location) {
  FILE* fp;
  location[strlen(location) - 1] = 0;
  printf("%s", location);
  fp = fopen(location, "r");

  if (!fp) {
    fprintf(stderr, "Error, could not find file: %s", location);
  }

  return fp;
}

off_t file_size(FILE** f) {
    fseek(*f, 0L, SEEK_END);
    long size = ftell(*f);
    fseek(*f, 0L, SEEK_SET);

    return size;
}

char* read_file(long bytes, FILE** f) {
    char* buffer = (char*) malloc(bytes* sizeof(char));
    fread(buffer, sizeof(char), bytes, *f);

    return buffer;
}

void read_file_by_packet_size(FILE** f, int pack_num, char* contents) {
    int offset = (pack_num * PACKET_SIZE) * sizeof(char);
    fseek(*f, offset, SEEK_SET);
    fread(contents, sizeof(char), PACKET_SIZE, *f);
}

int main(int argc, char** argv) {
    if (argc != 2) {
        fprintf(stderr, "Invalid argument count. Usage: ./server port");
        return EXIT_FAILURE;
    }

    int sockfd = socket(AF_INET, SOCK_DGRAM, 0);
    int port =atoi(argv[1]);
    char cli_req[PACKET_SIZE];

    struct sockaddr_storage client;
    struct sockaddr_in server;

    if (sockfd < 0) {
        perror("Failed to create udp socket.");
        return EXIT_FAILURE;
    }

    memset(&server, 0, sizeof(server));
    memset(&client, 0, sizeof(client));

    server.sin_family = AF_INET;
    server.sin_addr.s_addr = INADDR_ANY;
    server.sin_port = htons(port);

    if (bind(sockfd, (struct sockaddr*) &server, sizeof(server)) < 0) {
        fprintf(stderr, "Error, could not bind to the specified port.");
        return EXIT_FAILURE;
    }

    while(1) {
      socklen_t len = sizeof client;

      int res = recvfrom(sockfd, (char*) cli_req, PACKET_SIZE, MSG_WAITALL, (struct sockaddr*) &client, &len);
      printf("%s", cli_req);

      if (res == -1) {
          fprintf(stderr, "Error, could not receive data from client.");
      } else {
          fprintf(stdout, "Client file request: %s", cli_req);
      }

      // Trying to be efficient, so we open the file stream once, and then close it once.
      FILE* file_ptr = init_fstream(cli_req);

      off_t size = file_size(&file_ptr);
      fprintf(stdout, "File Size: %li", size);

      char* file_contents = read_file(size, &file_ptr);
      printf("File contents: %s", file_contents);

      // Just an example.
      char packet[PACKET_SIZE];
      read_file_by_packet_size(&file_ptr, 0, packet);
    }

    close(sockfd);

    return EXIT_SUCCESS;
}
