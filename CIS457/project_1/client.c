#include <arpa/inet.h>
#include <netinet/in.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/socket.h>
#include <unistd.h>

#define PACKET_SIZE 1024

#pragma pack(1)

struct packet {
  int packet_number;
  unsigned int size;
  char data[PACKET_SIZE];
};

int validate_args(int count) {
  if (count != 3) {
    fprintf(stderr, "Invalid argument count. Usage: ./client host port\n");
    return -1;
  }

  return 0;
}

int main(int argc, char** argv) {
  if ((validate_args(argc)) == -1) {
    return EXIT_FAILURE;
  }

  int sockfd = socket(AF_INET, SOCK_DGRAM, 0);
  if (sockfd < 0) {
    fprintf(stderr, "Failed to create udp socket!\n");
    return EXIT_FAILURE;
  }

  char* host = argv[1];
  int port = atoi(argv[2]);
  char* request = (char*) malloc(sizeof(char));

  struct sockaddr_in server;
  memset(&server, 0, sizeof(server));

  server.sin_family = AF_INET;
  server.sin_port = htons(port);
  server.sin_addr.s_addr = inet_addr(host);

  printf("Enter your file request: ");
  fgets(request, PACKET_SIZE, stdin);

  request[strlen(request)] = '\0';

  sendto(sockfd, request, strlen(request), 0, (struct sockaddr*)&server, sizeof(server));

  int packets_remaining;
  recv(sockfd, &packets_remaining, (int) sizeof(struct packet), MSG_CONFIRM);

  if (packets_remaining < 0) {
    printf("The server could not find the desired file.\n\nExiting...\n");
    return EXIT_SUCCESS; 
  } else {
    printf("Expected Packets: %d\n", packets_remaining);
  }

  struct packet* packets = (struct packet*) malloc (packets_remaining * sizeof(struct packet));
  struct packet current;
  
  // Store all packets
  for (int i = 0; i < packets_remaining; i++) {
    recv(sockfd, &current, (int) sizeof(struct packet), MSG_CONFIRM);
    packets[i] = current;
  }

  printf("All packets have been received.\n");

  char* dest = (char*) malloc(sizeof(char));
  printf("Enter your desired file destination: ");
  fgets(dest, PACKET_SIZE, stdin);
  dest[strlen(dest) - 1] = '\0';

  FILE* clear = fopen(dest, "w");
  fclose(clear);

  FILE* fp;
  fp = fopen(dest, "a");

  for (int i = 0; i < packets_remaining; i++) {
    fwrite(packets[i].data, sizeof(char), packets[i].size, fp);
  }

  fclose(fp);
  free(request);
  free(packets);
  close(sockfd);

  return EXIT_SUCCESS;
}
