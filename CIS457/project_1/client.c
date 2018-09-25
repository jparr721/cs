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
  char request[PACKET_SIZE];

  struct sockaddr_in server;
  memset(&server, 0, sizeof(server));

  server.sin_family = AF_INET;
  server.sin_port = htons(port);
  server.sin_addr.s_addr = inet_addr(host);

  printf("Enter your file request: ");
  fgets(request, PACKET_SIZE, stdin);

  request[strlen(request)] = 0;

  sendto(sockfd, request, strlen(request), 0, (struct sockaddr*)&server, sizeof(server));

  char response[PACKET_SIZE];
  int packets_remaining;

  recv(sockfd, &packets_remaining, PACKET_SIZE, MSG_CONFIRM);

  printf("Expected Packets: %d\n", packets_remaining);

  struct packet* packets = (struct packet*) malloc (packets_remaining * sizeof(struct packet));
  struct packet current;

  FILE* fp;
  fp = fopen("sample_copy.txt", "a");

  // Store all packets
  for (int i = 0; i < packets_remaining; i++) {
    recv(sockfd, &current, PACKET_SIZE, MSG_CONFIRM);
    packets[i] = current;
  }


  for (int i = 0; i < packets_remaining; i++) {
    fprintf(fp, "%s", packets[i].data);
  }

  fclose(fp);

  free(packets);
  close(sockfd);

  return EXIT_SUCCESS;
}
