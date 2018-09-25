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
  //printf("Packet Total Size: %d", (int) sizeof(struct packet));
  recv(sockfd, &packets_remaining, (int) sizeof(struct packet), MSG_CONFIRM);

  printf("Expected Packets: %d\n", packets_remaining);

  struct packet* packets = (struct packet*) malloc (packets_remaining * sizeof(struct packet));
  struct packet current;

  FILE* clear = fopen("sample_copy.txt", "w");
  fclose(clear);

  FILE* fp;
  fp = fopen("sample_copy.txt", "a");
  
  // Store all packets
  for (int i = 0; i < packets_remaining; i++) {
    recv(sockfd, &current, (int) sizeof(struct packet), MSG_CONFIRM);
    packets[i] = current;
    //packets[i].data[packets[i].size] = '\0';
    //printf("\n---------- PACKET DATA ----------\n%s\n", current.data);
  }


  for (int i = 0; i < packets_remaining; i++) {
    fwrite(packets[i].data, sizeof(char), packets[i].size, fp);
  }

  fclose(fp);

  free(packets);
  close(sockfd);

  return EXIT_SUCCESS;
}
