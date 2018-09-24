#include <stdio.h>
#include <string.h>
#include <unistd.h>
#include <stdlib.h>
#include <sys/socket.h>
#include <netinet/in.h>
#include <arpa/inet.h>

#define PACKET_SIZE 1024

struct packet {
  int packet_number;
  char data[PACKET_SIZE];
};

void reorder_packets(struct packet* packet_queue, int total_packets) {
  for (int i = 0; i < total_packets; i++) {
    for (int j = 0; j < total_packets; j++) {
      if (packet_queue[i].packet_number > packet_queue[j].packet_number) {
        struct packet temp = packet_queue[i];
        packet_queue[i] = packet_queue[j];
        packet_queue[j] = temp;
      }
    }
  }
}

int write_file_by_packet_size(char* location, struct packet* packet) {
  FILE* fp;

  fp = fopen(location, "a");

  if (fp) {
    fseek(fp, (packet->packet_number * PACKET_SIZE) * sizeof(char), SEEK_SET);
    fwrite(packet->data, sizeof(char), strlen(packet->data), fp);
    return 0;
  }

  fprintf(stderr, "Could not write data\n");
  return -1;
}

int main(int argc, char** argv) {
    if (argc != 3) {
        fprintf(stderr, "Invalid argument count. Usage: ./client host port");
        return EXIT_FAILURE;
    }

    int sockfd = socket(AF_INET, SOCK_DGRAM, 0);
    char* host = argv[1];
    int port = atoi(argv[2]);
    char request[PACKET_SIZE];

    if (sockfd < 0) {
        perror("Failed to create udp socket.");
        return EXIT_FAILURE;
    }

    struct sockaddr_in server;
    memset(&server, 0, sizeof(server));

    server.sin_family = AF_INET;
    server.sin_port = htons(port);
    server.sin_addr.s_addr = inet_addr(host);

    printf("Enter your file request: ");
    fgets(request, PACKET_SIZE, stdin);

    // Replacing a possible \n from the request char*.
    request[strlen(request)] = '\0';

    sendto(sockfd, request, strlen(request), 0, (struct sockaddr*) &server, sizeof(server));

    char response[PACKET_SIZE];
    int total_packets;
    int packets_remaining;

    recv(sockfd, &packets_remaining, PACKET_SIZE, MSG_CONFIRM);
    total_packets = packets_remaining;

    printf("Expected Packets: %d\n", packets_remaining);

    int* packet_data = (int*) malloc(packets_remaining * sizeof(int));

    for (int i = 0; i < packets_remaining; ++i) {
        packet_data[i] = 0;
    }

    struct packet* packets = (struct packet*) malloc (total_packets * sizeof(struct packet));

    while (packets_remaining > 0) {
      struct packet current_packet;

      // Receive the server's packet.
      recv(sockfd, &current_packet, PACKET_SIZE, MSG_CONFIRM);
      packets[current_packet.packet_number] = current_packet;


      packet_data[current_packet.packet_number] = 1;

      /* int packets_not_received = 0; */
      /* for (int i = 0; i < total_packets; ++i) { */
      /*     if (packet_data[i] == 0) { */
      /*         packets_not_received++; */
      /*     } */
      /* } */

      // Send the server the ACK packet.
      sendto(sockfd, &current_packet.packet_number, sizeof(int), 0, (struct sockaddr*) &server, sizeof(server));

      packets_remaining--;
    }

    FILE* fp;
    fp = fopen("sample_copy.txt", "a");

    for (int i = 0; i < total_packets; i++) {
      fprintf(fp, "%s", packets[i].data);
    }
    fclose(fp);

    free(packets);
    free(packet_data);
    close(sockfd);
}
