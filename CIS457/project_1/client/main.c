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

    while (packets_remaining > 0) {
        struct packet current_packet;

	  // Receive the server's packet.
    recv(sockfd, &current_packet, PACKET_SIZE, MSG_CONFIRM);

    packet_data[current_packet.packet_number] = 1;

	  // Send the server the ACK packet.
    sendto(sockfd, &current_packet.packet_number, sizeof(int), 0, (struct sockaddr*) &server, sizeof(server));

    int packets_not_received = 0;
    for (int i = 0; i < total_packets; ++i) {
        if (packet_data[i] == 0) {
            packets_not_received++;
        }
    }

      printf("Packet number: %d data: %s\n", current_packet.packet_number, current_packet.data);
      packets_remaining = packets_not_received;
      printf("Packet data: %d\n", current_packet.packet_number);
    }

    close(sockfd);
}
