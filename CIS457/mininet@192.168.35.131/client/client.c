#include <arpa/inet.h>
#include <netinet/in.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/socket.h>
#include <time.h>
#include <unistd.h>

#define PACKET_SIZE 1024

#pragma pack(1)

struct packet {
  int packet_number;
  unsigned int size;
  unsigned int type;
  char data[PACKET_SIZE];
  unsigned int chk;
};

unsigned short make_checksum(char* data, int length) {
  unsigned short chk = 0;
  unsigned int cur_length = length;
  printf("---- CHECKSUM (%d, %d) ----\n", (int) strlen(data), length);
  while (cur_length != 0) {
    chk -= *data++;
    cur_length--;
  }

  return chk;
}

int validate_args(int count) {
  if (count != 3) {
    fprintf(stderr, "Invalid argument count. Usage: ./client host port\n");
    return -1;
  }

  return 0;
}

int check_order(int len, struct packet* packets) {
  for (int i = 0; i < len; i++) {
    if (packets[i].packet_number != i + 1) {
      printf("Packets are not in order...\n");
      return 0;
    }
  }
  printf("Packets are in order!\n");
  return 1;
}

int reorder_packets(int len, struct packet* packets) {
  for (int i = 0; i < len; i++) {
    for (int j = i + 1; j < len; j++) {
      if (packets[i].packet_number > packets[j].packet_number) {
        struct packet temp_packet = packets[i];
        packets[i] = packets[j];
        packets[j] = temp_packet;
      }
    }
  }

  return 0;
}

int write_packet(int packets_remaining, struct packet* packets, char* dest) {
  FILE* fp;
  fp = fopen(dest, "a");

  for (int i = 0; i < packets_remaining; i++) {
    fwrite(packets[i].data, sizeof(char), packets[i].size, fp);
    return 1;
  }

  return 0;
}

int validate_checksum(struct packet packet) {
  unsigned short checksum = packet.chk;
  unsigned short new_checksum = make_checksum(packet.data, packet.size);
  if (checksum != new_checksum) {
    return -1;
  }

  return 1;
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

  int packets_remaining = -1;
  recv(sockfd, &packets_remaining, (int) sizeof(struct packet), MSG_CONFIRM);

  // Packets remaining is critical...
  if (packets_remaining == -1) {
    while (packets_remaining == -1) {
      recv(sockfd, &packets_remaining, (int) sizeof(struct packet), MSG_CONFIRM);
    }
  }
  sendto(sockfd, &packets_remaining, sizeof(int), 0, (struct sockaddr*) &server, sizeof(server));

  if (packets_remaining < 0) {
    printf("The server could not find the desired file.\n\nExiting...\n");
    return EXIT_SUCCESS;
  } else {
    printf("Expected Packets: %d\n", packets_remaining);
  }

  printf("Checking the transport buffer for a reply\n");

  int timeout = 2000;
  int packet_errno = -404;
  int expected_value = 0;
  clock_t begin;
  int total_packets = packets_remaining;
  struct packet* packets = (struct packet*) malloc (packets_remaining * sizeof(struct packet));

  begin = clock();

  while(packets_remaining > 0) {
    if (clock() - begin > timeout) {
      break;
    }

    // Scoop up the packet
    struct packet packet;
    int res = recv(sockfd, &packet, (int) sizeof(struct packet), MSG_CONFIRM);

    if (res == -1) {
      printf("Failed to receive packet\n");
      sleep(1/1000);
    } else {
      int num = packet.packet_number;
      printf("Got packet number: %d\n", num);

      //printf("---------- PACKET DATA ----------\n%s\n", packet.data);
      printf("Expected packet %d and got %d\n", expected_value, num);
      if (num == expected_value) {
        if (validate_checksum(packet) == 1) {
          printf("Sending ack number: %d\n", expected_value);

          printf("Data: \n%s\n", packet.data);
          printf("Size: %d\n", (int) sizeof(server));
          int req = sendto(sockfd, &expected_value, sizeof(int), 0, (struct sockaddr*) &server, sizeof(server));
          if (req == -1) {
            printf("Error in packet ack: %d sending\n", expected_value);
            sleep(1/1000);
          } else {
            printf("Saving to packet struct");
            printf("%d", expected_value);
            packets[expected_value] = packet;
            expected_value += 1;
            packets_remaining--;
          }
          begin = clock();
        } else {
          printf("Checksum invalid. Packet corruption detected, alerting the server\n");
          /* int checksum_res = sendto(sockfd, &packet_errno, sizeof(int), 0, (struct sockaddr*) &server, sizeof(server)); */
        }
      } else {
        // Packet has invalid checksum
        int req = sendto(sockfd, &packet_errno, sizeof(int), 0, (struct sockaddr*) &server, sizeof(server));

        if (req == -1) {
          printf("Error sending error response\n");
        }
      }
    }
  }

  printf("All packets have been received.\n");
  printf("Saving to: %s", request);

  FILE* fp;
  fp = fopen(request, "ae");

  if (check_order(total_packets, packets) == -1) {
    printf("Reordering packets...");
    reorder_packets(total_packets, packets);
  }

  for (int i = 0; i < total_packets; i++) {
    printf("%s", packets[i].data);
    fwrite(packets[i].data, sizeof(char), packets[i].size, fp);
  }

  fclose(fp);
  free(request);
  free(packets);
  close(sockfd);

  return EXIT_SUCCESS;
}
