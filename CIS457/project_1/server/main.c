#include <arpa/inet.h>
#include <errno.h>
#include <math.h>
#include <netdb.h>
#include <signal.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>

// Data received size
#define PACKET_SIZE 1024

// Frame size is arbitrary
#define FRAME_SIZE 128

// Window size is arbitrary
#define WINDOW_SIZE 5

/** Function prototypes **/
off_t file_size(FILE** f);
FILE* init_fstream(char* location);
char* read_fild(long bytes, FILE** f);
struct packet* construct_packet_transport_queue(off_t size, FILE* file_ptr);

struct packet {
  // For tracking packet order...
  int packet_number;
  // Holds the contents of the packet
  char data[PACKET_SIZE];
};


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

struct packet* construct_packet_transport_queue(off_t size, FILE* file_ptr) {
    int num_packets = (size / PACKET_SIZE);

    // Set remaining packets to the number we have left
    int packets_remaining = num_packets;

    // Queue up and tag the next packet
    struct packet* send_queue = (struct packet*) malloc (WINDOW_SIZE * sizeof(struct packet));

    // The packet to be sent
    struct packet current_packet;

    while (packets_remaining > 0) {
      if (packets_remaining > 1) {
        int i;
        for(i = 0; i < WINDOW_SIZE; i++) {
          // Tag our packets
          current_packet.packet_number = i;
          fread(current_packet.data, sizeof(char), PACKET_SIZE, file_ptr);
          // Store the current packet to our packet queue
          send_queue[i] = current_packet;
        }
      } else { // Our final packet in the window
        // Add remaining data without overshooting and filling our files with trash...
        int i;
        for (i = 0; i < WINDOW_SIZE; i++) {// If we're just a little bit over...
          if (size - (i * PACKET_SIZE) > PACKET_SIZE) {
            current_packet.packet_number = i;
            fread(current_packet.data, sizeof(char), PACKET_SIZE, file_ptr);

            send_queue[i] = current_packet;
          } else {
            int diff = size - (num_packets * PACKET_SIZE);

            current_packet.packet_number = i;
            fread(current_packet.data, sizeof(char), diff, file_ptr);

            send_queue[i] = current_packet;
          }
        }
      }
      packets_remaining--;
    }

    return send_queue;
}

int main(int argc, char** argv) {
    if (argc != 2) {
        fprintf(stderr, "Invalid argument count. Usage: ./server port");
        return EXIT_FAILURE;
    }

    int sockfd = socket(AF_INET, SOCK_DGRAM, 0);
    int port = atoi(argv[1]);
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

      if (res == -1) {
          fprintf(stderr, "Error, could not receive data from client.");
      } else {
          fprintf(stdout, "Client file request: %s", cli_req);
      }

      // Trying to be efficient, so we open the file stream once, and then close it once.
      FILE* file_ptr = init_fstream(cli_req);

      // Acquire file size for proper splitting
      off_t size = file_size(&file_ptr);
      fprintf(stdout, "File Size: %li", size);

      // Store the data into a content buffer
      char* file_contents = read_file(size, &file_ptr);

      /**********************************
       * Begin sliding window transfer
       **********************************/

      // "Gatekeeper" that keeps the window from closing on the last packet
      int window_gate = 0;

      // Begin the sliding window transfer of data
      int num_packets = (size / PACKET_SIZE);

      // Set remaining packets to the number we have left
      int packets_remaining = num_packets;

      // Initialize our packets to be sent over the wire
      struct packet* packet_queue = construct_packet_transport_queue(size, file_ptr);

      for (int i = 0; i < sizeof(packet_queue); i++) {
        printf("%s", packet_queue[i].data);
      }

      int ack = 0;

      // Send number of packets to receive
      sendto(sockfd, &packets_remaining, sizeof(int), MSG_CONFIRM, (struct sockaddr*) &client, sizeof client);

      while (packets_remaining > 0) {
        int i;
        int packets_to_send = packets_remaining;
        socklen_t len = sizeof client;

        // Incrementally send each packet in the window
        for (i = 0; i < packets_to_send; i++) {
          sendto(sockfd, &packet_queue[i], sizeof(struct packet) + 1, MSG_CONFIRM, (struct sockaddr*) &client, len);
        }

        // Receive acks
        int received = recvfrom(sockfd, &ack, sizeof(int), MSG_WAITALL, (struct sockaddr*) &client, &len);
        if (received == -1) {
          fprintf(stderr, "Failed to receive ack reply from client");
          printf("Packet dropped");
        }

        if (ack < packets_to_send) {
          packets_remaining -= ack;
        } else {
          printf("\n\nnAll packets sent successfully");
          packets_remaining -= packets_to_send;
        }

        printf("Remaining packets: %d\n", packets_remaining);
      }

      fclose(file_ptr);
    }


    close(sockfd);

    return EXIT_SUCCESS;
}
