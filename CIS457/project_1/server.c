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
char* read_file(long bytes, FILE** f);
struct packet* make_packet(off_t size, FILE* file_ptr);

#pragma pack(1)

struct packet {
  int packet_number;
  unsigned int size;
  char data[PACKET_SIZE];
};

FILE* init_fstream(char* location) {
  FILE* fp;

  location[strlen(location) - 1] = 0;
  fp = fopen(location, "r");

  if (fp) {
    fprintf(stdout, "Found File [%s]\n", location);
  } else {
    fprintf(stderr, "Error, could not find file: %s\n", location);
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

int validate_args(int count) {
  if (count != 2) {
    fprintf(stderr, "Invalid argument count. Usage: ./server port\n");
    return -1;
  }

  return 0;
}

struct packet* make_packets(off_t size, FILE* file_ptr) {
  int num_packets = ceil(size / PACKET_SIZE) + 1;

  int offset = 0;
  int bytes_left = (int) size;
  struct packet* packets = (struct packet*) malloc (WINDOW_SIZE * sizeof(struct packet));

  struct packet current;
  //printf("Packet Total Size: %d", (int) sizeof(struct packet));
  for (int i = 0; i < num_packets; i++) {

    offset = (i * PACKET_SIZE);

    if (size - (i * PACKET_SIZE) > PACKET_SIZE) {
      current.packet_number = i;
      bytes_left = bytes_left - PACKET_SIZE;
      
      fseek(file_ptr, offset, SEEK_SET);
      fread(current.data, sizeof(char), PACKET_SIZE, file_ptr);
      current.size = PACKET_SIZE;
      current.data[PACKET_SIZE] = '\0';
      packets[i] = current;

    } else {
      int diff = size - offset;
      current.packet_number = i;

      bytes_left = bytes_left - diff;
      
      fseek(file_ptr, offset, SEEK_SET);
      fread(current.data, sizeof(char), diff, file_ptr);
      current.size = diff;
      current.data[diff] = '\0';
      //printf("Last String Length: %d\n", (int) strlen(current.data));
      packets[i] = current;
    }
    //printf("Offset: %d\n", offset);
    //printf("\n---------- PACKET DATA ----------\n%s\n", current.data);
  }
  return packets;
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

  int port = atoi(argv[1]);
  char cli_req[PACKET_SIZE];

  struct sockaddr_storage client;
  struct sockaddr_in server;

  memset(&server, 0, sizeof(server));
  memset(&client, 0, sizeof(client));

  server.sin_family = AF_INET;
  server.sin_addr.s_addr = INADDR_ANY;
  server.sin_port = htons(port);

  int bound = bind(sockfd, (struct sockaddr*) &server, sizeof(server));
  if (bound == -1) {
    fprintf(stderr, "Failed to bind to port\n");
    return EXIT_FAILURE;
  }

  struct packet* packets;

  /** Begin out server Loop **/
  while(1) {
    socklen_t len = sizeof client;

    int req = recvfrom(sockfd, (char*) cli_req, PACKET_SIZE, 0, (struct sockaddr*) &client, &len);
    if (req == -1) {
      fprintf(stderr, "Error, could not receive client request\n");
    }

    printf("\n--------- START SESSION --------\n");

    // insert nullbyte
    cli_req[req] = 0;
    printf("Client has made a file request: %s\n", cli_req);

    // Load pointer to file
    FILE* file_ptr = init_fstream(cli_req);

    if (file_ptr) {

      off_t size = file_size(&file_ptr);
      printf("File size is %li bytes\n", size);

      //Send number of incoming packets
      int num_packets = ceil(size / PACKET_SIZE) + 1;
      printf("Packets to send: %d\n", num_packets);
      sendto(sockfd, &num_packets, sizeof(int), 0, (struct sockaddr*) &client, sizeof client);

      packets = make_packets(size, file_ptr);

      // int ack = 0;
      for (int i = 0; i < num_packets; i++) {
        //printf("Sending data: %s\n", packets[i].data);
        sendto(sockfd, &packets[i], sizeof(struct packet), MSG_CONFIRM, (struct sockaddr*) &client, len);
      }
      
      printf("All packets were sent to the client.\n");
      fclose(file_ptr);
    } else {
      printf("File [%s]  was not found.\n", cli_req);
    }
    printf("\n---------- END SESSION ----------\n");
  }

  free(packets);
  return EXIT_SUCCESS;
}

