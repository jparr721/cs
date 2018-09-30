#include <arpa/inet.h>
#include <errno.h>
#include <math.h>
#include <netdb.h>
#include <signal.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/types.h>
#include <time.h>
#include <unistd.h>

// Data received size
#define PACKET_SIZE 1024

// Frame size is arbitrary
#define FRAME_SIZE 128

// Window size is arbitrary
#define WINDOW_SIZE 5

// Our wait time to receive our ack packet
#define DURATION 500

// Our timeout time
#define TIMEOUT 2000

// Error packet (for corrupted packets)
#define PCK_ERR (-404)

// Regular packet
#define PCK_REG 1

/** Function prototypes **/
off_t file_size(FILE** f);
FILE* init_fstream(char* location);
char* read_file(long bytes, FILE** f);
struct packet* make_packet(off_t size, FILE* file_ptr);
unsigned short checksum(struct packet packet, int length);

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
  struct packet* packets = (struct packet*) malloc (num_packets * sizeof(struct packet));

  struct packet current;
  //printf("Packet Total Size: %d", (int) sizeof(struct packet));
  for (int i = 0; i < num_packets; i++) {

    offset = (i * PACKET_SIZE);
    printf("Making packet %d...\n", i);
    if (size - (i * PACKET_SIZE) > PACKET_SIZE) {
      current.packet_number = i;
      bytes_left = bytes_left - PACKET_SIZE;

      fseek(file_ptr, offset, SEEK_SET);
      fread(current.data, sizeof(char), PACKET_SIZE, file_ptr);
      current.size = PACKET_SIZE;
      current.data[PACKET_SIZE] = '\0';
			printf("Packet checksum params %d | %d\n", strlen(current.data), current.size);
      unsigned short checksum = make_checksum(current.data, current.size);
      current.type = PCK_REG;
      current.chk = checksum;
      packets[i] = current;

    } else {
      int diff = size - offset;
      current.packet_number = i;

      bytes_left = bytes_left - diff;

      fseek(file_ptr, offset, SEEK_SET);
      fread(current.data, sizeof(char), diff, file_ptr);
			current.size = diff;
			current.data[diff] = '\0';
      unsigned short checksum = make_checksum(current.data, strlen(current.data));
      current.chk = checksum;
      current.type = PCK_REG;
      //printf("Last String Length: %d\n", (int) strlen(current.data));i
      packets[i] = current;
			printf("Last Packet Checksum: %d | %d | %d\n", current.chk, current.size, (int) strlen(current.data));

    }
    //printf("Offset: %d\n", offset);
    //printf("\n---------- PACKET DATA ----------\n%s\n", current.data);
  }
  printf("Successfully made all packets.\n");
  return packets;
}

int set_window(int num_packets, int base) {
  if (WINDOW_SIZE > (num_packets - base)) {
    return num_packets - base;
  } else {
    return WINDOW_SIZE;
  }
}

int did_timeout(clock_t now) {
  if(clock() - now >= TIMEOUT) {
    return 1;
  }

  return 0;
}

int timer_running(clock_t now) {
  if (now == -1) {
    return 0;
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

      int next_frame = 0;
      int base = 0;
      int window = set_window(num_packets, base);

      packets = make_packets(size, file_ptr);

      while (base < num_packets) {
				printf("While Loop Params %d | %d | %d\n", next_frame, base, window);
        while (next_frame < base + window && next_frame < num_packets) {
          printf("Sending packet #: %d\n", next_frame);
          //printf("----------PACKET DATA---------\n%s\n", packets[next_frame].data);
          sendto(sockfd, &packets[next_frame], sizeof(struct packet), MSG_CONFIRM, (struct sockaddr*) &client, len);
          next_frame += 1;
        }

        clock_t time = clock();
        int ack = 0;

        while (timer_running(time) == 1 && did_timeout(time) != 1) {
          // Ack is expected to be the packet number

          int res = recvfrom(sockfd, &ack, sizeof(int), 0, (struct sockaddr*) &client, &len);
          printf("--- RESPONSE --- \n%d\n", ack);
          if (res == -1 || did_timeout(time) == 1) {
            printf("Error receiving ack number: %d from client\n", next_frame);
          } else {
            if (ack == PCK_ERR) {
            } else {
              printf("Got ack number: %d\n", ack);
              if (ack >= base) {
                base = ack + 1;
                time = -1;
              }
            }
          }
        }
        printf("Timeout: %d | %d\n", did_timeout(time), (int) time);
        if (did_timeout(time) == 1) {
          time = -1;
          next_frame = base;
        } else {
          printf("Shifting window\n");
          window = set_window(num_packets, base);
        }

      }

      printf("All packets were sent to the client.\n");
      fclose(file_ptr);
    } else {
      int response = -1;
      sendto(sockfd, &response, sizeof(struct packet), MSG_CONFIRM, (struct sockaddr*) &client, len);
      printf("File [%s]  was not found.\n", cli_req);
    }
    printf("\n---------- END SESSION ----------\n");
  }

  free(packets);
  return EXIT_SUCCESS;
}

