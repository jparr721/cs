#include <stdio.h>
#include <string.h>
#include <unistd.h>
#include <stdlib.h>
#include <sys/socket.h>
#include <netinet/in.h>
#include <arpa/inet.h>

#define PACKET_SIZE 1024

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

    sendto(sockfd, request, strlen(request), MSG_CONFIRM, (struct sockaddr*) &server, sizeof(server));

    char response[PACKET_SIZE];

    close(sockfd);
}
