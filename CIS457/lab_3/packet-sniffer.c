#include <arpa/inet.h>
#include <errno.h>
#include <netdb.h>
#include <net/ethernet.h>
#include <netinet/in.h>
#include <netinet/ether.h>
#include <netpacket/packet.h>
#include <net/if.h>
#include <signal.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/socket.h>
#include <sys/stat.h>
#include <sys/types.h>
#include <unistd.h>

int main(int argc, char** argv) {
  int packet_socket = socket(AF_PACKET, SOCK_RAW, htons(ETH_P_ALL));

  if (packet_socket < 0) {
    perror("socket");
    return EXIT_FAILURE;
  }

  struct sockaddr_ll server, client;
  server.sll_family=AF_PACKET;
  server.sll_protocol=htons(ETH_P_ALL);
  server.sll_ifindex=if_nametoindex("wlp2s0");

  if (bind(packet_socket, (struct sockaddr*) &server, sizeof(server)) < 0) {
    perror("bind");
    return EXIT_FAILURE;
  }

  while(1) {
    char buf[1514];
    socklen_t len = sizeof(client);
    int res = recvfrom(packet_socket, buf, sizeof(buf), 0, (struct sockaddr*) &client, &len);

    if (client.sll_pkttype == PACKET_OUTGOING) {
      continue;
    }

    printf("Received a %d byte packet, first byte is %02hhx\n", res, buf[0]);
  }

  return 0;
}
