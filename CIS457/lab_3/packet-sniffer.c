#include <arpa/inet.h>
#include <errno.h>
#include <netdb.h>
#include <net/ethernet.h>
#include <netinet/in.h>
#include <netinet/ether.h>
#include <netinet/ip.h>
#include <netinet/tcp.h>
#include <netinet/udp.h>
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

void print_ip_header(unsigned char* buf, int size) {
  struct sockaddr_in source, destination;

  struct iphdr *iph = (struct iphdr *)buf;
  unsigned short ip_header_len = iph->ihl*4;

  memset(&source, 0, sizeof(source));
  source.sin_addr.s_addr = iph->saddr;

  memset(&destination, 0, sizeof(destination));
  destination.sin_addr.s_addr = iph->daddr;

  printf("|- Source IP: %s\n", inet_ntoa(source.sin_addr));
  printf("|- Destination IP: %s\n", inet_ntoa(destination.sin_addr));
}

void parse_ethernet_header(unsigned char* buf, int size) {
  struct ethhdr *eth = (struct ethhdr*) buf;

  printf("Ethernet Header\n");
  printf("|- Destination %.2X-%.2X-%.2X-%.2X-%.2X-%.2X \n", eth->h_dest[0], eth->h_dest[1], eth->h_dest[2], eth->h_dest[3], eth->h_dest[4], eth->h_dest[5]);
  printf("|- Source Address %.2X-%.2X-%.2X-%.2X-%.2X-%.2X \n", eth->h_source[0], eth->h_source[1], eth->h_source[2], eth->h_source[3], eth->h_source[4], eth->h_source[5]);
  printf("|- Protocol %u\n", (unsigned short)eth->h_proto);
}

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
    print_ip_header(buf, len);
    parse_ethernet_header(buf, len);
  }

  return 0;
}
