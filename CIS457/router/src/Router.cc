#include "../include/router/Router.hpp"
#include "../include/router/TableLookup.hpp"
#include "../include/router/ARPHeader.hpp"
#include "../include/router/ETHHeader.hpp"
#include "../include/router/ICMPHeader.hpp"
#include "../include/router/IPHeader.hpp"

#include <arpa/inet.h>
#include <cerrno>
#include <cstring>
#include <iostream>
#include <linux/if_packet.h>
#include <net/ethernet.h>
#include <netinet/if_ether.h>
#include <sys/ioctl.h>
#include <sys/socket.h>
#include <unistd.h>

namespace router {
  ARPHeader* Router::build_arp_reply(
      struct ether_header *eh,
      struct ether_arp *arp_frame,
      uint8_t destination_mac
      ) {
    ARPHeader *r = new ARPHeader();
    // SOURCE MAC FORMAT
    r->ea.ea_hdr.ar_hrd = htons(ARPHRD_ETHER);
    // SOURCE MAC LENGTH
    r->ea.ea_hdr.ar_hln = ETHER_ADDR_LEN;
    // TARGET MAC
    std::memcpy(r->ea.arp_tha, &arp_frame->arp_sha, 6);
    // TARGET PROTOCOL
    std::memcpy(r->ea.arp_tpa, &arp_frame->arp_spa, 4);
    // SOURCE MAC ADDRESS
    std::memcpy(static_cast<uint8_t*>(r->ea.arp_sha), &destination_mac, 6);
		// SOURCE PROTOCOL ACCRESS
		std::memcpy(r->ea.arp_spa, &arp_frame->arp_tpa, 4);
    // PROTOCOL
    r->ea.ea_hdr.ar_pro = htons(ETH_P_IP);
    // PROTOCOL LENGTH
    r->ea.ea_hdr.ar_pln = sizeof(in_addr_t);
    // OP
    r->ea.ea_hdr.ar_op = htons(ARPOP_REPLY);
    // ETHERNET HEADER
    r->eh = *eh;
    return r;
  }

  ARPHeader* Router::build_arp_request(
      struct ether_header *eh,
      struct ether_arp *arp_fame,
      uint8_t hop_ip
      ){
    ARPHeader *r;
    // SOURCE MAC FORMAT
    r->ea.ea_hdr.ar_hrd = htons(ARPHRD_ETHER);
    // PROTOCOL
    r->ea.ea_hdr.ar_pro = htons(ETH_P_IP);
    // SOURCE MAC LENGTH
    r->ea.ea_hdr.ar_hln = ETHER_ADDR_LEN;
    // SOURCE PROTOCOL LENGTH
    r->ea.ea_hdr.ar_pln = sizeof(in_addr_t);
    // OP
    r->ea.ea_hdr.ar_op = htons(ARPOP_REQUEST);
    // ETHERNET HEADER
    r->eh = *eh;

    return r;
  }

  // Adapted from here: http://www.microhowto.info/howto/get_the_ip_address_of_a_network_interface_in_c_using_siocgifaddr.html
  uint8_t router::Router::get_dest_mac(
    struct ifaddrs *ifaddr,
    struct ifaddrs *tmp,
    uint8_t *arp_tpa,
    int socket
    ){
    struct ifreq ifr;
    uint8_t destination_mac_addr[6];

    for (tmp = ifaddr; tmp != nullptr; tmp = tmp->ifa_next) {
      if (tmp->ifa_addr->sa_family == AF_INET) {
        struct sockaddr_in* sa = (struct sockaddr_in*) tmp->ifa_addr;
        // Extract IP of captured packet
        char* ip_addr = inet_ntoa(sa->sin_addr);
        char arp_addr[50];
        // Build arp target IP to compare
        sprintf(arp_addr, "%d.%d.%d.%d", arp_tpa[0], arp_tpa[1], arp_tpa[2], arp_tpa[3]);

        if (std::strcmp(ip_addr, arp_addr) == 0) {
          // prepare our ifreq struct to use ioctl
          ifr.ifr_addr.sa_family = AF_INET;
          strcpy(ifr.ifr_name, tmp->ifa_name);
          if (ioctl(socket, SIOCGIFADDR, &ifr) == -1) {
            std::cerr << "Failed to run ioctl" << std::endl;
            return -1;
          }
          std::memcpy(destination_mac_addr, ifr.ifr_hwaddr.sa_data, 6);
        }
      }
    }

    return *destination_mac_addr;
  }

  /*
 * Checksum calculation.
 * Taken from: https://github.com/kohler/ipsumdump/blob/master/libclick-2.1/libsrc/in_cksum.c
 */
  uint16_t Router::checksum(unsigned char *addr, int len) {
      int nleft = len;
      const uint16_t *w = (const uint16_t *)addr;
      uint32_t sum = 0;
      uint16_t answer = 0;

      /*
       * Our algorithm is simple, using a 32 bit accumulator (sum), we add
       * sequential 16 bit words to it, and at the end, fold back all the
       * carry bits from the top 16 bits into the lower 16 bits.
       */
      while (nleft > 1)  {
    sum += *w++;
    nleft -= 2;
      }

      /* mop up an odd byte, if necessary */
      if (nleft == 1) {
    *(unsigned char *)(&answer) = *(const unsigned char *)w ;
    sum += answer;
      }

      /* add back carry outs from top 16 bits to low 16 bits */
      sum = (sum & 0xffff) + (sum >> 16);
      sum += (sum >> 16);
      /* guaranteed now that the lower 16 bits of sum are correct */

      answer = ~sum;              /* truncate to 16 bits */
      return answer;
  }

  uint8_t router::Router::get_src_mac(
    struct ifaddrs *ifaddr,
    struct ifaddrs *tmp,
    uint8_t if_ip,
    int socket,
    uint8_t destination_mac
    ){
    return destination_mac;
  }

  int Router::Start(std::string routing_table) {
    TableLookup routeTable(routing_table);

		int packet_socket;
    uint8_t local_addr[6];

    struct ifaddrs *ifaddr, *tmp;
    fd_set interfaces;
    FD_ZERO(&interfaces);

    if (getifaddrs(&ifaddr) == -1) {
      std::cerr << "getifaddrs machine broke" << std::endl;
      return EXIT_FAILURE;
    }

    for (tmp = ifaddr; tmp != nullptr; tmp = tmp->ifa_next) {
      if (tmp->ifa_addr->sa_family == AF_PACKET) {
        std::cout << "Interface: " << tmp->ifa_name << std::endl;

        if (!strncmp(&(tmp->ifa_name[3]), "eth", 3)) {
          std::cout << "Creating socket on interface: " << tmp->ifa_name << std::endl;

          //Get our mac
          struct sockaddr_ll *local_mac = (struct sockaddr_ll*) tmp->ifa_addr;
          std::memcpy(local_addr, local_mac->sll_addr, 6);
					std::cout << local_mac->sll_addr << std::endl;
          std::cout << "Mac addr: ";
          for (int i = 0; i < 5; ++i)
            std::cout << local_addr[i] << ":";

          std::cout << local_addr[5] << std::endl;

          packet_socket = socket(AF_PACKET, SOCK_RAW, htons(ETH_P_ALL));
          if (packet_socket < 0) {
            std::cerr << "socket machine broke [" << packet_socket << "]" << std::endl;
            return errno;
          }

          if (bind(packet_socket, tmp->ifa_addr, sizeof(struct sockaddr_ll)) == -1) {
            std::cerr << "bind machine broke" << std::endl;
          }
          listen(packet_socket, 10);
          FD_SET(packet_socket, &interfaces);
        }
      }
    }

    std::cout << "My body is ready" << std::endl;

    while (1) {
      char buf[1500], send_buffer[1500];
      struct sockaddr_ll recvaddr;
      struct ether_header *eh_incoming, *eh_outgoing;
      struct ether_arp *arp_frame;
      ARPHeader *rp_incoming, *rp_outgoing;
      IPHeader *ip_incoming;
			IPHeader *ip_outgoing = new IPHeader();
      ICMPHeader *icmp_incoming;
		  ICMPHeader *icmp_outgoing = new ICMPHeader();
      socklen_t recvaddrlen = sizeof(struct sockaddr_ll);

      fd_set tmp_set = interfaces;
      select(FD_SETSIZE, &tmp_set, NULL, NULL, NULL);

      for (int i = 0; i < FD_SETSIZE; ++i) {
        if (FD_ISSET(i, &tmp_set)) {
          int n = recvfrom(packet_socket, buf, 1500, 0, (sockaddr*) &recvaddr, &recvaddrlen);
          if (n < 0) {
            std::cerr << "recvfrom machine broke" << std::endl;
          }

          if (recvaddr.sll_pkttype == PACKET_OUTGOING) continue;

          std::cout << "Got a " << n << " byte packet" << std::endl;

          eh_incoming = (ether_header*) buf;
          rp_incoming = (ARPHeader*) (buf + sizeof(ether_header));
          ip_incoming = (IPHeader*) (buf + sizeof(ether_header));
          arp_frame = (ether_arp*) (buf + 14);

          eh_incoming->ether_type = ntohs(eh_incoming->ether_type);
          std::cout << "Type: " << eh_incoming->ether_type << std::endl;

          //If ARP request handled, build an arp reply
          if (eh_incoming->ether_type == ETHERTYPE_ARP) {
            std::cout << "Arp packet found" << std::endl;
            // Building arp reply here and storing into outgoing arp reply header
            rp_outgoing = build_arp_reply(eh_incoming, arp_frame, htons(1));
						std::memcpy(send_buffer, rp_outgoing, 1500);

            // Move data into Ethernet struct too
            std::cout << "Making ethernet header" << std::endl;
            eh_outgoing = (ether_header*) send_buffer;
            std::memcpy(eh_outgoing->ether_dhost, eh_incoming->ether_shost, 6);
            std::memcpy(eh_outgoing->ether_shost, eh_incoming->ether_dhost, 6);
            eh_outgoing->ether_type = htons(0x0806);

            // Send the damn thing
            std::cout << "Sending ARP reply" << std::endl;
						std::cout << eh_outgoing->ether_dhost << std::endl;
						std::cout << eh_outgoing->ether_shost << std::endl;
            if(send(i, send_buffer, 42, 0) == -1) {
              std::cout << "Error sending arp reply" << std::endl;
            }
          } else if (eh_incoming->ether_type == ETHERTYPE_IP) {
            std::cout << "IP/ICMP packet found" << std::endl;
            icmp_incoming = (ICMPHeader*) (buf + 34);
						std::cout << "IP/ICMP Type: " << icmp_incoming->type << std::endl;
            if (icmp_incoming->type == 8) {
              std::cout << "ICMP Echo request detected" << std::endl;

              std::memcpy(send_buffer, buf, 1500);

              // Copy data into the ICMP header
              std::cout << "Building the ICMP header" << std::endl;
              icmp_outgoing = (ICMPHeader*) (send_buffer + sizeof(ether_header) + sizeof(IPHeader));
              icmp_outgoing->type = 0;
              icmp_outgoing->checksum = 0;
              icmp_outgoing->checksum = checksum(reinterpret_cast<unsigned char*>(icmp_outgoing), (1500 - sizeof(ether_header) - sizeof(IPHeader)));

              // Copy data into the IP header
              ip_outgoing = reinterpret_cast<IPHeader*>(send_buffer + sizeof(ether_header));
              std::memcpy(ip_outgoing->src_ip, ip_incoming->dest_ip, 4);
              std::memcpy(ip_outgoing->dest_ip, ip_incoming->src_ip, 4);

              // Move data into the ether_header
              std::cout << "Building ICMP ethernet header" << std::endl;
              eh_outgoing = reinterpret_cast<ether_header*>(send_buffer);
              std::memcpy(eh_outgoing->ether_dhost, eh_incoming->ether_shost, 6);
              std::memcpy(eh_outgoing->ether_shost, eh_incoming->ether_dhost, 6);
              eh_outgoing->ether_type = htons(0x800);

              std::cout << "Sending ICMP response" << std::endl;
              if (send(i, send_buffer, n, 0) == -1) {
                std::cout << "There was an error sending the ICMP echo packet" << std::endl;
              }
            }
          }
        }
      }
    }

    freeifaddrs(ifaddr);
    return EXIT_SUCCESS;
  }
} // namespace router
