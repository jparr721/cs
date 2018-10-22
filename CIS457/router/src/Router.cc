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
#include <netinet/ip_icmp.h>
#include <sys/ioctl.h>
#include <sys/socket.h>
#include <sys/types.h>
#include <unistd.h>

namespace router {
  ARPHeader* Router::build_arp_reply(
      struct ether_header *eh,
      struct ether_arp *arp_frame,
      unsigned char local_addr[6]
      ) {
    ARPHeader *r = new ARPHeader();
    // SOURCE MAC FORMAT
    r->ea.ea_hdr.ar_hrd = htons(ARPHRD_ETHER);
    // SOURCE MAC LENGTH
    r->ea.ea_hdr.ar_hln = ETHER_ADDR_LEN;
    // TARGET MAC
    std::memcpy(r->ea.arp_tha, arp_frame->arp_sha, 6);
    // TARGET PROTOCOL
    std::memcpy(r->ea.arp_tpa, arp_frame->arp_spa, 4);
    // TARGET MAC
    std::memcpy(r->ea.arp_sha, local_addr, 6);
    // TARGET PROTOCOL ACCRESS
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

  bool Router::host_in_lookup_table(std::string host, std::unordered_map<std::string, std::string> lookup_table) {
    for (auto it = lookup_table.begin(); it != lookup_table.end(); ++it) {
      if (it->first.compare(host) == 0) {
        return true;
      }
    }

    return false;
  }

  ARPHeader* Router::build_arp_request(struct ether_header *eh, struct ether_arp *rp_frame, const unsigned char hop_ip[4]) {
    ARPHeader *r = new ARPHeader();
    // SOURCE MAC FORMAT
    r->ea.ea_hdr.ar_hrd = htons(ARPHRD_ETHER);
    // SET PROTOCOL
    r->ea.ea_hdr.ar_pro = htons(ETH_P_IP);

    // SET HARDWARE ADDRESS LENGTH
    r->ea.ea_hdr.ar_hln = ETHER_ADDR_LEN;
    // SET IP ADDRESS LENGTH
    r->ea.ea_hdr.ar_pln = sizeof(in_addr_t);
    // DELCARE AS AN ARP REQUEST
    r->ea.ea_hdr.ar_op = htons(ARPOP_REQUEST);

    uint8_t tmp_hw_addr[6];
    for (int i = 0; i < 6; ++i) {
      tmp_hw_addr[i] = 0;
    }

    // SET SOURCE HARDWARE ADDRESS
    std::memcpy(&r->ea.arp_sha, &rp_frame->arp_sha, 6);
    // SET TARGET HARDWARE PROTOCOL
    std::memcpy(&r->ea.arp_spa, &rp_frame->arp_spa, 4);
    // SET TARGET HARDWARE ADDRESS TO ALL ZERO
    std::memcpy(&r->ea.arp_tha, tmp_hw_addr, 6);
    // SET TARGET PROTOCOL ADDRESS AS THE HOP IP
    std::memcpy(&r->ea.arp_tpa, hop_ip, 4);
    // COPY ETHER HEADER
    std::memcpy(&r->eh, eh, sizeof(ether_header));
    // SETS ETHER TYPE TO ARP
    r->eh.ether_type = ntohs(2054);

   return r;
  }

  int Router::Start(std::string lookup) {
    // Load the relevant table to do lookup only for itself
    TableLookup router_lookup_table(lookup);

    int packet_socket;
    unsigned char local_addr[6];

    struct ifaddrs *ifaddr, *tmp;
    std::vector<int> interfaces;
    std::vector<unsigned char*> addresses;

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
          printf("Mac addr: ");
          for (int i = 0; i < 5; ++i)
            printf("%i:", local_addr[i]);

          printf("%i\n", local_addr[5]);

          packet_socket = socket(AF_PACKET, SOCK_RAW, htons(ETH_P_ALL));
          if (packet_socket < 0) {
            std::cerr << "socket machine broke [" << packet_socket << "]" << std::endl;
            return errno;
          }

          std::cout << "Created on descriptor " << packet_socket << std::endl;

          if (bind(packet_socket, tmp->ifa_addr, sizeof(struct sockaddr_ll)) == -1) {
            std::cerr << "bind machine broke" << std::endl;
          }
          interfaces.push_back(packet_socket);
          addresses.push_back(local_mac->sll_addr);
        }
      }
    }

    printf("Listening for packets on %d interfaces\n", interfaces.size());
    std::unordered_map<std::string, std::vector<std::string>> queue_map;

    while (1) {
      fd_set read_fds;
      FD_ZERO(&read_fds);
      int fd_max = 0;

      for (int i = 0; i < interfaces.size(); ++i) {
        if (interfaces[i] > fd_max) {
          fd_max = interfaces[i];
        }
        FD_SET(interfaces[i], &read_fds);
      }

      int activity = select(fd_max + 1, &read_fds, NULL, NULL, NULL);

      if (activity == -1) {
        printf("Unable to modify socket file descriptor.\n");
      }

      for (int i = 0; i < interfaces.size(); ++i) {
        if (FD_ISSET(interfaces[i], &read_fds)) {
          char buf[1500], send_buffer[1500];
          struct sockaddr_ll recvaddr;
          struct ether_header *eh_incoming, *eh_outgoing;
          struct ether_arp *arp_frame;
          ARPHeader *rp_incoming, *rp_outgoing;
          IPHeader *ip_incoming;
          IPHeader *ip_outgoing;
          ICMPHeader *icmp_incoming;
          ICMPHeader *icmp_outgoing;
          socklen_t recvaddrlen = sizeof(struct sockaddr_ll);

          int n = recvfrom(interfaces[i], buf, 1500, 0, (sockaddr*) &recvaddr, &recvaddrlen);
          if (n < 0) {
            std::cerr << "recvfrom machine broke" << std::endl;
          }

          if (recvaddr.sll_pkttype == PACKET_OUTGOING) continue;

          eh_incoming = (ether_header*) buf;
          rp_incoming = (ARPHeader*) (buf + sizeof(ether_header));
          ip_incoming = (IPHeader*) (buf + sizeof(ether_header));
          arp_frame = (ether_arp*) (buf + 14);

          printf("Incoming packet from %i.%i.%i.%i\n", ip_incoming->src_ip[0], ip_incoming->src_ip[1], ip_incoming->src_ip[2], ip_incoming->src_ip[3]);
          std::cout << "This packet is trying to reach: " << ip_incoming->dest_ip << std::endl;
          // Build the IP string for comparing later on
          std::string packet_ip = std::to_string(ip_incoming->dest_ip[0]) +"." +
            std::to_string(ip_incoming->dest_ip[1]) + "." +
            std::to_string(ip_incoming->dest_ip[2]) + "." +
            std::to_string(ip_incoming->dest_ip[3]);

          eh_incoming->ether_type = ntohs(eh_incoming->ether_type);

          //If ARP request handled, build an arp reply
          if (eh_incoming->ether_type == ETHERTYPE_ARP) {
            //std::cout << "Arp packet found" << std::endl;
            // Building arp reply here and storing into outgoing arp reply header
            rp_outgoing = build_arp_reply(eh_incoming, arp_frame, addresses[i]);
            std::memcpy(send_buffer, rp_outgoing, 1500);

            // Move data into Ethernet struct too
            eh_outgoing = (ether_header*) send_buffer;
            std::memcpy(eh_outgoing->ether_dhost, eh_incoming->ether_shost, 6);
            std::memcpy(eh_outgoing->ether_shost, addresses[i], 6);
            eh_outgoing->ether_type = htons(0x0806);

            // Send the damn thing
            std::cout << "Sending ARP reply" << std::endl;

            if(send(interfaces[i], send_buffer, 42, 0) == -1) {
              std::cout << "Error sending arp reply" << std::endl;
            }
          } else if (eh_incoming->ether_type == ETHERTYPE_IP) {
            std::cout << "IP/ICMP packet found" << std::endl;
            icmp_incoming = (ICMPHeader*) (buf + 34);
            if (host_in_lookup_table(std::string(reinterpret_cast<char*>(ip_incoming->dest_ip)), router_lookup_table.prefix_interface_table)) {
              std::cout << "The requested ip is in the lookup table, forwarding normally" << std::endl;
              // Add the normal forward code here
            } else {
              std::cout << "The requested ip is not in the prefix table, forwarding to the next router" << std::endl;
              rp_outgoing = build_arp_request(eh_incoming, arp_frame, reinterpret_cast<const unsigned char*>(router_lookup_table.hop_device_table.begin()->second.c_str()));
            }

            // Check if it's an ECHO packet
            /* if (icmp_incoming->type == ICMP_ECHO) { */
            /*   // First we build our ICMP goodies */
            /*   icmp_outgoing = (ICMPHeader*) (send_buffer + sizeof(ether_header) + sizeof(IPHeader)); */
            /*   icmp_outgoing->type = 0; */
            /*   icmp_outgoing->checksum = 0; */
            /*   icmp_outgoing->checksum = checksum(reinterpret_cast<unsigned char*>(icmp_outgoing), (1500 - sizeof(ether_header) - sizeof(IPHeader))); */

            /*   // Now we take the ip packet and put it into our outgoing ip packet. */
            /*   // This will enable us to add the new mac and reroute to the next router */
            /*   // (if needed). */
            /*   ip_outgoing = (IPHeader*) (send_buffer + sizeof(ether_header)); */
            /*   // Copy in the new ip address we want, which will by default be the host, */
            /*   // if it's not in lookup table though, we need to forward. */
            /*   if (host_in_lookup_table(std::string(reinterpret_cast<char*>(ip_incoming->dest_ip)), router_lookup_table.prefix_interface_table)) { */
            /*     // If our destination is in this set of hosts then forward normally */
            /*     std::memcpy(ip_outgoing->src_ip, ip_incoming->dest_ip, 4); */
            /*     std::memcpy(ip_outgoing->dest_ip, ip_incoming->src_ip, 4); */

            /*     // Now we build our new ethernet header since we are already on the */
            /*     // right router in this case */
            /*     std::cout << "Now building ICMP ethernet header" << std::endl; */
            /*     eh_outgoing = (ether_header*) send_buffer; */
            /*     std::memcpy(eh_outgoing->ether_dhost, eh_incoming->ether_shost, 6); */
            /*     std::memcpy(eh_outgoing->ether_shost, eh_incoming->ether_dhost, 6); */
            /*     eh_outgoing->ether_type = htons(0x800); */
            /*   } else { */
            /*     // Assign the router IP */
            /*     std::memcpy(ip_outgoing->src_ip, router_lookup_table.hop_device_table.begin()->first.c_str(), 4); */
            /*     std::memcpy(ip_outgoing->dest_ip, router_lookup_table.hop_device_table.begin()->second.c_str(), 4); */

            /*     // Now we build the new ethernet header from the mac address of the router */
            /*   } */
            /* } */
            // This is the old code, leaving it here breaks compilation FYI
            if (icmp_incoming->type == 8) {
              std::cout << "ICMP Echo request detected, beginning forward" << std::endl;

              std::memcpy(send_buffer, buf, 1500);

              // Copy data into the ICMP header
              icmp_outgoing = (ICMPHeader*) (send_buffer + sizeof(ether_header) + sizeof(IPHeader));
              icmp_outgoing->type = 0;
              icmp_outgoing->checksum = 0;
              icmp_outgoing->checksum = checksum(reinterpret_cast<unsigned char*>(icmp_outgoing), (1500 - sizeof(ether_header) - sizeof(IPHeader)));

              // Copy data into the IP heade0r
              ip_outgoing = (IPHeader*) (send_buffer + sizeof(ether_header));
              std::memcpy(ip_outgoing->src_ip, ip_incoming->dest_ip, 4);
              std::memcpy(ip_outgoing->dest_ip, ip_incoming->src_ip, 4);

              // Move data into the ether_header
              std::cout << "Building ICMP ethernet header" << std::endl;
              eh_outgoing = (ether_header*) send_buffer;
              std::memcpy(eh_outgoing->ether_dhost, eh_incoming->ether_shost, 6);
              std::memcpy(eh_outgoing->ether_shost, eh_incoming->ether_dhost, 6);
              eh_outgoing->ether_type = htons(0x800);
              std::string src_ip(reinterpret_cast<const char*>(ip_outgoing->src_ip), 6);
              std::cout << src_ip << std::endl;
              if (send(interfaces[i], send_buffer, n, 0) == -1) {
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
