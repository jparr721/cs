#include <router/Router.hpp>
#include <arpa/inet.h>
#include <cstring>
#include <ifaddrs.h>
#include <iostream>
#include <net/ethernet.h>
#include <netpacket/packet.h>
#include <sys/socket.h>
#include <sys/types.h>

int main(int argc, char** argv) {
  if (argc < 2) {
    std::cout << "usage: router routing_table" << std::endl;
  }
  std::string routing_table = argv[1];
  router::Router r;
  int router = r.Start(routing_table);
  if (router < 0) {
    std::cerr << "Failed to intialize routing interface..." << std::endl;
    return EXIT_FAILURE;
  }

  return EXIT_SUCCESS;
}
