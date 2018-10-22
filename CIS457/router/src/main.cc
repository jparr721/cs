#include "../include/router/Router.hpp"
#include <arpa/inet.h>
#include <cstring>
#include <ifaddrs.h>
#include <iostream>
#include <net/ethernet.h>
#include <netpacket/packet.h>
#include <string>
#include <sys/socket.h>
#include <sys/types.h>

int main(int argc, char** argv) {
  if (argc < 2) {
    std::cerr << "usage: router router_table" << std::endl;
    return EXIT_FAILURE;
  }
  std::string lookup_table = argv[1];

  router::Router r;
  int router = r.Start(lookup_table);
	std::cout << router << std::endl;
  if (router < 0) {
    std::cerr << "Failed to intialize routing interface... " << router  << std::endl;
    return router;
  }

  return EXIT_SUCCESS;
}
