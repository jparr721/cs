#include "../include/router/Router.hpp"
#include <arpa/inet.h>
#include <cstring>
#include <ifaddrs.h>
#include <iostream>
#include <net/ethernet.h>
#include <netpacket/packet.h>
#include <sys/socket.h>
#include <sys/types.h>

int main(int argc, char** argv) {
  router::Router r;
  int router = r.Start();
	std::cout << router << std::endl;
  if (router < 0) {
    std::cerr << "Failed to intialize routing interface... " << router  << std::endl;
    return router;
  }

  return EXIT_SUCCESS;
}
