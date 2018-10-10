/* #include <router/Router.hpp> */
#include "../include/router/Router.hpp"

#include <arpa/inet.h>
#include <errno.h>
#include <iostream>
#include <net/ethernet.h>
#include <netpacket/packet.h>
#include <sys/socket.h>
#include <sys/types.h>
#include <string>
#include <unistd.h>

namespace router {
  int Router::Start() {
    return 1;
  }
} // namespace router
