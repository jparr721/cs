/* #include <router/Router.hpp> */

#include "../include/router/Router.hpp"
#include <iostream>

int main(int argc, char** argv) {
  if (argc < 2) {
    std::cout << "usage: router routing_table" << std::endl;
  }
  router::Router r;
  int rt = r.Start();
  if (rt < 0) {
    std::cout << "router machine broke" << std::endl;
  }
}
