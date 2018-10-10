/* #include <router/Router.hpp> */

#include "../include/router/Router.hpp"
#include <iostream>

int main(int argc, char** argv) {
  router::Router r;
  int rt = r.Start();
  if (rt < 0) {
    std::cout << "router machine broke" << std::endl;
  }
}
