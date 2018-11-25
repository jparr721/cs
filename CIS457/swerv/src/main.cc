#include <swerv/core.h>

#include <iostream>

int main(int argc, char** argv) {
  swerver::Core server;
  int result = server.Run(argc, argv);
  return result;
}
