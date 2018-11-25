#include <swerv/core.h>

int main(int argc, char** argv) {
  swerver::Core server;
  int result = server.Run(argc, argv);
  return result;
}
