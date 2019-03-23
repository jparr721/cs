#include "slam.h"

int main(int argc, char** argv) {
  slam::Slam slammy;

  slammy.process_frame();

  return EXIT_SUCCESS;
}
