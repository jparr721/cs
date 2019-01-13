#include <linreg/linreg.h>

#include <iostream>

int main(int argc, char** argv) {
  LinReg lr("downloads.txt");
  auto out = lr.trendline(lr.left, lr.right);
  std::cout << std::get<0>(out) << "x + " << std::get<1>(out) << std::endl;

  return EXIT_SUCCESS;
}
