#include <linreg/linreg.h>

#include <algorithm>
#include <iostream>

int main(int argc, char** argv) {
  LinReg lr("downloads.txt");
  auto out = lr.trendline(lr.left, lr.right);
  std::cout << std::get<0>(out) << "x + " << std::get<1>(out) << std::endl;
  auto out2 = lr.polynomial_trendline(lr.left, lr.right, 3);
  std::for_each(out2.begin(), out2.end() - 1, [&](const double& v) {
    std::cout << v << " + ";
  });
  std::cout << out2[out2.size() - 1] << std::endl;

  return EXIT_SUCCESS;
}
