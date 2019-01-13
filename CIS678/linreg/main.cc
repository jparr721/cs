#include <linreg/linreg.h>

#include <algorithm>
#include <iostream>
#include <sstream>

int main(int argc, char** argv) {
  LinReg lr("downloads.txt");

  std::stringstream out_str;
  auto out = lr.trendline(lr.left, lr.right);
  out_str << std::get<0>(out) << " " << std::get<1>(out);
  std::cout << out_str.str() << std::endl;
  lr.write_to_file(out_str.str(), "linear.txt");

  std::stringstream out2_str;
  auto out2 = lr.polynomial_trendline(lr.left, lr.right, 3);
  std::for_each(out2.begin(), out2.end() - 1, [&](const double& v) {
    out2_str << v << " ";
  });
  std::cout << out2_str.str() << std::endl;
  lr.write_to_file(out2_str.str(), "polynomial.txt");

  return EXIT_SUCCESS;
}
