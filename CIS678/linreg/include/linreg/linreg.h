#ifndef LINREG_INCLUDE_LINREG_H
#define LINREG_INCLUDE_LINREG_H

#include <string>
#include <utility>
#include <vector>

class LinReg {
  public:
    LinReg(const std::string& filepath);
    std::pair<double, double> trendline(const std::vector<double>& x, const std::vector<double>& y);
    std::vector<double> polynomial_trendline(
        const std::vector<double>& x,
        const std::vector<double>& y,
        const int& order);
  private:
    std::vector<double> lines;
    bool read_infile(const std::string& file, std::vector<double>& lines);
};
#endif
