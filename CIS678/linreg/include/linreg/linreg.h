#ifndef LINREG_INCLUDE_LINREG_H
#define LINREG_INCLUDE_LINREG_H

#include <string>
#include <utility>
#include <vector>

class LinReg {
  public:
    std::vector<double> left;
    std::vector<double> right;

    LinReg(const std::string& filepath);
    std::pair<double, double> trendline(const std::vector<double>& x, const std::vector<double>& y);
    std::vector<double> polynomial_trendline(
        const std::vector<double>& x,
        const std::vector<double>& y,
        const int& terms);

    void print_vectors();
  private:
    bool read_infile(
        const std::string& file,
        std::vector<double>& left,
        std::vector<double>& right);
};
#endif
