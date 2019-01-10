#include <iostream>
#include <cstdlib>
#include <cstring>
#include <termios.h>

int main() {
  std::cout << "Disabling echo." << std::endl;
  termios t;
  tcgetattr(1, &t);
  termios tt = t;
  tt.c_lflag &= ~ECHO;
  tcsetattr(1, TCSANOW, &tt);

  std::string s;
  std::cout << "Enter secret word/phrase: ";
  std::getline(std::cin, s);
  std::cout << "\nYou Entered: " << s << std::endl;

  tcsetattr(1, TCSANOW, &t);
  std::cout << "Default behavior restored." << std::endl;
  std::cout << "Enter visible word/phrase: ";
  std::getline(std::cin, s);

  return EXIT_SUCCESS;
}
