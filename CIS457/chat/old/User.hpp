#ifndef CHAT_USER_HPP
#define CHAT_USER_HPP

#include <string>

namespace user {
class User {
  public:
    explicit User(std::string username);
    void set_username(std::string username);
    std::string get_username();
  private:
    std::string username;
};
} // namespace user

#endif
