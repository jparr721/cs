#include "./include/User.hpp"

namespace user {
User::User(std::string username) : username(username) {}

void User::set_username(std::string username) {
  this->username = username;
  return;
}

std::string User::get_username() {
  return this->username;
}

} // namespace user
