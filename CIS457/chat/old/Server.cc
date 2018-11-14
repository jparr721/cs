#include "./include/Server.hpp"
#include "./include/Crypto.hpp"

#include <cstdlib>
#include <cstring>
#include <iostream>
#include <cstdio>
#include <fstream>
#include <sstream>
#include <iostream>
#include <unistd.h>

namespace server {
  std::pair<std::string, int> Server::encrypt_string(std::string input, unsigned char key[32]) {
    yep::Crypto crypto;
    unsigned char* plaintext = const_cast<unsigned char*>(
        reinterpret_cast<const unsigned char*>(input.c_str()));
    unsigned char miv[16];
    unsigned char cipher[1024];
    int cipher_len = crypto.encrypt(plaintext, input.size(), key, miv, cipher);

    std::string str(cipher, cipher + sizeof cipher / sizeof cipher[0] );

    std::pair<std::string, int> result;

    result.first = str;
    result.second = cipher_len;

    return result;
  }

  bool Server::check_admin(const std::string& pass) {
    return pass == ADMIN_PASSWORD;
  }

  void* handler(void* args) {
    yep::Crypto crypto;
    Server::std_msg s;
    Server::thread t;

    RAND_bytes(s.iv, 16);
    std::memcpy(&t, args, sizeof(Server::thread));

    socklen_t sin_size = sizeof(t.client);

    for (;;) {
      char data[4096];
      int r = recv(t.socket, data, 4096, 0);
      if (r < 0) {
        std::cerr << "Failed to receive data from user: " <<
          t.username << ", exiting" << std::endl;
        break;
      }

      if (std::string(data) == "/quit" || r <= 0) {
        std::cout  << "Shutting down the server for user: " <<
          t.username << std::endl;
        break;
      }

      std::string message =std::string(data);
      unsigned char* cipher = reinterpret_cast<unsigned char*>(data);

      unsigned char miv[16];
      unsigned char decrypted_message[1024];

      // Fake 16 bit miv
      int fake = 0x10;

      if (message.size() > fake) {
        fake = message.size();
      }

      int message_len = crypto.decrypt(
          cipher,
          fake,
          t.key,
          miv,
          decrypted_message);

      std::cout << " <<< " << t.username << ": " << decrypted_message << std::endl;
      message = std::string(reinterpret_cast<char*>(decrypted_message));
      std::string command = t.instance->extract_command(message);
      std::cout << "Command:(" << command << ")" << std::endl;

      std::vector<std::string> command_args;
      std::istringstream iss(message);

    for(std::string val; iss >> message; ) {
        command_args.push_back(val);
      }

      std::cout << "Args: " << command_args.size() << std::endl;

      std::string outgoing;

      if (command.substr(0, 4) == "list") {
        std::string outlist;

        for (int i = 0; i < t.instance->users.size(); ++i) {
          outlist = outlist + "\n" + t.instance->users[i].username;
        }

        std::pair<std::string, int> out = t.instance->encrypt_string(outlist, t.key);
        send(t.socket, out.first.c_str(), out.second, 0);

      } else if (command == "/broadcast") {
        std::cout << "Broadcasting to all users" << std::endl;

        for (int i = 0; i < t.instance->users.size(); ++i) {
          std::cout << "Sending to: " << t.instance->users[i].username << ": " << command_args[1] << std::endl;
          std::pair<std::string, int> out = t.instance->encrypt_string(command_args[1], t.instance->users[i].key);
          send(t.instance->users[i].socket, out.first.c_str(), out.second, 0);
        }
      } else if (command == "/pm") {
        std::cout << "Sending personal message" << std::endl;

        for (int i = 0; i < t.instance->users.size(); ++i) {
          std::cout << "1: " << t.instance->users[i].username << std::endl;
          std::cout << "2: " << command_args[1] << std::endl;
          if (t.instance->users[i].username == command_args[1]) {
            std::pair<std::string, int> out = t.instance->encrypt_string(command_args[2], t.instance->users[1].key);
            send(t.instance->users[i].socket, out.first.c_str(), out.second, 00);
          }
        }
      } else if (command == "/kick") {
        std::string pass;

        if (t.instance->check_admin(command_args[2])) {
          std::string who = command_args[1];
          std::string kick_msg("kicked");

          for (int i = 0; i < t.instance->users.size(); ++i) {
            if (t.instance->users[i].username == who) {
              std::cout << "Bye Felicia!" << std::endl;
              Server::std_msg sm;
              send(t.instance->users[i].socket, kick_msg.c_str(), kick_msg.size(), 0);
              t.instance->users.erase(t.instance->users.begin() + i);
            }
          }
        }
      } else {
        std::cout << "Access denied" << std::endl;
      }
    }

    std::cout << "Closing handler thread" << std::endl;
    return nullptr;
  }

  int Server::Run() {
    return EXIT_SUCCESS;
  }

  } // namespace server
