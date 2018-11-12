#include "./include/ChatServer.hpp"

#include "./include/Succ.hpp"
#include "./include/Crypto.hpp"
#include <cstdlib>
#include <cstring>
#include <cstdio>
#include <fstream>
#include <sstream>
#include <iostream>
#include <openssl/conf.h>
#include <openssl/err.h>
#include <openssl/evp.h>
#include <openssl/pem.h>
#include <openssl/rand.h>
#include <sys/socket.h>
#include <tuple>
#include <unistd.h>


namespace server {

//std::vector<ChatServer::thread> ChatServer::users;

void* ChatServer::server_handler(void* args) {
  the::Succ SKOOMA_HIGH;
  ChatServer::std_message s;
  ChatServer::thread t;

  std::memcpy(&t, args, sizeof(ChatServer::thread));

  socklen_t sin_size = sizeof(t.client);

  while (true) {
    char data[4096];
    std::memset(data, 0, sizeof(data));
    int r = recv(t.socket, data, 4096, 0);


    if (r < 0) {
      std::cout << "I'VE GOTTEN ALL THE WAY GERE" << std::endl;
      break;
    }

    if (std::string(data) == "/quit" || r <= 0) {
      std::cout << "Shutting down the server connection to user: " << t.username << std::endl;
      //close(t.socket);
      // Break this worker
      break;
    }

    std::string message = std::string(data);
    //std::string message = SKOOMA_HIGH.decrypt(t.key, s.iv, s.cipher);
    std::cout << " <<< " << t.username << ": " << message << std::endl;

    std::string command = t.instance->extract_command(message);

    std::cout << "this the command:(" + command + ")" << std::endl;

    //std::cout << commands_args.size()
    std::vector<std::string> command_args;
    std::istringstream iss(message);
    for(std::string message; iss >> message; ) {
      command_args.push_back(message);
    }
    std::cout << "args: " << std::endl;
    std::cout << command_args.size() << std::endl;


    if (command == "list") {
      std::string outlist = "";
      for (int i = 0; i < t.instance->users.size(); i++) {
        outlist = outlist + "\n" + t.instance->users[i].username;
      }
      send(t.socket, outlist.c_str(), outlist.length(), 0);
    } else if (command == "broadcast") {
      std::cout << "we broadcasting" << std::endl;
      for (int i = 0; i < t.instance->users.size(); ++i) {
        std::cout << "sending to " + t.instance->users[i].username + ": " + command_args[1] << std::endl;
        send(t.instance->users[i].socket, command_args[1].c_str(), command_args[1].length(), 0);
      }
    } else if (command == "pm") {
      std::cout << "we pming" << std::endl;
      for (int i = 0; i < t.instance->users.size(); ++i) {
        std::cout << "username1 = " + t.instance->users[i].username << std::endl;
        std::cout << "username2 = " + command_args[1] << std::endl;
        if (t.instance->users[i].username == command_args[1])
          send(t.instance->users[i].socket, command_args[2].c_str(), command_args[2].length(), 0);
      }
    } else if (command == "kick") {
      std::string pass = "";

      if (t.instance->check_admin(command_args[2])) {
        std::string who = command_args[1];

        std::string kick_msg("kicked");

        for (int i = 0; i < t.instance->users.size(); ++i) {
          if (t.instance->users[i].username == who) {
            std::cout << "Bye Felicia!" << std::endl;
            // Encrypt our kick message
            ChatServer::std_message sm;
            //RAND_pseudo_bytes(sm.iv, 16);
            //sm.cipher = SKOOMA_HIGH.encrypt(cs.getUsers()[i].key, sm.iv, kick_msg);
            send(t.instance->users[i].socket, kick_msg.c_str(), kick_msg.length(), 0);
            t.instance->users.erase(t.instance->users.begin() + i);
          }
        }
      } else {
        std::cout << "access denied" << std::endl;
      }
    }
  }
  std::cout << "Closing handler thread" << std::endl;
  return NULL;
}

bool ChatServer::check_admin(const std::string& pass) {
  return pass == admin_password;
}

int ChatServer::handle_port() {
  std::cout << "Please enter the port for the server: " << std::flush;
  std::string port = "";
  std::getline(std::cin, port);

  return std::stoi(port);
}

std::string ChatServer::handle_input(std::string prompt = " >>> ") {
  std::cout << prompt << std::flush;
  std::string message = "";
  std::getline(std::cin, message);

  return message;
}

void ChatServer::broadcast(const std::string& message) {
  the::Succ succ;
  for (const auto &user : this->users) {
    ChatServer::std_message outgoing;
    RAND_pseudo_bytes(outgoing.iv, 16);
    outgoing.cipher = succ.encrypt(
        const_cast<unsigned char*>(reinterpret_cast<const unsigned char*>(user.key)),
        outgoing.iv,
        message);

    send(user.socket, message.c_str(), sizeof(ChatServer::std_message), 0);
  }
}

void ChatServer::list_users() {

  std::cout << "Listing users..." << std::endl;
  if (!this->users.empty()) {
    for (const auto &user : this->users) {
      std::cout << user.username << std::endl;
    }

    return;
  }

  std::cout << "No clients found." << std::endl;
}

std::string ChatServer::extract_command(const std::string& input) const {
  std::string command = "no_command";
  for (const auto v : input) {
    if  (v == '/') {
      // Strips everything before the command delimeter
      command = input.substr(input.find('/') + 1, input.find(' ')-1);
      return command;
    }
    return command;
  }
}

int ChatServer::RunServer() {
  the::Succ sad;
  yep::Crypto JEFF;
  int sock = socket(AF_INET, SOCK_STREAM, 0);

  sockaddr_in addr;
  addr.sin_family = AF_INET;
  addr.sin_addr.s_addr = INADDR_ANY;
  addr.sin_port = htons(handle_port());

  if (bind(sock, (struct sockaddr *)&addr, sizeof(addr)) < 0) {
    std::cout << "Failed to bind socket." << std::endl;
    return EXIT_FAILURE;
  }

  // listen for a new client connection
  listen(sock, 10);

  while (true) {
    sockaddr_in client;
    socklen_t sin_size = sizeof(client);
    std::cout << "Waiting for connection..." << std::endl;
    int clientsocket = accept(sock, reinterpret_cast<sockaddr*>(&client), &sin_size);
    if (clientsocket > -1) {
      std::cout << "Client conected" << std::endl;
    } else {
      std::cout << "oh no no no no" << std::endl;
    }

    
    char data[this->MAXDATASIZE];
    printf("im here accepting a connection\n");


    /*
      CRYPTO
    */

    unsigned char encrypted_key[256];
    
    // get that privkey
    EVP_PKEY *privkey;
    FILE *privf = fopen("rsa_priv.pem", "rb");
    privkey = PEM_read_PrivateKey(privf,NULL,NULL,NULL);
    unsigned char decrypted_key[32];

    // receive encrypted key

    recv(clientsocket, &encrypted_key, 256, 0);
    encrypted_key[256] = '\0';
    
    int decryptedkey_len = JEFF.rsa_decrypt(encrypted_key, 256, privkey, decrypted_key);

    
    std::cout << decrypted_key << std::endl;






    /*
      END
    */
    
    // Get username
    char username[100];
    std::memset(username, 0, sizeof(username));
    int r = recv(clientsocket, username, 100, 0);
    if  (r < 0) {
      const std::string err = "Failed to get username, exiting";
      send(clientsocket, err.c_str(), err.length(), 0);
      std::cerr << "failed to get username, killing session" << std::endl;
      close(clientsocket);
    }

    char actual_name[r];
    std::strcpy(actual_name, username);

    pthread_t client_r;
    thread* t = new thread();
    t->socket = clientsocket;
    t->username = std::string(actual_name);
    t->client = client;
    t->key = decrypted_key;
    t->instance = this;

    std::cout << "len: " << r << std::endl;
    std::cout << "key: " << t->key << std::endl;
    std::cout << "username: " << t->username << std::endl;
    std::cout << "sockID: " << t->socket << std::endl;
    // Add the user to our global ref
    this->users.push_back(*t);

    pthread_create(&client_r, NULL, ChatServer::server_handler, t);
    pthread_detach(client_r);

  }


  return EXIT_SUCCESS;
}

} // namespace server

int main() {
  server::ChatServer cs;
  cs.RunServer();

  return EXIT_SUCCESS;
}
