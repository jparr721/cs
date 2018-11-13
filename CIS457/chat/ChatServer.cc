#include "./include/ChatServer.hpp"

#include "./include/Succ.hpp"
#include "./include/Crypto.hpp"
#include <cstdlib>
#include <cstring>
#include <algorithm>
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

  std::pair<std::string, int> ChatServer::encrypt_string(std::string input, unsigned char key[32]) {
    yep::Crypto JEFF;
  unsigned char* plaintext = (unsigned char*)input.c_str();
  unsigned char miv[16];
  std::memset(miv, 0, 16);
  unsigned char ciphertext[1024];
  int ciphertext_len = JEFF.encrypt(plaintext, input.size(), key, miv, ciphertext);

  std::string str( ciphertext, ciphertext + sizeof ciphertext / sizeof ciphertext[0] );

  std::pair<std::string, int> result;

  result.first = str;
  result.second = ciphertext_len;
  
  return result;
}
  
void* ChatServer::server_handler(void* args) {
  the::Succ SKOOMA_HIGH;
  yep::Crypto JEFF;
  ChatServer::std_message s;
  ChatServer::thread t;

  RAND_bytes(s.iv, 16);

  std::memcpy(&t, args, sizeof(ChatServer::thread));

  socklen_t sin_size = sizeof(t.client);

  while (true) {
    char data[4096];
    std::memset(data, 0, sizeof(data));
    int r = recv(t.socket, data, 4096, 0);


    if (r < 0) {
      break;
    }

    if (std::string(data) == "/quit" || r <= 0) {
      std::cout << "Shutting down the server connection to user: " << t.username << std::endl;
      // Break this worker
      break;
    }

    /* crypto */
    std::string message = std::string(data);
    unsigned char* cipher = (unsigned char*) message.c_str();

    unsigned char miv[16];
    std::memset(miv, 0, 16);
    
    unsigned char decrypted_message[1024];
    std::memset(decrypted_message, 0, 1024);
    std::cout << "Got something: " << message << std::endl;

    std::cout << t.key << std::endl;

    int fake = 16;
    
    if (message.size() > fake) {
      fake = message.size();
    }
    
    int message_len = JEFF.decrypt(
          cipher,
          fake,
          t.key,
          miv,
          decrypted_message);
    std::cout << " <<< " << t.username << ": " << decrypted_message << std::endl;

    std::cout << decrypted_message << std::endl;

    message = std::string((char *) decrypted_message);
    
    std::string command = t.instance->extract_command(message);

    std::cout << "Command:(" + command + ")" << std::endl;

    std::vector<std::string> command_args;
    std::istringstream iss(message);

    for(std::string message; iss >> message; ) {
      command_args.push_back(message);
    }

    std::cout << "args: " << command_args.size() << std::endl;

    std::string outgoing;
    
    if (command.substr(0, 4) == "list") {
      std::string outlist = "";

      for (int i = 0; i < t.instance->users.size(); i++) {
        outlist = outlist + "\n" + t.instance->users[i].username;
      }

      std::pair<std::string, int> out = t.instance->encrypt_string(outlist, t.key);
      send(t.socket, out.first.c_str(), out.second, 0);
      
    } else if (command == "broadcast") {
      std::cout << "Sending broadcast packet" << std::endl;

      for (int i = 0; i < t.instance->users.size(); ++i) {
        std::cout << "sending to " + t.instance->users[i].username + ": " + command_args[1] << std::endl;

	std::pair<std::string, int> out = t.instance->encrypt_string(command_args[1], t.instance->users[i].key);
	
        send(t.instance->users[i].socket, out.first.c_str(), out.second, 0);
      }
    } else if (command == "pm") {
      std::cout << "Sending personal message" << std::endl;

      for (int i = 0; i < t.instance->users.size(); ++i) {
        std::cout << "username1 = " + t.instance->users[i].username << std::endl;
        std::cout << "username2 = " + command_args[1] << std::endl;
        if (t.instance->users[i].username == command_args[1]) {
	  std::pair<std::string, int> out = t.instance->encrypt_string(command_args[2], t.instance->users[i].key);
	  
	  send(t.instance->users[i].socket, out.first.c_str(), out.second, 0);
	}
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
  return nullptr;
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
    //RAND_pseudo_bytes(outgoing.iv, 16);
    /*outgoing.cipher = succ.encrypt(
        const_cast<unsigned char*>(reinterpret_cast<const unsigned char*>(user.key)),
        outgoing.iv,
        message);
    */
    
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
      
      command.erase(std::remove(command.begin(), command.end(), '\n'), command.end());
      
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
      std::cout << "Error, failed to connect client" << std::endl;
    }


    char data[this->MAXDATASIZE];
    std::cout << "Server now accepting connections" << std::endl;


    /*
      CRYPTO
    */

    unsigned char encrypted_key[256];

    // get that privkey
    EVP_PKEY *privkey;
    FILE *privf = fopen("rsa_priv.pem", "rb");
    privkey = PEM_read_PrivateKey(privf, nullptr, nullptr, nullptr);
    unsigned char decrypted_key[32];

    // receive encrypted key


    
    recv(clientsocket, &encrypted_key, 256, 0);
    //encrypted_key[256] = '\0';

    int decryptedkey_len = JEFF.rsa_decrypt(encrypted_key, 256, privkey, decrypted_key);

    decrypted_key[decryptedkey_len] = '\0';
    
    std::cout << "this the decrypted key: " << decrypted_key << std::endl;

    
    
    
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
      std::cerr << "Failed to get username, killing session" << std::endl;
      close(clientsocket);
    }

    char actual_name[r];
    std::strcpy(actual_name, username);

    pthread_t client_r;
    thread* t = new thread();
    t->socket = clientsocket;
    t->username = std::string(actual_name);
    t->client = client;
    memcpy(t->key, &decrypted_key, decryptedkey_len);
    t->instance = this;

    std::cout << "len: " << r << std::endl;
    std::cout << "key: " << t->key << std::endl;
    std::cout << "username: " << t->username << std::endl;
    std::cout << "sockID: " << t->socket << std::endl;
    // Add the user to our global ref
    this->users.push_back(*t);

    pthread_create(&client_r, nullptr, ChatServer::server_handler, t);
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
