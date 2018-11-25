#include <swerve/core.h>

#include <chrono>
#include <pthread.h>
#include <iostream>
#include <memory>
#include <sstream>
#include <sys/socket.h>

namespace swerver {
  std::string Core::get_logfile() {
    return this->logfile;
  }

  void Core::set_logfile(std::string logfile) {
    this->logfile = logfile;
  }

  std::string Core::get_docroot() {
    return this->docroot;
  }

  void Core::set_docroot(std::string docroot) {
    this->docroot = docroot;
  }

  int Core::get_port() {
    return this->port;
  }

  void Core::set_port(int port) {
    this->port = port;
  }

  void Core::usage() {
    const char* help =
      "Usage:"
      " swerver -p <PORT>       Specifies which port to run on."
      " swerver -docroot <DIR>  Specifies where the docroot will be."
      " swerver -logfile <FILE> Specifies where log files will be written to.";
      " swerver default         Run server with default settings.";

    std::cout << help << std::endl;
  }

  bool Core::handle_args(int argc, char** argv) {
    if (argc < 1) {
      usage();
      return false;
    }

    for (int i = 1; i < argc - 1; ++i) {
      switch (std::string(argv[i])) {
        case "-p":
         this->set_port(std::stoi(argv[i + 1]));
         break;
        case "-docroot":
         this->set_docroot(argv[i + 1]);
         break;
        case "-logfile":
         this->set_logfile(argv[i+ 1]);
         break;
        default:
         std::cerr << "Error, argument not supported" << std::endl;
         return false;
      }
    }

    return true;
  }

  void Core::send_http_repsonse(
      int socket,
      int code,
      bool keep_alive,
      ContentType content_type,
      std::string file_data) {
    std::string code_msg, connection_type, content_type_string;
    auto now = std::chrono::system_clock::now();
    auto in_time_t = std::chrono::system_clock::to_time_t(now);
    std::string html = "";

    switch (code) {
      case 200:
        code_msg = "200 OK\r\n";
        html = file_data;
        break;
      case 304:
        code_msg = "304 Not Modified\r\n";
        html = file_data;
        break;
      case 404:
        code_msg = "404 Not Found\r\n";
        html = html404;
        break;
      case 501:
        code_msg = "501 Not Implemented\r\n";
        html = html501;
        break;
      default:
        code_msg = std::to_string(code) + "\r\n";
        break;
    }

      if (keep_alive) {
        connection_type = "keep-alive\r\n";
      } else {
        connection_type = "close\r\n";
      }

      switch (content_type) {
        case 1:
          // Text
          content_type_string = "text/plain\r\n";
          break;
        case 2:
          // html
          content_type_string = "text/html\r\n";
          break;
        case 3:
          // jpeg
          content_type_string = "image/jpeg\r\n";
          break;
        case 4:
          // pdf
          content_type_string = "application/pdf\r\n";
          break;
      }

      std::ostringstream header;
      header << "HTTP/1.1 " << code_msg
        << "Date: " << std::put_time(std::localtime(&in_time_t))
        << "\r\n"
        << "Server: GVSU\r\n"
        // TODO - Add the rest of the fields
    }
  }

  void* Core::thread_handler(void* args) {
    Core::thread t;
    Core c;
    std::memcpy(&t, args, sizeof Core::thread);
    socklen_t sin_size = sizeof t.client;
    char line[5000];

    for (;;) {
      int in = recv(t.socket, line, 5000, 0);
      if (in < 0) {
        std::cout << "Failed to get data from client" << std::endl;
        c.send_http_response(404);
      }
    }
  }

  int Core::Run(int argc, char** argv) {
    if (!handle_args(argc, argv)) {
      return EXIT_FAILURE;
    }

    int sockfd = socket(AF_INET, SOCK_STREAM, 0);

    sockaddr_in server, client;
    server.sin_family = AF_INET;
    server.sin_addr.s_addr = INADDR_ANY;
    server.sin_port = htons(this->port);

    int b = bind(sockfd, reinterpret_cast<sockaddr*>&server, sizeof(server));
    if (b < 0) {
      std::cerr << "Failed to bind to port: " << this->port << std::endl;
      return EXIT_FAILURE:
    }

    std::cout << "Server ready for connections on port: " << this->port << std::endl;

    listen(sockfd, 10);

    for(;;) {
      // Shared pointer for safe memory usage
      std::shared_ptr<thread> t;
      t->instance = this;
      socklen_t sin_size = sizeof client;

      int clientfd = accept(sockfd, reinterpret_cast<sockaddr*(&client), &sin_size);
      t->socket = clientfd;
      pthread_t connection;

      int thread_status = pthread_create(&connection, nullptr, Core::thread_handler, t);
      if (thread_status < 0) {
        std::cerr << "Failed to create thread" << std::endl;
        return EXIT_FAILURE;
      }

      // I don't f**king care about you
      pthread_detach(connection);
    }

    return EXIT_SUCCESS;
  }
} // namespace swerver
