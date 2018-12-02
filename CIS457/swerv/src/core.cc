#include <swerv/core.h>

#include <boost/algorithm/string.hpp>
#include <chrono>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <memory>
#include <netinet/in.h>
#include <pthread.h>
#include <sstream>
#include <sys/socket.h>
#include <sys/stat.h>
#include <unistd.h>

namespace swerver {
  Core::ContentType Core::make_content_type(std::string ext) {
    if (ext == "html") {
      return Core::ContentType::html;
    } else if (ext == "txt") {
      return Core::ContentType::text;
    } else if (ext == "jpg") {
      return Core::ContentType::jpeg;
    } else if (ext == "pdf") {
      return Core::ContentType::pdf;
    } else {
      return Core::ContentType::html;
    }
  }

  void Core::log_request(std::string log) {
    std::cout << log << std::endl;
  }

  std::string Core::make_path(std::string dir, std::string file) const {
    std::filesystem::path p = std::filesystem::current_path();

    return std::string(p);
  }

  std::string Core::read_file(std::string filename) const {
    std::ifstream f(filename);
    if (!f.good()) return "";

    // Write to a string
    std::string str((std::istreambuf_iterator<char>(f)),
                     std::istreambuf_iterator<char>());

    return str;
  }

  void Core::usage() const {
    const char* help =
      "Usage:\n"
      " swerver -p <PORT>       Specifies which port to run on.\n"
      " swerver -docroot <DIR>  Specifies where the docroot will be.\n"
      " swerver -logfile <FILE> Specifies where log files will be written to.\n"
      " swerver default         Run server with default settings.\n";

    std::cout << help << std::endl;
  }

  std::string Core::check_file_mod(std::string path) {
    auto time = std::filesystem::last_write_time(path);
    auto cftime = decltype(time)::clock::to_time_t(time);
    auto final_time = std::asctime(std::localtime(&cftime));
    return std::string(final_time);
  }

  std::string Core::get_current_time() {
    auto now = std::chrono::system_clock::now();
    auto in_time_t = std::chrono::system_clock::to_time_t(now);
    auto current_time = std::put_time(std::localtime(&in_time_t), "%Y-%m-%d-%X");
    std::stringstream time;
    time << current_time;

    return time.str();
  }

  std::string Core::modded_since(std::string req, std::string filename, std::string ext) {
    if (ext == "html" || ext == "txt") {
      std::string mod_header;
      // Need to double check this
      mod_header = req.substr(0, 19);

      if (mod_header == "If-Modified-Since:") {
        std::string date_mod_since = req.substr(19, 29);
        std::string file_mod = check_file_mod(filename);

        if (date_mod_since == file_mod) {
          return file_mod;
        }
        return file_mod;
      }
    }

    return "";
  }

  bool Core::handle_args(int argc, char** argv) {
    if (argc < 2) {
      usage();
      return false;
    }

    if (std::string(argv[1]) == "default") {
      std::cout << "Running server with default configuration" << std::endl;
      return true;
    } else {
      for (int i = 1; i < argc - 1; ++i) {
        std::string opt(argv[i]);
        if (opt == "-p") {
           this->port = std::stoi(argv[i + 1]);
        } else if (opt == "-docroot") {
           this->docroot = std::string(argv[i + 1]);
        } else if (opt == "-logfile") {
           this->logfile = std::string(argv[i + 1]);
        } else {
           std::cerr << "Error, argument not supported" << std::endl;
           return false;
        }
      }
    }

    return true;
  }

  void Core::send_http_response(
      int socket,
      int code,
      bool keep_alive,
      Core::ContentType content_type,
      std::string filename,
      std::string last_modified,
      std::string file_data
      ) const {
    std::string code_msg, connection_type, content_type_string, pdf_header;
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
        html = this->html404;
        break;
      case 501:
        code_msg = "501 Not Implemented\r\n";
        html = this->html501;
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
      case Core::ContentType::text:
        // Text
        content_type_string = "text/plain\r\n";
        break;
      case Core::ContentType::html:
        // html
        content_type_string = "text/html\r\n";
        break;
      case Core::ContentType::jpeg:
        // jpeg
        content_type_string = "image/jpeg\r\n";
        break;
      case Core::ContentType::pdf:
        // pdf
        content_type_string = "application/pdf\r\n";
        pdf_header = "Content-Disposition: inline; filename=" + filename + "\r\n";
        break;
    }

    std::ostringstream header;
    header << "HTTP/1.1 " << code_msg
      << "Date: " << std::put_time(std::localtime(&in_time_t), "%Y-%m-%d-%X")
      << "\r\n"
      << "Server: GVSU\r\n"
      << "Accepted-Ranges: bytes\r\n"
      << "Connection: "
      << connection_type
      << "Content-Type: "
      << content_type_string
      << "\r\n"
      << "Content-Length: "
      << html.length()
      << "\r\n";
    if (content_type == Core::ContentType::pdf) {
      header << pdf_header;
    }

    if (last_modified != "") {
      header << "Last-Modified: "
        << last_modified
        << "\r\n\n";
    }

    send(socket, header.str().c_str(), header.str().size(), 0);
    std::cout << "{ Header: " << header.str() << "}" << std::endl;
  }

  bool Core::init_system_file(std::string path, std::string default_path) {
    struct stat info;

    if (stat(path.c_str(), &info) != 0) {
      std::cout << "Failed to find directory: " << path << " creating" << std::endl;

      const int dir_error = mkdir(path.c_str(), S_IRWXU | S_IRWXG | S_IROTH | S_IXOTH);

      if (dir_error < 0) {
        std::cout << "Failed to create directory with name: " << path << " creating with default name: " << default_path << std::endl;
      }

      if (stat(path.c_str(), &info) != 0) {
        std::cout << "Failed to find " << default_path << " creating now..." << std::endl;
        const int dir_error = mkdir(default_path.c_str(), S_IRWXU | S_IRWXG | S_IROTH | S_IXOTH);

        if (dir_error < 0) {
          std::cout << "Failed to create default directory or find it, check your directory path and try again. Exiting" << std::endl;
          return false;
        }

        // Reset our globals to the defaults
        if (default_path == ".doc") this->docroot = default_path;
        else if (default_path == ".log") this->logfile = default_path;
      }
    }

    return true;
  }

  void* Core::thread_handler(void* args) {
    Core::thread t;
    Core c;
    std::memcpy(&t, args, sizeof(Core::thread));
    socklen_t sin_size = sizeof t.socket;
    char line[5000];

    for (;;) {
      int in = recv(t.socket, line, 5000, 0);
      if (in < 0) {
        std::cout << "Failed to get data from client" << std::endl;
      }

      std::string input(line);

      c.log_request(input);

      std::string request_type = input.substr(0, 3);

      if (request_type == "GET") {
        bool file_request = false;
        std::string req = input.substr(5, input.substr(5).find(" ", 0));

        if (req == "") {
          c.send_http_response(t.socket, 200, true, c.Core::ContentType::html, "", "", c.html200Default);
        } else {
          std::vector<std::string> filename;
          auto content_type = c.make_content_type(filename[1]);
          boost::algorithm::split(filename, req, boost::is_any_of("."));

          // Check if the file exists first
          if (access(req.c_str(), F_OK) != -1) {
            std::string did_mod = c.modded_since(input, req, filename[1]);
            if (did_mod != "") {
              auto file_contents = c.read_file(req);
              if (file_contents == "") c.send_http_response(t.socket, 200, true, c.Core::ContentType::html, "", "", c.html200Failed);
              c.send_http_response(t.socket, 200, true, content_type, req, did_mod, file_contents);
            } else {
              auto file_contents = c.read_file(req);
              if (file_contents == "") c.send_http_response(t.socket, 200, true, c.Core::ContentType::html, "", "", c.html200Failed);
              c.send_http_response(t.socket, 304, true, content_type, req, did_mod, file_contents);
            }
          }
        }
      }
    }
  }

  int Core::Run(int argc, char** argv) {
    if (!this->handle_args(argc, argv)) {
      return EXIT_FAILURE;
    }

    bool err;
    // Create docroot
    err = this->init_system_file(this->docroot, ".doc");
    if (!err) return EXIT_FAILURE;

    // Create logfile root
    err = this->init_system_file(this->logfile, ".log");
    if (!err) return EXIT_FAILURE;

    int sockfd = socket(AF_INET, SOCK_STREAM, 0);

    sockaddr_in server, client;
    server.sin_family = AF_INET;
    server.sin_addr.s_addr = INADDR_ANY;
    server.sin_port = htons(this->port);

    int b = bind(sockfd, reinterpret_cast<sockaddr*>(&server), sizeof(server));
    if (b < 0) {
      std::cerr << "Failed to bind to port: " << this->port << std::endl;
      return EXIT_FAILURE;
    }

    std::cout << "Server ready for connections on port: " << this->port << std::endl;

    listen(sockfd, 10);

    for(;;) {
      // Shared pointer for safe memory usage
      Core::thread *t = new thread();
      t->instance = this;
      socklen_t sin_size = sizeof client;

      int clientfd = accept(sockfd, reinterpret_cast<sockaddr*>(&client), &sin_size);
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
