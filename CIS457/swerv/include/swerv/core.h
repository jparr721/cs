#pragma once

#include <string>

namespace swerver {
  class Core {
    public:
      enum ContentType {
        text,
        html,
        jpeg,
        pdf
      };

      Core() = default;
      ~Core() = default;

      int Run(int argc, char** argv);

      struct thread {
        int socket;
        // To access our goodies
        Core* instance;
      };

      void send_http_response(
          int socket,
          int code,
          bool keep_alive,
          Core::ContentType content_type,
          std::string filename,
          std::string last_modified,
          std::string file_data);

      // Getters and setters
      std::string get_docroot();
      void set_docroot(std::string root);

      std::string get_logfile();
      void set_logfile(std::string);

      int get_port();
      void set_port(int port);
    private:
      const std::string html404 =
        R"(
          <!DOCTYPE html>
          <html>
          <title>404! Whoops!</title>
          <body>
          <h1>
          Error! The requested resource could not be found!
          </h1>
          </body>
          </html>
        )";
      const std::string html501 =
        R"(
          <!DOCTYPE html>
          <html>
          <title>501!</title>
          <body>
          <h1>
          Request Type not implemented.
          </h1>
          </body>
          </html>
        )";
      std::string docroot = ".doc";
      std::string logfile = ".log";
      int port = 1024;

      static void* thread_handler(void* args);

      bool handle_args(int argc, char** argv);
      void usage();
  };
} // namespace swerver
