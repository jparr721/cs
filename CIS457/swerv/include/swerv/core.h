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
      void usage() const;

      void send_http_response(
          int socket,
          int code,
          bool keep_alive,
          Core::ContentType content_type,
          std::string filename,
          std::string last_modified,
          std::string file_data) const;

      template <typename T>
      auto read_file(std::string filename) const;
  };
} // namespace swerver
