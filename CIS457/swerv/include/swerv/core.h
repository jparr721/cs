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
      const std::string html304 =
        R"(
          <!DOCTYPE html>
          <html>
          <title>Not Modified</title>
          <body>
          <h1>
          File not modified.
          </h1>
          </body>
          </html>
        )";
      const std::string html200Failed =
        R"(
          <!DOCTYPE html>
          <html>
          <title>Error</title>
          <body>
          <h1>
          Failed to read file, sorry
          </h1>
          </body>
          </html>
        )";
      const std::string html200Default =
        R"(
          <!DOCTYPE html>
          <html>
          <title>Howdy</title>
          <body>
          <h1>
          What's up, dawg?
          </h1>
          </body>
          </html>
        )";
      std::string docroot = ".doc";
      std::string logfile = ".log";
      int port = 3000;

      static void* thread_handler(void* args);
      void usage() const;
      void send_http_response(
          int socket,
          int code,
          bool keep_alive,
          Core::ContentType content_type,
          std::string filename,
          std::string last_modified,
          std::string file_data) const;
      void log_request(std::string log);

      std::string check_file_mod(std::string path);
      std::string get_current_time();
      std::string modded_since(std::string req, std::string filename, std::string ext);
      std::string read_file(std::string filename) const;
      std::string make_path(std::string dir, std::string file) const;

      bool init_system_file(std::string path, std::string default_path);
      bool handle_args(int argc, char** argv);

      Core::ContentType make_content_type(std::string ext);
  };
} // namespace swerver
