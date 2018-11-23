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
          std::string last_modified);

      // Getters and setters
      std::string get_docroot();
      void set_docroot(std::string root);

      std::string get_logfile();
      void set_logfile(std::string);

      int get_port();
      void set_port(int port);
    private:
      std::string docroot = ".doc";
      std::string logfile = ".log";
      int port = 1024;

      static void* thread_handler(void* args);

      bool handle_args(int argc, char** argv);
      void usage();
  };
} // namespace swerver
