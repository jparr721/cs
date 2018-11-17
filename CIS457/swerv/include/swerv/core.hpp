#pragma once

#include <string>

namespace swerver {
  class Core {
    public:
      Core() = default;
      ~Core() = default;

      int Run();

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

      bool handle_args(char** argv);
      void usage();
  };
} // namespace swerver
