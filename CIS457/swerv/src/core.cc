#include <swerve/core.hpp>

#include <iostream>

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
      switch (argv[i]) {
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
         break;
      }
    }

    return true;
  }

  int Core::Run(int argc, char** argv) {

    if (!handle_args(argc, argv)) {
      return EXIT_FAILURE;
    }
  }
} // namespace swerver
