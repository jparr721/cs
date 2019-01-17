#include <algorithm>
#include <iostream>
#include <cstdlib>
#include <sys/wait.h>
#include <unistd.h>
#include <cstring>
#include <sys/time.h>
#include <vector>
#include <sys/resource.h>

int main(int argc, char** argv) {
  std::string instruction;
  struct rusage resources_used;

  for (;;) {
    std::cout << "Shellz0r> ";
    std::getline(std::cin, instruction);
    int instruction_size = instruction.length();

    if (instruction_size <= 0) {
      fprintf(stderr, "Invalid command sequence");
      return EXIT_FAILURE;
    } else if (instruction[instruction_size - 1] == '\n') {
      instruction[instruction_size - 1] = '\0';
    }

    if (instruction == "quit") {
      return EXIT_SUCCESS;
    }

    char* temp = strtok(const_cast<char*>(instruction.c_str()), " ");
    char* command = temp;
    std::vector<char*> flags;
    while (temp != NULL) {
      flags.push_back(temp);
      temp = strtok(NULL, " ");
    }
    flags.erase(flags.begin());

    pid_t pid;
    int status;
    long time_s;
    long time_ms;
    long last_context_switch;
    pid = fork();

    if (pid < 0) {
      std::cerr << "Fork machine broke" << std::endl;
      exit(EXIT_FAILURE);
    } else if (pid == 0) {
      if (execvp(command, flags.data()) < 0) {
        std::cerr << "Exec machine broke" << std::endl;
        exit(EXIT_FAILURE);
      } else {
        exit(EXIT_SUCCESS);
      }
    } else {
      wait(&status);

      if (getrusage(RUSAGE_CHILDREN, &resources_used) < 0) {
        std::cout << "Failed to get process stats" << std::endl;
      } else {
        time_s = resources_used.ru_utime.tv_sec;
        time_ms = resources_used.ru_utime.tv_usec;
        last_context_switch = resources_used.ru_nivcsw;
        std::cout << "Total cpu time: " << resources_used.ru_utime.tv_sec - time_s << "s " << time_ms << "ms " << std::endl;
        std::cout << "Context switches " << resources_used.ru_nivcsw - last_context_switch << std::endl;
      }
    }
  }

  return EXIT_SUCCESS;
}
