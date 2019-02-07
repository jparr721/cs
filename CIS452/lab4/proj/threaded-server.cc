#include <iostream>
#include <csignal>
#include <cstdlib>
#include <string>
#include "threaded-server.h"

namespace server {
  void Zappo::sig_handler(int) {
    std::cout << "Interrupt received" << std::endl;
    interrupt = 1;

    pthread_mutex_unlock(&req_mutex);
    pthread_mutex_unlock(&time_mutex);
  }

  std::uniform_int_distribution<int> make_rand(int low, int high) {
    std::uniform_int_distribution<int> dist(low, high);
    return dist;
  }

  void* Zappo::worker(void* arg) {
    pthread_mutex_lock(&req_mutex);
    pthread_mutex_lock(&time_mutex);
    std::random_device rd;
    std::mt19937 mt(rd());
    auto dist = make_rand(1, 10);
    auto rand_wait = make_rand(7, 10);

    auto *sp = static_cast<std::string*>(arg);
    std::string filename(*sp);

    size_t sleep_duration;
    if (dist(mt) > 8)
      sleep_duration = 1;
    else
      sleep_duration = rand_wait(mt);

    std::cout << "The file, good sir: " << filename << std::endl;

    time += sleep_duration;
    ++requests;

    pthread_mutex_unlock(&req_mutex);
    pthread_mutex_unlock(&time_mutex);

    pthread_exit(nullptr);
  }

  int Zappo::server() {
    sigaction action;
    action.sa_handler = &Zappo::sig_handler;
    action.sa_flags = 0;

    pthread_t thread;
    int status{0};

    std::string input;

    if (pthread_mutex_init(&req_mutex, nullptr) != 0) {
      std::cerr << "Failed to make requests mutex" << std::endl;
      return EXIT_FAILURE;
    }

    if (pthread_mutex_init(&time_mutex, nullptr) != 0) {
      std::cerr << "Failed to make time mutex" << std::endl;
      return EXIT_FAILURE;
    }

    if (sigaction(SIGINT, &action, nullptr) < 0) {
      std::cerr << "Failed to catch SIGINT" << std::endl;
      return EXIT_FAILURE;
    }

    while (!interrupt) {
      std::cout << "What file?: " << std::endl;
      std::getline(std::cin, input);

      if (input.length() == 0) {
        std::cout << "Oh, we got a wise guy eh? Enter a file like a real man!" << std::endl;
        continue;
      }

      if ((status = pthread_create(&thread, nullptr, &Zappo::worker, input.c_str())) != 0) {
        std::cerr << "Thread create error " << status<< std::endl;
        return EXIT_FAILURE;
      }
    }

    float proc = static_cast<float>(time) / requests;
    std::cout << "Processing time: " << time << std::endl;
    std::cout << "Average processing time: " << proc << std::endl;
    std::cout << "Total number of requests: " << requests << std::endl;

    pthread_mutex_destroy(&req_mutex);
    pthread_mutex_destroy(&time_mutex);

    return EXIT_SUCCESS;
  }
} // namespace server

int main() {
  server::Zappo z;
  return z.server();
}
