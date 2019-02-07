#ifndef THREADED_SERVER_H_
#define THREADED_SERVER_H_

#include <pthread.h>
#include <random>

namespace server {
  class Zappo {
    public:
      int server();
    private:
      void sig_handler(int signum);
      void* worker(void* arg);
      std::uniform_int_distribution<int> make_rand(int low, int high);

      size_t interrupt{0};
      size_t requests{0};
      size_t time{0};
      pthread_mutex_t req_mutex = PTHREAD_MUTEX_INITIALIZER;
      pthread_mutex_t time_mutex = PTHREAD_MUTEX_INITIALIZER;

  };
} // namespace server
#endif // THREADED_SERVER_H_
