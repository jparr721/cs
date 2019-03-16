#ifndef SLAM_H_
#define SLAM_H_

#include <string>

namespace slam {
  class Slam {
    public:
      void process_frame(std::string frame_name);
  };
} // namespace slam

#endif // SLAM_H_
