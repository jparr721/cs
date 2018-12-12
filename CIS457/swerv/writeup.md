# Swerv - Warp speed mini server

by Jarred, Kyle, and Thomas

### Challenges

The hardest parts of the server was the construction of headers and the movement of data over http. It's a very dense protocol so we found it particularly challenging when some things were not exactly cut and dry. In many cases, I struggled to see how some of the connections were formed or how they would be interpreted at the browser level. It was very interesting to see how you can manipulate the browser requests to show more of what you want. My team being distributed proposed some challenges as well particularly in the way of communication and just overall getting things done efficiently. Sometimes we would run into a situation where two people would be solving the same problem at the same time, which was interesting in its own way.

Overall, the project presented many interesting processes that we haven't seen before when it comes into the space of modern web development principles and it was very exciting to use them in practice. It's very enlightening to not see it exclusively as a black box.

### Compiling
You must create a build directory since CMake does not allow in-source builds
```
$ mkdir build
$ cd build
$ cmake ..
$ make
$ cd ..
$ ./swerver <OPTIONS>
```
### Usage
```
Usage:
      swerver -p <PORT>       Specifies which port to run on.
      swerver -docroot <DIR>  Specifies where the docroot will be.
      swerver -logfile <FILE> Specifies where log files will be written to.
      swerver default         Run server with default settings.
```