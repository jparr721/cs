# Swerv - Warp speed mini server

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
