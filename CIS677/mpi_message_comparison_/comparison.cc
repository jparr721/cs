#include <chrono>
#include <iostream>
#include <mpi.h>
#include <sstream>
#include <string>
#include <unistd.h>

class MPIComparison {
  const int MASTER = 0;
  const int TAG = 0;
  const int MSGSIZE = 100;
  const int MAX = 25;

  int compare_standard(int argc, char** argv);
  int compare_broadcast(int argc, char** argv);
};

int MPIComparison::compare_standard(int argc, char** argv) {
  int rank, source, num_nodes;
  std::string host("");
  std::string message("");
  std::ostringstram mess_buf;

  MPI_Init(&argc, &argv);
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  MPI_Comm_size(MPI_COMM_WORLD, &num_nodes);

  if (rank != this->MASTER) {
    gethostname(host, this->MAX);
    mess_buf << "Process: " << rank << ", host: " << host;
    message = mess_buf.str();
  } else {
    gethostname(host, this->MAX);
    std::cout << "Num nodes: " << num_nodes << std::endl;
    std::cout << "Master process: " << rank << "on host: " << host << std::endl;

    char mes[this->MSGSIZE];
    for (source = 1; source < num_nodes; ++source) {
      MPI_Recv(mes, MSGSIZE, MPI_CHAR, source, TAG, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
      std::cout << std::string(mes) << std::endl;
    }
  }

  MPI_Finalize();

  return EXIT_SUCCESS;
}

int MPIComparison::compare_broadcast(int argc, char** argv) {

}

int main(int argc, char** argv) {
  MPIComparison mpic;

  mpic.compare(argc, argv);

  return EXIT_SUCCESS;
}
