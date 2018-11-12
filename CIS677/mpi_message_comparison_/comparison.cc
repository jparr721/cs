#include <chrono>
#include <cstring>
#include <iostream>
#include <mpi.h>
#include <sstream>
#include <string>
#include <unistd.h>

class MPIComparison {
  public:
    int compare_standard(int argc, char** argv);
    int compare_broadcast(int argc, char** argv, int num_elements);
  private:
    const int MASTER = 0;
    const int TAG = 0;
    const int MSGSIZE = 100;
    const int MAX = 25;
};

int MPIComparison::compare_standard(int argc, char** argv) {
  int rank, source, num_nodes;
  std::string host("");
  std::string message("");
  std::ostringstream mess_buf;

  MPI_Init(&argc, &argv);
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  MPI_Comm_size(MPI_COMM_WORLD, &num_nodes);

  if (rank != this->MASTER) {
    gethostname(const_cast<char*>(host.c_str()), this->MAX);

    mess_buf << "Process: " << rank << ", host: " << host;

    message = mess_buf.str();
    MPI_Send(message.c_str(), message.size() + 1, MPI_CHAR, this->MASTER, this->TAG, MPI_COMM_WORLD);
  } else {
    gethostname(const_cast<char*>(host.c_str()), this->MAX);
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

int MPIComparison::compare_broadcast(int argc, char** argv, int num_elements) {
  int rank, num_nodes;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  MPI_Comm_size(MPI_COMM_WORLD, &num_nodes);
  std::string message("");
  std::string host("");
  std::ostringstream mess_buf;

  if (rank == this->MASTER) {
    // If we are the master, send our stuff to everyone
    gethostname(const_cast<char*>(host.c_str()), this->MAX);
    mess_buf << "Process: " << rank << ", host: " << host;
    message = mess_buf.str();

    for (int i = 0; i < num_nodes; ++i) {
      if (i != rank) {
        MPI_Send(message.c_str(), message.size() + 1, MPI_CHAR, i, TAG, MPI_COMM_WORLD);
      }
    }
  } else {
    char mes[MSGSIZE];
    MPI_Recv(mes, MSGSIZE, MPI_CHAR, MASTER, TAG, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
    std::cout << std::string(mes) << std::endl;
  }
}

int main(int argc, char** argv) {
  MPIComparison mpic;

  // How many times to send broadcast
  int iterations = std::stoi(argv[1]);

  // How many items to send
  int num_items = std::stoi(argv[2]);

  double total_standard_time = 0.0;
  double total_broadcast_time = 0.0;

  // Test over number of iterations
  for (int i = 0; i < iterations; ++i) {
    // Syncrhonize before beginning (to be safe)
    MPI_Barrier(MPI_COMM_WORLD);
    total_standard_time -= MPI_Wtime();
    mpic.compare_standard(argc, argv);
    // Wait for everything to finish running
    MPI_Barrier(MPI_COMM_WORLD);
    total_standard_time += MPI_Wtime();

    MPI_Barrier(MPI_COMM_WORLD);
    total_broadcast_time -= MPI_Wtime();
    mpic.compare_broadcast(argc, argv, num_items);
    MPI_Barrier(MPI_COMM_WORLD);
    total_broadcast_time += MPI_Wtime();
  }

  std::cout << "Average standard time: " << total_standard_time / num_items << std::endl;
  std::cout << "Average broadcast time: " << total_broadcast_time / num_items << std::endl;

  MPI_Finalize();

  return EXIT_SUCCESS;
}
