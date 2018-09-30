# Crystal
Simulates the Biological Crystal Growth via Diffusion-Limited Aggregation with OpenMP-based multi-threaded simulations. Compilation instructions and requirements are at the bottom of this document.

## Background
Inside of diffusion limited aggregation particles are introduced into a simulation one at a time and perform a brownian movement randomly "walking" about the plane until, in this case, the move limit is reached, or the particle collides with another "stuck" particle.

## Process
The process involved here is to first begin with a "seed" particle of sorts. This will allow for the particles to have something to run into since they're being processed one at a time. The rough overview of the code is as follows:

#### The Crystal Class:
This class is simply to define methods and properties relating to the formation of the crystal structure and also to define the behavior of particles when they're introducted into the system.

Everything is started in the Run function which houses the main logic relating to the particles formation and also supplies the initial openmp parallel call to begin the parallelization of the rest of the simulation.

## Results
In order to adequately benchmark speedup and compare the affects of parallelizing this code, it was ran with several iterations under varying circumstances. Those circumstances were: sequential, 2 cores, 4 cores, 8 cores, 16 cores. They boasted the following results:

##### Sequential: `21.983 minutes (1319 seconds)`
##### 2 Cores:
##### 4 Cores:
##### 8 Cores:
##### 16 Cores:
