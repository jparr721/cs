## Point - To - Point
- MSG: Addres, type, length, tag
- Process: Rank, Comm, `MPI_ANY_SOURCE`, `MPI_ANY_TAG`
- Utilities:
  - Status: Source, Tag, Errval, `MPI_GET_COUNT()`

### 4 Ways to send
- Standard: `MPI_Send()`
  - Implementation decides the semantics (number of blocking threads)
- Synchronous: `MPI_Ssend()`
  - Send completes when the receive succeeds ("true" blocking)
- Buffered: `MPI_Bsend()`
  - Send completes when the copy of the message completes, creates an extra memory copy and gives the advantage of not needing to worry about the next machine has finished since we have the data we need to proceed in our calculation. (Very high performance). Important to know the state of the other machine.
- Ready: `MPI_Rsend()`
  - Send starts when corresponding receive reached
| `MPI_Send(p2)` | `MPI_Send(p1)` |
|---------------|-----------------|
|  `MPI_Recv(p2)`  |`MPI_Recv(p1)`|
|                  |              |
|                  |              |

This leads to a deadlock if one sends and another tries to receive at the same time. Opt for the two-way strategy:

#### Two-Way
Utilizing `MPI_Sendrecv()` which combines the send and recv into one call and removes the deadlock

### How to Receive
`MPI_Recv()` function is used which returns when the data is received ("true" blocking)

### Non Blocking Communication
- Immeditate: `MPI_Isend()` / `MPI_Irecv()`
  - Calls return immediately
  - Need to check for completion
- Check: `MPI_Test()` & `MPI_Wait()`
  - Test will check if the operation is complete (ex. `MPI_Isend` returns immediately, so we don't know for sure when the message was sent. To do subsequent computation and not overwrite the buffer space, we can test if it's complete before writing the new message)
  - Waits until the computation has completed (Makes sense to use when main work is done and now we just need to write. Allows for multiple jobs to be tasked concurrently)

#### Ex. Image Processing
```c++
while (not_converged) {
  // send boundry values to neighbord
  // If value is on another machine, we can communicate across multiple machines non blocking to read all the data
  // Receive boundry values from neighbors
  // update non-boundry cells
  // Wait for send here once all other machines have finished in order to do writes, also have to wait for receives this lets you know you have all info
  // Update boundry cells
}
```

### Why do non blocking
- Overlaps communication and communication allowing them to happen at once
- Less likely to deadlock

#### Downsides
- A little bit extra overhead (mem copy)
- Extra checks and complexity
