# Torch-Spyre Distributed Tests

This directory contains unit tests for `torch.distributed` operations against the Spyre backend.

## Test Files

* `spyreccl_backend.py` : Tests for the SpyreCCL backend availability and device mapping.
* `test_barrier.py` : Tests for `torch.distributed.barrier()` operation
* `test_broadcast.py` : Tests for `torch.distributed.broadcast()` operation

## Running the Tests

### Prerequisites
* Torch Spyre must be installed and configured
* The `spyreccl` backend must be available
* Multiple Spyre devices must be available for multi-process tests

### Environment Variables

* **`AIU_WORLD_SIZE`**: Number of processes to spawn for distributed tests (default: 2)

### Running Tests with Config File (Recommended)

A dedicated config file is provided for running all distributed tests:

```bash
# Run all distributed tests
cd torch-spyre/tests
./run_test.sh configs/test_distributed_config.yaml
```

### Running Tests Directly with torchrun

You can also run the tests directly using `torchrun`:

```bash
# Run barrier tests with 2 processes
torchrun --nproc-per-node 2 -m pytest test_barrier.py

# Run broadcast tests with 4 processes
torchrun --nproc-per-node 4 -m pytest test_broadcast.py

# Run specific test
torchrun --nproc-per-node 2 -m pytest test_barrier.py::TestBarrier::test_barrier_synchronization_timing

# Run with verbose output
torchrun --nproc-per-node 2 -m pytest test_broadcast.py -v
```

The output from all processes are routed to a single terminal session which can make it difficult to read. To improve readability, you can use the `split_output.sh` script to split the output into separate files for each process.

```shell
# Logging directory into which each process will have an output file
export _LOGDIR=/tmp/pytest-torch-spyre-dist
# In addition to logging the output of rank 0, also display it to the screen
export _SHOW_PROGRESS=1

# Create log directory
mkdir -p "${_LOGDIR}"

# Run with split_output.sh wrapper
torchrun --nproc-per-node 4 --no-python bash split_output.sh python3 -u -m pytest test_broadcast.py
```

## References

* [PyTorch Distributed Documentation](https://pytorch.org/docs/stable/distributed.html)
* [torchrun Documentation](https://pytorch.org/docs/stable/elastic/run.html)
