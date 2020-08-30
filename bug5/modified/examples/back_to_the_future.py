"""Test future objects in Dask."""

from dask.distributed import Client
from torchy.utils import simulate_delay


@simulate_delay(0.5)
def function(input_):
    print("[function] Input is of type {}".format(type(input_)))
    return input_


if __name__ == '__main__':
    # Start a client
    client = Client()
    # Set up containers for inputs and futures
    inputs = [1, 2, 3]
    futures = []
    ffutures = []
    # Feed in futures
    for input_ in inputs:
        future = client.submit(function, input_)
        ffuture = client.submit(function, future)
        futures.append(future)
        ffutures.append(ffuture)
    # Print results
    for future, ffuture in zip(futures, ffutures):
        print("Future: {}, FFuture: {}".format(future.result(), ffuture.result()))

    # Future works as expected. Output is below:

    # --- OUTPUT ---
    # [function] Input is of type <class 'int'>
    # [function] Input is of type <class 'int'>
    # [function] Input is of type <class 'int'>
    # [function] Input is of type <class 'int'>
    # [function] Input is of type <class 'int'>
    # [function] Input is of type <class 'int'>
    # Future: 1, FFuture: 1
    # Future: 2, FFuture: 2
    # Future: 3, FFuture: 3
    # Received signal 15, shutting down
