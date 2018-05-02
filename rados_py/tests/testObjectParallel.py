import radosObject.radosObject as ro
import numpy as np
from mpi4py import MPI


comm = MPI.COMM_WORLD
rank = comm.rank


test_file = ro.RadosObject("MPI_TEST", "lena_testing", "open_or_create")

test_file[str(rank)]=np.random.randint(0,1024,size=(32,32),dtype=np.uint32)

test_file.save()

print(test_file)
comm.Barrier()

if rank is 0:
    print(test_file)
    test_file.delete()
