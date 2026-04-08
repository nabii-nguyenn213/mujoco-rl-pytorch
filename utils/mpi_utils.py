from mpi4py import MPI 

comm = MPI.COMM_WORLD
rank = comm.Get_rank()
world_size = comm.Get_size()

def get_rank(): 
    return rank 

def get_world_size(): 
    return world_size

def is_main_process(): 
    return rank==0

def barrier(): 
    comm.Barrier()

def gather(data, root=0): 
    return comm.gather(data, root=root)

def broadcast(data, root=0):
    return comm.bcast(data, root=root)
