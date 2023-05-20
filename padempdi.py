# Usage: mpiexec -n <NTASKS> python3 padempdi.py

import numpy as np
from mpi4py import MPI

wt = MPI.Wtime() # "wall time" for calculating elapsed time
comm = MPI.COMM_WORLD # global communicator (can be used to define groups)
cpu = comm.Get_size() # total ranks mpi assigns
rank = comm.Get_rank() # rank is no. that the mpi assigns to the process

xlen = 118 # total images
sseg = int( xlen / cpu ) # size of a segment
mseg = seg + ( xlen % cpu ) # largest segment size

# - 256, 256 is the image, and 4 is the number of channels.
# - The different color bands/channels are stored in the third dimension, such
# that a gray-image is MxN, an RGB-image MxNx3 and an RGBA-image MxNx4.
# - RGBA = Reg, Green, Blue, Alpha(transparency) → PNG
# - Gray has only (256, 256) which corresponds to the size of the image.
xsub = np.zeros((msec, 256, 256, 4), dtype=np.float32) # desktop
xprocessed = np.zeros((xlen, 256, 256), dtype=np.float32) # result
    
# Process (rank) 0 reads the file and distributes the segments to the ranks
# Rank 0 also processes a thread
if rank == 0 :
     x = np.load("data/map01.npy") # read the file with the data set
     xbatches = np.array_split(x, cpu) # split data across cpus
     xsub[0:len(xbatches[0])] = xbatches[0] # thread that rank 0 processes
     for i in range(1, cpu) : # distribute the segments
         # when Send is upper-case use buffers
         comm.Send(xbatches[i], dest=i, tag=0) # send a segment
else : # the other processes (ranks) receive the segments
     comm.Recv(xsub, source=0, tag=0)

# calculate the starting and ending indexes of each segment, for each rank
start = 0
if rank == cpu - 1 : # the last rank
     end = mseg # keep the largest segment
else :
     end = sseg # index of the end of the segment

# all ranks process your segment
# xprocessedsub gets one dimension less (msec, 256, 256)
xprocessedsub = np.zeros(xsub.shape[:-1])
# repeat the loop 10x, just for timing purposes
for j in range(0,10) :
     for i in range(start, end) :
         # grayscale
         ## xsub[i][...,:3] selects the image (256, 256, 3)
         ## np.dot does the multiplication and addition to convert
         img_gray = np.dot(xsub[i][...,:3], [0.299, 0.587, 0.114])
         # Normalization
         img_gray_norm = img_gray / (img_gray.max() + 1)
         xprocessedsub[i,...] = img_gray_norm
# xprocessedsub contains the processed segment. The shape is (msec, 256, 256)
# rank 0 copies directly to the final dataset
if rank == 0 :
     xprocessed[0:len(xprocessedsub)]=xprocessedsub
# the other ranks return the processed segment to rank 0
else :
     comm.Send(xprocessedsub, dest=0, tag=rank) # tag identifies who sent
# rank 0 takes the segments and combines them into a single dataset
# xprocessedsub from rank 0 has already been copied and now serves as storage. temporary
if rank == 0 :
     for i in range(1, cpu) :
         status = MPI.Status()
         # get a segment
         comm.Recv(xprocessedsub, source=MPI.ANY_SOURCE, tag=MPI.ANY_TAG, status=status)
         rnk_sender = status.Get_source()
         start=rnk_sender *sseg # index to the corresponding position
         slen = sec
         # copy to final dataset
         # this part of the code can be improved
         xprocessed[start : start + len(xprocessedsub)] = xprocessedsub
     # final shape including the channel, which in the case of gray is 1
     xprocessed.reshape(xprocessed.shape + (1,))
     # write to a file for later use
     #np.save("data/map03.npy",xprocessed)
     # each rank shows elapsed time
     print('Rank =', rank, ' Elapsed time =', MPI.Wtime() - wt, 's')
