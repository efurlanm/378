# CAP-378 Trabalho: PAD em PDI
# Uso: mpiexec -n <NTASKS> python3 padempdi.py

import numpy as np
from mpi4py import MPI

wt = MPI.Wtime()        # "wall time" para cálculo do tempo decorrido
comm = MPI.COMM_WORLD   # comunicador global (pode servir para definir grupos)
cpu = comm.Get_size()   # total de ranks que o mpi atribui
rank = comm.Get_rank()  # rank é o no. que o mpi atribui ao processo

xlen = 118                              # total de imagens
sseg = int( xlen / cpu )                # tamanho de um segmento
mseg = sseg + ( xlen % cpu )            # tamanho do maior segmento

# - 256, 256 é a imagem, e 4 é a qtde de canais.
# - The different color bands/channels are stored in the third dimension, such
#   that a gray-image is MxN, an RGB-image MxNx3 and an RGBA-image MxNx4.
# - RGBA = Reg, Green, Blue, Alpha(transparência) → PNG
# - Cinza tem só (256, 256) que corresponde ao tamanho da imagem.
xsub = np.zeros((mseg, 256, 256, 4), dtype=np.float32)    # área de trabalho
xprocessed = np.zeros((xlen, 256, 256), dtype=np.float32)   # resultado
    
# O processo (rank) 0 lê o arquivo e distribui os segmentos para os ranks
# O rank 0 também processa um segmento
if rank == 0 :
    x = np.load("data/map01.npy")       # lê o arquivo com o conjunto de dados
    xbatches = np.array_split(x, cpu)   # divide os dados entre as cpus
    xsub[0:len(xbatches[0])] = xbatches[0]                  # segmento que o rank 0 processa
    for i in range(1, cpu) :            # distribui os segmentos
        # quando Send é upper-case usa buffers
        comm.Send(xbatches[i], dest=i, tag=0)   # envia um segmento
else :      # os demais processos (ranks) recebem os segmentos
    comm.Recv(xsub, source=0, tag=0)

# calcula os índices inicial e final de cada segmento, para cada rank
start = 0
if rank == cpu - 1 :            # o último rank
    end = mseg                  # fica com o maior segmento
else :
    end = sseg                  # índice do final do segmento

# todos os ranks processam o seu segmento
# xprocessedsub fica com uma dimensão a menos (mseg, 256, 256)
xprocessedsub = np.zeros(xsub.shape[:-1])
# repete 10x o looping, apenas para fins de medição de tempo
for j in range(0,10) :
    for i in range(start, end) :
        # Grayscale
        ## xsub[i][...,:3] seleciona a imagem (256, 256, 3)
        ## np.dot faz a multiplicação e soma para converter 
        img_gray = np.dot(xsub[i][...,:3], [0.299, 0.587, 0.114])
        # Normalization
        img_gray_norm = img_gray / (img_gray.max() + 1)
        xprocessedsub[i,...] = img_gray_norm
# xprocessedsub contém o segmento processado. O shape é (mseg, 256, 256)
# o rank 0 copia direto para o dataset final
if rank == 0 :
    xprocessed[0:len(xprocessedsub)]=xprocessedsub
# os demais ranks retornam o segmento processado para o rank 0
else :
    comm.Send(xprocessedsub, dest=0, tag=rank)    # tag identifica quem mandou
# o rank 0 recebe os segmentos e os combina em um único dataset
# xprocessedsub do rank 0 já foi copiado e agora serve como armazen. temporário
if rank == 0 :
    for i in range(1, cpu) :
        status = MPI.Status()
        # recebe um segmento
        comm.Recv(xprocessedsub, source=MPI.ANY_SOURCE, tag=MPI.ANY_TAG, status=status)
        rnk_sender = status.Get_source()
        start= rnk_sender * sseg        # índice para a posição correspondente
        slen = sseg
        # copia para o dataset final
        # essa parte do codigo pode ser melhorada
        xprocessed[start : start + len(xprocessedsub)] = xprocessedsub
    # shape final incluindo o canal, que no caso de grey é 1
    xprocessed.reshape(xprocessed.shape + (1,))
    #grava em um arquivo para uso posterior
    #np.save("data/map03.npy",xprocessed)
    # cada rank mostra o tempo decorrido
    print('Rank =', rank, '  Elapsed time =', MPI.Wtime() - wt, 's')
