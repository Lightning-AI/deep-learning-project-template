#!/bin/bash
#SBATCH --partition=compute-od-gpu
#SBATCH --job-name=pl_test
#SBATCH --nodes=10
#SBATCH --exclusive
#SBATCH --output=%x_%j.out

export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/opt/nccl/build/lib:/opt/aws-ofi-nccl-install/lib
export NCCL_PROTO=simple
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/opt/aws-ofi-nccl/lib
export PATH=$PATH:/opt/amazon/efa/bin:/opt/amazon/openmpi/bin
export FI_EFA_FORK_SAFE=1
export FI_LOG_LEVEL=1
export FI_EFA_USE_DEVICE_RDMA=1 # use for p4dn
export NCCL_DEBUG=info
export OMPI_MCA_mtl_base_verbose=1
export FI_EFA_ENABLE_SHM_TRANSFER=0
export FI_PROVIDER=efa
export FI_EFA_TX_MIN_CREDITS=64
export NCCL_TREE_THRESHOLD=0

srun /home/knoriy/fsx/miniconda3/envs/pl/bin/python /home/knoriy/deep-learning-project-template/project/lit_mnist.py --accelerator gpu --strategy ddp
