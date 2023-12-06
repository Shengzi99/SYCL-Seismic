source /opt/spack/share/spack/setup-env.sh
spack load gcc@11.3.0
spack load cuda@11.8.0%gcc@=11.3.0
spack load numactl@2.0.14%gcc@=11.3.0
spack load libevent@2.1.12%gcc@=12.2.0

export DPCPP_HOME=/home/shenzitao/sycl_workspace
export LD_LIBRARY_PATH=$DPCPP_HOME/llvm/build/lib:$LD_LIBRARY_PATH
export LD_LIBRARY_PATH=/usr/lib/x86_64-linux-gnu/:$LD_LIBRARY_PATH

source ~/intel/oneapi/setvars.sh
export PATH=$DPCPP_HOME/llvm/build/bin:$PATH

export PROJHOME=/home/shenzitao/opt/proj-8.0.1
export UCXHOME=/home/shenzitao/opt/ucx-1.13.1
export PATH=$UCXHOME/bin:$PROJHOME/bin:$PATH
export LD_LIBRARY_PATH=$UCXHOME/lib:$PROJHOME/lib:$LD_LIBRARY_PATH

# export MPIHOME=/home/shenzitao/opt/openmpi-4.1.6-cuda
# export PATH=$MPIHOME/bin:$PATH
# export LD_LIBRARY_PATH=$MPIHOME/lib:$LD_LIBRARY_PATH

# spack load openmpi@4.0.5