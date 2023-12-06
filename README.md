# SYCL-Seismic
migrated FP16-LSRK using standard SYCL



## 坑！

1. 运行CUDA版本需要使用CUDA Aware MPI  
2. 如果SYCL找不到 CPU device，可能是MPI版本问题  
3. 如果SYCL找得到 CPU device，但```get_info<sycl::info::device::max_compute_units>()```结果和硬件的核心数不符（表现为Kernel跑得特别慢，开top看CPU利用率很低），也可能是MPI版本问题  
