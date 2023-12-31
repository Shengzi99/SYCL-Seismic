#================================================================
#   Copyright (C) 2021 Sangfor Ltd. All rights reserved.
#   
#   File Name：Makefile
#   Author: Wenqiang Wang
#   Created Time:2021-10-30
#   Discription:
#
#================================================================

# GPU_CUDA := ON
SYCL := ON

LayerMedium := #ON


FREE_SURFACE := ON
PML := ON
SOLVE_DISPLACEMENT := ON

# FLOAT16 := ON
SRCDIR := ./src_wave_con_med

CCHOME := /home/shenzitao/sycl_workspace/llvm/build/
# CCHOME := /home/shenzitao/intel/oneapi/compiler/2023.2.1/linux/
# CCHOME := /home/spack/spack/opt/spack/linux-debian12-zen2/gcc-12.2.0/gcc-11.3.0-ohhhwxjn5z4gq2lzccitxtcxv47dqigj
CUDAHOME := /home/spack/spack/opt/spack/linux-debian12-zen2/gcc-11.3.0/cuda-11.8.0-ehz25mlcpxpnrliwufrqvphysjy6gv5d
MPIHOME := /home/shenzitao/opt/openmpi-4.1.6-cuda
MPIHOME := /home/shenzitao/intel/oneapi/mpi/2021.10.0
PROJHOME := /home/shenzitao/opt/proj-8.0.1



# CC := $(CCHOME)/bin/gcc -pipe
CC := $(CCHOME)/bin/clang -pipe
# CC := $(CCHOME)/bin/icpx -pipe


#General Compiler
ifdef GPU_CUDA
GC := $(CUDAHOME)/bin/nvcc -rdc=true -maxrregcount=127 -arch=sm_70 #-Xptxas=-v 
else
# GC := $(CCHOME)/bin/g++ -pipe
GC := $(CCHOME)/bin/clang++ -pipe
# GC := $(CCHOME)/bin/icpx -pipe
endif


LIBS := -L$(CUDAHOME)/lib64 -lcudart -lcublas
INCS := -I$(CUDAHOME)/include 

LIBS += -L$(MPIHOME)/lib -lmpi
INCS += -I$(MPIHOME)/include 


LIBS += -L$(PROJHOME)/lib -lproj
INCS += -I$(PROJHOME)/include  



OBJDIR := ./obj/CGFDM3D-CJMVS
BINDIR := ./bin


CFLAGS := -c
LFLAGS :=

GCFLAGS := 

ifdef GPU_CUDA
#LFLAGS += -Xptxas=-v 

#LFLAGS += -arch=sm_70 -rdc=true -Xptxas=-v 
#GCFLAGS += --fmad=false 
GCFLAGS += -x cu
endif

vpath

vpath % $(SRCDIR)
vpath % $(OBJDIR)
vpath % $(BINDIR)


DFLAGS_LIST := GPU_CUDA FLOAT16 \
			   FREE_SURFACE PML SOLVE_DISPLACEMENT LayerMedium \
			   SYCL


DFLAGS := $(foreach flag,$(DFLAGS_LIST),$(if $($(flag)),-D$(flag)))


OBJS := cjson.o printInfo.o create_dir.o readParams.o \
		init_gpu.o cpu_Malloc.o init_grid.o init_MPI.o \
		run.o \
		coord.o terrain.o medium.o dealMedium.o crustMedium.o calc_CFL.o\
		contravariant.o \
		wave_deriv.o wave_rk.o freeSurface.o \
		init_pml_para.o pml_deriv.o pml_freeSurface.o \
		propagate.o \
		pml_rk.o \
		singleSource.o multiSource.o\
		data_io.o \
		MPI_send_recv.o MPI_send_recv_jac.o \
		PGV.o station.o\
		main.o MPI_send_recv_fp32.o

ifdef SYCL
OBJS += medium_sycl.o\
		contravariant_sycl.o\
		wave_deriv_sycl.o wave_rk_sycl.o freeSurface_sycl.o\
 	    init_pml_para_sycl.o pml_deriv_sycl.o pml_freeSurface_sycl.o\
		pml_rk_sycl.o


LFLAGS += -fno-fast-math -fsycl -fsycl-targets=nvptx64-nvidia-cuda,spir64_x86_64 
GCFLAGS += -fno-fast-math -fsycl -fsycl-targets=nvptx64-nvidia-cuda,spir64_x86_64 
CFLAGS += -fno-fast-math -fsycl -fsycl-targets=nvptx64-nvidia-cuda,spir64_x86_64 
endif


OBJS := $(addprefix $(OBJDIR)/,$(OBJS))


$(BINDIR)/CGFDM3D-CJMVS: $(OBJS)
	$(GC) $(LFLAGS) $(LIBS) $^ -o $@


$(OBJDIR)/%.o : $(SRCDIR)/%.cpp
	$(GC) $(CFLAGS) $(DFLAGS) $(GCFLAGS) $(INCS)  $^ -o $@

$(OBJDIR)/%.o : $(SRCDIR)/%.c
	$(CC) $(CFLAGS) $^ -o $@

$(OBJDIR)/%.o : $(SRCDIR)/device_sycl/%.cpp 
	$(GC) $(CFLAGS) $(DFLAGS) $(GCFLAGS) $(INCS)  $^ -o $@

env_test : env_test.cpp
	$(GC) $(CFLAGS) $(DFLAGS) $(GCFLAGS) $(INCS)  $^ -o $@.o
	$(GC) $(LFLAGS) $(LIBS) $@.o -o $@

clean:
	-rm $(OBJDIR)/* -rf
	-rm $(BINDIR)/* -rf
	-rm output -rf
