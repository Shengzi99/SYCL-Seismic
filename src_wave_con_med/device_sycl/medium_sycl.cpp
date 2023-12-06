#include "header_sycl.h"

void allocMedium_sycl( GRID grid, MEDIUM * medium, sycl::queue &Q )
{
	int _nx_ = grid._nx_;
	int _ny_ = grid._ny_;
	int _nz_ = grid._nz_;
	
	long long num = _nx_ * _ny_ * _nz_; 

	float * pMedium = NULL;
	long long size = sizeof( float ) * num * MEDIUMSIZE;

    pMedium = sycl::malloc_device<float>(size, Q);
    Q.memset(pMedium, 0, size);
	
	medium->mu       = pMedium;
	medium->lambda   = pMedium + num;
	medium->buoyancy = pMedium + num * 2;
    Q.wait();
}
void freeMedium_sycl( MEDIUM medium, sycl::queue &Q )
{	
	sycl::free(medium.mu, Q);
}
void allocMediumFLOAT_sycl( GRID grid, MEDIUM_FLOAT * medium, sycl::queue &Q )
{
	int _nx_ = grid._nx_;
	int _ny_ = grid._ny_;
	int _nz_ = grid._nz_;
	
	long long num = _nx_ * _ny_ * _nz_; 

	FLOAT * pMedium = NULL;
	long long size = sizeof( FLOAT ) * num * MEDIUMSIZE;

    pMedium = sycl::malloc_device<FLOAT>(size, Q);
    Q.memset(pMedium, 0, size);
	
	medium->mu       = pMedium;
	medium->lambda   = pMedium + num;
	medium->buoyancy = pMedium + num * 2;
    Q.wait();
}
void freeMediumFLOAT_sycl( MEDIUM_FLOAT medium, sycl::queue &Q )
{	
	sycl::free(medium.mu, Q);
}
void memcpyMed_FLOAT_sycl(GRID grid, MEDIUM_FLOAT medium, MEDIUM_FLOAT medium_device, int direction, sycl::queue &Q)
{
	long long size = sizeof( float ) * (grid._nx_ * grid._ny_ * grid._nz_) * MEDIUMSIZE;
	if(direction == HOST_TO_DEVICE)
		Q.memcpy(medium_device.mu, medium.mu, size);
	else
		Q.memcpy(medium.mu, medium_device.mu, size);
	Q.wait();
}