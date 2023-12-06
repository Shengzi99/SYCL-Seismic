#include "header_sycl.h"

#define P_alpha 1.0f
#define P_beta 2.0f
#define P_d 2.0f

#define cal_pml_alpha( l ) ( ( ( l ) < 0.0f ) ? 0.0f : ( pml_alpha0 * ( 1.0f - pow( ( l ) / L, P_alpha ) ) ) )
#define cal_pml_beta( l )  ( ( ( l ) < 0.0f ) ? 1.0f : ( 1.0f + ( pml_beta0 - 1.0f ) * pow( ( l ) / L, P_beta ) ) )
#define cal_pml_d( l )     ( ( ( l ) < 0.0f ) ? 0.0f : ( d0 * pow( ( l ) / L, P_d ) ) )

void allocPMLParameter_sycl( GRID grid, PML_ALPHA * pml_alpha, PML_BETA *pml_beta, PML_D * pml_d, sycl::queue &Q )
{
	int _nx_ = grid._nx_;
	int _ny_ = grid._ny_;
	int _nz_ = grid._nz_;

	float * pParams = NULL;
	long long size = sizeof( float ) * ( _nx_ + _ny_ + _nz_ ) * 3;

    pParams = sycl::malloc_device<float>(size, Q);
    Q.memset(pParams, 0, size);
	
	pml_alpha->x = pParams + 0 * _nx_;
	pml_beta->x  = pParams + 1 * _nx_;
	pml_d->x     = pParams + 2 * _nx_; 

	pParams = pParams + 3 * _nx_;

	pml_alpha->y = pParams + 0 * _ny_;
	pml_beta->y  = pParams + 1 * _ny_;
	pml_d->y     = pParams + 2 * _ny_;

	pParams = pParams + 3 * _ny_;

	pml_alpha->z = pParams + 0 * _nz_;
	pml_beta->z  = pParams + 1 * _nz_;
	pml_d->z     = pParams + 2 * _nz_;

    Q.wait();
}
void freePMLParamter_sycl( PML_ALPHA pml_alpha, PML_BETA pml_beta, PML_D pml_d, sycl::queue &Q )
{
    sycl::free(pml_alpha.x, Q);
}
void memcpyPMLParam_sycl(GRID grid, PML_ALPHA pml_alpha, PML_ALPHA pml_alpha_device, int direction, sycl::queue &Q)
{
    long long size = sizeof( float ) * ( grid._nx_ + grid._ny_ + grid._nz_ ) * 3;
    if(direction == HOST_TO_DEVICE)
        Q.memcpy(pml_alpha_device.x, pml_alpha.x, size);
    else
        Q.memcpy(pml_alpha.x, pml_alpha_device.x, size);
    Q.wait();
}