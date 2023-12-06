#include "header_sycl.h"

void allocContravariant_sycl( GRID grid, CONTRAVARIANT * con, sycl::queue &Q )
{
	int _nx_ = grid._nx_;
	int _ny_ = grid._ny_;
	int _nz_ = grid._nz_;
	
	long long num = _nx_ * _ny_ * _nz_; 
	
	float * pContravariant = NULL;
	long long size = sizeof( float ) * num * CONTRASIZE;

    pContravariant = sycl::malloc_device<float>(size, Q);
    Q.memset(pContravariant, 0, size);

	con->xi_x = pContravariant + num * 0;
	con->xi_y = pContravariant + num * 1;
	con->xi_z = pContravariant + num * 2;
                                                  
	con->et_x = pContravariant + num * 3;
	con->et_y = pContravariant + num * 4;
	con->et_z = pContravariant + num * 5;
                                                  
	con->zt_x = pContravariant + num * 6;
	con->zt_y = pContravariant + num * 7;
	con->zt_z = pContravariant + num * 8;
    Q.wait();
}
void freeContravariant_sycl( CONTRAVARIANT con, sycl::queue &Q )
{	
	sycl::free(con.xi_x, Q);
}
void memcpyCon_sycl( GRID grid, CONTRAVARIANT con, CONTRAVARIANT con_device, int direction, sycl::queue &Q)
{
    long long size = sizeof( float ) * grid._nx_ * grid._ny_ * grid._nz_ * CONTRASIZE;
    if(direction == HOST_TO_DEVICE)
        Q.memcpy(con_device.xi_x, con.xi_x, size);
    else
        Q.memcpy(con.xi_x, con_device.xi_x, size);
    Q.wait();
}
void allocContravariant_FLOAT_sycl( GRID grid, CONTRAVARIANT_FLOAT * con, sycl::queue &Q )
{
	int _nx_ = grid._nx_;
	int _ny_ = grid._ny_;
	int _nz_ = grid._nz_;
	
	long long num = _nx_ * _ny_ * _nz_; 

	FLOAT * pContravariant = NULL;
	long long size = sizeof( FLOAT ) * num * CONTRASIZE;

    pContravariant = sycl::malloc_device<FLOAT>(size, Q);
    Q.memset(pContravariant, 0, size);

	con->xi_x = pContravariant + num * 0;
	con->xi_y = pContravariant + num * 1;
	con->xi_z = pContravariant + num * 2;
                                                  
	con->et_x = pContravariant + num * 3;
	con->et_y = pContravariant + num * 4;
	con->et_z = pContravariant + num * 5;
                                                  
	con->zt_x = pContravariant + num * 6;
	con->zt_y = pContravariant + num * 7;
	con->zt_z = pContravariant + num * 8;
    Q.wait();
}
void freeContravariant_FLOAT_sycl( CONTRAVARIANT_FLOAT con, sycl::queue &Q )
{	
	sycl::free(con.xi_x, Q);
}
void memcpyCon_FLOAT_sycl( GRID grid, CONTRAVARIANT_FLOAT con, CONTRAVARIANT_FLOAT con_device, int direction, sycl::queue &Q)
{
    long long size = sizeof( FLOAT ) * grid._nx_ * grid._ny_ * grid._nz_ * CONTRASIZE;
    if(direction == HOST_TO_DEVICE)
        Q.memcpy(con_device.xi_x, con.xi_x, size);
    else
        Q.memcpy(con.xi_x, con_device.xi_x, size);
    Q.wait();
}


void allocJac_sycl( GRID grid, float ** Jac, sycl::queue &Q )
{
	int _nx_ = grid._nx_;
	int _ny_ = grid._ny_;
	int _nz_ = grid._nz_;
	
	long long num = _nx_ * _ny_ * _nz_; 

	float * pJac = NULL;
	long long size = sizeof( float ) * num;

    pJac = sycl::malloc_device<float>(size, Q);
    Q.memset(pJac, 0, size);
	
	*Jac = pJac;
    Q.wait();
}
void freeJac_sycl( float * Jac, sycl::queue &Q )
{
	sycl::free(Jac, Q);
}
void memcpyJac_sycl(GRID grid, float *Jac, float *Jac_device, int direction, sycl::queue &Q)
{
    long long size = sizeof( float ) * grid._nx_ * grid._ny_ * grid._nz_;
    if(direction == HOST_TO_DEVICE)
        Q.memcpy(Jac_device, Jac, size);
    else
        Q.memcpy(Jac, Jac_device, size);
    Q.wait();
}
void allocJac_FLOAT_sycl( GRID grid, FLOAT ** Jac, sycl::queue &Q )
{
	int _nx_ = grid._nx_;
	int _ny_ = grid._ny_;
	int _nz_ = grid._nz_;
	
	long long num = _nx_ * _ny_ * _nz_; 

	FLOAT * pJac = NULL;
	long long size = sizeof( FLOAT ) * num;

    pJac = sycl::malloc_device<FLOAT>(size, Q);
    Q.memset(pJac, 0, size);
	
	*Jac = pJac;
    Q.wait();
}
void freeJac_FLOAT_sycl( FLOAT * Jac, sycl::queue &Q )
{
	sycl::free(Jac, Q);
}
void memcpyJac_FLOAT_sycl(GRID grid, FLOAT *Jac, FLOAT *Jac_device, int direction, sycl::queue &Q)
{
    long long size = sizeof( FLOAT ) * grid._nx_ * grid._ny_ * grid._nz_;
    if(direction == HOST_TO_DEVICE)
        Q.memcpy(Jac_device, Jac, size);
    else
        Q.memcpy(Jac, Jac_device, size);
    Q.wait();
}


void allocMat3x3_sycl( GRID grid, Mat3x3 * _rDZ_DX, Mat3x3 * _rDZ_DY, sycl::queue &Q )
{
	int _nx_ = grid._nx_;
	int _ny_ = grid._ny_;
	
	long long num = _nx_ * _ny_; 
		
	float * pSurf = NULL;
	long long size = sizeof( float ) * num * 2 * 9;

	pSurf = sycl::malloc_device<float>(size, Q);
	Q.memset(pSurf, 0, size);
	
	_rDZ_DX->M11 = pSurf + 0 * num; 
	_rDZ_DX->M12 = pSurf + 1 * num; 
	_rDZ_DX->M13 = pSurf + 2 * num;
	_rDZ_DX->M21 = pSurf + 3 * num; 
	_rDZ_DX->M22 = pSurf + 4 * num; 
	_rDZ_DX->M23 = pSurf + 5 * num;
	_rDZ_DX->M31 = pSurf + 6 * num; 
	_rDZ_DX->M32 = pSurf + 7 * num; 
	_rDZ_DX->M33 = pSurf + 8 * num;

	pSurf		 = pSurf + 9 * num; 

	_rDZ_DY->M11 = pSurf + 0 * num; 
	_rDZ_DY->M12 = pSurf + 1 * num; 
	_rDZ_DY->M13 = pSurf + 2 * num;
	_rDZ_DY->M21 = pSurf + 3 * num; 
	_rDZ_DY->M22 = pSurf + 4 * num; 
	_rDZ_DY->M23 = pSurf + 5 * num;
	_rDZ_DY->M31 = pSurf + 6 * num; 
	_rDZ_DY->M32 = pSurf + 7 * num; 
	_rDZ_DY->M33 = pSurf + 8 * num;
	
	Q.wait();
}
void freeMat3x3_sycl( Mat3x3 _rDZ_DX, Mat3x3 _rDZ_DY, sycl::queue &Q )
{
	sycl::free(_rDZ_DX.M11, Q);
}
void memcpyMat3x3_sycl(GRID grid, Mat3x3 _rDZ_DX, Mat3x3 _rDZ_DX_device, int direction, sycl::queue &Q)
{
	long long size = sizeof( float ) * (grid._nx_ * grid._ny_) * 2 * 9;
	if(direction == HOST_TO_DEVICE)
		Q.memcpy(_rDZ_DX_device.M11, _rDZ_DX.M11, size);
	else
		Q.memcpy(_rDZ_DX.M11, _rDZ_DX_device.M11, size);
	Q.wait();
}
