/*================================================================
*   ESS, Southern University of Science and Technology
*   
*   File Name:propagate.cpp
*   Author: Wenqiang Wang, 11849528@mail.sustech.edu.cn
*   Created Time:2021-11-04
*   Discription:
*
================================================================*/
#include "header.h"
#ifdef SYCL
	#include "sycl/sycl.hpp"
#endif

void allocAuxPML( int nPML, int N1, int N2, AUX * h_Aux, AUX * Aux, AUX * t_Aux, AUX * m_Aux )
{
	
	long long num = nPML * N1 * N2; 
		
	float * pAux = NULL;
	long long size = sizeof( float ) * num * 9 * 4;
	
	//printf("num = %ld, nPML = %d, N1 = %d, N2 = %d\n", num, nPML, N1, N2  );


	CHECK( Malloc( ( void ** )&pAux, size ) );
	CHECK( Memset(  pAux, 0, size ) ); 
	//printf("pAux1 = %d\n", pAux );

	h_Aux->Vx  = pAux + 0 * num;
	h_Aux->Vy  = pAux + 1 * num;
	h_Aux->Vz  = pAux + 2 * num;
	h_Aux->Txx = pAux + 3 * num;
	h_Aux->Tyy = pAux + 4 * num;
	h_Aux->Tzz = pAux + 5 * num;
	h_Aux->Txy = pAux + 6 * num;
	h_Aux->Txz = pAux + 7 * num;
	h_Aux->Tyz = pAux + 8 * num;

	pAux 	 = pAux + 9 * num;
	//printf("pAux2 = %d\n", pAux );

	Aux->Vx  = pAux + 0 * num;
	Aux->Vy  = pAux + 1 * num;
	Aux->Vz  = pAux + 2 * num;
	Aux->Txx = pAux + 3 * num;
	Aux->Tyy = pAux + 4 * num;
	Aux->Tzz = pAux + 5 * num;
	Aux->Txy = pAux + 6 * num;
	Aux->Txz = pAux + 7 * num;
	Aux->Tyz = pAux + 8 * num;

	pAux    = pAux + 9 * num;
	//printf("pAux3 = %d\n", pAux );

	t_Aux->Vx  = pAux + 0 * num;
	t_Aux->Vy  = pAux + 1 * num;
	t_Aux->Vz  = pAux + 2 * num;
	t_Aux->Txx = pAux + 3 * num;
	t_Aux->Tyy = pAux + 4 * num;
	t_Aux->Tzz = pAux + 5 * num;
	t_Aux->Txy = pAux + 6 * num;
	t_Aux->Txz = pAux + 7 * num;
	t_Aux->Tyz = pAux + 8 * num;

	pAux 	 = pAux + 9 * num;

	//printf("pAux4 = %d\n", pAux );

	m_Aux->Vx  = pAux + 0 * num;
	m_Aux->Vy  = pAux + 1 * num;
	m_Aux->Vz  = pAux + 2 * num;
	m_Aux->Txx = pAux + 3 * num;
	m_Aux->Tyy = pAux + 4 * num;
	m_Aux->Tzz = pAux + 5 * num;
	m_Aux->Txy = pAux + 6 * num;
	m_Aux->Txz = pAux + 7 * num;
	m_Aux->Tyz = pAux + 8 * num;
	//printf("pAux4 = %d\n", pAux );

}


void allocPML( GRID grid, AUX4 *Aux4_1, AUX4 *Aux4_2, MPI_BORDER border )
{
	
	int nx = grid.nx;
	int ny = grid.ny;
	int nz = grid.nz;

	
	int nPML = grid.nPML;

	//printf( "nx = %d, nPML = %d\n", nx, nPML );


	memset( ( void * )Aux4_1, 0, sizeof( AUX4 ) );
	memset( ( void * )Aux4_2, 0, sizeof( AUX4 ) );


	if ( border.isx1 && nPML >= nx ) { printf( "The PML layer(%d) just bigger than nx(%d)\n", nPML, nx );  MPI_Abort( MPI_COMM_WORLD, 130 );}
	if ( border.isy1 && nPML >= ny ) { printf( "The PML layer(%d) just bigger than ny(%d)\n", nPML, ny );  MPI_Abort( MPI_COMM_WORLD, 130 );}
	if ( border.isz1 && nPML >= nz ) { printf( "The PML layer(%d) just bigger than nz(%d)\n", nPML, nz );  MPI_Abort( MPI_COMM_WORLD, 130 );}
                                                                                                                                     
	if ( border.isx2 && nPML >= nx ) { printf( "The PML layer(%d) just bigger than nx(%d)\n", nPML, nx );  MPI_Abort( MPI_COMM_WORLD, 130 );}
	if ( border.isy2 && nPML >= ny ) { printf( "The PML layer(%d) just bigger than ny(%d)\n", nPML, ny );  MPI_Abort( MPI_COMM_WORLD, 130 );}
	if ( border.isz2 && nPML >= nz ) { printf( "The PML layer(%d) just bigger than nz(%d)\n", nPML, nz );  MPI_Abort( MPI_COMM_WORLD, 130 );}

	if ( border.isx1 ) allocAuxPML( nPML, ny, nz, &( Aux4_1->h_Aux_x ), &( Aux4_1->Aux_x ), &( Aux4_1->t_Aux_x ), &( Aux4_1->m_Aux_x ) );
	if ( border.isy1 ) allocAuxPML( nPML, nx, nz, &( Aux4_1->h_Aux_y ), &( Aux4_1->Aux_y ), &( Aux4_1->t_Aux_y ), &( Aux4_1->m_Aux_y ) );
	if ( border.isz1 ) allocAuxPML( nPML, nx, ny, &( Aux4_1->h_Aux_z ), &( Aux4_1->Aux_z ), &( Aux4_1->t_Aux_z ), &( Aux4_1->m_Aux_z ) );
                                                                              
	if ( border.isx2 ) allocAuxPML( nPML, ny, nz, &( Aux4_2->h_Aux_x ), &( Aux4_2->Aux_x ), &( Aux4_2->t_Aux_x ), &( Aux4_2->m_Aux_x ) );
	if ( border.isy2 ) allocAuxPML( nPML, nx, nz, &( Aux4_2->h_Aux_y ), &( Aux4_2->Aux_y ), &( Aux4_2->t_Aux_y ), &( Aux4_2->m_Aux_y ) );

#ifndef FREE_SURFACE
	if ( border.isz2 ) allocAuxPML( nPML, nx, ny, &( Aux4_2->h_Aux_z ), &( Aux4_2->Aux_z ), &( Aux4_2->t_Aux_z ), &( Aux4_2->m_Aux_z ) );
#endif

}


void freePML( MPI_BORDER border,  AUX4 Aux4_1, AUX4 Aux4_2 )
{

	if ( border.isx1 )  Free( Aux4_1.h_Aux_x.Vx );
	if ( border.isy1 )  Free( Aux4_1.h_Aux_y.Vx );
	if ( border.isz1 )  Free( Aux4_1.h_Aux_z.Vx );
	                  
	if ( border.isx2 )  Free( Aux4_2.h_Aux_x.Vx );
	if ( border.isy2 )  Free( Aux4_2.h_Aux_y.Vx );
#ifndef FREE_SURFACE
	if ( border.isz2 )  Free( Aux4_2.h_Aux_z.Vx );
#endif
}


__GLOBAL__														
void pml_deriv_x( 									
	WAVE h_W, WAVE W, AUXILIARY h_Aux_x, AUXILIARY Aux_x,		
	FLOAT * XI_X, FLOAT * XI_Y, FLOAT * XI_Z, MEDIUM_FLOAT medium, 	
	float * pml_alpha_x, float * pml_beta_x, float * pml_d_x, 	
	int nPML, int _nx_, int _ny_, int _nz_, int FLAG, float rDH, int FB1, float DT )				
{																
#ifdef GPU_CUDA
	int i0 = threadIdx.x + blockIdx.x * blockDim.x;
	int j0 = threadIdx.y + blockIdx.y * blockDim.y;
	int k0 = threadIdx.z + blockIdx.z * blockDim.z;

#else
	int i0 = 0;
	int j0 = 0;
	int k0 = 0;
#endif
	


	int i, j, k;								
	long long index;
	long long pos;

	int nx = _nx_ - HALO - HALO;
	int ny = _ny_ - HALO - HALO;
	int nz = _nz_ - HALO - HALO;



	float mu = 0.0f;											
	float lambda = 0.0f;										
	float buoyancy = 0.0f;										
																
	float beta_x = 0.0f;										
	float d_x = 0.0f;											
	float alpha_d_x = 0.0f;										
																
	float xi_x = 0.0f; 	float xi_y = 0.0f; 	float xi_z = 0.0f;	
																
	float Txx_xi = 0.0f; float Tyy_xi = 0.0f; float Txy_xi = 0.0f; 
	float Txz_xi = 0.0f; float Tyz_xi = 0.0f; float Tzz_xi = 0.0f;
	float Vx_xi  = 0.0f; float Vy_xi  = 0.0f; float Vz_xi  = 0.0f;


	float Vx1  = 0.0;
	float Vy1  = 0.0;
	float Vz1  = 0.0;
	float Txx1 = 0.0;
	float Tyy1 = 0.0;
	float Tzz1 = 0.0;
	float Txy1 = 0.0;
	float Txz1 = 0.0;
	float Tyz1 = 0.0;


	float h_WVx ;		
	float h_WVy ;		
	float h_WVz ;		
	float h_WTxx;		
	float h_WTyy;		
	float h_WTzz;		
	float h_WTxy;		
	float h_WTxz;		
	float h_WTyz;

																
	int stride = FLAG * ( nx - nPML );				
	CALCULATE3D( i0, j0, k0, 0, nPML, 0, ny, 0, nz )
		i = i0 + HALO + stride;											
		j = j0 + HALO;													
		k = k0 + HALO;													
		index = INDEX(i, j, k);											
		pos	= Index3D( i0, j0, k0, nPML, ny, nz );//i0 + j0 * nPML + k0 * nPML * ny;			
															
		mu = (float)medium.mu[index]; 											
		lambda = (float)medium.lambda[index];							
		buoyancy = (float)medium.buoyancy[index];
		buoyancy *= Crho;
															
		beta_x = pml_beta_x[i]; 							
		d_x = pml_d_x[i];									
		alpha_d_x = d_x + pml_alpha_x[i];

		//if ( i0 == nPML - 10 && j0 == ny - 10 && k0 == nz - 10 )
		//printf( "beta_x = %f, d_x = %f, alpha_d_x = %f\n",  beta_x, d_x, alpha_d_x );


		xi_x = (float)XI_X[index]; xi_y = (float)XI_Y[index]; xi_z = (float)XI_Z[index];
															
		 Vx_xi = L( (float)W.Vx , FB1, xi ) * d_x;					
		 Vy_xi = L( (float)W.Vy , FB1, xi ) * d_x;					
		 Vz_xi = L( (float)W.Vz , FB1, xi ) * d_x;					
		Txx_xi = L( (float)W.Txx, FB1, xi ) * d_x;					
		Tyy_xi = L( (float)W.Tyy, FB1, xi ) * d_x;					
		Tzz_xi = L( (float)W.Tzz, FB1, xi ) * d_x;					
		Txy_xi = L( (float)W.Txy, FB1, xi ) * d_x;					
		Txz_xi = L( (float)W.Txz, FB1, xi ) * d_x;					
		Tyz_xi = L( (float)W.Tyz, FB1, xi ) * d_x;	


		Vx1  = DOT_PRODUCT3D( xi_x, xi_y, xi_z, Txx_xi, Txy_xi, Txz_xi ) * buoyancy;
		Vy1  = DOT_PRODUCT3D( xi_x, xi_y, xi_z, Txy_xi, Tyy_xi, Tyz_xi ) * buoyancy;
		Vz1  = DOT_PRODUCT3D( xi_x, xi_y, xi_z, Txz_xi, Tyz_xi, Tzz_xi ) * buoyancy;

		Txx1 = DOT_PRODUCT3D( xi_x, xi_y, xi_z, Vx_xi, Vy_xi, Vz_xi ) * lambda + 2.0f * mu * ( xi_x * Vx_xi );
		Tyy1 = DOT_PRODUCT3D( xi_x, xi_y, xi_z, Vx_xi, Vy_xi, Vz_xi ) * lambda + 2.0f * mu * ( xi_y * Vy_xi );
		Tzz1 = DOT_PRODUCT3D( xi_x, xi_y, xi_z, Vx_xi, Vy_xi, Vz_xi ) * lambda + 2.0f * mu * ( xi_z * Vz_xi );

		Txy1 = DOT_PRODUCT2D( xi_y, xi_x, Vx_xi, Vy_xi ) * mu;
		Txz1 = DOT_PRODUCT2D( xi_z, xi_x, Vx_xi, Vz_xi ) * mu;
		Tyz1 = DOT_PRODUCT2D( xi_z, xi_y, Vy_xi, Vz_xi ) * mu;
															
		h_Aux_x.Vx [pos] = ( Vx1  - alpha_d_x * Aux_x.Vx [pos] ) * DT;
		h_Aux_x.Vy [pos] = ( Vy1  - alpha_d_x * Aux_x.Vy [pos] ) * DT;
		h_Aux_x.Vz [pos] = ( Vz1  - alpha_d_x * Aux_x.Vz [pos] ) * DT;
		h_Aux_x.Txx[pos] = ( Txx1 - alpha_d_x * Aux_x.Txx[pos] ) * DT;
		h_Aux_x.Tyy[pos] = ( Tyy1 - alpha_d_x * Aux_x.Tyy[pos] ) * DT;
		h_Aux_x.Tzz[pos] = ( Tzz1 - alpha_d_x * Aux_x.Tzz[pos] ) * DT;
		h_Aux_x.Txy[pos] = ( Txy1 - alpha_d_x * Aux_x.Txy[pos] ) * DT;
		h_Aux_x.Txz[pos] = ( Txz1 - alpha_d_x * Aux_x.Txz[pos] ) * DT;
		h_Aux_x.Tyz[pos] = ( Tyz1 - alpha_d_x * Aux_x.Tyz[pos] ) * DT;
															
		h_WVx  = h_W.Vx [index];		
		h_WVy  = h_W.Vy [index];		
		h_WVz  = h_W.Vz [index];		
		h_WTxx = h_W.Txx[index];		
		h_WTyy = h_W.Tyy[index];		
		h_WTzz = h_W.Tzz[index];		
		h_WTxy = h_W.Txy[index];		
		h_WTxz = h_W.Txz[index];		
		h_WTyz = h_W.Tyz[index];

		h_WVx  += - beta_x * Aux_x.Vx [pos] * DT;		
		h_WVy  += - beta_x * Aux_x.Vy [pos] * DT;		
		h_WVz  += - beta_x * Aux_x.Vz [pos] * DT;		
		h_WTxx += - beta_x * Aux_x.Txx[pos] * DT;		
		h_WTyy += - beta_x * Aux_x.Tyy[pos] * DT;		
		h_WTzz += - beta_x * Aux_x.Tzz[pos] * DT;		
		h_WTxy += - beta_x * Aux_x.Txy[pos] * DT;		
		h_WTxz += - beta_x * Aux_x.Txz[pos] * DT;		
		h_WTyz += - beta_x * Aux_x.Tyz[pos] * DT;

		h_W.Vx [index] = h_WVx ;		
		h_W.Vy [index] = h_WVy ;		
		h_W.Vz [index] = h_WVz ;		
		h_W.Txx[index] = h_WTxx;		
		h_W.Tyy[index] = h_WTyy;		
		h_W.Tzz[index] = h_WTzz;		
		h_W.Txy[index] = h_WTxy;		
		h_W.Txz[index] = h_WTxz;		
		h_W.Tyz[index] = h_WTyz;
										
	END_CALCULATE3D( )											
}

__GLOBAL__
void pml_deriv_y( 
	WAVE h_W, WAVE W, AUXILIARY h_Aux_y, AUXILIARY Aux_y,
	FLOAT * ET_X, FLOAT * ET_Y, FLOAT * ET_Z, MEDIUM_FLOAT medium, 		
	float * pml_alpha_y, float * pml_beta_y, float *  pml_d_y, 		
	int nPML, int _nx_, int _ny_, int _nz_, int FLAG, float rDH, int FB2, float DT )	
{																	

#ifdef GPU_CUDA
	int i0 = threadIdx.x + blockIdx.x * blockDim.x;
	int j0 = threadIdx.y + blockIdx.y * blockDim.y;
	int k0 = threadIdx.z + blockIdx.z * blockDim.z;
#else
	int i0 = 0;
	int j0 = 0;
	int k0 = 0;
#endif

	int i, j, k;									
	long long index;
	long long pos;

	int nx = _nx_ - HALO - HALO;
	int ny = _ny_ - HALO - HALO;
	int nz = _nz_ - HALO - HALO;

	

	float mu = 0.0f;												
	float lambda = 0.0f;											
	float buoyancy = 0.0f;											
																	
	float beta_y = 0.0f;											
	float d_y = 0.0f;												
	float alpha_d_y = 0.0f;											
																	
	float et_x = 0.0f; 	float et_y = 0.0f; 	float et_z = 0.0f;		
																	
	float Txx_et = 0.0f; float Tyy_et = 0.0f; float Txy_et = 0.0f; 
	float Txz_et = 0.0f; float Tyz_et = 0.0f; float Tzz_et = 0.0f;
	float Vx_et  = 0.0f; float Vy_et  = 0.0f; float Vz_et  = 0.0f;

	float Vx2  = 0.0f;
	float Vy2  = 0.0f;
	float Vz2  = 0.0f;
	float Txx2 = 0.0f;
	float Tyy2 = 0.0f;
	float Tzz2 = 0.0f;
	float Txy2 = 0.0f;
	float Txz2 = 0.0f;
	float Tyz2 = 0.0f; 

	float h_WVx ;		
	float h_WVy ;		
	float h_WVz ;		
	float h_WTxx;		
	float h_WTyy;		
	float h_WTzz;		
	float h_WTxy;		
	float h_WTxz;		
	float h_WTyz;

																
																	
	int stride = FLAG * ( ny - nPML );					
	CALCULATE3D( i0, j0, k0, 0, nx, 0, nPML, 0, nz )
		i = i0 + HALO;													
		j = j0 + HALO + stride;											
		k = k0 + HALO;													
		index = INDEX(i, j, k);											
		pos	= Index3D( i0, j0, k0, nx, nPML, nz );//i0 + j0 * nx + k0 * nx * nPML;
																	
		mu = (float)medium.mu[index]; 											
		lambda = (float)medium.lambda[index];							
		buoyancy = (float)medium.buoyancy[index];
		buoyancy *= Crho;
																	
		beta_y = pml_beta_y[j]; 									
		d_y = pml_d_y[j];											
		alpha_d_y = d_y + pml_alpha_y[j]; 							
		
		et_x = (float)ET_X[index]; et_y = (float)ET_Y[index]; et_z = (float)ET_Z[index];
																	
		 Vx_et = L( (float)W.Vx , FB2, et ) * d_y;							
		 Vy_et = L( (float)W.Vy , FB2, et ) * d_y;							
		 Vz_et = L( (float)W.Vz , FB2, et ) * d_y;							
		Txx_et = L( (float)W.Txx, FB2, et ) * d_y;							
		Tyy_et = L( (float)W.Tyy, FB2, et ) * d_y;							
		Tzz_et = L( (float)W.Tzz, FB2, et ) * d_y;							
		Txy_et = L( (float)W.Txy, FB2, et ) * d_y;							
		Txz_et = L( (float)W.Txz, FB2, et ) * d_y;							
		Tyz_et = L( (float)W.Tyz, FB2, et ) * d_y;							

		Vx2  = DOT_PRODUCT3D( et_x, et_y, et_z, Txx_et, Txy_et, Txz_et ) * buoyancy;
		Vy2  = DOT_PRODUCT3D( et_x, et_y, et_z, Txy_et, Tyy_et, Tyz_et ) * buoyancy;
		Vz2  = DOT_PRODUCT3D( et_x, et_y, et_z, Txz_et, Tyz_et, Tzz_et ) * buoyancy;

		Txx2 = DOT_PRODUCT3D( et_x, et_y, et_z, Vx_et, Vy_et, Vz_et ) * lambda + 2.0f * mu * ( et_x * Vx_et );
		Tyy2 = DOT_PRODUCT3D( et_x, et_y, et_z, Vx_et, Vy_et, Vz_et ) * lambda + 2.0f * mu * ( et_y * Vy_et );
		Tzz2 = DOT_PRODUCT3D( et_x, et_y, et_z, Vx_et, Vy_et, Vz_et ) * lambda + 2.0f * mu * ( et_z * Vz_et );

		Txy2 = DOT_PRODUCT2D( et_y, et_x, Vx_et, Vy_et ) * mu;
		Txz2 = DOT_PRODUCT2D( et_z, et_x, Vx_et, Vz_et ) * mu;
		Tyz2 = DOT_PRODUCT2D( et_z, et_y, Vy_et, Vz_et ) * mu;

																	
		h_Aux_y.Vx [pos] = ( Vx2  - alpha_d_y * Aux_y.Vx [pos] ) * DT;		
		h_Aux_y.Vy [pos] = ( Vy2  - alpha_d_y * Aux_y.Vy [pos] ) * DT;		
		h_Aux_y.Vz [pos] = ( Vz2  - alpha_d_y * Aux_y.Vz [pos] ) * DT;		
		h_Aux_y.Txx[pos] = ( Txx2 - alpha_d_y * Aux_y.Txx[pos] ) * DT;		
		h_Aux_y.Tyy[pos] = ( Tyy2 - alpha_d_y * Aux_y.Tyy[pos] ) * DT;		
		h_Aux_y.Tzz[pos] = ( Tzz2 - alpha_d_y * Aux_y.Tzz[pos] ) * DT;		
		h_Aux_y.Txy[pos] = ( Txy2 - alpha_d_y * Aux_y.Txy[pos] ) * DT;		
		h_Aux_y.Txz[pos] = ( Txz2 - alpha_d_y * Aux_y.Txz[pos] ) * DT;		
		h_Aux_y.Tyz[pos] = ( Tyz2 - alpha_d_y * Aux_y.Tyz[pos] ) * DT;		
																	
		h_WVx  = h_W.Vx [index];		
		h_WVy  = h_W.Vy [index];		
		h_WVz  = h_W.Vz [index];		
		h_WTxx = h_W.Txx[index];		
		h_WTyy = h_W.Tyy[index];		
		h_WTzz = h_W.Tzz[index];		
		h_WTxy = h_W.Txy[index];		
		h_WTxz = h_W.Txz[index];		
		h_WTyz = h_W.Tyz[index];

		h_WVx  += - beta_y * Aux_y.Vx [pos] * DT;		
		h_WVy  += - beta_y * Aux_y.Vy [pos] * DT;		
		h_WVz  += - beta_y * Aux_y.Vz [pos] * DT;		
		h_WTxx += - beta_y * Aux_y.Txx[pos] * DT;		
		h_WTyy += - beta_y * Aux_y.Tyy[pos] * DT;		
		h_WTzz += - beta_y * Aux_y.Tzz[pos] * DT;		
		h_WTxy += - beta_y * Aux_y.Txy[pos] * DT;		
		h_WTxz += - beta_y * Aux_y.Txz[pos] * DT;		
		h_WTyz += - beta_y * Aux_y.Tyz[pos] * DT;

		h_W.Vx [index] = h_WVx ;		
		h_W.Vy [index] = h_WVy ;		
		h_W.Vz [index] = h_WVz ;		
		h_W.Txx[index] = h_WTxx;		
		h_W.Tyy[index] = h_WTyy;		
		h_W.Tzz[index] = h_WTzz;		
		h_W.Txy[index] = h_WTxy;		
		h_W.Txz[index] = h_WTxz;		
		h_W.Tyz[index] = h_WTyz;

	END_CALCULATE3D( )												
}

__GLOBAL__														
void pml_deriv_z( 
	WAVE h_W, WAVE W, AUXILIARY h_Aux_z, AUXILIARY Aux_z,	
	FLOAT * ZT_X, FLOAT * ZT_Y, FLOAT * ZT_Z, MEDIUM_FLOAT medium, 	
	float * pml_alpha_z, float * pml_beta_z, float *  pml_d_z,	
	int nPML, int _nx_, int _ny_, int _nz_, int FLAG, float rDH, int FB3, float DT )	
{																

#ifdef GPU_CUDA
	int i0 = threadIdx.x + blockIdx.x * blockDim.x;
	int j0 = threadIdx.y + blockIdx.y * blockDim.y;
	int k0 = threadIdx.z + blockIdx.z * blockDim.z;
#else
	int i0 = 0;
	int j0 = 0;
	int k0 = 0;
#endif

	int i, j, k;									
	long long index;
	long long pos;

	int nx = _nx_ - HALO - HALO;
	int ny = _ny_ - HALO - HALO;
	int nz = _nz_ - HALO - HALO;



	float mu = 0.0f;											
	float lambda = 0.0f;										
	float buoyancy = 0.0f;										
																
	float beta_z = 0.0f;										
	float d_z = 0.0f;											
	float alpha_d_z = 0.0f;										
																
	float zt_x = 0.0f; 	float zt_y = 0.0f; 	float zt_z = 0.0f;	
																
	float Txx_zt = 0.0f; float Tyy_zt = 0.0f; float Txy_zt = 0.0f; 
	float Txz_zt = 0.0f; float Tyz_zt = 0.0f; float Tzz_zt = 0.0f;
	float Vx_zt  = 0.0f; float Vy_zt  = 0.0f; float Vz_zt  = 0.0f; 

	float Vx3  = 0.0f;	
	float Vy3  = 0.0f;	
	float Vz3  = 0.0f;	
	float Txx3 = 0.0f;
	float Tyy3 = 0.0f;
	float Tzz3 = 0.0f;
	float Txy3 = 0.0f;
	float Txz3 = 0.0f;
	float Tyz3 = 0.0f;	


	float h_WVx ;		
	float h_WVy ;		
	float h_WVz ;		
	float h_WTxx;		
	float h_WTyy;		
	float h_WTzz;		
	float h_WTxy;		
	float h_WTxz;		
	float h_WTyz;

																
																
	int stride = FLAG * ( nz - nPML );				
	CALCULATE3D( i0, j0, k0, 0, nx, 0, ny, 0, nPML )
		i = i0 + HALO;													
		j = j0 + HALO;													
		k = k0 + HALO + stride;											
		index = INDEX(i, j, k);											
		pos	= Index3D( i0, j0, k0, nx, ny, nPML );//i0 + j0 * nx + k0 * nx * ny;
																
		mu = (float)medium.mu[index]; 											
		lambda = (float)medium.lambda[index];							
		buoyancy = (float)medium.buoyancy[index];
		buoyancy *= Crho;
		beta_z = pml_beta_z[k]; 								
		d_z = pml_d_z[k];										
		alpha_d_z = d_z + pml_alpha_z[k];						
		

		zt_x = (float)ZT_X[index]; zt_y = (float)ZT_Y[index]; zt_z = (float)ZT_Z[index];
																
		 Vx_zt = L( (float)W.Vx , FB3, zt ) * d_z;						
		 Vy_zt = L( (float)W.Vy , FB3, zt ) * d_z;						
		 Vz_zt = L( (float)W.Vz , FB3, zt ) * d_z;						
		Txx_zt = L( (float)W.Txx, FB3, zt ) * d_z;						
		Tyy_zt = L( (float)W.Tyy, FB3, zt ) * d_z;						
		Tzz_zt = L( (float)W.Tzz, FB3, zt ) * d_z;						
		Txy_zt = L( (float)W.Txy, FB3, zt ) * d_z;						
		Txz_zt = L( (float)W.Txz, FB3, zt ) * d_z;						
		Tyz_zt = L( (float)W.Tyz, FB3, zt ) * d_z;						
		

		Vx3  = DOT_PRODUCT3D( zt_x, zt_y, zt_z, Txx_zt, Txy_zt, Txz_zt ) * buoyancy;
		Vy3  = DOT_PRODUCT3D( zt_x, zt_y, zt_z, Txy_zt, Tyy_zt, Tyz_zt ) * buoyancy;
		Vz3  = DOT_PRODUCT3D( zt_x, zt_y, zt_z, Txz_zt, Tyz_zt, Tzz_zt ) * buoyancy;

		Txx3 = DOT_PRODUCT3D( zt_x, zt_y, zt_z, Vx_zt, Vy_zt, Vz_zt ) * lambda + 2.0f * mu * ( zt_x * Vx_zt );
		Tyy3 = DOT_PRODUCT3D( zt_x, zt_y, zt_z, Vx_zt, Vy_zt, Vz_zt ) * lambda + 2.0f * mu * ( zt_y * Vy_zt );
		Tzz3 = DOT_PRODUCT3D( zt_x, zt_y, zt_z, Vx_zt, Vy_zt, Vz_zt ) * lambda + 2.0f * mu * ( zt_z * Vz_zt );

		Txy3 = DOT_PRODUCT2D( zt_y, zt_x, Vx_zt, Vy_zt ) * mu;
		Txz3 = DOT_PRODUCT2D( zt_z, zt_x, Vx_zt, Vz_zt ) * mu;
		Tyz3 = DOT_PRODUCT2D( zt_z, zt_y, Vy_zt, Vz_zt ) * mu;


		h_Aux_z.Vx [pos] = ( Vx3  - alpha_d_z * Aux_z.Vx [pos] ) * DT;	
		h_Aux_z.Vy [pos] = ( Vy3  - alpha_d_z * Aux_z.Vy [pos] ) * DT;	
		h_Aux_z.Vz [pos] = ( Vz3  - alpha_d_z * Aux_z.Vz [pos] ) * DT;	
		h_Aux_z.Txx[pos] = ( Txx3 - alpha_d_z * Aux_z.Txx[pos] ) * DT;	
		h_Aux_z.Tyy[pos] = ( Tyy3 - alpha_d_z * Aux_z.Tyy[pos] ) * DT;	
		h_Aux_z.Tzz[pos] = ( Tzz3 - alpha_d_z * Aux_z.Tzz[pos] ) * DT;	
		h_Aux_z.Txy[pos] = ( Txy3 - alpha_d_z * Aux_z.Txy[pos] ) * DT;	
		h_Aux_z.Txz[pos] = ( Txz3 - alpha_d_z * Aux_z.Txz[pos] ) * DT;	
		h_Aux_z.Tyz[pos] = ( Tyz3 - alpha_d_z * Aux_z.Tyz[pos] ) * DT;	
															 	
		h_WVx  = h_W.Vx [index];		
		h_WVy  = h_W.Vy [index];		
		h_WVz  = h_W.Vz [index];		
		h_WTxx = h_W.Txx[index];		
		h_WTyy = h_W.Tyy[index];		
		h_WTzz = h_W.Tzz[index];		
		h_WTxy = h_W.Txy[index];		
		h_WTxz = h_W.Txz[index];		
		h_WTyz = h_W.Tyz[index];

		h_WVx  += - beta_z * Aux_z.Vx [pos] * DT;		
		h_WVy  += - beta_z * Aux_z.Vy [pos] * DT;		
		h_WVz  += - beta_z * Aux_z.Vz [pos] * DT;		
		h_WTxx += - beta_z * Aux_z.Txx[pos] * DT;		
		h_WTyy += - beta_z * Aux_z.Tyy[pos] * DT;		
		h_WTzz += - beta_z * Aux_z.Tzz[pos] * DT;		
		h_WTxy += - beta_z * Aux_z.Txy[pos] * DT;		
		h_WTxz += - beta_z * Aux_z.Txz[pos] * DT;		
		h_WTyz += - beta_z * Aux_z.Tyz[pos] * DT;

		h_W.Vx [index] = h_WVx ;		
		h_W.Vy [index] = h_WVy ;		
		h_W.Vz [index] = h_WVz ;		
		h_W.Txx[index] = h_WTxx;		
		h_W.Tyy[index] = h_WTyy;		
		h_W.Tzz[index] = h_WTzz;		
		h_W.Txy[index] = h_WTxy;		
		h_W.Txz[index] = h_WTxz;		
		h_W.Tyz[index] = h_WTyz;


	END_CALCULATE3D( )											
}


bool aux4_dev_init=false;
void pmlDeriv( GRID grid, WAVE h_W, WAVE W, CONTRAVARIANT_FLOAT con, MEDIUM_FLOAT medium, AUX4 Aux4_1, AUX4 Aux4_2, PML_ALPHA pml_alpha, PML_BETA pml_beta, PML_D pml_d, MPI_BORDER border, int FB1, int FB2, int FB3, float DT 
#ifdef SYCL
			   ,WAVE h_W_device, WAVE W_device, CONTRAVARIANT_FLOAT con_device, MEDIUM_FLOAT medium_device, AUX4 Aux4_1_device, AUX4 Aux4_2_device, 
			   PML_ALPHA pml_alpha_device, PML_BETA pml_beta_device, PML_D pml_d_device,
			   sycl::queue &Q
#endif
			)

{
	int _nx_ = grid._nx_;
	int _ny_ = grid._ny_;
	int _nz_ = grid._nz_;

	float rDH = grid.rDH;

	int nPML = grid.nPML;

	FLOAT * XI_X = con.xi_x; FLOAT * XI_Y = con.xi_y; FLOAT * XI_Z = con.xi_z;
	FLOAT * ET_X = con.et_x; FLOAT * ET_Y = con.et_y; FLOAT * ET_Z = con.et_z;
	FLOAT * ZT_X = con.zt_x; FLOAT * ZT_Y = con.zt_y; FLOAT * ZT_Z = con.zt_z;

	float * pml_alpha_x = pml_alpha.x; float * pml_beta_x = pml_beta.x; float * pml_d_x = pml_d.x;
	float * pml_alpha_y = pml_alpha.y; float * pml_beta_y = pml_beta.y; float * pml_d_y = pml_d.y;
	float * pml_alpha_z = pml_alpha.z; float * pml_beta_z = pml_beta.z; float * pml_d_z = pml_d.z;

#ifdef GPU_CUDA

	int nx = _nx_ - 2 * HALO;
	int ny = _ny_ - 2 * HALO;
	int nz = _nz_ - 2 * HALO;

	dim3 thread( 8, 8, 8);
	dim3 blockX;
	blockX.x = ( nPML + thread.x - 1 ) / thread.x;
	blockX.y = ( ny   + thread.y - 1 ) / thread.y;
	blockX.z = ( nz   + thread.z - 1 ) / thread.z;

	dim3 blockY;
	blockY.x = ( nx   + thread.x - 1 ) / thread.x;
	blockY.y = ( nPML + thread.y - 1 ) / thread.y;
	blockY.z = ( nz   + thread.z - 1 ) / thread.z;

	dim3 blockZ;
	blockZ.x = ( nx   + thread.x - 1 ) / thread.x;
	blockZ.y = ( ny   + thread.y - 1 ) / thread.y;
	blockZ.z = ( nPML + thread.z - 1 ) / thread.z;

	
	if ( border.isx1 )  pml_deriv_x<<< blockX, thread >>>( h_W, W, Aux4_1.h_Aux_x, Aux4_1.Aux_x, XI_X, XI_Y, XI_Z, medium, pml_alpha_x, pml_beta_x, pml_d_x, nPML, _nx_, _ny_, _nz_, 0, rDH, FB1, DT );
	if ( border.isy1 )  pml_deriv_y<<< blockY, thread >>>( h_W, W, Aux4_1.h_Aux_y, Aux4_1.Aux_y, ET_X, ET_Y, ET_Z, medium, pml_alpha_y, pml_beta_y, pml_d_y, nPML, _nx_, _ny_, _nz_, 0, rDH, FB2, DT );
	if ( border.isz1 )  pml_deriv_z<<< blockZ, thread >>>( h_W, W, Aux4_1.h_Aux_z, Aux4_1.Aux_z, ZT_X, ZT_Y, ZT_Z, medium, pml_alpha_z, pml_beta_z, pml_d_z, nPML, _nx_, _ny_, _nz_, 0, rDH, FB3, DT );

	if ( border.isx2 )  pml_deriv_x<<< blockX, thread >>>( h_W, W, Aux4_2.h_Aux_x, Aux4_2.Aux_x, XI_X, XI_Y, XI_Z, medium, pml_alpha_x, pml_beta_x, pml_d_x, nPML, _nx_, _ny_, _nz_, 1, rDH, FB1, DT );
	if ( border.isy2 )  pml_deriv_y<<< blockY, thread >>>( h_W, W, Aux4_2.h_Aux_y, Aux4_2.Aux_y, ET_X, ET_Y, ET_Z, medium, pml_alpha_y, pml_beta_y, pml_d_y, nPML, _nx_, _ny_, _nz_, 1, rDH, FB2, DT );
#ifndef FREE_SURFACE
	if ( border.isz2 )  pml_deriv_z<<< blockZ, thread >>>( h_W, W, Aux4_2.h_Aux_z, Aux4_2.Aux_z, ZT_X, ZT_Y, ZT_Z, medium, pml_alpha_z, pml_beta_z, pml_d_z, nPML, _nx_, _ny_, _nz_, 1, rDH, FB3, DT );
#endif

	CHECK( cudaDeviceSynchronize( ));
	

#elif defined SYCL
	XI_X = con_device.xi_x; XI_Y = con_device.xi_y; XI_Z = con_device.xi_z;
	ET_X = con_device.et_x; ET_Y = con_device.et_y; ET_Z = con_device.et_z;
	ZT_X = con_device.zt_x; ZT_Y = con_device.zt_y; ZT_Z = con_device.zt_z;

	pml_alpha_x = pml_alpha_device.x; pml_beta_x = pml_beta_device.x; pml_d_x = pml_d_device.x;
	pml_alpha_y = pml_alpha_device.y; pml_beta_y = pml_beta_device.y; pml_d_y = pml_d_device.y;
	pml_alpha_z = pml_alpha_device.z; pml_beta_z = pml_beta_device.z; pml_d_z = pml_d_device.z;

	if ( border.isx1 )  pml_deriv_x_sycl( h_W_device, W_device, Aux4_1_device.h_Aux_x, Aux4_1_device.Aux_x, XI_X, XI_Y, XI_Z, medium_device, pml_alpha_x, pml_beta_x, pml_d_x, nPML, _nx_, _ny_, _nz_, 0, rDH, FB1, DT, Q);
	if ( border.isy1 )  pml_deriv_y_sycl( h_W_device, W_device, Aux4_1_device.h_Aux_y, Aux4_1_device.Aux_y, ET_X, ET_Y, ET_Z, medium_device, pml_alpha_y, pml_beta_y, pml_d_y, nPML, _nx_, _ny_, _nz_, 0, rDH, FB2, DT, Q);
	if ( border.isz1 )  pml_deriv_z_sycl( h_W_device, W_device, Aux4_1_device.h_Aux_z, Aux4_1_device.Aux_z, ZT_X, ZT_Y, ZT_Z, medium_device, pml_alpha_z, pml_beta_z, pml_d_z, nPML, _nx_, _ny_, _nz_, 0, rDH, FB3, DT, Q);

	if ( border.isx2 )  pml_deriv_x_sycl( h_W_device, W_device, Aux4_2_device.h_Aux_x, Aux4_2_device.Aux_x, XI_X, XI_Y, XI_Z, medium_device, pml_alpha_x, pml_beta_x, pml_d_x, nPML, _nx_, _ny_, _nz_, 1, rDH, FB1, DT, Q);
	if ( border.isy2 )  pml_deriv_y_sycl( h_W_device, W_device, Aux4_2_device.h_Aux_y, Aux4_2_device.Aux_y, ET_X, ET_Y, ET_Z, medium_device, pml_alpha_y, pml_beta_y, pml_d_y, nPML, _nx_, _ny_, _nz_, 1, rDH, FB2, DT, Q);
#ifndef FREE_SURFACE
	if ( border.isz2 )  pml_deriv_z_sycl( h_W_device, W_device, Aux4_2_device.h_Aux_z, Aux4_2_device.Aux_z, ZT_X, ZT_Y, ZT_Z, medium_device, pml_alpha_z, pml_beta_z, pml_d_z, nPML, _nx_, _ny_, _nz_, 1, rDH, FB3, DT, Q);
#endif

#else //GPU_CUDA

	XI_X = con.xi_x; XI_Y = con.xi_y; XI_Z = con.xi_z;
	ET_X = con.et_x; ET_Y = con.et_y; ET_Z = con.et_z;
	ZT_X = con.zt_x; ZT_Y = con.zt_y; ZT_Z = con.zt_z;

	pml_alpha_x = pml_alpha.x; pml_beta_x = pml_beta.x; pml_d_x = pml_d.x;
	pml_alpha_y = pml_alpha.y; pml_beta_y = pml_beta.y; pml_d_y = pml_d.y;
	pml_alpha_z = pml_alpha.z; pml_beta_z = pml_beta.z; pml_d_z = pml_d.z;

	if ( border.isx1 )  pml_deriv_x( h_W, W, Aux4_1.h_Aux_x, Aux4_1.Aux_x, XI_X, XI_Y, XI_Z, medium, pml_alpha_x, pml_beta_x, pml_d_x, nPML, _nx_, _ny_, _nz_, 0, rDH, FB1, DT );
	if ( border.isy1 )  pml_deriv_y( h_W, W, Aux4_1.h_Aux_y, Aux4_1.Aux_y, ET_X, ET_Y, ET_Z, medium, pml_alpha_y, pml_beta_y, pml_d_y, nPML, _nx_, _ny_, _nz_, 0, rDH, FB2, DT );
	if ( border.isz1 )  pml_deriv_z( h_W, W, Aux4_1.h_Aux_z, Aux4_1.Aux_z, ZT_X, ZT_Y, ZT_Z, medium, pml_alpha_z, pml_beta_z, pml_d_z, nPML, _nx_, _ny_, _nz_, 0, rDH, FB3, DT );

	if ( border.isx2 )  pml_deriv_x( h_W, W, Aux4_2.h_Aux_x, Aux4_2.Aux_x, XI_X, XI_Y, XI_Z, medium, pml_alpha_x, pml_beta_x, pml_d_x, nPML, _nx_, _ny_, _nz_, 1, rDH, FB1, DT );
	if ( border.isy2 )  pml_deriv_y( h_W, W, Aux4_2.h_Aux_y, Aux4_2.Aux_y, ET_X, ET_Y, ET_Z, medium, pml_alpha_y, pml_beta_y, pml_d_y, nPML, _nx_, _ny_, _nz_, 1, rDH, FB2, DT );
#ifndef FREE_SURFACE
	if ( border.isz2 )  pml_deriv_z( h_W, W, Aux4_2.h_Aux_z, Aux4_2.Aux_z, ZT_X, ZT_Y, ZT_Z, medium, pml_alpha_z, pml_beta_z, pml_d_z, nPML, _nx_, _ny_, _nz_, 1, rDH, FB3, DT );
#endif

#endif //GPU_CUDA


}




