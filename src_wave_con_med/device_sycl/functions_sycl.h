#include "header_sycl.h"

// wave_deriv
void allocWave_sycl( GRID grid, WAVE * h_W, WAVE * W, WAVE * t_W, WAVE * m_W, sycl::queue &Q);
void freeWave_sycl( WAVE h_W, WAVE W, WAVE t_W, WAVE m_W, sycl::queue &Q);
void memcpyWave_sycl(GRID grid, WAVE &h_W, WAVE &h_W_device, int direction, sycl::queue &Q);
void wave_deriv_sycl( WAVE h_W, WAVE W, CONTRAVARIANT_FLOAT con, MEDIUM_FLOAT medium, 
#ifdef PML
	PML_BETA pml_beta,
#endif
	int _nx_, int _ny_, int _nz_, float rDH, int FB1, int FB2, int FB3, float DT, sycl::queue &Q);

// wave_rk
void wave_rk0_sycl(sycl::item<1> &it, FLOAT * h_W, FLOAT * W, FLOAT * t_W, FLOAT * m_W, long long WStride );
void wave_rk1_sycl(sycl::item<1> &it, FLOAT * h_W, FLOAT * W, FLOAT * t_W, FLOAT * m_W, long long WStride);
void wave_rk2_sycl(sycl::item<1> &it, FLOAT * h_W, FLOAT * W, FLOAT * t_W, FLOAT * m_W, long long WStride );
void wave_rk3_sycl(sycl::item<1> &it, FLOAT * h_W, FLOAT * W, FLOAT * t_W, FLOAT * m_W, long long WStride );
void waveRk_sycl( GRID grid, int irk, FLOAT * h_W, FLOAT * W, FLOAT * t_W, FLOAT * m_W, sycl::queue &Q );

// pml_deriv
void allocAuxPML_sycl( int nPML, int N1, int N2, AUX * h_Aux, AUX * Aux, AUX * t_Aux, AUX * m_Aux, sycl::queue &Q);
void allocPML_sycl( GRID grid, AUX4 *Aux4_1, AUX4 *Aux4_2, MPI_BORDER border, sycl::queue &Q);
void freePML_sycl( MPI_BORDER border,  AUX4 Aux4_1, AUX4 Aux4_2, sycl::queue &Q);
void memcpyPML_sycl(GRID grid, AUX4 &Aux4_1, AUX4 &Aux4_2, AUX4 &Aux4_1_device, AUX4 &Aux4_2_device, MPI_BORDER border, int direction, sycl::queue &Q);
void pml_deriv_x_sycl( 									
	WAVE h_W, WAVE W, AUXILIARY h_Aux_x, AUXILIARY Aux_x,		
	FLOAT * XI_X, FLOAT * XI_Y, FLOAT * XI_Z, MEDIUM_FLOAT medium, 	
	float * pml_alpha_x, float * pml_beta_x, float * pml_d_x, 	
	int nPML, int _nx_, int _ny_, int _nz_, int FLAG, float rDH, int FB1, float DT, sycl::queue &Q );
void pml_deriv_y_sycl( 
	WAVE h_W, WAVE W, AUXILIARY h_Aux_y, AUXILIARY Aux_y,
	FLOAT * ET_X, FLOAT * ET_Y, FLOAT * ET_Z, MEDIUM_FLOAT medium, 		
	float * pml_alpha_y, float * pml_beta_y, float *  pml_d_y, 		
	int nPML, int _nx_, int _ny_, int _nz_, int FLAG, float rDH, int FB2, float DT, sycl::queue &Q );
void pml_deriv_z_sycl( 
	WAVE h_W, WAVE W, AUXILIARY h_Aux_z, AUXILIARY Aux_z,	
	FLOAT * ZT_X, FLOAT * ZT_Y, FLOAT * ZT_Z, MEDIUM_FLOAT medium, 	
	float * pml_alpha_z, float * pml_beta_z, float *  pml_d_z,	
	int nPML, int _nx_, int _ny_, int _nz_, int FLAG, float rDH, int FB3, float DT, sycl::queue &Q );

// pml_rk
void wave_pml_rk0_sycl(sycl::item<1> &it, float * h_W, float * W, float * t_W, float * m_W, long long WStride );
void wave_pml_rk1_sycl(sycl::item<1> &it, float * h_W, float * W, float * t_W, float * m_W, long long WStride );
void wave_pml_rk2_sycl(sycl::item<1> &it, float * h_W, float * W, float * t_W, float * m_W, long long WStride );
void wave_pml_rk3_sycl(sycl::item<1> &it, float * h_W, float * W, float * t_W, float * m_W, long long WStride );
void pmlRk_sycl( GRID grid, MPI_BORDER border, int irk, AUX4 Aux4_1, AUX4 Aux4_2, sycl::queue &Q );


// pml param
void allocPMLParameter_sycl( GRID grid, PML_ALPHA * pml_alpha, PML_BETA *pml_beta, PML_D * pml_d, sycl::queue &Q );
void freePMLParamter_sycl( PML_ALPHA pml_alpha, PML_BETA pml_beta, PML_D pml_d, sycl::queue &Q );
void memcpyPMLParam_sycl(GRID grid, PML_ALPHA pml_alpha, PML_ALPHA pml_alpha_device, int direction, sycl::queue &Q);

//contravariant
void allocContravariant_sycl( GRID grid, CONTRAVARIANT * con, sycl::queue &Q );
void freeContravariant_sycl( CONTRAVARIANT con, sycl::queue &Q );
void memcpyCon_sycl( GRID grid, CONTRAVARIANT con, CONTRAVARIANT con_device, int direction, sycl::queue &Q);

void allocContravariant_FLOAT_sycl( GRID grid, CONTRAVARIANT_FLOAT * con, sycl::queue &Q );
void freeContravariant_FLOAT_sycl( CONTRAVARIANT_FLOAT con, sycl::queue &Q );
void memcpyCon_FLOAT_sycl( GRID grid, CONTRAVARIANT_FLOAT con, CONTRAVARIANT_FLOAT con_device, int direction, sycl::queue &Q);

void allocJac_sycl( GRID grid, float ** Jac, sycl::queue &Q );
void freeJac_sycl( float * Jac, sycl::queue &Q );
void memcpyJac_sycl(GRID grid, float *Jac, float *Jac_device, int direction, sycl::queue &Q);

void allocJac_FLOAT_sycl( GRID grid, FLOAT ** Jac, sycl::queue &Q );
void freeJac_FLOAT_sycl( FLOAT * Jac, sycl::queue &Q );
void memcpyJac_FLOAT_sycl(GRID grid, FLOAT *Jac, FLOAT *Jac_device, int direction, sycl::queue &Q);

void allocMat3x3_sycl( GRID grid, Mat3x3 * _rDZ_DX, Mat3x3 * _rDZ_DY, sycl::queue &Q );
void freeMat3x3_sycl( Mat3x3 _rDZ_DX, Mat3x3 _rDZ_DY, sycl::queue &Q );
void memcpyMat3x3_sycl(GRID grid, Mat3x3 _rDZ_DX, Mat3x3 _rDZ_DX_device, int direction, sycl::queue &Q);

//medium
void allocMedium_sycl( GRID grid, MEDIUM * medium, sycl::queue &Q );
void freeMedium_sycl( MEDIUM medium, sycl::queue &Q );
void allocMediumFLOAT_sycl( GRID grid, MEDIUM_FLOAT * medium, sycl::queue &Q );
void freeMediumFLOAT_sycl( MEDIUM_FLOAT medium, sycl::queue &Q );
void memcpyMed_FLOAT_sycl(GRID grid, MEDIUM_FLOAT medium, MEDIUM_FLOAT medium_device, int direction, sycl::queue &Q);


// freeSurface
void free_surface_deriv_sycl(WAVE h_W, WAVE W, CONTRAVARIANT_FLOAT con, MEDIUM_FLOAT medium, FLOAT * Jac, Mat3x3 _rDZ_DX, Mat3x3 _rDZ_DY, 
#ifdef PML
	PML_BETA pml_beta,
#endif
	int _nx_, int _ny_, int _nz_, float rDH, int FB1, int FB2, int FB3, float DT, sycl::queue &Q);


// pml_freeSurface
void pml_free_surface_x_sycl(		 									
	WAVE h_W, WAVE W, AUXILIARY h_Aux_x, AUXILIARY Aux_x,			
	FLOAT * ZT_X, FLOAT * ZT_Y, FLOAT * ZT_Z, MEDIUM_FLOAT medium, 		
	Mat3x3 _rDZ_DX, Mat3x3 _rDZ_DY,									
	float *  pml_d_x, int nPML, int _nx_, int _ny_, int _nz_, int FLAG, float rDH, int FB1, float DT, sycl::queue &Q);
void pml_free_surface_y_sycl( 											
	WAVE h_W, WAVE W, AUXILIARY h_Aux_y, AUXILIARY Aux_y,			
	FLOAT * ZT_X, FLOAT * ZT_Y, FLOAT * ZT_Z, MEDIUM_FLOAT medium, 		
	Mat3x3 _rDZ_DX, Mat3x3 _rDZ_DY,									
	float *  pml_d_y, int nPML, int _nx_, int _ny_, int _nz_, int FLAG, float rDH, int FB2, float DT, sycl::queue &Q);


// utils

bool checkWave(GRID grid, WAVE &h_W, WAVE &h_W_device, sycl::queue &Q);