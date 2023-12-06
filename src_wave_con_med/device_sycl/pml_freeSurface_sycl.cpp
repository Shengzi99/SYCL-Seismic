#include "header_sycl.h"

static sycl::range local{1,8,8}; //original

void pml_free_surface_x_sycl(		 									
	WAVE h_W, WAVE W, AUXILIARY h_Aux_x, AUXILIARY Aux_x,			
	FLOAT * ZT_X, FLOAT * ZT_Y, FLOAT * ZT_Z, MEDIUM_FLOAT medium, 		
	Mat3x3 _rDZ_DX, Mat3x3 _rDZ_DY,									
	float *  pml_d_x, int nPML, int _nx_, int _ny_, int _nz_, int FLAG, float rDH, int FB1, float DT, sycl::queue &Q)
{
	int nx = _nx_ - 2 * HALO;
	int ny = _ny_ - 2 * HALO;
	int nz = _nz_ - 2 * HALO;

	// sycl::range local{1,8,8};
	sycl::range global_x{1,
						 (ny+local[1]-1)/local[1]*local[1],
						 (nPML+local[2]-1)/local[2]*local[2]};

	Q.parallel_for(sycl::nd_range{global_x, local}, [=](sycl::nd_item<3> nd_item){
		int i0 = nd_item.get_global_id(2);
		int j0 = nd_item.get_global_id(1);

		int k0;										
		int i, j, k;								
		long long index;
		long long pos;

		int nx = _nx_ - HALO - HALO;
		int ny = _ny_ - HALO - HALO;
		int nz = _nz_ - HALO - HALO;


		int indexOnSurf; 												
		float mu = 0.0f;												
		float lambda = 0.0f;											
																		
		float d_x = 0.0f;												
																		
		float Vx_xi  = 0.0f; float Vy_xi  = 0.0f; float Vz_xi  = 0.0f;	
		float Vx_zt  = 0.0f; float Vy_zt  = 0.0f; float Vz_zt  = 0.0f;	
		float zt_x  = 0.0f;  float zt_y  = 0.0f;  float zt_z  = 0.0f;	

		float Txx3 = 0.0;
		float Tyy3 = 0.0;
		float Tzz3 = 0.0;
		float Txy3 = 0.0;
		float Txz3 = 0.0;
		float Tyz3 = 0.0;


		float h_Aux_xTxx;																									
		float h_Aux_xTyy;																									
		float h_Aux_xTzz;																									
		float h_Aux_xTxy;																									
		float h_Aux_xTxz;																									
		float h_Aux_xTyz;																									
																		
		int stride = FLAG * ( nx - nPML );					
		k0 = nz - 1;										
		CALCULATE2D_SYCL( i0, j0, 0, nPML, 0, ny )		
			i = i0 + HALO + stride;								
			j = j0 + HALO;										
			k = k0 + HALO;										
			index = INDEX(i, j, k);								
			indexOnSurf = INDEX( i, j, 0 ); 					
			//pos	= i0 + j0 * nPML + k0 * nPML * ny;
			pos	= Index3D( i0, j0, k0, nPML, ny, nz );//i0 + j0 * nPML + k0 * nPML * ny;			
																		
			mu = (float)medium.mu[index]; 										
			lambda = (float)medium.lambda[index];								
																		
			d_x = pml_d_x[i];											
			

			zt_x = (float)ZT_X[index]; 	zt_y = (float)ZT_Y[index]; 	zt_z = (float)ZT_Z[index];															
			
			Vx_xi = L( ( float )W.Vx, FB1, xi ) * d_x;																							
			Vy_xi = L( ( float )W.Vy, FB1, xi ) * d_x;																							
			Vz_xi = L( ( float )W.Vz, FB1, xi ) * d_x;																							
																																		
			Vx_zt = DOT_PRODUCT3D( _rDZ_DX.M11[indexOnSurf], _rDZ_DX.M12[indexOnSurf], _rDZ_DX.M13[indexOnSurf], Vx_xi, Vy_xi, Vz_xi );	
			Vy_zt = DOT_PRODUCT3D( _rDZ_DX.M21[indexOnSurf], _rDZ_DX.M22[indexOnSurf], _rDZ_DX.M23[indexOnSurf], Vx_xi, Vy_xi, Vz_xi );	
			Vz_zt = DOT_PRODUCT3D( _rDZ_DX.M31[indexOnSurf], _rDZ_DX.M32[indexOnSurf], _rDZ_DX.M33[indexOnSurf], Vx_xi, Vy_xi, Vz_xi );	
			

			Txx3 = DOT_PRODUCT3D( zt_x, zt_y, zt_z, Vx_zt, Vy_zt, Vz_zt ) * lambda + 2.0f * mu * ( zt_x * Vx_zt );
			Tyy3 = DOT_PRODUCT3D( zt_x, zt_y, zt_z, Vx_zt, Vy_zt, Vz_zt ) * lambda + 2.0f * mu * ( zt_y * Vy_zt );
			Tzz3 = DOT_PRODUCT3D( zt_x, zt_y, zt_z, Vx_zt, Vy_zt, Vz_zt ) * lambda + 2.0f * mu * ( zt_z * Vz_zt );

			Txy3 = DOT_PRODUCT2D( zt_y, zt_x, Vx_zt, Vy_zt ) * mu;
			Txz3 = DOT_PRODUCT2D( zt_z, zt_x, Vx_zt, Vz_zt ) * mu;
			Tyz3 = DOT_PRODUCT2D( zt_z, zt_y, Vy_zt, Vz_zt ) * mu;
																																		
			h_Aux_xTxx = h_Aux_x.Txx[pos];																									
			h_Aux_xTyy = h_Aux_x.Tyy[pos];																									
			h_Aux_xTzz = h_Aux_x.Tzz[pos];																									
			h_Aux_xTxy = h_Aux_x.Txy[pos];																									
			h_Aux_xTxz = h_Aux_x.Txz[pos];																									
			h_Aux_xTyz = h_Aux_x.Tyz[pos];																									

			h_Aux_xTxx += Txx3 * DT;																									
			h_Aux_xTyy += Tyy3 * DT;																									
			h_Aux_xTzz += Tzz3 * DT;																									
			h_Aux_xTxy += Txy3 * DT;																									
			h_Aux_xTxz += Txz3 * DT;																									
			h_Aux_xTyz += Tyz3 * DT;																									

			h_Aux_x.Txx[pos] = h_Aux_xTxx;																									
			h_Aux_x.Tyy[pos] = h_Aux_xTyy;																									
			h_Aux_x.Tzz[pos] = h_Aux_xTzz;																									
			h_Aux_x.Txy[pos] = h_Aux_xTxy;																									
			h_Aux_x.Txz[pos] = h_Aux_xTxz;																									
			h_Aux_x.Tyz[pos] = h_Aux_xTyz;																									
										
		END_CALCULATE2D_SYCL( )
	}).wait();
}


void pml_free_surface_y_sycl( 											
	WAVE h_W, WAVE W, AUXILIARY h_Aux_y, AUXILIARY Aux_y,			
	FLOAT * ZT_X, FLOAT * ZT_Y, FLOAT * ZT_Z, MEDIUM_FLOAT medium, 		
	Mat3x3 _rDZ_DX, Mat3x3 _rDZ_DY,									
	float *  pml_d_y, int nPML, int _nx_, int _ny_, int _nz_, int FLAG, float rDH, int FB2, float DT, sycl::queue &Q)
{																	
	int nx = _nx_ - 2 * HALO;
	int ny = _ny_ - 2 * HALO;
	int nz = _nz_ - 2 * HALO;

	// sycl::range local{1,8,8};
	sycl::range global_y{1,
						 (nPML+local[1]-1)/local[1]*local[1],
						 (nx+local[2]-1)/local[2]*local[2],};

	Q.parallel_for(sycl::nd_range{global_y, local}, [=](sycl::nd_item<3> nd_item){
		int i0 = nd_item.get_global_id(2);
		int j0 = nd_item.get_global_id(1);

		int k0;											
		int i, j, k;									
		long long index;
		long long pos;

		int nx = _nx_ - HALO - HALO;
		int ny = _ny_ - HALO - HALO;
		int nz = _nz_ - HALO - HALO;

		int indexOnSurf; 								
		float mu = 0.0f;												
		float lambda = 0.0f;											
																		
		float d_y = 0.0f;												
																		
		float Vx_et  = 0.0f; float Vy_et  = 0.0f; float Vz_et  = 0.0f; 	
		float Vx_zt  = 0.0f; float Vy_zt  = 0.0f; float Vz_zt  = 0.0f;	
		float zt_x  = 0.0f;  float zt_y  = 0.0f;  float zt_z  = 0.0f;	
			
		float Txx3 = 0.0f;
		float Tyy3 = 0.0f;
		float Tzz3 = 0.0f;
		float Txy3 = 0.0f;
		float Txz3 = 0.0f;
		float Tyz3 = 0.0f; 

		float h_Aux_yTxx;																									
		float h_Aux_yTyy;																									
		float h_Aux_yTzz;																									
		float h_Aux_yTxy;																									
		float h_Aux_yTxz;																									
		float h_Aux_yTyz;																									

		int stride = FLAG * ( ny - nPML );					
		k0 = nz - 1;											
		CALCULATE2D_SYCL( i0, j0, 0, nx, 0, nPML )	
			i = i0 + HALO;									
			j = j0 + HALO + stride;							
			k = k0 + HALO;									
			index = INDEX(i, j, k);							
			indexOnSurf = INDEX( i, j, 0 ); 				
			//pos	= i0 + j0 * nx + k0 * nx * nPML;
			pos	= Index3D( i0, j0, k0, nx, nPML, nz );//i0 + j0 * nx + k0 * nx * nPML;
																		
			mu = (float)medium.mu[index]; 										
			lambda = (float)medium.lambda[index];								
																		
			d_y = pml_d_y[j];											
			
			zt_x = (float)ZT_X[index]; 	zt_y = (float)ZT_Y[index]; 	zt_z = (float)ZT_Z[index];															
																																		
			Vx_et = L( ( float )W.Vx, FB2, et ) * d_y;																							
			Vy_et = L( ( float )W.Vy, FB2, et ) * d_y;																							
			Vz_et = L( ( float )W.Vz, FB2, et ) * d_y;																							
																																		
			Vx_zt = DOT_PRODUCT3D( _rDZ_DY.M11[indexOnSurf], _rDZ_DY.M12[indexOnSurf], _rDZ_DY.M13[indexOnSurf], Vx_et, Vy_et, Vz_et );	
			Vy_zt = DOT_PRODUCT3D( _rDZ_DY.M21[indexOnSurf], _rDZ_DY.M22[indexOnSurf], _rDZ_DY.M23[indexOnSurf], Vx_et, Vy_et, Vz_et );	
			Vz_zt = DOT_PRODUCT3D( _rDZ_DY.M31[indexOnSurf], _rDZ_DY.M32[indexOnSurf], _rDZ_DY.M33[indexOnSurf], Vx_et, Vy_et, Vz_et );	
			

			Txx3 = DOT_PRODUCT3D( zt_x, zt_y, zt_z, Vx_zt, Vy_zt, Vz_zt ) * lambda + 2.0f * mu * ( zt_x * Vx_zt );
			Tyy3 = DOT_PRODUCT3D( zt_x, zt_y, zt_z, Vx_zt, Vy_zt, Vz_zt ) * lambda + 2.0f * mu * ( zt_y * Vy_zt );
			Tzz3 = DOT_PRODUCT3D( zt_x, zt_y, zt_z, Vx_zt, Vy_zt, Vz_zt ) * lambda + 2.0f * mu * ( zt_z * Vz_zt );

			Txy3 = DOT_PRODUCT2D( zt_y, zt_x, Vx_zt, Vy_zt ) * mu;
			Txz3 = DOT_PRODUCT2D( zt_z, zt_x, Vx_zt, Vz_zt ) * mu;
			Tyz3 = DOT_PRODUCT2D( zt_z, zt_y, Vy_zt, Vz_zt ) * mu;


			h_Aux_yTxx = h_Aux_y.Txx[pos];																									
			h_Aux_yTyy = h_Aux_y.Tyy[pos];																									
			h_Aux_yTzz = h_Aux_y.Tzz[pos];																									
			h_Aux_yTxy = h_Aux_y.Txy[pos];																									
			h_Aux_yTxz = h_Aux_y.Txz[pos];																									
			h_Aux_yTyz = h_Aux_y.Tyz[pos];																									

			h_Aux_yTxx += Txx3 * DT;																									
			h_Aux_yTyy += Tyy3 * DT;																									
			h_Aux_yTzz += Tzz3 * DT;																									
			h_Aux_yTxy += Txy3 * DT;																									
			h_Aux_yTxz += Txz3 * DT;																									
			h_Aux_yTyz += Tyz3 * DT;																									

			h_Aux_y.Txx[pos] = h_Aux_yTxx;																									
			h_Aux_y.Tyy[pos] = h_Aux_yTyy;																									
			h_Aux_y.Tzz[pos] = h_Aux_yTzz;																									
			h_Aux_y.Txy[pos] = h_Aux_yTxy;																									
			h_Aux_y.Txz[pos] = h_Aux_yTxz;																									
			h_Aux_y.Tyz[pos] = h_Aux_yTyz;																									


		END_CALCULATE2D_SYCL( )			
	}).wait();									
}