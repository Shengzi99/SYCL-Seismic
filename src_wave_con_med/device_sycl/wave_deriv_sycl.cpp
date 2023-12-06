#include "header_sycl.h"
#include<cmath>
using namespace std;

#ifdef PML
#define TIMES_PML_BETA_X * pml_beta_x
#define TIMES_PML_BETA_Y * pml_beta_y
#define TIMES_PML_BETA_Z * pml_beta_z
#else
#define TIMES_PML_BETA_X 
#define TIMES_PML_BETA_Y 
#define TIMES_PML_BETA_Z 
#endif


void allocWave_sycl( GRID grid, WAVE * h_W, WAVE * W, WAVE * t_W, WAVE * m_W, sycl::queue &Q)
{
	int _nx_ = grid._nx_;
	int _ny_ = grid._ny_;
	int _nz_ = grid._nz_;
	
	//printf( "_nx_ = %d, _ny_ = %d, _nz_ = %d\n", _nx_, _ny_, _nz_  );
	long long num = _nx_ * _ny_ * _nz_; 
		
	FLOAT * pWave = NULL;
	long long size = sizeof( FLOAT ) * num * WAVESIZE * 4;

    pWave = sycl::malloc_device<FLOAT>(size, Q);
    Q.memset(pWave, 0, size);

	h_W->Vx  = pWave + 0 * num;
	h_W->Vy  = pWave + 1 * num;
	h_W->Vz  = pWave + 2 * num;
	h_W->Txx = pWave + 3 * num;
	h_W->Tyy = pWave + 4 * num;
	h_W->Tzz = pWave + 5 * num;
	h_W->Txy = pWave + 6 * num;
	h_W->Txz = pWave + 7 * num;
	h_W->Tyz = pWave + 8 * num;

	pWave 	 = pWave + 9 * num;
	
	W->Vx  = pWave + 0 * num;
	W->Vy  = pWave + 1 * num;
	W->Vz  = pWave + 2 * num;
	W->Txx = pWave + 3 * num;
	W->Tyy = pWave + 4 * num;
	W->Tzz = pWave + 5 * num;
	W->Txy = pWave + 6 * num;
	W->Txz = pWave + 7 * num;
	W->Tyz = pWave + 8 * num;

	pWave    = pWave + 9 * num;

	t_W->Vx  = pWave + 0 * num;
	t_W->Vy  = pWave + 1 * num;
	t_W->Vz  = pWave + 2 * num;
	t_W->Txx = pWave + 3 * num;
	t_W->Tyy = pWave + 4 * num;
	t_W->Tzz = pWave + 5 * num;
	t_W->Txy = pWave + 6 * num;
	t_W->Txz = pWave + 7 * num;
	t_W->Tyz = pWave + 8 * num;

	pWave 	 = pWave + 9 * num;

	m_W->Vx  = pWave + 0 * num;
	m_W->Vy  = pWave + 1 * num;
	m_W->Vz  = pWave + 2 * num;
	m_W->Txx = pWave + 3 * num;
	m_W->Tyy = pWave + 4 * num;
	m_W->Tzz = pWave + 5 * num;
	m_W->Txy = pWave + 6 * num;
	m_W->Txz = pWave + 7 * num;
	m_W->Tyz = pWave + 8 * num;
    
    Q.wait();
}
void freeWave_sycl( WAVE h_W, WAVE W, WAVE t_W, WAVE m_W, sycl::queue &Q)
{
	sycl::free(h_W.Vx, Q);
}
void memcpyWave_sycl(GRID grid, WAVE &h_W, WAVE &h_W_device, int direction, sycl::queue &Q)
{
	long long num = grid._nx_ * grid._ny_ * grid._nz_; 
	long long size = sizeof( FLOAT ) * num * WAVESIZE * 4;

    if(direction == HOST_TO_DEVICE){
        Q.memcpy(h_W_device.Vx, h_W.Vx, size);
    }
    else{
        Q.memcpy(h_W.Vx, h_W_device.Vx, size);
    }
    Q.wait();
}
void checkWave(GRID grid, WAVE &h_W, WAVE &h_W_device, sycl::queue &Q){
    long long num = grid._nx_ * grid._ny_ * grid._nz_; 
	long long size = sizeof( FLOAT ) * num * WAVESIZE * 4;
    long long data_num = num * WAVESIZE * 4;

    FLOAT * pWave_copy = NULL;
    pWave_copy = (FLOAT*)malloc(size);
    Q.memcpy(pWave_copy, h_W_device.Vx, size).wait();

    FLOAT * pWave_host = h_W.Vx;
    
    double sum_err = 0.0; 
    float max_err = 0.0;
    long long inf_cnt=0, inf_cnt_d = 0, nz_cnt=0, nz_cnt_d=0, diff_cnt=0;
    for(size_t i=0; i<data_num; i++){
        if( (finite(pWave_copy[i])) && (finite(pWave_host[i])) )
        {
            float cur_err = fabs(pWave_copy[i]-pWave_host[i]);
            max_err = cur_err>max_err?cur_err:max_err;
            sum_err += cur_err;
            
            if(cur_err!=0.0){
                diff_cnt++;
                // printf("%.16f,  %.16f,  %.16f, index: %lu\n", pWave_copy[i], pWave_host[i], cur_err, i);
            }
            if(fabs(pWave_copy[i])!=0.0){
                nz_cnt_d++;
                // printf("%.16f,  %.16f,  %.16f, index: %lu, diff: %s\n", pWave_copy[i], pWave_host[i], cur_err, i, (cur_err!=0.0)?"true":"false");
            }
            if(fabs(pWave_host[i])!=0.0){
                nz_cnt++;
                // printf("%.16f,  %.16f,  %.16f, index: %lu, diff: %s\n", pWave_copy[i], pWave_host[i], cur_err, i, (cur_err!=0.0)?"true":"false");
            }
        }
        if(!finite(pWave_copy[i])) inf_cnt_d++;
        if(!finite(pWave_host[i])) inf_cnt++;
    }

    free(pWave_copy);

    printf("Sum Err: %.9lf, Max Err: %.9f, Avg Err: %.9lf.    inf cnt(h/d): %lld/%lld, nz cnt(h/d): %lld/%lld, diff cnt: %lld\n", sum_err, max_err, sum_err/diff_cnt, inf_cnt, inf_cnt_d, nz_cnt, nz_cnt_d, diff_cnt);
}

static sycl::range local{4, 4, 32};
void wave_deriv_sycl( WAVE h_W_device, WAVE W_device, CONTRAVARIANT_FLOAT con_device, MEDIUM_FLOAT medium_device, 
#ifdef PML
	PML_BETA pml_beta_device,
#endif
	int _nx_, int _ny_, int _nz_, float rDH, int FB1, int FB2, int FB3, float DT, sycl::queue &Q)
{
    int nx = _nx_ - 2 * HALO;
	int ny = _ny_ - 2 * HALO;
	int nz = _nz_ - 2 * HALO;
    // sycl::range local{4, 4, 32};

    sycl::range global{(nz+local[0]-1)/local[0]*local[0], (ny+local[1]-1)/local[1]*local[1], (nx+local[2]-1)/local[2]*local[2]};
    // sycl::range global{(nz+local[0]-1)/local[0]*local[0], (nx+local[2]-1)/local[2]*local[2], (ny+local[1]-1)/local[1]*local[1]};
    // sycl::range global{(nx+local[2]-1)/local[2]*local[2], (nz+local[0]-1)/local[0]*local[0], (ny+local[1]-1)/local[1]*local[1]};
    Q.parallel_for(sycl::nd_range{global, local}, [=](sycl::nd_item<3> nd_item){
        int _nx = _nx_ - HALO;
        int _ny = _ny_ - HALO;
        int _nz = _nz_ - HALO;
        int i = nd_item.get_global_id(2) + HALO;
        int j = nd_item.get_global_id(1) + HALO;
        int k = nd_item.get_global_id(0) + HALO;
        

        long long index;

#ifdef PML
        float pml_beta_x;
        float pml_beta_y;
        float pml_beta_z;
#endif
        float mu;												
        float lambda;											
        float buoyancy;											                                                 
        float xi_x;   float xi_y; 	float xi_z;		
        float et_x;   float et_y; 	float et_z;		
        float zt_x;   float zt_y; 	float zt_z;		                                              
        float Txx_xi; float Tyy_xi; float Txy_xi; 	
        float Txx_et; float Tyy_et; float Txy_et; 	
        float Txx_zt; float Tyy_zt; float Txy_zt; 	
        float Txz_xi; float Tyz_xi; float Tzz_xi;	
        float Txz_et; float Tyz_et; float Tzz_et;	
        float Txz_zt; float Tyz_zt; float Tzz_zt;	
        float Vx_xi ; float Vx_et ; float Vx_zt ;	
        float Vy_xi ; float Vy_et ; float Vy_zt ;	
        float Vz_xi ; float Vz_et ; float Vz_zt ;
        float Vx1;	  float Vx2;	float Vx3;
        float Vy1;	  float Vy2;	float Vy3;
        float Vz1;	  float Vz2;	float Vz3;
        float Txx1;	  float Txx2;   float Txx3;
        float Tyy1;	  float Tyy2;	float Tyy3;
        float Tzz1;   float Tzz2;	float Tzz3;
        float Txy1;   float Txy2;	float Txy3;
        float Txz1;   float Txz2;	float Txz3;
        float Tyz1;   float Tyz2;	float Tyz3;
        float h_WVx ;  
        float h_WVy ;  
        float h_WVz ;  
        float h_WTxx;
        float h_WTyy;
        float h_WTzz;
        float h_WTxy;
        float h_WTxz;
        float h_WTyz;

        CALCULATE3D_SYCL( i, j, k, HALO, _nx, HALO, _ny, HALO, _nz )
            index = INDEX( i, j, k );
            mu = (float)medium_device.mu[index]; 											
            lambda = (float)medium_device.lambda[index];										
            buoyancy = (float)medium_device.buoyancy[index];
            buoyancy *= Crho;

#ifdef PML
            pml_beta_x = pml_beta_device.x[i];
            pml_beta_y = pml_beta_device.y[j];
            pml_beta_z = pml_beta_device.z[k];
#endif

            xi_x = (float)con_device.xi_x[index]; xi_y = (float)con_device.xi_y[index]; xi_z = (float)con_device.xi_z[index];	
            et_x = (float)con_device.et_x[index]; et_y = (float)con_device.et_y[index]; et_z = (float)con_device.et_z[index];	
            zt_x = (float)con_device.zt_x[index]; zt_y = (float)con_device.zt_y[index]; zt_z = (float)con_device.zt_z[index];	

            Vx_xi = L( (float)W_device.Vx , FB1, xi ) TIMES_PML_BETA_X; 				
            Vy_xi = L( (float)W_device.Vy , FB1, xi ) TIMES_PML_BETA_X; 				
            Vz_xi = L( (float)W_device.Vz , FB1, xi ) TIMES_PML_BETA_X; 				
            Txx_xi = L( (float)W_device.Txx, FB1, xi ) TIMES_PML_BETA_X; 				
            Tyy_xi = L( (float)W_device.Tyy, FB1, xi ) TIMES_PML_BETA_X; 				
            Tzz_xi = L( (float)W_device.Tzz, FB1, xi ) TIMES_PML_BETA_X; 				
            Txy_xi = L( (float)W_device.Txy, FB1, xi ) TIMES_PML_BETA_X; 				
            Txz_xi = L( (float)W_device.Txz, FB1, xi ) TIMES_PML_BETA_X; 				
            Tyz_xi = L( (float)W_device.Tyz, FB1, xi ) TIMES_PML_BETA_X; 				

            Vx_et = L( (float)W_device.Vx , FB2, et ) TIMES_PML_BETA_Y;				
            Vy_et = L( (float)W_device.Vy , FB2, et ) TIMES_PML_BETA_Y;				
            Vz_et = L( (float)W_device.Vz , FB2, et ) TIMES_PML_BETA_Y;				
            Txx_et = L( (float)W_device.Txx, FB2, et ) TIMES_PML_BETA_Y;				
            Tyy_et = L( (float)W_device.Tyy, FB2, et ) TIMES_PML_BETA_Y;				
            Tzz_et = L( (float)W_device.Tzz, FB2, et ) TIMES_PML_BETA_Y;				
            Txy_et = L( (float)W_device.Txy, FB2, et ) TIMES_PML_BETA_Y;				
            Txz_et = L( (float)W_device.Txz, FB2, et ) TIMES_PML_BETA_Y;				
            Tyz_et = L( (float)W_device.Tyz, FB2, et ) TIMES_PML_BETA_Y;				

            Vx_zt = L( (float)W_device.Vx , FB3, zt ) TIMES_PML_BETA_Z;				
            Vy_zt = L( (float)W_device.Vy , FB3, zt ) TIMES_PML_BETA_Z;				
            Vz_zt = L( (float)W_device.Vz , FB3, zt ) TIMES_PML_BETA_Z;				
            Txx_zt = L( (float)W_device.Txx, FB3, zt ) TIMES_PML_BETA_Z;				
            Tyy_zt = L( (float)W_device.Tyy, FB3, zt ) TIMES_PML_BETA_Z;				
            Tzz_zt = L( (float)W_device.Tzz, FB3, zt ) TIMES_PML_BETA_Z;				
            Txy_zt = L( (float)W_device.Txy, FB3, zt ) TIMES_PML_BETA_Z;				
            Txz_zt = L( (float)W_device.Txz, FB3, zt ) TIMES_PML_BETA_Z;				
            Tyz_zt = L( (float)W_device.Tyz, FB3, zt ) TIMES_PML_BETA_Z;

            Vx1  = DOT_PRODUCT3D( xi_x, xi_y, xi_z, Txx_xi, Txy_xi, Txz_xi ) * buoyancy;
            Vx2  = DOT_PRODUCT3D( et_x, et_y, et_z, Txx_et, Txy_et, Txz_et ) * buoyancy;
            Vx3  = DOT_PRODUCT3D( zt_x, zt_y, zt_z, Txx_zt, Txy_zt, Txz_zt ) * buoyancy;
            Vy1  = DOT_PRODUCT3D( xi_x, xi_y, xi_z, Txy_xi, Tyy_xi, Tyz_xi ) * buoyancy;
            Vy2  = DOT_PRODUCT3D( et_x, et_y, et_z, Txy_et, Tyy_et, Tyz_et ) * buoyancy;
            Vy3  = DOT_PRODUCT3D( zt_x, zt_y, zt_z, Txy_zt, Tyy_zt, Tyz_zt ) * buoyancy;
            Vz1  = DOT_PRODUCT3D( xi_x, xi_y, xi_z, Txz_xi, Tyz_xi, Tzz_xi ) * buoyancy;
            Vz2  = DOT_PRODUCT3D( et_x, et_y, et_z, Txz_et, Tyz_et, Tzz_et ) * buoyancy;
            Vz3  = DOT_PRODUCT3D( zt_x, zt_y, zt_z, Txz_zt, Tyz_zt, Tzz_zt ) * buoyancy;

            Txx1 = DOT_PRODUCT3D( xi_x, xi_y, xi_z, Vx_xi, Vy_xi, Vz_xi ) * lambda + 2.0f * mu * ( xi_x * Vx_xi );
            Txx2 = DOT_PRODUCT3D( et_x, et_y, et_z, Vx_et, Vy_et, Vz_et ) * lambda + 2.0f * mu * ( et_x * Vx_et );
            Txx3 = DOT_PRODUCT3D( zt_x, zt_y, zt_z, Vx_zt, Vy_zt, Vz_zt ) * lambda + 2.0f * mu * ( zt_x * Vx_zt );
            Tyy1 = DOT_PRODUCT3D( xi_x, xi_y, xi_z, Vx_xi, Vy_xi, Vz_xi ) * lambda + 2.0f * mu * ( xi_y * Vy_xi );
            Tyy2 = DOT_PRODUCT3D( et_x, et_y, et_z, Vx_et, Vy_et, Vz_et ) * lambda + 2.0f * mu * ( et_y * Vy_et );
            Tyy3 = DOT_PRODUCT3D( zt_x, zt_y, zt_z, Vx_zt, Vy_zt, Vz_zt ) * lambda + 2.0f * mu * ( zt_y * Vy_zt );
            Tzz1 = DOT_PRODUCT3D( xi_x, xi_y, xi_z, Vx_xi, Vy_xi, Vz_xi ) * lambda + 2.0f * mu * ( xi_z * Vz_xi );
            Tzz2 = DOT_PRODUCT3D( et_x, et_y, et_z, Vx_et, Vy_et, Vz_et ) * lambda + 2.0f * mu * ( et_z * Vz_et );
            Tzz3 = DOT_PRODUCT3D( zt_x, zt_y, zt_z, Vx_zt, Vy_zt, Vz_zt ) * lambda + 2.0f * mu * ( zt_z * Vz_zt );

            Txy1 = DOT_PRODUCT2D( xi_y, xi_x, Vx_xi, Vy_xi ) * mu;
            Txy2 = DOT_PRODUCT2D( et_y, et_x, Vx_et, Vy_et ) * mu;
            Txy3 = DOT_PRODUCT2D( zt_y, zt_x, Vx_zt, Vy_zt ) * mu;
            Txz1 = DOT_PRODUCT2D( xi_z, xi_x, Vx_xi, Vz_xi ) * mu;
            Txz2 = DOT_PRODUCT2D( et_z, et_x, Vx_et, Vz_et ) * mu;
            Txz3 = DOT_PRODUCT2D( zt_z, zt_x, Vx_zt, Vz_zt ) * mu;
            Tyz1 = DOT_PRODUCT2D( xi_z, xi_y, Vy_xi, Vz_xi ) * mu;
            Tyz2 = DOT_PRODUCT2D( et_z, et_y, Vy_et, Vz_et ) * mu;
            Tyz3 = DOT_PRODUCT2D( zt_z, zt_y, Vy_zt, Vz_zt ) * mu;

            h_WVx  	= ( Vx1  + Vx2  + Vx3  ) * DT;
            h_WVy  	= ( Vy1  + Vy2  + Vy3  ) * DT;
            h_WVz  	= ( Vz1  + Vz2  + Vz3  ) * DT;
            h_WTxx 	= ( Txx1 + Txx2 + Txx3 ) * DT;
            h_WTyy 	= ( Tyy1 + Tyy2 + Tyy3 ) * DT;
            h_WTzz 	= ( Tzz1 + Tzz2 + Tzz3 ) * DT;
            h_WTxy 	= ( Txy1 + Txy2 + Txy3 ) * DT;
            h_WTxz 	= ( Txz1 + Txz2 + Txz3 ) * DT;
            h_WTyz 	= ( Tyz1 + Tyz2 + Tyz3 ) * DT;
            
            h_W_device.Vx [index] 	= h_WVx ;
            h_W_device.Vy [index] 	= h_WVy ;
            h_W_device.Vz [index] 	= h_WVz ;
            h_W_device.Txx[index] 	= h_WTxx;
            h_W_device.Tyy[index] 	= h_WTyy;
            h_W_device.Tzz[index] 	= h_WTzz;
            h_W_device.Txy[index] 	= h_WTxy;
            h_W_device.Txz[index] 	= h_WTxz;
            h_W_device.Tyz[index] 	= h_WTyz;
        END_CALCULATE3D_SYCL( )

    }).wait();
}