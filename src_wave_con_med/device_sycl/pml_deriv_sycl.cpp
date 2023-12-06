#include "header_sycl.h"

void allocAuxPML_sycl( int nPML, int N1, int N2, AUX * h_Aux, AUX * Aux, AUX * t_Aux, AUX * m_Aux, sycl::queue &Q)
{
	long long num = nPML * N1 * N2; 
		
	float * pAux = NULL;
	long long size = sizeof( float ) * num * 9 * 4;

    pAux = sycl::malloc_device<float>(size, Q);
    Q.memset(pAux, 0, size);

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

    Q.wait();
}
void allocPML_sycl( GRID grid, AUX4 *Aux4_1, AUX4 *Aux4_2, MPI_BORDER border, sycl::queue &Q)
{
	int nx = grid.nx;
	int ny = grid.ny;
	int nz = grid.nz;
	int nPML = grid.nPML;
    memset(( void * )Aux4_1, 0, sizeof(AUX4));
    memset(( void * )Aux4_2, 0, sizeof(AUX4));
	if ( border.isx1 ) allocAuxPML_sycl( nPML, ny, nz, &( Aux4_1->h_Aux_x ), &( Aux4_1->Aux_x ), &( Aux4_1->t_Aux_x ), &( Aux4_1->m_Aux_x ), Q);
	if ( border.isy1 ) allocAuxPML_sycl( nPML, nx, nz, &( Aux4_1->h_Aux_y ), &( Aux4_1->Aux_y ), &( Aux4_1->t_Aux_y ), &( Aux4_1->m_Aux_y ), Q);
	if ( border.isz1 ) allocAuxPML_sycl( nPML, nx, ny, &( Aux4_1->h_Aux_z ), &( Aux4_1->Aux_z ), &( Aux4_1->t_Aux_z ), &( Aux4_1->m_Aux_z ), Q);
                                                                              
	if ( border.isx2 ) allocAuxPML_sycl( nPML, ny, nz, &( Aux4_2->h_Aux_x ), &( Aux4_2->Aux_x ), &( Aux4_2->t_Aux_x ), &( Aux4_2->m_Aux_x ), Q);
	if ( border.isy2 ) allocAuxPML_sycl( nPML, nx, nz, &( Aux4_2->h_Aux_y ), &( Aux4_2->Aux_y ), &( Aux4_2->t_Aux_y ), &( Aux4_2->m_Aux_y ), Q);
#ifndef FREE_SURFACE
	if ( border.isz2 ) allocAuxPML_sycl( nPML, nx, ny, &( Aux4_2->h_Aux_z ), &( Aux4_2->Aux_z ), &( Aux4_2->t_Aux_z ), &( Aux4_2->m_Aux_z ), Q);
#endif
}
void freePML_sycl( MPI_BORDER border,  AUX4 Aux4_1, AUX4 Aux4_2, sycl::queue &Q)
{
	if ( border.isx1 )  sycl::free( Aux4_1.h_Aux_x.Vx, Q);
	if ( border.isy1 )  sycl::free( Aux4_1.h_Aux_y.Vx, Q);
	if ( border.isz1 )  sycl::free( Aux4_1.h_Aux_z.Vx, Q);
	                  
	if ( border.isx2 )  sycl::free( Aux4_2.h_Aux_x.Vx, Q);
	if ( border.isy2 )  sycl::free( Aux4_2.h_Aux_y.Vx, Q);
#ifndef FREE_SURFACE
	if ( border.isz2 )  sycl::free( Aux4_2.h_Aux_z.Vx, Q);
#endif
}
void memcpyPML_sycl(GRID grid, AUX4 &Aux4_1, AUX4 &Aux4_2, AUX4 &Aux4_1_device, AUX4 &Aux4_2_device, MPI_BORDER border, int direction, sycl::queue &Q){
    int nx = grid.nx;
	int ny = grid.ny;
	int nz = grid.nz;
	int nPML = grid.nPML;
    if(direction == HOST_TO_DEVICE){
        if ( border.isx1 )  Q.memcpy(Aux4_1_device.h_Aux_x.Vx, Aux4_1.h_Aux_x.Vx, sizeof( float ) * nPML * ny * nz * 9 * 4);
        if ( border.isy1 )  Q.memcpy(Aux4_1_device.h_Aux_y.Vx, Aux4_1.h_Aux_y.Vx, sizeof( float ) * nPML * nx * nz * 9 * 4);
        if ( border.isz1 )  Q.memcpy(Aux4_1_device.h_Aux_z.Vx, Aux4_1.h_Aux_z.Vx, sizeof( float ) * nPML * nx * ny * 9 * 4);
                        
        if ( border.isx2 )  Q.memcpy(Aux4_2_device.h_Aux_x.Vx, Aux4_2.h_Aux_x.Vx, sizeof( float ) * nPML * ny * nz * 9 * 4);
        if ( border.isy2 )  Q.memcpy(Aux4_2_device.h_Aux_y.Vx, Aux4_2.h_Aux_y.Vx, sizeof( float ) * nPML * nx * nz * 9 * 4);
#ifndef FREE_SURFACE
        if ( border.isz2 )  Q.memcpy(Aux4_2_device.h_Aux_z.Vx, Aux4_2.h_Aux_z.Vx, sizeof( float ) * nPML * nx * ny * 9 * 4);
#endif
    }
    else{
        if ( border.isx1 )  Q.memcpy(Aux4_1.h_Aux_x.Vx, Aux4_1_device.h_Aux_x.Vx, sizeof( float ) * nPML * ny * nz * 9 * 4);
        if ( border.isy1 )  Q.memcpy(Aux4_1.h_Aux_y.Vx, Aux4_1_device.h_Aux_y.Vx, sizeof( float ) * nPML * nx * nz * 9 * 4);
        if ( border.isz1 )  Q.memcpy(Aux4_1.h_Aux_z.Vx, Aux4_1_device.h_Aux_z.Vx, sizeof( float ) * nPML * nx * ny * 9 * 4);
                        
        if ( border.isx2 )  Q.memcpy(Aux4_2.h_Aux_x.Vx, Aux4_2_device.h_Aux_x.Vx, sizeof( float ) * nPML * ny * nz * 9 * 4);
        if ( border.isy2 )  Q.memcpy(Aux4_2.h_Aux_y.Vx, Aux4_2_device.h_Aux_y.Vx, sizeof( float ) * nPML * nx * nz * 9 * 4);
#ifndef FREE_SURFACE
        if ( border.isz2 )  Q.memcpy(Aux4_2.h_Aux_z.Vx, Aux4_2_device.h_Aux_z.Vx, sizeof( float ) * nPML * nx * ny * 9 * 4);
#endif
    }
    Q.wait();
}

static sycl::range local{8, 8, 8};
void pml_deriv_x_sycl( 									
	WAVE h_W, WAVE W, AUXILIARY h_Aux_x, AUXILIARY Aux_x,		
	FLOAT * XI_X, FLOAT * XI_Y, FLOAT * XI_Z, MEDIUM_FLOAT medium, 	
	float * pml_alpha_x, float * pml_beta_x, float * pml_d_x, 	
	int nPML, int _nx_, int _ny_, int _nz_, int FLAG, float rDH, int FB1, float DT, sycl::queue &Q)
{
    int nx = _nx_ - 2 * HALO;
	int ny = _ny_ - 2 * HALO;
	int nz = _nz_ - 2 * HALO;
    // sycl::range local{8, 8, 8};
    sycl::range globalX{(nz+local[0]-1)/local[0]*local[0], (ny+local[1]-1)/local[1]*local[1], (nPML+local[2]-1)/local[2]*local[2]};
    Q.parallel_for(sycl::nd_range{globalX, local}, [=](sycl::nd_item<3> nd_item){
        int i0 = nd_item.get_global_id(2);
        int j0 = nd_item.get_global_id(1);
        int k0 = nd_item.get_global_id(0);

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
        CALCULATE3D_SYCL( i0, j0, k0, 0, nPML, 0, ny, 0, nz )
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
                                            
        END_CALCULATE3D_SYCL( )
    }).wait();
}

void pml_deriv_y_sycl( 
	WAVE h_W, WAVE W, AUXILIARY h_Aux_y, AUXILIARY Aux_y,
	FLOAT * ET_X, FLOAT * ET_Y, FLOAT * ET_Z, MEDIUM_FLOAT medium, 		
	float * pml_alpha_y, float * pml_beta_y, float *  pml_d_y, 		
	int nPML, int _nx_, int _ny_, int _nz_, int FLAG, float rDH, int FB2, float DT, sycl::queue &Q )
{
    int nx = _nx_ - 2 * HALO;
	int ny = _ny_ - 2 * HALO;
	int nz = _nz_ - 2 * HALO;
    // sycl::range local{8, 8, 8};
    sycl::range globalX{(nz+local[0]-1)/local[0]*local[0], (nPML+local[1]-1)/local[1]*local[1], (nx+local[2]-1)/local[2]*local[2]};
    Q.parallel_for(sycl::nd_range{globalX, local}, [=](sycl::nd_item<3> nd_item){
        int i0 = nd_item.get_global_id(2);
        int j0 = nd_item.get_global_id(1);
        int k0 = nd_item.get_global_id(0);

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
        CALCULATE3D_SYCL( i0, j0, k0, 0, nx, 0, nPML, 0, nz )
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

        END_CALCULATE3D_SYCL( )
    }).wait();
}

void pml_deriv_z_sycl( 
	WAVE h_W, WAVE W, AUXILIARY h_Aux_z, AUXILIARY Aux_z,	
	FLOAT * ZT_X, FLOAT * ZT_Y, FLOAT * ZT_Z, MEDIUM_FLOAT medium, 	
	float * pml_alpha_z, float * pml_beta_z, float *  pml_d_z,	
	int nPML, int _nx_, int _ny_, int _nz_, int FLAG, float rDH, int FB3, float DT, sycl::queue &Q )
{
    int nx = _nx_ - 2 * HALO;
	int ny = _ny_ - 2 * HALO;
	int nz = _nz_ - 2 * HALO;
    // sycl::range local{8, 8, 8};
    sycl::range globalX{(nPML+local[0]-1)/local[0]*local[0], (ny+local[1]-1)/local[1]*local[1], (nx+local[2]-1)/local[2]*local[2]};
    Q.parallel_for(sycl::nd_range{globalX, local}, [=](sycl::nd_item<3> nd_item){
        int i0 = nd_item.get_global_id(2);
        int j0 = nd_item.get_global_id(1);
        int k0 = nd_item.get_global_id(0);

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
        CALCULATE3D_SYCL( i0, j0, k0, 0, nx, 0, ny, 0, nPML )
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


        END_CALCULATE3D_SYCL( )
    }).wait();
}