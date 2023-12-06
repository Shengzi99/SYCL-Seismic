#include "header_sycl.h"
void wave_rk0_sycl(sycl::id<1> &idx, FLOAT * h_W, FLOAT * W, FLOAT * t_W, FLOAT * m_W, long long WStride )
{
    int i = idx[0];
	float h_w, w, t_w, m_w;
	CALCULATE1D_SYCL( i, 0, WStride ) 
		h_w = (float)h_W[i];
		m_w = (float)W[i];

		t_w = m_w + beta1  * h_w;
		w   = m_w + alpha2 * h_w;

		m_W[i] = m_w;
		t_W[i] = t_w;
		W[i] = w;
	END_CALCULATE1D_SYCL( )
}

void wave_rk1_sycl(sycl::id<1> &idx, FLOAT * h_W, FLOAT * W, FLOAT * t_W, FLOAT * m_W, long long WStride)
{
    int i = idx[0];
	float h_w, w, t_w, m_w;
	CALCULATE1D_SYCL( i, 0, WStride ) 
		h_w = (float)h_W[i];
		t_w = (float)t_W[i];
		m_w = (float)m_W[i];

		t_w += beta2 * h_w;
		w   = m_w + alpha3 * h_w;
		
		t_W[i] = t_w;
		W[i] = w;
	END_CALCULATE1D_SYCL( )
}

void wave_rk2_sycl(sycl::id<1> &idx, FLOAT * h_W, FLOAT * W, FLOAT * t_W, FLOAT * m_W, long long WStride )
{
    int i = idx[0];
	float h_w, w, t_w, m_w;
	CALCULATE1D_SYCL( i, 0, WStride ) 
		h_w = (float)h_W[i];
		t_w = (float)t_W[i];
		m_w = (float)m_W[i];

		t_w += beta3 * h_w;
		w    = m_w + h_w;

		t_W[i] = t_w;
		W[i] = w;
	END_CALCULATE1D_SYCL( )
}

void wave_rk3_sycl(sycl::id<1> &idx, FLOAT * h_W, FLOAT * W, FLOAT * t_W, FLOAT * m_W, long long WStride )
{
    int i = idx[0];
	float h_w, w, t_w, m_w;
	CALCULATE1D_SYCL( i, 0, WStride ) 
		h_w = (float)h_W[i];
		t_w = (float)t_W[i];

		w = t_w +  beta4 * h_w;

		W[i] = w;
	END_CALCULATE1D_SYCL( )
}

void waveRk_sycl( GRID grid, int irk, FLOAT * h_W, FLOAT * W, FLOAT * t_W, FLOAT * m_W, sycl::queue &Q )
{
	long long num = grid._nx_ * grid._ny_ * grid._nz_ * WAVESIZE;

    switch (irk){
    case 0:
        Q.parallel_for(num, [=](sycl::id<1> idx){
            wave_rk0_sycl(idx, h_W, W, t_W, m_W, num);
        });
        break;
    case 1:
        Q.parallel_for(num, [=](sycl::id<1> idx){
            wave_rk1_sycl(idx, h_W, W, t_W, m_W, num);
        });
        break;
    case 2:
        Q.parallel_for(num, [=](sycl::id<1> idx){
            wave_rk2_sycl(idx, h_W, W, t_W, m_W, num);
        });
        break;
    case 3:
        Q.parallel_for(num, [=](sycl::id<1> idx){
            wave_rk3_sycl(idx, h_W, W, t_W, m_W, num);
        });
        break;
    }
    Q.wait();

}