#include "header_sycl.h"

void wave_pml_rk0_sycl(sycl::id<1> &idx, float *h_W, float *W, float *t_W, float *m_W, long long WStride)
{
	int i = idx[0];
	CALCULATE1D_SYCL(i, 0, WStride)
	m_W[i] = W[i];
	t_W[i] = m_W[i] + beta1 * h_W[i];
	W[i] = m_W[i] + alpha2 * h_W[i];
	END_CALCULATE1D_SYCL()
}

void wave_pml_rk1_sycl(sycl::id<1> &idx, float *h_W, float *W, float *t_W, float *m_W, long long WStride)
{
	int i = idx[0];
	CALCULATE1D_SYCL(i, 0, WStride)
	t_W[i] += beta2 * h_W[i];
	W[i] = m_W[i] + alpha3 * h_W[i];
	END_CALCULATE1D_SYCL()
}

void wave_pml_rk2_sycl(sycl::id<1> &idx, float *h_W, float *W, float *t_W, float *m_W, long long WStride)
{
	int i = idx[0];
	CALCULATE1D_SYCL(i, 0, WStride)
	t_W[i] += beta3 * h_W[i];
	W[i] = m_W[i] + h_W[i];
	END_CALCULATE1D_SYCL()
}

void wave_pml_rk3_sycl(sycl::id<1> &idx, float *h_W, float *W, float *t_W, float *m_W, long long WStride)
{
	int i = idx[0];
	CALCULATE1D_SYCL(i, 0, WStride)
	W[i] = t_W[i] + beta4 * h_W[i];
	END_CALCULATE1D_SYCL()
}
void pmlRk_sycl(GRID grid, MPI_BORDER border, int irk, AUX4 Aux4_1, AUX4 Aux4_2, sycl::queue &Q)
{
	int nx = grid.nx;
	int ny = grid.ny;
	int nz = grid.nz;
	int nPML = grid.nPML;
	long long numx = nPML * ny * nz * WAVESIZE;
	long long numy = nPML * nx * nz * WAVESIZE;
	long long numz = nPML * nx * ny * WAVESIZE;

	if (border.isx1)
		switch (irk)
		{
		case 0:
			Q.parallel_for(numx, [=](sycl::id<1> idx)
						   { wave_pml_rk0_sycl(idx, Aux4_1.h_Aux_x.Vx, Aux4_1.Aux_x.Vx, Aux4_1.t_Aux_x.Vx, Aux4_1.m_Aux_x.Vx, numx); });
			break;
		case 1:
			Q.parallel_for(numx, [=](sycl::id<1> idx)
						   { wave_pml_rk1_sycl(idx, Aux4_1.h_Aux_x.Vx, Aux4_1.Aux_x.Vx, Aux4_1.t_Aux_x.Vx, Aux4_1.m_Aux_x.Vx, numx); });
			break;
		case 2:
			Q.parallel_for(numx, [=](sycl::id<1> idx)
						   { wave_pml_rk2_sycl(idx, Aux4_1.h_Aux_x.Vx, Aux4_1.Aux_x.Vx, Aux4_1.t_Aux_x.Vx, Aux4_1.m_Aux_x.Vx, numx); });
			break;
		case 3:
			Q.parallel_for(numx, [=](sycl::id<1> idx)
						   { wave_pml_rk3_sycl(idx, Aux4_1.h_Aux_x.Vx, Aux4_1.Aux_x.Vx, Aux4_1.t_Aux_x.Vx, Aux4_1.m_Aux_x.Vx, numx); });
			break;
		}

	if (border.isy1)
		switch (irk)
		{
		case 0:
			Q.parallel_for(numy, [=](sycl::id<1> idx)
						   { wave_pml_rk0_sycl(idx, Aux4_1.h_Aux_y.Vx, Aux4_1.Aux_y.Vx, Aux4_1.t_Aux_y.Vx, Aux4_1.m_Aux_y.Vx, numy); });
			break;
		case 1:
			Q.parallel_for(numy, [=](sycl::id<1> idx)
						   { wave_pml_rk1_sycl(idx, Aux4_1.h_Aux_y.Vx, Aux4_1.Aux_y.Vx, Aux4_1.t_Aux_y.Vx, Aux4_1.m_Aux_y.Vx, numy); });
			break;
		case 2:
			Q.parallel_for(numy, [=](sycl::id<1> idx)
						   { wave_pml_rk2_sycl(idx, Aux4_1.h_Aux_y.Vx, Aux4_1.Aux_y.Vx, Aux4_1.t_Aux_y.Vx, Aux4_1.m_Aux_y.Vx, numy); });
			break;
		case 3:
			Q.parallel_for(numy, [=](sycl::id<1> idx)
						   { wave_pml_rk3_sycl(idx, Aux4_1.h_Aux_y.Vx, Aux4_1.Aux_y.Vx, Aux4_1.t_Aux_y.Vx, Aux4_1.m_Aux_y.Vx, numy); });
			break;
		}

	if (border.isz1)
		switch (irk)
		{
		case 0:
			Q.parallel_for(numz, [=](sycl::id<1> idx)
						   { wave_pml_rk0_sycl(idx, Aux4_1.h_Aux_z.Vx, Aux4_1.Aux_z.Vx, Aux4_1.t_Aux_z.Vx, Aux4_1.m_Aux_z.Vx, numz); });
			break;
		case 1:
			Q.parallel_for(numz, [=](sycl::id<1> idx)
						   { wave_pml_rk1_sycl(idx, Aux4_1.h_Aux_z.Vx, Aux4_1.Aux_z.Vx, Aux4_1.t_Aux_z.Vx, Aux4_1.m_Aux_z.Vx, numz); });
			break;
		case 2:
			Q.parallel_for(numz, [=](sycl::id<1> idx)
						   { wave_pml_rk2_sycl(idx, Aux4_1.h_Aux_z.Vx, Aux4_1.Aux_z.Vx, Aux4_1.t_Aux_z.Vx, Aux4_1.m_Aux_z.Vx, numz); });
			break;
		case 3:
			Q.parallel_for(numz, [=](sycl::id<1> idx)
						   { wave_pml_rk3_sycl(idx, Aux4_1.h_Aux_z.Vx, Aux4_1.Aux_z.Vx, Aux4_1.t_Aux_z.Vx, Aux4_1.m_Aux_z.Vx, numz); });
			break;
		}

	//
	if (border.isx2)
		switch (irk)
		{
		case 0:
			Q.parallel_for(numx, [=](sycl::id<1> idx)
						   { wave_pml_rk0_sycl(idx, Aux4_2.h_Aux_x.Vx, Aux4_2.Aux_x.Vx, Aux4_2.t_Aux_x.Vx, Aux4_2.m_Aux_x.Vx, numx); });
			break;
		case 1:
			Q.parallel_for(numx, [=](sycl::id<1> idx)
						   { wave_pml_rk1_sycl(idx, Aux4_2.h_Aux_x.Vx, Aux4_2.Aux_x.Vx, Aux4_2.t_Aux_x.Vx, Aux4_2.m_Aux_x.Vx, numx); });
			break;
		case 2:
			Q.parallel_for(numx, [=](sycl::id<1> idx)
						   { wave_pml_rk2_sycl(idx, Aux4_2.h_Aux_x.Vx, Aux4_2.Aux_x.Vx, Aux4_2.t_Aux_x.Vx, Aux4_2.m_Aux_x.Vx, numx); });
			break;
		case 3:
			Q.parallel_for(numx, [=](sycl::id<1> idx)
						   { wave_pml_rk3_sycl(idx, Aux4_2.h_Aux_x.Vx, Aux4_2.Aux_x.Vx, Aux4_2.t_Aux_x.Vx, Aux4_2.m_Aux_x.Vx, numx); });
			break;
		}

	if (border.isy2)
		switch (irk)
		{
		case 0:
			Q.parallel_for(numy, [=](sycl::id<1> idx)
						   { wave_pml_rk0_sycl(idx, Aux4_2.h_Aux_y.Vx, Aux4_2.Aux_y.Vx, Aux4_2.t_Aux_y.Vx, Aux4_2.m_Aux_y.Vx, numy); });
			break;
		case 1:
			Q.parallel_for(numy, [=](sycl::id<1> idx)
						   { wave_pml_rk1_sycl(idx, Aux4_2.h_Aux_y.Vx, Aux4_2.Aux_y.Vx, Aux4_2.t_Aux_y.Vx, Aux4_2.m_Aux_y.Vx, numy); });
			break;
		case 2:
			Q.parallel_for(numy, [=](sycl::id<1> idx)
						   { wave_pml_rk2_sycl(idx, Aux4_2.h_Aux_y.Vx, Aux4_2.Aux_y.Vx, Aux4_2.t_Aux_y.Vx, Aux4_2.m_Aux_y.Vx, numy); });
			break;
		case 3:
			Q.parallel_for(numy, [=](sycl::id<1> idx)
						   { wave_pml_rk3_sycl(idx, Aux4_2.h_Aux_y.Vx, Aux4_2.Aux_y.Vx, Aux4_2.t_Aux_y.Vx, Aux4_2.m_Aux_y.Vx, numy); });
			break;
		}

#ifndef FREE_SURFACE
	if (border.isz2)
		switch (irk)
		{
		case 0:
			Q.parallel_for(numz, [=](sycl::id<1> idx)
						   { wave_pml_rk0_sycl(idx, Aux4_2.h_Aux_z.Vx, Aux4_2.Aux_z.Vx, Aux4_2.t_Aux_z.Vx, Aux4_2.m_Aux_z.Vx, numz); });
			break;
		case 1:
			Q.parallel_for(numz, [=](sycl::id<1> idx)
						   { wave_pml_rk1_sycl(idx, Aux4_2.h_Aux_z.Vx, Aux4_2.Aux_z.Vx, Aux4_2.t_Aux_z.Vx, Aux4_2.m_Aux_z.Vx, numz); });
			break;
		case 2:
			Q.parallel_for(numz, [=](sycl::id<1> idx)
						   { wave_pml_rk2_sycl(idx, Aux4_2.h_Aux_z.Vx, Aux4_2.Aux_z.Vx, Aux4_2.t_Aux_z.Vx, Aux4_2.m_Aux_z.Vx, numz); });
			break;
		case 3:
			Q.parallel_for(numz, [=](sycl::id<1> idx)
						   { wave_pml_rk3_sycl(idx, Aux4_2.h_Aux_z.Vx, Aux4_2.Aux_z.Vx, Aux4_2.t_Aux_z.Vx, Aux4_2.m_Aux_z.Vx, numz); });
			break;
		}

#endif
	Q.wait();
}
