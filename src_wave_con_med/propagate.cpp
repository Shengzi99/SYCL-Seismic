/*================================================================
*   ESS, Southern University of Science and Technology
*   
*   File Name:propagate.cpp
*   Author: Wenqiang Wang, 11849528@mail.sustech.edu.cn
*   Created Time:2021-11-03
*   Discription:
*
================================================================*/
#include "header.h"

void checkWave(GRID grid, WAVE &h_W){
    long long num = grid._nx_ * grid._ny_ * grid._nz_; 
	long long size = sizeof( FLOAT ) * num * WAVESIZE * 4;
    long long data_num = num * WAVESIZE * 4;

    FLOAT * pWave_host = h_W.Vx;

    long long inf_cnt=0, nz_cnt=0;
    for(size_t i=0; i<data_num; i++){
        if( (finite(pWave_host[i])) )
        {
            if(fabs(pWave_host[i])!=0.0){
                nz_cnt++;
                // printf("%.16f,  %.16f,  %.16f, index: %lu, diff: %s\n", pWave_copy[i], pWave_host[i], cur_err, i, (cur_err!=0.0)?"true":"false");
            }
        }
        if(!finite(pWave_host[i])) inf_cnt++;
    }

    printf("inf cnt-h: %lld, nz cnt-h: %lld\n", inf_cnt, nz_cnt);
}
void isMPIBorder( GRID grid, MPI_COORD thisMPICoord, MPI_BORDER * border )
{

	if ( 0 == thisMPICoord.X ) border->isx1 = 1;   if ( ( grid.PX - 1 ) == thisMPICoord.X ) border->isx2 = 1;
	if ( 0 == thisMPICoord.Y ) border->isy1 = 1;   if ( ( grid.PY - 1 ) == thisMPICoord.Y ) border->isy2 = 1;
	if ( 0 == thisMPICoord.Z ) border->isz1 = 1;   if ( ( grid.PZ - 1 ) == thisMPICoord.Z ) border->isz2 = 1;

}

const int TNUM = 32;
const char* TNAME[TNUM] = {
	"waveDeriv", 
	"freeSurfaceDeriv",
	"pmlDeriv",
	"pmlFreeSurfaceDeriv",
	"waveRk",
	"pmlRk"
};
double TIMER[TNUM];
size_t TCOUNT[TNUM];

void propagate( 
MPI_Comm comm_cart, MPI_COORD thisMPICoord, MPI_NEIGHBOR mpiNeighbor,
GRID grid, PARAMS params, SEND_RECV_DATA_FLOAT sr,
WAVE h_W, WAVE W, WAVE t_W, WAVE m_W,
Mat3x3 _rDZ_DX, Mat3x3 _rDZ_DY,
CONTRAVARIANT_FLOAT con, FLOAT * Jac, MEDIUM_FLOAT medium, 
SOURCE_FILE_INPUT src_in, long long * srcIndex, MOMENT_RATE momentRate, MOMENT_RATE momentRateSlice, 
SLICE slice, SLICE_DATA sliceData, SLICE_DATA sliceDataCpu )
{
	float DT = params.DT;
	int NT = params.TMAX / DT;
	float DH = grid.DH;


	int IT_SKIP = params.IT_SKIP;
	int sliceFreeSurf = params.sliceFreeSurf;
	SLICE freeSurfSlice;
	locateFreeSurfSlice( grid, &freeSurfSlice );
	
	
	
	SLICE_DATA freeSurfData, freeSurfDataCpu;
	PGV pgv, cpuPgv;



	int stationNum;
	STATION station, station_cpu;
	stationNum = readStationIndex( grid );

	if ( stationNum > 0 )
	{
		allocStation( &station, stationNum, NT );
		station_cpu = station;
#ifdef GPU_CUDA
		allocStation_cpu( &station_cpu, stationNum, NT );
#endif
		initStationIndex( grid, station_cpu ); 
#ifdef GPU_CUDA
		stationCPU2GPU( station, station_cpu, stationNum );
#endif 
	}



	int thisRank;
	MPI_Comm_rank( MPI_COMM_WORLD, &thisRank );


	int _nx_ = grid._nx_;
	int _ny_ = grid._ny_;
	int _nz_ = grid._nz_;
	

	
	int IsFreeSurface = 0;
#ifdef FREE_SURFACE
	if ( thisMPICoord.Z == grid.PZ - 1 )
		IsFreeSurface = 1;
#endif
	if ( IsFreeSurface )
	{
		allocatePGV( grid, &pgv );
		if ( sliceFreeSurf )
			allocSliceData( grid, freeSurfSlice, &freeSurfData );
#ifdef GPU_CUDA
		allocatePGV_cpu( grid, &cpuPgv );
		if ( sliceFreeSurf )
			allocSliceData_cpu( grid, freeSurfSlice, &freeSurfDataCpu );
#endif
	}
	

	MPI_BORDER border = { 0 };
	isMPIBorder( grid, thisMPICoord, &border );
	
	AUX4 Aux4_1, Aux4_2;


#ifdef PML
	allocPML( grid, &Aux4_1, &Aux4_2, border );

	PML_ALPHA pml_alpha;
	PML_BETA pml_beta;
	PML_D pml_d;

	allocPMLParameter( grid, &pml_alpha, &pml_beta, &pml_d );
	init_pml_parameter( params, grid, border, pml_alpha, pml_beta, pml_d );

#endif


	SOURCE S = { 0 } ;//{ _nx_ / 2, _ny_ / 2, _nz_ / 2 };
	locateSource( params, grid, &S );
	
	int useMultiSource = params.useMultiSource;
	int useSingleSource = params.useSingleSource;


	int it = 0, irk = 0;
	int FB1 = 0;	int FB2 = 0;	int FB3 = 0;

	int FB[8][3] =
	{
		{ -1, -1, -1 },
		{  1,  1, -1 },
		{  1,  1,  1 },
		{ -1, -1,  1 },
		{ -1,  1, -1 },
		{  1, -1, -1 },
		{  1, -1,  1 },
		{ -1,  1,  1 },
	};// F = 1, B = -1




	int nGauss = 3;
	int lenGauss = nGauss * 2 + 1;
	int gaussPoints =  lenGauss * lenGauss * lenGauss;
	float  * gaussFactor = ( float * ) malloc( gaussPoints * sizeof( float ) );

	int gPos = 0;
	float sumGauss = 0.0;
	float factorGauss = 0.0;
	int gaussI = 0, gaussJ = 0, gaussK = 0;
	float ra = 0.5 * nGauss;
	for( gaussK = - nGauss; gaussK < nGauss + 1; gaussK ++ )
	{
		for( gaussJ = - nGauss; gaussJ < nGauss + 1; gaussJ ++ )
		{
			for( gaussI = - nGauss; gaussI < nGauss + 1; gaussI ++ )
			{
				gPos = ( gaussI + nGauss ) + ( gaussJ + nGauss ) * lenGauss + ( gaussK + nGauss ) * lenGauss * lenGauss;
				float D1 = GAUSS_FUN( gaussI, ra, 0.0);
				float D2 = GAUSS_FUN( gaussJ, ra, 0.0);
				float D3 = GAUSS_FUN( gaussK, ra, 0.0);
				float amp = D1*D2*D3 / 0.998125703461425;				
				gaussFactor[gPos] = amp;
				sumGauss += amp;
			}
		}
	}



	/*
	int FB[8][3] =
	{
		{ -1, -1, -1 },
		{  1,  1,  1 },
		{ -1, -1, -1 },
		{  1,  1,  1 },
		{ -1, -1, -1 },
		{  1,  1,  1 },
		{ -1, -1, -1 },
		{  1,  1,  1 },
	};// F = 1, B = -1
	*/

	//GaussField( grid, W );
	//useSingleSource = 0;
	
	if ( useMultiSource )	
		calculateMomentRate( src_in, medium, Jac, momentRate, srcIndex, DH );

	MPI_Barrier( comm_cart );
	long long midClock = clock( ), stepClock = 0;


	for(int i=0;i<TNUM;i++) {TIMER[i]=0.0; TCOUNT[i]=0;}
#ifdef SYCL
	sycl::queue Q{sycl::cpu_selector_v};

	putchar('\n');
	std::cout<<"SYCL kernel device: "<<Q.get_device().get_info<sycl::info::device::name>()<<std::endl;
	std::cout<<"Max Compute Units: "<<Q.get_device().get_info<sycl::info::device::max_compute_units>()<<std::endl;
	putchar('\n');
	
	WAVE h_W_device, W_device, t_W_device, m_W_device;
	allocWave_sycl( grid, &h_W_device, &W_device, &t_W_device, &m_W_device, Q);

	PML_ALPHA pml_alpha_device;
	PML_BETA pml_beta_device;
	PML_D pml_d_device;
	allocPMLParameter_sycl(grid, &pml_alpha_device, &pml_beta_device, &pml_d_device, Q);

	AUX4 Aux4_1_device, Aux4_2_device;
	allocPML_sycl(grid, &Aux4_1_device, &Aux4_2_device, border, Q);

	CONTRAVARIANT_FLOAT con_device;
	FLOAT *Jac_device;
	MEDIUM_FLOAT medium_device;

	allocContravariant_FLOAT_sycl(grid, &con_device, Q);
	allocJac_FLOAT_sycl(grid, &Jac_device, Q);
	allocMediumFLOAT_sycl(grid, &medium_device, Q);

	Mat3x3 _rDZ_DX_device, _rDZ_DY_device;
	allocMat3x3_sycl(grid, &_rDZ_DX_device, &_rDZ_DY_device, Q);
#endif//SYCL


	for ( it = 0; it < NT; it ++ )
	{
		//loadPointSource( S, W, _nx_, _ny_, _nz_, Jac, it, 0, DT, DH );
		if( useSingleSource ) 
			loadPointSource( S, W, _nx_, _ny_, _nz_, Jac, it, 0, DT, DH, params.rickerfc );
		if ( useMultiSource )
			addMomenteRate( grid, src_in, W, Jac, srcIndex, momentRate, momentRateSlice, it, 0, DT, DH, gaussFactor, nGauss, IsFreeSurface );

#ifdef SYCL
		memcpyWave_sycl(grid, h_W, h_W_device, HOST_TO_DEVICE, Q);
		memcpyCon_FLOAT_sycl(grid, con, con_device, HOST_TO_DEVICE, Q);
		memcpyJac_FLOAT_sycl(grid, Jac, Jac_device, HOST_TO_DEVICE, Q);
		memcpyMed_FLOAT_sycl(grid, medium, medium_device, HOST_TO_DEVICE, Q);
		memcpyPMLParam_sycl(grid, pml_alpha, pml_alpha_device, HOST_TO_DEVICE, Q);

		memcpyPML_sycl(grid, Aux4_1, Aux4_2, Aux4_1_device, Aux4_2_device, border, HOST_TO_DEVICE, Q);
		memcpyMat3x3_sycl(grid, _rDZ_DX, _rDZ_DX_device, HOST_TO_DEVICE, Q);
#endif


		FB1 = FB[it % 8][0]; FB2 = FB[it % 8][1]; FB3 = FB[it % 8][2];
		for ( irk = 0; irk < 4; irk ++ )
		{
			MPI_Barrier( comm_cart );
			mpiSendRecv( grid, comm_cart, mpiNeighbor, W, sr );
			
#ifdef SYCL
		memcpyWave_sycl(grid, h_W, h_W_device, HOST_TO_DEVICE, Q);
#endif

#ifdef PML
			TIMER[0] -= MPI_Wtime(); TCOUNT[0]++;
			waveDeriv( grid, h_W, W, 
					   con, medium, 
					   pml_beta, 
					   FB1, FB2, FB3, DT
 #ifdef SYCL
					   ,h_W_device, W_device, 
					   con_device, medium_device,
					   pml_beta_device,
					   Q
 #endif
					   );
			TIMER[0] += MPI_Wtime();
			
			if ( IsFreeSurface ){ 
				TIMER[1] -= MPI_Wtime(); TCOUNT[1]++;
				freeSurfaceDeriv( grid, h_W, W, 
								con, medium, Jac,
								_rDZ_DX, _rDZ_DY, 
								pml_beta, 
								FB1, FB2, FB3, DT 
 #ifdef SYCL
								,h_W_device, W_device, 
								con_device, medium_device, Jac_device,
								_rDZ_DX_device, _rDZ_DY_device, 
								pml_beta_device,
								Q
 #endif
								);
				TIMER[1] += MPI_Wtime();
			}

			TIMER[2] -= MPI_Wtime(); TCOUNT[2]++;
			pmlDeriv( grid, h_W, W, 
					  con, medium, 
					  Aux4_1, Aux4_2, 
					  pml_alpha,  pml_beta, pml_d,	
					  border, 
					  FB1, FB2, FB3, DT
 #ifdef SYCL
					  ,h_W_device, W_device,
					  con_device, medium_device,
					  Aux4_1_device, Aux4_2_device,
					  pml_alpha_device, pml_beta_device, pml_d_device,
					  Q
 #endif
					);
			TIMER[2] += MPI_Wtime();


			if ( IsFreeSurface ){
				TIMER[3] -= MPI_Wtime(); TCOUNT[3]++;
				pmlFreeSurfaceDeriv( grid, h_W, W, 
									con, medium, 
									Aux4_1, Aux4_2, 
									_rDZ_DX, _rDZ_DY, 
									pml_d, 
									border, 
									FB1, FB2, DT
 #ifdef SYCL
									,h_W_device, W_device, 
									con_device, medium_device, 
									Aux4_1_device, Aux4_2_device, 
									_rDZ_DX_device, _rDZ_DY_device, 
									pml_d_device, 
									Q
 #endif
									);
				TIMER[3] += MPI_Wtime();
			}

			TIMER[4] -= MPI_Wtime(); TCOUNT[4]++;
 #ifdef SYCL
			waveRk_sycl( grid, irk, h_W_device.Vx, W_device.Vx, t_W_device.Vx, m_W_device.Vx, Q);
 #else
			waveRk( grid, irk, h_W.Vx, W.Vx, t_W.Vx, m_W.Vx );
 #endif
			TIMER[4] += MPI_Wtime();


			TIMER[5] -= MPI_Wtime(); TCOUNT[5]++;
 #ifdef SYCL
			pmlRk_sycl( grid, border, irk, Aux4_1_device, Aux4_2_device, Q);
 #else
			pmlRk( grid, border, irk, Aux4_1, Aux4_2 );
 #endif
			TIMER[5] += MPI_Wtime();
 

#else//PML

			waveDeriv( grid, h_W, W, 
					   con, medium,
					   FB1, FB2, FB3, DT
 #ifdef SYCL
					   ,h_W_device, W_device, 
					   con_device, medium_device
					   Q
 #endif
					   );	
			
			if ( IsFreeSurface ) freeSurfaceDeriv ( grid, h_W, W, 
													con, medium, Jac,
													_rDZ_DX, _rDZ_DY,
													FB1, FB2, FB3, DT 
 #ifdef SYCL
													,h_W_device, W_device, 
													con_device, medium_device, Jac_device,
													_rDZ_DX_device, _rDZ_DY_device,
													Q
 #endif
													);
 #ifdef SYCL
			waveRk_sycl( grid, irk, h_W_device.Vx, W_device.Vx, t_W_device.Vx, m_W_device.Vx, Q);
 #else
			waveRk( grid, irk, h_W.Vx, W.Vx, t_W.Vx, m_W.Vx );
 #endif

#endif//PML
			FB1 *= - 1; FB2 *= - 1; FB3 *= - 1; //reverse 

#ifdef SYCL
		memcpyWave_sycl(grid, h_W, h_W_device, DEVICE_TO_HOST, Q);
		memcpyCon_FLOAT_sycl(grid, con, con_device, DEVICE_TO_HOST, Q);
		memcpyJac_FLOAT_sycl(grid, Jac, Jac_device, DEVICE_TO_HOST, Q);
		memcpyMed_FLOAT_sycl(grid, medium, medium_device, DEVICE_TO_HOST, Q);
		memcpyPMLParam_sycl(grid, pml_alpha, pml_alpha_device, DEVICE_TO_HOST, Q);

		memcpyPML_sycl(grid, Aux4_1, Aux4_2, Aux4_1_device, Aux4_2_device, border, DEVICE_TO_HOST, Q);
		memcpyMat3x3_sycl(grid, _rDZ_DX, _rDZ_DX_device, DEVICE_TO_HOST, Q);
#endif
		} // for loop of irk: Range Kutta Four Step

		if ( stationNum >0 ) storageStation( grid, NT, stationNum, station, W, it );
		if ( IsFreeSurface ) 	comparePGV( grid, thisMPICoord, W, pgv );





		if ( it % IT_SKIP == 0  )
		{
			data2D_XYZ_out( thisMPICoord, params, grid, W, slice, sliceData, sliceDataCpu, 'T', it ); //V mean data dump Vx Vy Vz. T means data dump Txx Tyy Tzz Txy Txz Tzz
			data2D_XYZ_out( thisMPICoord, params, grid, W, slice, sliceData, sliceDataCpu, 'V', it ); //V mean data dump Vx Vy Vz. T means data dump Txx Tyy Tzz Txy Txz Tzz
			if ( sliceFreeSurf && IsFreeSurface )
				data2D_XYZ_out( thisMPICoord, params, grid, W, freeSurfSlice, freeSurfData, freeSurfDataCpu, 'F', it ); //F means data dump FreeSurfVx FreeSurfVy FreeSurfVz
		}
		MPI_Barrier( comm_cart );
		if ( ( 0 == thisRank ) && ( it % 10 == 0 ) )
		{
			printf( "it = %8d. ", it );
			//if ( 0 == it ) midClock = 0;
			stepClock = clock( ) - midClock;
			//midClock  = clock( ) - startClock;
			midClock  = stepClock + midClock;
			printf("Step time loss: %8.3lfs. Total time loss: %8.3lfs.\n", stepClock * 1.0 / ( CLOCKS_PER_SEC * 1.0 ), midClock * 1.0 / ( CLOCKS_PER_SEC * 1.0 ) );
#ifdef SYCL
			checkWave(grid, h_W, h_W_device, Q);
#else
			checkWave(grid, h_W);
#endif
			printf("\n Timer:\n");
			for(int i=0;i<6;i++){
				printf("\t [%-25s] Acc Time: %.8lf, Avg Time: %.8lf, Calls: %lu\n", TNAME[i], TIMER[i], TIMER[i]/TCOUNT[i], TCOUNT[i]);
			}
			putchar('\n');
		}
		
	
	}// for loop of it: The time iterator of NT steps

#ifdef SYCL
	freeWave_sycl( h_W_device, W_device, t_W_device, m_W_device, Q);
	freePMLParamter_sycl(pml_alpha_device, pml_beta_device, pml_d_device, Q);
	freePML_sycl(border, Aux4_1_device, Aux4_2_device, Q);
	freeContravariant_FLOAT_sycl(con_device, Q);
	freeJac_FLOAT_sycl(Jac_device, Q);
	freeMediumFLOAT_sycl(medium_device, Q);
	freeMat3x3_sycl(_rDZ_DX_device, _rDZ_DY_device, Q);
#endif//SYCL


	if ( stationNum > 0 )
	{
#ifdef GPU_CUDA
		stationGPU2CPU( station, station_cpu, stationNum, NT );
#endif
		write( params, grid, thisMPICoord, station_cpu, NT, stationNum );

	}
		



	
	free( gaussFactor );

#ifdef PML

	freePML( border, Aux4_1, Aux4_2 );

	freePMLParamter( pml_alpha, pml_beta, pml_d );

#endif






	if ( stationNum > 0 )
	{
		freeStation( station );
#ifdef GPU_CUDA
		freeStation_cpu( station_cpu );
#endif
	}




	if ( IsFreeSurface )
	{

		outputPGV( params, grid, thisMPICoord, pgv, cpuPgv );
		freePGV( pgv );
		if ( sliceFreeSurf ) 
			freeSliceData( grid, freeSurfSlice, freeSurfData );
#ifdef GPU_CUDA
		freePGV_cpu( cpuPgv );
		if ( sliceFreeSurf ) 
			freeSliceData_cpu( grid, freeSurfSlice, freeSurfDataCpu );
#endif
	}


}
