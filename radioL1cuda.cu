/*  
    Copyright Stephen Hardy 2013 
    Released under BSD license: http://opensource.org/licenses/BSD-2-Clause 
*/

#include "radioL1cuda.h"


#include <thrust/functional.h>
#include <thrust/transform.h>
#include <thrust/iterator/zip_iterator.h>
#include <thrust/iterator/counting_iterator.h>
#include <thrust/iterator/constant_iterator.h>
#include <thrust/random.h>
#include <thrust/inner_product.h>
#include <thrust/sequence.h>
#include <thrust/copy.h>
#include <thrust/pair.h>
#include <thrust/count.h>
#include <thrust/tuple.h>
#include <thrust/fill.h>
#include <thrust/replace.h>
#include <iostream>
#include <vector>

#include <stdio.h>
#include <stdlib.h>
#include <math.h>

#include <cuda.h>
#include <cufft.h>

#ifdef _OPENMP
  #include <omp.h>
  #define TRUE  1
  #define FALSE 0
#else
  #define omp_get_thread_num() 0
#endif


// this is the block size for partitioning the visibilities and intensities calculation on the GPU
// Blocksizes of 128 and 256 were trialled - 256 gave almost 100% utilisation.
#define BLOCK_SIZE 256

/* 

Structure of this file:

Section 1: 
CUDA kernels for calculating the matrix multiplications and transpose multiplications for the 
visibility matrix. These multiplications are given in 3 versions - for delta function pixels, for Gaussian pixels appropximated in a narrow field, and for Gaussian pixels in a wide field.

Section 2:
C++ functions for calling the above CUDA kernels, managing memory and multiprocessing across multiple CUDA devices

Section 3:
Utility functions for implementation of the FISTA algorithm

Section 4:
The Sl1M algorithm including FISTA steps.

Section 5:
Algorithm for finding the maximum eigenvalue for the projection matrix, and for finding an approximate starting point using a gridded fourier transform

*/


namespace radiocuda
{


// Section 1 - CUDA kernels

// 1.1 Delta function pixels
//
// 1. Calculate transform to real intensities from complex visibilities for delta function pixels
// 2. Calculate transform to complex visibilities from real intensities for delta function pixels
// 3. Calculate transform to complex intensities from complex visibilities for delta function pixels
// 4. Calculate transform to complex visibilities from complex intensities for delta function pixels




/**
   Cuda kernel for calculating real intensities from visibilities by multiplying by the
   conjugate transpose of the measurement matrix (formulated for delta function sources).
   Profiling this code shows 99% utilisation of the GPU.
 **/

__global__ void kernel_calcintensdelta(ftype *usr, ftype *vsr, ftype *wsr, ftype *visrr, ftype *visir,int nvis, ftype *xsr, ftype *ysr, ftype *irr,int npix)
{
  __shared__ ftype uss[BLOCK_SIZE];
  __shared__ ftype vss[BLOCK_SIZE];
  __shared__ ftype wss[BLOCK_SIZE];
  __shared__ ftype visrs[BLOCK_SIZE];
  __shared__ ftype visis[BLOCK_SIZE];

  int bx = blockIdx.x;
  int tx = threadIdx.x;
  int posind = bx * BLOCK_SIZE + tx;  // index of the position that is being calculated by this thread

// we break the visibilities to be summed over into groups of BLOCK_SIZE
  ftype ival = 0.0;
  for(int m = 0; m < (nvis - 1)/BLOCK_SIZE + 1; m++)
  {
    int ind = m * BLOCK_SIZE + tx;

    // copy the common data related to visibilities in each block into shared memory
    if(ind < nvis)
    {
      uss[tx]=usr[ind];
      vss[tx]=vsr[ind];
      wss[tx]=wsr[ind];
      visrs[tx]=visrr[ind];
      visis[tx]=visir[ind];
    }
    else // if the number of visibilities is not an exact multiple of 256, fill the rest with zero
    {
      uss[tx]=0.0f;
      vss[tx]=0.0f;
      wss[tx]=0.0f;
      visrs[tx]=0.0f;
      visis[tx]=0.0f;
    }
    __syncthreads(); // Don't go on until all the data is loaded into shared memory

    // calculate the value of the sum for the block
    for(int k = 0; k < BLOCK_SIZE; k++)
    {
      int visind = m*BLOCK_SIZE + k;

      if((posind < npix) && (visind < nvis))
      {
	ftype x = xsr[posind];
	ftype y = ysr[posind];
	ftype wf = sqrtfun(1.0f - x*x - y*y) - 1.0f;
	ftype sph = (ftype)(-2.0f * M_PI)*(x*uss[k] + y*vss[k]+ wf * wss[k]); // visibility kernel
	ftype cosph;
	ftype sinph;
	sincosfun(sph,&sinph,&cosph);
	ival += visrs[k]*cosph + visis[k]*sinph;  // complex conjugate makes this a '+'
	}
    }
    __syncthreads();
  }

  if(posind < npix)
    irr[posind] = ival;

  __syncthreads();

}



/**
   Cuda kernel for calculating complex visibilities from intensities by multiplying by the
   measurement matrix (formulated for delta function sources).

 **/

__global__ void kernel_calcvisdelta(ftype *usr, ftype *vsr, ftype *wsr, ftype *visrr, ftype *visir,int nvis, ftype *xsr, ftype *ysr, ftype *irr,int npix)
{
  __shared__ ftype xss[BLOCK_SIZE];
  __shared__ ftype yss[BLOCK_SIZE];
  __shared__ ftype irs[BLOCK_SIZE];

  int bx = blockIdx.x;
  int tx = threadIdx.x;
  int visind = bx * BLOCK_SIZE + tx;

  ftype vvalr = 0.0;
  ftype vvali = 0.0;
  for(int m = 0; m < (npix - 1)/BLOCK_SIZE + 1; m++)
  {
    int ind = m * BLOCK_SIZE + tx;

    if(ind < npix)
    {
      xss[tx]=xsr[ind];
      yss[tx]=ysr[ind];
      irs[tx]=irr[ind];
    }
    else
    {
      xss[tx]=0.0f;
      yss[tx]=0.0f;
      irs[tx]=0.0f;
    }
    __syncthreads();


    for(int k = 0; k < BLOCK_SIZE; k++)
    {
      int posind = m*BLOCK_SIZE + k;

      if((posind < npix) && (visind < nvis))
      {
	ftype u = usr[visind];
	ftype v = vsr[visind];
	ftype w = wsr[visind];
	ftype wf = sqrtfun(1.0f - xss[k]*xss[k] - yss[k]*yss[k]) - 1.0f;
	ftype sph = (ftype)(-2.0 * M_PI)*(xss[k]*u + yss[k]*v + wf * w);
	ftype cosph;
	ftype sinph;
	sincosfun(sph,&sinph,&cosph);
	
	//	printf("%d %d %d %d %f %f %f %f %f %f %f %f %f %f %f\n",bx,tx,visind,posind,x,y,wf,uss[k],vss[k],wss[k],sph,cosph,sinph,visrs[k],visis[k]);

	vvalr += cosph*irs[k];
	vvali += sinph*irs[k];
	}
    }


    __syncthreads();
  }
  if(visind < nvis)
  {    
    visrr[visind] = vvalr;
    visir[visind] = vvali;
  }
  __syncthreads();

}


/**
   Cuda kernel for calculating real intensities from visibilities by multiplying by the
   conjugate transpose of the measurement matrix (formulated for delta function sources).

 **/

__global__ void kernel_calccompintensdelta(ftype *usr, ftype *vsr, ftype *wsr, ftype *visrr, ftype *visir,int nvis, ftype *xsr, ftype *ysr, ftype *irr, ftype *iir, int npix)
{
  __shared__ ftype uss[BLOCK_SIZE];
  __shared__ ftype vss[BLOCK_SIZE];
  __shared__ ftype wss[BLOCK_SIZE];
  __shared__ ftype visrs[BLOCK_SIZE];
  __shared__ ftype visis[BLOCK_SIZE];

  int bx = blockIdx.x;
  int tx = threadIdx.x;
  int posind = bx * BLOCK_SIZE + tx;

  ftype ivalr = 0.0;
  ftype ivali = 0.0;
  for(int m = 0; m < (nvis - 1)/BLOCK_SIZE + 1; m++)
  {
    int ind = m * BLOCK_SIZE + tx;

    if(ind < nvis)
    {
      uss[tx]=usr[ind];
      vss[tx]=vsr[ind];
      wss[tx]=wsr[ind];
      visrs[tx]=visrr[ind];
      visis[tx]=visir[ind];
    }
    else
    {
      uss[tx]=0.0f;
      vss[tx]=0.0f;
      wss[tx]=0.0f;
      visrs[tx]=0.0f;
      visis[tx]=0.0f;
    }
    __syncthreads();


    for(int k = 0; k < BLOCK_SIZE; k++)
    {
      int visind = m*BLOCK_SIZE + k;

      if((posind < npix) && (visind < nvis))
      {
	ftype x = xsr[posind];
	ftype y = ysr[posind];
	ftype wf = sqrtfun(1.0f - x*x - y*y) - 1.0f;
	ftype sph = (ftype)(-2.0 * M_PI)*(x*uss[k] + y*vss[k]+ wf * wss[k]);
	ftype cosph;
	ftype sinph;
	sincosfun(sph,&sinph,&cosph);
	
	//	printf("%d %d %d %d %f %f %f %f %f %f %f %f %f %f %f\n",bx,tx,visind,posind,x,y,wf,uss[k],vss[k],wss[k],sph,cosph,sinph,visrs[k],visis[k]);

	ivalr += visrs[k]*cosph + visis[k]*sinph;  // complex conjugate makes this a '+'
	ivali += visis[k]*cosph - visrs[k]*sinph;  // complex conjugate makes this a '-'
	}
    }


    __syncthreads();
  }
  if(posind < npix)
  {
    irr[posind] = ivalr;
    iir[posind] = ivali;
  }
  __syncthreads();

}

/**
   Cuda kernel for calculating complex visibilities from complex intensities by multiplying by the
   measurement matrix (formulated for delta function sources).

 **/

__global__ void kernel_calcvisdeltacomp(ftype *usr, ftype *vsr, ftype *wsr, ftype *visrr, ftype *visir,int nvis, ftype *xsr, ftype *ysr, ftype *irr, ftype *iri, int npix)
{
  __shared__ ftype xss[BLOCK_SIZE];
  __shared__ ftype yss[BLOCK_SIZE];
  __shared__ ftype irrs[BLOCK_SIZE];
  __shared__ ftype iris[BLOCK_SIZE];

  int bx = blockIdx.x;
  int tx = threadIdx.x;
  int visind = bx * BLOCK_SIZE + tx;

  ftype vvalr = 0.0;
  ftype vvali = 0.0;
  for(int m = 0; m < (npix - 1)/BLOCK_SIZE + 1; m++)
  {
    int ind = m * BLOCK_SIZE + tx;

    if(ind < npix)
    {
      xss[tx]=xsr[ind];
      yss[tx]=ysr[ind];
      irrs[tx]=irr[ind];
      iris[tx]=iri[ind];
    }
    else
    {
      xss[tx]=0.0f;
      yss[tx]=0.0f;
      irrs[tx]=0.0f;
      iris[tx]=0.0f;
    }
    __syncthreads();


    for(int k = 0; k < BLOCK_SIZE; k++)
    {
      int posind = m*BLOCK_SIZE + k;

      if((posind < npix) && (visind < nvis))
      {
	ftype u = usr[visind];
	ftype v = vsr[visind];
	ftype w = wsr[visind];
	ftype wf = sqrtfun(1.0f - xss[k]*xss[k] - yss[k]*yss[k]) - 1.0f;
	ftype sph = (ftype)(-2.0 * M_PI)*(xss[k]*u + yss[k]*v + wf * w);
	ftype cosph;
	ftype sinph;
	sincosfun(sph,&sinph,&cosph);
	
	//	printf("%d %d %d %d %f %f %f %f %f %f %f %f %f %f %f\n",bx,tx,visind,posind,x,y,wf,uss[k],vss[k],wss[k],sph,cosph,sinph,visrs[k],visis[k]);

	vvalr += cosph*irrs[k] - sinph*iris[k];
	vvali += sinph*irrs[k] + cosph*iris[k];
	}
    }


    __syncthreads();
  }
  if(visind < nvis)
  {    
    visrr[visind] = vvalr;
    visir[visind] = vvali;
  }
  __syncthreads();

}



// 1.2 Gaussian function pixels
//
// 1. Calculate transform to real intensities from complex visibilities for gaussian function pixels
// 2. Calculate transform to complex visibilities from real intensities for gaussian function pixels
// 3. Calculate transform to complex intensities from complex visibilities for gaussian function pixels
// 4. Calculate transform to complex visibilities from complex intensities for gaussian function pixels


/**
   Cuda kernel for calculating real intensities from visibilities by multiplying by the
   conjugate transpose of the measurement matrix (formulated for gaussian function sources).
   Profiling this code shows 99% utilisation of the GPU.
 **/

__global__ void kernel_calcintensgauss(ftype *usr, ftype *vsr, ftype *wsr, ftype *visrr, ftype *visir,int nvis, ftype *xsr, ftype *ysr, ftype *ssr, ftype *irr,int npix)
{
  __shared__ ftype uss[BLOCK_SIZE];
  __shared__ ftype vss[BLOCK_SIZE];
  __shared__ ftype wss[BLOCK_SIZE];
  __shared__ ftype visrs[BLOCK_SIZE];
  __shared__ ftype visis[BLOCK_SIZE];

  int bx = blockIdx.x;
  int tx = threadIdx.x;
  int posind = bx * BLOCK_SIZE + tx;  // index of the position that is being calculated by this thread

// we break the visibilities to be summed over into groups of BLOCK_SIZE
  ftype ival = 0.0;
  for(int m = 0; m < (nvis - 1)/BLOCK_SIZE + 1; m++)
  {
    int ind = m * BLOCK_SIZE + tx;

    // copy the common data related to visibilities in each block into shared memory
    if(ind < nvis)
    {
      uss[tx]=usr[ind];
      vss[tx]=vsr[ind];
      wss[tx]=wsr[ind];
      visrs[tx]=visrr[ind];
      visis[tx]=visir[ind];
    }
    else
    {
      uss[tx]=0.0f;
      vss[tx]=0.0f;
      wss[tx]=0.0f;
      visrs[tx]=0.0f;
      visis[tx]=0.0f;
    }
    __syncthreads();

    // calculate the value of the sum for the block
    for(int k = 0; k < BLOCK_SIZE; k++)
    {
      int visind = m*BLOCK_SIZE + k;

      if((posind < npix) && (visind < nvis))
      {
	ftype x = xsr[posind];
	ftype y = ysr[posind];
	ftype s = ssr[posind];
	ftype s2 = s*s;
	ftype mps2 = -(ftype)M_PI * s2;
	ftype s4 = s2*s2;


	ftype u = uss[k];
	ftype v = vss[k];
	ftype w = wss[k];
	ftype u2 = u*u;
	ftype v2 = v*v;
	ftype w2 = w*w;
	ftype r2 = u2 + v2;
	ftype s2w1 = s2*w;
	ftype s4w2f = 1.0 + s4*w2;
	ftype s4w1f = s4*r2*w;
	ftype slm2w1 = (x*x + y*y)*w;
	ftype sph = (ftype)(-2.0) * (y*v + x*u);
	ftype cosph;
	ftype sinph;
	sincosfun((ftype)M_PI*(sph + (slm2w1 - s4w1f) ) / s4w2f, &sinph, &cosph);
	ftype expa = mps2 *(r2 + (sph + slm2w1)*w) / s4w2f;      
	ftype ef = expfun(expa)/s4w2f;                             
	ftype rep = ef * (cosph - s2w1 * sinph);
	ftype imp = ef * (s2w1 * cosph + sinph);

	ival += visrs[k]*rep + visis[k]*imp;  // complex conjugate makes this a '+'
	}
    }
    __syncthreads();
  }

  if(posind < npix)
    irr[posind] = ival;

  __syncthreads();

}

/**
   Cuda kernel for calculating complex visibilities from intensities by multiplying by the
   measurement matrix (formulated for gaussian function sources).

 **/

__global__ void kernel_calcvisgauss(ftype *usr, ftype *vsr, ftype *wsr, ftype *visrr, ftype *visir,int nvis, ftype *xsr, ftype *ysr, ftype *ssr, ftype *irr,int npix)
{
  __shared__ ftype xss[BLOCK_SIZE];
  __shared__ ftype yss[BLOCK_SIZE];
  __shared__ ftype sss[BLOCK_SIZE];
  __shared__ ftype irs[BLOCK_SIZE];

  int bx = blockIdx.x;
  int tx = threadIdx.x;
  int visind = bx * BLOCK_SIZE + tx;

  ftype vvalr = 0.0;
  ftype vvali = 0.0;
  for(int m = 0; m < (npix - 1)/BLOCK_SIZE + 1; m++)
  {
    int ind = m * BLOCK_SIZE + tx;

    if(ind < npix)
    {
      xss[tx]=xsr[ind];
      yss[tx]=ysr[ind];
      sss[tx]=ssr[ind];
      irs[tx]=irr[ind];
    }
    else
    {
      xss[tx]=0.0f;
      yss[tx]=0.0f;
      sss[tx]=0.0f;
      irs[tx]=0.0f;
    }
    __syncthreads();


    for(int k = 0; k < BLOCK_SIZE; k++)
    {
      int posind = m*BLOCK_SIZE + k;

      if((posind < npix) && (visind < nvis))
      {
	ftype u = usr[visind];
	ftype v = vsr[visind];
	ftype w = wsr[visind];
	ftype u2 = u*u;
	ftype v2 = v*v;
	ftype w2 = w*w;
	ftype r2 = u2 + v2;

	ftype x = xss[k];
	ftype y = yss[k];
	ftype s = sss[k];
	ftype s2 = s*s;
	ftype s2w1 = s2*w;
	ftype mps2 = -(ftype)M_PI * s2;
	ftype s4 = s2*s2;
	ftype s4w2f = 1.0 + s4*w2;
	ftype s4w1f = s4*r2*w;
	ftype slm2w1 = (x*x + y*y)*w;
	ftype sph = (ftype)(-2.0) * (y*v + x*u);
	ftype cosph;
	ftype sinph;
	sincosfun((ftype)M_PI*(sph + (slm2w1 - s4w1f) ) / s4w2f, &sinph, &cosph);
	ftype expa = mps2 *(r2 + (sph + slm2w1)*w) / s4w2f;      
	ftype ef = expfun(expa)/s4w2f;                              

	vvalr += ef * (cosph - s2w1 * sinph) * irs[k];
	vvali += ef * (s2w1 * cosph + sinph) * irs[k];
      }
    }


    __syncthreads();
  }
  if(visind < nvis)
  {    
    visrr[visind] = vvalr;
    visir[visind] = vvali;
  }
  __syncthreads();

}


/**
   Cuda kernel for calculating complex intensities from visibilities by multiplying by the
   conjugate transpose of the measurement matrix (formulated for gaussian function sources).

 **/

__global__ void kernel_calccompintensgauss(ftype *usr, ftype *vsr, ftype *wsr, ftype *visrr, ftype *visir,int nvis, ftype *xsr, ftype *ysr, ftype *ssr, ftype *irr, ftype *iir, int npix)
{
  __shared__ ftype uss[BLOCK_SIZE];
  __shared__ ftype vss[BLOCK_SIZE];
  __shared__ ftype wss[BLOCK_SIZE];
  __shared__ ftype visrs[BLOCK_SIZE];
  __shared__ ftype visis[BLOCK_SIZE];

  int bx = blockIdx.x;
  int tx = threadIdx.x;
  int posind = bx * BLOCK_SIZE + tx;

  ftype ivalr = 0.0;
  ftype ivali = 0.0;
  for(int m = 0; m < (nvis - 1)/BLOCK_SIZE + 1; m++)
  {
    int ind = m * BLOCK_SIZE + tx;

    if(ind < nvis)
    {
      uss[tx]=usr[ind];
      vss[tx]=vsr[ind];
      wss[tx]=wsr[ind];
      visrs[tx]=visrr[ind];
      visis[tx]=visir[ind];
    }
    else
    {
      uss[tx]=0.0f;
      vss[tx]=0.0f;
      wss[tx]=0.0f;
      visrs[tx]=0.0f;
      visis[tx]=0.0f;
    }
    __syncthreads();


    for(int k = 0; k < BLOCK_SIZE; k++)
    {
      int visind = m*BLOCK_SIZE + k;

      if((posind < npix) && (visind < nvis))
      {
	ftype x = xsr[posind];
	ftype y = ysr[posind];
	ftype s = ssr[posind];

	ftype s2 = s*s;
	ftype mps2 = -(ftype)M_PI * s2;
	ftype s4 = s2*s2;

	ftype u = uss[k];
	ftype v = vss[k];
	ftype w = wss[k];
	ftype u2 = u*u;
	ftype v2 = v*v;
	ftype w2 = w*w;
	ftype r2 = u2 + v2;
	ftype s2w1 = s2*w;
	ftype s4w2f = 1.0 + s4*w2;
	ftype s4w1f = s4*r2*w;
	ftype slm2w1 = (x*x + y*y)*w;
	ftype sph = (ftype)(-2.0) * (y*v + x*u);
	ftype cosph;
	ftype sinph;
	sincosfun((ftype)M_PI*(sph + (slm2w1 - s4w1f) ) / s4w2f, &sinph, &cosph);
	ftype expa = mps2 *(r2 + (sph + slm2w1)*w) / s4w2f;      
	ftype ef = expfun(expa)/s4w2f;                             
	ftype rep = ef * (cosph - s2w1 * sinph);
	ftype imp = ef * (s2w1 * cosph + sinph);

	ivalr += visrs[k]*rep + visis[k]*imp;  // complex conjugate makes this a '+'
	ivali += visis[k]*rep - visrs[k]*imp;  // complex conjugate makes this a '-'
	}
    }

    __syncthreads();
  }
  if(posind < npix)
  {
    irr[posind] = ivalr;
    iir[posind] = ivali;
  }
  __syncthreads();

}



/**
   Cuda kernel for calculating complex visibilities from complex intensities by multiplying by the
   measurement matrix (formulated for gaussian function sources).

 **/

__global__ void kernel_calcvisgausscomp(ftype *usr, ftype *vsr, ftype *wsr, ftype *visrr, ftype *visir,int nvis, ftype *xsr, ftype *ysr, ftype *ssr, ftype *irr, ftype *iri, int npix)
{
  __shared__ ftype xss[BLOCK_SIZE];
  __shared__ ftype yss[BLOCK_SIZE];
  __shared__ ftype sss[BLOCK_SIZE];
  __shared__ ftype irrs[BLOCK_SIZE];
  __shared__ ftype iris[BLOCK_SIZE];

  int bx = blockIdx.x;
  int tx = threadIdx.x;
  int visind = bx * BLOCK_SIZE + tx;

  ftype vvalr = 0.0;
  ftype vvali = 0.0;
  for(int m = 0; m < (npix - 1)/BLOCK_SIZE + 1; m++)
  {
    int ind = m * BLOCK_SIZE + tx;

    if(ind < npix)
    {
      xss[tx]=xsr[ind];
      yss[tx]=ysr[ind];
      sss[tx]=ssr[ind];
      irrs[tx]=irr[ind];
      iris[tx]=iri[ind];
    }
    else
    {
      xss[tx]=0.0f;
      yss[tx]=0.0f;
      sss[tx]=0.0f;
      irrs[tx]=0.0f;
      iris[tx]=0.0f;
    }
    __syncthreads();


    for(int k = 0; k < BLOCK_SIZE; k++)
    {
      int posind = m*BLOCK_SIZE + k;

      if((posind < npix) && (visind < nvis))
      {
	ftype u = usr[visind];
	ftype v = vsr[visind];
	ftype w = wsr[visind];
	ftype u2 = u*u;
	ftype v2 = v*v;
	ftype w2 = w*w;
	ftype r2 = u2 + v2;

	ftype x = xss[k];
	ftype y = yss[k];
	ftype s = sss[k];
	ftype s2 = s*s;
	ftype s2w1 = s2*w;
	ftype mps2 = -(ftype)M_PI * s2;
	ftype s4 = s2*s2;
	ftype s4w2f = 1.0 + s4*w2;
	ftype s4w1f = s4*r2*w;
	ftype slm2w1 = (x*x + y*y)*w;
	ftype sph = (ftype)(-2.0) * (y*v + x*u);
	ftype cosph;
	ftype sinph;
	sincosfun((ftype)M_PI*(sph + (slm2w1 - s4w1f) ) / s4w2f, &sinph, &cosph);
	ftype expa = mps2 *(r2 + (sph + slm2w1)*w) / s4w2f;      
	ftype ef = expfun(expa)/s4w2f;                              
	ftype rep = ef * (cosph - s2w1 * sinph);
	ftype imp = ef * (s2w1 * cosph + sinph);

	vvalr += rep*irrs[k] - imp*iris[k];
	vvali += imp*irrs[k] + rep*iris[k];
      }
    }


    __syncthreads();
  }
  if(visind < nvis)
  {    
    visrr[visind] = vvalr;
    visir[visind] = vvali;
  }
  __syncthreads();

}


// Section 2 - C++ function to call CUDA kernels

// 2.1 Delta function pixels

/**

   2.1.1 calculateDeltaVisibilitiesFromIntensities

   Takes
       xs, ys, is   - positions (xs, ys) and  intensities of the model components to be calculated
                    - positions and sigmas are in sine coordinates ( delta M_PI/ (180 3600) where delta is the pixel size in arc seconds)
       us, vs, ws   - (u,v,w) triple of the visibilities to calculate
       realp, imp   - real and imaginary parts of the visibilities

   All vectors are host vectors.

   Code is structured to make use of 2 GPUs as per AWS GPU instances.

   To make it go faster we use CUDA to do the basic calculation. EC2 GPU instances have
   two instances, so we use OMP to break the calculation in half and do each half on 
   a different GPU. Number of threads used by should be set to 2 globally before this
   function is run.

 **/



void calculateDeltaVisibilitiesFromIntensities(hvecf const & xs, hvecf const & ys, hvecf const &is, hvecf const & us, hvecf const & vs, hvecf const & ws, hvecf & realp, hvecf & imp)
{
  int splitpt = us.size()/2; // half the calculation goes on each GPU card
#pragma omp parallel
  {
#pragma omp sections
    {
#pragma omp section
      {
	int res1 = cudaSetDevice(omp_get_thread_num());

	// Copy host vectors to the device
	dvecf xs1(xs);
	dvecf ys1(ys);
	dvecf is1(is);
	dvecf us1(us);
	dvecf vs1(vs);
	dvecf ws1(ws);

	// result will be in the following
	dvecf resr(splitpt);
	dvecf resi(splitpt);
	
	// extract raw pointers from the device vectors to pass into cuda call
	ftype * usr = thrust::raw_pointer_cast( &us1[0] );
	ftype * vsr = thrust::raw_pointer_cast( &vs1[0] );
	ftype * wsr = thrust::raw_pointer_cast( &ws1[0] );
	ftype * isr = thrust::raw_pointer_cast( &is1[0] );
	ftype * xsr = thrust::raw_pointer_cast( &xs1[0] );
	ftype * ysr = thrust::raw_pointer_cast( &ys1[0] );
	ftype * resrr = thrust::raw_pointer_cast( &resr[0] );
	ftype * resir = thrust::raw_pointer_cast( &resi[0] );
	int nvis = splitpt;
	int npix = xs.size();
	
	// call the appropriate CUDA kernel
	dim3 dimGrid((nvis - 1)/BLOCK_SIZE + 1);
	dim3 dimBlock(BLOCK_SIZE);
	kernel_calcvisdelta<<<dimGrid,dimBlock>>>(usr,vsr,wsr,resrr,resir,nvis,xsr,ysr,isr,npix);

	// copy the result into the first half of the returned vectors
	thrust::copy(resr.begin(),resr.end(),realp.begin());
	thrust::copy(resi.begin(),resi.end(),imp.begin());

      }
#pragma omp section
      {
	int res2 = cudaSetDevice(omp_get_thread_num());

	dvecf xs1(xs);
	dvecf ys1(ys);
	dvecf is1(is);
	dvecf us1(us);
	dvecf vs1(vs);
	dvecf ws1(ws);


	dvecf resr(us.size()-splitpt);
	dvecf resi(us.size()-splitpt);

	ftype * usr = thrust::raw_pointer_cast( &us1[0] ); usr += splitpt;
	ftype * vsr = thrust::raw_pointer_cast( &vs1[0] ); vsr += splitpt;
	ftype * wsr = thrust::raw_pointer_cast( &ws1[0] ); wsr += splitpt;
	ftype * isr = thrust::raw_pointer_cast( &is1[0] );
	ftype * xsr = thrust::raw_pointer_cast( &xs1[0] );
	ftype * ysr = thrust::raw_pointer_cast( &ys1[0] );
	ftype * resrr = thrust::raw_pointer_cast( &resr[0] );
	ftype * resir = thrust::raw_pointer_cast( &resi[0] );
	int nvis = us1.size() - splitpt;
	int npix = xs.size();
	
	dim3 dimGrid((nvis - 1)/BLOCK_SIZE + 1);
	dim3 dimBlock(BLOCK_SIZE);
	kernel_calcvisdelta<<<dimGrid,dimBlock>>>(usr,vsr,wsr,resrr,resir,nvis,xsr,ysr,isr,npix);

	// copy the result into the second half of the returned vector
	thrust::copy(resr.begin(),resr.end(),realp.begin()+splitpt);
	thrust::copy(resi.begin(),resi.end(),imp.begin()+splitpt);
      }
    }
  }
}


/**

   2.1.2 calculateDeltaIntensitiesFromVisibilities

   Takes
       xs, ys   - positions (xs, ys) and sigmas (ss) and intensities of the model components to be calculated
                        - positions and sigmas are in sine coordinates ( delta M_PI/ (180 3600) where delta is the pixel size in arc seconds)
       us, vs, ws   - (u,v,w) triple of the visibilities
       realp, imp   - real and imaginary parts of the visibilities to calculate
       is - real part of the intensities projected from the visibilities

   All vectors are host vectors.

 **/


void calculateDeltaIntensitiesFromVisibilities(hvecf const & xs, hvecf const & ys, hvecf const & us, hvecf const & vs, hvecf const & ws, hvecf const & realp, hvecf const & imp, hvecf & is)
{
  int splitpt = xs.size()/2;
#pragma omp parallel
  {
#pragma omp sections
    {
#pragma omp section
      {
// We run this part on the first GPU selected using the thread number
	int res1 = cudaSetDevice(omp_get_thread_num());

	// copy host memory to the device.
	dvecf xs1(xs);
	dvecf ys1(ys);
	dvecf us1(us);
	dvecf vs1(vs);
	dvecf ws1(ws);
	dvecf realp1(realp);
	dvecf imp1(imp);

	// make a result vector on the device that is half size
	dvecf res(splitpt);
	
	// get raw pointers to cuda memory to pass into the cuda function
	ftype * usr = thrust::raw_pointer_cast( &us1[0] );
	ftype * vsr = thrust::raw_pointer_cast( &vs1[0] );
	ftype * wsr = thrust::raw_pointer_cast( &ws1[0] );
	ftype * visrr = thrust::raw_pointer_cast( &realp1[0] );
	ftype * visir = thrust::raw_pointer_cast( &imp1[0] );
	ftype * xsr = thrust::raw_pointer_cast( &xs1[0] );
	ftype * ysr = thrust::raw_pointer_cast( &ys1[0] );
	ftype * resr = thrust::raw_pointer_cast( &res[0] );
	
	//specify the size of the vectors
	int nvis = us1.size();
	int npix = splitpt;
	
	// we just do one dimensional gridding as we are calculating a matrix times a vector
	// but we don't need to store the vector
	dim3 dimGrid((npix - 1)/BLOCK_SIZE + 1);
	dim3 dimBlock(BLOCK_SIZE);

	kernel_calcintensdelta<<<dimGrid,dimBlock>>>(usr,vsr,wsr,visrr,visir,nvis,xsr,ysr,resr,npix);

	thrust::copy(res.begin(),res.end(),is.begin());

      }
#pragma omp section
      {
      // This second part we calculate on the second GPU - all the visibilities are needed
      // but we only compute over half the xsr and ysrs, and the output is copied into the second half of is
	int res2 = cudaSetDevice(omp_get_thread_num());

	dvecf xs1(xs);
	dvecf ys1(ys);
	dvecf us1(us);
	dvecf vs1(vs);
	dvecf ws1(ws);
	dvecf realp1(realp);
	dvecf imp1(imp);

	dvecf res(xs.size() - splitpt);
	
	ftype * usr = thrust::raw_pointer_cast( &us1[0] );
	ftype * vsr = thrust::raw_pointer_cast( &vs1[0] );
	ftype * wsr = thrust::raw_pointer_cast( &ws1[0] );
	ftype * visrr = thrust::raw_pointer_cast( &realp1[0] );
	ftype * visir = thrust::raw_pointer_cast( &imp1[0] );
	ftype * xsr = thrust::raw_pointer_cast( &xs1[0] ); xsr += splitpt;
	ftype * ysr = thrust::raw_pointer_cast( &ys1[0] ); ysr += splitpt;
	ftype * resr = thrust::raw_pointer_cast( &res[0] );
	int nvis = us1.size();
	int npix = xs.size() - splitpt;
	
	dim3 dimGrid((npix - 1)/BLOCK_SIZE + 1);
	dim3 dimBlock(BLOCK_SIZE);

	kernel_calcintensdelta<<<dimGrid,dimBlock>>>(usr,vsr,wsr,visrr,visir,nvis,xsr,ysr,resr,npix);

	thrust::copy(res.begin(),res.end(),is.begin()+splitpt);
      }
    }
  }
}


/**

   2.1.3 calculateComplexDeltaIntensitiesFromVisibilities

   Takes
       xs, ys   - positions (xs, ys) and sigmas (ss) and intensities of the model components to be calculated
                        - positions and sigmas are in sine coordinates ( delta M_PI/ (180 3600) where delta is the pixel size in arc seconds)
       us, vs, ws   - (u,v,w) triple of the visibilities
       realp, imp   - real and imaginary parts of the visibilities to calculate
      isr, isi - real and imaginary parts of the intensities projected from the visibilities

   All vectors are host vectors.

 **/

void calculateComplexDeltaIntensitiesFromVisibilities(hvecf const & xs, hvecf const & ys, hvecf const & us, hvecf const & vs, hvecf const & ws, hvecf const & realp, hvecf const & imp, hvecf & isr, hvecf & isi)
{
  int splitpt = xs.size()/2;
#pragma omp parallel
  {
#pragma omp sections
    {
#pragma omp section
      {
	int res1 = cudaSetDevice(omp_get_thread_num());
	dvecf xs1(xs);
	dvecf ys1(ys);
	dvecf us1(us);
	dvecf vs1(vs);
	dvecf ws1(ws);
	dvecf realp1(realp);
	dvecf imp1(imp);

	dvecf ires(splitpt);
	dvecf rres(splitpt);
	
	ftype * usr = thrust::raw_pointer_cast( &us1[0] );
	ftype * vsr = thrust::raw_pointer_cast( &vs1[0] );
	ftype * wsr = thrust::raw_pointer_cast( &ws1[0] );
	ftype * visrr = thrust::raw_pointer_cast( &realp1[0] );
	ftype * visir = thrust::raw_pointer_cast( &imp1[0] );
	ftype * xsr = thrust::raw_pointer_cast( &xs1[0] );
	ftype * ysr = thrust::raw_pointer_cast( &ys1[0] );
	ftype * rresr = thrust::raw_pointer_cast( &rres[0] );
	ftype * iresr = thrust::raw_pointer_cast( &ires[0] );
	int nvis = us1.size();
	int npix = splitpt;
	
	dim3 dimGrid((npix - 1)/BLOCK_SIZE + 1);
	dim3 dimBlock(BLOCK_SIZE);


	kernel_calccompintensdelta<<<dimGrid,dimBlock>>>(usr,vsr,wsr,visrr,visir,nvis,xsr,ysr,rresr,iresr,npix);

	thrust::copy(rres.begin(),rres.end(),isr.begin());
	thrust::copy(ires.begin(),ires.end(),isi.begin());

      }
#pragma omp section
      {
	int res2 = cudaSetDevice(omp_get_thread_num());

	dvecf xs1(xs);
	dvecf ys1(ys);
	dvecf us1(us);
	dvecf vs1(vs);
	dvecf ws1(ws);
	dvecf realp1(realp);
	dvecf imp1(imp);

	dvecf rres(xs.size() - splitpt);
	dvecf ires(xs.size() - splitpt);
	
	ftype * usr = thrust::raw_pointer_cast( &us1[0] );
	ftype * vsr = thrust::raw_pointer_cast( &vs1[0] );
	ftype * wsr = thrust::raw_pointer_cast( &ws1[0] );
	ftype * visrr = thrust::raw_pointer_cast( &realp1[0] );
	ftype * visir = thrust::raw_pointer_cast( &imp1[0] );
	ftype * xsr = thrust::raw_pointer_cast( &xs1[0] ); xsr += splitpt;
	ftype * ysr = thrust::raw_pointer_cast( &ys1[0] ); ysr += splitpt;
	ftype * rresr = thrust::raw_pointer_cast( &rres[0] );
	ftype * iresr = thrust::raw_pointer_cast( &ires[0] );
	int nvis = us1.size();
	int npix = xs.size() - splitpt;
	
	dim3 dimGrid((npix - 1)/BLOCK_SIZE + 1);
	dim3 dimBlock(BLOCK_SIZE);

	kernel_calccompintensdelta<<<dimGrid,dimBlock>>>(usr,vsr,wsr,visrr,visir,nvis,xsr,ysr,rresr,iresr,npix);

	thrust::copy(rres.begin(),rres.end(),isr.begin()+splitpt);
	thrust::copy(ires.begin(),ires.end(),isi.begin()+splitpt);
      }
    }
  }
}


/**

   2.1.4 calculateDeltaVisibilitiesFromComplexIntensities

   Takes
       xs, ys, isr, isi   - positions (xs, ys) and  intensities of the model components to be calculated
                    - positions and sigmas are in sine coordinates ( delta M_PI/ (180 3600) where delta is the pixel size in arc seconds)
       us, vs, ws   - (u,v,w) triple of the visibilities to calculate
       realp, imp   - real and imaginary parts of the visibilities

   All vectors are host vectors.

 **/



void calculateDeltaVisibilitiesFromComplexIntensities(hvecf const & xs, hvecf const & ys, hvecf const &isr, hvecf const &isi, hvecf const & us, hvecf const & vs, hvecf const & ws, hvecf & realp, hvecf & imp)
{
  int splitpt = us.size()/2;
#pragma omp parallel
  {
#pragma omp sections
    {
#pragma omp section
      {
	int res1 = cudaSetDevice(omp_get_thread_num());
	dvecf xs1(xs);
	dvecf ys1(ys);
	dvecf isr1(isr);
	dvecf isi1(isi);
	dvecf us1(us);
	dvecf vs1(vs);
	dvecf ws1(ws);


	dvecf resr(splitpt);
	dvecf resi(splitpt);
	
	ftype * usr = thrust::raw_pointer_cast( &us1[0] );
	ftype * vsr = thrust::raw_pointer_cast( &vs1[0] );
	ftype * wsr = thrust::raw_pointer_cast( &ws1[0] );
	ftype * isrr = thrust::raw_pointer_cast( &isr1[0] );
	ftype * isir = thrust::raw_pointer_cast( &isi1[0] );
	ftype * xsr = thrust::raw_pointer_cast( &xs1[0] );
	ftype * ysr = thrust::raw_pointer_cast( &ys1[0] );
	ftype * resrr = thrust::raw_pointer_cast( &resr[0] );
	ftype * resir = thrust::raw_pointer_cast( &resi[0] );
	int nvis = splitpt;
	int npix = xs.size();
	
	dim3 dimGrid((nvis - 1)/BLOCK_SIZE + 1);
	dim3 dimBlock(BLOCK_SIZE);


	kernel_calcvisdeltacomp<<<dimGrid,dimBlock>>>(usr,vsr,wsr,resrr,resir,nvis,xsr,ysr,isrr,isir,npix);

	thrust::copy(resr.begin(),resr.end(),realp.begin());
	thrust::copy(resi.begin(),resi.end(),imp.begin());

      }
#pragma omp section
      {
	int res2 = cudaSetDevice(omp_get_thread_num());

	dvecf xs1(xs);
	dvecf ys1(ys);
	dvecf isr1(isr);
	dvecf isi1(isi);
	dvecf us1(us);
	dvecf vs1(vs);
	dvecf ws1(ws);


	dvecf resr(us.size()-splitpt);
	dvecf resi(us.size()-splitpt);

	ftype * usr = thrust::raw_pointer_cast( &us1[0] ); usr += splitpt;
	ftype * vsr = thrust::raw_pointer_cast( &vs1[0] ); vsr += splitpt;
	ftype * wsr = thrust::raw_pointer_cast( &ws1[0] ); wsr += splitpt;
	ftype * isrr = thrust::raw_pointer_cast( &isr1[0] );
	ftype * isir = thrust::raw_pointer_cast( &isi1[0] );
	ftype * xsr = thrust::raw_pointer_cast( &xs1[0] );
	ftype * ysr = thrust::raw_pointer_cast( &ys1[0] );
	ftype * resrr = thrust::raw_pointer_cast( &resr[0] );
	ftype * resir = thrust::raw_pointer_cast( &resi[0] );
	int nvis = us1.size() - splitpt;
	int npix = xs.size();
	
	dim3 dimGrid((nvis - 1)/BLOCK_SIZE + 1);
	dim3 dimBlock(BLOCK_SIZE);

	kernel_calcvisdeltacomp<<<dimGrid,dimBlock>>>(usr,vsr,wsr,resrr,resir,nvis,xsr,ysr,isrr,isir,npix);

	thrust::copy(resr.begin(),resr.end(),realp.begin()+splitpt);
	thrust::copy(resi.begin(),resi.end(),imp.begin()+splitpt);
      }
    }
  }
}


// 2.2 Gaussian function pixels


/**

   2.2.1 calculateGaussIntensitiesFromVisibilities

   Takes
       xs, ys, ss   - positions (xs, ys) and sigmas (ss) and intensities of the model components to be calculated
                    - positions and sigmas are in sine coordinates ( delta M_PI/ (180 3600) where delta is the pixel size in arc seconds)
       us, vs, ws   - (u,v,w) triple of the visibilities
       realp, imp   - real and imaginary parts of the visibilities to calculate
      is - real part of the intensities projected from the visibilities

   All vectors are host vectors.

 **/

void calculateGaussIntensitiesFromVisibilities(hvecf const & xs, hvecf const & ys, hvecf const & ss, hvecf const & us, hvecf const & vs, hvecf const & ws, hvecf const & realp, hvecf const & imp, hvecf & is)
{
  int splitpt = xs.size()/2;
#pragma omp parallel
  {
#pragma omp sections
    {
#pragma omp section
      {
// We run this part on the first GPU selected using the thread number
	int res1 = cudaSetDevice(omp_get_thread_num());

	// copy host memory to the device.
	dvecf xs1(xs);
	dvecf ys1(ys);
	dvecf ss1(ss);
	dvecf us1(us);
	dvecf vs1(vs);
	dvecf ws1(ws);
	dvecf realp1(realp);
	dvecf imp1(imp);

	// make a result vector on the device that is half size
	dvecf res(splitpt);
	
	// get raw pointers to cuda memory to pass into the cuda function
	ftype * usr = thrust::raw_pointer_cast( &us1[0] );
	ftype * vsr = thrust::raw_pointer_cast( &vs1[0] );
	ftype * wsr = thrust::raw_pointer_cast( &ws1[0] );
	ftype * visrr = thrust::raw_pointer_cast( &realp1[0] );
	ftype * visir = thrust::raw_pointer_cast( &imp1[0] );
	ftype * xsr = thrust::raw_pointer_cast( &xs1[0] );
	ftype * ysr = thrust::raw_pointer_cast( &ys1[0] );
	ftype * ssr = thrust::raw_pointer_cast( &ss1[0] );
	ftype * resr = thrust::raw_pointer_cast( &res[0] );
	
	//specify the size of the vectors
	int nvis = us1.size();
	int npix = splitpt;
	
	// we just do one dimensional gridding as we are calculating a matrix times a vector
	// but we don't need to store the vector
	dim3 dimGrid((npix - 1)/BLOCK_SIZE + 1);
	dim3 dimBlock(BLOCK_SIZE);

	kernel_calcintensgauss<<<dimGrid,dimBlock>>>(usr,vsr,wsr,visrr,visir,nvis,xsr,ysr,ssr,resr,npix);

	thrust::copy(res.begin(),res.end(),is.begin());

      }
#pragma omp section
      {
	int res2 = cudaSetDevice(omp_get_thread_num());

	dvecf xs1(xs);
	dvecf ys1(ys);
	dvecf ss1(ss);
	dvecf us1(us);
	dvecf vs1(vs);
	dvecf ws1(ws);
	dvecf realp1(realp);
	dvecf imp1(imp);

	dvecf res(xs.size() - splitpt);
	
	ftype * usr = thrust::raw_pointer_cast( &us1[0] );
	ftype * vsr = thrust::raw_pointer_cast( &vs1[0] );
	ftype * wsr = thrust::raw_pointer_cast( &ws1[0] );
	ftype * visrr = thrust::raw_pointer_cast( &realp1[0] );
	ftype * visir = thrust::raw_pointer_cast( &imp1[0] );
	ftype * xsr = thrust::raw_pointer_cast( &xs1[0] ); xsr += splitpt;
	ftype * ysr = thrust::raw_pointer_cast( &ys1[0] ); ysr += splitpt;
	ftype * ssr = thrust::raw_pointer_cast( &ss1[0] ); ssr += splitpt;
	ftype * resr = thrust::raw_pointer_cast( &res[0] );
	int nvis = us1.size();
	int npix = xs.size() - splitpt;
	
	dim3 dimGrid((npix - 1)/BLOCK_SIZE + 1);
	dim3 dimBlock(BLOCK_SIZE);

	kernel_calcintensgauss<<<dimGrid,dimBlock>>>(usr,vsr,wsr,visrr,visir,nvis,xsr,ysr,ssr,resr,npix);

	thrust::copy(res.begin(),res.end(),is.begin()+splitpt);
      }
    }
  }
}


/**

   2.2.2 calculateGaussVisibilitiesFromIntensities

   Takes
       xs, ys, ss, is   - positions (xs, ys) and  intensities of the model components to be calculated
                        - positions and sigmas are in sine coordinates ( delta M_PI/ (180 3600) where delta is the pixel size in arc seconds)
       us, vs, ws       - (u,v,w) triple of the visibilities to calculate
       realp, imp       - real and imaginary parts of the visibilities

   All vectors are host vectors.

 **/


void calculateGaussVisibilitiesFromIntensities(hvecf const & xs, hvecf const & ys, hvecf const & ss, hvecf const &is, hvecf const & us, hvecf const & vs, hvecf const & ws, hvecf & realp, hvecf & imp)
{
  int splitpt = us.size()/2;
#pragma omp parallel
  {
#pragma omp sections
    {
#pragma omp section
      {
	int res1 = cudaSetDevice(omp_get_thread_num());
	dvecf xs1(xs);
	dvecf ys1(ys);
	dvecf ss1(ss);
	dvecf is1(is);
	dvecf us1(us);
	dvecf vs1(vs);
	dvecf ws1(ws);


	dvecf resr(splitpt);
	dvecf resi(splitpt);
	
	ftype * usr = thrust::raw_pointer_cast( &us1[0] );
	ftype * vsr = thrust::raw_pointer_cast( &vs1[0] );
	ftype * wsr = thrust::raw_pointer_cast( &ws1[0] );
	ftype * isr = thrust::raw_pointer_cast( &is1[0] );
	ftype * xsr = thrust::raw_pointer_cast( &xs1[0] );
	ftype * ysr = thrust::raw_pointer_cast( &ys1[0] );
	ftype * ssr = thrust::raw_pointer_cast( &ss1[0] );
	ftype * resrr = thrust::raw_pointer_cast( &resr[0] );
	ftype * resir = thrust::raw_pointer_cast( &resi[0] );
	int nvis = splitpt;
	int npix = xs.size();
	
	dim3 dimGrid((nvis - 1)/BLOCK_SIZE + 1);
	dim3 dimBlock(BLOCK_SIZE);


	kernel_calcvisgauss<<<dimGrid,dimBlock>>>(usr,vsr,wsr,resrr,resir,nvis,xsr,ysr,ssr,isr,npix);

	thrust::copy(resr.begin(),resr.end(),realp.begin());
	thrust::copy(resi.begin(),resi.end(),imp.begin());

      }
#pragma omp section
      {
	int res2 = cudaSetDevice(omp_get_thread_num());

	dvecf xs1(xs);
	dvecf ys1(ys);
	dvecf ss1(ss);
	dvecf is1(is);
	dvecf us1(us);
	dvecf vs1(vs);
	dvecf ws1(ws);


	dvecf resr(us.size()-splitpt);
	dvecf resi(us.size()-splitpt);

	ftype * usr = thrust::raw_pointer_cast( &us1[0] ); usr += splitpt;
	ftype * vsr = thrust::raw_pointer_cast( &vs1[0] ); vsr += splitpt;
	ftype * wsr = thrust::raw_pointer_cast( &ws1[0] ); wsr += splitpt;
	ftype * isr = thrust::raw_pointer_cast( &is1[0] );
	ftype * xsr = thrust::raw_pointer_cast( &xs1[0] );
	ftype * ysr = thrust::raw_pointer_cast( &ys1[0] );
	ftype * ssr = thrust::raw_pointer_cast( &ss1[0] );
	ftype * resrr = thrust::raw_pointer_cast( &resr[0] );
	ftype * resir = thrust::raw_pointer_cast( &resi[0] );
	int nvis = us1.size() - splitpt;
	int npix = xs.size();
	
	dim3 dimGrid((nvis - 1)/BLOCK_SIZE + 1);
	dim3 dimBlock(BLOCK_SIZE);

	kernel_calcvisgauss<<<dimGrid,dimBlock>>>(usr,vsr,wsr,resrr,resir,nvis,xsr,ysr,ssr,isr,npix);

	thrust::copy(resr.begin(),resr.end(),realp.begin()+splitpt);
	thrust::copy(resi.begin(),resi.end(),imp.begin()+splitpt);
      }
    }
  }
}

/**

   2.2.3 calculateComplexGaussIntensitiesFromVisibilities

   Takes
       xs, ys, ss   - positions (xs, ys) and sigmas (ss) and intensities of the model components to be calculated
                    - positions and sigmas are in sine coordinates ( delta M_PI/ (180 3600) where delta is the pixel size in arc seconds)
       us, vs, ws   - (u,v,w) triple of the visibilities
       realp, imp   - real and imaginary parts of the visibilities to calculate
       isr, isi - real and imaginary parts of the intensities projected from the visibilities

   All vectors are host vectors.

 **/


void calculateComplexGaussIntensitiesFromVisibilities(hvecf const & xs, hvecf const & ys, hvecf const & ss, hvecf const & us, hvecf const & vs, hvecf const & ws, hvecf const & realp, hvecf const & imp, hvecf & isr, hvecf & isi)
{
  int splitpt = xs.size()/2;
#pragma omp parallel
  {
#pragma omp sections
    {
#pragma omp section
      {
	int res1 = cudaSetDevice(omp_get_thread_num());
	dvecf xs1(xs);
	dvecf ys1(ys);
	dvecf ss1(ss);
	dvecf us1(us);
	dvecf vs1(vs);
	dvecf ws1(ws);
	dvecf realp1(realp);
	dvecf imp1(imp);

	dvecf ires(splitpt);
	dvecf rres(splitpt);
	
	ftype * usr = thrust::raw_pointer_cast( &us1[0] );
	ftype * vsr = thrust::raw_pointer_cast( &vs1[0] );
	ftype * wsr = thrust::raw_pointer_cast( &ws1[0] );
	ftype * visrr = thrust::raw_pointer_cast( &realp1[0] );
	ftype * visir = thrust::raw_pointer_cast( &imp1[0] );
	ftype * xsr = thrust::raw_pointer_cast( &xs1[0] );
	ftype * ysr = thrust::raw_pointer_cast( &ys1[0] );
	ftype * ssr = thrust::raw_pointer_cast( &ss1[0] );
	ftype * rresr = thrust::raw_pointer_cast( &rres[0] );
	ftype * iresr = thrust::raw_pointer_cast( &ires[0] );
	int nvis = us1.size();
	int npix = splitpt;
	
	dim3 dimGrid((npix - 1)/BLOCK_SIZE + 1);
	dim3 dimBlock(BLOCK_SIZE);


	kernel_calccompintensgauss<<<dimGrid,dimBlock>>>(usr,vsr,wsr,visrr,visir,nvis,xsr,ysr,ssr,rresr,iresr,npix);

	thrust::copy(rres.begin(),rres.end(),isr.begin());
	thrust::copy(ires.begin(),ires.end(),isi.begin());

      }
#pragma omp section
      {
	int res2 = cudaSetDevice(omp_get_thread_num());

	dvecf xs1(xs);
	dvecf ys1(ys);
	dvecf ss1(ss);
	dvecf us1(us);
	dvecf vs1(vs);
	dvecf ws1(ws);
	dvecf realp1(realp);
	dvecf imp1(imp);

	dvecf rres(xs.size() - splitpt);
	dvecf ires(xs.size() - splitpt);
	
	ftype * usr = thrust::raw_pointer_cast( &us1[0] );
	ftype * vsr = thrust::raw_pointer_cast( &vs1[0] );
	ftype * wsr = thrust::raw_pointer_cast( &ws1[0] );
	ftype * visrr = thrust::raw_pointer_cast( &realp1[0] );
	ftype * visir = thrust::raw_pointer_cast( &imp1[0] );
	ftype * xsr = thrust::raw_pointer_cast( &xs1[0] ); xsr += splitpt;
	ftype * ysr = thrust::raw_pointer_cast( &ys1[0] ); ysr += splitpt;
	ftype * ssr = thrust::raw_pointer_cast( &ss1[0] ); ssr += splitpt;
	ftype * rresr = thrust::raw_pointer_cast( &rres[0] );
	ftype * iresr = thrust::raw_pointer_cast( &ires[0] );
	int nvis = us1.size();
	int npix = xs.size() - splitpt;
	
	dim3 dimGrid((npix - 1)/BLOCK_SIZE + 1);
	dim3 dimBlock(BLOCK_SIZE);

	kernel_calccompintensgauss<<<dimGrid,dimBlock>>>(usr,vsr,wsr,visrr,visir,nvis,xsr,ysr,ssr,rresr,iresr,npix);

	thrust::copy(rres.begin(),rres.end(),isr.begin()+splitpt);
	thrust::copy(ires.begin(),ires.end(),isi.begin()+splitpt);
      }
    }
  }
}


/**

   2.2.4 calculateGaussVisibilitiesFromComplexIntensities

   Takes
       xs, ys, ss, isr, isi   - positions (xs, ys) and  intensities of the model components to be calculated
                              - positions and sigmas are in sine coordinates ( delta M_PI/ (180 3600) where delta is the pixel size in arc seconds)
       us, vs, ws             - (u,v,w) triple of the visibilities to calculate
       realp, imp             - real and imaginary parts of the visibilities

   All vectors are host vectors.

 **/


void calculateGaussVisibilitiesFromComplexIntensities(hvecf const & xs, hvecf const & ys, hvecf const & ss, hvecf const &isr, hvecf const &isi, hvecf const & us, hvecf const & vs, hvecf const & ws, hvecf & realp, hvecf & imp)
{
  int splitpt = us.size()/2;
#pragma omp parallel
  {
#pragma omp sections
    {
#pragma omp section
      {
	int res1 = cudaSetDevice(omp_get_thread_num());
	dvecf xs1(xs);
	dvecf ys1(ys);
	dvecf ss1(ss);
	dvecf isr1(isr);
	dvecf isi1(isi);
	dvecf us1(us);
	dvecf vs1(vs);
	dvecf ws1(ws);


	dvecf resr(splitpt);
	dvecf resi(splitpt);
	
	ftype * usr = thrust::raw_pointer_cast( &us1[0] );
	ftype * vsr = thrust::raw_pointer_cast( &vs1[0] );
	ftype * wsr = thrust::raw_pointer_cast( &ws1[0] );
	ftype * isrr = thrust::raw_pointer_cast( &isr1[0] );
	ftype * isir = thrust::raw_pointer_cast( &isi1[0] );
	ftype * xsr = thrust::raw_pointer_cast( &xs1[0] );
	ftype * ysr = thrust::raw_pointer_cast( &ys1[0] );
	ftype * ssr = thrust::raw_pointer_cast( &ss1[0] );
	ftype * resrr = thrust::raw_pointer_cast( &resr[0] );
	ftype * resir = thrust::raw_pointer_cast( &resi[0] );
	int nvis = splitpt;
	int npix = xs.size();
	
	dim3 dimGrid((nvis - 1)/BLOCK_SIZE + 1);
	dim3 dimBlock(BLOCK_SIZE);


	kernel_calcvisgausscomp<<<dimGrid,dimBlock>>>(usr,vsr,wsr,resrr,resir,nvis,xsr,ysr,ssr,isrr,isir,npix);

	thrust::copy(resr.begin(),resr.end(),realp.begin());
	thrust::copy(resi.begin(),resi.end(),imp.begin());

      }
#pragma omp section
      {
	int res2 = cudaSetDevice(omp_get_thread_num());

	dvecf xs1(xs);
	dvecf ys1(ys);
	dvecf ss1(ss);
	dvecf isr1(isr);
	dvecf isi1(isi);
	dvecf us1(us);
	dvecf vs1(vs);
	dvecf ws1(ws);


	dvecf resr(us.size()-splitpt);
	dvecf resi(us.size()-splitpt);

	ftype * usr = thrust::raw_pointer_cast( &us1[0] ); usr += splitpt;
	ftype * vsr = thrust::raw_pointer_cast( &vs1[0] ); vsr += splitpt;
	ftype * wsr = thrust::raw_pointer_cast( &ws1[0] ); wsr += splitpt;
	ftype * isrr = thrust::raw_pointer_cast( &isr1[0] );
	ftype * isir = thrust::raw_pointer_cast( &isi1[0] );
	ftype * xsr = thrust::raw_pointer_cast( &xs1[0] );
	ftype * ysr = thrust::raw_pointer_cast( &ys1[0] );
	ftype * ssr = thrust::raw_pointer_cast( &ss1[0] );
	ftype * resrr = thrust::raw_pointer_cast( &resr[0] );
	ftype * resir = thrust::raw_pointer_cast( &resi[0] );
	int nvis = us1.size() - splitpt;
	int npix = xs.size();
	
	dim3 dimGrid((nvis - 1)/BLOCK_SIZE + 1);
	dim3 dimBlock(BLOCK_SIZE);

	kernel_calcvisgausscomp<<<dimGrid,dimBlock>>>(usr,vsr,wsr,resrr,resir,nvis,xsr,ysr,ssr,isrr,isir,npix);

	thrust::copy(resr.begin(),resr.end(),realp.begin()+splitpt);
	thrust::copy(resi.begin(),resi.end(),imp.begin()+splitpt);
      }
    }
  }
}


// Section 3 - Utility functions for implementing SL1M

// These are thrust functors that allow for simple operations on host or device vectors - allows cuda processing without explicitly writing cuda kernels

typedef thrust::tuple<ftype,ftype> Tuple2;

// functor to take the modulus squared of a set of complex numbers
template<typename T>
struct modulus_squared : public thrust::unary_function<Tuple2,T>
{
  __host__ __device__ T operator()(const Tuple2 &x) const
    {
      return thrust::get<0>(x)*thrust::get<0>(x) + thrust::get<1>(x)*thrust::get<1>(x);
    }
};

// Claculate the norm squared of a complex vector represented by two vectors
ftype normsquared(hvecf const & rep, hvecf const & imp)
{
    return thrust::transform_reduce(thrust::make_zip_iterator(thrust::make_tuple(rep.begin(),imp.begin())),
				    thrust::make_zip_iterator(thrust::make_tuple(rep.end(),imp.end())),
				    modulus_squared<ftype>(),
				    0.0,
				    thrust::plus<ftype>()      
				    );
}

// functor to perform a subtraction and threshold operation without enforcing positivity
template<typename T>
struct subtractAndThreshold : public thrust::binary_function<T,T,T>
{
  __host__ 
  __device__ subtractAndThreshold(ftype lambdain, ftype Lin) : lambda(lambdain), L(Lin) { d = lambda / L; }

  __host__ 
    __device__ T operator()(const T &xout, const T &y) const 
  {
    ftype v = y - xout/L;
    if(v > d) v -= d;
    else if(v < -d) v += d;
    else v = 0.0;
    
    return v;
  }

  ftype lambda, L, d;

};

// functor to perform a subtraction and threshold operation _with_ enforcing positivity
template<typename T>
struct subtractAndThresholdPositivity : public thrust::binary_function<T,T,T>
{
  __host__ 
  __device__ subtractAndThresholdPositivity(ftype lambdain, ftype Lin) : lambda(lambdain), L(Lin) { d = lambda / L; }


  __host__ 
    __device__ T operator()(const T &xout, const T &y) const 
  {
    ftype v = y - xout/L;
    if(v > d) v -= d;
    else v = 0.0;
    
    return v;
  }

  ftype lambda, L, d;

};

// Functor to take the absolute value of an element of a vector
template<typename T>
struct absoluteValue : public thrust::unary_function<T,T>
{
  __host__ __device__ T operator()(const T &x) const
  {
    return x < T(0) ? -x : x;
  }
};

// Functor to do an interpolation between two vectors
template<typename T>
struct interpolateEstimates : public thrust::binary_function<T,T,T>
{
  __host__ 
  __device__ interpolateEstimates(ftype tvin) : tv(tvin) {}


  __host__ 
    __device__ T operator()(const T &x1, const T &x0) const 
  {
    return x1 + tv * (x1 - x0);
  }

  ftype tv;
};



// Section 4 : SL1M algorithm implementation

/**

   4.1 deltaL1Minimisation - L1 minimisation for a given set of data and visibility positions using SL1M derived for delta function pixels

   Takes
	lambda       - ponalty for L1 minimisation
	L            - maximum eigenvalue of the projection matrix 
	maxiters     - maximum number of iterations to make
        us, vs, ws   - (u,v,w) triple of the visibilities to calculate
        visi, visr   - real and imaginary parts of the observed visibilities
        xs, ys       - positions (xs, ys) of the model components to be calculated
	xout         - result of the minimsation - the intensity at the corresponding xs,ys position
	l1err        - l1 norm of the result
	l2err	     - L2 mismatch between the observed visibilities and the predicted visibilities
 **/



int deltaL1Minimisation(ftype lambda, ftype L, int maxiters, hvecf const & us, hvecf const & vs, hvecf const & ws, hvecf const & visr, hvecf const & visi, hvecf const & xs, hvecf const & ys, hvecf & xout, ftype & l1err, ftype & l2err)
{
  int iter = 0;
  //  int old_nzero = 0;
  ftype t2 = 1.0, t1 = 1.0, tv = 1.0;


  l1err = 0.0;
  l2err = 0.0;
  ftype totalerr = 0.0;
  //  ftype oldtotalerr = 0.0;
  
  hvecf xin = xout;
  hvecf y = xout;

  hvecf residualr(visr.size(),0.0f);
  hvecf residuali(visr.size(),0.0f);
  while(iter < maxiters)
  {
    calculateDeltaVisibilitiesFromIntensities(xs,ys,y,us,vs,ws,residualr,residuali); // Multiply by matrix

    thrust::transform(residualr.begin(), residualr.end(), visr.begin(), residualr.begin(), thrust::minus<ftype>());
    thrust::transform(residuali.begin(), residuali.end(), visi.begin(), residuali.begin(), thrust::minus<ftype>());

    // calculate l2norm
    l2err = normsquared(residualr,residuali);

    // calculate l1norm
    l1err = transform_reduce(y.begin(), y.end(), absoluteValue<ftype>(),0.0,thrust::plus<ftype>());
    totalerr = l2err + lambda * l1err;


    calculateDeltaIntensitiesFromVisibilities(xs,ys,us,vs,ws,residualr,residuali,xout); // Multiply by conjugate transpose
    
    // update estimate and
    // shrink-threshold and positivity

    thrust::transform(xout.begin(), xout.end(),
		      y.begin(),
		      xout.begin(),
		      subtractAndThresholdPositivity<ftype>(lambda, L)
		      );

    
    // calculate step size
    t2 = (1.0 + sqrt(1.0 + 4.0 * t1 * t1))/2.0;
    tv = (t1 - 1.0)/t2;
		      
    thrust::transform(xout.begin(), xout.end(),
		      xin.begin(),
		      y.begin(),
		      interpolateEstimates<ftype>(tv)
		      );

    thrust::copy(xout.begin(), xout.end(), xin.begin());
    t1 = t2;

/*    if(iter %20 == 0) // output every 20th step size (for diagnostic purposed)
    {
      hvecf xwrite = xout;
      
      char nbuf[200];
      sprintf(nbuf, "res%d.dat", iter);
      FILE * dat = fopen(nbuf,"wb");
      int sz = 128;
      fwrite(&sz,sizeof(int),1,dat);
      fwrite(&sz,sizeof(int),1,dat);
      fwrite(&(xwrite[0]),sizeof(ftype),xwrite.size(),dat);
      fclose(dat);
    }
*/
    // calculate number of non-zero entries

    int countz = thrust::count(xout.begin(), xout.end(), 0.0f);

    std::cout << "Finished iteration " << iter << " with error " << l2err << " " << l1err << " " << totalerr << 
      " non-zero: " << xout.size() - countz <<  std::endl;


    //    if((iter != 0) && (abs(totalerr - oldtotalerr)/abs(totalerr) < TOTALERRTOL) && abs(old_nzero - countz) < SAMECOUNTLIMIT)
    //  {
    //	break;
    //  }


    //oldtotalerr = totalerr;
    //old_nzero = countz;
    iter++;


  }

  return iter;

}


/**

   4.2 gaussL1Minimisation - L1 minimisation for a given set of data and visibility positions using SL1M derived for gaussian function pixels

   Takes
	lambda       - ponalty for L1 minimisation
	L            - maximum eigenvalue of the projection matrix 
	maxiters     - maximum number of iterations to make
        us, vs, ws   - (u,v,w) triple of the visibilities to calculate
        visi, visr   - real and imaginary parts of the observed visibilities
        xs, ys, ss   - positions (xs, ys) and scales (ss) of the model components to be calculated
	xout         - result of the minimsation - the intensity at the corresponding xs,ys position
	l1err        - l1 norm of the result
	l2err	     - L2 mismatch between the observed visibilities and the predicted visibilities
 **/

int gaussL1Minimisation(ftype lambda, ftype L, int maxiters, hvecf const & us, hvecf const & vs, hvecf const & ws, hvecf const & visr, hvecf const & visi, hvecf const & xs, hvecf const & ys, hvecf const & ss, hvecf & xout, ftype & l1err, ftype & l2err)
{
  int iter = 0;
  //  int old_nzero = 0;
  ftype t2 = 1.0, t1 = 1.0, tv = 1.0;


  l1err = 0.0;
  l2err = 0.0;
  ftype totalerr = 0.0;
  //  ftype oldtotalerr = 0.0;
  
  hvecf xin = xout;
  hvecf y = xout;

  hvecf residualr(visr.size(),0.0f);
  hvecf residuali(visr.size(),0.0f);
  while(iter < maxiters)
  {
    calculateGaussVisibilitiesFromIntensities(xs,ys,ss,y,us,vs,ws,residualr,residuali); // Multiply by matrix

    thrust::transform(residualr.begin(), residualr.end(), visr.begin(), residualr.begin(), thrust::minus<ftype>());
    thrust::transform(residuali.begin(), residuali.end(), visi.begin(), residuali.begin(), thrust::minus<ftype>());

    // calculate l2norm
    l2err = normsquared(residualr,residuali);

    // calculate l1norm
    l1err = transform_reduce(y.begin(), y.end(), absoluteValue<ftype>(),0.0,thrust::plus<ftype>());
    totalerr = l2err + lambda * l1err;


    calculateGaussIntensitiesFromVisibilities(xs,ys,ss,us,vs,ws,residualr,residuali,xout); // Multiply by conjugate transpose
    
    // update estimate and
    // shrink-threshold and positivity

    thrust::transform(xout.begin(), xout.end(),
		      y.begin(),
		      xout.begin(),
		      subtractAndThresholdPositivity<ftype>(lambda, L)
		      );

    
    // calculate step size
    t2 = (1.0 + sqrt(1.0 + 4.0 * t1 * t1))/2.0;
    tv = (t1 - 1.0)/t2;
		      
    thrust::transform(xout.begin(), xout.end(),
		      xin.begin(),
		      y.begin(),
		      interpolateEstimates<ftype>(tv)
		      );

    thrust::copy(xout.begin(), xout.end(), xin.begin());
    t1 = t2;

/*    if(iter %100 == 0)
    {
      hvecf xwrite = xout;
      
      char nbuf[200];
      sprintf(nbuf, "res%d.dat", iter);
      FILE * dat = fopen(nbuf,"wb");
      int sz = 128;
      fwrite(&sz,sizeof(int),1,dat);
      fwrite(&sz,sizeof(int),1,dat);
      fwrite(&(xwrite[0]),sizeof(ftype),xwrite.size(),dat);
      fclose(dat);
    }*/

    // calculate number of non-zero entries

    int countz = thrust::count(xout.begin(), xout.end(), 0.0f);

    std::cout << "Finished iteration " << iter << " with error " << l2err << " " << l1err << " " << totalerr << 
      " non-zero: " << xout.size() - countz <<  std::endl;


    //    if((iter != 0) && (abs(totalerr - oldtotalerr)/abs(totalerr) < 1e-9) && abs(old_nzero - countz) < 1)
    //  {
    //	break;
    //  }


    //oldtotalerr = totalerr;
    //old_nzero = countz;
    iter++;


  }

  return iter;

}

/**

   4.3 SynthesisL1Minimisation - general interface for L1 minimisation for a given set of data and visibility positions using SL1M

   if ssf has non-zero size, it is assumed that it is the same size as xsf, ysf and holds the scales of the corresponding pixels.
   Gaussian pixels are then used in the calculation

   Takes
        usf, vsf, wsf    - (u,v,w) triple of the visibilities to calculate
        chvsre, chvsim   - real and imaginary parts of the observed visibilities
        xsf, ysf, ssf    - positions (xs, ys) and scales (ss) of the model components to be calculated
	imin 	  	 - initial conditions for the intensities at the positions (xsf, ysf) to start from
	imout            - result of the minimsation - the intensity at the corresponding xsf,ysf position
	maxEV            - maximum eigenvalue of the projection matrix - if zero on input it will be calculated
	lambda           - penalty for L1 minimisation
	maxiters         - maximum number of iterations to make

 **/


void SynthesisL1Minimisation
(
 hvecf const & usf,
 hvecf const & vsf,
 hvecf const & wsf,
 hvecf const & chvsre,
 hvecf const & chvsim,
 hvecf const & xsf,
 hvecf const & ysf,
 hvecf const & ssf,
 hvecf const & imin,
 hvecf & imout,
 ftype maxEV,
 ftype lambda,
 ftype maxiters
)
{

  bool hassigmas = (ssf.size() != 0);

  if(maxEV == 0.0)
  {
    hvecf evr(xsf.size());
    hvecf evi(xsf.size());

    if(hassigmas)
      maxEV = radiocuda::calculateMaxEigenvalueAndVector(100,xsf,ysf,ssf,usf,vsf,wsf,evr,evi);
    else
      maxEV = radiocuda::calculateMaxEigenvalueAndVector(100,xsf,ysf,usf,vsf,wsf,evr,evi);
  }

  ftype L = maxEV;
  ftype l1err = 0.0, l2err = 0.0;

  int iters = 0;
  imout = imin;

  if(hassigmas)
    iters = radiocuda::gaussL1Minimisation(lambda,L,maxiters,usf,vsf,wsf,chvsre,chvsim,xsf,ysf,ssf,imout,l1err,l2err);
  else
    iters = radiocuda::deltaL1Minimisation(lambda,L,maxiters,usf,vsf,wsf,chvsre,chvsim,xsf,ysf,imout,l1err,l2err);

  std::cout << "Iterations: " << iters << "L1 err: " << l1err << " L2 err: " << l2err << std::endl;
}


// Section 5:
// Algorithm for finding the maximum eigenvalue for the projection matrix, and for finding an approximate starting point using a gridded fourier transform


// Calculate the maximum eigen value of the projection matrix specified by the u,v,w coordinates of the visibilities and the 
// coordinates of the sampled delta function pixels.

ftype calculateMaxEigenvalueAndVector(int maxiters, hvecf const & xs, hvecf const & ys, hvecf const & us, hvecf const & vs, hvecf const & ws, hvecf & evr, hvecf & evi)
{

  // stick a random vector in evr and evi using thrust random routine - could be moved on to device if it proves to
  // be a performance bottleneck

  hvecf hvr1(evr.size());
  hvecf hvr2(evr.size());
  thrust::default_random_engine rng;
  thrust::uniform_real_distribution<ftype> dist(0.0f, 1.0f);
  for (size_t i = 0; i < evr.size(); i++)
  {  
    hvr1[i] = static_cast<ftype>(dist(rng));
    hvr2[i] = static_cast<ftype>(dist(rng));
  }
  evr = hvr1;
  evi = hvr2;
  
  // Define a temp vector for the intermediate values calculated in visibility space
  hvecf tempr(us.size());
  hvecf tempi(us.size());

  // loop - two projections, renormalisation, check convergence
  ftype norm = 0.0, oldnorm = 1e31;
  int iter = 0;

//  std::cout << xs[1] << " " << ys[1] << " " << us[1] << " " <<vs[1] << " " << evr[1] << std::endl;

  while(iter < maxiters)
  {

    calculateDeltaVisibilitiesFromComplexIntensities(xs,ys,evr,evi,us,vs,ws,tempr,tempi); // Multiply by matrix
    calculateComplexDeltaIntensitiesFromVisibilities(xs,ys,us,vs,ws,tempr,tempi,evr,evi); // Multiply by conjugate transpose

    
    norm = sqrt(normsquared(evr,evi));


    thrust::transform(evr.begin(), evr.end(), 
		      thrust::make_constant_iterator<ftype>(1.0/norm), 
		      evr.begin(),
		      thrust::multiplies<ftype>()
		      );
    thrust::transform(evi.begin(), evi.end(), 
		      thrust::make_constant_iterator<ftype>(1.0/norm), 
		      evi.begin(),
		      thrust::multiplies<ftype>()
		      );

    iter++;
    if(abs(oldnorm - norm)/abs(norm) < MAXEVTOL) break;
    oldnorm = norm;
  }

  std::cerr << "Performed " << iter << " iterations in calculation of maximum EigenValue = " << norm << std::endl;
  return norm;
}



// Calculate the maximum eigen value of the projection matrix specified by the u,v,w coordinates of the visibilities and the 
// coordinates and scales of the sampled gaussian pixels 

ftype calculateMaxEigenvalueAndVector(int maxiters, hvecf const & xs, hvecf const & ys, hvecf const & ss, hvecf const & us, hvecf const & vs, hvecf const & ws, hvecf & evr, hvecf & evi)
{

  // stick a random vector in evr and evi using thrust random routine - could be moved on to device if it proves to
  // be a performance bottleneck

  hvecf hvr1(evr.size());
  hvecf hvr2(evr.size());
  thrust::default_random_engine rng;
  thrust::uniform_real_distribution<ftype> dist(0.0f, 1.0f);
  for (size_t i = 0; i < evr.size(); i++)
  {  
    hvr1[i] = static_cast<ftype>(dist(rng));
    hvr2[i] = static_cast<ftype>(dist(rng));
  }
  evr = hvr1;
  evi = hvr2;
  
  // Define a temp vector for the intermediate values calculated in visibility space
  hvecf tempr(us.size());
  hvecf tempi(us.size());

  // loop - two projections, renormalisation, check convergence
  ftype norm = 0.0, oldnorm = 1e31;
  int iter = 0;

  std::cerr << "Calculating maximum eigevalue" << std::endl;

  while(iter < maxiters)
  {
    calculateGaussVisibilitiesFromComplexIntensities(xs,ys,ss,evr,evi,us,vs,ws,tempr,tempi); // Multiply by matrix
    calculateComplexGaussIntensitiesFromVisibilities(xs,ys,ss,us,vs,ws,tempr,tempi,evr,evi); // Multiply by conjugate transpose

    norm = sqrt(normsquared(evr,evi));


    thrust::transform(evr.begin(), evr.end(), 
		      thrust::make_constant_iterator<ftype>(1.0/norm), 
		      evr.begin(),
		      thrust::multiplies<ftype>()
		      );
    thrust::transform(evi.begin(), evi.end(), 
		      thrust::make_constant_iterator<ftype>(1.0/norm), 
		      evi.begin(),
		      thrust::multiplies<ftype>()
		      );

    iter++;
    if(abs(oldnorm - norm)/abs(norm) < MAXEVTOL) break;
    oldnorm = norm;
  }

  std::cerr << "Performed " << iter << " iterations in calculation of maximum EigenValue = " << norm << std::endl;
  return norm;
}


/**

   5.3 calculateInitialConditionsOnGrid

   Calculates an approximate solution to the inverse problem by spatting visibilities onto 
   a rectangular grid at w=0 and then doing the inverse Fourier transform

   Takes
       us, vs, ws   - (u,v,w) triple of the visibilities to calculate
       vr, vi       - real and imaginary parts of the visibilities
       Ds           - pixel spacing in arcsecond
       sz           - size of output (sz x sz pixel image)
       res          - the resulting image

 **/

void calculateInitialConditionsOnGrid(
				      hvecf const & us,
				      hvecf const & vs,
				      hvecf const & ws,
				      hvecf const & vr,
				      hvecf const & vi,
				      ftype Ds,
				      int sz,
				      hvecf & res
				      )
{
  hvecf rp(sz*sz,0.0);
  hvecf ip(sz*sz,0.0);
  hvecf wt(sz*sz,0.0);
  hvecf ires(sz*sz,0.0);
  res = hvecf(sz*sz,0.0);

  int hsz = sz/2;
  ftype del = sz * Ds;
  for(int i = 0; i < us.size(); i++)
  {
    int hp1 = (int)(del * us[i]) + hsz;
    int vp1 = (int)(del * vs[i]) + hsz;
    int hp2 = -(int)(del * us[i]) + hsz;
    int vp2 = -(int)(del * vs[i]) + hsz;
    if(!(hp1 < 0||hp1 >=sz||hp2 < 0||hp2 >=sz||vp1 < 0||vp1 >=sz||vp2 < 0||vp2 >=sz))
    {
      hp1 -= hsz;
      hp2 -= hsz;
      vp1 -= hsz;
      vp2 -= hsz;
      if(hp1 < 0) hp1 += sz;
      if(hp2 < 0) hp2 += sz;
      if(vp1 < 0) vp1 += sz;
      if(vp2 < 0) vp2 += sz;
      rp[vp2*sz+hp2]+=vr[i];
      ip[vp2*sz+hp2]-=vi[i];
      rp[vp1*sz+hp1]+=vr[i];
      ip[vp1*sz+hp1]+=vi[i];
      wt[vp2*sz+hp2]+=1;
      wt[vp1*sz+hp1]+=1;
    }
  }
  for(int i = 0; i < sz*sz; i++)
  {
    if(wt[i]) { rp[i] /= wt[i]; ip[i] /= wt[i]; }
  }

  ftype * rpr = thrust::raw_pointer_cast( &rp[0] );
  ftype * ipr = thrust::raw_pointer_cast( &ip[0] );
  ftype * resr = thrust::raw_pointer_cast( &ires[0] );

  // inverse Fourier transform
  // into a new vector

  cufftHandle plan;
  cufftComplex *data;
  cudaMalloc((void**)&data, sizeof(cufftComplex)*sz*sz);

  float * tmp_d = (float *)data;

  cudaMemcpy2D(tmp_d,2*sizeof(tmp_d[0]),rpr, 1*sizeof(rpr[0]),sizeof(rpr[0]),sz*sz,cudaMemcpyHostToDevice);
  cudaMemcpy2D(tmp_d+1,2*sizeof(tmp_d[0]),ipr, 1*sizeof(ipr[0]),sizeof(ipr[0]),sz*sz,cudaMemcpyHostToDevice);

  cufftPlan2d(&plan, sz, sz, CUFFT_C2C);
  cufftExecC2C(plan, data, data, CUFFT_INVERSE);
  
  cudaMemcpy2D(resr,sizeof(resr[0]),tmp_d,2*sizeof(tmp_d[0]),sizeof(tmp_d[0]),sz*sz,cudaMemcpyDeviceToHost);

// divide by number of elements and shift back
  for(int y = 0; y < sz; y++)
  {
    int ny = y - hsz;
    if(ny < 0) ny += sz;
    for(int x = 0; x< sz; x++)
    {
      int nx = x - hsz;
      if(nx < 0) nx += sz;
      res[ny*sz + nx] = ires[y*sz + x]/(sz*sz);
    }
  }
  cufftDestroy(plan);
  cudaFree(data);
}



}
