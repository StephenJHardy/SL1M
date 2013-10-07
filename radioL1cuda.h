#ifndef RADIOL1CUDA_H
#define RADIOL1CUDA_H

/*  
    Copyright Stephen Hardy 2013 
    Released under BSD license: http://opensource.org/licenses/BSD-2-Clause 
*/

// edit the following to switch between float and double versions of the code
#include "config.h"

// we use thrust vectors to do memory management. There is a (very) slight overhead, but
// it is much more elegant than mallocing and copying data around.

#include <thrust/host_vector.h>
#include <thrust/device_vector.h>

typedef thrust::host_vector<ftype> hvecf;
typedef thrust::device_vector<ftype> dvecf;

namespace radiocuda
{

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

  void SynthesisL1Minimisation (
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
			       );


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
					);



  // The folowing functions are used internally to implement the SL1M algorithm.
  // Documnetation to their interfaces is found in radioL1cuda.cu

  void calculateDeltaIntensitiesFromVisibilities(hvecf const & xs, hvecf const & ys, hvecf const & us, hvecf const & vs, hvecf const & ws, hvecf const & realp, hvecf const & imp, hvecf & is);

  void calculateDeltaVisibilitiesFromIntensities(hvecf const & xs, hvecf const & ys, hvecf const &is, hvecf const & us, hvecf const & vs, hvecf const & ws, hvecf & realp, hvecf & imp);

  int deltaL1Minimisation(ftype lambda, ftype L, int maxiters, hvecf const & us, hvecf const & vs, hvecf const & ws, hvecf const & visr, hvecf const & visi, hvecf const & xs, hvecf const & ys, hvecf & xout, ftype & l1err, ftype & l2err);

  void calculateComplexDeltaIntensitiesFromVisibilities(hvecf const & xs, hvecf const & ys, hvecf const & us, hvecf const & vs, hvecf const & ws, hvecf const & realp, hvecf const & imp, hvecf & isr, hvecf &isi);

  void calculateDeltaVisibilitiesFromComplexIntensities(hvecf const & xs, hvecf const & ys, hvecf const &isr, hvecf const &isi, hvecf const & us, hvecf const & vs, hvecf const & ws, hvecf & realp, hvecf & imp);

  ftype calculateMaxEigenvalueAndVector(int maxiters, hvecf const & xs, hvecf const & ys, hvecf const & us, hvecf const & vs, hvecf const & ws, hvecf & evr, hvecf & evi);



  void calculateGaussIntensitiesFromVisibilities(hvecf const & xs, hvecf const & ys, hvecf const & ss, hvecf const & us, hvecf const & vs, hvecf const & ws, hvecf const & realp, hvecf const & imp, hvecf & is);

  void calculateGaussVisibilitiesFromIntensities(hvecf const & xs, hvecf const & ys, hvecf const & ss, hvecf const &is, hvecf const & us, hvecf const & vs, hvecf const & ws, hvecf & realp, hvecf & imp);

  int deltaL1Minimisation(ftype lambda, ftype L, int maxiters, hvecf const & us, hvecf const & vs, hvecf const & ws, hvecf const & visr, hvecf const & visi, hvecf const & xs, hvecf const & ys, hvecf const & ss, hvecf & xout, ftype & l1err, ftype & l2err);

  void calculateComplexGaussIntensitiesFromVisibilities(hvecf const & xs, hvecf const & ys, hvecf const & ss, hvecf const & us, hvecf const & vs, hvecf const & ws, hvecf const & realp, hvecf const & imp, hvecf & isr, hvecf &isi);

  void calculateGaussVisibilitiesFromComplexIntensities(hvecf const & xs, hvecf const & ys, hvecf const & ss, hvecf const &isr, hvecf const &isi, hvecf const & us, hvecf const & vs, hvecf const & ws, hvecf & realp, hvecf & imp);

  ftype calculateMaxEigenvalueAndVector(int maxiters, hvecf const & xs, hvecf const & ys, hvecf const & ss, hvecf const & us, hvecf const & vs, hvecf const & ws, hvecf & evr, hvecf & evi);

}


#endif // #ifndef RADIOL1CUDA_H
