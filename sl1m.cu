#include "config.h"

#include <cuda.h>
#include <thrust/host_vector.h>
#include <thrust/device_vector.h>
#include <thrust/copy.h>

#include <math.h>
#include <stdio.h>

#include <iostream>
#include <vector>

#include "radioL1cuda.h"
#include "readvises.h"

#ifdef _OPENMP
  #include <omp.h>
  #define TRUE  1
  #define FALSE 0
#else
  #define omp_get_thread_num() 0
#endif

void configure2devicecuda()
{
#ifdef _OPENMP
   (void) omp_set_dynamic(FALSE);
   if (omp_get_dynamic()) {printf("Warning: dynamic adjustment of threads has been set\n");}
   (void) omp_set_num_threads(2);
#pragma omp parallel
   {
#pragma omp sections
     {
#pragma omp section
       {
	 cudaSetDevice(omp_get_thread_num());
       }
#pragma omp section
       {
	 cudaSetDevice(omp_get_thread_num());
       }
     }
   }
#else
   fprintf(stderr,"Must be compiled with OPEN MP"); 
   abort();
#endif
}

#include <string>
#include <iostream>
#include <algorithm>
#include <tclap/CmdLine.h>

int main(int argc, char *argv[])
{


  try {  

    TCLAP::CmdLine cmd("SL1M - synthesis through L1 minimisation", ' ', "0.1");

    // scale of gaussian representing a component in pixels
    TCLAP::ValueArg<ftype> gaussianSzArg("g","gaussian","gaussian component size in pixels (0.0 = delta function pixels)",false,(ftype)0.0,"float",cmd);

    // use fft to generate initial approximate solution
    TCLAP::SwitchArg fftInitSwitch("f","fftinit","use FFT to calculate initial approximation", cmd, false);

    // read initial approximation from a file
    TCLAP::ValueArg<std::string> initialInNameArg("a","initialin","read initial appoximation from this file",false,"","string",cmd);

    // output the initial approximating used
    TCLAP::ValueArg<std::string> initialOutNameArg("k","initialout","output the initial approximation to this file",false,"","string",cmd);

    // threshold on size of visibility components
    TCLAP::ValueArg<ftype> visThreshArg("t","threshold","discard visibilities greater than this size (0.0 = use all)",false,(ftype)0.0,"int",cmd);

    // limit on number of records
    TCLAP::ValueArg<int> recordsNumArg("r","records","Max records to process (0 = all)",false,0,"int",cmd);

    // maximum number of iterations to take
    TCLAP::ValueArg<int> maxItersArg("m","maxiters","Maximum number of FISTA steps to take",false,300,"int",cmd);

    // what channel to process
    TCLAP::ValueArg<int> channelNumArg("c","channel","Channel number to process",false,0,"int",cmd);

    // pixel size in arc seconds
    TCLAP::ValueArg<ftype> pixelSzArg("p","pixel","pixel size in arc seconds",false,(ftype)30.0,"float",cmd);

    // what size of image to produce
    TCLAP::ValueArg<int> imageSzArg("s","size","Size of output image (assumed square)",false,1024,"float",cmd);

    // regularisation parameter for L1 minimisation
    TCLAP::ValueArg<ftype> lambdaArg("l","lambda","regularisation parameter for L1 minimisation",true,(ftype)0.0,"float",cmd);

    // name of the out binary raster file
    TCLAP::ValueArg<std::string> outputNameArg("o","output","Output file name",false,"output.dat","string",cmd);

    // name of the input binary data file
    TCLAP::ValueArg<std::string> inputNameArg("i","input","Input file name",false,"syndat2n3pc.bin","string",cmd);

    // Parse the argv array.
    cmd.parse( argc, argv );
    
    // Get the value parsed by each arg. 
    std::string inName = inputNameArg.getValue();
    std::string outName = outputNameArg.getValue();
    int channel = channelNumArg.getValue();
    int recordsMax = recordsNumArg.getValue();
    ftype visThresh = visThreshArg.getValue();
    int imSize = imageSzArg.getValue();
    ftype pixelSize = pixelSzArg.getValue();
    ftype gaussianSize = gaussianSzArg.getValue();
    ftype lambdaSize = lambdaArg.getValue();
    int maxIters = maxItersArg.getValue();
    bool fftInit = fftInitSwitch.getValue();
    std::string initInName = initialInNameArg.getValue();
    std::string initOutName = initialOutNameArg.getValue();


    // load in the visibilities from binary file off disk
    ftype *usin=0, *vsin=0,*wsin=0,*vsrein=0,*vsimin=0;
    int nchan = 0, nrec = 0;
    ftype basef = 0.0, channelw = 0.0;
    
    //everything comes off disk as ftypes
    readVises(inName.c_str(), nrec, nchan, basef, channelw, usin, vsin, wsin, vsrein, vsimin);

    std::cout << nchan << " channels with base at " << basef << " and channel width " << channelw << std::endl;

    // prepare visibility related data stored in 
    hvecf usfp(nrec);
    hvecf vsfp(nrec);
    hvecf wsfp(nrec);
    hvecf chvsrep(nrec);
    hvecf chvsimp(nrec);
  
    if(visThresh != 0.0)
      std::cout << "Limiting size of visibilities to " << visThresh << std::endl;

    int count = 0;
    for(int i = 0; i < nrec; i++) {
      ftype rp = vsrein[i*nchan + channel];
      ftype ip = vsimin[i*nchan + channel];
      if((rp*rp+ip*ip < visThresh) || (visThresh == 0.0)) {
	ftype scale = basef + channel *channelw;
	usfp[count] = (ftype)(usin[i]*scale);
	vsfp[count] = (ftype)(vsin[i]*scale);
	wsfp[count] = (ftype)(wsin[i]*scale);
	chvsrep[count] = (ftype)vsrein[i*nchan + channel];
	chvsimp[count] = (ftype)vsimin[i*nchan + channel];
	count++;
	if(recordsMax && (count >= recordsMax)) break; // dont read any more if we have enough
      }
    } 

    std::cout << "Working with "<< count << " of " << nrec << " records for this channel"<<std::endl;

    // transfer the data into the host vectors, so they can be moved to the graphics card as
    // required
    hvecf usf(usfp.begin(),usfp.begin()+count);
    hvecf vsf(vsfp.begin(),vsfp.begin()+count);
    hvecf wsf(wsfp.begin(),wsfp.begin()+count);
    hvecf chvsre(chvsrep.begin(),chvsrep.begin()+count);
    hvecf chvsim(chvsimp.begin(),chvsimp.begin()+count);
    nrec = count;
    
    // sampling related constants
    int sz = imSize;
    int hsz = imSize/2;
    pixelSize *= (ftype)M_PI / (180.0 * 3600.0);

    ftype sigma0 = gaussianSize;

    // set up a vector of x,y,scale pixel positions for each pixel in the image
    hvecf xsf(sz*sz);
    hvecf ysf(sz*sz);

    for(int y = 0; y < sz; y++)  {
      for(int x = 0; x < sz; x++) {
        xsf[y*sz + x] = (x - hsz)*pixelSize;
        ysf[y*sz + x] = (y - hsz)*pixelSize;
      } 
    }

    hvecf ssf;
    if(sigma0) {
      ssf = hvecf(sz*sz,1.0);
      for(int y = 0; y < sz; y++) 
	for(int x = 0; x < sz; x++)
	  ssf[y*sz + x] = (sigma0 * pixelSize);
    }

    hvecf xinit(xsf.size(), ftype(0.0));
  
    if(initialInNameArg.isSet())
    {
      FILE * dat = fopen(initInName.c_str(),"rb");
      int sz2;
      fread(&sz2,sizeof(int),1,dat);
      fread(&sz2,sizeof(int),1,dat);
      fread(&(xinit[0]),sizeof(ftype),xsf.size(),dat);
      fclose(dat);
    }
    else if(fftInit)
    {
      radiocuda::calculateInitialConditionsOnGrid(usf,vsf,wsf,chvsre,chvsim,pixelSize,sz,xinit);
    }
    
    /*  hvecf xwrite = xinit;
	char nbuf[200];
	sprintf(nbuf, "init.dat");
	FILE * dat = fopen(nbuf,"wb");
	fwrite(&sz,sizeof(int),1,dat);
	fwrite(&sz,sizeof(int),1,dat);
	fwrite(&(xwrite[0]),sizeof(ftype),xwrite.size(),dat);
	fclose(dat);
    */


    hvecf xout(xsf.size(), ftype(0.0));

    ftype lambda = lambdaSize;
    ftype maxEV = 0.0;
    
    radiocuda::SynthesisL1Minimisation(usf,vsf,wsf,chvsre,chvsim,xsf,ysf,ssf,xinit,xout,maxEV,lambda,maxIters);
   
    {
      FILE * dat = fopen(outName.c_str(),"wb");
      fwrite(&sz,sizeof(int),1,dat);
      fwrite(&sz,sizeof(int),1,dat);
      fwrite(&(xout[0]),sizeof(ftype),xout.size(),dat);
      fclose(dat);
    }

  }
  catch (TCLAP::ArgException &e)  // catch any exceptions
  { 
    std::cerr << "error: " << e.error() << " for arg " << e.argId() << std::endl; 
  }
  
}

