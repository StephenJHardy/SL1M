#ifndef READVISES_H
#define READVISES_H

#include "config.h"

/* 
To avoid having to parse general uvfits files or measurement sets files from Casa to extract the visibility data, 
this code accepts a very simple binary format that can be extracted reasonably simply after preprocessing has been 
done (i.e. flagging, calibration, and merging polarisations if required).

The file format is binary and little endian. This reads in as "ftype" data - changing types requires different data files at the moment
1-4: integer number of visibility records
5-8: integer number of channels
9-12: float base frequency of channel 0
13-16: float frequency spacing of the channels
4*nrec: float u coordinates
4*nrec: float v coordinates
4*nrec: float w coordinates
nrec*nchan*4: real parts of the visibility data, for each visibility record, nchan floats are read in
nrec*nchan*4: imaginary parts of the visibility data, for each visibility record, nchan floats are read in

NOTE: this call mallocs memory (hence the pointer pass by reference) the user is resonsible for free-ing these: us, vs, ws, vises_re, vises_im.

TODO: replace this horrible hack - probably make the code callable from python so that this can be avoided altogether.

*/

void 
readVises(const char * fname, int &nrec, int &nchan, ftype &basef, ftype &chanw, ftype * &us, ftype * &vs, ftype * &ws, ftype * &vises_re, ftype * &vises_im);

#endif
