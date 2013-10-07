#include "readvises.h"
#include <stdio.h>
#include <stdlib.h>


void readVises(const char * fname, int &nrec, int &nchan, ftype &basef, ftype &chanw, ftype * &us, ftype * &vs, ftype * &ws, ftype * &vises_re, ftype * &vises_im)
{
  FILE * f = 0;
  f = fopen(fname, "rb");
#ifdef TEST
  if(f) printf("open FILE succeeded\n");
  else printf("open FILE failed\n");
#endif

  fread(&nrec, sizeof(nrec),1,f);
  fread(&nchan, sizeof(nchan),1,f);

  fread(&basef, sizeof(basef),1,f);
  fread(&chanw, sizeof(chanw),1,f);

#ifdef TEST
  printf("nrec = %d nchan = %d\n",nrec,nchan);
#endif

  us = (ftype *) malloc(nrec * sizeof(ftype));
  vs = (ftype *) malloc(nrec * sizeof(ftype));
  ws = (ftype *) malloc(nrec * sizeof(ftype));
  vises_re = (ftype *) malloc(nrec * nchan * sizeof(ftype));
  vises_im = (ftype *) malloc(nrec * nchan * sizeof(ftype));


#ifdef TEST
  if(us && vs && ws && vises_re && vises_im) printf("Malloc succeeded\n");
#endif

  int rb = 0;
  rb = fread(us,sizeof(ftype),nrec,f);
#ifdef TEST
  printf("Read us : %d ftypes\n",rb);
#endif
  rb = fread(vs,sizeof(ftype),nrec,f);
#ifdef TEST
  printf("Read vs : %d ftypes\n",rb);
#endif
  rb = fread(ws,sizeof(ftype),nrec,f);
#ifdef TEST
  printf("Read ws : %d ftypes\n",rb);
#endif
  rb = fread(vises_re,sizeof(ftype),nrec*nchan,f);
#ifdef TEST
  printf("Read vises_re : %d ftypes\n",rb);
#endif
  rb = fread(vises_im,sizeof(ftype),nrec*nchan,f);
#ifdef TEST
  printf("Read vises_im : %d ftypes\n",rb);
#endif
  fclose(f);
}
