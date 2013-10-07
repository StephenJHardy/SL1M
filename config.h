#ifndef CONFIG_H 
#define CONFIG_H

/* redefine the following to switch between float and double versions of the code
   remember this impacts the format of the files read in */

#define ftype float
#define expfun exp
#define sqrtfun sqrt
#define sincosfun sincos

#define MAXEVTOL (ftype)1e-5
#define SAMECOUNTLIMIT 6
#define TOTALERRTOL (ftype)1e-9
#endif
