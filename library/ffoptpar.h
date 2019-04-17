#ifndef GUARD_ffoptpar_h
#define GUARD_ffoptpar_h

#include "petsc.h"
#include "initialize.h"
#include "filters.h"
#include "logging.h"
#include "ffopt.h"

double ffoptpar(int ndofAll, double *dofAll, double *dofgradAll, void *data);

#endif

