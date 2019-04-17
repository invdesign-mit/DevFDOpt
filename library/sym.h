#ifndef GUARD_sym_h
#define GUARD_sym_h

#include "petsc.h"
#include "ffopt.h"

void symx(PetscReal *u_half, PetscReal *u_full, int mx,int my,int nlayers,int mxcells,int mycells, int transpose);

double ffoptsym(int ndof_half, double *dof_half, double *dofgrad_half, void *data);

double ffoptsym_maximinconstraint(int ndofhalf_with_dummy, double *dofhalf_with_dummy, double *dofgradhalf_with_dummy, void *data);

#endif
