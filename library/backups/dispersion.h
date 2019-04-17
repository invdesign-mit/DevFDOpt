#ifndef GUARD_dispersion_h
#define GUARD_dispersion_h

#include "petsc.h"
#include "type.h"
#include "initialize.h"

typedef struct{

  PetscReal freq;
  PetscReal nmed;
  PetscReal theta_rad;
  PetscReal d1theta;
  PetscReal d2theta;
  PetscReal d3theta;
  PetscReal d4theta;

}pwparams;

typedef PetscScalar (*paramfun)(PetscReal x, PetscScalar *val, void *data);

PetscErrorCode dispersiveV_2d(MPI_Comm comm, DM da, Vec v0, Vec v1, Vec v2, Vec v3, Vec v4, Vec u, paramfun f, void *params, PetscReal x_offset, PetscReal *z_ref, PetscReal hx, PetscReal hz, int verbose);

PetscScalar planewave_phaseonly(PetscReal x, PetscScalar *val, void *data);

PetscScalar planewave(PetscReal x, PetscScalar *val, void *data);

#endif
