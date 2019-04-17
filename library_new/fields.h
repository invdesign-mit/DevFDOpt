#ifndef GUARD_fields_h
#define GUARD_fields_h

#include "petsc.h"
#include "type.h"
#include "initialize.h"

typedef struct{

  PetscReal freq;
  PetscReal nmed;
  PetscReal theta_rad;
  PetscReal phi_rad;

}pwparams;

typedef struct{

  PetscReal freq;
  PetscReal nmed;
  PetscReal x_far;
  PetscReal y_far;
  PetscReal z_far;
  PetscReal z_local;

}ffparams;

typedef PetscScalar (*paramfun)(PetscReal x, PetscReal y, PetscScalar *val, void *data);

PetscErrorCode vec_at_xyslab_linpol(MPI_Comm comm, DM da, Vec v, Vec u, paramfun f, void *params, PetscInt pol, PetscReal x_offset, PetscReal y_offset, PetscReal *z_ref, PetscReal hx, PetscReal hy, PetscReal hz, int verbose);

PetscScalar planewave(PetscReal x, PetscReal y, PetscScalar *val, void *data);

PetscScalar farfield2d(PetscReal x_local, PetscReal y_local, PetscScalar *val, void *data);

PetscScalar farfield3d(PetscReal x_local, PetscReal y_local, PetscScalar *val, void *data);

PetscScalar nearfieldlens(PetscReal x_local, PetscReal y_local, PetscScalar *val, void *data);

#endif
