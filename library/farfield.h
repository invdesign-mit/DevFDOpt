#ifndef GUARD_farfield_h
#define GUARD_farfield_h

#include "petsc.h"
#include "type.h"
#include "initialize.h"

PetscErrorCode ff2dxz(MPI_Comm comm, DM da, Vec v, PetscReal x, PetscReal z, PetscReal freq, PetscReal xp_offset, PetscReal *zp_ref, PetscReal hx, PetscReal hz, int verbose);

PetscErrorCode ff2dxz_asymp(MPI_Comm comm, DM da, Vec v, PetscReal r, PetscReal theta_far_rad, PetscReal freq, PetscReal xp_offset, PetscReal *zp_ref, PetscReal hx, PetscReal hz, int verbose);

PetscErrorCode pwsrc_2dxz(MPI_Comm comm, DM da, Vec v, PetscReal freq, PetscReal refractive_index, PetscReal theta, PetscScalar amp, PetscReal x_offset, PetscReal *z_ref, PetscReal hx, PetscReal hz, int verbose);

#endif
