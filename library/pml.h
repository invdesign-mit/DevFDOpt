#ifndef GUARD_pml_h
#define GUARD_pml_h

#include "petsc.h"
#include "initialize.h"

#define m 4
#define R cexp(-16)
#define kappa_max 1.0
#define amax 0
#define ma 4
#define lprim_0 0

void stretch_dl(PetscScalar *dl_stretched[Naxis][Ngt], PetscScalar omega, GridInfo gi);

void generate_s_factor(PetscScalar omega, PetscScalar *s_prim, PetscScalar *s_dual, PetscScalar *dl_prim, PetscScalar *dl_dual, int N, int Npml_n, int Npml_p);

#endif
