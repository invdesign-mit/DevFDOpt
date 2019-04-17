#ifndef GUARD_lasing_h
#define GUARD_lasing_h

#include "petsc.h"
#include "initialize.h"
#include "filters.h"
#include "logging.h"
#include "dof2domdirect.h"
#include "solver.h"
#include "output.h"

typedef struct{

  Mat M0;
  Vec x;
  Vec epsDiff;
  Vec epsBkg;
  PetscScalar omega;
  KSP ksp;
  int *its;
  int maxit;
  GridInfo *gi;
  DOFInfo *dofi;
  ParDataGrid dg;
  int printdof;
  FiltersToolBox *flt;

  PetscReal jmax;
  PetscReal jmin;
  
}Lasingdata;

PetscErrorCode array2vec(PetscScalar *pt, Vec v, GridInfo *gi, DM da);

PetscErrorCode vec2array(MPI_Comm comm, Vec v, PetscScalar *pt, GridInfo *gi, DM da);

double lasing(int ndofAll, double *dofAll, double *dofgradAll, void *data);

double lasing2(int ndofAll, double *dofAll, double *dofgradAll, void *data);

#endif

