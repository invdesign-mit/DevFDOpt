#ifndef GUARD_ffopt_h
#define GUARD_ffopt_h

#include "petsc.h"
#include "initialize.h"
#include "filters.h"
#include "logging.h"

typedef struct{

  int colour;
  MPI_Comm subcomm;
  Mat M0;
  Vec b;
  Vec x;
  Vec epsDiff;
  Vec epsBkg;
  PetscScalar omega;
  Vec ffvec;
  KSP ksp;
  int *its;
  int maxit;
  DOFInfo *dofi;
  ParDataGrid dg;
  int printdof;
  FiltersToolBox *flt;

}Farfielddata;

PetscScalar ffgrad(MPI_Comm comm, PetscScalar *dof, PetscScalar *dofgrad, const Mat M0, const Vec b, Vec x, const Vec epsDiff, const Vec epsBkg, const PetscScalar omega, const Vec ffvec, KSP ksp, int *its, int maxit, DOFInfo *dofi, DM da);

double ffopt(int ndofAll, double *dofAll, double *dofgradAll, void *data);

double ffopt_maximinconstraint(int ndofAll_with_dummy, double *dofAll_with_dummy, double *dofgradAll_with_dummy, void *data);

double dummy_obj(int ndofAll_with_dummy, double *dofAll_with_dummy, double *dofgradAll_with_dummy, void *data);

#endif

