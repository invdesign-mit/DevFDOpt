#ifndef GUARD_optfuncs_h
#define GUARD_optfuncs_h

#include "petsc.h"
#include "initialize.h"
#include "filters.h"

typedef double (*optfunc)(int DegFree,double *epsopt, double *grad, void *data);

typedef struct{

  int colour;
  int ncells_per_comm;
  MPI_Comm subcomm;
  int ndof_eps;
  int ID;
  Mat M0;
  Vec *b;
  Vec *x;
  Vec epsDiff;
  Vec epsBkg;
  PetscScalar omega;
  Vec *v;
  Vec u;
  PetscReal *ampr;
  PetscReal *ampphi;
  optfunc func;
  KSP *ksp;
  int *its;
  int maxit;
  DOFInfo *dofi;
  ParDataGrid dg;
  int printdof;
  FiltersToolBox *flt;

}DotObjData;

PetscScalar ffgrad(MPI_Comm comm, PetscScalar *dof, PetscScalar *dofgrad, const Mat M0, const Vec b, Vec x, const Vec epsDiff, const Vec epsBkg, const PetscScalar omega, const Vec ffvec, KSP ksp, int *its, int maxit, DOFInfo *dofi, DM da);

PetscScalar phgrad(MPI_Comm comm, PetscScalar *dof, PetscReal ampphi, PetscScalar *dofgrad, PetscScalar *grad_ampphi, const Mat M0, const Vec b, Vec x, const Vec epsDiff, const Vec epsBkg, const PetscScalar omega, const Vec v, Vec u, KSP ksp, int *its, int maxit, DOFInfo *dofi, DM da);

PetscScalar pherrgrad(MPI_Comm comm, PetscScalar *dof, PetscReal ampr, PetscReal ampphi, PetscScalar *dofgrad, PetscScalar *grad_ampr, PetscScalar *grad_ampphi, const Mat M0, const Vec b, Vec x, const Vec epsDiff, const Vec epsBkg, const PetscScalar omega, const Vec v, Vec u, KSP ksp, int *its, int maxit, DOFInfo *dofi, DM da);

double maximinconstraint(int ndofAll_with_dummy, double *dofAll_with_dummy, double *dofgradAll_with_dummy, void *data);

double minimaxconstraint(int ndofAll_with_dummy, double *dofAll_with_dummy, double *dofgradAll_with_dummy, void *data);

double dummy_obj(int ndofAll_with_dummy, double *dofAll_with_dummy, double *dofgradAll_with_dummy, void *data);

double ffopt(int ndofAll, double *dofAll, double *dofgradAll, void *data);

double phopt(int ndofAll, double *dofAll, double *dofgradAll, void *data);

double pherropt(int ndofAll, double *dofAll, double *dofgradAll, void *data);

double phoptmulti(int ndofAll, double *dofAll, double *dofgradAll, void *data);

double pherroptmulti(int ndofAll, double *dofAll, double *dofgradAll, void *data);

#endif

