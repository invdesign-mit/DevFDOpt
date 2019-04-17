#ifndef GUARD_phopt_h
#define GUARD_phopt_h

#include "petsc.h"
#include "initialize.h"
#include "filters.h"

typedef struct{

  int colour;
  MPI_Comm subcomm;
  Mat M0;
  Vec epsDiff;
  Vec epsBkg; 
  PetscScalar omega;
  KSP ksp; 
  int *its;
  int maxit;
  DOFInfo *dofi;
  ParDataGrid dg;
  Vec x0;
  Vec x1;
  Vec x2;
  Vec x3;
  Vec x4;
  Vec b0;
  Vec b1;
  Vec b2;
  Vec b3;
  Vec b4;
  Vec v0;
  Vec v1;
  Vec v2;
  Vec v3;
  Vec v4;
  Vec u;
  PetscReal *obj0;
  PetscReal *obj1;
  PetscReal *obj2;
  PetscReal *obj3;
  PetscReal *obj4;
  PetscScalar *grad0;
  PetscScalar *grad1;
  PetscScalar *grad2;
  PetscScalar *grad3;
  PetscScalar *grad4;
  PetscReal *d0amp_mag;
  PetscReal *d1amp_mag;
  PetscReal *d2amp_mag;
  PetscReal *d3amp_mag;
  PetscReal *d4amp_mag;
  PetscReal *d0amp_phi;
  PetscReal *d1amp_phi;
  PetscReal *d2amp_phi;
  PetscReal *d3amp_phi;
  PetscReal *d4amp_phi;
  PetscInt *solve_order;
  int printdof;
  FiltersToolBox *flt;
  PetscScalar *eps_dof;
  PetscReal *amp_dof;
  PetscReal *mask_order;
  PetscInt magsq;
  PetscReal norm_local;
  PetscReal norm_global;

}Phasefielddata;

void phgrad(void *data);

double phopt_dispsum_amponly(int n, double *amp, double *amp_grad, void *data);

double phopt_dispsum_epscell(int ndofcell, double *epscell, double *epscell_grad, void *data);

double phopt_dispsum_epsglobal_localabs(int ndofAll, double *epsAll, double *epsAll_grad, void *data);

double phopt_dispsum_epsglobal_globalabs(int ndofAll, double *epsAll, double *epsAll_grad, void *data);

#endif

