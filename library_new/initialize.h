#ifndef GUARD_initialize_h
#define GUARD_initialize_h

#include "petsc.h"
#include "hdf5.h"
#include "type.h"
#include "input.h"

typedef struct {
  PetscScalar comp[Naxis];  // x,y,z component of field
} Field;  // used for DAVecGetArray()

typedef struct {

  PetscInt N[Naxis];  // # of grid points in x, y, z
  PetscInt Ntot;  // total # of unknowns
  BC bc[Naxis]; // boundary conditions at -x, -y, -z ends; for simplicity, we set BC+ == BC-
  PetscScalar *dl[Naxis][Ngt];  // dx, dy, dz at primary and dual grid locations
  PetscInt Npml[Naxis][Nsign]; //number of pml layers along x, y, z at + and - ends.
  PetscScalar exp_neg_ikL[Naxis];  // exp(-ik Lx), exp(-ik Ly), exp(-ik Lz)

} GridInfo;

typedef struct{

  DM da;  // distributed array
  PetscInt Nlocal_tot;  // total # of local unknowns
  PetscInt Nlocal[Naxis];  // # of local grid points in x, y, z
  PetscInt start[Naxis]; // local starting points in x, y, z
  PetscInt Nlocal_g[Naxis];  // # of local grid points in x, y, z including ghost points
  PetscInt start_g[Naxis]; // local starting points in x, y, z including ghost points
  Vec vecTemp; // template vector.  Also used as a temporary storage of a vector
  ISLocalToGlobalMapping map;  // local-to-global index mapping

} ParDataGrid;

typedef struct{
  
  PetscInt nlayers;
  PetscInt Mx;
  PetscInt My;
  PetscInt *Mz;
  PetscInt Mzslab;
  PetscInt Nxo;
  PetscInt Nyo;
  PetscInt *Nzo;
  PetscInt ndof;

} DOFInfo;

typedef struct{

  KrylovType solverID; // 0 BiCG, 1 QMR 
  PetscBool use_mat_sym; //check if matrix is symmetric and, if true, use a symalg
  PetscInt max_iter;  // maximum number of iteration of BiCG
  PetscReal tol;  // tolerance of BiCG
  PetscInt relres_interval;  // number of BiCG iterations between snapshots of approximate solutions

} SolverInfo;

PetscErrorCode setGridInfo(MPI_Comm comm, const char *inputfile_name, GridInfo *gi);

PetscErrorCode setDOFInfo(MPI_Comm comm, const char *inputfile_name, DOFInfo *dofi);

PetscErrorCode setParDataGrid(MPI_Comm comm, ParDataGrid *dg, GridInfo gi);

PetscErrorCode setSolverInfo(const char *flag_prefix, SolverInfo *si);

#endif
