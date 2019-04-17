#ifndef GUARD_output_h
#define GUARD_output_h

#include "petsc.h"
#include "hdf5.h"
#include "type.h"

void writetofile(MPI_Comm comm, char *name, PetscScalar *data, PetscInt n);

void writetofiledouble(MPI_Comm comm, char *name, double *data, PetscInt n);

PetscErrorCode saveVecHDF5(MPI_Comm comm, Vec vec, const char *filename, const char *dsetname);

PetscErrorCode saveVecMfile(MPI_Comm comm, Vec vec, const char *filename, const char *dsetname);

#endif
