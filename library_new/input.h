#ifndef GUARD_input_h
#define GUARD_input_h

#include "petsc.h"
#include "hdf5.h"
#include "type.h"

/**
 * h5get_data
 * -----------
 * Retrieve an array data stored under a given data set name in an HDF5 file.
 */
PetscErrorCode h5get_data(hid_t file_id, const char *dataset_name, hid_t mem_type_id, void *buf);

/**
 * ri2c
 * -----------
 * Construct a complex array from an array with alternating real and imaginary values as elements.
 */
PetscErrorCode ri2c(const void *pri, void *pc, const int numelem);

PetscErrorCode getreal(const char *flag, double *var, double autoval);

PetscErrorCode getint(const char *flag, int *var, int autoval);

PetscErrorCode getstr(const char *flag, char *filename, const char default_filename[]);

PetscErrorCode getintarray(const char *flag, int *z, int *nz, int default_val);

PetscErrorCode getrealarray(const char *flag, PetscReal *r, int *nr, PetscReal default_val);

void readfromfile(char *name, PetscScalar *data, PetscInt n);

void readfromfiledouble(char *name, double *data, PetscInt n);

PetscErrorCode loadVecHDF5(MPI_Comm comm, Vec vec, const char *inputfile_name, const char *dataset_name);

#endif
