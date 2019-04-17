#ifndef GUARD_mat_h
#define GUARD_mat_h

#include <assert.h>
#include "initialize.h"
#include "type.h"
#include "logging.h"
#include "vec.h"
#include "pml.h"
#include "petsc.h"

#define MATRIX_TYPE MATAIJ
#define MATRIX_SYM_TYPE MATSBAIJ
//#define MATRIX_TYPE MATMPIAIJ
//#define MATRIX_SYM_TYPE MATMPISBAIJ
//#define MATRIX_TYPE MATSEQAIJ
//#define MATRIX_SYM_TYPE MATSEQSBAIJ

PetscErrorCode create_doublecurl_op(MPI_Comm comm, Mat *M, Mat *Curl, PetscScalar omega, Vec mu, GridInfo gi, ParDataGrid dg);

#endif
