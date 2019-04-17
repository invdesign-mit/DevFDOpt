#ifndef GUARD_vec_h
#define GUARD_vec_h

#include "initialize.h"
#include "type.h"

#include "petsc.h"

typedef PetscErrorCode (*FunctionSetComponentAt)(PetscScalar *component, Axis axis, const PetscInt ind[], GridInfo *gi);

PetscErrorCode createFieldArray(Vec *field, FunctionSetComponentAt setComponentAt, GridInfo gi, ParDataGrid dg);

PetscErrorCode set_mask_prim_at(PetscScalar *mask_prim_value, Axis axis, const PetscInt ind[], GridInfo *gi);

PetscErrorCode set_mask_dual_at(PetscScalar *mask_dual_value, Axis axis, const PetscInt ind[], GridInfo *gi);

PetscErrorCode set_double_Fbc_at(PetscScalar *double_Fbc_value, Axis axis, const PetscInt ind[], GridInfo *gi);

PetscErrorCode set_index_at(PetscScalar *index_value, Axis axis, const PetscInt ind[], GridInfo *gi);

PetscErrorCode maskInf2One(Vec vec, GridInfo gi, ParDataGrid dg);

PetscErrorCode infMaskVec(Vec vec, GridInfo gi, ParDataGrid dg);
	 
PetscErrorCode complementMaskVec(Vec vec, GridInfo gi, ParDataGrid dg);

PetscErrorCode DMDAVecSetValue(DM da, Vec v, PetscInt ix0, PetscInt iy0, PetscInt iz0, PetscInt ic0, PetscScalar val);

PetscErrorCode DMDAVecGetValue(MPI_Comm comm, DM da, Vec v, PetscInt ix0, PetscInt iy0, PetscInt iz0, PetscInt ic0, PetscScalar *val);

#endif
