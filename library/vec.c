#include "vec.h"

#undef __FUNCT__
#define __FUNCT__ "setFieldArray"
PetscErrorCode setFieldArray(Vec field, FunctionSetComponentAt setComponentAt, GridInfo gi, ParDataGrid dg)
{
	PetscFunctionBegin;
	PetscErrorCode ierr;

	Field ***field_array;  // 3D array that images Vec field
	ierr = DMDAVecGetArray(dg.da, field, &field_array); CHKERRQ(ierr);

	/** Get corners and widths of Yee's grid included in this proces. */
	PetscInt ox, oy, oz;  // coordinates of beginning corner of Yee's grid in this process
	PetscInt nx, ny, nz;  // local widths of Yee's grid in this process
	ierr = DMDAGetCorners(dg.da, &ox, &oy, &oz, &nx, &ny, &nz); CHKERRQ(ierr);

	PetscInt ind[Naxis], axis;  // x, y, z indices of grid point
	for (ind[Zz] = oz; ind[Zz] < oz+nz; ++ind[Zz]) {
		for (ind[Yy] = oy; ind[Yy] < oy+ny; ++ind[Yy]) {
			for (ind[Xx] = ox; ind[Xx] < ox+nx; ++ind[Xx]) {
				for (axis = 0; axis < Naxis; ++axis) {
					/** field_array is just the array-representation of the vector field.  So, 
					setting values on field_array is actually setting values on field.*/
					/** setComponentAt() may modify gi internally, but the original gi is intact
					becaues gi has already been copied. */
					ierr = setComponentAt(&field_array[ind[Zz]][ind[Yy]][ind[Xx]].comp[axis], (Axis)axis, ind, &gi); CHKERRQ(ierr);
				}
			}
		}
	}

	ierr = DMDAVecRestoreArray(dg.da, field, &field_array); CHKERRQ(ierr);


	PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "createFieldArray"
PetscErrorCode createFieldArray(Vec *field, FunctionSetComponentAt setComponentAt, GridInfo gi, ParDataGrid dg)
{
	PetscFunctionBegin;
	PetscErrorCode ierr;

	ierr = VecDuplicate(dg.vecTemp, field); CHKERRQ(ierr);
	ierr = setFieldArray(*field, setComponentAt, gi, dg); CHKERRQ(ierr);

	PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "set_mask_prim_at"
/**
 * set_mask_prim_at
 * -------------
 * Mask the primary fields at the negative boundaries according to their boundary conditions.
 */
PetscErrorCode set_mask_prim_at(PetscScalar *mask_prim_value, Axis axis, const PetscInt ind[], GridInfo *gi)
{
	PetscFunctionBegin;

	*mask_prim_value = 1.0;
	Axis u = (Axis)((axis+1) % Naxis);
	if (ind[u]==0) {
		if (gi->bc[u]==PEC) {
			*mask_prim_value = 0.0;
		}
	}

	u = (Axis)((axis+2) % Naxis);
	if (ind[u]==0) {
		if (gi->bc[u]==PEC) {
			*mask_prim_value = 0.0;
		}
	}

	PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "set_mask_dual_at"
/**
 * set_mask_dual_at
 * -------------
 * Mask the dual fields at the negative boundaries according to their boundary conditions.
 */
PetscErrorCode set_mask_dual_at(PetscScalar *mask_dual_value, Axis axis, const PetscInt ind[], GridInfo *gi)
{
	PetscFunctionBegin;

	*mask_dual_value = 1.0;
	if (ind[axis]==0) {
		if (gi->bc[axis]==PEC) {
			*mask_dual_value = 0.0;
		}
	}

	PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "set_double_Fbc_at"
/**
 * set_double_Fbc_at
 * -------------
 * Set an element of the vector that scales fields at boundaries by a factor of 2.
 */
PetscErrorCode set_double_Fbc_at(PetscScalar *double_Fbc_value, Axis axis, const PetscInt ind[], GridInfo *gi)
{
	PetscFunctionBegin;

	*double_Fbc_value = 2.0;

	Axis axis0 = axis;
	Axis axis1 = (Axis)((axis+1) % Naxis);
	Axis axis2 = (Axis)((axis+2) % Naxis);

	PetscInt ind0 = ind[axis0];
	PetscInt ind1 = ind[axis1];
	PetscInt ind2 = ind[axis2];

	BC bc=PMC;

	if (gi->bc[axis1]==bc && ind1==0) {
	  *double_Fbc_value *= 2.0;
	}

	if (gi->bc[axis2]==bc && ind2==0) {
	  *double_Fbc_value *= 2.0;
	}
	

	PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "set_maskInf2One_at"
/**
 * set_maskInf2One_at
 * -------------
 * Create a vector whose Inf elements are replaced by 1.0's
 */
PetscErrorCode set_maskInf2One_at(PetscScalar *value, Axis axis, const PetscInt ind[], GridInfo *gi)
{
	PetscFunctionBegin;
	if (PetscIsInfOrNanScalar(*value)) {
		*value = 1.0;
	}

	PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "maskInf2One"
/**
 * maskInf2One
 * -----------------
 * For a given vector, replace every Inf element to 1.0. 
 * otherwise.
 */
PetscErrorCode maskInf2One(Vec vec, GridInfo gi, ParDataGrid dg)
{
	PetscFunctionBegin;
	PetscErrorCode ierr; 

	ierr = setFieldArray(vec, set_maskInf2One_at, gi, dg); CHKERRQ(ierr);

	PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "set_infMask_at"
/**
 * set_infMask_at
 * -------------
 * Create a vector that masks out infinitely large elements of a vector.
 */
PetscErrorCode set_infMask_at(PetscScalar *value, Axis axis, const PetscInt ind[], GridInfo *gi)
{
	PetscFunctionBegin;
	if (PetscIsInfOrNanScalar(*value)) {
		*value = 0.0;
	} else {
		*value = 1.0;
	}

	PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "infMaskVec"
/**
 * infMaskVec
 * -----------------
 * For a given vector, replace every element of the vector with 0.0 if the element is Inf, and 1.0 
 * otherwise.
 */
PetscErrorCode infMaskVec(Vec vec, GridInfo gi, ParDataGrid dg)
{
	PetscFunctionBegin;
	PetscErrorCode ierr; 

	ierr = setFieldArray(vec, set_infMask_at, gi, dg); CHKERRQ(ierr);

	PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "set_complementMask_at"
/**
 * set_complementMask_at
 * ---------------------
 * Replace an element of the vector with 1.0 if the element is zero, and 0.0 otherwise.
 */
PetscErrorCode set_complementMask_at(PetscScalar *value, Axis axis, const PetscInt ind[], GridInfo *gi)
{
	PetscFunctionBegin;
	*value = (*value==0.0 ? 1.0 : 0.0);
	PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "complementMaskVec"
/**
 * complementMaskVec
 * -----------------
 * For a given vector, replace every element of the vector with 1.0 if the element is zero, and 0.0 
 * otherwise.
 */
PetscErrorCode complementMaskVec(Vec vec, GridInfo gi, ParDataGrid dg)
{
	PetscFunctionBegin;
	PetscErrorCode ierr; 

	ierr = setFieldArray(vec, set_complementMask_at, gi, dg); CHKERRQ(ierr);

	PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "set_index_at"
/**
 * set_index_at
 * -------------
 */
PetscErrorCode set_index_at(PetscScalar *index_value, Axis axis, const PetscInt ind[], GridInfo *gi)
{
	PetscFunctionBegin;

	*index_value = axis + Naxis*ind[Xx] + Naxis*gi->N[Xx]*ind[Yy] + Naxis*gi->N[Xx]*gi->N[Yy]*ind[Zz];

	PetscFunctionReturn(0);
}


#undef __FUNCT__
#define __FUNCT__ "DMDAVecSetValue"
PetscErrorCode DMDAVecSetValue(DM da, Vec v, PetscInt ix0, PetscInt iy0, PetscInt iz0, PetscInt ic0, PetscScalar val)
{
  PetscErrorCode ierr;

  Field ***v_array;  // 3D array that images Vec v
  ierr = DMDAVecGetArray(da, v, &v_array); CHKERRQ(ierr);

  /** Get corners and widths of Yee's grid included in this proces. */
  PetscInt ox, oy, oz;  // coordinates of beginning corner of Yee's grid in this process
  PetscInt nx, ny, nz;  // local widths of Yee's grid in this process
  ierr = DMDAGetCorners(da, &ox, &oy, &oz, &nx, &ny, &nz); CHKERRQ(ierr);

  PetscInt ix,iy,iz,ic;
  PetscReal xprime, dr;
  for (iz = oz; iz < oz+nz; ++iz) {
    for (iy = oy; iy < oy+ny; ++iy) {
      for (ix = ox; ix < ox+nx; ++ix) {
        for (ic = 0; ic < Naxis; ++ic) {
          if(ix==ix0 && iy==iy0 && iz==iz0 && ic==ic0){
	    v_array[iz][iy][ix].comp[ic]=val;
          }
        }
      }
    }
  }

  ierr = DMDAVecRestoreArray(da, v, &v_array); CHKERRQ(ierr);



  return ierr;
}

#undef __FUNCT__
#define __FUNCT__ "DMDAVecGetValue"
PetscErrorCode DMDAVecGetValue(MPI_Comm comm, DM da, Vec v, PetscInt ix0, PetscInt iy0, PetscInt iz0, PetscInt ic0, PetscScalar *val)
{
  PetscErrorCode ierr;
  PetscScalar local_val=0;

  Field ***v_array;  // 3D array that images Vec v
  ierr = DMDAVecGetArray(da, v, &v_array); CHKERRQ(ierr);

  /** Get corners and widths of Yee's grid included in this proces. */
  PetscInt ox, oy, oz;  // coordinates of beginning corner of Yee's grid in this process
  PetscInt nx, ny, nz;  // local widths of Yee's grid in this process
  ierr = DMDAGetCorners(da, &ox, &oy, &oz, &nx, &ny, &nz); CHKERRQ(ierr);

  PetscInt ix,iy,iz,ic;
  PetscReal xprime, dr;
  for (iz = oz; iz < oz+nz; ++iz) {
    for (iy = oy; iy < oy+ny; ++iy) {
      for (ix = ox; ix < ox+nx; ++ix) {
        for (ic = 0; ic < Naxis; ++ic) {
          if(ix==ix0 && iy==iy0 && iz==iz0 && ic==ic0){
	    local_val=v_array[iz][iy][ix].comp[ic];
          }
        }
      }
    }
  }

  ierr = DMDAVecRestoreArray(da, v, &v_array); CHKERRQ(ierr);

  MPI_Allreduce(&local_val,val,1,MPIU_SCALAR,MPI_SUM,comm);

  return ierr;
}
