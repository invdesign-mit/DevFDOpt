#include "mat.h"

	/** Below, 13 is the maximum number of nonzero elements in a diagonal portion of 
	  a local submatrix. e.g., if G=H and F=E, CHE is 9-by-9 and distributed among 3 processors, 
	  entire row (0,1,2) compose a submatrix in processsor 0, and row (3,4,5) in 
	  processor 1, row (6,7,8) in processor 2.  In processor 1, the diagonal portion 
	  means 3-by-3 square matrix at the diagonal, which is composed of row (3,4,5) and 
	  column (3,4,5).  Each row of CHE corresponds to one of Ex, Ey, Ez component of 
	  some cell in the grid.

	  When CHE is multiplied to a vector x, which has 3*Nx*Ny*Nz E field components, a 
	  row of CHE generates an output E field component out of 13 input E field 
	  components; each output E field component is involved in 4 curl loops, in each of 
	  which 3 extra E field components are introduced.  Therefore, an output E field 
	  component is the result of interactions between 1(itself) + 4(# of curl loops) * 
	  3(# of extra E field components in each loop) = 13 input E field components.

	  If the cell containing the output E field component is in interior of a local 
	  portion of the Yee's grid, then all 4 curl loops lie inside the local grid.  
	  Therefore, at most 13 E field components can be in the diagonal portion of a local 
	  submatrix.

	  On the other hand, if the cell is at a boundary of a local grid, then some of 4 
	  curl loops lie outside the local grid.  As an extreme case, if the local grid is
	  composed of only one cell, then only 3 E field components are in the local grid, 
	  and therefore 10 E field components are in the off-diagonal portion of the 
	  submatrix. */

#undef __FUNCT__
#define __FUNCT__ "setDp"
/**
 * setDp
 * ------------
 * For a matrix row indexed by (w, i, j, k), set the forward (s==Pos) or backward (s==Neg)
 * difference of the p-component of the field in the v-direction, with extra scale multiplied.
 */
PetscErrorCode setDp(Mat A, Sign s, Axis w, PetscInt i, PetscInt j, PetscInt k, Axis p, Axis v, PetscScalar scale, PetscScalar *dl_stretched[Naxis][Ngt], GridInfo gi)
{
	PetscFunctionBegin;
	PetscErrorCode ierr;

	/** Set dv. */
	PetscInt coord[] = {i, j, k};
	PetscInt ind = coord[v];
	PetscScalar dv = dl_stretched[v][s][ind];  // for forward (s==Pos) difference, use dl at dual grid locations

	/** Set the row and column indices of the matrix elements to set. */
	MatStencil indGw;  // grid point indices of Gw (mapped to row index of A)
	MatStencil indFp[2];  // current and next grid point indices of Fp (mapped to column indices of A).  The next grid point can be either in +d or -d direction

	indGw.c = w;
	indGw.i = i;
	indGw.j = j;
	indGw.k = k;

	indFp[0].c = p;
	indFp[0].i = coord[Xx];
	indFp[0].j = coord[Yy];
	indFp[0].k = coord[Zz];

	if (s == Pos) {
		++coord[v];
	} else {
		assert(s == Neg);
		--coord[v];
	}
	indFp[1].c = p;
	indFp[1].i = coord[Xx];
	indFp[1].j = coord[Yy];
	indFp[1].k = coord[Zz];

	/** Set Nv. */
	PetscInt Nv = gi.N[v];

	PetscScalar dFp[2];
	PetscInt num_dFp = 2;

	/** Two matrix elements in a single row to be set at once. */
	if (s == Pos) {
		dFp[0] = -scale/dv; dFp[1] = scale/dv;  // forward difference
	} else {
		dFp[0] = scale/dv; dFp[1] = -scale/dv;  // backward difference
	}

	/** Handle boundary conditions. */
	if (s == Pos) {  // forward difference
		if (ind == Nv-1) {
			if (gi.bc[v] == Bloch) {
				dFp[1] *= gi.exp_neg_ikL[v];
			} else {
				dFp[1] = 0.0;
			}
		}
	} else {  // backward difference
		assert(s == Neg);
		if (ind == 0) {
			if (gi.bc[v] == PMC) {
				dFp[0] *= 2.0;
			}

			if (gi.bc[v] == Bloch) {
				dFp[1] /= gi.exp_neg_ikL[v];
			} else {
				dFp[1] = 0.0;
			}
		}
	}

	/** Below, ADD_VALUES is used instead of INSERT_VALUES to deal with cases of 
	  gi.bc[Pp]==gi.bc[Pp]==Bloch and Nv==1.  In such a case, v==0 and v==Nv-1 coincide, 
	  so inserting dFp[1] after dFp[0] overwrites dFp[0], which is not what we want.  
	  On the other hand, if we add dFp[1] to dFp[0], it is equivalent to insert 
	  -scale/dv + scale/dv = 0.0, and this is what should be done. */
	ierr = MatSetValuesStencil(A, 1, &indGw, num_dFp, indFp, dFp, ADD_VALUES); CHKERRQ(ierr);  // Gw <-- scale * (d/dv)Fp

	PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "setCF"
/**
 * setCF
 * -----
 * Set up the curl(F) operator matrix CF for given F == E or H.
 */
PetscErrorCode setCF(Mat CF, GridType gtype, PetscScalar *dl_stretched[Naxis][Ngt], GridInfo gi, ParDataGrid dg)
{
	PetscFunctionBegin;
	PetscErrorCode ierr;
	
	Sign s = (Sign)((gtype+1) % Ngt);  // Pos for gtype==Prim, Neg for gtype==Dual

	/** Get corners and widths of Yee's grid included in this proces. */
	PetscInt ox, oy, oz;  // coordinates of beginning corner of Yee's grid in this process
	PetscInt nx, ny, nz;  // local widths of Yee's grid in this process
	ierr = DMDAGetCorners(dg.da, &ox, &oy, &oz, &nx, &ny, &nz); CHKERRQ(ierr);

	PetscInt i, j, k, axis;  // x, y, z indices of grid point
	for (k = oz; k < oz+nz; ++k) {
		for (j = oy; j < oy+ny; ++j) {
			for (i = ox; i < ox+nx; ++i) {
				for (axis = 0; axis < Naxis; ++axis) {  // direction of curl
					Axis n = (Axis) axis;
					Axis h = (Axis)((axis+1) % Naxis);  // horizontal axis
					Axis v = (Axis)((axis+2) % Naxis);  // vertical axis

					ierr = setDp(CF, s, n, i, j, k, v, h, 1.0, dl_stretched, gi); CHKERRQ(ierr);
					ierr = setDp(CF, s, n, i, j, k, h, v, -1.0, dl_stretched, gi); CHKERRQ(ierr);
				}
			}
		}
	}

	PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "createCE"
/**
 * createCE
 * --------
 * Create the matrix CE, the curl operator on E fields.
 */
PetscErrorCode createCE(MPI_Comm comm, Mat *CE, PetscScalar *dl_stretched[Naxis][Ngt], GridInfo gi, ParDataGrid dg)
{
	PetscFunctionBegin;
	PetscErrorCode ierr;

	Vec maskE, maskH;
	ierr = createFieldArray(&maskE, set_mask_prim_at, gi, dg);
	ierr = createFieldArray(&maskH, set_mask_dual_at, gi, dg);
	
	ierr = MatCreate(comm, CE); CHKERRQ(ierr);
	ierr = MatSetSizes(*CE, dg.Nlocal_tot, dg.Nlocal_tot, PETSC_DETERMINE, PETSC_DETERMINE); CHKERRQ(ierr);
	ierr = MatSetType(*CE, MATRIX_TYPE); CHKERRQ(ierr);
	ierr = MatSetFromOptions(*CE);
	ierr = MatMPIAIJSetPreallocation(*CE, 4, PETSC_NULL, 2, PETSC_NULL); CHKERRQ(ierr);
	ierr = MatSeqAIJSetPreallocation(*CE, 4, PETSC_NULL); CHKERRQ(ierr);
	ierr = MatSetLocalToGlobalMapping(*CE, dg.map, dg.map); CHKERRQ(ierr);
	ierr = MatSetStencil(*CE, Naxis, dg.Nlocal_g, dg.start_g, Naxis); CHKERRQ(ierr);
	ierr = setCF(*CE, Prim, dl_stretched, gi, dg); CHKERRQ(ierr);
	ierr = MatAssemblyBegin(*CE, MAT_FINAL_ASSEMBLY); CHKERRQ(ierr);
	ierr = MatAssemblyEnd(*CE, MAT_FINAL_ASSEMBLY); CHKERRQ(ierr);

	ierr = MatDiagonalScale(*CE, maskH, maskE); CHKERRQ(ierr);
	
	VecDestroy(&maskE);
	VecDestroy(&maskH);

	PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "createCH"
/**
 * createCH
 * --------
 * Create the matrix CH, the curl operator on H fields.
 */
PetscErrorCode createCH(MPI_Comm comm, Mat *CH, PetscScalar *dl_stretched[Naxis][Ngt], GridInfo gi, ParDataGrid dg)
{
	PetscFunctionBegin;
	PetscErrorCode ierr;

	Vec maskE, maskH;
	ierr = createFieldArray(&maskE, set_mask_prim_at, gi, dg);
	ierr = createFieldArray(&maskH, set_mask_dual_at, gi, dg);
	
	ierr = MatCreate(comm, CH); CHKERRQ(ierr);
	ierr = MatSetSizes(*CH, dg.Nlocal_tot, dg.Nlocal_tot, PETSC_DETERMINE, PETSC_DETERMINE); CHKERRQ(ierr);
	ierr = MatSetType(*CH, MATRIX_TYPE); CHKERRQ(ierr);
	ierr = MatSetFromOptions(*CH);
	ierr = MatMPIAIJSetPreallocation(*CH, 4, PETSC_NULL, 2, PETSC_NULL); CHKERRQ(ierr);
	ierr = MatSeqAIJSetPreallocation(*CH, 4, PETSC_NULL); CHKERRQ(ierr);
	ierr = MatSetLocalToGlobalMapping(*CH, dg.map, dg.map); CHKERRQ(ierr);
	ierr = MatSetStencil(*CH, Naxis, dg.Nlocal_g, dg.start_g, Naxis); CHKERRQ(ierr);
	ierr = setCF(*CH, Dual, dl_stretched, gi, dg); CHKERRQ(ierr);
	ierr = MatAssemblyBegin(*CH, MAT_FINAL_ASSEMBLY); CHKERRQ(ierr);
	ierr = MatAssemblyEnd(*CH, MAT_FINAL_ASSEMBLY); CHKERRQ(ierr);

	ierr = MatDiagonalScale(*CH, maskE, maskH); CHKERRQ(ierr);

	VecDestroy(&maskE);
	VecDestroy(&maskH);

	PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "create_doublecurl_op"
PetscErrorCode create_doublecurl_op(MPI_Comm comm, Mat *M, Mat *Curl, PetscScalar omega, Vec mu, GridInfo gi, ParDataGrid dg)
{
	PetscFunctionBegin;
	PetscErrorCode ierr;

	TimeStamp ts_m;
	initTimeStamp(&ts_m);

	Vec param, paramMask; 
   	Vec muinv;  // store various inverse vectors
	Mat CH, CE;  // curl operators on E and H
	Mat CHE;

	PetscPrintf(comm,"\tStart creating Curl(mu^-1 Curl) matrix with PML\n");

	/** Stretch coords. */
	PetscScalar *dl_stretched[Naxis][Ngt];
	Axis axis;
	GridType gt;
	for(axis=Xx;axis<Naxis;axis++){
	  for(gt=Prim;gt<Ngt;gt++){
	    dl_stretched[axis][gt]=(PetscScalar *) malloc(gi.N[axis]*sizeof(PetscScalar));
	  }
	}
	stretch_dl(dl_stretched,omega,gi);
		
	/** Set up the matrix CH, the curl operator on H fields. */
	ierr = createCH(comm, &CH, dl_stretched, gi, dg); CHKERRQ(ierr);
	ierr = updateTimeStamp(comm, &ts_m, "creating CH matrix"); CHKERRQ(ierr);

	/** Set up the matrix CE, the curl operator on E fields. */
	ierr = createCE(comm, &CE, dl_stretched, gi, dg); CHKERRQ(ierr);
	*Curl = CE;
	ierr = updateTimeStamp(comm, &ts_m, "creating CE matrix"); CHKERRQ(ierr);

	/** Multiply with muinv */
	ierr = VecDuplicate(dg.vecTemp, &muinv); CHKERRQ(ierr);
	ierr = VecSet(muinv, 1.0); CHKERRQ(ierr);
	ierr = VecPointwiseDivide(muinv, muinv, mu); CHKERRQ(ierr);
	ierr = MatDiagonalScale(CH, PETSC_NULL, muinv); CHKERRQ(ierr);

	MatMatMult(CH, CE, MAT_INITIAL_MATRIX, 13.0/(4.0+4.0), &CHE); CHKERRQ(ierr); // CHE = CH*(invMu)*CE
	*M = CHE;
	ierr = updateTimeStamp(comm, &ts_m, "creating CHE matrix"); CHKERRQ(ierr);

	MatDestroy(&CH);
	VecDestroy(&muinv); 
	for(axis=Xx;axis<Naxis;axis++){
	  for(gt=Prim;gt<Ngt;gt++){
	    free(dl_stretched[axis][gt]);
	  }
	}

	PetscPrintf(comm,"\tEnd creating Curl(mu^-1 Curl) matrix with PML\n");

	PetscFunctionReturn(0);
}
