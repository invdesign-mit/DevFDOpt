#include "initialize.h"

#undef __FUNCT__
#define __FUNCT__ "setGridInfo"
PetscErrorCode setGridInfo(MPI_Comm comm, const char *inputfile_name, GridInfo *gi)
{
  
  PetscErrorCode ierr;
  PetscBool flg;

  hid_t inputfile_id;
  inputfile_id = H5Fopen(inputfile_name, H5F_ACC_RDONLY, H5P_DEFAULT);

  PetscInt  axis, gt;
  PetscReal tmp1d[Naxis];
  PetscReal tmp2d[Naxis][Nsign];
  PetscReal tmp;

  //get 3-element array that defines the boundary conditions on the negative end
  //Note BC+ = BC-
  //Note 0 pec, 1 pmc, 2 bloch 
  ierr = h5get_data(inputfile_id, "/bc", H5T_NATIVE_DOUBLE, tmp1d); CHKERRQ(ierr);
  for (axis = 0; axis < Naxis; ++axis) {
    gi->bc[axis] = (PetscInt) tmp1d[axis];
  }
  PetscPrintf(comm, "From %s, boundary conditions at negative ends: [%d, %d, %d]. NOTE: 0 pec, 1 pmc, 2 bloch.\n", inputfile_name, gi->bc[Xx], gi->bc[Yy], gi->bc[Zz]);

  //read the raw grids and determine dl and N
  char *dset[3]={"/xraw","/yraw","/zraw"};
  hid_t dset_id, dspace_id;
  hsize_t lraw_size;
  PetscReal *lraw[Naxis];
  PetscScalar *ldual[Naxis];
  PetscInt i;
  for(axis=Xx;axis<Naxis;axis++){

    dset_id = H5Dopen(inputfile_id, dset[axis], H5P_DEFAULT);
    dspace_id = H5Dget_space(dset_id);
    H5Sget_simple_extent_dims(dspace_id, &lraw_size, NULL);
    lraw[axis] = (PetscReal *) malloc( lraw_size * sizeof(PetscReal) );
    H5Dread(dset_id, H5T_NATIVE_DOUBLE, H5S_ALL, H5S_ALL, H5P_DEFAULT, lraw[axis]);
    H5Sclose(dspace_id);
    H5Dclose(dset_id);
    gi->N[axis] = lraw_size - 1;

    ldual[axis] = (PetscScalar *) malloc(lraw_size*sizeof(PetscScalar));
    gi->dl[axis][Prim] = (PetscScalar *) malloc(gi->N[axis]*sizeof(PetscScalar));
    gi->dl[axis][Dual] = (PetscScalar *) malloc(gi->N[axis]*sizeof(PetscScalar));
    
    for(i=0;i<gi->N[axis];i++) ldual[axis][i+1]= (PetscScalar) ( (lraw[axis][i]+lraw[axis][i+1])/2 + PETSC_i*0 );

    if(gi->bc[axis]==2)
      ldual[axis][0] =  ldual[axis][gi->N[axis]] - (PetscScalar) (lraw[axis][gi->N[axis]] - lraw[axis][0] + PETSC_i*0);
    else
      ldual[axis][0] = -ldual[axis][1] + (PetscScalar) (2*lraw[axis][0] + PETSC_i*0);
       
    for(i=0;i<gi->N[axis];i++){
      gi->dl[axis][Prim][i] = ldual[axis][i+1] - ldual[axis][i];
      gi->dl[axis][Dual][i] = (PetscScalar) ( lraw[axis][i+1] - lraw[axis][i] + PETSC_i * 0 );
    }

    free(lraw[axis]);
    free(ldual[axis]);

  }
  gi->Ntot = gi->N[Xx] * gi->N[Yy] * gi->N[Zz] * Naxis;
  PetscPrintf(comm, "From %s,[Nx, Ny, Nz] = [%d, %d, %d] with %d unknowns.\n", inputfile_name, gi->N[Xx], gi->N[Yy], gi->N[Zz], gi->Ntot); 
  
  //get 3x2 array Npml[3][2]
  ierr = h5get_data(inputfile_id, "/Npml", H5T_NATIVE_DOUBLE, tmp2d); CHKERRQ(ierr);
  for (axis = 0; axis < Naxis; ++axis) {
    gi->Npml[axis][Neg] = (PetscInt) tmp2d[axis][Neg];
    gi->Npml[axis][Pos] = (PetscInt) tmp2d[axis][Pos];
  }
  PetscPrintf(comm, "From %s, Npml array: \n \t [x-, x+] [%d, %d] \n \t [y-, y+] [%d, %d] \n \t [z-, z+] [%d, %d] \n", inputfile_name, gi->Npml[Xx][Neg], gi->Npml[Xx][Pos], gi->Npml[Yy][Neg], gi->Npml[Yy][Pos], gi->Npml[Zz][Neg], gi->Npml[Zz][Pos]);

  //get 3x2 array that defines bloch factors {e(-i kx Lx), e(-i ky Ly), e(-i kz Lz)}
  PetscReal e_ikL[Naxis][Nri];
  ierr = h5get_data(inputfile_id, "/e_ikL", H5T_NATIVE_DOUBLE, e_ikL); CHKERRQ(ierr);
  ierr = ri2c(e_ikL, gi->exp_neg_ikL, Naxis); CHKERRQ(ierr);
  PetscPrintf(comm, "From %s, exp(-i kx Lx) = %g + 1i*%g\n", inputfile_name, creal(gi->exp_neg_ikL[Xx]), cimag(gi->exp_neg_ikL[Xx]));
  PetscPrintf(comm, "From %s, exp(-i ky Ly) = %g + 1i*%g\n", inputfile_name, creal(gi->exp_neg_ikL[Yy]), cimag(gi->exp_neg_ikL[Yy]));
  PetscPrintf(comm, "From %s, exp(-i kz Lz) = %g + 1i*%g\n", inputfile_name, creal(gi->exp_neg_ikL[Zz]), cimag(gi->exp_neg_ikL[Zz]));

  H5Fclose(inputfile_id);

  PetscFunctionReturn(0);

}

#undef __FUNCT__
#define __FUNCT__ "setDOFInfo"
PetscErrorCode setDOFInfo(MPI_Comm comm, const char *inputfile_name, DOFInfo *dofi)
{

  PetscErrorCode ierr;
  PetscBool flg;

  hid_t inputfile_id;
  inputfile_id = H5Fopen(inputfile_name, H5F_ACC_RDONLY, H5P_DEFAULT);

  int i;
  PetscReal *tmp1d;
  PetscReal tmp;
  ierr = h5get_data(inputfile_id, "/nlayers", H5T_NATIVE_DOUBLE, &tmp); CHKERRQ(ierr);
  dofi->nlayers=(PetscInt) tmp;
  ierr = h5get_data(inputfile_id, "/Mx", H5T_NATIVE_DOUBLE, &tmp); CHKERRQ(ierr);
  dofi->Mx=(PetscInt) tmp;
  ierr = h5get_data(inputfile_id, "/My", H5T_NATIVE_DOUBLE, &tmp); CHKERRQ(ierr);
  dofi->My=(PetscInt) tmp;
  ierr = h5get_data(inputfile_id, "/Nxo", H5T_NATIVE_DOUBLE, &tmp); CHKERRQ(ierr);
  dofi->Nxo=(PetscInt) tmp;
  ierr = h5get_data(inputfile_id, "/Nyo", H5T_NATIVE_DOUBLE, &tmp); CHKERRQ(ierr);
  dofi->Nyo=(PetscInt) tmp;  
  ierr = h5get_data(inputfile_id, "/Mzslab", H5T_NATIVE_DOUBLE, &tmp); CHKERRQ(ierr);
  dofi->Mzslab=(PetscInt) tmp;

  tmp1d=(PetscReal *) malloc(dofi->nlayers*sizeof(PetscReal));
  dofi->Mz=(PetscInt *) malloc(dofi->nlayers*sizeof(PetscInt));
  dofi->Nzo=(PetscInt *) malloc(dofi->nlayers*sizeof(PetscInt));
  ierr = h5get_data(inputfile_id, "/Mz", H5T_NATIVE_DOUBLE, tmp1d); CHKERRQ(ierr);
  for(i=0;i<dofi->nlayers;i++){
    dofi->Mz[i] = (PetscInt) tmp1d[i];
  }
  ierr = h5get_data(inputfile_id, "/Nzo", H5T_NATIVE_DOUBLE, tmp1d); CHKERRQ(ierr);
  for(i=0;i<dofi->nlayers;i++){
    dofi->Nzo[i] = (PetscInt) tmp1d[i];
  }
  dofi->ndof=0;
  for(i=0;i<dofi->nlayers;i++){
    dofi->ndof=dofi->ndof+ dofi->Mx * dofi->My * ((dofi->Mzslab==0) ? dofi->Mz[i] : 1);
  }
  free(tmp1d);
  
  if(dofi->Mzslab) 
    PetscPrintf(comm,"\tSlab Geometry specified with %d layers.\n",dofi->nlayers);
  else
    PetscPrintf(comm,"\tFull DOFs within Mz and %d layers (Non-Slab Geometry).\n",dofi->nlayers);
  PetscPrintf(comm,"From %s, (Nxo,Mx): (%d,%d)\n",inputfile_name,dofi->Nxo,dofi->Mx);
  PetscPrintf(comm,"From %s, (Nyo,My): (%d,%d)\n",inputfile_name,dofi->Nyo,dofi->My);
  char buffer[PETSC_MAX_PATH_LEN], tmpchar[PETSC_MAX_PATH_LEN];
  sprintf(buffer,"From %s, (Nzo,Mz): ",inputfile_name);
  for(i=0;i<dofi->nlayers;i++){
    sprintf(tmpchar,"(%d,%d) ",dofi->Nzo[i],dofi->Mz[i]);
    strcat(buffer,tmpchar);
  }
  PetscPrintf(comm,"%s\n",buffer);
  PetscPrintf(comm,"From %s, the total number of dofs is %d.\n",inputfile_name,dofi->ndof);

  H5Fclose(inputfile_id);

  PetscFunctionReturn(0);

}


#undef __FUNCT__
#define __FUNCT__ "setParDataGrid"
PetscErrorCode setParDataGrid(MPI_Comm comm, ParDataGrid *dg, GridInfo gi)
{

  PetscErrorCode ierr;

  /** Create distributed array (DA) representing Yee's grid, and set it in grid info. */
  const DMBoundaryType ptype = DM_BOUNDARY_PERIODIC;
  const DMDAStencilType stype = DMDA_STENCIL_BOX;
  const PetscInt dof = Naxis;
  const PetscInt swidth = 1;

  ierr = DMDACreate3d(comm,  // MPI communicator
		      ptype, ptype, ptype, stype,  // type of peroodicity and stencil
		      gi.N[Xx], gi.N[Yy], gi.N[Zz],   // global # of grid points in x,y,z
		      PETSC_DECIDE, PETSC_DECIDE, PETSC_DECIDE,  // # of processes in x,y,z
		      dof, swidth,  // degree of freedom, width of stencil
		      PETSC_NULL, PETSC_NULL, PETSC_NULL,  // # of grid points in each process in x,y,z
		      &dg->da); CHKERRQ(ierr);

  ierr = DMDAGetGhostCorners(dg->da, &dg->start_g[Xx], &dg->start_g[Yy], &dg->start_g[Zz], &dg->Nlocal_g[Xx], &dg->Nlocal_g[Yy], &dg->Nlocal_g[Zz]); CHKERRQ(ierr);
  ierr = DMDAGetCorners(dg->da, &dg->start[Xx], &dg->start[Yy], &dg->start[Zz], &dg->Nlocal[Xx], &dg->Nlocal[Yy], &dg->Nlocal[Zz]); CHKERRQ(ierr);
  dg->Nlocal_tot = dg->Nlocal[Xx] * dg->Nlocal[Yy] * dg->Nlocal[Zz] * Naxis;


  /** Create a template vector.  Other vectors are created to have duplicate
      structure of this. */
  ierr = DMCreateGlobalVector(dg->da, &dg->vecTemp); CHKERRQ(ierr);

  /** Get local-to-global mapping from DA. */
  ierr = DMGetLocalToGlobalMapping(dg->da, &dg->map); CHKERRQ(ierr);

  PetscPrintf(comm,"\tDone setting up Parallel Data Grid Context.\n");

  PetscFunctionReturn(0);

}

#undef __FUNCT__
#define __FUNCT__ "setSolverInfo"
PetscErrorCode setSolverInfo(const char *flag_prefix, SolverInfo *si)
{

  PetscErrorCode ierr;
  PetscBool flg;
  PetscInt tmp;
  PetscReal tmpr;

  char tmpflg[PETSC_MAX_PATH_LEN];
  
  strcpy(tmpflg,flag_prefix);
  strcat(tmpflg,"_solverID");
  getint(tmpflg,&tmp,0);
  si->solverID = (KrylovType) tmp;

  strcpy(tmpflg,flag_prefix);
  strcat(tmpflg,"_use_mat_sym");
  getint(tmpflg,&tmp,0);
  si->use_mat_sym = (PetscBool) tmp;

  strcpy(tmpflg,flag_prefix);
  strcat(tmpflg,"_max_iter");
  getint(tmpflg,&tmp,1000000);
  si->max_iter = (PetscInt) tmp;

  strcpy(tmpflg,flag_prefix);
  strcat(tmpflg,"_tol");
  getreal(tmpflg,&tmpr,1e-6);
  si->tol = (PetscReal) tmpr;
  
  strcpy(tmpflg,flag_prefix);
  strcat(tmpflg,"_relres_interval");
  getint(tmpflg,&tmp,1);
  si->relres_interval = (PetscInt) tmp;
  
  PetscFunctionReturn(0);

}
