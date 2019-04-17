#include "lasing.h"

extern int count;
extern TimeStamp global_time;

#undef __FUNCT__
#define __FUNCT__ "array2vec"
PetscErrorCode array2vec(PetscScalar *pt, Vec v, GridInfo *gi, DM da)
{

  PetscInt Nx = gi->N[Xx];
  PetscInt Ny = gi->N[Yy];
  //PetscInt Nz = gi->N[Zz];

  PetscErrorCode ierr;

  Field ***v_array;  // 3D array that images Vec v
  ierr = DMDAVecGetArray(da, v, &v_array); CHKERRQ(ierr);

  // Get corners and widths of Yee's grid included in this proces. 
  PetscInt ox, oy, oz;  // coordinates of beginning corner of Yee's grid in this process
  PetscInt nx, ny, nz;  // local widths of Yee's grid in this process
  ierr = DMDAGetCorners(da, &ox, &oy, &oz, &nx, &ny, &nz); CHKERRQ(ierr);

  int ip;
  PetscInt ix,iy,iz,ic;
  for (iz = oz; iz < oz+nz; ++iz) {
    for (iy = oy; iy < oy+ny; ++iy) {
      for (ix = ox; ix < ox+nx; ++ix) {
	for (ic = 0; ic < Naxis; ++ic) {
	  ip = ic + Naxis * ix + Naxis * Nx * iy + Naxis * Nx * Ny * iz;
	  v_array[iz][iy][ix].comp[ic]=pt[ip];
	}
      }
    }
  }

  ierr = DMDAVecRestoreArray(da, v, &v_array); CHKERRQ(ierr);

  return ierr;
}

#undef __FUNCT__
#define __FUNCT__ "vec2array"
PetscErrorCode vec2array(MPI_Comm comm, Vec v, PetscScalar *pt, GridInfo *gi, DM da)
{

  PetscInt Nx = gi->N[Xx];
  PetscInt Ny = gi->N[Yy];
  PetscInt Nz = gi->N[Zz];
  
  PetscErrorCode ierr;

  PetscInt i,ip;
  PetscScalar *local_pt;
  PetscInt Ntot=Naxis*Nx*Ny*Nz;
  local_pt = (PetscScalar *) malloc(Ntot*sizeof(PetscScalar));
  for(i=0;i<Ntot;i++) local_pt[i]=0.0+PETSC_i*0.0;
  MPI_Barrier(comm);

  Field ***v_array;  // 3D array that images Vec v
  ierr = DMDAVecGetArray(da, v, &v_array); CHKERRQ(ierr);

  // Get corners and widths of Yee's grid included in this proces. 
  PetscInt ox, oy, oz;  // coordinates of beginning corner of Yee's grid in this process
  PetscInt nx, ny, nz;  // local widths of Yee's grid in this process
  ierr = DMDAGetCorners(da, &ox, &oy, &oz, &nx, &ny, &nz); CHKERRQ(ierr);

  PetscInt ix,iy,iz,ic;
  for (iz = oz; iz < oz+nz; ++iz) {
    for (iy = oy; iy < oy+ny; ++iy) {
      for (ix = ox; ix < ox+nx; ++ix) {
	for (ic = 0; ic < Naxis; ++ic) {
	  ip = ic + Naxis * ix + Naxis * Nx * iy + Naxis * Nx * Ny * iz;
	  local_pt[ip] = local_pt[ip]+v_array[iz][iy][ix].comp[ic];
	}
      }
    }
  }

  ierr = DMDAVecRestoreArray(da, v, &v_array); CHKERRQ(ierr);
  MPI_Barrier(comm);

  MPI_Allreduce(local_pt,pt,Ntot,MPIU_SCALAR,MPI_SUM,comm);
  MPI_Barrier(comm);

  free(local_pt);

  return ierr;
}

#undef __FUNCT__
#define __FUNCT__ "lasing"
double lasing(int ndofAll, double *dofAll, double *dofgradAll, void *data)
{

  Lasingdata *ptdata = (Lasingdata *) data;

  Mat M0 = ptdata->M0;
  Vec x = ptdata->x;
  Vec epsDiff = ptdata->epsDiff;
  Vec epsBkg = ptdata->epsBkg;
  PetscScalar omega = ptdata->omega;
  KSP ksp = ptdata->ksp;
  int *its = ptdata->its;
  int maxit = ptdata->maxit;
  GridInfo *gi = ptdata->gi;
  DOFInfo *dofi = ptdata->dofi;
  ParDataGrid dg = ptdata->dg;
  int printdof = ptdata->printdof;

  PetscReal jmax = ptdata->jmax;
  PetscReal jmin = ptdata->jmin;
  
  char tmpstr[PETSC_MAX_PATH_LEN];
  sprintf(tmpstr,"starting computation step %d (global timecheck)",count);
  updateTimeStamp(PETSC_COMM_WORLD,&global_time,tmpstr);

  int i;
  int ndof=dofi->ndof; //only the epsilon degrees of freedom
  PetscScalar *dof,*dofgrad; //complex values of epsilon degrees of freedom just for filtering purposes
  FiltersToolBox *flt = ptdata->flt;
  PetscScalar *_u,*_ugrad;
  dof = (PetscScalar *) malloc(ndof*sizeof(PetscScalar));
  dofgrad = (PetscScalar *) malloc(ndof*sizeof(PetscScalar));
  _u = (PetscScalar *) malloc(ndof*sizeof(PetscScalar));
  _ugrad = (PetscScalar *) malloc(ndof*sizeof(PetscScalar));
  for(i=0;i<ndof;i++){
    dof[i]=dofAll[i]+PETSC_i*0.0;
  }
  filters_apply(PETSC_COMM_WORLD,dof,_u,flt,1);

  int ndof_j=ndofAll-ndof; //the degrees of freedom pertaining to the source distribution
  PetscScalar *j = (PetscScalar *) malloc(ndof_j/2*sizeof(PetscScalar));
  PetscScalar *jgrad = (PetscScalar *) malloc(ndof_j/2*sizeof(PetscScalar));
  for(i=0;i<ndof_j/2;i++)
    j[i]= (jmin + (jmax-jmin) * dofAll[ndof+i]) * cexp(PETSC_i * 2.0 * M_PI * dofAll[ndof+i+ndof_j/2]);

  Vec J,b,Jconj,eps,negW2eps,xconj,u,gradeps,gradJ;
  Mat M;
  VecDuplicate(x,&J);
  VecDuplicate(x,&b);
  VecDuplicate(x,&Jconj);
  VecDuplicate(x,&eps);
  VecDuplicate(x,&negW2eps);
  VecDuplicate(x,&xconj);
  VecDuplicate(x,&u);
  VecDuplicate(x,&gradeps);
  VecDuplicate(x,&gradJ);
  MatDuplicate(M0,MAT_COPY_VALUES,&M);
  
  array2vec(j,J,gi,dg.da);
  VecCopy(J,b);
  VecCopy(J,Jconj);
  VecScale(b,-PETSC_i*omega);
  VecConjugate(Jconj);

  multilayer_forward(_u,eps,dofi,dg.da);
  VecPointwiseMult(eps,eps,epsDiff);
  VecAXPY(eps,1.0,epsBkg);
  VecCopy(eps,negW2eps);
  VecScale(negW2eps,-omega*omega);
  MatDiagonalSet(M,negW2eps,ADD_VALUES);

  SolveMatrixDirect(PETSC_COMM_WORLD,ksp,M,b,x,its,maxit);
  VecCopy(x,xconj);
  VecConjugate(xconj);
  PetscScalar Estar_dot_J, Jstar_dot_J;
  VecTDot(xconj,J,&Estar_dot_J);
  VecTDot(Jconj,J,&Jstar_dot_J);
  double objval = -creal(Estar_dot_J)/creal(Jstar_dot_J);

  PetscPrintf(PETSC_COMM_WORLD,"---at step %d, the objval is %.16g \n",count,objval);
  
  KSPSolveTranspose(ksp,Jconj,u);

  VecPointwiseMult(gradeps,u,x);
  VecScale(gradeps,-omega*omega/creal(Jstar_dot_J));
  VecPointwiseMult(gradeps,gradeps,epsDiff);
  multilayer_backward(PETSC_COMM_WORLD,gradeps,_ugrad,dofi,dg.da);
  filters_apply(PETSC_COMM_WORLD,_ugrad,dofgrad,flt,-1);

  VecCopy(xconj,gradJ);
  VecAXPY(gradJ,-PETSC_i*omega,u);
  VecScale(gradJ,1/(2.0*Jstar_dot_J));
  VecAXPY(gradJ,-creal(Estar_dot_J)/(Jstar_dot_J*Jstar_dot_J),Jconj);
  VecScale(gradJ,-1.0);
  vec2array(PETSC_COMM_WORLD,gradJ,jgrad,gi,dg.da);

  for(i=0;i<ndof;i++)
    dofgradAll[i]=creal(dofgrad[i]);

  for(i=0;i<ndof_j/2;i++){
    dofgradAll[ndof+i]=2.0*(jmax-jmin)*creal(jgrad[i]*cexp(PETSC_i*2.0*M_PI*dofAll[ndof+i+ndof_j/2]));
    dofgradAll[ndof+i+ndof_j/2]=2.0*2.0*M_PI*creal(jgrad[i]*PETSC_i*j[i]);
  }

  VecDestroy(&J);
  VecDestroy(&b);
  VecDestroy(&Jconj);
  VecDestroy(&eps);
  VecDestroy(&negW2eps);
  VecDestroy(&xconj);
  VecDestroy(&u);
  VecDestroy(&gradeps);
  VecDestroy(&gradJ);
  MatDestroy(&M);
  
  char output_filename[PETSC_MAX_PATH_LEN];
  if((count%printdof)==0){
    sprintf(output_filename,"outputdof_step%d.txt",count);
    writetofiledouble(PETSC_COMM_WORLD,output_filename,dofAll,ndofAll);
    sprintf(tmpstr,"outputing the dofs at step %d",count);
    updateTimeStamp(PETSC_COMM_WORLD,&global_time,tmpstr);
  }

  sprintf(tmpstr,"end of computation step %d (global timecheck)",count);
  updateTimeStamp(PETSC_COMM_WORLD,&global_time,tmpstr);

  count++;
  free(_u);
  free(_ugrad);
  free(dof);
  free(dofgrad);
  free(j);
  free(jgrad);

  return objval;

}

