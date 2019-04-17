#include "petsc.h"
#include "petscsys.h"
#include "hdf5.h"
#include "nlopt.h"
#include <assert.h>
#include "libFDOPT.h"

int count=0;
TimeStamp global_time;

PetscErrorCode set_J(MPI_Comm comm, DM da, Vec v, PetscInt iz0);
PetscErrorCode set_u(MPI_Comm comm, DM da, Vec v, PetscInt iz0);

#undef __FUNCT__
#define __FUNCT__ "main"
PetscErrorCode main(int argc, char **argv)
{
  PetscInitialize(&argc,&argv,NULL,NULL);
  int size;
  MPI_Comm_size(MPI_COMM_WORLD, &size);
  PetscPrintf(PETSC_COMM_WORLD,"\tThe total number of processors is %d\n",size);
  
  GridInfo gi;
  DOFInfo dofi;
  ParDataGrid dg;
  setGridInfo(PETSC_COMM_WORLD,"domain.h5",&gi);
  setDOFInfo(PETSC_COMM_WORLD,"domain.h5",&dofi);
  setParDataGrid(PETSC_COMM_WORLD,&dg,gi);

  Vec srcJ,u;
  PetscInt zj0=gi.Npml[2][0]+4;
  PetscInt zu0=gi.N[2]-gi.Npml[2][1]-4;
  VecDuplicate(dg.vecTemp,&srcJ);
  VecDuplicate(dg.vecTemp,&u);
  PetscPrintf(PETSC_COMM_WORLD,"zj0 %d\n",zj0);
  PetscPrintf(PETSC_COMM_WORLD,"zu0 %d\n",zu0);
  set_J(PETSC_COMM_WORLD,dg.da,srcJ, zj0);
  set_u(PETSC_COMM_WORLD,dg.da,   u, zu0);
  
  Vec mu;
  VecDuplicate(dg.vecTemp,&mu);
  VecSet(mu,1.0+PETSC_i*0.0);

  Vec b,x,y;
  VecDuplicate(dg.vecTemp,&b);
  VecDuplicate(dg.vecTemp,&x);
  VecDuplicate(dg.vecTemp,&y);

  int maxit=15, its=100;
  KSP ksp;
  PC pc;
  setupKSPDirect(PETSC_COMM_WORLD,&ksp,&pc,maxit);

  PetscReal freq, omega;
  double epsbkg,epsdiff;
  getreal("-freq",&freq,0.7);
  getreal("-epsdiff",&epsdiff,2.0);
  getreal("-epsbkg",&epsbkg,1.0);
  Vec eps;
  VecDuplicate(dg.vecTemp,&eps);

  int i;
  double *dof;
  PetscScalar *_dof;
  dof = (double *) malloc(dofi.ndof*sizeof(double));
  _dof = (PetscScalar *) malloc(dofi.ndof*sizeof(PetscScalar));
  readfromfiledouble("dof.txt",dof,dofi.ndof);
  for(i=0;i<dofi.ndof;i++)
    _dof[i]=dof[i]+PETSC_i*0;

  multilayer_forward(_dof,eps,&dofi,dg.da);
  VecScale(eps,epsdiff + PETSC_i*0);
  VecShift(eps,epsbkg);
  saveVecHDF5(PETSC_COMM_WORLD,eps,"epsilonraw.h5","eps");

  omega=2.0*M_PI*freq;
  VecCopy(srcJ,b);
  VecScale(b,-PETSC_i*omega);

  Mat M0,Curl,Meps;
  create_doublecurl_op(PETSC_COMM_WORLD,&M0,&Curl,omega,mu,gi,dg);
  MatDuplicate(M0,MAT_COPY_VALUES,&Meps);
  MatDiagonalSet(Meps,eps,ADD_VALUES);
  SolveMatrixDirect(PETSC_COMM_WORLD,ksp,Meps,b,x,&its,maxit);
  saveVecHDF5(PETSC_COMM_WORLD,x,"rcwax_fwd.h5","x");
  KSPSolveTranspose(ksp,u,y);
  saveVecHDF5(PETSC_COMM_WORLD,y,"rcwax_adj.h5","y");
  
  PetscScalar objval;
  VecTDot(u,x,&objval);

  VecPointwiseMult(y,y,x);
  VecScale(y,omega*omega*epsdiff);
  multilayer_backward(PETSC_COMM_WORLD,y,_dof,&dofi,dg.da);
  for(i=0;i<dofi.ndof;i++)
    dof[i]=2.0*creal(conj(objval)*_dof[i]);
  
  PetscPrintf(PETSC_COMM_WORLD,"objval: freq, %0.18g, %0.18g, %0.18g \n",freq,creal(objval),cimag(objval),\
	      creal(objval)*creal(objval)+cimag(objval)*cimag(objval));

  writetofiledouble(PETSC_COMM_WORLD,"gradient.dat",dof,dofi.ndof);
  
  VecDestroy(&eps);
  VecDestroy(&x);
  VecDestroy(&b);
  VecDestroy(&u);
  VecDestroy(&srcJ);
  VecDestroy(&mu);
  MatDestroy(&Meps);
  MatDestroy(&M0);
  MatDestroy(&Curl);

  free(dof);
  free(_dof);
  
  PetscFinalize();
  return 0;

}

PetscErrorCode set_J(MPI_Comm comm, DM da, Vec v, PetscInt iz0)
{

  PetscScalar val;

  PetscErrorCode ierr;

  Field ***v_array;  // 3D array that images Vec v
  ierr = DMDAVecGetArray(da, v, &v_array); CHKERRQ(ierr);

  /** Get corners and widths of Yee's grid included in this proces. */
  PetscInt ox, oy, oz;  // coordinates of beginning corner of Yee's grid in this process
  PetscInt nx, ny, nz;  // local widths of Yee's grid in this process
  ierr = DMDAGetCorners(da, &ox, &oy, &oz, &nx, &ny, &nz); CHKERRQ(ierr);

  PetscInt ix,iy,iz,ic;
  for (iz = oz; iz < oz+nz; ++iz) {
    for (iy = oy; iy < oy+ny; ++iy) {
      for (ix = ox; ix < ox+nx; ++ix) {
        for (ic = 0; ic < Naxis; ++ic) {
          val=0.0;
          if(iz==iz0 && ic==0){
            val = 1.0+PETSC_i*0;
          }
          v_array[iz][iy][ix].comp[ic]=val;

        }
      }
    }
  }

  ierr = DMDAVecRestoreArray(da, v, &v_array); CHKERRQ(ierr);

  return ierr;
}

PetscErrorCode set_u(MPI_Comm comm, DM da, Vec v, PetscInt iz0)
{

  PetscScalar val;

  PetscErrorCode ierr;

  Field ***v_array;  // 3D array that images Vec v
  ierr = DMDAVecGetArray(da, v, &v_array); CHKERRQ(ierr);

  /** Get corners and widths of Yee's grid included in this proces. */
  PetscInt ox, oy, oz;  // coordinates of beginning corner of Yee's grid in this process
  PetscInt nx, ny, nz;  // local widths of Yee's grid in this process
  ierr = DMDAGetCorners(da, &ox, &oy, &oz, &nx, &ny, &nz); CHKERRQ(ierr);

  PetscInt ix,iy,iz,ic;
  for (iz = oz; iz < oz+nz; ++iz) {
    for (iy = oy; iy < oy+ny; ++iy) {
      for (ix = ox; ix < ox+nx; ++ix) {
        for (ic = 0; ic < Naxis; ++ic) {
          val=0.0;
          if(iz==iz0 && ic==0){
            val = 1.0+PETSC_i*0;
          }
          v_array[iz][iy][ix].comp[ic]=val;

        }
      }
    }
  }

  ierr = DMDAVecRestoreArray(da, v, &v_array); CHKERRQ(ierr);

  return ierr;
}
