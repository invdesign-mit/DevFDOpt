#include "ffopt.h"

extern int count;

#undef __FUNCT__
#define __FUNCT__ "ffgrad"
PetscScalar ffgrad(MPI_Comm comm, PetscScalar *dof, PetscScalar *dofgrad, const Mat M0, const Vec b, Vec x, const Vec epsDiff, const Vec epsBkg, const PetscScalar omega, const Vec ffvec, KSP ksp, int *its, int maxit, DOFInfo *dofi, DM da)
{

  Vec eps,grad,negW2eps;
  VecDuplicate(x,&eps);
  VecDuplicate(x,&grad);
  VecDuplicate(x,&negW2eps);

  multilayer_forward(dof,eps,dofi,da);
  VecPointwiseMult(eps,eps,epsDiff);
  VecAXPY(eps,1.0,epsBkg);

  Mat M;
  MatDuplicate(M0,MAT_COPY_VALUES,&M);
  VecCopy(eps,negW2eps);
  VecScale(negW2eps,-omega*omega);
  MatDiagonalSet(M,negW2eps,ADD_VALUES);

  SolveMatrixDirect(comm,ksp,M,b,x,its,maxit);
  PetscScalar xfar;
  VecTDot(ffvec,x,&xfar);

  Vec u;
  VecDuplicate(x,&u);
  KSPSolveTranspose(ksp,ffvec,u);
  VecPointwiseMult(grad,u,x);
  VecScale(grad,omega*omega);
  VecDestroy(&u);

  VecPointwiseMult(grad,grad,epsDiff);
  multilayer_backward(comm,grad,dofgrad,dofi,da);

  VecDestroy(&eps);
  VecDestroy(&grad);
  VecDestroy(&negW2eps);
  MatDestroy(&M);

  PetscPrintf(comm,"xfar = %g + i*(%g) \n",creal(xfar),cimag(xfar));
  return xfar;
}

#undef __FUNCT__
#define __FUNCT__ "ffopt"
double ffopt(int ndofAll, double *dofAll, double *dofgradAll, void *data)
{

  Farfielddata *ptdata = (Farfielddata *) data;

  int colour = ptdata->colour;
  MPI_Comm subcomm = ptdata->subcomm;
  Mat M0 = ptdata->M0;
  Vec b = ptdata->b;
  Vec x = ptdata->x;
  Vec epsDiff = ptdata->epsDiff;
  Vec epsBkg = ptdata->epsBkg;
  PetscScalar omega = ptdata->omega;
  Vec ffvec = ptdata->ffvec;
  KSP ksp = ptdata->ksp;
  int *its = ptdata->its;
  int maxit = ptdata->maxit;
  DOFInfo *dofi = ptdata->dofi;
  ParDataGrid dg = ptdata->dg;
  int printdof = ptdata->printdof;

  int ndof=dofi->ndof;
  PetscScalar *dof,*dofgrad;
  dof = (PetscScalar *) malloc(ndof*sizeof(PetscScalar));
  dofgrad = (PetscScalar *) malloc(ndof*sizeof(PetscScalar));
  int i;
  for(i=0;i<ndof;i++){
    dof[i]=dofAll[i+colour*ndof]+PETSC_i*0.0;
  }

  PetscScalar xfar=ffgrad(subcomm, dof,dofgrad, M0,b,x, epsDiff,epsBkg, omega, ffvec, ksp,its,maxit, dofi,dg.da);
  MPI_Barrier(subcomm);
  MPI_Barrier(PETSC_COMM_WORLD);

  int subrank;
  MPI_Comm_rank(subcomm, &subrank);
  if(subrank>0) xfar=0;
  MPI_Barrier(subcomm);
  MPI_Barrier(PETSC_COMM_WORLD);

  PetscScalar xfartotal;
  MPI_Allreduce(&xfar,&xfartotal,1,MPIU_SCALAR,MPI_SUM,PETSC_COMM_WORLD);
  PetscPrintf(PETSC_COMM_WORLD,"xfar_total = %g + i*(%g) \n",creal(xfartotal),cimag(xfartotal));
  double xfarsq=creal(xfartotal * conj(xfartotal));
  PetscPrintf(PETSC_COMM_WORLD,"******xfarsq at step %d is %g \n",count,xfarsq);

  double *tmp;
  tmp = (double *) malloc(ndofAll*sizeof(double));
  for(i=0;i<ndofAll;i++){
    if(subrank==0 && i>=colour*ndof && i<colour*ndof+ndof)
      tmp[i]= 2.0*creal(conj(xfartotal)*dofgrad[i-colour*ndof]);
    else
      tmp[i] = 0.0;
  }
  MPI_Allreduce(tmp,dofgradAll,ndofAll,MPI_DOUBLE,MPI_SUM,PETSC_COMM_WORLD);

  char output_filename[PETSC_MAX_PATH_LEN];
  if((count%printdof)==0){
    sprintf(output_filename,"outputdof_step%d.txt",count);
    writetofiledouble(PETSC_COMM_WORLD,output_filename,dofAll,ndofAll);
  }

  count++;
  free(dof);
  free(dofgrad);
  free(tmp);
  return xfarsq;

}
