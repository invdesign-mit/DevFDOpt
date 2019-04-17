#include "ffoptpar.h"

extern int count;
extern TimeStamp global_time;

#undef __FUNCT__
#define __FUNCT__ "ffoptpar"
double ffoptpar(int ndofcell, double *dofcell, double *dofgradcell, void *data)
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

  char tmpstr[PETSC_MAX_PATH_LEN];
  sprintf(tmpstr,"starting computation step %d (global timecheck)",count);
  updateTimeStamp(PETSC_COMM_WORLD,&global_time,tmpstr);

  PetscScalar *dof,*dofgrad;
  dof = (PetscScalar *) malloc(ndofcell*sizeof(PetscScalar));
  dofgrad = (PetscScalar *) malloc(ndofcell*sizeof(PetscScalar));
  int i;
  for(i=0;i<ndofcell;i++){
    dof[i]=dofcell[i]+PETSC_i*0.0;
  }

  FiltersToolBox *flt = ptdata->flt;
  PetscScalar *_u,*_ugrad;
  _u = (PetscScalar *) malloc(ndofcell*sizeof(PetscScalar));
  _ugrad = (PetscScalar *) malloc(ndofcell*sizeof(PetscScalar));
  filters_apply(subcomm,dof,_u,flt,1);
  PetscScalar xfar=ffgrad(subcomm, _u,_ugrad, M0,b,x, epsDiff,epsBkg, omega, ffvec, ksp,its,maxit, dofi,dg.da);
  filters_apply(subcomm,_ugrad,dofgrad,flt,-1);
  MPI_Barrier(subcomm);
  MPI_Barrier(PETSC_COMM_WORLD);

  sprintf(tmpstr,"solving all unit cells at step %d (global timecheck)",count);
  updateTimeStamp(PETSC_COMM_WORLD,&global_time,tmpstr);

  int subrank;
  MPI_Comm_rank(subcomm, &subrank);
  if(subrank>0) xfar=0;
  MPI_Barrier(subcomm);
  MPI_Barrier(PETSC_COMM_WORLD);

  PetscScalar xfartotal;
  MPI_cellreduce(&xfar,&xfartotal,1,MPIU_SCALAR,MPI_SUM,PETSC_COMM_WORLD);
  PetscPrintf(PETSC_COMM_WORLD,"xfar_total = %g + i*(%g) \n",creal(xfartotal),cimag(xfartotal));
  double xfarsq=creal(xfartotal * conj(xfartotal));
  PetscPrintf(PETSC_COMM_WORLD,"******xfarsq at step %d is %g \n",count,xfarsq);

  sprintf(tmpstr,"reducing the objective function at step %d (global timecheck)",count);
  updateTimeStamp(PETSC_COMM_WORLD,&global_time,tmpstr);

  for(i=0;i<ndofcell;i++){
    dofgradcell[i]=2.0*creal(conj(xfartotal)*dofgrad[i]);
  }

  char output_filename[PETSC_MAX_PATH_LEN];
  if((count%printdof)==0){
    sprintf(output_filename,"outputdof_cell%d_step%d.txt",colour,count);
    writetofiledouble(subcomm,output_filename,dofcell,ndofcell);
  }

  sprintf(tmpstr,"end of computation step %d (global timecheck)",count);
  updateTimeStamp(PETSC_COMM_WORLD,&global_time,tmpstr);

  count++;
  free(_u);
  free(_ugrad);
  free(dof);
  free(dofgrad);
  return xfarsq;

}


