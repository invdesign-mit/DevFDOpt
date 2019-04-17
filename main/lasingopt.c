#include "petsc.h"
#include "petscsys.h"
#include "hdf5.h"
#include "nlopt.h"
#include <assert.h>
#include "libFDOPT.h"

int count=0;
int mma_verbose;
TimeStamp global_time;

void make_array(double *x, double val, int n);
double optimize_generic(int DegFree, double *epsopt, double *lb, double *ub, void *objdata, void **constrdata, optfunc obj, optfunc *constraint, int min_or_max, int nconstraints);

#undef __FUNCT__
#define __FUNCT__ "main"
PetscErrorCode main(int argc, char **argv)
{

  MPI_Init(NULL, NULL);
  PetscInitialize(&argc,&argv,NULL,NULL);
  int size;
  MPI_Comm_size(MPI_COMM_WORLD, &size);
  PetscPrintf(PETSC_COMM_WORLD,"\tThe total number of processors is %d\n",size);
  
  /*---------------*/
  PetscPrintf(PETSC_COMM_WORLD,"*****Lasing Threshold Optimization*****\n");
  TimeStamp ts;
  initTimeStamp(&ts);
    
  PetscReal freq;
  getreal("-freq",&freq,1.0);
  PetscScalar omega=2.0*M_PI*freq;

  int printdof;
  getint("-print_dof",&printdof,1);

  GridInfo gi;
  char tmpstr[PETSC_MAX_PATH_LEN];
  getstr("-inputfile_name",tmpstr,"domain.h5");
  setGridInfo(PETSC_COMM_WORLD,tmpstr,&gi);
  MPI_Barrier(PETSC_COMM_WORLD);

  DOFInfo dofi;
  ParDataGrid dg;
  FiltersToolBox flt;
  setDOFInfo(PETSC_COMM_WORLD,tmpstr,&dofi);
  setParDataGrid(PETSC_COMM_WORLD,&dg,gi);
  filters_initialize(PETSC_COMM_WORLD,&flt,dofi);

  MPI_Barrier(PETSC_COMM_WORLD);
  updateTimeStamp(PETSC_COMM_WORLD,&ts,"setting up grid, pardata, dof and filter infos");    

  //---epsilon inputs---
  Vec epsDiff, epsBkg;
  VecDuplicate(dg.vecTemp,&epsDiff);
  VecSet(epsDiff,0.0);
  getstr("-epsDiff_name",tmpstr,"epsDiff.h5");
  loadVecHDF5(PETSC_COMM_WORLD,epsDiff,tmpstr,"/eps");

  VecDuplicate(dg.vecTemp,&epsBkg);
  VecSet(epsBkg,0.0);
  getstr("-epsBkg_name",tmpstr,"epsBkg.h5");
  loadVecHDF5(PETSC_COMM_WORLD,epsBkg,tmpstr,"/eps");

  MPI_Barrier(PETSC_COMM_WORLD);
  updateTimeStamp(PETSC_COMM_WORLD,&ts,"loading input epsDiff and epsBkg from h5 files");    

  //--------------

  Vec mu;
  VecDuplicate(dg.vecTemp,&mu);
  VecSet(mu,1.0+PETSC_i*0.0);

  Mat M0,Curl;
  Vec x;
  KSP ksp;
  PC pc;
  int maxit=15;
  int its=1000;
  create_doublecurl_op(PETSC_COMM_WORLD,&M0,&Curl,omega,mu,gi,dg);
  VecDuplicate(dg.vecTemp,&x);
  setupKSPDirect(PETSC_COMM_WORLD,&ksp,&pc,maxit);
  MPI_Barrier(PETSC_COMM_WORLD);
  updateTimeStamp(PETSC_COMM_WORLD,&ts,"setting up curlxcurl operator, b, x and ksp");

  Lasingdata data;
  data.its=&its;
  data.M0=M0;
  data.x=x;
  data.epsDiff=epsDiff;
  data.epsBkg=epsBkg;
  data.omega=omega;
  data.ksp=ksp;
  data.maxit=maxit;
  data.gi=&gi;
  data.dofi=&dofi;
  data.dg=dg;
  data.printdof=printdof;
  data.flt=&flt;
  getreal("-jmin",&(data.jmin),0.0);
  getreal("-jmax",&(data.jmax),1.0);

  int ndofeps=dofi.ndof;
  int ndof_j=2 * gi.N[Xx]*gi.N[Yy]*gi.N[Zz]*Naxis;
  int ndofAll=ndofeps+ndof_j;
  PetscPrintf(PETSC_COMM_WORLD,"Total number of DOFs including the source DOFs is %d \n",ndofAll);
  
  double *dofAll;
  dofAll = (double *) malloc(ndofAll*sizeof(double));
  getstr("-init_dof_name",tmpstr,"dofAll.txt");
  readfromfiledouble(tmpstr,dofAll,ndofAll);

  initTimeStamp(&global_time);
  
  int Job;
  getint("-Job",&Job,1);
  if(Job==0){
    
    int print_Efields;
    getint("-print_Efields",&print_Efields,0);
    
    PetscScalar *dof,*_dof;
    int i;
    Vec eps;
    
    dof  = (PetscScalar *) malloc(ndofeps*sizeof(PetscScalar));
    _dof = (PetscScalar *) malloc(ndofeps*sizeof(PetscScalar));
    for(i=0;i<ndofeps;i++) dof[i]=dofAll[i]+PETSC_i*0.0;

    filters_apply(PETSC_COMM_WORLD,dof,_dof,&flt,1);

    VecDuplicate(dg.vecTemp,&eps);
    multilayer_forward(_dof,eps,&dofi,dg.da);
    VecPointwiseMult(eps,eps,epsDiff);
    VecAXPY(eps,1.0,epsBkg);
    saveVecHDF5(PETSC_COMM_WORLD,eps,"epsilon.h5","eps");
    MPI_Barrier(PETSC_COMM_WORLD);

    if(print_Efields){
      VecScale(eps,-omega*omega);
      MatDiagonalSet(M0,eps,ADD_VALUES);
      PetscScalar *b_array=(PetscScalar *)malloc(ndof_j/2*sizeof(PetscScalar));
      for(i=0;i<ndof_j/2;i++) b_array[i] = -PETSC_i * omega *
				(data.jmin + (data.jmax-data.jmin)*dofAll[ndofeps + i]) *
				cexp(PETSC_i * 2.0 * M_PI * dofAll[ndofeps + i + ndof_j/2]);
      Vec b;
      VecDuplicate(dg.vecTemp,&b);
      array2vec(b_array,b,&gi,dg.da);
      SolveMatrixDirect(PETSC_COMM_WORLD,ksp,M0,b,x,&its,maxit);
      MPI_Barrier(PETSC_COMM_WORLD);
      saveVecHDF5(PETSC_COMM_WORLD,x,"Efield.h5","E");
      MPI_Barrier(PETSC_COMM_WORLD);
      free(b_array);
      VecDestroy(&b);
    }
    
    VecDestroy(&eps);
    free(dof);
    free(_dof);
    
  }else if(Job==1){
    
    double *lb,*ub;
    lb=(double *) malloc(ndofAll*sizeof(double));
    ub=(double *) malloc(ndofAll*sizeof(double));
    make_array(lb,0.0,ndofAll);
    make_array(ub,1.0,ndofAll);
    optimize_generic(ndofAll, dofAll, lb,ub, &data, NULL, lasing2, NULL, 1, 0);

  }else if(Job==-1){

    double *dofgradtot;
    dofgradtot = (double *) malloc(ndofAll*sizeof(double));

    int p;
    double s,s0=0,ds=0.01,s1=1.0;
    PetscReal objval;
    getint("-change_dof_at",&p,dofi.ndof/2);
    for(s=s0;s<s1;s+=ds){
      dofAll[p]=s;
      objval=lasing2(ndofAll,dofAll,dofgradtot,&data);
      PetscPrintf(PETSC_COMM_WORLD,"objval: %.16g, %.16g, %.16g\n",dofAll[p],objval,dofgradtot[p]);
    }

    free(dofgradtot);
  }

  VecDestroy(&dg.vecTemp);
  VecDestroy(&mu);
  DMDestroy(&dg.da);
  
  VecDestroy(&epsDiff);
  VecDestroy(&epsBkg);
  VecDestroy(&x);  
  MatDestroy(&M0);
  MatDestroy(&Curl);
  KSPDestroy(&ksp);
  MatDestroy(&flt.W);
  VecDestroy(&flt.rho_grad);
  free(dofAll);

  PetscFinalize();
  return 0;

}

