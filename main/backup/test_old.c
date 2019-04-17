#include "petsc.h"
#include "petscsys.h"
#include "hdf5.h"
#include "nlopt.h"
#include <assert.h>
#include "libFDOPT.h"

int count=0;
double testcons(int ndof, double *dof, double *dofgrad, void *data);

#undef __FUNCT__
#define __FUNCT__ "main"
PetscErrorCode main(int argc, char **argv)
{

  MPI_Init(NULL, NULL);
  PetscInitialize(&argc,&argv,NULL,NULL);
  int size;
  MPI_Comm_size(MPI_COMM_WORLD, &size);
  PetscPrintf(PETSC_COMM_WORLD,"\tThe total number of processors is %d\n",size);
  
  int testjob;
  getint("-testjob",&testjob,7);
  if(testjob==0){
    PetscPrintf(PETSC_COMM_WORLD,"***Testing the parallel functionality***\n");

    char inputfile_name[PETSC_MAX_PATH_LEN];
    const char defaultfile_name[PETSC_MAX_PATH_LEN] = "opt3d.h5";
    getstr("-inputfile_name", inputfile_name, defaultfile_name);

    GridInfo gi;
    setGridInfo(inputfile_name, &gi);

    const int max_nfreq=100;
    PetscReal freq[max_nfreq];
    int nfreq=max_nfreq;
    getrealarray("-freqs",freq,&nfreq,1.0);

    //check if the number of processors is divisible by the number of subcomms 
    int ncomms, np_per_comm;
    if(!(size%nfreq==0)) SETERRQ(PETSC_COMM_WORLD,1,"The number of processes must be divisible by the number of frequencies.");
    ncomms=nfreq;
    np_per_comm=size/ncomms;
    
    //calculate the colour of each subcomm ( = rank of each processor / number of processors in each subcomm )
    //note once calculated, the colour is fixed throughout the entire run
    int rank;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm subcomm;
    int colour = rank/np_per_comm;
    MPI_Comm_split(MPI_COMM_WORLD, colour, rank, &subcomm);

    //create parallel data grids and PETSc objects under each subcomm and assign different values
    ParDataGrid dg;
    setParDataGrid(subcomm, &dg, gi);
    Vec v;
    VecDuplicate(dg.vecTemp,&v);
    VecSet(v,freq[colour]);
    PetscScalar s;
    VecSum(v,&s);
    PetscBarrier(NULL);
    PetscPrintf(subcomm,"the sum of v under the subcomm[%d] is %g.\n",colour,s);
    VecDestroy(&v);
  
  }else if(testjob==1){

    PetscPrintf(PETSC_COMM_WORLD,"***Job=1 that involves indirect dm scatters is no longer needed.\n"); 
 
  }else if(testjob==2){

    PetscPrintf(PETSC_COMM_WORLD,"***Job=2 that involves indirect dm scatters is no longer needed.\n"); 

  }else if(testjob==3){

    PetscPrintf(PETSC_COMM_WORLD,"***Testing direct dof-to-domain functions***\n");

    char inputfile_name[PETSC_MAX_PATH_LEN];
    const char defaultfile_name[PETSC_MAX_PATH_LEN] = "opt3d.h5";
    getstr("-inputfile_name", inputfile_name, defaultfile_name);

    TimeStamp ts;
    initTimeStamp(&ts);

    GridInfo gi;
    setGridInfo(inputfile_name, &gi);
    DOFInfo dofi;
    setDOFInfo(inputfile_name, &dofi);
    ParDataGrid dg;
    setParDataGrid(PETSC_COMM_WORLD,&dg,gi);
    
    updateTimeStamp(PETSC_COMM_WORLD,&ts,"setting up GridInfo, DOFInfo and ParDataGrid");

    PetscScalar *dof;
    dof=(PetscScalar *) malloc(dofi.ndof*sizeof(PetscScalar));
    int i;
    for(i=0;i<dofi.ndof;i++){
      dof[i]= 1.0 + PETSC_i * 0.0;
    } 

    updateTimeStamp(PETSC_COMM_WORLD,&ts,"initializing dof array");

    Vec Vdof;
    VecDuplicate(dg.vecTemp,&Vdof);
    multilayer_forward(dof,Vdof,&dofi,dg.da);

    updateTimeStamp(PETSC_COMM_WORLD,&ts,"directly populating domain vector from dof array");

    int outputVdof=0;
    if(outputVdof){
      saveVecHDF5(PETSC_COMM_WORLD, Vdof, "Vdof.h5", "data");
      updateTimeStamp(PETSC_COMM_WORLD,&ts,"writing parallel vector to hdf5");
    }

    PetscScalar *grad;
    grad=(PetscScalar *) malloc(dofi.ndof*sizeof(PetscScalar));
    multilayer_backward(PETSC_COMM_WORLD,Vdof,grad,&dofi,dg.da);

    updateTimeStamp(PETSC_COMM_WORLD,&ts,"condensing domain vector into dof array");

    writetofile(PETSC_COMM_WORLD,"grad.txt",grad,dofi.ndof);

    updateTimeStamp(PETSC_COMM_WORLD,&ts,"writing C array to file");

    VecDestroy(&Vdof);
    free(dof);
    free(grad);

  }else if(testjob==4){

    PetscPrintf(PETSC_COMM_WORLD,"***Testing gradients from direct dof-to-domain functions***\n");

    char inputfile_name[PETSC_MAX_PATH_LEN];
    const char defaultfile_name[PETSC_MAX_PATH_LEN] = "opt3d.h5";
    getstr("-inputfile_name", inputfile_name, defaultfile_name);

    TimeStamp ts;
    initTimeStamp(&ts);

    GridInfo gi;
    setGridInfo(inputfile_name, &gi);
    DOFInfo dofi;
    setDOFInfo(inputfile_name, &dofi);
    ParDataGrid dg;
    setParDataGrid(PETSC_COMM_WORLD,&dg,gi);
    updateTimeStamp(PETSC_COMM_WORLD,&ts,"setting up GridInfo, DOFInfo and ParDataGrid");

    PetscScalar *dof;
    dof=(PetscScalar *) malloc(dofi.ndof*sizeof(PetscScalar));
    readfromfile("dof.txt",dof,dofi.ndof);
    updateTimeStamp(PETSC_COMM_WORLD,&ts,"reading C array from file");

    Vec Vdof,U,Grad;
    VecDuplicate(dg.vecTemp,&Vdof);
    VecDuplicate(dg.vecTemp,&U);
    VecDuplicate(dg.vecTemp,&Grad);
    PetscScalar *grad;
    grad=(PetscScalar *) malloc(dofi.ndof*sizeof(PetscScalar));

    PetscInt idof;
    getint("-change_dof_at",&idof,0);
    PetscReal dp,d0=1.0,d1=2.0,ds=0.01;
    PetscScalar s;
    for(dp=d0;dp<d1;dp=dp+ds){
      
      dof[idof] = dp + PETSC_i*0.0;
      
      multilayer_forward(dof,Vdof,&dofi,dg.da);
      updateTimeStamp(PETSC_COMM_WORLD,&ts,"directly populating domain vector from dof array");

      //sum[exp(3 v^2)]
      VecPointwiseMult(U,Vdof,Vdof);
      VecScale(U,3.0);
      VecExp(U);
      VecSum(U,&s);
      s=s/dofi.ndof;
      VecPointwiseMult(Grad,U,Vdof);
      VecScale(Grad,6.0/dofi.ndof);
      updateTimeStamp(PETSC_COMM_WORLD,&ts,"processing domain vector and computing the gradient vector");

      multilayer_backward(PETSC_COMM_WORLD,Grad,grad,&dofi,dg.da);
      updateTimeStamp(PETSC_COMM_WORLD,&ts,"condensing domain vector into dof array");

      PetscPrintf(PETSC_COMM_WORLD,"---dp,val,grad: %g, %0.16g, %0.16g \n",dp,creal(s),creal(grad[idof]));
    
    }
      
    VecDestroy(&Vdof);
    VecDestroy(&U);
    VecDestroy(&Grad);
    free(dof);
    free(grad);

  }else if(testjob==5){

    PetscPrintf(PETSC_COMM_WORLD,"***Testing pml or Maxwell op creation***\n");

    char inputfile_name[PETSC_MAX_PATH_LEN];
    const char defaultfile_name[PETSC_MAX_PATH_LEN] = "opt3d.h5";
    getstr("-inputfile_name", inputfile_name, defaultfile_name);

    TimeStamp ts;
    initTimeStamp(&ts);

    GridInfo gi;
    setGridInfo(inputfile_name, &gi);
    DOFInfo dofi;
    setDOFInfo(inputfile_name, &dofi);
    ParDataGrid dg;
    setParDataGrid(PETSC_COMM_WORLD,&dg,gi);
    updateTimeStamp(PETSC_COMM_WORLD,&ts,"setting up GridInfo, DOFInfo and ParDataGrid");

    PetscScalar omega=2*PI;

    PetscInt testpml=0;
    if(testpml){
      PetscScalar *dl_stretched[Naxis][Ngt];
      PetscInt axis,gt;
      for(axis=Xx;axis<Naxis;axis++){
	for(gt=Prim;gt<Ngt;gt++){
	  dl_stretched[axis][gt]=(PetscScalar *) malloc(gi.N[axis]*sizeof(PetscScalar));
	}
      }
      updateTimeStamp(PETSC_COMM_WORLD,&ts,"allocating scdl array");
    
      stretch_dl(dl_stretched,omega,gi);
      updateTimeStamp(PETSC_COMM_WORLD,&ts,"generating PML-stretched coords");

      writetofile(PETSC_COMM_WORLD,"scdxprim.txt",dl_stretched[Xx][Prim],gi.N[Xx]);
      writetofile(PETSC_COMM_WORLD,"scdxdual.txt",dl_stretched[Xx][Dual],gi.N[Xx]);
      updateTimeStamp(PETSC_COMM_WORLD,&ts,"writing C arrays to files");

      for(axis=Xx;axis<Naxis;axis++){
	for(gt=Prim;gt<Ngt;gt++){
	  free(dl_stretched[axis][gt]);
	}
      }
    }

    PetscInt testMop=1;
    if(testMop){
      Mat M, Curl;
      Vec mu;
      VecDuplicate(dg.vecTemp,&mu);
      VecSet(mu,1.0+PETSC_i*0.0);
      create_doublecurl_op(PETSC_COMM_WORLD,&M,&Curl,omega,mu,gi,dg);
      
      MatDestroy(&M);
      MatDestroy(&Curl);
      VecDestroy(&mu);
    
    }

  }else if(testjob==6){

    PetscPrintf(PETSC_COMM_WORLD,"***Testing solver***\n");

    char inputfile_name[PETSC_MAX_PATH_LEN];
    const char defaultfile_name[PETSC_MAX_PATH_LEN] = "opt3d.h5";
    getstr("-inputfile_name", inputfile_name, defaultfile_name);

    TimeStamp ts;
    initTimeStamp(&ts);

    GridInfo gi;
    setGridInfo(inputfile_name, &gi);
    DOFInfo dofi;
    setDOFInfo(inputfile_name, &dofi);
    ParDataGrid dg;
    setParDataGrid(PETSC_COMM_WORLD,&dg,gi);
    updateTimeStamp(PETSC_COMM_WORLD,&ts,"setting up GridInfo, DOFInfo and ParDataGrid");

    SolverInfo si;
    char flag_prefix[PETSC_MAX_PATH_LEN];
    getstr("-solver_flag", flag_prefix, "-fd3d");
    setSolverInfo(flag_prefix, &si);

    PetscScalar omega=2*PI;
    Mat M, Curl;
    Vec mu,eps,b,x;
    
    VecDuplicate(dg.vecTemp,&mu);
    VecSet(mu,1.0+PETSC_i*0.0);
    create_doublecurl_op(PETSC_COMM_WORLD,&M,&Curl,omega,mu,gi,dg);
    
    VecDuplicate(dg.vecTemp,&eps);
    VecSet(eps,1.0+PETSC_i*0.0);
    VecScale(eps,-omega*omega);
    MatDiagonalSet(M,eps,ADD_VALUES);
    
    VecDuplicate(dg.vecTemp,&b);
    VecSet(b,0.0);
    VecSetValue(b,gi.Ntot/2,-PETSC_i*omega,ADD_VALUES);
    
    VecDuplicate(dg.vecTemp,&x);
    VecSetRandom(x,PETSC_NULL);

    solveEq(PETSC_COMM_WORLD,M,x,b,PETSC_NULL,PETSC_NULL,&si);

    VecDestroy(&mu);
    VecDestroy(&eps);
    VecDestroy(&b);
    VecDestroy(&x);
    MatDestroy(&M);
    MatDestroy(&Curl);

  }else if(testjob==7){
    PetscPrintf(PETSC_COMM_WORLD,"***Testing solver with arbitrary geometric inputs***\n");

    TimeStamp ts;
    initTimeStamp(&ts);
    
    GridInfo gi;
    DOFInfo dofi;
    ParDataGrid dg;
    SolverInfo si;
    char tmpstr[PETSC_MAX_PATH_LEN];
    getstr("-inputfile_name",tmpstr,"domain.h5");
    setGridInfo(tmpstr,&gi);
    setDOFInfo(tmpstr,&dofi);
    setParDataGrid(PETSC_COMM_WORLD,&dg,gi);
    getstr("-flag_prefix",tmpstr,"-fd3d");
    setSolverInfo(tmpstr,&si);
    updateTimeStamp(PETSC_COMM_WORLD,&ts,"setting up grid, pardata, dof and solver infos");

    Vec srcJ;
    VecDuplicate(dg.vecTemp,&srcJ);
    loadVecHDF5(PETSC_COMM_WORLD,srcJ,"srcJ.h5","/srcJ");
    updateTimeStamp(PETSC_COMM_WORLD,&ts,"loading srcJ from h5");

    PetscReal freq;
    getreal("-freq",&freq,1.0);
    PetscScalar omega=2*PI*freq + PETSC_i*0.0;
    
    Vec mu;
    VecDuplicate(dg.vecTemp,&mu);
    VecSet(mu,1.0+PETSC_i*0.0);
    Mat M,Curl;
    create_doublecurl_op(PETSC_COMM_WORLD,&M,&Curl,omega,mu,gi,dg);
    updateTimeStamp(PETSC_COMM_WORLD,&ts,"creating the operator Curl mu^-1 Curl with PML");    

    Vec eps;
    VecDuplicate(dg.vecTemp,&eps);
    VecSet(eps,0.0);
    loadVecHDF5(PETSC_COMM_WORLD,eps,"eps.h5","/eps");

    Vec negw2eps;
    VecDuplicate(dg.vecTemp,&negw2eps);
    VecCopy(eps,negw2eps);
    VecScale(negw2eps,-omega*omega);
    MatDiagonalSet(M,negw2eps,ADD_VALUES);
    Vec b;
    VecDuplicate(dg.vecTemp,&b);
    VecCopy(srcJ,b);
    VecScale(b,-PETSC_i*omega);
    updateTimeStamp(PETSC_COMM_WORLD,&ts,"setting up the full Maxwell operator and RHS");

    Vec x;
    VecDuplicate(dg.vecTemp,&x);
    //VecSetRandom(x,PETSC_NULL);
    VecSet(x,0.0);

    if(si.use_mat_sym) MatIsSymmetric(M,0.0,&si.use_mat_sym);
    solveEq(PETSC_COMM_WORLD,M,x,b,PETSC_NULL,PETSC_NULL,&si);
    updateTimeStamp(PETSC_COMM_WORLD,&ts,"solving the Maxwell's equations");
    

    saveVecHDF5(PETSC_COMM_WORLD,x,"solution.h5","x");
    updateTimeStamp(PETSC_COMM_WORLD,&ts,"saving the E field solution to h5");
    
    MatDestroy(&M);
    MatDestroy(&Curl);
    VecDestroy(&srcJ);
    VecDestroy(&mu);
    VecDestroy(&eps);
    VecDestroy(&negw2eps);
    VecDestroy(&b);
    VecDestroy(&x);
    VecDestroy(&dg.vecTemp);
    DMDestroy(&dg.da);

    free(gi.dl[Xx][Prim]);
    free(gi.dl[Yy][Prim]);
    free(gi.dl[Zz][Prim]);
    free(gi.dl[Xx][Dual]);
    free(gi.dl[Yy][Dual]);
    free(gi.dl[Zz][Dual]);

  }else if(testjob==8){
    PetscPrintf(PETSC_COMM_WORLD,"***Testing direct solver for 2D***\n");

    TimeStamp ts;
    initTimeStamp(&ts);
    
    GridInfo gi;
    DOFInfo dofi;
    ParDataGrid dg;
    SolverInfo si;
    char tmpstr[PETSC_MAX_PATH_LEN];
    getstr("-inputfile_name",tmpstr,"domain.h5");
    setGridInfo(tmpstr,&gi);
    setDOFInfo(tmpstr,&dofi);
    setParDataGrid(PETSC_COMM_WORLD,&dg,gi);
    getstr("-flag_prefix",tmpstr,"-fd2d");
    setSolverInfo(tmpstr,&si);
    updateTimeStamp(PETSC_COMM_WORLD,&ts,"setting up grid, pardata, dof and solver infos");

    Vec srcJ;
    VecDuplicate(dg.vecTemp,&srcJ);
    //loadVecHDF5(PETSC_COMM_WORLD,srcJ,"srcJ.h5","/srcJ");
    //updateTimeStamp(PETSC_COMM_WORLD,&ts,"loading srcJ from h5");
    PetscReal zj_ref=2.0;
    PetscReal angle;
    getreal("-angle",&angle,0);
    pwsrc_2dxz(PETSC_COMM_WORLD,dg.da,srcJ,1.0,1.5,angle,1.0+0*PETSC_i,0,&zj_ref,0.02,0.02,1);
    saveVecHDF5(PETSC_COMM_WORLD,srcJ,"srcJ.h5","srcJ");

    PetscReal freq;
    getreal("-freq",&freq,1.0);
    PetscScalar omega=2*PI*freq + PETSC_i*0.0;
    
    Vec mu;
    VecDuplicate(dg.vecTemp,&mu);
    VecSet(mu,1.0+PETSC_i*0.0);
    Mat M,Curl;
    create_doublecurl_op(PETSC_COMM_WORLD,&M,&Curl,omega,mu,gi,dg);
    updateTimeStamp(PETSC_COMM_WORLD,&ts,"creating the operator Curl mu^-1 Curl with PML");    

    Vec eps;
    VecDuplicate(dg.vecTemp,&eps);
    VecSet(eps,0.0);
    loadVecHDF5(PETSC_COMM_WORLD,eps,"eps.h5","/eps");

    Vec negw2eps;
    VecDuplicate(dg.vecTemp,&negw2eps);
    VecCopy(eps,negw2eps);
    VecScale(negw2eps,-omega*omega);
    MatDiagonalSet(M,negw2eps,ADD_VALUES);
    Vec b;
    VecDuplicate(dg.vecTemp,&b);
    VecCopy(srcJ,b);
    VecScale(b,-PETSC_i*omega);
    updateTimeStamp(PETSC_COMM_WORLD,&ts,"setting up the full Maxwell operator and RHS");

    Vec x;
    VecDuplicate(dg.vecTemp,&x);

    KSP ksp;
    PC pc;
    int maxit=15;
    int its=100;
    setupKSPDirect(PETSC_COMM_WORLD,&ksp,&pc,maxit);
    SolveMatrixDirect(PETSC_COMM_WORLD,ksp,M,b,x,&its,maxit);
    
    Vec ffvec;
    VecDuplicate(dg.vecTemp,&ffvec);
    PetscReal zp_ref=1.2;
    ff2dxz(PETSC_COMM_WORLD,dg.da,ffvec,2,20000,1.0,0,&zp_ref,0.02,0.02,1);

    PetscScalar ffx;
    VecTDot(ffvec,x,&ffx);
    PetscPrintf(PETSC_COMM_WORLD,"ffx = %g + %g * i\n",creal(ffx),cimag(ffx));
    VecDestroy(&ffvec);

    saveVecHDF5(PETSC_COMM_WORLD,x,"solution.h5","x");
    updateTimeStamp(PETSC_COMM_WORLD,&ts,"saving the E field solution to h5");
    
    MatDestroy(&M);
    MatDestroy(&Curl);
    VecDestroy(&srcJ);
    VecDestroy(&mu);
    VecDestroy(&eps);
    VecDestroy(&negw2eps);
    VecDestroy(&b);
    VecDestroy(&x);
    VecDestroy(&dg.vecTemp);
    DMDestroy(&dg.da);

    free(gi.dl[Xx][Prim]);
    free(gi.dl[Yy][Prim]);
    free(gi.dl[Zz][Prim]);
    free(gi.dl[Xx][Dual]);
    free(gi.dl[Yy][Dual]);
    free(gi.dl[Zz][Dual]);

  }else if(testjob==9){
    PetscPrintf(PETSC_COMM_WORLD,"***Testing gradient with direct solver for 2D***\n");

    TimeStamp ts;
    initTimeStamp(&ts);
    
    GridInfo gi;
    DOFInfo dofi;
    ParDataGrid dg;
    SolverInfo si;
    char tmpstr[PETSC_MAX_PATH_LEN];
    getstr("-inputfile_name",tmpstr,"domain.h5");
    setGridInfo(tmpstr,&gi);
    setDOFInfo(tmpstr,&dofi);
    setParDataGrid(PETSC_COMM_WORLD,&dg,gi);
    getstr("-flag_prefix",tmpstr,"-fd2d");
    setSolverInfo(tmpstr,&si);
    updateTimeStamp(PETSC_COMM_WORLD,&ts,"setting up grid, pardata, dof and solver infos");

    PetscReal freq;
    getreal("-freq",&freq,1.0);
    PetscScalar omega=2*PI*freq + PETSC_i*0.0;

    Vec srcJ;
    VecDuplicate(dg.vecTemp,&srcJ);
    PetscReal zj_ref=1.0;
    PetscReal angle;
    getreal("-angle",&angle,0);
    pwsrc_2dxz(PETSC_COMM_WORLD,dg.da,srcJ,1.0,1.5,angle,1.0+0*PETSC_i,0,&zj_ref,0.02,0.02,1);

    Vec mu;
    VecDuplicate(dg.vecTemp,&mu);
    VecSet(mu,1.0+PETSC_i*0.0);
    Mat M0,Curl;
    create_doublecurl_op(PETSC_COMM_WORLD,&M0,&Curl,omega,mu,gi,dg);
    updateTimeStamp(PETSC_COMM_WORLD,&ts,"creating the operator Curl mu^-1 Curl with PML");    

    Vec epsDiff, epsBkg;
    VecDuplicate(dg.vecTemp,&epsDiff);
    VecSet(epsDiff,0.0);
    loadVecHDF5(PETSC_COMM_WORLD,epsDiff,"epsDiff.h5","/epsDiff");
    VecDuplicate(dg.vecTemp,&epsBkg);
    VecSet(epsBkg,0.0);
    loadVecHDF5(PETSC_COMM_WORLD,epsBkg,"epsBkg.h5","/epsBkg");

    Vec b;
    VecDuplicate(dg.vecTemp,&b);
    VecCopy(srcJ,b);
    VecScale(b,-PETSC_i*omega);
    updateTimeStamp(PETSC_COMM_WORLD,&ts,"setting up the full Maxwell operator and RHS");

    Vec x;
    VecDuplicate(dg.vecTemp,&x);

    KSP ksp;
    PC pc;
    int maxit=15;
    int its=100;
    setupKSPDirect(PETSC_COMM_WORLD,&ksp,&pc,maxit);

    PetscScalar *dof,*dofgrad;
    dof=(PetscScalar *) malloc(dofi.ndof*sizeof(PetscScalar));
    dofgrad=(PetscScalar *) malloc(dofi.ndof*sizeof(PetscScalar));
    readfromfile("dof.txt",dof,dofi.ndof);
    
    Vec ffvec;
    VecDuplicate(dg.vecTemp,&ffvec);
    PetscReal zp_ref=3.0;
    ff2dxz(PETSC_COMM_WORLD,dg.da,ffvec,2,20000,1.0,0,&zp_ref,0.02,0.02,1);

    Vec eps;
    VecDuplicate(dg.vecTemp,&eps);
    multilayer_forward(dof,eps,&dofi,dg.da);
    VecPointwiseMult(eps,eps,epsDiff);
    VecAXPY(eps,1.0,epsBkg);
    saveVecHDF5(PETSC_COMM_WORLD,eps,"eps.h5","eps");
    updateTimeStamp(PETSC_COMM_WORLD,&ts,"saving the epsilon profile to h5");
    VecDestroy(&eps);

    int p;
    double s,s0=0,ds=0.01,s1=1.0;
    PetscScalar xfar;
    getint("-change_dof_at",&p,0);
    for(s=s0;s<s1;s+=ds){
      dof[p]=s+PETSC_i*0;
      xfar=ffgrad(PETSC_COMM_WORLD, dof,dofgrad, M0,b,x, epsDiff,epsBkg, omega, ffvec, ksp,&its,maxit, &dofi,dg.da);
      PetscPrintf(PETSC_COMM_WORLD,"xfar: %g, %g, %g\n",creal(dof[p]),creal(xfar),creal(dofgrad[p]));
    }

    MatDestroy(&M0);
    MatDestroy(&Curl);
    VecDestroy(&srcJ);
    VecDestroy(&mu);
    VecDestroy(&epsDiff);
    VecDestroy(&epsBkg);
    VecDestroy(&ffvec);
    VecDestroy(&b);
    VecDestroy(&x);
    VecDestroy(&dg.vecTemp);
    DMDestroy(&dg.da);
    KSPDestroy(&ksp);
    PCDestroy(&pc);
    
    free(dof);
    free(dofgrad);

    free(gi.dl[Xx][Prim]);
    free(gi.dl[Yy][Prim]);
    free(gi.dl[Zz][Prim]);
    free(gi.dl[Xx][Dual]);
    free(gi.dl[Yy][Dual]);
    free(gi.dl[Zz][Dual]);

  }else if(testjob==10){

    int ncells;
    getint("-numcells",&ncells,2);
    //check if the number of processors is divisible by the number of subcomms 
    int ncomms=ncells, np_per_comm;
    if(!(size%ncomms==0)) SETERRQ(PETSC_COMM_WORLD,1,"The number of processes must be divisible by the number of subcomms.");
    np_per_comm=size/ncomms;
    PetscPrintf(PETSC_COMM_WORLD,"\tThe number of subcomms (= # of cells) is %d.\n\tEach cell is being simulated across %d processors.\n",ncomms,np_per_comm);
    
    //calculate the colour of each subcomm ( = rank of each processor / number of processors in each subcomm )
    //note once calculated, the colour is fixed throughout the entire run
    int rank;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm subcomm;
    int colour = rank/np_per_comm;
    MPI_Comm_split(MPI_COMM_WORLD, colour, rank, &subcomm);
    
    /*---------------*/
    PetscPrintf(PETSC_COMM_WORLD,"*****Tesing the 2D farfield solver with embarassingly parallel setup******\n");
    TimeStamp ts;
    initTimeStamp(&ts);
    
    GridInfo gi;
    DOFInfo dofi;
    char tmpstr[PETSC_MAX_PATH_LEN];
    getstr("-inputfile_name",tmpstr,"domain.h5");
    setGridInfo(tmpstr,&gi);
    setDOFInfo(tmpstr,&dofi);
    PetscPrintf(PETSC_COMM_WORLD,"Note: since there are %d unit cells, each with %d dofs, there are %d dofs in total.\n",ncells,dofi.ndof,dofi.ndof*ncells);

    ParDataGrid dg;
    setParDataGrid(subcomm,&dg,gi);

    MPI_Barrier(PETSC_COMM_WORLD);
    updateTimeStamp(PETSC_COMM_WORLD,&ts,"setting up grid, pardata and dof infos");    

    /*---inputs---*/
    Vec epsDiff, epsBkg;
    VecDuplicate(dg.vecTemp,&epsDiff);
    VecSet(epsDiff,0.0);
    loadVecHDF5(subcomm,epsDiff,"epsDiff.h5","/epsDiff");
    VecDuplicate(dg.vecTemp,&epsBkg);
    VecSet(epsBkg,0.0);
    loadVecHDF5(subcomm,epsBkg,"epsBkg.h5","/epsBkg");

    PetscScalar *dofAll, *dof, *dofgradAll, *dofgrad;
    dofAll = (PetscScalar *) malloc(dofi.ndof*ncells*sizeof(PetscScalar));
    dof = (PetscScalar *) malloc(dofi.ndof*sizeof(PetscScalar));
    dofgradAll = (PetscScalar *) malloc(dofi.ndof*ncells*sizeof(PetscScalar));
    dofgrad = (PetscScalar *) malloc(dofi.ndof*sizeof(PetscScalar));
    readfromfile("dof.txt",dofAll,dofi.ndof*ncells);
    int i;
    for(i=0;i<dofi.ndof;i++){
      dof[i]=dofAll[i+colour*dofi.ndof];
    }

    MPI_Barrier(PETSC_COMM_WORLD);
    updateTimeStamp(PETSC_COMM_WORLD,&ts,"loading input epsDiff and epsBkg from h5 files, dofAll from text file");    

    /*------*/
    PetscReal freq;
    getreal("-freq",&freq,1.0);
    PetscScalar omega=2*PI*freq + PETSC_i*0.0;

    Vec srcJ;
    VecDuplicate(dg.vecTemp,&srcJ);
    PetscReal nsub,angle,x_offset,zj_ref,hx,hz;
    PetscScalar amp=1.0+PETSC_i*0;
    getreal("-nsub",&nsub,1.5);
    getreal("-angle",&angle,0);
    getreal("-zjref",&zj_ref,1.0);
    getreal("-hx",&hx,0.02);
    getreal("-hz",&hz,0.02);
    x_offset=colour*dofi.Mx*hx;
    MPI_Barrier(subcomm);
    MPI_Barrier(PETSC_COMM_WORLD);
    pwsrc_2dxz(subcomm,dg.da,srcJ,freq,nsub,angle,amp,x_offset,&zj_ref,hx,hz,1);
    MPI_Barrier(subcomm);
    MPI_Barrier(PETSC_COMM_WORLD);

    Vec ffvec;
    VecDuplicate(dg.vecTemp,&ffvec);
    PetscReal far_x, far_z, zp_ref;
    getreal("-far_x",&far_x,dofi.Mx*ncells/2);
    getreal("-far_z",&far_z,200000);
    getreal("-zpref",&zp_ref,3.0);
    MPI_Barrier(subcomm);
    MPI_Barrier(PETSC_COMM_WORLD);
    ff2dxz(subcomm,dg.da,ffvec,far_x,far_z,freq,x_offset,&zp_ref,hx,hz,1);
    MPI_Barrier(subcomm);
    MPI_Barrier(PETSC_COMM_WORLD);

    Vec mu;
    VecDuplicate(dg.vecTemp,&mu);
    VecSet(mu,1.0+PETSC_i*0.0);
    Mat M0,Curl;
    create_doublecurl_op(subcomm,&M0,&Curl,omega,mu,gi,dg);
    VecDestroy(&mu);

    Vec b;
    VecDuplicate(dg.vecTemp,&b);
    VecCopy(srcJ,b);
    VecScale(b,-PETSC_i*omega);

    Vec x;
    VecDuplicate(dg.vecTemp,&x);

    KSP ksp;
    PC pc;
    int maxit=15;
    int its=100;
    setupKSPDirect(subcomm,&ksp,&pc,maxit);

    MPI_Barrier(subcomm);
    MPI_Barrier(PETSC_COMM_WORLD);
    updateTimeStamp(PETSC_COMM_WORLD,&ts,"setting up srcJ, farfield convoluter, curlxcurl operator, b, x and ksp");    

    /*-----compute-----*/
    PetscScalar xfar=ffgrad(subcomm, dof,dofgrad, M0,b,x, epsDiff,epsBkg, omega, ffvec, ksp,&its,maxit, &dofi,dg.da);
    
    int subrank;
    MPI_Comm_rank(subcomm, &subrank);
    if(subrank>0) xfar=0;
    MPI_Barrier(subcomm);
    MPI_Barrier(PETSC_COMM_WORLD);

    PetscScalar xfartotal;
    MPI_Allreduce(&xfar,&xfartotal,1,MPIU_SCALAR,MPI_SUM,PETSC_COMM_WORLD);
    PetscPrintf(PETSC_COMM_WORLD,"xfar_total = %g + i*(%g) \n",creal(xfartotal),cimag(xfartotal));

    for(i=0;i<dofi.ndof*ncells;i++) 
      dofgradAll[i]=0+PETSC_i*0;
    if(subrank==0){
      for(i=0;i<dofi.ndof;i++){
	dofgradAll[i+colour*dofi.ndof]=dofgrad[i];
      }
    }
    PetscScalar *dofgradAll_total;
    dofgradAll_total=(PetscScalar *) malloc(dofi.ndof*ncells*sizeof(PetscScalar));
    MPI_Allreduce(dofgradAll,dofgradAll_total,dofi.ndof*ncells,MPIU_SCALAR,MPI_SUM,PETSC_COMM_WORLD);
    writetofile(PETSC_COMM_WORLD,"dofgradAlltotal.txt",dofgradAll_total,dofi.ndof*ncells);

  }else if(testjob==11){

    int ncells;
    getint("-numcells",&ncells,2);
    //check if the number of processors is divisible by the number of subcomms 
    int ncomms=ncells, np_per_comm;
    if(!(size%ncomms==0)) SETERRQ(PETSC_COMM_WORLD,1,"The number of processes must be divisible by the number of subcomms.");
    np_per_comm=size/ncomms;
    PetscPrintf(PETSC_COMM_WORLD,"\tThe number of subcomms (= # of cells) is %d.\n\tEach cell is being simulated across %d processors.\n",ncomms,np_per_comm);
    
    //calculate the colour of each subcomm ( = rank of each processor / number of processors in each subcomm )
    //note once calculated, the colour is fixed throughout the entire run
    int rank;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm subcomm;
    int colour = rank/np_per_comm;
    MPI_Comm_split(MPI_COMM_WORLD, colour, rank, &subcomm);
    
    /*---------------*/
    PetscPrintf(PETSC_COMM_WORLD,"*****Tesing the 2D farfield solver and the total gradient calcuation with an embarassingly parallel setup******\n");
    TimeStamp ts;
    initTimeStamp(&ts);
    
    GridInfo gi;
    DOFInfo dofi;
    char tmpstr[PETSC_MAX_PATH_LEN];
    getstr("-inputfile_name",tmpstr,"domain.h5");
    setGridInfo(tmpstr,&gi);
    setDOFInfo(tmpstr,&dofi);
    PetscPrintf(PETSC_COMM_WORLD,"Note: since there are %d unit cells, each with %d dofs, there are %d dofs in total.\n",ncells,dofi.ndof,dofi.ndof*ncells);

    ParDataGrid dg;
    setParDataGrid(subcomm,&dg,gi);

    MPI_Barrier(PETSC_COMM_WORLD);
    updateTimeStamp(PETSC_COMM_WORLD,&ts,"setting up grid, pardata and dof infos");    

    /*---epsilon inputs---*/
    Vec epsDiff, epsBkg;
    VecDuplicate(dg.vecTemp,&epsDiff);
    VecSet(epsDiff,0.0);
    loadVecHDF5(subcomm,epsDiff,"epsDiff.h5","/epsDiff");
    VecDuplicate(dg.vecTemp,&epsBkg);
    VecSet(epsBkg,0.0);
    loadVecHDF5(subcomm,epsBkg,"epsBkg.h5","/epsBkg");

    MPI_Barrier(PETSC_COMM_WORLD);
    updateTimeStamp(PETSC_COMM_WORLD,&ts,"loading input epsDiff and epsBkg from h5 files, dofAll from text file");    
    /*------*/
    PetscReal freq;
    getreal("-freq",&freq,1.0);
    PetscScalar omega=2*PI*freq + PETSC_i*0.0;

    Vec srcJ,ffvec;
    VecDuplicate(dg.vecTemp,&srcJ);
    VecDuplicate(dg.vecTemp,&ffvec);
    int asymp=0;
    getint("-asymp",&asymp,0);
    if(asymp){
      PetscReal nsub,angle,x_offset,zj_ref,hx,hz;
      PetscScalar amp=10000.0+PETSC_i*0;
      getreal("-nsub",&nsub,1.5);
      getreal("-angle",&angle,0);
      getreal("-zjref",&zj_ref,1.0);
      getreal("-hx",&hx,0.02);
      getreal("-hz",&hz,0.02);
      x_offset=colour*dofi.Mx*hx-dofi.Mx*hx*ncells/2;
      MPI_Barrier(subcomm);
      MPI_Barrier(PETSC_COMM_WORLD);
      pwsrc_2dxz(subcomm,dg.da,srcJ,freq,nsub,angle,amp,x_offset,&zj_ref,hx,hz,1);
      MPI_Barrier(subcomm);
      MPI_Barrier(PETSC_COMM_WORLD);
      
      PetscReal far_r, far_angle, zp_ref;
      getreal("-far_r",&far_r,100000);
      getreal("-far_angle",&far_angle,0);
      getreal("-zpref",&zp_ref,3.0);
      MPI_Barrier(subcomm);
      MPI_Barrier(PETSC_COMM_WORLD);
      ff2dxz_asymp(subcomm,dg.da,ffvec,far_r,far_angle,freq,x_offset,&zp_ref,hx,hz,1);
      MPI_Barrier(subcomm);
      MPI_Barrier(PETSC_COMM_WORLD);
    }else{
      PetscReal nsub,angle,x_offset,zj_ref,hx,hz;
      PetscScalar amp=10000.0+PETSC_i*0;
      getreal("-nsub",&nsub,1.5);
      getreal("-angle",&angle,-30);
      getreal("-zjref",&zj_ref,1.0);
      getreal("-hx",&hx,0.02);
      getreal("-hz",&hz,0.02);
      x_offset=colour*dofi.Mx*hx;
      MPI_Barrier(subcomm);
      MPI_Barrier(PETSC_COMM_WORLD);
      pwsrc_2dxz(subcomm,dg.da,srcJ,freq,nsub,angle,amp,x_offset,&zj_ref,hx,hz,1);
      MPI_Barrier(subcomm);
      MPI_Barrier(PETSC_COMM_WORLD);

      PetscReal far_x, far_z, zp_ref;
      getreal("-far_x",&far_x,dofi.Mx*ncells/2);
      getreal("-far_z",&far_z,200000);
      getreal("-zpref",&zp_ref,3.0);
      MPI_Barrier(subcomm);
      MPI_Barrier(PETSC_COMM_WORLD);
      ff2dxz(subcomm,dg.da,ffvec,far_x,far_z,freq,x_offset,&zp_ref,hx,hz,1);
      MPI_Barrier(subcomm);
      MPI_Barrier(PETSC_COMM_WORLD);
    }

    Vec mu;
    VecDuplicate(dg.vecTemp,&mu);
    VecSet(mu,1.0+PETSC_i*0.0);
    Mat M0,Curl;
    create_doublecurl_op(subcomm,&M0,&Curl,omega,mu,gi,dg);
    VecDestroy(&mu);

    Vec b;
    VecDuplicate(dg.vecTemp,&b);
    VecCopy(srcJ,b);
    VecScale(b,-PETSC_i*omega);

    Vec x;
    VecDuplicate(dg.vecTemp,&x);

    KSP ksp;
    PC pc;
    int maxit=15;
    int its=1000;
    setupKSPDirect(subcomm,&ksp,&pc,maxit);

    MPI_Barrier(subcomm);
    MPI_Barrier(PETSC_COMM_WORLD);
    updateTimeStamp(PETSC_COMM_WORLD,&ts,"setting up srcJ, farfield convoluter, curlxcurl operator, b, x and ksp");    

    /*-----compute-----*/
    double *dofAll, *dofgradAll;
    dofAll = (double *) malloc(dofi.ndof*ncells*sizeof(double));
    dofgradAll = (double *) malloc(dofi.ndof*ncells*sizeof(double));
    readfromfiledouble("dof.txt",dofAll,dofi.ndof*ncells);

    FiltersToolBox flt;
    filters_initialize(subcomm,&flt,dofi);
    
    Farfielddata ffdat={colour,subcomm, M0,b,x, epsDiff,epsBkg, omega, ffvec, ksp,&its,maxit, &dofi,dg, 10000, &flt};
    int p;
    double s,s0=0,ds=0.01,s1=1.0;
    PetscReal xfarsq;
    getint("-change_dof_at",&p,0);
    for(s=s0;s<s1;s+=ds){
      dofAll[p]=s;
      xfarsq=ffopt(dofi.ndof*ncells,dofAll,dofgradAll,&ffdat);
      PetscPrintf(PETSC_COMM_WORLD,"xfarsq: %g, %g, %g\n",dofAll[p],xfarsq,dofgradAll[p]);
    }



    /*
    double test_a=10.123;
    void *constrdata[]={&test_a};
    optfunc constraint[]={testcons};
    optimize_eps(dofi.ndof*ncells, dofAll, &ffdat, constrdata, ffopt, constraint, 1, 0);
    */

  }else if(testjob==12){
    /*
    pwparams pw={1.0,1.0, 1.1111,1.234,2.72,8.4,24};
    PetscScalar val[5]={0,0,0,0,0};
    planewave(1.0,val,&pw);
    PetscPrintf(PETSC_COMM_WORLD,"\tplanewave vals: \n \
\tval[0]: %g + i (%g),\n\
\tval[1]: %g + i (%g),\n\
\tval[2]: %g + i (%g),\n\
\tval[3]: %g + i (%g),\n\
\tval[4]: %g + i (%g),\n",\
		creal(val[0]),cimag(val[0]),\
		creal(val[1]),cimag(val[1]),\
		creal(val[2]),cimag(val[2]),\
		creal(val[3]),cimag(val[3]),\
		creal(val[4]),cimag(val[4]));
    */

    /*****/
    int ncells;
    getint("-numcells",&ncells,2);
    //check if the number of processors is divisible by the number of subcomms 
    int ncomms=ncells, np_per_comm;
    if(!(size%ncomms==0)) SETERRQ(PETSC_COMM_WORLD,1,"The number of processes must be divisible by the number of subcomms.");
    np_per_comm=size/ncomms;
    PetscPrintf(PETSC_COMM_WORLD,"\tThe number of subcomms (= # of cells) is %d.\n\tEach cell is being simulated across %d processors.\n",ncomms,np_per_comm);
    
    //calculate the colour of each subcomm ( = rank of each processor / number of processors in each subcomm )
    //note once calculated, the colour is fixed throughout the entire run
    int rank;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm subcomm;
    int colour = rank/np_per_comm;
    MPI_Comm_split(MPI_COMM_WORLD, colour, rank, &subcomm);
    
    /*---------------*/
    PetscPrintf(PETSC_COMM_WORLD,"*****Tesing the 2D farfield solver and the total gradient calcuation with an embarassingly parallel setup******\n");
    TimeStamp ts;
    initTimeStamp(&ts);
    
    GridInfo gi;
    DOFInfo dofi;
    char tmpstr[PETSC_MAX_PATH_LEN];
    getstr("-inputfile_name",tmpstr,"domain.h5");
    setGridInfo(tmpstr,&gi);
    setDOFInfo(tmpstr,&dofi);
    PetscPrintf(PETSC_COMM_WORLD,"Note: since there are %d unit cells, each with %d dofs, there are %d dofs in total.\n",ncells,dofi.ndof,dofi.ndof*ncells);

    ParDataGrid dg;
    setParDataGrid(subcomm,&dg,gi);

    FiltersToolBox flt;
    filters_initialize(subcomm,&flt,dofi);

    MPI_Barrier(PETSC_COMM_WORLD);
    updateTimeStamp(PETSC_COMM_WORLD,&ts,"setting up grid, pardata, dof and filter infos");    

    /*---epsilon inputs---*/
    Vec epsDiff, epsBkg;
    VecDuplicate(dg.vecTemp,&epsDiff);
    VecSet(epsDiff,0.0);
    loadVecHDF5(subcomm,epsDiff,"epsDiff.h5","/epsDiff");
    VecDuplicate(dg.vecTemp,&epsBkg);
    VecSet(epsBkg,0.0);
    loadVecHDF5(subcomm,epsBkg,"epsBkg.h5","/epsBkg");

    MPI_Barrier(PETSC_COMM_WORLD);
    updateTimeStamp(PETSC_COMM_WORLD,&ts,"loading input epsDiff and epsBkg from h5 files, dofAll from text file");    
    /*--------------------*/

    PetscReal freq;
    getreal("-freq",&freq,1.0);
    PetscScalar omega=2*PI*freq + PETSC_i*0.0;
    Vec mu;
    VecDuplicate(dg.vecTemp,&mu);
    VecSet(mu,1.0+PETSC_i*0.0);
    Mat M0,Curl;
    create_doublecurl_op(subcomm,&M0,&Curl,omega,mu,gi,dg);
    VecDestroy(&mu);
    KSP ksp;
    PC pc;
    int maxit=15;
    int its=1000;
    setupKSPDirect(subcomm,&ksp,&pc,maxit);
    MPI_Barrier(subcomm);
    MPI_Barrier(PETSC_COMM_WORLD);
    /*------------------------*/

    Vec v0,v1,v2,v3,v4,u;
    Vec b0,b1,b2,b3,b4,usrc;
    Vec x0,x1,x2,x3,x4;
    VecDuplicate(dg.vecTemp,&v0);
    VecDuplicate(dg.vecTemp,&v1);
    VecDuplicate(dg.vecTemp,&v2);
    VecDuplicate(dg.vecTemp,&v3);
    VecDuplicate(dg.vecTemp,&v4);
    VecDuplicate(dg.vecTemp,&u);
    VecDuplicate(dg.vecTemp,&b0);
    VecDuplicate(dg.vecTemp,&b1);
    VecDuplicate(dg.vecTemp,&b2);
    VecDuplicate(dg.vecTemp,&b3);
    VecDuplicate(dg.vecTemp,&b4);
    VecDuplicate(dg.vecTemp,&usrc);
    VecDuplicate(dg.vecTemp,&x0);
    VecDuplicate(dg.vecTemp,&x1);
    VecDuplicate(dg.vecTemp,&x2);
    VecDuplicate(dg.vecTemp,&x3);
    VecDuplicate(dg.vecTemp,&x4);

    PetscReal x_offset,zref,zsrc,nmed_in,nmed_out,hx,hz;
    getreal("-zsrc",&zsrc,1.0);
    getreal("-zref",&zref,3.0);
    getreal("-nmed_in",&nmed_in,1.5);
    getreal("-nmed_out",&nmed_out,1.0);
    getreal("-hx",&hx,0.02);
    getreal("-hz",&hz,0.02);
    x_offset=colour*dofi.Mx*hx-dofi.Mx*hx*ncells/2;
    MPI_Barrier(subcomm);
    MPI_Barrier(PETSC_COMM_WORLD);

    PetscReal obj0;
    PetscReal obj1;
    PetscReal obj2;
    PetscReal obj3;
    PetscReal obj4;
    PetscScalar *grad0;
    PetscScalar *grad1;
    PetscScalar *grad2;
    PetscScalar *grad3;
    PetscScalar *grad4;
    PetscReal d0amp_mag;
    PetscReal d1amp_mag;
    PetscReal d2amp_mag;
    PetscReal d3amp_mag;
    PetscReal d4amp_mag;
    PetscReal d0amp_phi;
    PetscReal d1amp_phi;
    PetscReal d2amp_phi;
    PetscReal d3amp_phi;
    PetscReal d4amp_phi;
    PetscInt solve_order[5]={1,1,1,1,1};
    PetscReal mask_order[5]={1,0.01,0.0001,0.0000001,0.000000001};
    grad0 = (PetscScalar *) malloc(dofi.ndof*sizeof(PetscScalar));
    grad1 = (PetscScalar *) malloc(dofi.ndof*sizeof(PetscScalar));
    grad2 = (PetscScalar *) malloc(dofi.ndof*sizeof(PetscScalar));
    grad3 = (PetscScalar *) malloc(dofi.ndof*sizeof(PetscScalar));
    grad4 = (PetscScalar *) malloc(dofi.ndof*sizeof(PetscScalar));
    Phasefielddata data;
    data.colour=colour,data.subcomm=subcomm,data.M0=M0,data.epsDiff=epsDiff,data.epsBkg=epsBkg,data.omega=omega,data.ksp=ksp,data.its=&its,data.maxit=maxit,data.dofi=&dofi,data.dg=dg;
    data.x0=x0,data.x1=x1,data.x2=x2,data.x3=x3,data.x4=x4;
    data.obj0=&obj0,data.obj1=&obj1,data.obj2=&obj2,data.obj3=&obj3,data.obj4=&obj4;
    data.grad0=grad0,data.grad1=grad1,data.grad2=grad2,data.grad3=grad3,data.grad4=grad4;
    data.d0amp_mag=&d0amp_mag,data.d1amp_mag=&d1amp_mag,data.d2amp_mag=&d2amp_mag,data.d3amp_mag=&d3amp_mag,data.d4amp_mag=&d4amp_mag;
    data.d0amp_phi=&d0amp_phi,data.d1amp_phi=&d1amp_phi,data.d2amp_phi=&d2amp_phi,data.d3amp_phi=&d3amp_phi,data.d4amp_phi=&d4amp_phi;
    data.solve_order=solve_order;
    data.mask_order=mask_order;
    data.printdof=100000,data.flt=&flt;
    data.magsq=1;

    int i;
    double *eps_dofAll,*eps_dofAllgrad;
    PetscScalar *eps_dof;
    eps_dofAll = (double *) malloc(dofi.ndof*ncells*sizeof(double));
    eps_dofAllgrad = (double *) malloc(dofi.ndof*ncells*sizeof(double));
    eps_dof = (PetscScalar *) malloc(dofi.ndof*sizeof(PetscScalar));
    readfromfiledouble("dof.txt",eps_dofAll,dofi.ndof*ncells);
    for(i=0;i<dofi.ndof;i++){
      eps_dof[i]=eps_dofAll[i+colour*dofi.ndof]+PETSC_i*0.0;
    }
    PetscReal amp_dof[2]={1.0,0.0};
    data.eps_dof=eps_dof,data.amp_dof=amp_dof;
    
    PetscScalar norm_cell_scalar;
    PetscReal theta_in,theta_out;
    PetscReal theta0=M_PI/4,theta1=M_PI/4+0.01,step_theta=0.01;
    PetscReal d1theta_out, d2theta_out, d3theta_out, d4theta_out;
    pwparams pw_in ={freq,nmed_in,  0,0,0,0,0};
    pwparams pw_out={freq,nmed_out, 0,0,0,0,0}; 
    PetscInt p;
    getint("-change_dof_at",&p,dofi.ndof/2);
    PetscReal s,s0,s1,ds;
    PetscReal obj_total,amp_grad[2]={0,0};
    PetscReal *_eps_dof, *_eps_grad;
    _eps_dof=(PetscReal *) malloc(dofi.ndof*sizeof(PetscReal));
    _eps_grad=(PetscReal *) malloc(dofi.ndof*sizeof(PetscReal));
    for(i=0;i<dofi.ndof;i++) _eps_dof[i]=creal(eps_dof[i]);
    for(theta_in=theta0;theta_in<theta1;theta_in+=step_theta){
      
      /*
	theta_out  =    pow(theta_in,5) -    pow(theta_in,4) +   pow(theta_in,3) -   pow(theta_in,2) + pow(theta_in,1) - 1.0;
	d1theta_out=  5*pow(theta_in,4) -  4*pow(theta_in,3) + 3*pow(theta_in,2) - 2*pow(theta_in,1) + 1.0;
	d2theta_out= 20*pow(theta_in,3) - 12*pow(theta_in,2) + 6*pow(theta_in,1) - 2.0;
	d3theta_out= 60*pow(theta_in,2) - 24*pow(theta_in,1) + 6.0;
	d4theta_out=120*pow(theta_in,1) - 24.0; 
      */
      
      theta_out   =  (M_PI/4)*sin(0.1*theta_in)*0.1;
      d1theta_out =  (M_PI/4)*cos(0.1*theta_in)*0.1;
      d2theta_out = -(M_PI/4)*sin(0.1*theta_in)*0.1;
      d3theta_out = -(M_PI/4)*cos(0.1*theta_in)*0.1;
      d4theta_out =  (M_PI/4)*sin(0.1*theta_in)*0.1;
      
      pw_in.theta_rad = theta_in;
      pw_in.d1theta = 1.0;
      pw_in.d2theta = 0.0;
      pw_in.d3theta = 0.0;
      pw_in.d4theta = 0.0;
      pw_out.theta_rad = theta_out;
      pw_out.d1theta = d1theta_out;
      pw_out.d2theta = d2theta_out;
      pw_out.d3theta = d3theta_out;
      pw_out.d4theta = d4theta_out;
      dispersiveV_2d(subcomm,dg.da, b0,b1,b2,b3,b4,usrc, planewave,&pw_in,  x_offset,&zsrc,hx,hz,1); 
      dispersiveV_2d(subcomm,dg.da, v0,v1,v2,v3,v4,u,    planewave,&pw_out, x_offset,&zref,hx,hz,1); 
      VecScale(b0,-PETSC_i*omega);
      VecScale(b1,-PETSC_i*omega);
      VecScale(b2,-PETSC_i*omega);
      VecScale(b3,-PETSC_i*omega);
      VecScale(b4,-PETSC_i*omega);
      MPI_Barrier(subcomm);
      MPI_Barrier(PETSC_COMM_WORLD);

      data.b0=b0,data.b1=b1,data.b2=b2,data.b3=b3,data.b4=b4;
      data.v0=v0,data.v1=v1,data.v2=v2,data.v3=v3,data.v4=v4,data.u=u;      
      
      VecSum(u,&norm_cell_scalar);
      data.norm_local=creal(norm_cell_scalar);
      data.norm_global=creal(norm_cell_scalar)*ncells;

      s0=0,s1=M_PI,ds=0.01;
      //s0=0,s1=1.0,ds=0.01;
      for(s=s0;s<s1;s+=ds){
	//_eps_dof[p]=s;
	//amp_dof[0]=s;
	amp_dof[1]=s;
	//eps_dofAll[p]=s;
	//obj_total=phopt_dispsum_epsglobal_globalabs(dofi.ndof*ncells,eps_dofAll,eps_dofAllgrad,&data);
	//PetscPrintf(PETSC_COMM_WORLD,"eps, dispsum, grad: %g, %.16g, %.16g\n",eps_dofAll[p],obj_total,eps_dofAllgrad[p]);
	//obj_total=phopt_dispsum_epsglobal_localabs(dofi.ndof*ncells,eps_dofAll,eps_dofAllgrad,&data);
	//PetscPrintf(PETSC_COMM_WORLD,"eps, dispsum, grad: %g, %.16g, %.16g\n",eps_dofAll[p],obj_total,eps_dofAllgrad[p]);
	//obj_total=phopt_dispsum_epscell(dofi.ndof,_eps_dof,_eps_grad,&data);
	//PetscPrintf(subcomm,"cell%d, eps, dispsum, grad: %g, %.16g, %.16g\n",colour,_eps_dof[p],obj_total,_eps_grad[p]);
	//phgrad(&data);
	//PetscPrintf(subcomm,"theta: %g, obj0: %g, obj1: %g, obj2: %g, obj3: %g, obj4: %g\n",theta_in,obj0,obj1,obj2,obj3,obj4);
	/*
	PetscPrintf(subcomm,"eps_dof[%d], obj0, grad: %g, %.16g, %.16g\n",p,creal(eps_dof[p]),obj0,creal(grad0[p])); 
	PetscPrintf(subcomm,"eps_dof[%d], obj1, grad: %g, %.16g, %.16g\n",p,creal(eps_dof[p]),obj1,creal(grad1[p])); 
	PetscPrintf(subcomm,"eps_dof[%d], obj2, grad: %g, %.16g, %.16g\n",p,creal(eps_dof[p]),obj2,creal(grad2[p])); 
	PetscPrintf(subcomm,"eps_dof[%d], obj3, grad: %g, %.16g, %.16g\n",p,creal(eps_dof[p]),obj3,creal(grad3[p])); 
	PetscPrintf(subcomm,"eps_dof[%d], obj4, grad: %g, %.16g, %.16g\n",p,creal(eps_dof[p]),obj4,creal(grad4[p])); 
	*/
	/*
	PetscPrintf(subcomm,"amp_mag, obj0, grad: %g, %.16g, %.16g\n",amp_dof[0],obj0,d0amp_mag); 
	PetscPrintf(subcomm,"amp_mag, obj1, grad: %g, %.16g, %.16g\n",amp_dof[0],obj1,d1amp_mag); 
	PetscPrintf(subcomm,"amp_mag, obj2, grad: %g, %.16g, %.16g\n",amp_dof[0],obj2,d2amp_mag); 
	PetscPrintf(subcomm,"amp_mag, obj3, grad: %g, %.16g, %.16g\n",amp_dof[0],obj3,d3amp_mag); 
	PetscPrintf(subcomm,"amp_mag, obj4, grad: %g, %.16g, %.16g\n",amp_dof[0],obj4,d4amp_mag); 
	*/
	/*
	PetscPrintf(subcomm,"amp_phi, obj0, grad: %g, %.16g, %.16g\n",amp_dof[1],obj0,d0amp_phi); 
	PetscPrintf(subcomm,"amp_phi, obj1, grad: %g, %.16g, %.16g\n",amp_dof[1],obj1,d1amp_phi); 
	PetscPrintf(subcomm,"amp_phi, obj2, grad: %g, %.16g, %.16g\n",amp_dof[1],obj2,d2amp_phi); 
	PetscPrintf(subcomm,"amp_phi, obj3, grad: %g, %.16g, %.16g\n",amp_dof[1],obj3,d3amp_phi); 
	PetscPrintf(subcomm,"amp_phi, obj4, grad: %g, %.16g, %.16g\n",amp_dof[1],obj4,d4amp_phi); 
	*/
	obj_total=phopt_dispsum_amponly(2,amp_dof,amp_grad,&data);
	PetscPrintf(PETSC_COMM_WORLD,"amp_phi, dispsum, grad: %g, %.16g, %.16g\n",amp_dof[1],obj_total,amp_grad[1]); 

      }
      
    }


  }

  PetscFinalize();
  return 0;

}

double testcons(int ndof, double *dof, double *dofgrad, void *data)
{
  double *a=(double *)data;
  int i;
  for(i=0;i<ndof;i++){
    dofgrad[i]=2;
  }
  PetscPrintf(PETSC_COMM_WORLD,"---this is testcons at step %d, taking %g\n",count,*a);

  return 1;

}
