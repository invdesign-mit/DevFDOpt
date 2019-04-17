#include "petsc.h"
#include "petscsys.h"
#include "hdf5.h"
#include "nlopt.h"
#include <assert.h>
#include "libFDOPT.h"

int count=0;
const int MAXFREQS=1000;
const int MAXANGLES=1000;
int mma_verbose;
TimeStamp global_time;

#undef __FUNCT__
#define __FUNCT__ "main"
PetscErrorCode main(int argc, char **argv)
{

  MPI_Init(NULL, NULL);
  PetscInitialize(&argc,&argv,NULL,NULL);
  int size;
  MPI_Comm_size(MPI_COMM_WORLD, &size);
  PetscPrintf(PETSC_COMM_WORLD,"\tThe total number of processors is %d\n",size);
  
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
    
  int myrank;
  MPI_Comm_rank(MPI_COMM_WORLD,&myrank);
  if(myrank==0)
    mma_verbose=1;

  /*---------------*/
  PetscPrintf(PETSC_COMM_WORLD,"*****Large-area Optimizer for Far Field*****\n");
  TimeStamp ts;
  initTimeStamp(&ts);
    
  int nfreq=MAXFREQS, nangle=MAXANGLES;
  PetscReal freqs[MAXFREQS], angles[MAXANGLES];
  getrealarray("-freqs",freqs,&nfreq,1.0);
  getrealarray("-angles",angles,&nangle,0.0);
  PetscScalar omega[nfreq];

  int printdof;
  getint("-print_dof",&printdof,1);

  int i,j,jpt;
  GridInfo gi[nfreq*nangle];
  char tmpstr1[PETSC_MAX_PATH_LEN];
  char tmpstr2[PETSC_MAX_PATH_LEN];
  getstr("-inputfile_name",tmpstr1,"domain");
  for(i=0;i<nfreq;i++){
    omega[i]=2*M_PI*freqs[i]+PETSC_i*0.0;
    for(j=0;j<nangle;j++){
      jpt=j*nfreq+i;
      sprintf(tmpstr2,"%s_freq%d_angle%d.h5",tmpstr1,i,j);
      setGridInfo(subcomm,tmpstr2,gi+jpt);
    }
  }
  MPI_Barrier(subcomm);
  MPI_Barrier(PETSC_COMM_WORLD);

  DOFInfo dofi;
  ParDataGrid dg;
  FiltersToolBox flt;
  sprintf(tmpstr2,"%s_freq0_angle0.h5",tmpstr1);
  setDOFInfo(subcomm,tmpstr2,&dofi);
  setParDataGrid(subcomm,&dg,gi[0]);
  filters_initialize(subcomm,&flt,dofi);
  
  MPI_Barrier(subcomm);
  MPI_Barrier(PETSC_COMM_WORLD);
  updateTimeStamp(PETSC_COMM_WORLD,&ts,"setting up grid, pardata, dof and filter infos");    

  //---epsilon inputs---
  Vec epsDiff[nfreq], epsBkg[nfreq];
  for(i=0;i<nfreq;i++){
    sprintf(tmpstr2,"%s_freq%d_epsDiff.h5",tmpstr1,i);
    VecDuplicate(dg.vecTemp,epsDiff+i);
    VecSet(epsDiff[i],0.0);
    loadVecHDF5(subcomm,epsDiff[i],tmpstr2,"/epsDiff");

    sprintf(tmpstr2,"%s_freq%d_epsBkg.h5",tmpstr1,i);
    VecDuplicate(dg.vecTemp,epsBkg+i);
    VecSet(epsBkg[i],0.0);
    loadVecHDF5(subcomm,epsBkg[i],tmpstr2,"/epsBkg");
  }

  MPI_Barrier(subcomm);
  MPI_Barrier(PETSC_COMM_WORLD);
  updateTimeStamp(PETSC_COMM_WORLD,&ts,"loading input epsDiff and epsBkg from h5 files, dofAll from text file");    

  //--------------
  int nget=MAXFREQS*MAXANGLES;
  PetscReal nsub[nfreq];
  PetscReal ff_x[nfreq*nangle], ff_z[nfreq*nangle];
  PetscReal x_offset,zp_ref,zj_ref,hx,hz;
  PetscScalar amp=1.0+PETSC_i*0;
  PetscInt vec_verbose;
  getrealarray("-nsub",nsub,&nget,1.5);
  nget=MAXFREQS*MAXANGLES;
  getrealarray("-ffx",ff_x,&nget,0);
  nget=MAXFREQS*MAXANGLES;
  getrealarray("-ffz",ff_z,&nget,100000);

  getreal("-zjref",&zj_ref,1.0);
  getreal("-zpref",&zp_ref,3.0);
  getreal("-hx",&hx,0.02);
  getreal("-hz",&hz,0.02);
  getint("-verbose_for_J_and_FFVec",&vec_verbose,1);
  x_offset=colour*dofi.Mx*hx - dofi.Mx*hx*ncells/2;

  MPI_Barrier(subcomm);
  MPI_Barrier(PETSC_COMM_WORLD);

  Vec srcJ[nfreq*nangle],ffvec[nfreq*nangle];
  for(i=0;i<nfreq;i++){
    for(j=0;j<nangle;j++){
      jpt=j*nfreq+i;
      VecDuplicate(dg.vecTemp,srcJ+jpt);
      VecDuplicate(dg.vecTemp,ffvec+jpt);
      pwsrc_2dxz(subcomm,dg.da,srcJ[jpt],freqs[i],nsub[i],angles[j],amp,x_offset,&zj_ref,hx,hz,vec_verbose);
      ff2dxz(subcomm,dg.da,ffvec[jpt],ff_x[jpt],ff_z[jpt],freqs[i],x_offset,&zp_ref,hx,hz,vec_verbose);
    }
  }
  MPI_Barrier(subcomm);
  MPI_Barrier(PETSC_COMM_WORLD);
  updateTimeStamp(PETSC_COMM_WORLD,&ts,"setting up srcJ and farfield convoluter");    
  //------

  
  Vec mu;
  VecDuplicate(dg.vecTemp,&mu);
  VecSet(mu,1.0+PETSC_i*0.0);

  Mat M0[nfreq*nangle],Curl[nfreq*nangle];
  Vec b[nfreq*nangle], x[nfreq*nangle];
  KSP ksp[nfreq*nangle];
  PC pc[nfreq*nangle];
  int maxit=15;
  int its[nfreq*nangle];
  Farfielddata ffdat[nfreq*nangle];

  for(i=0;i<nfreq;i++){
    for(j=0;j<nangle;j++){
      jpt=j*nfreq+i;
      create_doublecurl_op(subcomm,M0+jpt,Curl+jpt,omega[i],mu,gi[jpt],dg);
      VecDuplicate(dg.vecTemp,b+jpt);
      VecCopy(srcJ[jpt],b[jpt]);
      VecScale(b[jpt],-PETSC_i*omega[i]);
      VecDuplicate(dg.vecTemp,x+jpt);
      setupKSPDirect(subcomm,ksp+jpt,pc+jpt,maxit);
      MPI_Barrier(subcomm);
      MPI_Barrier(PETSC_COMM_WORLD);
      updateTimeStamp(PETSC_COMM_WORLD,&ts,"setting up curlxcurl operator, b, x and ksp");    

      its[jpt]=1000;
      (ffdat+jpt)->colour=colour;
      (ffdat+jpt)->subcomm=subcomm;
      (ffdat+jpt)->M0=M0[jpt];
      (ffdat+jpt)->b=b[jpt];
      (ffdat+jpt)->x=x[jpt];
      (ffdat+jpt)->epsDiff=epsDiff[i];
      (ffdat+jpt)->epsBkg=epsBkg[i];
      (ffdat+jpt)->omega=omega[i];
      (ffdat+jpt)->ffvec=ffvec[jpt];
      (ffdat+jpt)->ksp=ksp[jpt];
      (ffdat+jpt)->its=its+jpt;
      (ffdat+jpt)->maxit=maxit;
      (ffdat+jpt)->dofi=&dofi;
      (ffdat+jpt)->dg=dg;
      (ffdat+jpt)->printdof=printdof;
      (ffdat+jpt)->flt=&flt;
      
    }
  }

  double *dofAll;
  dofAll = (double *) malloc(dofi.ndof*ncells*sizeof(double));
  getstr("-init_dof_name",tmpstr1,"dof.txt");
  readfromfiledouble(tmpstr1,dofAll,dofi.ndof*ncells);

  initTimeStamp(&global_time);

  int Job;
  getint("-Job",&Job,1);
  if(Job==0){
    
    int ifreq, jangle;
    getint("-print_ifreq",&ifreq,0);
    getint("-print_jangle",&jangle,0);
    jpt=jangle*nfreq+ifreq;

    int print_Efields;
    getint("-print_Efields",&print_Efields,0);
    
    int ndof=flt.ndof;
    PetscScalar *dof,*_dof;
    int i;
    Vec eps;
    
    dof  = (PetscScalar *) malloc(ndof*sizeof(PetscScalar));
    _dof = (PetscScalar *) malloc(ndof*sizeof(PetscScalar));
    for(i=0;i<ndof;i++) dof[i]=dofAll[i+colour*ndof]+PETSC_i*0.0;
    filters_apply(subcomm,dof,_dof,&flt,1);
    //
    int subrank;
    MPI_Comm_rank(subcomm, &subrank);
    double *tmp_dofAll,*_dofAll;
    int ndofAll=dofi.ndof*ncells;
    tmp_dofAll = (double *) malloc(ndofAll*sizeof(double));
    _dofAll = (double *) malloc(ndofAll*sizeof(double));
    for(i=0;i<ndofAll;i++){
      if(subrank==0 && i>=colour*ndof && i<colour*ndof+ndof)
	tmp_dofAll[i]= creal(_dof[i-colour*ndof]);
      else
	tmp_dofAll[i] = 0.0;
    }
    MPI_Allreduce(tmp_dofAll,_dofAll,ndofAll,MPI_DOUBLE,MPI_SUM,PETSC_COMM_WORLD);
    writetofiledouble(PETSC_COMM_WORLD,"filtered_dof.txt",_dofAll,ndofAll);
    free(tmp_dofAll);
    free(_dofAll);
    //
    VecDuplicate(dg.vecTemp,&eps);
    multilayer_forward(_dof,eps,&dofi,dg.da);
    VecPointwiseMult(eps,eps,epsDiff[ifreq]);
    VecAXPY(eps,1.0,epsBkg[ifreq]);
    sprintf(tmpstr1,"epsilon_cell%03d.h5",colour);
    saveVecHDF5(subcomm,eps,tmpstr1,"eps");
    MPI_Barrier(subcomm);
    MPI_Barrier(PETSC_COMM_WORLD);

    if(print_Efields){
      VecScale(eps,-omega[ifreq]*omega[ifreq]);
      MatDiagonalSet(M0[jpt],eps,ADD_VALUES);
      SolveMatrixDirect(subcomm,ksp[jpt],M0[jpt],b[jpt],x[jpt],its+jpt,maxit);
      MPI_Barrier(subcomm);
      MPI_Barrier(PETSC_COMM_WORLD);
      sprintf(tmpstr1,"Efield_cell%03d.h5",colour);
      saveVecHDF5(subcomm,x[jpt],tmpstr1,"E");
      MPI_Barrier(subcomm);
      MPI_Barrier(PETSC_COMM_WORLD);
    }
    
    VecDestroy(&eps);
    free(dof);
    free(_dof);
    
  }else if(Job==1){
    
    int opt_nfreq=nfreq, opt_nangle=nangle;
    int ifreq[nfreq], jangle[nangle];
    getintarray("-opt_ifreq",ifreq,&opt_nfreq,0);
    getintarray("-opt_jangle",jangle,&opt_nangle,0);
    int opt_num=opt_nfreq*opt_nangle;

    if(opt_num==1){
      
      jpt=jangle[0]*nfreq+ifreq[0];
      double *lb,*ub;
      lb=(double *) malloc(dofi.ndof*ncells*sizeof(double));
      ub=(double *) malloc(dofi.ndof*ncells*sizeof(double));
      make_array(lb,0.0,dofi.ndof*ncells);
      make_array(ub,1.0,dofi.ndof*ncells);
      optimize_generic(dofi.ndof*ncells, dofAll, lb,ub, ffdat+jpt, NULL, ffopt, NULL, 1, 0);

      free(lb);
      free(ub);
      
    }else{

      double dummy_var;
      getreal("-init_dummy_var",&dummy_var,0.01);
      int ntot=dofi.ndof*ncells+1;
      double *doftot, *lbtot, *ubtot;
      doftot = (double *) malloc(ntot*sizeof(double));
      lbtot = (double *) malloc(ntot*sizeof(double));
      ubtot = (double *) malloc(ntot*sizeof(double));
      for(i=0;i<ntot-1;i++){
	doftot[i]=dofAll[i];
	lbtot[i]=0;
	ubtot[i]=1;
      }
      doftot[ntot-1]=dummy_var;
      lbtot[ntot-1]=0;
      ubtot[ntot-1]=1.0/0.0;

      Farfielddata *data[opt_num];
      optfunc funcs[opt_num];
      int id;
      for(i=0;i<opt_nfreq;i++){
	for(j=0;j<opt_nangle;j++){
	jpt=jangle[j]*nfreq+ifreq[i];
	id=j*opt_nfreq+i;
	data[id]=ffdat+jpt;
	funcs[id]=ffopt_maximinconstraint;
	} 
      }
      optimize_generic(ntot, doftot, lbtot,ubtot, NULL, data, dummy_obj, funcs, 1, opt_num);

      free(doftot);
      free(lbtot);
      free(ubtot);

      }

  }else{

    /*-----compute-----*/
    int ifreq, jangle;
    getint("-test_ifreq",&ifreq,0);
    getint("-test_jangle",&jangle,0);
    jpt=jangle*nfreq+ifreq;

    int ntot=dofi.ndof*ncells+1;
    double *doftot,*dofgradtot;
    doftot = (double *) malloc(ntot*sizeof(double));
    dofgradtot = (double *) malloc(ntot*sizeof(double));
    for(i=0;i<ntot-1;i++){
      doftot[i]=dofAll[i];
    }
    doftot[ntot-1]=0.01;

    int p;
    double s,s0=0,ds=0.01,s1=1.0;
    PetscReal xfarsq;
    getint("-change_dof_at",&p,dofi.ndof/2);
    for(s=s0;s<s1;s+=ds){
      doftot[p]=s;
      xfarsq=ffopt_maximinconstraint(ntot,doftot,dofgradtot,ffdat+jpt);
      PetscPrintf(PETSC_COMM_WORLD,"xfarsq: %g, %g, %g\n",doftot[p],xfarsq,dofgradtot[p]);
    }

    free(doftot);
    free(dofgradtot);
  }


  VecDestroy(&dg.vecTemp);
  VecDestroy(&mu);
  DMDestroy(&dg.da);
  
  for(i=0;i<nfreq;i++){
    VecDestroy(epsDiff+i);
    VecDestroy(epsBkg+i);
    for(j=0;j<nangle;j++){
      jpt=j*nfreq+i;
      VecDestroy(srcJ+jpt);
      VecDestroy(ffvec+jpt);
      VecDestroy(b+jpt);
      VecDestroy(x+jpt);  
      MatDestroy(M0+jpt);
      MatDestroy(Curl+jpt);
      KSPDestroy(ksp+jpt);
    }
  }
  MatDestroy(&flt.W);
  VecDestroy(&flt.rho_grad);
  free(dofAll);

  PetscFinalize();
  return 0;

}

