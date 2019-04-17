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
int mma_dof;
int import_sigma;
int export_sigma;
char *export_data;
double *import_data;

#undef __FUNCT__
#define __FUNCT__ "main"
PetscErrorCode main(int argc, char **argv)
{

  MPI_Init(NULL, NULL);
  PetscInitialize(&argc,&argv,NULL,NULL);
  int size;
  MPI_Comm_size(MPI_COMM_WORLD, &size);
  PetscPrintf(PETSC_COMM_WORLD,"\tThe total number of processors is %d\n",size);
  
  int ncells_x, ncells_y, ncells;
  getint("-numcells_x",&ncells_x,2);
  getint("-numcells_y",&ncells_y,2);
  ncells=ncells_x*ncells_y;

  int ncells_per_comm;
  getint("-ncells_per_comm",&ncells_per_comm,1);
  int ncomms=ncells/ncells_per_comm;

  //check if the number of processors is divisible by the number of subcomms 
  int np_per_comm;
  if(!(size%ncomms==0)) SETERRQ(PETSC_COMM_WORLD,1,"The number of processes must be divisible by the number of subcomms.");
  np_per_comm=size/ncomms;
  PetscPrintf(PETSC_COMM_WORLD,"\tThe number of subcomms is %d.\n\tEach subcomm sequentially handles a portion of %d cells.\n\tEach cell is being simulated across %d processors.\n",ncomms,ncells_per_comm,np_per_comm);
  PetscPrintf(PETSC_COMM_WORLD,"\n\tExample:\n\
\tncells_x= 2000 (each cell is 0.5 lambda)\n\
\tncells_y= 1\n\
\tncells  = 2000\n\
\tncells_per_comm = 40\n\
\tncomms = ncells/ncells_per_comm = 2000/40 = 50\n\
\tnproc_total = 2000\n\
\tnp_per_comm = nproc_total/ncomms = 2000/50 = 40.\n");
    
  //calculate the colour of each subcomm ( = rank of each processor / number of processors in each subcomm )
  //note once calculated, the colour is fixed throughout the entire run
  int rank;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  MPI_Comm subcomm;
  int colour = rank/np_per_comm;
  MPI_Comm_split(MPI_COMM_WORLD, colour, rank, &subcomm);

  int colour_x[ncells_per_comm],colour_y[ncells_per_comm];
  int icell_comm,icell_world;
  for(icell_comm=0;icell_comm<ncells_per_comm;icell_comm++){
    icell_world=icell_comm+colour*ncells_per_comm;
    colour_x[icell_comm]=icell_world%ncells_x;
    colour_y[icell_comm]=icell_world/ncells_x;
  }
    
  //---------------
  PetscPrintf(PETSC_COMM_WORLD,"*****Large-area Topology Optimizer*****\n");
  TimeStamp ts;
  initTimeStamp(&ts);
    
  int nfreq=MAXFREQS, nangle=MAXANGLES;
  PetscReal freqs[MAXFREQS], angles_theta[MAXANGLES], angles_phi[MAXANGLES];
  getrealarray("-freqs",freqs,&nfreq,1.0);
  getrealarray("-angles_theta",angles_theta,&nangle,0.0);
  getrealarray("-angles_phi",angles_phi,&nangle,0.0);
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
  updateTimeStamp(PETSC_COMM_WORLD,&ts,"loading input epsDiff and epsBkg from h5 files");    

  //--------------
  int nget;
  int i_cell;
  PetscInt pol_in[nfreq*nangle];
  PetscReal hx,hy,hz;
  PetscReal x_offset,y_offset,zj_ref;
  PetscInt vec_verbose;
  PetscReal nsub[nfreq];
  Vec srcJ[nfreq*nangle][ncells_per_comm],u_j;
  pwparams pw_in;
  nget=nfreq*nangle;
  getintarray("-input_polz",pol_in,&nget,1);
  getreal("-hx",&hx,0.02);
  getreal("-hy",&hy,0.02);
  getreal("-hz",&hz,0.02);
  getreal("-zjref",&zj_ref,1.0);
  getint("-verbose_for_J_and_v",&vec_verbose,1);
  nget=nfreq;
  getrealarray("-nsub",nsub,&nget,1.5);
  VecDuplicate(dg.vecTemp,&u_j);
  for(i_cell=0;i_cell<ncells_per_comm;i_cell++){
    x_offset=colour_x[i_cell]*dofi.Mx*hx - dofi.Mx*hx*ncells_x/2;
    y_offset=colour_y[i_cell]*dofi.My*hy - dofi.My*hy*ncells_y/2;
    for(i=0;i<nfreq;i++){
      for(j=0;j<nangle;j++){
	jpt=j*nfreq+i;
	pw_in.freq=freqs[i];
	pw_in.nmed=nsub[i];
	pw_in.theta_rad=angles_theta[j]*M_PI/180;
	pw_in.phi_rad=angles_phi[j]*M_PI/180;
	VecDuplicate(dg.vecTemp,&srcJ[jpt][i_cell]);
	vec_at_xyslab_linpol(subcomm,dg.da, srcJ[jpt][i_cell], u_j, planewave, &pw_in, pol_in[jpt], x_offset,y_offset,&zj_ref, hx,hy,hz, vec_verbose);
      }
    }
  }

  PetscInt pol_out[nfreq*nangle];
  PetscReal zp_ref;
  PetscReal nsup[nfreq];
  Vec v[nfreq*nangle][ncells_per_comm],u;
  PetscInt voptions;
  nget=nfreq*nangle;
  getintarray("-output_polz",pol_out,&nget,1);
  getreal("-zpref",&zp_ref,3.0);
  getint("-verbose_for_J_and_FFVec",&vec_verbose,1);
  nget=nfreq;
  getrealarray("-nsup",nsup,&nget,1.0);
  VecDuplicate(dg.vecTemp,&u);
  getint("-voptions",&voptions,1);
  PetscPrintf(PETSC_COMM_WORLD,"voptions 0 [NF plane wave], 1 [FF focus 2d], 2 [FF focus 3d], 3 [NF plane wave phase error], 4 [NF lens phase]\n");
  if(voptions==0 || voptions==3){
    pwparams pw_out;
    PetscReal angles_theta_out[nfreq*nangle],angles_phi_out[nfreq*nangle];
    nget=nfreq*nangle;
    getrealarray("-angles_theta_out",angles_theta_out,&nget,0);
    nget=nfreq*nangle;
    getrealarray("-angles_phi_out",angles_phi_out,&nget,0);
    for(i_cell=0;i_cell<ncells_per_comm;i_cell++){
      x_offset=colour_x[i_cell]*dofi.Mx*hx - dofi.Mx*hx*ncells_x/2;
      y_offset=colour_y[i_cell]*dofi.My*hy - dofi.My*hy*ncells_y/2;
      for(i=0;i<nfreq;i++){
	for(j=0;j<nangle;j++){
	  jpt=j*nfreq+i;
	  pw_out.freq=freqs[i];
	  pw_out.nmed=nsup[i];
	  pw_out.theta_rad=angles_theta_out[j]*M_PI/180;
	  pw_out.phi_rad=angles_phi_out[j]*M_PI/180;
	  VecDuplicate(dg.vecTemp,&v[jpt][i_cell]);
	  vec_at_xyslab_linpol(subcomm,dg.da, v[jpt][i_cell], u, planewave, &pw_out, pol_out[jpt], x_offset,y_offset,&zp_ref, hx,hy,hz, vec_verbose);
	}
      }
    }
  }else if(voptions==1 || voptions==2 || voptions==4 ){
    ffparams ff_out;
    PetscReal x_far[nfreq*nangle],y_far[nfreq*nangle],z_far[nfreq*nangle];
    nget=nfreq*nangle;
    getrealarray("-xfar",x_far,&nget,0);
    nget=nfreq*nangle; 
    getrealarray("-yfar",y_far,&nget,0);
    nget=nfreq*nangle;
    getrealarray("-zfar",z_far,&nget,1000);
    for(i_cell=0;i_cell<ncells_per_comm;i_cell++){
      x_offset=colour_x[i_cell]*dofi.Mx*hx - dofi.Mx*hx*ncells_x/2;
      y_offset=colour_y[i_cell]*dofi.My*hy - dofi.My*hy*ncells_y/2;
      for(i=0;i<nfreq;i++){
	for(j=0;j<nangle;j++){
	  jpt=j*nfreq+i;
	  ff_out.freq=freqs[i];
	  ff_out.nmed=nsup[i];
	  ff_out.x_far=x_far[jpt];
	  ff_out.y_far=y_far[jpt];
	  ff_out.z_far=z_far[jpt];
	  ff_out.z_local=zp_ref;
	  VecDuplicate(dg.vecTemp,&v[jpt][i_cell]);
	  if(voptions==1)
	    vec_at_xyslab_linpol(subcomm,dg.da, v[jpt][i_cell], u, farfield2d, &ff_out, pol_out[jpt], x_offset,y_offset,&zp_ref, hx,hy,hz, vec_verbose);
	  else if(voptions==2)
	    vec_at_xyslab_linpol(subcomm,dg.da, v[jpt][i_cell], u, farfield3d, &ff_out, pol_out[jpt], x_offset,y_offset,&zp_ref, hx,hy,hz, vec_verbose);
	  else if(voptions==4)
	    vec_at_xyslab_linpol(subcomm,dg.da, v[jpt][i_cell], u, nearfieldlens, &ff_out, pol_out[jpt], x_offset,y_offset,&zp_ref, hx,hy,hz, vec_verbose);
	}
      }
    }

  }

  MPI_Barrier(subcomm);
  MPI_Barrier(PETSC_COMM_WORLD);
  updateTimeStamp(PETSC_COMM_WORLD,&ts,"setting up srcJ and v");    
  //------
  
  Vec mu;
  VecDuplicate(dg.vecTemp,&mu);
  VecSet(mu,1.0+PETSC_i*0.0);

  Mat M0[nfreq*nangle],Curl[nfreq*nangle];
  int maxit=15;
  Vec b[nfreq*nangle][ncells_per_comm], x[nfreq*nangle][ncells_per_comm];
  KSP ksp[nfreq*nangle][ncells_per_comm];
  PC pc[nfreq*nangle][ncells_per_comm];
  int its[nfreq*nangle][ncells_per_comm];
  DotObjData vdat[nfreq*nangle];

  int ndof_eps=dofi.ndof*ncells;
  
  PetscReal ampr, ampphi[nfreq*nangle];
  int include_amplitude_dof;
  getreal("-amp_mag",&ampr,1.0);
  nget=nfreq*nangle;
  getrealarray("-amp_phi",ampphi,&nget,0.0);
  getint("-include_amplitude_dof",&include_amplitude_dof,0);
  for(i=0;i<nfreq;i++){
    for(j=0;j<nangle;j++){
      jpt=j*nfreq+i;
      create_doublecurl_op(subcomm,M0+jpt,Curl+jpt,omega[i],mu,gi[jpt],dg);
      for(i_cell=0;i_cell<ncells_per_comm;i_cell++){
	VecDuplicate(dg.vecTemp,&b[jpt][i_cell]);
	VecCopy(srcJ[jpt][i_cell],b[jpt][i_cell]);
	VecScale(b[jpt][i_cell],-PETSC_i*omega[i]);
	VecDuplicate(dg.vecTemp,&x[jpt][i_cell]);
	setupKSPDirect(subcomm,&ksp[jpt][i_cell],&pc[jpt][i_cell],maxit);
	its[jpt][i_cell]=1000;
      }
      MPI_Barrier(subcomm);
      MPI_Barrier(PETSC_COMM_WORLD);
      updateTimeStamp(PETSC_COMM_WORLD,&ts,"setting up curlxcurl operator, b, x and ksp");    

      (vdat+jpt)->colour=colour;
      (vdat+jpt)->ncells_per_comm=ncells_per_comm;
      (vdat+jpt)->subcomm=subcomm;
      (vdat+jpt)->ndof_eps=ndof_eps;
      (vdat+jpt)->ID=jpt;
      (vdat+jpt)->M0=M0[jpt];
      (vdat+jpt)->b=b[jpt];
      (vdat+jpt)->x=x[jpt];
      (vdat+jpt)->epsDiff=epsDiff[i];
      (vdat+jpt)->epsBkg=epsBkg[i];
      (vdat+jpt)->omega=omega[i];
      (vdat+jpt)->v=v[jpt];
      (vdat+jpt)->u=u;
      (vdat+jpt)->ampr=&ampr;
      (vdat+jpt)->ampphi=ampphi+jpt;
      if(voptions==0) (vdat+jpt)->func= (include_amplitude_dof) ? phoptmulti : phopt;
      if(voptions==1) (vdat+jpt)->func= ffopt;
      if(voptions==2) (vdat+jpt)->func= ffopt;
      if(voptions==3) (vdat+jpt)->func= (include_amplitude_dof) ? pherroptmulti : pherropt;
      if(voptions==4) (vdat+jpt)->func= (include_amplitude_dof) ? phoptmulti : phopt;
      if(voptions>4) (vdat+jpt)->func=phopt;
      if(voptions<0) (vdat+jpt)->func=phopt;
      (vdat+jpt)->ksp=ksp[jpt];
      (vdat+jpt)->its=its[jpt];
      (vdat+jpt)->maxit=maxit;
      (vdat+jpt)->dofi=&dofi;
      (vdat+jpt)->dg=dg;
      (vdat+jpt)->printdof=printdof;
      (vdat+jpt)->flt=&flt;
      
    }
  }

  //setup MMA
  int exportsigma;
  char tmp_filename[PETSC_MAX_PATH_LEN];
  getint("-mma_import",&import_sigma,0);
  getint("-mma_export",&exportsigma,1);

  int myrank;
  MPI_Comm_rank(MPI_COMM_WORLD,&myrank);
  export_sigma=0;
  if(myrank==0) {
    mma_verbose=1;
    export_sigma=exportsigma;
    getstr("-mma_sigma_file",tmp_filename,"mma_latest.txt");
    export_data=tmp_filename;
  }

  if(import_sigma){
    int sigma_size;
    getint("-mma_size",&sigma_size,1);
    getstr("-mma_sigma_file",tmp_filename,"mma_latest.txt");
    import_data = (double *) malloc(sigma_size*sizeof(double));
    
    FILE *fp;
    fp = fopen(tmp_filename,"r");
    for (i=0;i<sigma_size;i++) fscanf(fp,"%lf",&import_data[i]);
    fclose(fp);
  }

  MPI_Barrier(subcomm);
  MPI_Barrier(PETSC_COMM_WORLD);  
  //setup MMA

  int ndofAll;
  double *dofAll;
  getstr("-init_dof_name",tmpstr1,"dof.txt");
  if(include_amplitude_dof){
    if(voptions==0 || voptions==4){
      ndofAll=ndof_eps+nfreq*nangle;
      dofAll=(double *) malloc(ndofAll*sizeof(double));
      readfromfiledouble(tmpstr1,dofAll,ndof_eps);
      for(i=0;i<nfreq;i++){
	for(j=0;j<nangle;j++){
	  jpt=j*nfreq+i;
	  dofAll[ndof_eps+jpt]=ampphi[jpt];
	}
      }
    }else if(voptions==3){
      ndofAll=ndof_eps+1+nfreq*nangle;
      dofAll=(double *) malloc(ndofAll*sizeof(double));
      readfromfiledouble(tmpstr1,dofAll,ndof_eps);
      dofAll[ndof_eps]=ampr;
      for(i=0;i<nfreq;i++){
	for(j=0;j<nangle;j++){
	  jpt=j*nfreq+i;
	  dofAll[ndof_eps+1+jpt]=ampphi[jpt];
	}
      }
    }else{
      ndofAll=ndof_eps;
      dofAll=(double *) malloc(ndofAll*sizeof(double));
      readfromfiledouble(tmpstr1,dofAll,ndofAll);
    }
  }else{
    ndofAll=ndof_eps;
    dofAll=(double *) malloc(ndofAll*sizeof(double));
    readfromfiledouble(tmpstr1,dofAll,ndofAll);
  }

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
    PetscScalar *dof[ncells_per_comm],*_dof[ncells_per_comm];
    int i;
    Vec eps;

    Mat Mfull;
    MatDuplicate(M0[jpt],MAT_COPY_VALUES,&Mfull);
    for(i_cell=0;i_cell<ncells_per_comm;i_cell++){
      dof[i_cell]  = (PetscScalar *) malloc(ndof*sizeof(PetscScalar));
      _dof[i_cell] = (PetscScalar *) malloc(ndof*sizeof(PetscScalar));
      for(i=0;i<ndof;i++){
	icell_world=i_cell+colour*ncells_per_comm;
	dof[i_cell][i]=dofAll[i+icell_world*ndof]+PETSC_i*0.0;
      }
      filters_apply(subcomm,dof[i_cell],_dof[i_cell],&flt,1);
      VecDuplicate(dg.vecTemp,&eps);
      multilayer_forward(_dof[i_cell],eps,&dofi,dg.da);
      VecPointwiseMult(eps,eps,epsDiff[ifreq]);
      VecAXPY(eps,1.0,epsBkg[ifreq]);
      sprintf(tmpstr1,"epsilon_cellx%03d_celly%03d.h5",colour_x[i_cell],colour_y[i_cell]);
      saveVecHDF5(subcomm,eps,tmpstr1,"eps");

      sprintf(tmpstr1,"srcJ_cellx%03d_celly%03d.h5",colour_x[i_cell],colour_y[i_cell]);
      saveVecHDF5(subcomm,srcJ[jpt][i_cell],tmpstr1,"srcJ");
      sprintf(tmpstr1,"v_cellx%03d_celly%03d.h5",colour_x[i_cell],colour_y[i_cell]);
      saveVecHDF5(subcomm,v[jpt][i_cell],tmpstr1,"v");
      MPI_Barrier(subcomm);
      MPI_Barrier(PETSC_COMM_WORLD);
    
      if(print_Efields){
	VecScale(eps,-omega[ifreq]*omega[ifreq]);
	MatDiagonalSet(Mfull,eps,ADD_VALUES);
	SolveMatrixDirect(subcomm,ksp[jpt][i_cell],Mfull,b[jpt][i_cell],x[jpt][i_cell],&its[jpt][i_cell],maxit);
	sprintf(tmpstr1,"Efield_cellx%03d_celly%03d.h5",colour_x[i_cell],colour_y[i_cell]);
	saveVecHDF5(subcomm,x[jpt][i_cell],tmpstr1,"E");
	MPI_Barrier(subcomm);
	MPI_Barrier(PETSC_COMM_WORLD);
      }      
    }

    for(i_cell=0;i_cell<ncells_per_comm;i_cell++){
      free(dof[i_cell]);
      free(_dof[i_cell]);
    }
    MatDestroy(&Mfull);
    VecDestroy(&eps);
    
  }else if(Job==1){

    int opt_nfreq=nfreq, opt_nangle=nangle;
    int ifreq[nfreq], jangle[nangle];
    getintarray("-opt_ifreq",ifreq,&opt_nfreq,0);
    getintarray("-opt_jangle",jangle,&opt_nangle,0);
    int opt_num=opt_nfreq*opt_nangle;

    if(opt_num==1){

      mma_dof=ndofAll;
          
      jpt=jangle[0]*nfreq+ifreq[0];
      double *lb,*ub;
      lb=(double *) malloc(ndofAll*sizeof(double));
      ub=(double *) malloc(ndofAll*sizeof(double));
      make_array(lb,0.0,ndof_eps);
      make_array(ub,1.0,ndof_eps);
      if(include_amplitude_dof){
	if(voptions==0 || voptions==4){
	  for(i=ndof_eps;i<ndofAll;i++){
	    lb[i]=ampphi[i-ndof_eps];
	    ub[i]=ampphi[i-ndof_eps];      
	  }
	  getreal("-lb_ampphi",lb+ndof_eps+jpt,0);
	  getreal("-ub_ampphi",ub+ndof_eps+jpt,2*M_PI);
	}else if(voptions==3){
	  lb[ndof_eps]=ampr;
	  ub[ndof_eps]=ampr;
	  for(i=ndof_eps+1;i<ndofAll;i++){
	    lb[i]=ampphi[i-ndof_eps-1];
	    ub[i]=ampphi[i-ndof_eps-1];      
	  }
	  getreal("-lb_ampmag",lb+ndof_eps,0.5);
	  getreal("-ub_ampmag",ub+ndof_eps,1.5);
	  getreal("-lb_ampphi",lb+ndof_eps+1+jpt,0);
	  getreal("-ub_ampphi",ub+ndof_eps+1+jpt,2*M_PI);
	}
      }
      if(voptions==3)
	optimize_generic(ndofAll, dofAll, lb,ub, vdat+jpt, NULL, (vdat+jpt)->func, NULL, 0, 0);
      else
	optimize_generic(ndofAll, dofAll, lb,ub, vdat+jpt, NULL, (vdat+jpt)->func, NULL, 1, 0);

      free(lb);
      free(ub);
      
    }else{

      double dummy_var;
      getreal("-init_dummy_var",&dummy_var,0.01);
      int ntot=ndofAll+1;
      double *doftot, *lbtot, *ubtot;
      doftot = (double *) malloc(ntot*sizeof(double));
      lbtot = (double *) malloc(ntot*sizeof(double));
      ubtot = (double *) malloc(ntot*sizeof(double));
      for(i=0;i<ndofAll;i++){
	doftot[i]=dofAll[i];
	lbtot[i]=0;
	ubtot[i]=1;
      }
      doftot[ntot-1]=dummy_var;
      lbtot[ntot-1]=0;
      ubtot[ntot-1]=1.0/0.0;

      double lbphi[opt_num], ubphi[opt_num];
      if(include_amplitude_dof){
	if(voptions==0 || voptions==4){
	  for(i=ndof_eps;i<ndofAll;i++){
	    lbtot[i]=ampphi[i-ndof_eps];
	    ubtot[i]=ampphi[i-ndof_eps];      
	  }
	  nget=opt_num;
	  getrealarray("-lb_ampphi",lbphi,&nget,0.0);
	  nget=opt_num;
	  getrealarray("-ub_ampphi",ubphi,&nget,1.0);
	}else if(voptions==3){
	  lbtot[ndof_eps]=ampr;
	  ubtot[ndof_eps]=ampr;
	  for(i=ndof_eps+1;i<ndofAll;i++){
	    lbtot[i]=ampphi[i-ndof_eps-1];
	    ubtot[i]=ampphi[i-ndof_eps-1];      
	  }
	  getreal("-lb_ampmag",lbtot+ndof_eps,0.5);
	  getreal("-ub_ampmag",ubtot+ndof_eps,1.5);
	  nget=opt_num;
	  getrealarray("-lb_ampphi",lbphi,&nget,0.0);
	  nget=opt_num;
	  getrealarray("-ub_ampphi",ubphi,&nget,1.0);
	}
      }


      mma_dof=ntot;
      
      DotObjData *data[opt_num];
      optfunc funcs[opt_num];
      int id;
      for(i=0;i<opt_nfreq;i++){
	for(j=0;j<opt_nangle;j++){
	jpt=jangle[j]*nfreq+ifreq[i];
	id=j*opt_nfreq+i;
	data[id]=vdat+jpt;

	if(include_amplitude_dof){
	  if(voptions==0 || voptions==4){
	    *(lbtot+ndof_eps+jpt)=lbphi[id];
	    *(ubtot+ndof_eps+jpt)=ubphi[id];
	  }else if(voptions==3){
	    *(lbtot+ndof_eps+1+jpt)=lbphi[id];
	    *(ubtot+ndof_eps+1+jpt)=ubphi[id];
	  }
	}

	if(voptions==3)
	  funcs[id]=minimaxconstraint;
	else
	  funcs[id]=maximinconstraint;
	} 
      }


      if(voptions==3)
	optimize_generic(ntot, doftot, lbtot,ubtot, NULL, data, dummy_obj, funcs, 0, opt_num);
      else
	optimize_generic(ntot, doftot, lbtot,ubtot, NULL, data, dummy_obj, funcs, 1, opt_num);

      free(doftot);
      free(lbtot);
      free(ubtot);

      }

  }else{

    //-----compute-----
    int ifreq, jangle;
    getint("-test_ifreq",&ifreq,0);
    getint("-test_jangle",&jangle,0);
    jpt=jangle*nfreq+ifreq;

    int ntot=ndofAll+1;
    double *doftot,*dofgradtot;
    doftot = (double *) malloc(ntot*sizeof(double));
    dofgradtot = (double *) malloc(ntot*sizeof(double));
    for(i=0;i<ntot-1;i++){
      doftot[i]=dofAll[i];
    }
    doftot[ntot-1]=0.01;

    int p;
    double s,s0,ds,s1;
    PetscReal objval;
    getint("-change_dof_at",&p,dofi.ndof/2);
    getreal("-change_s0",&s0,0.0);
    getreal("-change_s1",&s1,1.0);
    getreal("-change_ds",&ds,0.01);
    for(s=s0;s<s1;s+=ds){
      doftot[p]=s;
      if(voptions==3)
	objval=minimaxconstraint(ntot,doftot,dofgradtot,vdat+jpt);
      else
	objval=maximinconstraint(ntot,doftot,dofgradtot,vdat+jpt);
      PetscPrintf(PETSC_COMM_WORLD,"objval: %g, %.24g, %.24g\n",doftot[p],objval,dofgradtot[p]);
    }

    MPI_Barrier(subcomm);
    MPI_Barrier(PETSC_COMM_WORLD);
    
    free(doftot);
    free(dofgradtot);
  }

  VecDestroy(&dg.vecTemp);
  VecDestroy(&mu);
  DMDestroy(&dg.da);

  VecDestroy(&u);
  VecDestroy(&u_j);
  for(i=0;i<nfreq;i++){
    VecDestroy(epsDiff+i);
    VecDestroy(epsBkg+i);
    for(j=0;j<nangle;j++){
      jpt=j*nfreq+i;
      for(i_cell=0;i_cell<ncells_per_comm;i_cell++){
	VecDestroy(&srcJ[jpt][i_cell]);
	VecDestroy(&v[jpt][i_cell]);
	VecDestroy(&b[jpt][i_cell]);
	VecDestroy(&x[jpt][i_cell]);  
	KSPDestroy(&ksp[jpt][i_cell]);
      }
      MatDestroy(M0+jpt);
      MatDestroy(Curl+jpt);
    }
  }
  MatDestroy(&flt.W);
  VecDestroy(&flt.rho_grad);
  free(dofAll);
  
  PetscFinalize();
  return 0;

}

