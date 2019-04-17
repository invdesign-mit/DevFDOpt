#include "petsc.h"
#include "petscsys.h"
#include "hdf5.h"
#include "nlopt.h"
#include <assert.h>
#include "libFDOPT.h"

int count=0;
int mma_verbose;

void angular_disp_linear(PetscReal theta_in, const PetscReal *anglemap, pwparams* pw);

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
  PetscPrintf(PETSC_COMM_WORLD,"*****Large-area Optimizer*****\n");
  TimeStamp ts;
  initTimeStamp(&ts);
    
  GridInfo gi;
  DOFInfo dofi;
  char tmpstr[PETSC_MAX_PATH_LEN];
  char tmpsubstr[PETSC_MAX_PATH_LEN];
  getstr("-inputfile_name",tmpstr,"domain.h5");
  setGridInfo(tmpstr,&gi);
  setDOFInfo(tmpstr,&dofi);
  PetscPrintf(PETSC_COMM_WORLD,"Note: since there are %d unit cells, each with %d dofs, there are %d dofs in total.\n",ncells,dofi.ndof,dofi.ndof*ncells);

  ParDataGrid dg;
  setParDataGrid(subcomm,&dg,gi);

  FiltersToolBox flt;
  filters_initialize(subcomm,&flt,dofi);
  
  MPI_Barrier(subcomm);
  MPI_Barrier(PETSC_COMM_WORLD);
  updateTimeStamp(PETSC_COMM_WORLD,&ts,"setting up grid, pardata, dof and filter infos");    

  /*---epsilon inputs---*/
  Vec epsDiff, epsBkg;
  VecDuplicate(dg.vecTemp,&epsDiff);
  VecSet(epsDiff,0.0);
  getstr("-epsDiff_h5_name",tmpstr,"epsDiff.h5");
  getstr("-epsDiff_dset_name",tmpsubstr,"/epsDiff");
  loadVecHDF5(subcomm,epsDiff,tmpstr,tmpsubstr);

  VecDuplicate(dg.vecTemp,&epsBkg);
  VecSet(epsBkg,0.0);
  getstr("-epsBkg_h5_name",tmpstr,"epsBkg.h5");
  getstr("-epsBkg_dset_name",tmpsubstr,"/epsBkg");
  loadVecHDF5(subcomm,epsBkg,tmpstr,tmpsubstr);  

  MPI_Barrier(subcomm);
  MPI_Barrier(PETSC_COMM_WORLD);
  updateTimeStamp(PETSC_COMM_WORLD,&ts,"loading input epsDiff and epsBkg from h5 files, dofAll from text file");    
    
  /*------*/
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
  updateTimeStamp(PETSC_COMM_WORLD,&ts,"setting up curlxcurl operator and ksp");    
  /*------*/
  Phasefielddata data;

  Vec v0,v1,v2,v3,v4,u;
  Vec b0,b1,b2,b3,b4,usrc;
  Vec x0,x1,x2,x3,x4;
  PetscReal obj0=0;
  PetscReal obj1=0;
  PetscReal obj2=0;
  PetscReal obj3=0;
  PetscReal obj4=0;
  PetscScalar *grad0;
  PetscScalar *grad1;
  PetscScalar *grad2;
  PetscScalar *grad3;
  PetscScalar *grad4;
  PetscScalar *eps_dof;
  PetscReal d0amp_mag=0;
  PetscReal d1amp_mag=0;
  PetscReal d2amp_mag=0;
  PetscReal d3amp_mag=0;
  PetscReal d4amp_mag=0;
  PetscReal d0amp_phi=0;
  PetscReal d1amp_phi=0;
  PetscReal d2amp_phi=0;
  PetscReal d3amp_phi=0;
  PetscReal d4amp_phi=0;
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
  grad0 = (PetscScalar *) malloc(dofi.ndof*sizeof(PetscScalar));
  grad1 = (PetscScalar *) malloc(dofi.ndof*sizeof(PetscScalar));
  grad2 = (PetscScalar *) malloc(dofi.ndof*sizeof(PetscScalar));
  grad3 = (PetscScalar *) malloc(dofi.ndof*sizeof(PetscScalar));
  grad4 = (PetscScalar *) malloc(dofi.ndof*sizeof(PetscScalar));
  eps_dof = (PetscScalar *) malloc(dofi.ndof*sizeof(PetscScalar));
  PetscReal amp_dof[2]={1.0,0.0};
  PetscReal tmpamp;
  getreal("-amp_mag",&tmpamp,1.0);
  amp_dof[0]=tmpamp;
  getreal("-amp_phi",&tmpamp,0.0);
  amp_dof[1]=tmpamp;

  PetscReal x_offset,zref,zsrc,nmed_in,nmed_out,hx,hz;
  getreal("-zsrc",&zsrc,1.0);
  getreal("-zref",&zref,3.0);
  getreal("-nmed_in",&nmed_in,1.5);
  getreal("-nmed_out",&nmed_out,1.0);
  getreal("-hx",&hx,0.02);
  getreal("-hz",&hz,0.02);
  x_offset=colour*dofi.Mx*hx-dofi.Mx*hx*ncells/2;
  PetscReal theta_in;
  PetscInt nget=4;
  PetscReal anglemap[4]={0,0,M_PI/2,M_PI/2};
  getreal("-incident_angle(radian)",&theta_in,0);
  getrealarray("-angle_map(in,out)",anglemap,&nget,0);
  pwparams pw_in ={freq,nmed_in,  theta_in,1.0,0,0,0};
  pwparams pw_out={freq,nmed_out, 0,0,0,0,0};
  angular_disp_linear(theta_in,anglemap,&pw_out);
  PetscInt solve_order[5]={1,0,0,0,0};
  PetscReal mask_order[5]={1,0,0,0,0};
  nget=5;
  getintarray("-solve_order",solve_order,&nget,1);
  getrealarray("-mask_order",mask_order,&nget,1.0);
  MPI_Barrier(subcomm);
  MPI_Barrier(PETSC_COMM_WORLD);

  dispersiveV_2d(subcomm,dg.da, b0,b1,b2,b3,b4,usrc, planewave,&pw_in,  x_offset,&zsrc,hx,hz,1);
  dispersiveV_2d(subcomm,dg.da, v0,v1,v2,v3,v4,u,    planewave,&pw_out, x_offset,&zref,hx,hz,1);
  VecScale(b0,-PETSC_i*omega);
  VecScale(b1,-PETSC_i*omega);
  VecScale(b2,-PETSC_i*omega);
  VecScale(b3,-PETSC_i*omega);
  VecScale(b4,-PETSC_i*omega);

  int tmpprint;
  getint("-print_dof",&tmpprint,100);
  int magsq;
  getint("-magsq",&magsq,0);
  PetscScalar norm_cell_scalar;
  VecSum(u,&norm_cell_scalar);

  data.b0=b0,data.b1=b1,data.b2=b2,data.b3=b3,data.b4=b4;
  data.v0=v0,data.v1=v1,data.v2=v2,data.v3=v3,data.v4=v4,data.u=u;
  data.colour=colour,data.subcomm=subcomm,data.M0=M0,data.epsDiff=epsDiff,data.epsBkg=epsBkg,data.omega=omega,data.ksp=ksp,data.its=&its,data.maxit=maxit,data.dofi=&dofi,data.dg=dg;
  data.x0=x0,data.x1=x1,data.x2=x2,data.x3=x3,data.x4=x4;
  data.obj0=&obj0,data.obj1=&obj1,data.obj2=&obj2,data.obj3=&obj3,data.obj4=&obj4;
  data.grad0=grad0,data.grad1=grad1,data.grad2=grad2,data.grad3=grad3,data.grad4=grad4;
  data.d0amp_mag=&d0amp_mag,data.d1amp_mag=&d1amp_mag,data.d2amp_mag=&d2amp_mag,data.d3amp_mag=&d3amp_mag,data.d4amp_mag=&d4amp_mag;
  data.d0amp_phi=&d0amp_phi,data.d1amp_phi=&d1amp_phi,data.d2amp_phi=&d2amp_phi,data.d3amp_phi=&d3amp_phi,data.d4amp_phi=&d4amp_phi;
  data.solve_order=solve_order;
  data.mask_order=mask_order;
  data.printdof=tmpprint,data.flt=&flt;
  data.magsq=magsq;
  data.eps_dof=eps_dof;
  data.amp_dof=amp_dof;
  data.norm_local=creal(norm_cell_scalar);
  data.norm_global=creal(norm_cell_scalar)*ncells;

  MPI_Barrier(subcomm);
  MPI_Barrier(PETSC_COMM_WORLD);
  /*------*/

  double *dofAll;
  dofAll = (double *) malloc(dofi.ndof*ncells*sizeof(double));
  
  getstr("-init_dof_name",tmpstr,"dof.txt");
  readfromfiledouble(tmpstr,dofAll,dofi.ndof*ncells);

  int Job;
  getint("-Job",&Job,1);
  if(Job==0){
    
    int ndofAll=dofi.ndof*ncells;
    int p;
    PetscReal obj_total;
    getint("-change_dof_at",&p,dofi.ndof/2);
    double s,s0=0,s1=1.0,ds=0.01;
    double *dofAll_grad;
    dofAll_grad = (double *) malloc(dofi.ndof*ncells*sizeof(double));
    for(s=s0;s<s1;s+=ds){
      dofAll[p]=s;
      obj_total=phopt_dispsum_epsglobal_globalabs(ndofAll,dofAll,dofAll_grad,&data);
      PetscPrintf(PETSC_COMM_WORLD,"eps, dispsum, grad: %g, %.16g, %.16g\n",dofAll[p],obj_total,dofAll_grad[p]);
    }
    
  }else if(Job==1){

    int ndof=flt.ndof;
    PetscScalar *dofcell;
    int i;
    dofcell  = (PetscScalar *) malloc(ndof*sizeof(PetscScalar));
    for(i=0;i<ndof;i++) dofcell[i]=dofAll[i+colour*ndof]+PETSC_i*0.0;
    filters_apply(subcomm,dofcell,data.eps_dof,&flt,1);
    
    double obj_total, amp_grad[2]={0,0};
    double s,s0=0,s1=2*M_PI,ds=0.01;
    for(s=s0;s<s1;s+=ds){
      amp_dof[1]=s;
      obj_total=phopt_dispsum_amponly(2,amp_dof,amp_grad,&data);
      PetscPrintf(PETSC_COMM_WORLD,"amp_phi, dispsum, grad: %g, %.16g, %.16g\n",amp_dof[1],obj_total,amp_grad[1]); 
    }
    free(dofcell);

  }else if(Job==2){

    int ndofAll=dofi.ndof*ncells;
    double *lb,*ub;
    lb=(double *) malloc(ndofAll*sizeof(double));
    ub=(double *) malloc(ndofAll*sizeof(double));
    make_array(lb,0.0,ndofAll);
    make_array(ub,1.0,ndofAll);
    optimize_generic(ndofAll, dofAll, lb,ub, &data, NULL, phopt_dispsum_epsglobal_globalabs, NULL, 1, 0);
    MPI_Barrier(subcomm);
    MPI_Barrier(PETSC_COMM_WORLD);
    
    free(lb);
    free(ub);

  }
    
  VecDestroy(&dg.vecTemp);
  VecDestroy(&epsDiff);
  VecDestroy(&epsBkg);
  MatDestroy(&M0);
  MatDestroy(&Curl);
  DMDestroy(&dg.da);
  KSPDestroy(&ksp);
  free(dofAll);

  VecDestroy(&v0);
  VecDestroy(&v1);
  VecDestroy(&v2);
  VecDestroy(&v3);
  VecDestroy(&v4);
  VecDestroy(&u);
  VecDestroy(&b0);
  VecDestroy(&b1);
  VecDestroy(&b2);
  VecDestroy(&b3);
  VecDestroy(&b4);
  VecDestroy(&usrc);
  VecDestroy(&x0);
  VecDestroy(&x1);
  VecDestroy(&x2);
  VecDestroy(&x3);
  VecDestroy(&x4);

  free(grad0);
  free(grad1);
  free(grad2);
  free(grad3);
  free(grad4);
  free(eps_dof);

  PetscFinalize();
  return 0;

}

void angular_disp_linear(PetscReal theta_in, const PetscReal *anglemap, pwparams* pw)
{
  PetscReal d1theta=(anglemap[1]-anglemap[3])/(anglemap[0]-anglemap[2]);
  
  pw->theta_rad = d1theta * ( theta_in - anglemap[0] ) + anglemap[1];
  pw->d1theta   = d1theta;
  pw->d2theta   = 0;
  pw->d3theta   = 0;
  pw->d4theta   = 0;

}
