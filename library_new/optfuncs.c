#include "optfuncs.h"

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

typedef PetscScalar (*functwo)(PetscScalar v, PetscScalar x);

#undef __FUNCT__
#define __FUNCT__ "private_manipulate_vx"
PetscErrorCode private_manipulate_vx(DM da, Vec u, Vec v, Vec x, Vec w, functwo f)
{
  PetscErrorCode ierr;

  Field ***_u,***_v,***_x,***_w;
  ierr = DMDAVecGetArray(da, u, &_u); CHKERRQ(ierr);
  ierr = DMDAVecGetArray(da, v, &_v); CHKERRQ(ierr);
  ierr = DMDAVecGetArray(da, x, &_x); CHKERRQ(ierr);
  ierr = DMDAVecGetArray(da, w, &_w); CHKERRQ(ierr);


  /** Get corners and widths of Yee's grid included in this proces. */
  PetscInt ox, oy, oz;  // coordinates of beginning corner of Yee's grid in this process
  PetscInt nx, ny, nz;  // local widths of Yee's grid in this process
  ierr = DMDAGetCorners(da, &ox, &oy, &oz, &nx, &ny, &nz); CHKERRQ(ierr);

  PetscScalar val[5];
  PetscInt ix,iy,iz,ic;
  for (iz = oz; iz < oz+nz; ++iz) {
    for (iy = oy; iy < oy+ny; ++iy) {
      for (ix = ox; ix < ox+nx; ++ix) {
        for (ic = 0; ic < Naxis; ++ic) {
          if(fabs(_u[iz][iy][ix].comp[ic])>0.01){
	    _w[iz][iy][ix].comp[ic]=f(_v[iz][iy][ix].comp[ic],_x[iz][iy][ix].comp[ic]);	    
          }else{
	    _w[iz][iy][ix].comp[ic]=0.0+PETSC_i*0.0;
          }
        }
      }
    }
  }

  ierr = DMDAVecRestoreArray(da, u, &_u); CHKERRQ(ierr);
  ierr = DMDAVecRestoreArray(da, v, &_v); CHKERRQ(ierr);
  ierr = DMDAVecRestoreArray(da, x, &_x); CHKERRQ(ierr);
  ierr = DMDAVecRestoreArray(da, w, &_w); CHKERRQ(ierr);

  return ierr;


}

PetscScalar func_phase_overlap(PetscScalar v, PetscScalar x)
{
  if(cabs(x)>0)
    return conj(v)*x/cabs(x);
  else 
    return 0.0+PETSC_i*0.0;
}

PetscScalar func_phase_overlap_dot(PetscScalar v, PetscScalar x)
{
  if(cabs(x)>0)
    return conj(v)/cabs(x) - creal(conj(v)*x)*conj(x)/pow(cabs(x),3);
  else
    return 0.0+PETSC_i*0.0;
}

#undef __FUNCT__
#define __FUNCT__ "phgrad"
PetscScalar phgrad(MPI_Comm comm, PetscScalar *dof, PetscReal ampphi, PetscScalar *dofgrad, PetscScalar *grad_ampphi, const Mat M0, const Vec b, Vec x, const Vec epsDiff, const Vec epsBkg, const PetscScalar omega, const Vec v, Vec u, KSP ksp, int *its, int maxit, DOFInfo *dofi, DM da)
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

  Vec ampv;
  Vec ampvxhat,adjoint_rhs,adjoint_var;
  PetscScalar phase_overlap_sum;

  VecDuplicate(x,&ampv);
  VecDuplicate(x,&ampvxhat);
  VecDuplicate(x,&adjoint_rhs);
  VecDuplicate(x,&adjoint_var);

  PetscScalar amp=cos(ampphi) + PETSC_i * sin(ampphi);
  VecCopy(v,ampv);
  VecScale(ampv,amp);

  SolveMatrixDirect(comm,ksp,M,b,x,its,maxit);
  private_manipulate_vx(da,u,ampv,x,ampvxhat,func_phase_overlap);
  VecSum(ampvxhat,&phase_overlap_sum);
  *grad_ampphi=-PETSC_i*phase_overlap_sum;

  PetscReal testdbgx,testdbgv,testdbgb, testdbg1, testdbg2;
  VecNorm(x,NORM_INFINITY,&testdbgx);
  VecNorm(v,NORM_INFINITY,&testdbgv);
  VecNorm(b,NORM_INFINITY,&testdbgb);
  VecNorm(ampv,NORM_INFINITY,&testdbg1);
  VecNorm(ampvxhat,NORM_INFINITY,&testdbg2);
  //PetscPrintf(comm,"DEBUG: %g %g %g %g %g\n",testdbgx,testdbgv,testdbgb,testdbg1,testdbg2);
  
  private_manipulate_vx(da,u,ampv,x,adjoint_rhs,func_phase_overlap_dot);
  KSPSolveTranspose(ksp,adjoint_rhs,adjoint_var);
  VecPointwiseMult(grad,adjoint_var,x);
  VecPointwiseMult(grad,grad,epsDiff);
  VecScale(grad,omega*omega);
  multilayer_backward(comm,grad,dofgrad,dofi,da);

  VecDestroy(&eps);
  VecDestroy(&negW2eps);
  MatDestroy(&M);

  VecDestroy(&ampv);
  VecDestroy(&ampvxhat);
  VecDestroy(&adjoint_rhs);
  VecDestroy(&adjoint_var);
  VecDestroy(&grad);
  
  PetscPrintf(comm,"phase_overlap_sum_local = %g + i*(%g), amp: %g %g \n",creal(phase_overlap_sum),cimag(phase_overlap_sum),creal(amp),cimag(amp));
  //PetscPrintf(comm,"DEBUG: nancheck %d %d %d %d %d\n",isnan(ampphi),isnan(cos(ampphi)),isnan(sin(ampphi)),isnan(creal(amp)),isnan(cimag(amp)));
  return phase_overlap_sum;

}

#undef __FUNCT__
#define __FUNCT__ "pherrgrad"
PetscScalar pherrgrad(MPI_Comm comm, PetscScalar *dof, PetscReal ampr, PetscReal ampphi, PetscScalar *dofgrad, PetscScalar *grad_ampr, PetscScalar *grad_ampphi, const Mat M0, const Vec b, Vec x, const Vec epsDiff, const Vec epsBkg, const PetscScalar omega, const Vec v, Vec u, KSP ksp, int *its, int maxit, DOFInfo *dofi, DM da)
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

  Vec ampv;
  Vec xdiffampv,adjoint_rhs,adjoint_var;
  PetscScalar phase_err_sum;

  VecDuplicate(x,&ampv);
  VecDuplicate(x,&xdiffampv);
  VecDuplicate(x,&adjoint_rhs);
  VecDuplicate(x,&adjoint_var);

  PetscScalar amp=ampr*(cos(ampphi) + PETSC_i * sin(ampphi));
  VecCopy(v,ampv);
  VecScale(ampv,amp);

  SolveMatrixDirect(comm,ksp,M,b,x,its,maxit);
  VecWAXPY(xdiffampv,-1.0,ampv,x);
  VecCopy(xdiffampv,adjoint_rhs);
  VecConjugate(adjoint_rhs);
  VecPointwiseMult(xdiffampv,xdiffampv,adjoint_rhs);
  VecSum(xdiffampv,&phase_err_sum);
  KSPSolveTranspose(ksp,adjoint_rhs,adjoint_var);
  VecPointwiseMult(grad,adjoint_var,x);
  VecPointwiseMult(grad,grad,epsDiff);
  VecScale(grad,omega*omega);
  multilayer_backward(comm,grad,dofgrad,dofi,da);

  PetscScalar ampgrad;
  VecPointwiseMult(xdiffampv,adjoint_rhs,v);
  VecSum(xdiffampv,&ampgrad);
  *grad_ampr= -ampgrad * (cos(ampphi) + PETSC_i * sin(ampphi));
  *grad_ampphi= -ampgrad * amp * PETSC_i;
  
  VecDestroy(&eps);
  VecDestroy(&negW2eps);
  MatDestroy(&M);

  VecDestroy(&ampv);
  VecDestroy(&xdiffampv);
  VecDestroy(&adjoint_rhs);
  VecDestroy(&adjoint_var);
  VecDestroy(&grad);

  return phase_err_sum;

}


#undef __FUNCT__
#define __FUNCT__ "maximinconstraint"
double maximinconstraint(int ndofAll_with_dummy, double *dofAll_with_dummy, double *dofgradAll_with_dummy, void *data)
{

  DotObjData *ptdata = (DotObjData *) data;
  optfunc func = ptdata->func;

  int ndofAll=ndofAll_with_dummy-1;
  double *dofAll,*dofgradAll;
  dofAll=(double *) malloc(ndofAll*sizeof(double));
  dofgradAll=(double *) malloc(ndofAll*sizeof(double));
  int i;
  for(i=0;i<ndofAll;i++){
    dofAll[i]=dofAll_with_dummy[i];
  }

  double obj=func(ndofAll,dofAll,dofgradAll,data);
  
  for(i=0;i<ndofAll;i++){
    dofgradAll_with_dummy[i]=-1.0*dofgradAll[i];
  }
  dofgradAll_with_dummy[ndofAll]=1.0;

  count--;
  free(dofAll);
  free(dofgradAll);
  return dofAll_with_dummy[ndofAll]-obj;

}

#undef __FUNCT__
#define __FUNCT__ "minimaxconstraint"
double minimaxconstraint(int ndofAll_with_dummy, double *dofAll_with_dummy, double *dofgradAll_with_dummy, void *data)
{

  DotObjData *ptdata = (DotObjData *) data;
  optfunc func = ptdata->func;

  int ndofAll=ndofAll_with_dummy-1;
  double *dofAll,*dofgradAll;
  dofAll=(double *) malloc(ndofAll*sizeof(double));
  dofgradAll=(double *) malloc(ndofAll*sizeof(double));
  int i;
  for(i=0;i<ndofAll;i++){
    dofAll[i]=dofAll_with_dummy[i];
  }

  double obj=func(ndofAll,dofAll,dofgradAll,data);
  
  for(i=0;i<ndofAll;i++){
    dofgradAll_with_dummy[i]=1.0*dofgradAll[i];
  }
  dofgradAll_with_dummy[ndofAll]=-1.0;

  count--;
  free(dofAll);
  free(dofgradAll);
  return obj-dofAll_with_dummy[ndofAll];

}

#undef __FUNCT__
#define __FUNCT__ "dummy_obj"
double dummy_obj(int ndofAll_with_dummy, double *dofAll_with_dummy, double *dofgradAll_with_dummy, void *data)
{

  if(dofgradAll_with_dummy){
    int i;
    for(i=0;i<ndofAll_with_dummy-1;i++){
      dofgradAll_with_dummy[i]=0;
    }
    dofgradAll_with_dummy[ndofAll_with_dummy-1]=1.0;
  }
  PetscPrintf(PETSC_COMM_WORLD,"******dummy value at step %d is %g \n",count,dofAll_with_dummy[ndofAll_with_dummy-1]);

  count++;
  return dofAll_with_dummy[ndofAll_with_dummy-1];

}

#undef __FUNCT__
#define __FUNCT__ "ffopt"
double ffopt(int ndofAll, double *dofAll, double *dofgradAll, void *data)
{

  DotObjData *ptdata = (DotObjData *) data;

  int colour = ptdata->colour;
  int ncells_per_comm = ptdata->ncells_per_comm;
  MPI_Comm subcomm = ptdata->subcomm;
  Mat M0 = ptdata->M0;
  Vec *b = ptdata->b;
  Vec *x = ptdata->x;
  Vec epsDiff = ptdata->epsDiff;
  Vec epsBkg = ptdata->epsBkg;
  PetscScalar omega = ptdata->omega;
  Vec *ffvec = ptdata->v;
  KSP *ksp = ptdata->ksp;
  int *its = ptdata->its;
  int maxit = ptdata->maxit;
  DOFInfo *dofi = ptdata->dofi;
  ParDataGrid dg = ptdata->dg;
  int printdof = ptdata->printdof;

  int ndof=dofi->ndof;
  PetscScalar *dof[ncells_per_comm],*dofgrad[ncells_per_comm];
  int icell,icell_world,i;
  FiltersToolBox *flt = ptdata->flt;
  PetscScalar xfar=0;
  PetscScalar *_u,*_ugrad;
  _u = (PetscScalar *) malloc(ndof*sizeof(PetscScalar));
  _ugrad = (PetscScalar *) malloc(ndof*sizeof(PetscScalar));

  for(icell=0;icell<ncells_per_comm;icell++){
    dof[icell] = (PetscScalar *) malloc(ndof*sizeof(PetscScalar));
    dofgrad[icell] = (PetscScalar *) malloc(ndof*sizeof(PetscScalar));
    for(i=0;i<ndof;i++){
      icell_world=icell+colour*ncells_per_comm;
      dof[icell][i]=dofAll[i+icell_world*ndof]+PETSC_i*0.0;
    }
    filters_apply(subcomm,dof[icell],_u,flt,1);
    xfar+=ffgrad(subcomm, _u,_ugrad, M0,b[icell],x[icell], epsDiff,epsBkg, omega, ffvec[icell], ksp[icell],its+icell,maxit, dofi,dg.da);
    filters_apply(subcomm,_ugrad,dofgrad[icell],flt,-1);
  }
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
  int j;
  tmp = (double *) malloc(ndofAll*sizeof(double));
  for(i=0;i<ndofAll;i++){
    if(subrank==0 && i>=colour*ncells_per_comm*ndof && i<(colour+1)*ncells_per_comm*ndof){
      j=(i-colour*ncells_per_comm*ndof)%ndof;
      icell=(i-colour*ncells_per_comm*ndof)/ndof;
      tmp[i]= 2.0*creal(conj(xfartotal)*dofgrad[icell][j]);
    }else{
      tmp[i] = 0.0;
    }
  }

  MPI_Allreduce(tmp,dofgradAll,ndofAll,MPI_DOUBLE,MPI_SUM,PETSC_COMM_WORLD);

  char output_filename[PETSC_MAX_PATH_LEN];
  if((count%printdof)==0){
    sprintf(output_filename,"outputdof_step%d_jpt%d.txt",count,ptdata->ID);
    writetofiledouble(PETSC_COMM_WORLD,output_filename,dofAll,ndofAll);
  }

  count++;
  free(_u);
  free(_ugrad);
  for(icell=0;icell<ncells_per_comm;icell++){
    free(dof[icell]);
    free(dofgrad[icell]);
  }
  free(tmp);
  return xfarsq;

}

#undef __FUNCT__
#define __FUNCT__ "phopt"
double phopt(int ndofAll, double *dofAll, double *dofgradAll, void *data)
{

  DotObjData *ptdata = (DotObjData *) data;

  int colour = ptdata->colour;
  int ncells_per_comm = ptdata->ncells_per_comm;
  MPI_Comm subcomm = ptdata->subcomm;
  Mat M0 = ptdata->M0;
  Vec *b = ptdata->b;
  Vec *x = ptdata->x;
  Vec epsDiff = ptdata->epsDiff;
  Vec epsBkg = ptdata->epsBkg;
  PetscScalar omega = ptdata->omega;
  Vec *v = ptdata->v;
  Vec u = ptdata->u;
  KSP *ksp = ptdata->ksp;
  int *its = ptdata->its;
  int maxit = ptdata->maxit;
  DOFInfo *dofi = ptdata->dofi;
  ParDataGrid dg = ptdata->dg;
  int printdof = ptdata->printdof;
  int ndof_eps = ptdata->ndof_eps;
  
  PetscReal ampphi;
  if(ndofAll==ndof_eps){
    ampphi=*(ptdata->ampphi);
    if(ampphi<1e-16){
      ampphi=0;
      *(ptdata->ampphi)=0;
    }
  }else{
    ampphi=dofAll[ndofAll-1];
    *(ptdata->ampr)=1.0;
    *(ptdata->ampphi)=ampphi;
    if(ampphi<1e-16){
      ampphi=0;
      dofAll[ndofAll-1]=0;
      *(ptdata->ampphi)=0;
    }
  }

      
  PetscScalar grad_ampphi=0, tmp_grad_ampphi;
  
  int ndof=dofi->ndof;
  PetscScalar *dof[ncells_per_comm],*dofgrad[ncells_per_comm];
  int icell,icell_world,i;
  FiltersToolBox *flt = ptdata->flt;
  PetscScalar *_u,*_ugrad;
  _u = (PetscScalar *) malloc(ndof*sizeof(PetscScalar));
  _ugrad = (PetscScalar *) malloc(ndof*sizeof(PetscScalar));
  PetscScalar phoverlap=0;
  for(icell=0;icell<ncells_per_comm;icell++){
    dof[icell] = (PetscScalar *) malloc(ndof*sizeof(PetscScalar));
    dofgrad[icell] = (PetscScalar *) malloc(ndof*sizeof(PetscScalar));
    icell_world=icell+colour*ncells_per_comm;
    for(i=0;i<ndof;i++){
      dof[icell][i]=dofAll[i+icell_world*ndof]+PETSC_i*0.0;
    }
    filters_apply(subcomm,dof[icell],_u,flt,1);
    phoverlap+=phgrad(subcomm, _u,2.0*M_PI*ampphi, _ugrad,&tmp_grad_ampphi, M0,b[icell],x[icell], epsDiff,epsBkg, omega, v[icell],u, ksp[icell],its+icell,maxit, dofi,dg.da);
    grad_ampphi+=tmp_grad_ampphi;
    filters_apply(subcomm,_ugrad,dofgrad[icell],flt,-1);
  }
  MPI_Barrier(subcomm);
  MPI_Barrier(PETSC_COMM_WORLD);

  PetscScalar norm;
  VecSum(u,&norm);
  norm=norm*ncells_per_comm;
  
  int subrank;
  MPI_Comm_rank(subcomm, &subrank);
  if(subrank>0){
    phoverlap=0;
    grad_ampphi=0;
    norm=0;
  }
  MPI_Barrier(subcomm);
  MPI_Barrier(PETSC_COMM_WORLD);

  PetscScalar phoverlaptotal,normtotal;
  MPI_Allreduce(&phoverlap,&phoverlaptotal,1,MPIU_SCALAR,MPI_SUM,PETSC_COMM_WORLD);
  MPI_Allreduce(&norm,&normtotal,1,MPIU_SCALAR,MPI_SUM,PETSC_COMM_WORLD);
  PetscPrintf(PETSC_COMM_WORLD,"phoverlap_total = %g + i*(%g) \n",creal(phoverlaptotal),cimag(phoverlaptotal));
  double phobj=creal(phoverlaptotal)/creal(normtotal);
  PetscPrintf(PETSC_COMM_WORLD,"******phobj at step %d is %g \n",count,phobj);

  PetscScalar grad_ampphi_total;
  MPI_Allreduce(&grad_ampphi,&grad_ampphi_total,1,MPIU_SCALAR,MPI_SUM,PETSC_COMM_WORLD);
  PetscPrintf(PETSC_COMM_WORLD,"******the phase at step %d is %g with gradient %g \n",count,ampphi,creal(grad_ampphi_total)/creal(normtotal));
  
  double *tmp;
  int j;
  tmp = (double *) malloc(ndofAll*sizeof(double));
  for(i=0;i<ndofAll;i++){
    if(subrank==0 && i>=colour*ncells_per_comm*ndof && i<(colour+1)*ncells_per_comm*ndof){
      j=(i-colour*ncells_per_comm*ndof)%ndof;
      icell=(i-colour*ncells_per_comm*ndof)/ndof;
      tmp[i]= creal(dofgrad[icell][j])/creal(normtotal);
    }else{
      tmp[i] = 0.0;
    }
  }
  MPI_Allreduce(tmp,dofgradAll,ndofAll,MPI_DOUBLE,MPI_SUM,PETSC_COMM_WORLD);
  if(ndofAll>ndof_eps){
    dofgradAll[ndofAll-1]=2.0*M_PI*creal(grad_ampphi_total)/creal(normtotal);
  }

  char output_filename[PETSC_MAX_PATH_LEN];
  if((count%printdof)==0){
    sprintf(output_filename,"outputdof_step%d_jpt%d.txt",count,ptdata->ID);
    writetofiledouble(PETSC_COMM_WORLD,output_filename,dofAll,ndofAll);
  }
  
  count++;
  free(_u);
  free(_ugrad);
  for(icell=0;icell<ncells_per_comm;icell++){
    free(dof[icell]);
    free(dofgrad[icell]);
  }
  free(tmp);
  return phobj;

}


#undef __FUNCT__
#define __FUNCT__ "pherropt"
double pherropt(int ndofAll, double *dofAll, double *dofgradAll, void *data)
{

  DotObjData *ptdata = (DotObjData *) data;

  int colour = ptdata->colour;
  int ncells_per_comm = ptdata->ncells_per_comm;
  MPI_Comm subcomm = ptdata->subcomm;
  Mat M0 = ptdata->M0;
  Vec *b = ptdata->b;
  Vec *x = ptdata->x;
  Vec epsDiff = ptdata->epsDiff;
  Vec epsBkg = ptdata->epsBkg;
  PetscScalar omega = ptdata->omega;
  Vec *v = ptdata->v;
  Vec u = ptdata->u;
  KSP *ksp = ptdata->ksp;
  int *its = ptdata->its;
  int maxit = ptdata->maxit;
  DOFInfo *dofi = ptdata->dofi;
  ParDataGrid dg = ptdata->dg;
  int printdof = ptdata->printdof;
  int ndof_eps = ptdata->ndof_eps;
  
  PetscReal ampr, ampphi;
  if(ndofAll==ndof_eps){
    ampr=*(ptdata->ampr);
    ampphi=*(ptdata->ampphi);
  }else{
    ampr=dofAll[ndofAll-2];
    ampphi=dofAll[ndofAll-1];
    *(ptdata->ampr)=ampr;
    *(ptdata->ampphi)=ampphi;
  }
  PetscScalar grad_ampr=0, tmp_grad_ampr, grad_ampphi=0, tmp_grad_ampphi;
  
  int ndof=dofi->ndof;
  PetscScalar *dof[ncells_per_comm],*dofgrad[ncells_per_comm];
  int icell,icell_world,i;
  FiltersToolBox *flt = ptdata->flt;
  PetscScalar *_u,*_ugrad;
  _u = (PetscScalar *) malloc(ndof*sizeof(PetscScalar));
  _ugrad = (PetscScalar *) malloc(ndof*sizeof(PetscScalar));
  PetscScalar pherr=0;
  for(icell=0;icell<ncells_per_comm;icell++){
    dof[icell] = (PetscScalar *) malloc(ndof*sizeof(PetscScalar));
    dofgrad[icell] = (PetscScalar *) malloc(ndof*sizeof(PetscScalar));
    icell_world=icell+colour*ncells_per_comm;
    for(i=0;i<ndof;i++){
      dof[icell][i]=dofAll[i+icell_world*ndof]+PETSC_i*0.0;
    }
    filters_apply(subcomm,dof[icell],_u,flt,1);
    pherr+=pherrgrad(subcomm, _u,ampr,ampphi, _ugrad,&tmp_grad_ampr,&tmp_grad_ampphi, M0,b[icell],x[icell], epsDiff,epsBkg, omega, v[icell],u, ksp[icell],its+icell,maxit, dofi,dg.da);
    grad_ampr+=tmp_grad_ampr;
    grad_ampphi+=tmp_grad_ampphi;
    filters_apply(subcomm,_ugrad,dofgrad[icell],flt,-1);
  }
  MPI_Barrier(subcomm);
  MPI_Barrier(PETSC_COMM_WORLD);

  int subrank;
  MPI_Comm_rank(subcomm, &subrank);
  if(subrank>0){
    pherr=0;
    grad_ampr=0;
    grad_ampphi=0;
  }
  MPI_Barrier(subcomm);
  MPI_Barrier(PETSC_COMM_WORLD);

  PetscScalar pherrtotal;
  MPI_Allreduce(&pherr,&pherrtotal,1,MPIU_SCALAR,MPI_SUM,PETSC_COMM_WORLD);
  PetscPrintf(PETSC_COMM_WORLD,"pherr_total = %g + i*(%g) \n",creal(pherrtotal),cimag(pherrtotal));
  double phobj=creal(pherrtotal);
  PetscPrintf(PETSC_COMM_WORLD,"******phobj at step %d is %g \n",count,phobj);
  PetscPrintf(PETSC_COMM_WORLD,"******phase and amplitude at step %d is %0.16g %0.16g\n",count,ampr,ampphi);

  PetscScalar grad_ampr_total, grad_ampphi_total;
  MPI_Allreduce(&grad_ampphi,&grad_ampphi_total,1,MPIU_SCALAR,MPI_SUM,PETSC_COMM_WORLD);
  MPI_Allreduce(&grad_ampr,&grad_ampr_total,1,MPIU_SCALAR,MPI_SUM,PETSC_COMM_WORLD);
  
  double *tmp;
  int j;
  tmp = (double *) malloc(ndofAll*sizeof(double));
  for(i=0;i<ndofAll;i++){
    if(subrank==0 && i>=colour*ncells_per_comm*ndof && i<(colour+1)*ncells_per_comm*ndof){
      j=(i-colour*ncells_per_comm*ndof)%ndof;
      icell=(i-colour*ncells_per_comm*ndof)/ndof;
      tmp[i]= 2.0*creal(dofgrad[icell][j]);
    }else{
      tmp[i] = 0.0;
    }
  }
  MPI_Allreduce(tmp,dofgradAll,ndofAll,MPI_DOUBLE,MPI_SUM,PETSC_COMM_WORLD);
  if(ndofAll>ndof_eps){
    dofgradAll[ndofAll-2]=2.0*creal(grad_ampr_total);
    dofgradAll[ndofAll-1]=2.0*creal(grad_ampphi_total);
  }

  char output_filename[PETSC_MAX_PATH_LEN];
  if((count%printdof)==0){
    sprintf(output_filename,"outputdof_step%d_jpt%d.txt",count,ptdata->ID);
    writetofiledouble(PETSC_COMM_WORLD,output_filename,dofAll,ndofAll);
  }
  
  count++;
  free(_u);
  free(_ugrad);
  for(icell=0;icell<ncells_per_comm;icell++){
    free(dof[icell]);
    free(dofgrad[icell]);
  }
  free(tmp);
  return phobj;

}

#undef __FUNCT__
#define __FUNCT__ "phoptmulti"
double phoptmulti(int ndofAll, double *dofAll, double *dofgradAll, void *data)
{

  DotObjData *ptdata = (DotObjData *) data;

  int ndof_eps=ptdata->ndof_eps;
  int ID=ptdata->ID;

  int ndof=ndof_eps+1;
  double *dof, *dofgrad;
  dof=(double *)malloc(ndof*sizeof(double));
  dofgrad=(double *)malloc(ndof*sizeof(double));
  int i;
  for(i=0;i<ndof_eps;i++){
    dof[i]=dofAll[i];
  }
  dof[ndof_eps]=dofAll[ndof_eps+ID];
  double phobj=phopt(ndof,dof,dofgrad,data);
  for(i=0;i<ndof_eps;i++){
    dofgradAll[i]=dofgrad[i];
  }
  dofgradAll[ndof_eps+ID]=dofgrad[ndof_eps];

  free(dof);
  free(dofgrad);

  return phobj;

}

#undef __FUNCT__
#define __FUNCT__ "pherroptmulti"
double pherroptmulti(int ndofAll, double *dofAll, double *dofgradAll, void *data)
{

  DotObjData *ptdata = (DotObjData *) data;

  int ndof_eps=ptdata->ndof_eps;
  int ID=ptdata->ID;
  
  int ndof=ndof_eps+2;
  double *dof, *dofgrad;
  dof=(double *)malloc(ndof*sizeof(double));
  dofgrad=(double *)malloc(ndof*sizeof(double));
  int i;
  for(i=0;i<ndof_eps;i++){
    dof[i]=dofAll[i];
  }
  dof[ndof_eps]=dofAll[ndof_eps];
  dof[ndof_eps+1]=dofAll[ndof_eps+ID+1];
  double phobj=pherropt(ndof,dof,dofgrad,data);
  for(i=0;i<ndof_eps;i++){
    dofgradAll[i]=dofgrad[i];
  }
  dofgradAll[ndof_eps]=dofgrad[ndof_eps];
  dofgradAll[ndof_eps+ID+1]=dofgrad[ndof_eps+1];

  free(dof);
  free(dofgrad);

  return phobj;
}

