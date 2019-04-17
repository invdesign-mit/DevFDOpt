#include "phopt.h"

extern int count;

typedef PetscScalar (*functwo)(PetscScalar v, PetscScalar x);
typedef PetscScalar (*funcfour)(PetscScalar v, PetscScalar x, PetscScalar v1, PetscScalar x1);

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

#undef __FUNCT__
#define __FUNCT__ "private_manipulate_vxvx"
PetscErrorCode private_manipulate_vxvx(DM da, Vec u, Vec v, Vec x, Vec v1, Vec x1, Vec w, funcfour f)
{
  PetscErrorCode ierr;

  Field ***_u,***_v,***_x,***_w,***_v1,***_x1;
  ierr = DMDAVecGetArray(da, u, &_u); CHKERRQ(ierr);
  ierr = DMDAVecGetArray(da, v, &_v); CHKERRQ(ierr);
  ierr = DMDAVecGetArray(da, x, &_x); CHKERRQ(ierr);
  ierr = DMDAVecGetArray(da, v1, &_v1); CHKERRQ(ierr);
  ierr = DMDAVecGetArray(da, x1, &_x1); CHKERRQ(ierr);
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
	    _w[iz][iy][ix].comp[ic]=f(_v[iz][iy][ix].comp[ic],_x[iz][iy][ix].comp[ic],_v1[iz][iy][ix].comp[ic],_x1[iz][iy][ix].comp[ic]);	    
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
  ierr = DMDAVecRestoreArray(da, v1, &_v1); CHKERRQ(ierr);
  ierr = DMDAVecRestoreArray(da, x1, &_x1); CHKERRQ(ierr);
  ierr = DMDAVecRestoreArray(da, w, &_w); CHKERRQ(ierr);

  return ierr;


}

PetscScalar func_F0(PetscScalar v, PetscScalar x)
{
  if(cabs(x)>0)
    return conj(v)*x/cabs(x);
  else 
    return 0.0+PETSC_i*0.0;
}

PetscScalar func_F0dot(PetscScalar v, PetscScalar x)
{
  if(cabs(x)>0)
    return conj(v)/cabs(x) - creal(conj(v)*x)*conj(x)/pow(cabs(x),3);
  else
    return 0.0+PETSC_i*0.0;
}

PetscScalar func_F1(PetscScalar v, PetscScalar x, PetscScalar v1, PetscScalar x1)
{
  if(cabs(x)>0)
    return pow( cimag(x1/x) - creal(v1) , 2) + PETSC_i*0.0;
  else 
    return 0.0+PETSC_i*0.0;
}

PetscScalar func_F1dot_c1(PetscScalar v, PetscScalar x, PetscScalar v1, PetscScalar x1)
{

  PetscReal delta_phi1 = cimag(x1/x) - creal(v1);
  if(cabs(x)>0)
    return 2*delta_phi1/x * (-PETSC_i);
  else 
    return 0.0+PETSC_i*0.0;
}

PetscScalar func_F1dot_c0(PetscScalar v, PetscScalar x, PetscScalar v1, PetscScalar x1)
{

  PetscReal delta_phi1 = cimag(x1/x) - creal(v1);
  if(cabs(x)>0)
    return 2*delta_phi1*x1/(x*x) * PETSC_i;
  else 
    return 0.0+PETSC_i*0.0;
}

#undef __FUNCT__
#define __FUNCT__ "phgrad"
void phgrad(void *data)
{
  Phasefielddata *ptdata = (Phasefielddata *) data;

  int colour = ptdata->colour;
  MPI_Comm subcomm = ptdata->subcomm;
  Mat M0 = ptdata->M0;
  Vec epsDiff = ptdata->epsDiff;
  Vec epsBkg = ptdata->epsBkg;
  PetscScalar omega = ptdata->omega;
  KSP ksp = ptdata->ksp;
  int *its = ptdata->its;
  int maxit = ptdata->maxit;
  DOFInfo *dofi = ptdata->dofi;
  ParDataGrid dg = ptdata->dg;
  Vec x0 = ptdata->x0;
  Vec x1 = ptdata->x1;
  Vec x2 = ptdata->x2;
  Vec x3 = ptdata->x3;
  Vec x4 = ptdata->x4;
  Vec b0 = ptdata->b0;
  Vec b1 = ptdata->b1;
  Vec b2 = ptdata->b2;
  Vec b3 = ptdata->b3;
  Vec b4 = ptdata->b4;
  Vec v0 = ptdata->v0;
  Vec v1 = ptdata->v1;
  Vec v2 = ptdata->v2;
  Vec v3 = ptdata->v3;
  Vec v4 = ptdata->v4;
  Vec u  = ptdata->u;
  PetscReal *obj0 = ptdata->obj0;
  PetscReal *obj1 = ptdata->obj1;
  PetscReal *obj2 = ptdata->obj2;
  PetscReal *obj3 = ptdata->obj3;
  PetscReal *obj4 = ptdata->obj4;
  PetscScalar *grad0 = ptdata->grad0;
  PetscScalar *grad1 = ptdata->grad1;
  PetscScalar *grad2 = ptdata->grad2;
  PetscScalar *grad3 = ptdata->grad3;
  PetscScalar *grad4 = ptdata->grad4;
  PetscReal *d0amp_mag = ptdata->d0amp_mag;
  PetscReal *d1amp_mag = ptdata->d1amp_mag;
  PetscReal *d2amp_mag = ptdata->d2amp_mag;
  PetscReal *d3amp_mag = ptdata->d3amp_mag;
  PetscReal *d4amp_mag = ptdata->d4amp_mag;
  PetscReal *d0amp_phi = ptdata->d0amp_phi;
  PetscReal *d1amp_phi = ptdata->d1amp_phi;
  PetscReal *d2amp_phi = ptdata->d2amp_phi;
  PetscReal *d3amp_phi = ptdata->d3amp_phi;
  PetscReal *d4amp_phi = ptdata->d4amp_phi;
  PetscInt *solve_order = ptdata->solve_order;
  PetscScalar *eps_dof = ptdata->eps_dof;
  PetscReal *amp_dof = ptdata->amp_dof;

  Vec eps,negW2eps;
  Mat M;
  VecDuplicate(x0,&eps);
  VecDuplicate(x0,&negW2eps);
  MatDuplicate(M0,MAT_COPY_VALUES,&M);

  multilayer_forward(eps_dof,eps,dofi,dg.da);
  VecPointwiseMult(eps,eps,epsDiff);
  VecAXPY(eps,1.0,epsBkg);
  VecCopy(eps,negW2eps);
  VecScale(negW2eps,-omega*omega);
  MatDiagonalSet(M,negW2eps,ADD_VALUES);

  Vec v0amp;
  Vec f0,f1,f2,f3,f4;
  Vec c0,c1,c2,c3,c4;
  Vec y0,y1,y2,y3,y4;
  Vec g0,g1,g2,g3,g4;
  Vec kappa;
  VecDuplicate(x0,&v0amp);
  VecDuplicate(x0,&f0);
  VecDuplicate(x0,&f1);
  VecDuplicate(x0,&f2);
  VecDuplicate(x0,&f3);
  VecDuplicate(x0,&f4);
  VecDuplicate(x0,&c0);
  VecDuplicate(x0,&c1);
  VecDuplicate(x0,&c2);
  VecDuplicate(x0,&c3);
  VecDuplicate(x0,&c4);
  VecDuplicate(x0,&y0);
  VecDuplicate(x0,&y1);
  VecDuplicate(x0,&y2);
  VecDuplicate(x0,&y3);
  VecDuplicate(x0,&y4);
  VecDuplicate(x0,&g0);
  VecDuplicate(x0,&g1);
  VecDuplicate(x0,&g2);
  VecDuplicate(x0,&g3);
  VecDuplicate(x0,&g4);
  VecDuplicate(x0,&kappa);

  PetscReal amp_mag=amp_dof[0];
  PetscReal amp_phi=amp_dof[1];
  PetscScalar amp = amp_mag * cexp(PETSC_i * amp_phi);
  PetscScalar tmp;

  VecCopy(epsDiff,kappa);
  VecScale(kappa,omega*omega);

  VecCopy(v0,v0amp);
  VecScale(v0amp,amp);

  if(solve_order[0]){

    SolveMatrixDirect(subcomm,ksp,M,b0,x0,its,maxit);
    private_manipulate_vx(dg.da,u,v0amp,x0,f0,func_F0);
    VecSum(f0,&tmp);
    *obj0=creal(tmp);

    private_manipulate_vx(dg.da,u,v0amp,x0,c0,func_F0dot);
    KSPSolveTranspose(ksp,c0,y0);
    VecPointwiseMult(g0,y0,x0);
    VecPointwiseMult(g0,g0,kappa);
    multilayer_backward(subcomm,g0,grad0,dofi,dg.da);

    *d0amp_mag=0;
    *d0amp_phi=0;

  }

  if(solve_order[1]){
    //if(solve_order[0]<=0) SETERRQ(PETSC_COMM_SELF,1,"solve_order[0] must be > 0.");

    SolveMatrixDirect(subcomm,ksp,M,b1,x1,its,maxit);
    private_manipulate_vxvx(dg.da,u,v0amp,x0,v1,x1,f1,func_F1);
    VecSum(f1,&tmp);
    *obj1=creal(tmp);

    private_manipulate_vxvx(dg.da,u,v0amp,x0,v1,x1,c0,func_F1dot_c0);
    KSPSolveTranspose(ksp,c0,y0);
    VecPointwiseMult(g0,y0,x0);
    private_manipulate_vxvx(dg.da,u,v0amp,x0,v1,x1,c1,func_F1dot_c1);
    KSPSolveTranspose(ksp,c1,y1);
    VecPointwiseMult(g1,y1,x1);
    VecAXPY(g1,1.0,g0);
    VecPointwiseMult(g1,g1,kappa);
    multilayer_backward(subcomm,g1,grad1,dofi,dg.da);

    *d1amp_mag=0;
    *d1amp_phi=0;

  }

  if(solve_order[2]){
    //if(solve_order[0]<=0 || solve_order[1]<=0) SETERRQ(PETSC_COMM_SELF,1,"solve_order[0,1] must be > 0.");

    *obj2=0;
    VecSet(g2,0.0);
    multilayer_backward(subcomm,g2,grad2,dofi,dg.da);
    *d2amp_mag=0;
    *d2amp_phi=0;

  }

  if(solve_order[3]){
    //if(solve_order[0]<=0 || solve_order[1]<=0 || solve_order[2]<=0) SETERRQ(PETSC_COMM_SELF,1,"solve_order[0,1,2] must be > 0.");

    *obj3=0;
    VecSet(g3,0.0);
    multilayer_backward(subcomm,g3,grad3,dofi,dg.da);
    *d3amp_mag=0;
    *d3amp_phi=0;

  }

  if(solve_order[4]){
    //if(solve_order[0]<=0 || solve_order[1]<=0 || solve_order[2]<=0 || solve_order[3]<=0) SETERRQ(PETSC_COMM_SELF,1,"solve_order[0,1,2,3] must be > 0.");
    
    *obj4=0;
    VecSet(g4,0.0);
    multilayer_backward(subcomm,g4,grad4,dofi,dg.da);
    *d4amp_mag=0;
    *d4amp_phi=0;

  }

  VecDestroy(&eps);
  VecDestroy(&negW2eps);
  MatDestroy(&M);

  VecDestroy(&v0amp);
  VecDestroy(&f0);
  VecDestroy(&f1);
  VecDestroy(&f2);
  VecDestroy(&f3);
  VecDestroy(&f4);
  VecDestroy(&c0);
  VecDestroy(&c1);
  VecDestroy(&c2);
  VecDestroy(&c3);
  VecDestroy(&c4);
  VecDestroy(&y0);
  VecDestroy(&y1);
  VecDestroy(&y2);
  VecDestroy(&y3);
  VecDestroy(&y4);
  VecDestroy(&g0);
  VecDestroy(&g1);
  VecDestroy(&g2);
  VecDestroy(&g3);
  VecDestroy(&g4);
  VecDestroy(&kappa);

}

/*****************************************/

PetscReal magfun(PetscReal x, int sq)
{
  PetscReal val;
  if(sq)
    val=x*x;
  else
    val=fabs(x);

  return val;
}

PetscReal magfun_prime(PetscReal x, int sq)
{

  PetscReal val;
  if(sq)
    val=2*x;
  else
    val=(PetscReal) ((x>=0) - (x<0));

  return val;
}

#undef __FUNCT__
#define __FUNCT__ "phopt_dispsum_amponly"
double phopt_dispsum_amponly(int n, double *amp, double *amp_grad, void *data)
{

  Phasefielddata *ptdata = (Phasefielddata *) data;
  MPI_Comm subcomm = ptdata->subcomm;
  PetscReal *mask_order = ptdata->mask_order;
  int magsq = ptdata->magsq;

  ptdata->amp_dof= (PetscReal *) amp;
  //PetscPrintf(PETSC_COMM_WORLD,"***amplitude (r,phi) from argument at the beginning is %.8g , %.8g \n",amp[0],amp[1]);
  //PetscPrintf(PETSC_COMM_WORLD,"***amplitude (r,phi) from void data at the beginning is %.8g , %.8g \n",ptdata->amp_dof[0],ptdata->amp_dof[1]);

  phgrad(data);
  MPI_Barrier(subcomm);
  MPI_Barrier(PETSC_COMM_WORLD);

  PetscReal obj0=0,d0amp_mag=0,d0amp_phi=0;
  PetscReal obj1=0,d1amp_mag=0,d1amp_phi=0;
  PetscReal obj2=0,d2amp_mag=0,d2amp_phi=0;
  PetscReal obj3=0,d3amp_mag=0,d3amp_phi=0;
  PetscReal obj4=0,d4amp_mag=0,d4amp_phi=0;
  int subrank;
  MPI_Comm_rank(subcomm, &subrank);
  if(subrank==0) {
    obj0 = *(ptdata->obj0);
    d0amp_mag = *(ptdata->d0amp_mag);
    d0amp_phi = *(ptdata->d0amp_phi);
    obj1 = *(ptdata->obj1);
    d1amp_mag = *(ptdata->d1amp_mag);
    d1amp_phi = *(ptdata->d1amp_phi);
    obj2 = *(ptdata->obj2);
    d2amp_mag = *(ptdata->d2amp_mag);
    d2amp_phi = *(ptdata->d2amp_phi);
    obj3 = *(ptdata->obj3);
    d3amp_mag = *(ptdata->d3amp_mag);
    d3amp_phi = *(ptdata->d3amp_phi);
    obj4 = *(ptdata->obj4);
    d4amp_mag = *(ptdata->d4amp_mag);
    d4amp_phi = *(ptdata->d4amp_phi);

  }
  MPI_Barrier(subcomm);
  MPI_Barrier(PETSC_COMM_WORLD);

  PetscReal obj0_total,d0amp_mag_total,d0amp_phi_total;
  MPI_Allreduce(&obj0,&obj0_total,1,MPIU_REAL,MPI_SUM,PETSC_COMM_WORLD);
  MPI_Allreduce(&d0amp_mag,&d0amp_mag_total,1,MPIU_REAL,MPI_SUM,PETSC_COMM_WORLD);
  MPI_Allreduce(&d0amp_phi,&d0amp_phi_total,1,MPIU_REAL,MPI_SUM,PETSC_COMM_WORLD);
  PetscPrintf(PETSC_COMM_WORLD,"***obj0_total at step %d is %.8g \n",count,obj0_total);

  PetscReal obj1_total,d1amp_mag_total,d1amp_phi_total;
  MPI_Allreduce(&obj1,&obj1_total,1,MPIU_REAL,MPI_SUM,PETSC_COMM_WORLD);
  MPI_Allreduce(&d1amp_mag,&d1amp_mag_total,1,MPIU_REAL,MPI_SUM,PETSC_COMM_WORLD);
  MPI_Allreduce(&d1amp_phi,&d1amp_phi_total,1,MPIU_REAL,MPI_SUM,PETSC_COMM_WORLD);
  PetscPrintf(PETSC_COMM_WORLD,"***obj1_total at step %d is %.8g \n",count,obj1_total);

  PetscReal obj2_total,d2amp_mag_total,d2amp_phi_total;
  MPI_Allreduce(&obj2,&obj2_total,1,MPIU_REAL,MPI_SUM,PETSC_COMM_WORLD);
  MPI_Allreduce(&d2amp_mag,&d2amp_mag_total,1,MPIU_REAL,MPI_SUM,PETSC_COMM_WORLD);
  MPI_Allreduce(&d2amp_phi,&d2amp_phi_total,1,MPIU_REAL,MPI_SUM,PETSC_COMM_WORLD);
  PetscPrintf(PETSC_COMM_WORLD,"***obj2_total at step %d is %.8g \n",count,obj2_total);

  PetscReal obj3_total,d3amp_mag_total,d3amp_phi_total;
  MPI_Allreduce(&obj3,&obj3_total,1,MPIU_REAL,MPI_SUM,PETSC_COMM_WORLD);
  MPI_Allreduce(&d3amp_mag,&d3amp_mag_total,1,MPIU_REAL,MPI_SUM,PETSC_COMM_WORLD);
  MPI_Allreduce(&d3amp_phi,&d3amp_phi_total,1,MPIU_REAL,MPI_SUM,PETSC_COMM_WORLD);
  PetscPrintf(PETSC_COMM_WORLD,"***obj3_total at step %d is %.8g \n",count,obj3_total);

  PetscReal obj4_total,d4amp_mag_total,d4amp_phi_total;
  MPI_Allreduce(&obj4,&obj4_total,1,MPIU_REAL,MPI_SUM,PETSC_COMM_WORLD);
  MPI_Allreduce(&d4amp_mag,&d4amp_mag_total,1,MPIU_REAL,MPI_SUM,PETSC_COMM_WORLD);
  MPI_Allreduce(&d4amp_phi,&d4amp_phi_total,1,MPIU_REAL,MPI_SUM,PETSC_COMM_WORLD);
  PetscPrintf(PETSC_COMM_WORLD,"***obj4_total at step %d is %.8g \n",count,obj4_total);
  
  PetscReal disp_sum = mask_order[0] * obj0_total \
    + mask_order[1] * magfun(obj1_total,magsq) \
    + mask_order[2] * magfun(obj2_total,magsq) \
    + mask_order[3] * magfun(obj3_total,magsq) \
    + mask_order[4] * magfun(obj4_total,magsq);
  PetscPrintf(PETSC_COMM_WORLD,"***dispersion sum at step %d is %.8g \n",count,disp_sum);

  if(amp_grad){
    amp_grad[0]= mask_order[0] * d0amp_mag_total \
      + mask_order[1] * magfun_prime(obj1_total,magsq) * d1amp_mag_total \
      + mask_order[2] * magfun_prime(obj2_total,magsq) * d2amp_mag_total \
      + mask_order[3] * magfun_prime(obj3_total,magsq) * d3amp_mag_total \
      + mask_order[4] * magfun_prime(obj4_total,magsq) * d4amp_mag_total;
    amp_grad[1]= mask_order[0] * d0amp_phi_total \
      + mask_order[1] * magfun_prime(obj1_total,magsq) * d1amp_phi_total \
      + mask_order[2] * magfun_prime(obj2_total,magsq) * d2amp_phi_total \
      + mask_order[3] * magfun_prime(obj3_total,magsq) * d3amp_phi_total \
      + mask_order[4] * magfun_prime(obj4_total,magsq) * d4amp_phi_total;
  }

  PetscPrintf(PETSC_COMM_WORLD,"***amplitude (r,phi) at step %d is %.16g , %.16g \n",count,ptdata->amp_dof[0],ptdata->amp_dof[1]);

  count++;
  return disp_sum;

}

#undef __FUNCT__
#define __FUNCT__ "phopt_dispsum_epscell"
double phopt_dispsum_epscell(int ndofcell, double *epscell, double *epscell_grad, void *data)
{

  Phasefielddata *ptdata = (Phasefielddata *) data;
  int colour = ptdata->colour;
  MPI_Comm subcomm = ptdata->subcomm;
  PetscReal *mask_order = ptdata->mask_order;
  FiltersToolBox *flt = ptdata->flt;
  int printdof = ptdata->printdof;
  int magsq = ptdata->magsq;

  PetscScalar *u,*_u0,*_u1,*_u2,*_u3,*_u4;
   u  = (PetscScalar *) malloc(ndofcell*sizeof(PetscScalar));
  _u0 = (PetscScalar *) malloc(ndofcell*sizeof(PetscScalar));
  _u1 = (PetscScalar *) malloc(ndofcell*sizeof(PetscScalar));
  _u2 = (PetscScalar *) malloc(ndofcell*sizeof(PetscScalar));
  _u3 = (PetscScalar *) malloc(ndofcell*sizeof(PetscScalar));
  _u4 = (PetscScalar *) malloc(ndofcell*sizeof(PetscScalar));
  int i;
  for(i=0;i<ndofcell;i++) u[i] = epscell[i] + PETSC_i * 0.0;
  filters_apply(subcomm,u,ptdata->eps_dof,flt,1);
  phgrad(data);
  filters_apply(subcomm,ptdata->grad0,_u0,flt,-1);
  filters_apply(subcomm,ptdata->grad1,_u1,flt,-1);
  filters_apply(subcomm,ptdata->grad2,_u2,flt,-1);
  filters_apply(subcomm,ptdata->grad3,_u3,flt,-1);
  filters_apply(subcomm,ptdata->grad4,_u4,flt,-1);

  PetscReal g0,g1,g2,g3,g4;
  PetscReal dispsum_cell;
  g0=mask_order[0]*        *(ptdata->obj0);
  g1=mask_order[1]* magfun(*(ptdata->obj1),magsq);
  g2=mask_order[2]* magfun(*(ptdata->obj2),magsq);
  g3=mask_order[3]* magfun(*(ptdata->obj3),magsq);
  g4=mask_order[4]* magfun(*(ptdata->obj4),magsq);
  dispsum_cell= g0+g1+g2+g3+g4;

  PetscPrintf(subcomm,"***obj0 in the %dth cell at step %d is %.8g \n",colour,count,*(ptdata->obj0));
  PetscPrintf(subcomm,"***obj1 in the %dth cell at step %d is %.8g \n",colour,count,*(ptdata->obj1));
  PetscPrintf(subcomm,"***obj2 in the %dth cell at step %d is %.8g \n",colour,count,*(ptdata->obj2));
  PetscPrintf(subcomm,"***obj3 in the %dth cell at step %d is %.8g \n",colour,count,*(ptdata->obj3));
  PetscPrintf(subcomm,"***obj4 in the %dth cell at step %d is %.8g \n",colour,count,*(ptdata->obj4));
  PetscPrintf(subcomm,"***dispersion sum in the %dth cell at step %d is %.8g \n",colour,count,dispsum_cell);

  if(epscell_grad){
    for(i=0;i<ndofcell;i++){
      g0 = mask_order[0] *                                    creal(_u0[i]);
      g1 = mask_order[1] * magfun_prime(*ptdata->obj1,magsq) *creal(_u1[i]);
      g2 = mask_order[2] * magfun_prime(*ptdata->obj2,magsq) *creal(_u2[i]);
      g3 = mask_order[3] * magfun_prime(*ptdata->obj3,magsq) *creal(_u3[i]);
      g4 = mask_order[4] * magfun_prime(*ptdata->obj4,magsq) *creal(_u4[i]);
      epscell_grad[i] = g0+g1+g2+g3+g4;
    }
  }

  char output_filename[PETSC_MAX_PATH_LEN];
  if((count%printdof)==0){
    sprintf(output_filename,"cell%d_outputdof_step%d.txt",colour,count);
    writetofiledouble(subcomm,output_filename,epscell,ndofcell);
  }

  count++;
  free(u);
  free(_u0);
  free(_u1);
  free(_u2);
  free(_u3);
  free(_u4);
  return dispsum_cell;

}



#undef __FUNCT__
#define __FUNCT__ "phopt_dispsum_epsglobal_localabs"
double phopt_dispsum_epsglobal_localabs(int ndofAll, double *epsAll, double *epsAll_grad, void *data)
{

  Phasefielddata *ptdata = (Phasefielddata *) data;
  int colour = ptdata->colour;
  MPI_Comm subcomm = ptdata->subcomm;
  PetscReal *mask_order = ptdata->mask_order;
  DOFInfo *dofi = ptdata->dofi;
  int printdof = ptdata->printdof;
  int magsq = ptdata->magsq;

  int i;
  int ndof=dofi->ndof;
  PetscScalar *u;
  u = (PetscScalar *) malloc(ndof*sizeof(PetscScalar));
  for(i=0;i<ndof;i++){
    u[i]=epsAll[i+colour*ndof]+PETSC_i*0.0;
  }

  FiltersToolBox *flt = ptdata->flt;
  PetscScalar *_u0,*_u1,*_u2,*_u3,*_u4;
  _u0 = (PetscScalar *) malloc(ndof*sizeof(PetscScalar));
  _u1 = (PetscScalar *) malloc(ndof*sizeof(PetscScalar));
  _u2 = (PetscScalar *) malloc(ndof*sizeof(PetscScalar));
  _u3 = (PetscScalar *) malloc(ndof*sizeof(PetscScalar));
  _u4 = (PetscScalar *) malloc(ndof*sizeof(PetscScalar));
  filters_apply(subcomm,u,ptdata->eps_dof,flt,1);
  phgrad(data);
  filters_apply(subcomm,ptdata->grad0,_u0,flt,-1);
  filters_apply(subcomm,ptdata->grad1,_u1,flt,-1);
  filters_apply(subcomm,ptdata->grad2,_u2,flt,-1);
  filters_apply(subcomm,ptdata->grad3,_u3,flt,-1);
  filters_apply(subcomm,ptdata->grad4,_u4,flt,-1);
  MPI_Barrier(subcomm);
  MPI_Barrier(PETSC_COMM_WORLD);

  int subrank;
  MPI_Comm_rank(subcomm, &subrank);

  PetscReal g0,g1,g2,g3,g4;
  PetscReal dispsum_local=0, dispsum_global;
  if(subrank==0) {
    g0=mask_order[0]*        *(ptdata->obj0);
    g1=mask_order[1]* magfun(*(ptdata->obj1),magsq);
    g2=mask_order[2]* magfun(*(ptdata->obj2),magsq);
    g3=mask_order[3]* magfun(*(ptdata->obj3),magsq);
    g4=mask_order[4]* magfun(*(ptdata->obj4),magsq);
    dispsum_local= g0+g1+g2+g3+g4;
  }

  MPI_Allreduce(&dispsum_local,&dispsum_global,1,MPIU_REAL,MPI_SUM,PETSC_COMM_WORLD);  

  PetscPrintf(PETSC_COMM_WORLD,"***dispersion sum at step %d is %.8g \n",count,dispsum_global);
  
  double *tmp;
  tmp = (double *) malloc(ndofAll*sizeof(double));
  for(i=0;i<ndofAll;i++){
    if(subrank==0 && i>=colour*ndof && i<colour*ndof+ndof){
      g0 = mask_order[0] *                                    creal(_u0[i-colour*ndof]);
      g1 = mask_order[1] * magfun_prime(*ptdata->obj1,magsq) *creal(_u1[i-colour*ndof]);
      g2 = mask_order[2] * magfun_prime(*ptdata->obj2,magsq) *creal(_u2[i-colour*ndof]);
      g3 = mask_order[3] * magfun_prime(*ptdata->obj3,magsq) *creal(_u3[i-colour*ndof]);
      g4 = mask_order[4] * magfun_prime(*ptdata->obj4,magsq) *creal(_u4[i-colour*ndof]);
      tmp[i]= g0+g1+g2+g3+g4;
    }else{
      tmp[i] = 0.0;
    }
  }
  MPI_Allreduce(tmp,epsAll_grad,ndofAll,MPI_DOUBLE,MPI_SUM,PETSC_COMM_WORLD);

  char output_filename[PETSC_MAX_PATH_LEN];
  if((count%printdof)==0){
    sprintf(output_filename,"outputdof_step%d.txt",count);
    writetofiledouble(PETSC_COMM_WORLD,output_filename,epsAll,ndofAll);
  }

  free(u);
  free(_u0);
  free(_u1);
  free(_u2);
  free(_u3);
  free(_u4);
  free(tmp);
  count++;
  return dispsum_global;
  
}



#undef __FUNCT__
#define __FUNCT__ "phopt_dispsum_epsglobal_globalabs"
double phopt_dispsum_epsglobal_globalabs(int ndofAll, double *epsAll, double *epsAll_grad, void *data)
{

  Phasefielddata *ptdata = (Phasefielddata *) data;
  int colour = ptdata->colour;
  MPI_Comm subcomm = ptdata->subcomm;
  PetscReal *mask_order = ptdata->mask_order;
  DOFInfo *dofi = ptdata->dofi;
  int printdof = ptdata->printdof;
  int magsq = ptdata->magsq;

  int i;
  int ndof=dofi->ndof;
  PetscScalar *u;
  u = (PetscScalar *) malloc(ndof*sizeof(PetscScalar));
  for(i=0;i<ndof;i++){
    u[i]=epsAll[i+colour*ndof]+PETSC_i*0.0;
  }

  FiltersToolBox *flt = ptdata->flt;
  PetscScalar *_u0,*_u1,*_u2,*_u3,*_u4;
  _u0 = (PetscScalar *) malloc(ndof*sizeof(PetscScalar));
  _u1 = (PetscScalar *) malloc(ndof*sizeof(PetscScalar));
  _u2 = (PetscScalar *) malloc(ndof*sizeof(PetscScalar));
  _u3 = (PetscScalar *) malloc(ndof*sizeof(PetscScalar));
  _u4 = (PetscScalar *) malloc(ndof*sizeof(PetscScalar));
  filters_apply(subcomm,u,ptdata->eps_dof,flt,1);
  phgrad(data);
  filters_apply(subcomm,ptdata->grad0,_u0,flt,-1);
  filters_apply(subcomm,ptdata->grad1,_u1,flt,-1);
  filters_apply(subcomm,ptdata->grad2,_u2,flt,-1);
  filters_apply(subcomm,ptdata->grad3,_u3,flt,-1);
  filters_apply(subcomm,ptdata->grad4,_u4,flt,-1);
  MPI_Barrier(subcomm);
  MPI_Barrier(PETSC_COMM_WORLD);

  int subrank;
  MPI_Comm_rank(subcomm, &subrank);

  PetscReal obj_local[5]={0,0,0,0,0};
  PetscReal obj_global[5]={0,0,0,0,0};
  if(subrank==0) {
    obj_local[0] = *(ptdata->obj0);
    obj_local[1] = *(ptdata->obj1);
    obj_local[2] = *(ptdata->obj2);
    obj_local[3] = *(ptdata->obj3);
    obj_local[4] = *(ptdata->obj4);
  }
  MPI_Allreduce(obj_local,obj_global,5,MPIU_REAL,MPI_SUM,PETSC_COMM_WORLD);  

  PetscReal dispsum_global = mask_order[0] * obj_global[0] \
    + mask_order[1] * magfun(obj_global[1],magsq) \
    + mask_order[2] * magfun(obj_global[2],magsq) \
    + mask_order[3] * magfun(obj_global[3],magsq) \
    + mask_order[4] * magfun(obj_global[4],magsq);
  dispsum_global/=ptdata->norm_global;

  PetscPrintf(PETSC_COMM_WORLD,"***obj0 at step %d is %.8g \n",count,obj_global[0]);
  PetscPrintf(PETSC_COMM_WORLD,"***obj1 at step %d is %.8g \n",count,obj_global[1]);
  PetscPrintf(PETSC_COMM_WORLD,"***obj2 at step %d is %.8g \n",count,obj_global[2]);
  PetscPrintf(PETSC_COMM_WORLD,"***obj3 at step %d is %.8g \n",count,obj_global[3]);
  PetscPrintf(PETSC_COMM_WORLD,"***obj4 at step %d is %.8g \n",count,obj_global[4]);
  PetscPrintf(PETSC_COMM_WORLD,"***dispersion sum at step %d is %.8g \n",count,dispsum_global);
  
  PetscReal g0,g1,g2,g3,g4;
  double *tmp;
  tmp = (double *) malloc(ndofAll*sizeof(double));
  for(i=0;i<ndofAll;i++){
    if(subrank==0 && i>=colour*ndof && i<colour*ndof+ndof){
      g0 = mask_order[0] *                                    creal(_u0[i-colour*ndof]);
      g1 = mask_order[1] * magfun_prime(obj_global[1],magsq) *creal(_u1[i-colour*ndof]);
      g2 = mask_order[2] * magfun_prime(obj_global[2],magsq) *creal(_u2[i-colour*ndof]);
      g3 = mask_order[3] * magfun_prime(obj_global[3],magsq) *creal(_u3[i-colour*ndof]);
      g4 = mask_order[4] * magfun_prime(obj_global[4],magsq) *creal(_u4[i-colour*ndof]);
      tmp[i]= (g0+g1+g2+g3+g4)/ptdata->norm_global;
    }else{
      tmp[i] = 0.0;
    }
  }
  MPI_Allreduce(tmp,epsAll_grad,ndofAll,MPIU_REAL,MPI_SUM,PETSC_COMM_WORLD);

  char output_filename[PETSC_MAX_PATH_LEN];
  if((count%printdof)==0){
    sprintf(output_filename,"outputdof_step%d.txt",count);
    writetofiledouble(PETSC_COMM_WORLD,output_filename,epsAll,ndofAll);
  }

  free(u);
  free(_u0);
  free(_u1);
  free(_u2);
  free(_u3);
  free(_u4);
  free(tmp);
  count++;
  return dispsum_global;
  
}

