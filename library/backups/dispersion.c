#include "dispersion.h"

#undef __FUNCT__
#define __FUNCT__ "dispersiveV_2d"
PetscErrorCode dispersiveV_2d(MPI_Comm comm, DM da, Vec v0, Vec v1, Vec v2, Vec v3, Vec v4, Vec u, paramfun f, void *params, PetscReal x_offset, PetscReal *z_ref, PetscReal hx, PetscReal hz, int verbose)
{
  PetscErrorCode ierr;

  PetscInt izref=floor(*z_ref/hz);
  *z_ref=(izref+1/2)*hz;
  if(verbose) PetscPrintf(comm,"\tGenerating the dispersive vector (to 4th order) in 2D x-z plane for Ey polarization.\n\
\tthe ref plane is positioned at z_index = %d corresponding to z = %g while xprime_offset = %g\n",izref,*z_ref,x_offset);

  Field ***_v0,***_v1,***_v2,***_v3,***_v4,***_u;
  ierr = DMDAVecGetArray(da, v0, &_v0); CHKERRQ(ierr);
  ierr = DMDAVecGetArray(da, v1, &_v1); CHKERRQ(ierr);
  ierr = DMDAVecGetArray(da, v2, &_v2); CHKERRQ(ierr);
  ierr = DMDAVecGetArray(da, v3, &_v3); CHKERRQ(ierr);
  ierr = DMDAVecGetArray(da, v4, &_v4); CHKERRQ(ierr);
  ierr = DMDAVecGetArray(da, u , &_u ); CHKERRQ(ierr);

  /** Get corners and widths of Yee's grid included in this proces. */
  PetscInt ox, oy, oz;  // coordinates of beginning corner of Yee's grid in this process
  PetscInt nx, ny, nz;  // local widths of Yee's grid in this process
  ierr = DMDAGetCorners(da, &ox, &oy, &oz, &nx, &ny, &nz); CHKERRQ(ierr);

  PetscScalar val[5];
  PetscReal x;
  PetscInt ix,iy,iz,ic;
  for (iz = oz; iz < oz+nz; ++iz) {
    for (iy = oy; iy < oy+ny; ++iy) {
      for (ix = ox; ix < ox+nx; ++ix) {
        for (ic = 0; ic < Naxis; ++ic) {
          /** v_array is just the array-representation of the vector v.  So,
              setting values on v_array is actually setting values on v.*/
          if(iz==izref && ic==1){
            x = x_offset + (ix+1/2)*hx;
            f(x,val,params);
            _v0[iz][iy][ix].comp[ic]=val[0];
            _v1[iz][iy][ix].comp[ic]=val[1];
            _v2[iz][iy][ix].comp[ic]=val[2];
	    _v3[iz][iy][ix].comp[ic]=val[3];
            _v4[iz][iy][ix].comp[ic]=val[4];
	    _u[iz][iy][ix].comp[ic]=1.0+PETSC_i*0.0;

          }else{
            _v0[iz][iy][ix].comp[ic]=0.0+PETSC_i*0.0;
            _v1[iz][iy][ix].comp[ic]=0.0+PETSC_i*0.0;
            _v2[iz][iy][ix].comp[ic]=0.0+PETSC_i*0.0;
            _v3[iz][iy][ix].comp[ic]=0.0+PETSC_i*0.0;
            _v4[iz][iy][ix].comp[ic]=0.0+PETSC_i*0.0;
	    _u[iz][iy][ix].comp[ic]=0.0+PETSC_i*0.0;
          }
        }
      }
    }
  }

  ierr = DMDAVecRestoreArray(da, v0, &_v0); CHKERRQ(ierr);
  ierr = DMDAVecRestoreArray(da, v1, &_v1); CHKERRQ(ierr);
  ierr = DMDAVecRestoreArray(da, v2, &_v2); CHKERRQ(ierr);
  ierr = DMDAVecRestoreArray(da, v3, &_v3); CHKERRQ(ierr);
  ierr = DMDAVecRestoreArray(da, v4, &_v4); CHKERRQ(ierr);
  ierr = DMDAVecRestoreArray(da, u , &_u ); CHKERRQ(ierr);

  return ierr;


}


#undef __FUNCT__
#define __FUNCT__ "planewave_phaseonly"
PetscScalar planewave_phaseonly(PetscReal x, PetscScalar *val, void *data)
{
  pwparams *ptdata=(pwparams *) data;
  PetscReal freq = ptdata->freq;
  PetscReal nmed = ptdata->nmed;
  PetscReal theta = ptdata->theta_rad;
  PetscReal d1theta = ptdata->d1theta;
  PetscReal d2theta = ptdata->d2theta;
  PetscReal d3theta = ptdata->d3theta;
  PetscReal d4theta = ptdata->d4theta;
  PetscReal kappa = 2 * M_PI * freq * nmed * x;
  PetscScalar I1 = 0.0 + 1.0 * PETSC_i;

  val[0]=cexp( I1 * kappa * sin(theta) );

  val[1]= kappa * cos(theta) * d1theta;
  val[2]=-kappa * sin(theta) * pow(d1theta,2) +  kappa * cos(theta) * d2theta;
  val[3]=-kappa * cos(theta) * pow(d1theta,3) - 3 * kappa * sin(theta) * d1theta * d2theta + kappa * cos(theta) * d3theta;
  val[4]= kappa * sin(theta) * pow(d1theta,4) - 6 * kappa * cos(theta) * pow(d1theta,2) * d2theta - 3 * kappa * sin(theta) * pow(d2theta,2) - 4 * kappa * sin(theta) * d1theta * d3theta + kappa * cos(theta) * d4theta;
  
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "planewave"
PetscScalar planewave(PetscReal x, PetscScalar *val, void *data)
{
  pwparams *ptdata=(pwparams *) data;
  PetscReal freq = ptdata->freq;
  PetscReal nmed = ptdata->nmed;
  PetscReal theta = ptdata->theta_rad;
  PetscReal d1theta = ptdata->d1theta;
  PetscReal d2theta = ptdata->d2theta;
  PetscReal d3theta = ptdata->d3theta;
  PetscReal d4theta = ptdata->d4theta;
  PetscReal kappa = 2 * M_PI * freq * nmed * x;
  PetscScalar I1 = 0.0 + 1.0 * PETSC_i;
  val[0]=cexp( I1 * kappa * sin(theta) );

  val[1]=val[0] * I1 * kappa * cos(theta) * d1theta;

  val[2]=val[1] * ( I1*kappa*cos(theta)*d1theta - \
		    tan(theta)*d1theta + \
		    d2theta/d1theta );
  
  val[3]=val[1] * ( -pow(d1theta,2) - \
		    pow(kappa,2)*pow(cos(theta),2)*pow(d1theta,2) - \
		    3*I1*kappa*sin(theta)*pow(d1theta,2) + \
		    3*I1*kappa*cos(theta)*d2theta - \
		    3*tan(theta)*d2theta + \
		    d3theta/d1theta );

  val[4]=val[1] * ( (-4)*I1*kappa*cos(theta)*pow(d1theta,3) - \
		    I1*pow(kappa,3)*pow(cos(theta),3)*pow(d1theta,3) + \
		    3*pow(kappa,2)*sin(2*theta)*pow(d1theta,3) + \
		    tan(theta)*pow(d1theta,3) + \
		    3*I1*kappa*sin(theta)*tan(theta)*pow(d1theta,3) - \
		    6*d1theta*d2theta - \
		    3*pow(kappa,2)*d1theta*d2theta - \
		    3*pow(kappa,2)*cos(2*theta)*d1theta*d2theta - \
		    18*I1*kappa*sin(theta)*d1theta*d2theta + \
		    (3*I1*kappa*cos(theta)*pow(d2theta,2))/d1theta - \
		    (3*tan(theta)*pow(d2theta,2))/d1theta + 4*I1*kappa*cos(theta)*d3theta - \
		    4*tan(theta)*d3theta + d4theta/d1theta );
  PetscFunctionReturn(0);
}
