#include "fields.h"

#undef __FUNCT__
#define __FUNCT__ "vec_at_xyslab_linpol"
PetscErrorCode vec_at_xyslab_linpol(MPI_Comm comm, DM da, Vec v, Vec u, paramfun f, void *params, PetscInt pol, PetscReal x_offset, PetscReal y_offset, PetscReal *z_ref, PetscReal hx, PetscReal hy, PetscReal hz, int verbose)
{
  PetscErrorCode ierr;

  PetscInt izref=floor(*z_ref/hz);
  *z_ref=(izref+1/2)*hz;
  if(verbose) PetscPrintf(comm,"\tGenerating the vector v over the xy plane for either source, far-field (z) projection or near-field phase matching.\n\
\tthe ref plane is positioned at z_index = %d corresponding to z = %g while x_offset, y_offset = %g, %g\n",izref,*z_ref,x_offset,y_offset);

  Field ***_v,***_u;
  ierr = DMDAVecGetArray(da, v, &_v); CHKERRQ(ierr);
  ierr = DMDAVecGetArray(da, u, &_u); CHKERRQ(ierr);

  /** Get corners and widths of Yee's grid included in this proces. */
  PetscInt ox, oy, oz;  // coordinates of beginning corner of Yee's grid in this process
  PetscInt nx, ny, nz;  // local widths of Yee's grid in this process
  ierr = DMDAGetCorners(da, &ox, &oy, &oz, &nx, &ny, &nz); CHKERRQ(ierr);

  PetscScalar val;
  PetscReal x,y;
  PetscInt ix,iy,iz,ic;
  for (iz = oz; iz < oz+nz; ++iz) {
    for (iy = oy; iy < oy+ny; ++iy) {
      for (ix = ox; ix < ox+nx; ++ix) {
        for (ic = 0; ic < Naxis; ++ic) {
          /** v_array is just the array-representation of the vector v.  So,
              setting values on v_array is actually setting values on v.*/
          if(iz==izref && ic==pol){
            x = x_offset + (ix+1/2)*hx;
            y = y_offset + (iy+1/2)*hy;
            f(x,y,&val,params);
            _v[iz][iy][ix].comp[ic]=val;
	    _u[iz][iy][ix].comp[ic]=1.0+PETSC_i*0.0;

          }else{
            _v[iz][iy][ix].comp[ic]=0.0+PETSC_i*0.0;
	    _u[iz][iy][ix].comp[ic]=0.0+PETSC_i*0.0;
          }
        }
      }
    }
  }

  ierr = DMDAVecRestoreArray(da, v, &_v); CHKERRQ(ierr);
  ierr = DMDAVecRestoreArray(da, u, &_u); CHKERRQ(ierr);

  return ierr;


}

#undef __FUNCT__
#define __FUNCT__ "planewave"
PetscScalar planewave(PetscReal x, PetscReal y, PetscScalar *val, void *data)
{
  pwparams *ptdata=(pwparams *) data;
  PetscReal freq = ptdata->freq;
  PetscReal nmed = ptdata->nmed;
  PetscReal theta = ptdata->theta_rad;
  PetscReal phi = ptdata->phi_rad;
  PetscReal kappa_x = 2 * M_PI * freq * nmed * x;
  PetscReal kappa_y = 2 * M_PI * freq * nmed * y;
  PetscScalar I1 = 0.0 + 1.0 * PETSC_i;

  *val=cexp( I1 * kappa_x * sin(theta) * cos(phi) + I1 * kappa_y * sin(theta) * sin(phi) );

  PetscFunctionReturn(0);
}


#undef __FUNCT__
#define __FUNCT__ "farfield2d"
PetscScalar farfield2d(PetscReal x_local, PetscReal y_local, PetscScalar *val, void *data)
{
  ffparams *ptdata=(ffparams *) data;
  PetscReal freq = ptdata->freq;
  PetscReal nmed = ptdata->nmed;
  PetscReal x_far = ptdata->x_far;
  PetscReal y_far = ptdata->y_far;
  PetscReal z_far = ptdata->z_far;
  PetscReal z_local = ptdata->z_local;
  PetscReal k = 2 * M_PI * freq * nmed;
  PetscReal dr = sqrt(pow(x_far-x_local,2)+pow(y_far-y_local,2)+pow(z_far-z_local,2));

  *val=cexp( - PETSC_i * k * dr ) / sqrt(dr);

  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "farfield3d"
PetscScalar farfield3d(PetscReal x_local, PetscReal y_local, PetscScalar *val, void *data)
{
  ffparams *ptdata=(ffparams *) data;
  PetscReal freq = ptdata->freq;
  PetscReal nmed = ptdata->nmed;
  PetscReal x_far = ptdata->x_far;
  PetscReal y_far = ptdata->y_far;
  PetscReal z_far = ptdata->z_far;
  PetscReal z_local = ptdata->z_local;
  PetscReal k = 2 * M_PI * freq * nmed;
  PetscReal dr = sqrt(pow(x_far-x_local,2)+pow(y_far-y_local,2)+pow(z_far-z_local,2));

  *val=cexp( - PETSC_i * k * dr ) / dr;

  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "nearfieldlens"
PetscScalar nearfieldlens(PetscReal x_local, PetscReal y_local, PetscScalar *val, void *data)
{
  ffparams *ptdata=(ffparams *) data;
  PetscReal freq = ptdata->freq;
  PetscReal nmed = ptdata->nmed;
  PetscReal x_far = ptdata->x_far;
  PetscReal y_far = ptdata->y_far;
  PetscReal z_far = ptdata->z_far;
  PetscReal z_local = ptdata->z_local;
  PetscReal k = 2 * M_PI * freq * nmed;
  PetscReal dr = sqrt(pow(x_far-x_local,2)+pow(y_far-y_local,2)+pow(z_far-z_local,2));

  *val=cexp( PETSC_i * k * (dr-z_far) );

  PetscFunctionReturn(0);
}
