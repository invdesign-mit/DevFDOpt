#include "farfield.h"

#undef __FUNCT__
#define __FUNCT__ "hankel01"
PetscScalar hankel01(double z)
{
  return j0(z) + PETSC_i*y0(z);
}

#undef __FUNCT__
#define __FUNCT__ "ff2dxz"
PetscErrorCode ff2dxz(MPI_Comm comm, DM da, Vec v, PetscReal x, PetscReal z, PetscReal freq, PetscReal xp_offset, PetscReal *zp_ref, PetscReal hx, PetscReal hz, int verbose) 
{

  PetscInt izprime=floor(*zp_ref/hz);
  *zp_ref=(izprime+1/2)*hz;
  PetscReal zprime=*zp_ref;
  if(verbose) PetscPrintf(comm,"\tGenerating the far-field convoluter in 2D x-z plane for Ey polarization.\n\
\tNote that the exact Hankel function is used here.\n\
\tthe ref line is set at z_index = %d corresponding to z = %g while xprime_offset = %g\n",izprime,*zp_ref,xp_offset);
  
  PetscScalar val;

  PetscErrorCode ierr;

  PetscReal k=2*M_PI*freq;

  Field ***v_array;  // 3D array that images Vec v
  ierr = DMDAVecGetArray(da, v, &v_array); CHKERRQ(ierr);

  /** Get corners and widths of Yee's grid included in this proces. */
  PetscInt ox, oy, oz;  // coordinates of beginning corner of Yee's grid in this process
  PetscInt nx, ny, nz;  // local widths of Yee's grid in this process
  ierr = DMDAGetCorners(da, &ox, &oy, &oz, &nx, &ny, &nz); CHKERRQ(ierr);

  PetscInt ix,iy,iz,ic;
  PetscReal xprime, dr;
  for (iz = oz; iz < oz+nz; ++iz) {
    for (iy = oy; iy < oy+ny; ++iy) {
      for (ix = ox; ix < ox+nx; ++ix) {
	for (ic = 0; ic < Naxis; ++ic) {
	  /** v_array is just the array-representation of the vector v.  So,
	      setting values on v_array is actually setting values on v.*/
	  val=0.0;
	  if(iz==izprime && ic==1){
	    xprime = xp_offset + (ix+1/2)*hx;
	    dr = sqrt((x-xprime)*(x-xprime) + (z-zprime)*(z-zprime));
	    //val = (PETSC_i * k/4) * hankel01(k*dr);
	    val = cexp(-PETSC_i * k * dr)/sqrt(dr);
	  }
	  v_array[iz][iy][ix].comp[ic]=val;
		    
	}
      }
    }
  }

  ierr = DMDAVecRestoreArray(da, v, &v_array); CHKERRQ(ierr);



  return ierr;
}

#undef __FUNCT__
#define __FUNCT__ "ff2dxz_asymp"
PetscErrorCode ff2dxz_asymp(MPI_Comm comm, DM da, Vec v, PetscReal r, PetscReal theta_far_rad, PetscReal freq, PetscReal xp_offset, PetscReal *zp_ref, PetscReal hx, PetscReal hz, int verbose) 
{

  PetscInt izprime=floor(*zp_ref/hz);
  *zp_ref=(izprime+1/2)*hz;
  PetscReal zprime=*zp_ref;
  if(verbose) PetscPrintf(comm,"\tGenerating the far-field convoluter in 2D x-z plane for Ey polarization.\n\
\tNote that the exact Hankel function is used here.\n\
\tthe ref line is set at z_index = %d corresponding to z = %g while xprime_offset = %g\n",izprime,*zp_ref,xp_offset);
  
  PetscScalar val;

  PetscErrorCode ierr;

  PetscReal k=2*M_PI*freq;

  Field ***v_array;  // 3D array that images Vec v
  ierr = DMDAVecGetArray(da, v, &v_array); CHKERRQ(ierr);

  /** Get corners and widths of Yee's grid included in this proces. */
  PetscInt ox, oy, oz;  // coordinates of beginning corner of Yee's grid in this process
  PetscInt nx, ny, nz;  // local widths of Yee's grid in this process
  ierr = DMDAGetCorners(da, &ox, &oy, &oz, &nx, &ny, &nz); CHKERRQ(ierr);

  PetscInt ix,iy,iz,ic;
  PetscReal xprime;
  for (iz = oz; iz < oz+nz; ++iz) {
    for (iy = oy; iy < oy+ny; ++iy) {
      for (ix = ox; ix < ox+nx; ++ix) {
	for (ic = 0; ic < Naxis; ++ic) {
	  /** v_array is just the array-representation of the vector v.  So,
	      setting values on v_array is actually setting values on v.*/
	  val=0.0;
	  if(iz==izprime && ic==1){
	    xprime = xp_offset + (ix+1/2)*hx;
	    val = ( cexp(-PETSC_i*k*r)/sqrt(r) ) * cexp(PETSC_i*k*sin(theta_far_rad)*xprime);
	  }
	  v_array[iz][iy][ix].comp[ic]=val;
		    
	}
      }
    }
  }

  ierr = DMDAVecRestoreArray(da, v, &v_array); CHKERRQ(ierr);



  return ierr;
}

#undef __FUNCT__
#define __FUNCT__ "pwsrc_2dxz"
PetscErrorCode pwsrc_2dxz(MPI_Comm comm, DM da, Vec v, PetscReal freq, PetscReal refractive_index, PetscReal theta, PetscScalar amp, PetscReal x_offset, PetscReal *z_ref, PetscReal hx, PetscReal hz, int verbose) 
{

  PetscReal kx=2*M_PI*freq*refractive_index*sin(theta*M_PI/180);

  PetscInt iz0=floor(*z_ref/hz);
  *z_ref=(iz0+1/2)*hz;
  if(verbose) PetscPrintf(comm,"\tGenerating the plane wave source with angle %g deg (kx=%g) in 2D x-z plane for Jy polarization.\n\
\tthe source line is set at z_index = %d corresponding to z = %g while x_offset = %g\n",theta,kx,iz0,*z_ref,x_offset);
  
  PetscScalar val;

  PetscErrorCode ierr;

  Field ***v_array;  // 3D array that images Vec v
  ierr = DMDAVecGetArray(da, v, &v_array); CHKERRQ(ierr);

  /** Get corners and widths of Yee's grid included in this proces. */
  PetscInt ox, oy, oz;  // coordinates of beginning corner of Yee's grid in this process
  PetscInt nx, ny, nz;  // local widths of Yee's grid in this process
  ierr = DMDAGetCorners(da, &ox, &oy, &oz, &nx, &ny, &nz); CHKERRQ(ierr);

  PetscInt ix,iy,iz,ic;
  PetscReal x;
  for (iz = oz; iz < oz+nz; ++iz) {
    for (iy = oy; iy < oy+ny; ++iy) {
      for (ix = ox; ix < ox+nx; ++ix) {
	for (ic = 0; ic < Naxis; ++ic) {
	  /** v_array is just the array-representation of the vector v.  So,
	      setting values on v_array is actually setting values on v.*/
	  val=0.0;
	  if(iz==iz0 && ic==1){
	    x = x_offset + (ix+1/2)*hx;
	    val = amp*cexp(PETSC_i * kx * x);
	  }
	  v_array[iz][iy][ix].comp[ic]=val;
		    
	}
      }
    }
  }

  ierr = DMDAVecRestoreArray(da, v, &v_array); CHKERRQ(ierr);



  return ierr;
}
