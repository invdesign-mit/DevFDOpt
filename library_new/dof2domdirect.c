#include "dof2domdirect.h"

#undef __FUNCT__
#define __FUNCT__ "multilayer_forward"
PetscErrorCode multilayer_forward(PetscScalar *pt, Vec v, DOFInfo *dofi, DM da)
{

  PetscInt nlayers = dofi->nlayers;
  PetscInt Mx = dofi->Mx;
  PetscInt My = dofi->My;
  PetscInt *Mz = dofi->Mz;
  PetscInt Mzslab = dofi->Mzslab;
  PetscInt Nxo = dofi->Nxo;
  PetscInt Nyo = dofi->Nyo;
  PetscInt *Nzo = dofi->Nzo;
  PetscScalar val;

  PetscErrorCode ierr;

  Field ***v_array;  // 3D array that images Vec v
  ierr = DMDAVecGetArray(da, v, &v_array); CHKERRQ(ierr);

  /** Get corners and widths of Yee's grid included in this proces. */
  PetscInt ox, oy, oz;  // coordinates of beginning corner of Yee's grid in this process
  PetscInt nx, ny, nz;  // local widths of Yee's grid in this process
  ierr = DMDAGetCorners(da, &ox, &oy, &oz, &nx, &ny, &nz); CHKERRQ(ierr);

  PetscInt ix,iy,iz,ic,px,py,pz,ilayer,indp;
  for (iz = oz; iz < oz+nz; ++iz) {
    for (iy = oy; iy < oy+ny; ++iy) {
      for (ix = ox; ix < ox+nx; ++ix) {
	for (ic = 0; ic < Naxis; ++ic) {
	  /** v_array is just the array-representation of the vector v.  So,
	      setting values on v_array is actually setting values on v.*/
	  val=0.0;
	  if(ix>=Nxo && ix<Nxo+Mx){
	    px=ix-Nxo;
	    if(iy>=Nyo && iy<Nyo+My){
	      py=iy-Nyo;
	      for(ilayer=0;ilayer<nlayers;ilayer++){
		if(iz>=Nzo[ilayer] && iz<Nzo[ilayer]+Mz[ilayer]){
		    pz = (Mzslab==0) ? iz-Nzo[ilayer] : 0 ;
		    indp = ilayer + nlayers*px + nlayers*Mx*py + nlayers*Mx*My*pz;
		    val=pt[indp];
		}
	      }
	    }
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
#define __FUNCT__ "multilayer_backward"
PetscErrorCode multilayer_backward(MPI_Comm comm, Vec v, PetscScalar *pt, DOFInfo *dofi, DM da)
{

  PetscInt nlayers = dofi->nlayers;
  PetscInt Mx = dofi->Mx;
  PetscInt My = dofi->My;
  PetscInt *Mz = dofi->Mz;
  PetscInt Mzslab = dofi->Mzslab;
  PetscInt Nxo = dofi->Nxo;
  PetscInt Nyo = dofi->Nyo;
  PetscInt *Nzo = dofi->Nzo;
  PetscInt ndof = dofi->ndof;

  PetscErrorCode ierr;

  PetscInt i;
  PetscScalar *local_pt;
  local_pt = (PetscScalar *) malloc(ndof*sizeof(PetscScalar));
  for(i=0;i<ndof;i++) local_pt[i]=0.0+PETSC_i*0.0;
  MPI_Barrier(comm);

  Field ***v_array;  // 3D array that images Vec v
  ierr = DMDAVecGetArray(da, v, &v_array); CHKERRQ(ierr);

  /** Get corners and widths of Yee's grid included in this proces. */
  PetscInt ox, oy, oz;  // coordinates of beginning corner of Yee's grid in this process
  PetscInt nx, ny, nz;  // local widths of Yee's grid in this process
  ierr = DMDAGetCorners(da, &ox, &oy, &oz, &nx, &ny, &nz); CHKERRQ(ierr);

  PetscInt ix,iy,iz,ic,px,py,pz,ilayer,indp;
  for (iz = oz; iz < oz+nz; ++iz) {
    for (iy = oy; iy < oy+ny; ++iy) {
      for (ix = ox; ix < ox+nx; ++ix) {
	for (ic = 0; ic < Naxis; ++ic) {
	  if(ix>=Nxo && ix<Nxo+Mx){
	    px=ix-Nxo;
	    if(iy>=Nyo && iy<Nyo+My){
	      py=iy-Nyo;
	      for(ilayer=0;ilayer<nlayers;ilayer++){
		if(iz>=Nzo[ilayer] && iz<Nzo[ilayer]+Mz[ilayer]){
		    pz = (Mzslab==0) ? iz-Nzo[ilayer] : 0 ;
		    indp = ilayer + nlayers*px + nlayers*Mx*py + nlayers*Mx*My*pz;
		    local_pt[indp] = local_pt[indp]+v_array[iz][iy][ix].comp[ic];
		}
	      }
	    }
	  }
		    
	}
      }
    }
  }

  ierr = DMDAVecRestoreArray(da, v, &v_array); CHKERRQ(ierr);
  MPI_Barrier(comm);

  MPI_Allreduce(local_pt,pt,ndof,MPIU_SCALAR,MPI_SUM,comm);
  MPI_Barrier(comm);

  free(local_pt);

  return ierr;
}
