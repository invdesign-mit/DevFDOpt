#include "sym.h"

extern int count;

#undef __FUNCT__
#define __FUNCT__ "symx"
void symx(PetscReal *u_half, PetscReal *u_full, int mx,int my,int nlayers,int mxcells,int mycells, int transpose)
{

  int nx_half=mx*mxcells;
  int mxcells_full=2*mxcells;
  
  int iycell_half_tandem, ixcell_half_tandem, iy_half_tandem, ix_half_tandem;
  int ilayer;
  int i_half_tandem;
  int ix_half_natural;
  int ix1_full_natural,ix2_full_natural;
  int ix1_full_tandem, ix2_full_tandem;
  int ix1cell_full_tandem,ix2cell_full_tandem;
  int iy_full_tandem,iycell_full_tandem;
  int i1_full_tandem,i2_full_tandem;
  
  for(iycell_half_tandem=0; iycell_half_tandem<mycells; iycell_half_tandem++){
    for(ixcell_half_tandem=0; ixcell_half_tandem<mxcells; ixcell_half_tandem++){
      for(iy_half_tandem=0; iy_half_tandem<my; iy_half_tandem++){
	for(ix_half_tandem=0; ix_half_tandem<mx; ix_half_tandem++){
	  for(ilayer=0; ilayer<nlayers; ilayer++){

	    i_half_tandem = ilayer + nlayers*ix_half_tandem + nlayers*mx*iy_half_tandem + nlayers*mx*my*ixcell_half_tandem + nlayers*mx*my*mxcells*iycell_half_tandem;

	    ix_half_natural=ix_half_tandem+mx*ixcell_half_tandem;
	    
	    //ix1_full_natural=nx_half+ix_half_natural;
	    //ix2_full_natural=nx_half-ix_half_natural-1;
	    ix1_full_natural=ix_half_natural;
	    ix2_full_natural=2*nx_half-ix_half_natural-1;

	    ix1_full_tandem=ix1_full_natural%mx;
	    ix1cell_full_tandem=ix1_full_natural/mx;
	    ix2_full_tandem=ix2_full_natural%mx;
	    ix2cell_full_tandem=ix2_full_natural/mx;
	    
	    iy_full_tandem=iy_half_tandem;
	    iycell_full_tandem=iycell_half_tandem;

	    i1_full_tandem = ilayer + nlayers*ix1_full_tandem + nlayers*mx*iy_full_tandem + nlayers*mx*my*ix1cell_full_tandem + nlayers*mx*my*mxcells_full*iycell_full_tandem;
	    i2_full_tandem = ilayer + nlayers*ix2_full_tandem + nlayers*mx*iy_full_tandem + nlayers*mx*my*ix2cell_full_tandem + nlayers*mx*my*mxcells_full*iycell_full_tandem;

	    if(transpose==0){
	      u_full[i1_full_tandem]=u_half[i_half_tandem];
	      u_full[i2_full_tandem]=u_half[i_half_tandem];
	    }else{
	      u_half[i_half_tandem]=u_full[i1_full_tandem]+u_full[i2_full_tandem];
	    }

	  }
	}
      }
    }
  }

	    
}


#undef __FUNCT__
#define __FUNCT__ "ffoptsym"
double ffoptsym(int ndof_half, double *dof_half, double *dofgrad_half, void *data)
{

  Farfielddata *ptdata = (Farfielddata *) data;

  DOFInfo *dofi = ptdata->dofi;
  
  int ndofAll=2*ndof_half;
  int ncells_half=ndof_half/dofi->ndof;
  double *dofAll=(double *)malloc(ndofAll*sizeof(double));
  double *gradAll=(double *)malloc(ndofAll*sizeof(double));
  symx(dof_half, dofAll, dofi->Mx,dofi->My,dofi->nlayers,ncells_half,1, 0);
  MPI_Barrier(PETSC_COMM_WORLD);
  
  double objval = ffopt(ndofAll,dofAll,gradAll,data);
  symx(dofgrad_half,gradAll, dofi->Mx,dofi->My,dofi->nlayers,ncells_half,1, 1);

  free(dofAll);
  free(gradAll);

  return objval;
}

#undef __FUNCT__
#define __FUNCT__ "ffoptsym_maximinconstraint"
double ffoptsym_maximinconstraint(int ndofhalf_with_dummy, double *dofhalf_with_dummy, double *dofgradhalf_with_dummy, void *data)
{
  int ndofhalf=ndofhalf_with_dummy-1;
  double *dofhalf,*dofgradhalf;
  dofhalf=(double *) malloc(ndofhalf*sizeof(double));
  dofgradhalf=(double *) malloc(ndofhalf*sizeof(double));
  int i;
  for(i=0;i<ndofhalf;i++){
    dofhalf[i]=dofhalf_with_dummy[i];
  }

  double obj=ffoptsym(ndofhalf,dofhalf,dofgradhalf,data);

  for(i=0;i<ndofhalf;i++){
    dofgradhalf_with_dummy[i]=-1.0*dofgradhalf[i];
  }
  dofgradhalf_with_dummy[ndofhalf]=1.0;

  count--;
  free(dofhalf);
  free(dofgradhalf);
  return dofhalf_with_dummy[ndofhalf]-obj;

}
