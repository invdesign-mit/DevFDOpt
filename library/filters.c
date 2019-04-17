#include "filters.h"

#undef __FUNCT__
#define __FUNCT__ "create_density_filter_1d"
void create_density_filter_1d(MPI_Comm comm, Mat *Wout, int rp, int ndof_per_layer, int num_layers)
{

  PetscPrintf(comm,"Creating the linear (conic) density filter.\n");

  Mat W;
  int i,j,k,ilayer,ns,ne;
  int N=ndof_per_layer * num_layers;

  MatCreate(comm,&W);
  MatSetType(W,MATMPIAIJ);
  MatSetSizes(W,PETSC_DECIDE,PETSC_DECIDE, N,N);
  MatMPIAIJSetPreallocation(W, 2*rp+1, PETSC_NULL, 2*rp+1, PETSC_NULL);

  int nbh;
  int *col;
  PetscScalar *val;
  col = (int *) malloc((2*rp+1)*sizeof(int));
  val = (PetscScalar *) malloc((2*rp+1)*sizeof(PetscScalar));

  MatGetOwnershipRange(W, &ns, &ne);

  for(i=ns;i<ne;i++){
    //for each row (i), determine the layer index (ilayer) and the position index within the layer (j) 
    j=i%ndof_per_layer;
    ilayer=i/ndof_per_layer;
    //for each element i, there will be 2*rp + 1 neighbors ( including i ) within the distance rp 
    for(k=0;k<2*rp+1;k++){ //consider each of the 2*rp + 1 neighbors
      nbh = k+j-rp;  //the position index of each neighbor element
      if(nbh>=0 && nbh<ndof_per_layer){ // if the position index of the neighbor is within the layer
	col[k]=nbh + ilayer * ndof_per_layer;  //determine the ultimate position index of the neighbor element 
      }else if(nbh<0){  //if the position index of the neighbor is beyond the negative end (nbh<0) 
	col[k]=nbh+ndof_per_layer + ilayer * ndof_per_layer; //shift the position to the one near the positive end
      }else{  //if the position index of the neighbor is beyond the positive end (nbh>ndof_per_layer)
	col[k]=nbh-ndof_per_layer + ilayer * ndof_per_layer; //shift the position to the one near the negative end
      }
      val[k]=((double)(rp-abs(j-nbh)))/((double)(rp*rp)) + PETSC_i * 0; // the weight is given by (rp - distance btn the neighbor and the current element) normalized to 1 (conical weights) 
    }
    MatSetValues(W,1,&i,2*rp+1,col,val,INSERT_VALUES);
  }

  MatAssemblyBegin(W, MAT_FINAL_ASSEMBLY);
  MatAssemblyEnd(W, MAT_FINAL_ASSEMBLY);

  *Wout = W;

  free(col);
  free(val);
}

int findmin(int a, int b)
{

  return (a < b) ? a : b;

}
//******************
#undef __FUNCT__
#define __FUNCT__ "create_density_filter_2d"
void create_density_filter_2d(MPI_Comm comm, Mat *Wout, PetscReal rp, int mx_per_layer, int my_per_layer, int num_layers)
{

  PetscPrintf(comm,"Creating the linear (conic) density filter 2D.\n");

  int mx=mx_per_layer;
  int my=my_per_layer;
  int n=num_layers;
  int N=mx*my*n;

  int prealloc_tot=ceil(4*(rp+1)*(rp+1));

  Mat W;
  MatCreate(comm,&W);
  MatSetType(W,MATMPIAIJ);
  MatSetSizes(W,PETSC_DECIDE,PETSC_DECIDE, N,N);
  MatMPIAIJSetPreallocation(W, prealloc_tot, PETSC_NULL, prealloc_tot, PETSC_NULL);

  int i,j,ik,ns,ne;
  int ix,iy,ic,jx,jy,jc;
  int lx,ux,ly,uy;
  PetscReal rcol;
  PetscScalar val;

  if(rp>1.0){
      
    MatGetOwnershipRange(W, &ns, &ne);
    
    for(i=ns;i<ne;i++){
      //for each row (i), determine the layer index (ic) and the position index within the layer (ix,iy)
      ic=(ik=i)%n;
      ix=(ik/=n)%mx;
      iy=(ik/=mx)%my;

      //for each element i, define a box bounded by 2*rp
      lx = (floor(ix-rp)>0) ? floor(ix-rp) : 0;
      ux = (ceil(ix+rp)<mx) ?  ceil(ix+rp) : mx-1;
      ly = (floor(iy-rp)>0) ? floor(iy-rp) : 0;
      uy = (ceil(iy+rp)<my) ?  ceil(iy+rp) : my-1;
      jc = ic;

      //identify the neighbors within the circle of radius rp
      for(jy=ly;jy<=uy;jy++){
	for(jx=lx;jx<=ux;jx++){
	  rcol=sqrt(pow(ix-jx,2)+pow(iy-jy,2));
	  if(rcol<rp){
	    val = (rp-rcol)/rp + PETSC_i*0.0;
	    j = jc + n*jx + n*mx*jy;
	    MatSetValues(W,1,&i,1,&j,&val,INSERT_VALUES);
	  }
	}
      }

    }
  }

  MatAssemblyBegin(W, MAT_FINAL_ASSEMBLY);
  MatAssemblyEnd(W, MAT_FINAL_ASSEMBLY);

  if(rp<=1.0) MatShift(W,1.0);
  
  //Normalize W
  Vec u,v;
  MatCreateVecs(W,&u,&v);
  VecSet(u,1.0);
  MatMult(W,u,v);
  VecReciprocal(v);
  MatDiagonalScale(W,v,NULL);
  VecDestroy(&u);
  VecDestroy(&v);

  *Wout = W;

}

#undef __FUNCT__
#define __FUNCT__ "threshold_projection_filter"
void threshold_projection_filter(Vec rho_in, Vec rho_out, Vec rho_grad, double filter_threshold, double filter_steepness)
{

  PetscScalar eta=filter_threshold + 1e-10 + PETSC_i*0;
  PetscScalar beta=filter_steepness + PETSC_i*0;

  VecSet(rho_out,0.0);
  VecSet(rho_grad,0.0);

  PetscScalar *rin,*rout,*rg;
  VecGetArray(rho_in,&rin);
  VecGetArray(rho_out,&rout);
  VecGetArray(rho_grad,&rg);
  
  int i,ns,ne;
  VecGetOwnershipRange(rho_in, &ns, &ne);

  for(i=ns;i<ne;i++){
    if(creal(rin[i-ns])>=0 && creal(rin[i-ns])<=creal(eta)){
      rout[i-ns]=eta * ( cexp(-beta*(1-rin[i-ns]/eta)) - (1-rin[i-ns]/eta)*cexp(-beta) );
      rg[i-ns]  =eta * ( (beta/eta)*cexp(-beta*(1-rin[i-ns]/eta)) + cexp(-beta)/eta );
    }else if(creal(rin[i-ns])>creal(eta) && creal(rin[i-ns])<=1.0){
      rout[i-ns]=(1-eta) * ( 1 - cexp(-beta*(rin[i-ns]-eta)/(1-eta)) + (rin[i-ns]-eta)*cexp(-beta)/(1-eta) ) + eta;
      rg[i-ns]  =(1-eta) * ( beta/(1-eta) * cexp(-beta*(rin[i-ns]-eta)/(1-eta)) + cexp(-beta)/(1-eta) );
    }else if(creal(rin[i-ns])<0){
      rout[i-ns]=0+PETSC_i*0;
      rg[i-ns]  =0+PETSC_i*0;
    }else{
      rout[i-ns]=1+PETSC_i*0;
      rg[i-ns]  =0+PETSC_i*0;
    }
  }
  
  VecRestoreArray(rho_in,&rin);
  VecRestoreArray(rho_out,&rout);
  VecRestoreArray(rho_grad,&rg);

}

#undef __FUNCT__
#define __FUNCT__ "array2mpi"
PetscErrorCode array2mpi(PetscScalar *pt, Vec v)
{
PetscErrorCode ierr;
int j, ns, ne;

  ierr = VecGetOwnershipRange(v,&ns,&ne);
  for(j=ns;j<ne;j++)
    { ierr=VecSetValue(v,j,pt[j],INSERT_VALUES);
      CHKERRQ(ierr);
    }

  ierr = VecAssemblyBegin(v); CHKERRQ(ierr);
  ierr = VecAssemblyEnd(v);  CHKERRQ(ierr);

  PetscFunctionReturn(0);
}


#undef __FUNCT__
#define __FUNCT__ "mpi2array"
PetscErrorCode mpi2array(Vec v, PetscScalar *pt, int n)
{
  PetscErrorCode ierr;
  int i;
  PetscScalar *_a;
  Vec V_SEQ;
  VecScatter ctx;

  ierr = VecScatterCreateToAll(v,&ctx,&V_SEQ);CHKERRQ(ierr);
  ierr = VecScatterBegin(ctx,v,V_SEQ,INSERT_VALUES,SCATTER_FORWARD);CHKERRQ(ierr);
  ierr = VecScatterEnd(ctx,v,V_SEQ,INSERT_VALUES,SCATTER_FORWARD);CHKERRQ(ierr);
  ierr = VecGetArray(V_SEQ,&_a);CHKERRQ(ierr);
  for (i = 0; i < n; i++) pt[i] = _a[i];
  ierr = VecRestoreArray(V_SEQ,&_a);CHKERRQ(ierr);
  ierr = VecScatterDestroy(&ctx);CHKERRQ(ierr);
  ierr = VecDestroy(&V_SEQ);CHKERRQ(ierr);

  PetscFunctionReturn(0);
}

/*=================================================================*/
/*=================================================================*/
/*=================================================================*/
/*=================================================================*/
/*=================================================================*/
/*=================================================================*/


#undef __FUNCT__
#define __FUNCT__ "smear_projection"
void smear_projection(Vec rho_in, Vec rho_out, FiltersToolBox *flt)
{

  Vec tmp;
  VecDuplicate(rho_in,&tmp);
  MatMult(flt->W,rho_in,tmp);
  threshold_projection_filter(tmp,rho_out,flt->rho_grad,flt->filter_threshold,flt->filter_steepness);
  VecDestroy(&tmp);

} 

#undef __FUNCT__
#define __FUNCT__ "smear_projection_undo"
void smear_projection_undo(Vec rho_in, Vec rho_out, FiltersToolBox *flt)
{
  Vec tmp;
  VecDuplicate(rho_in,&tmp);
  VecPointwiseMult(tmp,rho_in,flt->rho_grad);
  MatMultTranspose(flt->W,tmp,rho_out);
  VecDestroy(&tmp);
}

#undef __FUNCT__
#define __FUNCT__ "filters_initialize"
void filters_initialize(MPI_Comm comm, FiltersToolBox *flt, DOFInfo dofi)
{

  PetscPrintf(comm,"\tInitializing the filter tool box. The default is no filter. Note that the filter radius (in pixels) could be a fraction for 2d density filter.\n");

  int tmp_i;
  double tmp_f;
  getint("-filter_choice",&tmp_i,0);
  flt->filter_choice=tmp_i;
  getreal("-filter_threshold",&tmp_f,0.5);
  flt->filter_threshold=tmp_f;
  getreal("-filter_steepness",&tmp_f,0);
  flt->filter_steepness=tmp_f;
  getint("-filter_dim",&tmp_i,1);
  flt->filter_dim=tmp_i;

  flt->ndof=dofi.Mx*dofi.My*dofi.nlayers;
  Mat W;
  if(flt->filter_dim==1){
    getint("-filter_radius_in_pixels",&tmp_i,1);
    create_density_filter_1d(comm,&W, tmp_i, dofi.Mx, dofi.nlayers);
  }else if(flt->filter_dim==2){
    getreal("-filter_radius_in_pixels",&tmp_f,1);
    create_density_filter_2d(comm,&W, tmp_f, dofi.Mx, dofi.My, dofi.nlayers);
  }
  flt->W=W;

  Vec rho_grad;
  MatCreateVecs(flt->W,&rho_grad,PETSC_NULL);
  flt->rho_grad=rho_grad;

}

#undef __FUNCT__
#define __FUNCT__ "filters_apply"
void filters_apply(MPI_Comm comm, PetscScalar *u_in, PetscScalar *u_out, FiltersToolBox *flt, int filter_direction)
{

  FilterType filter[3];

  if(filter_direction>=0){
    filter[0]=smear_projection;
    filter[1]=smear_projection; //morph_open/close etc
    filter[2]=smear_projection; //morph_open/close etc
  }else{
    filter[0]=smear_projection_undo;
    filter[1]=smear_projection_undo; //morph_open/close_undo etc
    filter[2]=smear_projection_undo; //morph_open/close_undo etc
  }

  Vec rho_in,rho_out;
  VecDuplicate(flt->rho_grad,&rho_in);
  VecDuplicate(flt->rho_grad,&rho_out);

  array2mpi(u_in,rho_in);
  filter[flt->filter_choice](rho_in,rho_out,flt);
  mpi2array(rho_out,u_out,flt->ndof);

  MPI_Barrier(comm);
  VecDestroy(&rho_in);
  VecDestroy(&rho_out);

}

