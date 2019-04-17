#include "pml.h"

PetscScalar calc_s_factor(PetscScalar omega, PetscScalar depth, PetscScalar Lpml);

#undef __FUNCT__
#define __FUNCT__ "stretch_dl"
void stretch_dl(PetscScalar *dl_stretched[Naxis][Ngt], PetscScalar omega, GridInfo gi)
{
  Axis i;
  int j;
  PetscScalar *s_factor[Naxis][Ngt];
  for(i=Xx;i<Naxis;i++){
    s_factor[i][Prim] = (PetscScalar *) malloc(gi.N[i]*sizeof(PetscScalar));
    s_factor[i][Dual] = (PetscScalar *) malloc(gi.N[i]*sizeof(PetscScalar));
    generate_s_factor(omega, s_factor[i][Prim], s_factor[i][Dual], gi.dl[i][Prim], gi.dl[i][Dual], gi.N[i], gi.Npml[i][Neg], gi.Npml[i][Pos]);
    for(j=0;j<gi.N[i];j++){
      dl_stretched[i][Prim][j]=gi.dl[i][Prim][j]*s_factor[i][Prim][j];
      dl_stretched[i][Dual][j]=gi.dl[i][Dual][j]*s_factor[i][Dual][j];
    }
    free(s_factor[i][Prim]);
    free(s_factor[i][Dual]);
  }

}

#undef __FUNCT__
#define __FUNCT__ "generate_s_factor"
void generate_s_factor(PetscScalar omega, PetscScalar *s_prim, PetscScalar *s_dual, PetscScalar *dl_prim, PetscScalar *dl_dual, int N, int Npml_n, int Npml_p)
{
  int i,j;
  PetscScalar *lprim, *ldual;
  lprim = (PetscScalar *) malloc((N+1)*sizeof(PetscScalar));
  ldual = (PetscScalar *) malloc((N+1)*sizeof(PetscScalar));
  lprim[0]=lprim_0;
  ldual[0]=lprim_0 - dl_prim[0]/2;
  PetscScalar dLprim, dLdual;
  for(i=1;i<N+1;i++){
    dLprim=0;
    dLdual=0;
    for(j=0;j<i;j++){
      dLprim = dLprim + dl_prim[j];
      dLdual = dLdual + dl_dual[j];
    }
    lprim[i] = lprim[0] + dLdual;
    ldual[i] = ldual[0] + dLprim;
  }
  int Npml[2] = {Npml_n, Npml_p};
  PetscScalar lpml[2]; //locations of the PML interfaces at negative [0] and positive [1] ends
  lpml[0] = lprim[Npml[0]];
  lpml[1] = lprim[N-Npml[1]];
  PetscScalar Lpml[2],depth;
  Lpml[0]=lpml[0]-lprim[0];
  Lpml[1]=lprim[N]-lpml[1];

  ldual=ldual+1;
  
  for(i=0;i<N;i++){
    if(creal(lprim[i])<creal(lpml[0])){
      depth=cabs(lprim[i]-lpml[0]);
      s_prim[i]=calc_s_factor(omega,depth,Lpml[0]);
    }else if(creal(lprim[i])>creal(lpml[1])){
      depth=cabs(lprim[i]-lpml[1]);
      s_prim[i]=calc_s_factor(omega,depth,Lpml[1]);
    }else{
      s_prim[i]=1.0;
    }

    if(creal(ldual[i])<creal(lpml[0])){
      depth=cabs(ldual[i]-lpml[0]);
      s_dual[i]=calc_s_factor(omega,depth,Lpml[0]);
    }else if(creal(ldual[i])>creal(lpml[1])){
      depth=cabs(ldual[i]-lpml[1]);
      s_dual[i]=calc_s_factor(omega,depth,Lpml[1]);
    }else{
      s_dual[i]=1.0;
    }
  }

  free(lprim);
  free(ldual-1);
 
}

PetscScalar calc_s_factor(PetscScalar omega, PetscScalar depth, PetscScalar Lpml)
{
  
  PetscScalar lnR = clog(R);

  PetscScalar sigma_max = -(m+1) * lnR/(2*Lpml);
  PetscScalar sigma = sigma_max * cpow(depth/Lpml,m);
  
  PetscScalar kappa = 1 + (kappa_max - 1) * cpow(depth/Lpml,m);

  PetscScalar a = amax * cpow(1 - depth/Lpml,ma);

  return kappa + sigma / (a + PETSC_i * omega);
}

