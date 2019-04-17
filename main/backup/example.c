#include "petsc.h"
#include "petscsys.h"

#undef __FUNCT__
#define __FUNCT__ "main"
PetscErrorCode main(int argc, char **argv)
{

  MPI_Init(NULL, NULL);
  PetscInitialize(&argc,&argv,NULL,NULL);
  int size;
  MPI_Comm_size(MPI_COMM_WORLD, &size);
  PetscPrintf(PETSC_COMM_WORLD,"\tThe total number of processors is %d\n",size);
  
  //check if the number of processors is divisible by the number of subcomms 
  int ncomms, np_per_comm;
  PetscOptionsGetInt(NULL,"-ncomms",&ncomms,NULL);
  if(!(size%ncomms==0)) SETERRQ(PETSC_COMM_WORLD,1,"The number of processes must be a multiple of ncomms so that it is divisible by the number of subcomms.");
  np_per_comm=size/ncomms;
  PetscPrintf(PETSC_COMM_WORLD,"\tThe number of subcomms is %d.\n\tEach subcomm has %d processors.\n",ncomms,np_per_comm);
    
  //calculate the colour of each subcomm ( = rank of each processor / number of processors in each subcomm )
  //note once calculated, the colour is fixed throughout the entire run
  int rank;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  MPI_Comm subcomm;
  int colour = rank/np_per_comm;
  MPI_Comm_split(MPI_COMM_WORLD, colour, rank, &subcomm);
    
  Vec u;
  PetscScalar *_u;
  int i,ns,ne;
  PetscScalar tmp;

  VecCreateMPI(subcomm,PETSC_DECIDE,10,&u);
  VecSet(u,1.0+PETSC_i*0);

  VecGetArray(u,&_u);
  VecGetOwnershipRange(u,&ns,&ne);
  for(i=ns;i<ne;i++){
    VecGetValues(u,1,&i,&tmp);
    PetscPrintf(PETSC_COMM_SELF,"colour %d, u[%d]_array = %g + i * (%g), u[%d]_vec = %g + i * %g \n",colour,i,creal(_u[i]),cimag(_u[i]),creal(tmp),cimag(tmp));
  }
  VecRestoreArray(u,&_u);

  PetscFinalize();
  return 0;

}

