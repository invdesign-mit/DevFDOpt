#include "output.h"

#undef __FUNCT__
#define __FUNCT__ "saveVecHDF5"
PetscErrorCode saveVecHDF5(MPI_Comm comm, Vec vec, const char *filename, const char *dsetname)
{
  PetscObjectSetName((PetscObject) vec, dsetname);
  PetscErrorCode ierr;
  PetscViewer viewer;
  ierr=PetscViewerHDF5Open(comm,filename,FILE_MODE_WRITE,&viewer); CHKERRQ(ierr);
  ierr=PetscViewerHDF5PushGroup(viewer, "/"); CHKERRQ(ierr);
  ierr=VecView(vec,viewer); CHKERRQ(ierr);
  ierr=PetscViewerDestroy(&viewer); CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "saveVecMfile"
PetscErrorCode saveVecMfile(MPI_Comm comm, Vec vec, const char *filename, const char *dsetname)
{
  PetscObjectSetName((PetscObject) vec, dsetname);
  PetscErrorCode ierr;
  PetscViewer viewer;
  ierr = PetscViewerASCIIOpen(comm, filename, &viewer); CHKERRQ(ierr);
  ierr = PetscViewerSetFormat(viewer, PETSC_VIEWER_ASCII_MATLAB); CHKERRQ(ierr);
  ierr = VecView(vec,viewer); CHKERRQ(ierr);
  ierr = PetscViewerDestroy(&viewer); CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "writetofile"
void writetofile(MPI_Comm comm, char *name, PetscScalar *data, PetscInt n)
{

  int rank;
  MPI_Comm_rank(comm, &rank);

  if(rank==0){
    FILE *ptf;
    int i;
    ptf = fopen(name,"w");
    for (i=0;i<n;i++)
      {
	fprintf(ptf,"%.16e + %.16ej \n",creal(data[i]),cimag(data[i]));
      }
    fclose(ptf);
  }

}

#undef __FUNCT__
#define __FUNCT__ "writetofiledouble"
void writetofiledouble(MPI_Comm comm, char *name, double *data, PetscInt n)
{

  int rank;
  MPI_Comm_rank(comm, &rank);

  if(rank==0){
    FILE *ptf;
    int i;
    ptf = fopen(name,"w");
    for (i=0;i<n;i++)
      {
	fprintf(ptf,"%.16e \n",data[i]);
      }
    fclose(ptf);
  }

}

