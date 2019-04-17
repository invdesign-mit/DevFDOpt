#include "input.h"

#undef __FUNCT__
#define __FUNCT__ "h5get_data"
/**
 * h5get_data
 * -----------
 * Retrieve an array data stored under a given data set name in an HDF5 file.
 */
PetscErrorCode h5get_data(hid_t inputfile_id, const char *dataset_name, hid_t mem_type_id, void *buf)
{
  PetscFunctionBegin;

  hid_t dataset_id;
  herr_t status;

  dataset_id = H5Dopen(inputfile_id, dataset_name, H5P_DEFAULT);
  status = H5Dread(dataset_id, mem_type_id, H5S_ALL, H5S_ALL, H5P_DEFAULT, buf);
  status = H5Dclose(dataset_id);

  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "ri2c"
/**
 * ri2c
 * -----------
 * Construct a complex array from an array with alternating real and imaginary values as elements.
 */
PetscErrorCode ri2c(const void *pri, void *pc, const int numelem)
{
  PetscFunctionBegin;

  PetscReal *pRI = (PetscReal *) pri;
  PetscScalar *pC = (PetscScalar *) pc;

  int i;
  for (i = 0; i < numelem; ++i) {
    *pC = *pRI++;
    *pC++ += PETSC_i * (*pRI++);
  }

  PetscFunctionReturn(0);
}

//functions to read from stdin which can take in default values 
#undef __FUNCT__
#define __FUNCT__ "getreal"
PetscErrorCode getreal(const char *flag, double *var, double autoval)
{
  PetscErrorCode ierr;
  PetscBool flg;
  ierr=PetscOptionsGetReal(PETSC_NULL,flag,var,&flg); CHKERRQ(ierr);
  if(!flg) *var=autoval;
  ierr=PetscPrintf(PETSC_COMM_WORLD,"--%s is %g \n",flag,*var); CHKERRQ(ierr);

  PetscFunctionReturn(ierr);
}

#undef __FUNCT__
#define __FUNCT__ "getint"
PetscErrorCode getint(const char *flag, int *var, int autoval)
{
  PetscErrorCode ierr;
  PetscBool flg;
  ierr=PetscOptionsGetInt(PETSC_NULL,flag,var,&flg); CHKERRQ(ierr);
  if(!flg) *var=autoval;
  ierr=PetscPrintf(PETSC_COMM_WORLD,"--%s is %d \n",flag,*var); CHKERRQ(ierr);

  PetscFunctionReturn(ierr);
}

#undef __FUNCT__
#define __FUNCT__ "getstr"
PetscErrorCode getstr(const char *flag, char *strin, const char default_strin[])
{
  PetscErrorCode ierr;
  PetscBool flg;
  ierr=PetscOptionsGetString(PETSC_NULL,flag,strin,PETSC_MAX_PATH_LEN-1,&flg); CHKERRQ(ierr);
  if(!flg) strcpy(strin,default_strin);
  ierr=PetscPrintf(PETSC_COMM_WORLD,"--%s is %s \n",flag,strin);

  PetscFunctionReturn(ierr);
}

#undef __FUNCT__
#define __FUNCT__ "getintarray"
PetscErrorCode getintarray(const char *flag, int *z, int *nz, int default_val)
{
  PetscErrorCode ierr;
  PetscBool flg;
  int i,nget=*nz;
  char buffer[PETSC_MAX_PATH_LEN], tmp[PETSC_MAX_PATH_LEN];
  ierr=PetscOptionsGetIntArray(PETSC_NULL,flag,z,&nget,&flg); CHKERRQ(ierr);
  if(nget!=*nz){
    PetscPrintf(PETSC_COMM_WORLD,"!!!!WARNING: %d values expected for %s but only received %d.\n",*nz,flag,nget);
    *nz=nget;
  }
  if(!flg) {
    for(i=0;i<*nz;i++){
      z[i]=default_val;
    }
  }
  strcpy(buffer," ");
  for(i=0;i<*nz;i++){
    sprintf(tmp,"%d, ",z[i]);
    strcat(buffer,tmp);
  }
  PetscPrintf(PETSC_COMM_WORLD,"--%s is %s total: %d  \n",flag,buffer,*nz);

  PetscFunctionReturn(ierr);

}

#undef __FUNCT__
#define __FUNCT__ "getrealarray"
PetscErrorCode getrealarray(const char *flag, PetscReal *r, int *nr, PetscReal default_val)
{
  PetscErrorCode ierr;
  PetscBool flg;
  int i,nget=*nr;
  char buffer[PETSC_MAX_PATH_LEN], tmp[PETSC_MAX_PATH_LEN];
  ierr=PetscOptionsGetRealArray(PETSC_NULL,flag,r,&nget,&flg); CHKERRQ(ierr);
  if(nget!=*nr){
    PetscPrintf(PETSC_COMM_WORLD,"!!!!WARNING: %d values expected for %s but only received %d.\n",*nr,flag,nget);
    *nr=nget;
  }
  if(!flg) {
    for(i=0;i<*nr;i++){
      r[i]=default_val;
    }
  }
  strcpy(buffer," ");
  for(i=0;i<*nr;i++){
    sprintf(tmp,"%g, ",r[i]);
    strcat(buffer,tmp);
  }
  PetscPrintf(PETSC_COMM_WORLD,"--%s is %s total: %d  \n",flag,buffer,*nr);

  PetscFunctionReturn(ierr);

}

#undef __FUNCT__
#define __FUNCT__ "readfromfile"
void readfromfile(char *name, PetscScalar *data, PetscInt n)
{

  FILE *ptf;
  int i;
  PetscReal tmp;
  ptf = fopen(name,"r");
  for (i=0;i<n;i++)
    {
      fscanf(ptf,"%lf",&tmp);
      data[i] = tmp + PETSC_i * 0.0;
    }
  fclose(ptf);

}

#undef __FUNCT__
#define __FUNCT__ "readfromfiledouble"
void readfromfiledouble(char *name, double *data, PetscInt n)
{

  FILE *ptf;
  int i;
  PetscReal tmp;
  ptf = fopen(name,"r");
  for (i=0;i<n;i++)
    {
      fscanf(ptf,"%lf",&tmp);
      data[i] = tmp;
    }
  fclose(ptf);

}

#undef __FUNCT__
#define __FUNCT__ "loadVecHDF5"
PetscErrorCode loadVecHDF5(MPI_Comm comm, Vec vec, const char *inputfile_name, const char *dataset_name)
{
  PetscFunctionBegin;
  PetscErrorCode ierr;

  PetscViewer viewer;
  ierr = PetscViewerHDF5Open(comm, inputfile_name, FILE_MODE_READ, &viewer); CHKERRQ(ierr);

  ierr = PetscViewerHDF5PushGroup(viewer, "/"); CHKERRQ(ierr);  // assume that all datasets are under "/".
  ierr = PetscObjectSetName((PetscObject) vec, ++dataset_name); CHKERRQ(ierr);  // ++ to remove '/'

  ierr = VecLoad(vec, viewer); CHKERRQ(ierr);

  ierr = PetscViewerHDF5PopGroup(viewer); CHKERRQ(ierr);

  ierr = PetscViewerDestroy(&viewer); CHKERRQ(ierr);

  PetscFunctionReturn(0);
}
