#include "lasing.h"

extern int count;
extern TimeStamp global_time;

#undef __FUNCT__
#define __FUNCT__ "lasing2"
double lasing2(int ndofAll, double *dofAll, double *dofgradAll, void *data)
{

  Lasingdata *ptdata = (Lasingdata *) data;

  Mat M0 = ptdata->M0;
  Vec x = ptdata->x;
  Vec epsDiff = ptdata->epsDiff;
  Vec epsBkg = ptdata->epsBkg;
  PetscScalar omega = ptdata->omega;
  KSP ksp = ptdata->ksp;
  int *its = ptdata->its;
  int maxit = ptdata->maxit;
  GridInfo *gi = ptdata->gi;
  DOFInfo *dofi = ptdata->dofi;
  ParDataGrid dg = ptdata->dg;
  int printdof = ptdata->printdof;

  PetscReal jmax = ptdata->jmax;
  PetscReal jmin = ptdata->jmin;
  
  char tmpstr[PETSC_MAX_PATH_LEN];
  sprintf(tmpstr,"starting computation step %d (global timecheck)",count);
  updateTimeStamp(PETSC_COMM_WORLD,&global_time,tmpstr);

  int i;
  int ndof=dofi->ndof; //only the epsilon degrees of freedom
  PetscScalar *dof,*dofgrad; //complex values of epsilon degrees of freedom just for filtering purposes
  FiltersToolBox *flt = ptdata->flt;
  PetscScalar *_u,*_ugrad;
  dof = (PetscScalar *) malloc(ndof*sizeof(PetscScalar));
  dofgrad = (PetscScalar *) malloc(ndof*sizeof(PetscScalar));
  _u = (PetscScalar *) malloc(ndof*sizeof(PetscScalar));
  _ugrad = (PetscScalar *) malloc(ndof*sizeof(PetscScalar));
  for(i=0;i<ndof;i++){
    dof[i]=dofAll[i]+PETSC_i*0.0;
  }
  filters_apply(PETSC_COMM_WORLD,dof,_u,flt,1);

  int ndof_j=ndofAll-ndof; //the degrees of freedom pertaining to the source distribution
  PetscScalar *j = (PetscScalar *) malloc(ndof_j/2*sizeof(PetscScalar));
  PetscScalar *jgrad = (PetscScalar *) malloc(ndof_j/2*sizeof(PetscScalar));
  for(i=0;i<ndof_j/2;i++)
    j[i]= (jmin + (jmax-jmin) * dofAll[ndof+i]) * cexp(PETSC_i * 2.0 * M_PI * dofAll[ndof+i+ndof_j/2]);

  Vec J,b,Jconj,epsbar,epsbarJstar,eps,negW2eps,xconj,u,gradeps,gradJ;
  Mat M;
  VecDuplicate(x,&J);
  VecDuplicate(x,&b);
  VecDuplicate(x,&Jconj);
  VecDuplicate(x,&epsbar);
  VecDuplicate(x,&epsbarJstar);
  VecDuplicate(x,&eps);
  VecDuplicate(x,&negW2eps);
  VecDuplicate(x,&xconj);
  VecDuplicate(x,&u);
  VecDuplicate(x,&gradeps);
  VecDuplicate(x,&gradJ);
  MatDuplicate(M0,MAT_COPY_VALUES,&M);

  multilayer_forward(_u,epsbar,dofi,dg.da);
  
  array2vec(j,J,gi,dg.da);
  VecPointwiseMult(b,epsbar,J);
  VecScale(b,-PETSC_i*omega);
  VecCopy(J,Jconj);
  VecConjugate(Jconj);
  VecPointwiseMult(epsbarJstar,epsbar,Jconj);
  
  VecCopy(epsbar,eps);
  VecPointwiseMult(eps,eps,epsDiff);
  VecAXPY(eps,1.0,epsBkg);
  VecCopy(eps,negW2eps);
  VecScale(negW2eps,-omega*omega);
  MatDiagonalSet(M,negW2eps,ADD_VALUES);

  SolveMatrixDirect(PETSC_COMM_WORLD,ksp,M,b,x,its,maxit);
  VecCopy(x,xconj);
  VecConjugate(xconj);
  PetscScalar epsbar_Jstar_dot_E, epsbar_Jstar_dot_J;
  VecTDot(epsbarJstar,x,&epsbar_Jstar_dot_E);
  VecTDot(epsbarJstar,J,&epsbar_Jstar_dot_J);
  double objval = -creal(epsbar_Jstar_dot_E)/creal(epsbar_Jstar_dot_J);

  PetscPrintf(PETSC_COMM_WORLD,"---at step %d, the thresholdval is %.16g \n",count,objval);
  
  KSPSolveTranspose(ksp,epsbarJstar,u);

  Vec tmp;
  VecDuplicate(x,&tmp);
  VecPointwiseMult(gradeps,u,x);
  VecScale(gradeps,-omega*omega/creal(epsbar_Jstar_dot_J));
  VecPointwiseMult(gradeps,gradeps,epsDiff);
  VecPointwiseMult(tmp,Jconj,J);
  VecAXPY(gradeps,creal(epsbar_Jstar_dot_E)/pow(creal(epsbar_Jstar_dot_J),2),tmp);
  VecPointwiseMult(tmp,xconj,J);
  VecAXPY(gradeps,-1.0/creal(epsbar_Jstar_dot_J),tmp);
  multilayer_backward(PETSC_COMM_WORLD,gradeps,_ugrad,dofi,dg.da);
  filters_apply(PETSC_COMM_WORLD,_ugrad,dofgrad,flt,-1);

  VecCopy(xconj,gradJ);
  VecPointwiseMult(gradJ,gradJ,epsbar);
  VecPointwiseMult(tmp,epsbar,u);
  VecAXPY(gradJ,-PETSC_i*omega,tmp);
  VecScale(gradJ,-1/(2.0*epsbar_Jstar_dot_J));
  VecAXPY(gradJ,creal(epsbar_Jstar_dot_E)/pow(creal(epsbar_Jstar_dot_J),2),epsbarJstar);
  vec2array(PETSC_COMM_WORLD,gradJ,jgrad,gi,dg.da);

  VecDestroy(&tmp);
  
  for(i=0;i<ndof;i++)
    dofgradAll[i]=creal(dofgrad[i]);

  for(i=0;i<ndof_j/2;i++){
    dofgradAll[ndof+i]=2.0*(jmax-jmin)*creal(jgrad[i]*cexp(PETSC_i*2.0*M_PI*dofAll[ndof+i+ndof_j/2]));
    dofgradAll[ndof+i+ndof_j/2]=2.0*2.0*M_PI*creal(jgrad[i]*PETSC_i*j[i]);
  }

  VecDestroy(&J);
  VecDestroy(&b);
  VecDestroy(&Jconj);
  VecDestroy(&epsbar);
  VecDestroy(&epsbarJstar);
  VecDestroy(&eps);
  VecDestroy(&negW2eps);
  VecDestroy(&xconj);
  VecDestroy(&u);
  VecDestroy(&gradeps);
  VecDestroy(&gradJ);
  MatDestroy(&M);
  
  char output_filename[PETSC_MAX_PATH_LEN];
  if((count%printdof)==0){
    sprintf(output_filename,"outputdof_step%d.txt",count);
    writetofiledouble(PETSC_COMM_WORLD,output_filename,dofAll,ndofAll);
    sprintf(tmpstr,"outputing the dofs at step %d",count);
    updateTimeStamp(PETSC_COMM_WORLD,&global_time,tmpstr);
  }

  sprintf(tmpstr,"end of computation step %d (global timecheck)",count);
  updateTimeStamp(PETSC_COMM_WORLD,&global_time,tmpstr);

  count++;
  free(_u);
  free(_ugrad);
  free(dof);
  free(dofgrad);
  free(j);
  free(jgrad);

  return objval;

}

