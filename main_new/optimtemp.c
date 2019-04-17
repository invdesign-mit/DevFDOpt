#include "petsc.h"
#include "nlopt.h"
#include "libFDOPT.h"

double optimize_generic(int DegFree, double *epsopt, double *lb, double *ub, void *objdata, void **constrdata, optfunc obj, optfunc *constraint, int min_or_max, int nconstraints)
{

  PetscPrintf(PETSC_COMM_WORLD, "\t***How to pass constraints: if there is only one constraint (func,data),\n\
\t   void *constrdata[]={&data};\n\
\t   optfunc constraint[]={func};\n\
\t   int nconstraints=1\n\
\tOtherwise, e.g. nconstraints>1, you can also do:\n\
\t   void *constrdata[nconstraints];\n\
\t   optfunc constraint[nconstraints];\n\
\t   constrdata[0]=&data0; ...\n\
\t   constraint[0]=func0; ...\n\
\tIf no constraints, pass NULL.\n\
\tNote the key concept here is that *constrdata[] is \"an array of addresses\" to data.\n");

  int alg, localalg, maxeval, maxtime;
  getint("-alg",&alg,41);
  getint("-localalg",&localalg,40);
  getint("-maxeval",&maxeval,1000);
  getint("-maxtime",&maxtime,100000);

  nlopt_opt opt;
  nlopt_opt local_opt;
  int i;

  double maxf;
  opt = nlopt_create(alg, DegFree);
  nlopt_set_lower_bounds(opt,lb);
  nlopt_set_upper_bounds(opt,ub);
  nlopt_set_maxeval(opt,maxeval);
  nlopt_set_maxtime(opt,maxtime);
  if(alg==11) nlopt_set_vector_storage(opt,4000);
  if(localalg){
    local_opt=nlopt_create(localalg, DegFree);
    nlopt_set_ftol_rel(local_opt, 1e-11);
    nlopt_set_maxeval(local_opt,10000);
    nlopt_set_local_optimizer(opt,local_opt);
  }

  if(nconstraints){
    for(i=0;i<nconstraints;i++){
      nlopt_add_inequality_constraint(opt,constraint[i],constrdata[i],1e-8);
    }
  }

  double result=0;
  if(obj){
    if(min_or_max==0)
      nlopt_set_min_objective(opt,obj,objdata);
    else
      nlopt_set_max_objective(opt,obj,objdata);
    result=nlopt_optimize(opt,epsopt,&maxf);
  }

  nlopt_destroy(opt);
  if(localalg) nlopt_destroy(local_opt);

  return result;

}

void make_array(double *x, double val, int n)
{
  int i;
  for(i=0;i<n;i++){
    x[i]=val;
  }

}
