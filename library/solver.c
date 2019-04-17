#include "solver.h"

#undef __FUNCT__
#define __FUNCT__ "vecDot"
PetscErrorCode vecDot(const Vec x1, const Vec x2, const Vec y1, const Vec y2, PetscScalar *val)
{
  PetscFunctionBegin;
  PetscErrorCode ierr;

  PetscScalar val1, val2;
  ierr = VecDot(x1, y1, &val1); CHKERRQ(ierr);
  ierr = VecDot(x2, y2, &val2); CHKERRQ(ierr);
  *val = val1 + val2;

  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "vecTDot"
PetscErrorCode vecTDot(const Vec x1, const Vec x2, const Vec y1, const Vec y2, PetscScalar *val)
{
  PetscFunctionBegin;
  PetscErrorCode ierr;

  PetscScalar val1, val2;
  ierr = VecTDot(x1, y1, &val1); CHKERRQ(ierr);
  ierr = VecTDot(x2, y2, &val2); CHKERRQ(ierr);
  *val = val1 + val2;

  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "vec2Norm"
PetscErrorCode vec2Norm(const Vec x1, const Vec x2, PetscReal *val)
{
  PetscFunctionBegin;
  PetscErrorCode ierr;

  PetscReal val1, val2;
  ierr = VecNorm(x1, NORM_2, &val1); CHKERRQ(ierr);
  ierr = VecNorm(x2, NORM_2, &val2); CHKERRQ(ierr);
  *val = PetscSqrtScalar(val1*val1 + val2*val2);

  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "vecNormalize"
PetscErrorCode vecNormalize(Vec x1, Vec x2, PetscReal *val)
{
  PetscFunctionBegin;
  PetscErrorCode ierr;

  PetscReal val_temp;
  ierr = vec2Norm(x1, x2, &val_temp); CHKERRQ(ierr);
  ierr = VecScale(x1, 1.0/val_temp); CHKERRQ(ierr);
  ierr = VecScale(x2, 1.0/val_temp); CHKERRQ(ierr);
  if (val != PETSC_NULL) *val = val_temp;

  PetscFunctionReturn(0);
}



#undef __FUNCT__
#define __FUNCT__ "bicg"
PetscErrorCode bicg(MPI_Comm comm, const Mat A, Vec x, const Vec b, const PetscInt max_iter, const PetscReal tol, const PetscInt relres_interval)
{
	PetscFunctionBegin;
	PetscErrorCode ierr;

	PetscPrintf(comm,"\tUsing BiCG algorithm for Asymmetric matrices\n");

	Vec r;  // residual for x
	ierr = VecDuplicate(x, &r); CHKERRQ(ierr);
	ierr = MatMult(A, x, r); CHKERRQ(ierr);
	ierr = VecAYPX(r, -1.0, b); CHKERRQ(ierr);  // r = b - A*x

	Vec s;  // residual for y
	ierr = VecDuplicate(x, &s); CHKERRQ(ierr);
	ierr = VecCopy(r, s); CHKERRQ(ierr);
	//ierr = MatMult(A, r, s); CHKERRQ(ierr);  // Fletcher's choice
	ierr = VecConjugate(s); CHKERRQ(ierr);  // this makes s^T * r = conj(r)^T * r = <r, r>

	Vec p;
	ierr = VecDuplicate(x, &p); CHKERRQ(ierr);
	ierr = VecCopy(r, p); CHKERRQ(ierr);  // p = r

	Vec q;
	ierr = VecDuplicate(x, &q); CHKERRQ(ierr);
	ierr = VecCopy(s, q); CHKERRQ(ierr);  // q = s

	PetscReal norm_r, norm_b;
	ierr = VecNorm(r, NORM_INFINITY, &norm_r); CHKERRQ(ierr);
	ierr = VecNorm(b, NORM_INFINITY, &norm_b); CHKERRQ(ierr);

	PetscReal rel_res = norm_r / norm_b;  // relative residual

	PetscScalar sr;  // s^T * r
	ierr = VecTDot(s, r, &sr); CHKERRQ(ierr);

	Vec Ap;  // A*p
	ierr = VecDuplicate(x, &Ap); CHKERRQ(ierr);

	Vec Aq;  // A^T * q
	ierr = VecDuplicate(x, &Aq); CHKERRQ(ierr);

	PetscScalar qAp;  // q^T * Ap
	PetscScalar alpha;  // sr/qAp
	PetscScalar gamma;  // sr_curr / sr_prev

	PetscInt num_iter;
	for (num_iter = 0; (max_iter <= 0 || num_iter < max_iter) && rel_res > tol; ++num_iter) {
		if (num_iter % relres_interval == 0) {
		  PetscPrintf(comm,"\tnumiter: %d\trelres: %e\n",num_iter,rel_res);
		}
		ierr = MatMult(A, p, Ap); CHKERRQ(ierr);  // Ap = A*p
		ierr = MatMultTranspose(A, q, Aq); CHKERRQ(ierr);  // Aq = A^T * q
		ierr = VecTDot(q, Ap, &qAp); CHKERRQ(ierr);  // qAp = q^T * Ap

		alpha = sr / qAp;

		ierr = VecAXPY(x, alpha, p); CHKERRQ(ierr);  // x = x + alpha * p

		ierr = VecAXPY(r, -alpha, Ap); CHKERRQ(ierr);  // r = r - alpha * Ap
		ierr = VecAXPY(s, -alpha, Aq); CHKERRQ(ierr);  // s = s - alpha * Aq

		gamma = sr;
		ierr = VecTDot(s, r, &sr); CHKERRQ(ierr);  // sr = s^T * r
		gamma = sr / gamma;  // gamma = sr_curr / sr_prev

		ierr = VecAYPX(p, gamma, r); CHKERRQ(ierr);  // p = r + gamma * p
		ierr = VecAYPX(q, gamma, s); CHKERRQ(ierr);  // q = s + gamma * q

		ierr = VecNorm(r, NORM_INFINITY, &norm_r); CHKERRQ(ierr);
		rel_res = norm_r / norm_b;

	}

	ierr = VecDestroy(&r); CHKERRQ(ierr);
	ierr = VecDestroy(&s); CHKERRQ(ierr);
	ierr = VecDestroy(&p); CHKERRQ(ierr);
	ierr = VecDestroy(&q); CHKERRQ(ierr);
	ierr = VecDestroy(&Ap); CHKERRQ(ierr);
	ierr = VecDestroy(&Aq); CHKERRQ(ierr);

	PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "bicgSymmetric"
PetscErrorCode bicgSymmetric(MPI_Comm comm, const Mat A, Vec x, const Vec b, const PetscInt max_iter, const PetscReal tol, const PetscInt relres_interval)
{
  PetscFunctionBegin;
  PetscErrorCode ierr;

  PetscPrintf(comm,"\tUsing BiCG algorithm for Symmetric matrices\n");

  Vec r;  // residual for x
  ierr = VecDuplicate(x, &r); CHKERRQ(ierr);
  ierr = MatMult(A, x, r); CHKERRQ(ierr);
  ierr = VecAYPX(r, -1.0, b); CHKERRQ(ierr);  // r = b - A*x

  Vec p;
  ierr = VecDuplicate(x, &p); CHKERRQ(ierr);
  ierr = VecCopy(r, p); CHKERRQ(ierr);  // p = r

  PetscScalar rr;
  ierr = VecTDot(r, r, &rr); CHKERRQ(ierr);  // rr = r^T * r

  PetscReal norm_r, norm_b;
  ierr = VecNorm(r, NORM_INFINITY, &norm_r); CHKERRQ(ierr);
  ierr = VecNorm(b, NORM_INFINITY, &norm_b); CHKERRQ(ierr);

  PetscReal rel_res = norm_r / norm_b;  // relative residual

  Vec Ap;  // A*p
  ierr = VecDuplicate(x, &Ap); CHKERRQ(ierr);

  PetscScalar pAp;  // p^T * Ap
  PetscScalar alpha;  // sr/qAp
  PetscScalar gamma;  // rr_curr / rr_prev

  PetscInt num_iter;

  for (num_iter = 0; (max_iter <= 0 || num_iter < max_iter) && rel_res > tol; ++num_iter) {
    if (num_iter % relres_interval == 0) {
      PetscPrintf(comm,"\tnumiter: %d\trelres: %e\n",num_iter,rel_res);
    }
    ierr = MatMult(A, p, Ap); CHKERRQ(ierr);  // Ap = A*p

    ierr = VecTDot(p, Ap, &pAp); CHKERRQ(ierr);  // pAp = p^T * Ap
    alpha = rr / pAp;

    ierr = VecAXPY(x, alpha, p); CHKERRQ(ierr);  // x = x + alpha * p
    ierr = VecAXPY(r, -alpha, Ap); CHKERRQ(ierr);  // r = r - alpha * Ap

    gamma = rr;
    ierr = VecTDot(r, r, &rr); CHKERRQ(ierr);  // rr = r^T * r
    gamma = rr / gamma;  // gamma = rr_curr / rr_prev

    ierr = VecAYPX(p, gamma, r); CHKERRQ(ierr);  // p = r + gamma * p

    ierr = VecNorm(r, NORM_INFINITY, &norm_r); CHKERRQ(ierr);
    rel_res = norm_r / norm_b;
  }

  ierr = VecDestroy(&r); CHKERRQ(ierr);
  ierr = VecDestroy(&p); CHKERRQ(ierr);
  ierr = VecDestroy(&Ap); CHKERRQ(ierr);

  PetscFunctionReturn(0);
}


#undef __FUNCT__
#define __FUNCT__ "solveEq"
PetscErrorCode solveEq(MPI_Comm comm, Mat A, Vec x, Vec b, Vec LPC, Vec RPC, SolverInfo *si)
{

  IterativeSolver solver;
  TimeStamp ts_s;

  if(si->use_mat_sym)
    solver=bicgSymmetric;
  else
    solver=bicg;

  /*
  if(LPC){
    MatDiagonalScale(A,LPC,PETSC_NULL);
    VecPointwiseMult(b,LPC,b);
  }
  if(RPC){
    MatDiagonalScale(A,PETSC_NULL,RPC);
  }
  */

  initTimeStamp(&ts_s);
  solver(comm,A,x,b,si->max_iter,si->tol,si->relres_interval);
  updateTimeStamp(comm,&ts_s,"solving Ax=b iteratively");
  
  /*
  if(LPC){
    Vec inv;
    VecDuplicate(LPC,&inv);
    VecSet(inv,1.0);
    VecPointwiseDivide(inv,inv,LPC);
    MatDiagonalScale(A,inv,PETSC_NULL);
    VecPointwiseMult(b,inv,b);
    VecDestroy(&inv);
  }
  if(RPC){
    Vec inv;
    VecDuplicate(RPC,&inv);
    VecSet(inv,1.0);
    VecPointwiseDivide(inv,inv,RPC);
    MatDiagonalScale(A,PETSC_NULL,inv);
    VecPointwiseMult(x,RPC,x);
    VecDestroy(&inv);
  }
  */

  PetscFunctionReturn(0);

}

#undef __FUNCT__
#define __FUNCT__ "setupKSPDirect"
PetscErrorCode setupKSPDirect(MPI_Comm comm, KSP *kspout, PC *pcout, int maxit)
{
  PetscErrorCode ierr;
  KSP ksp;
  PC pc;

  ierr = KSPCreate(comm,&ksp);CHKERRQ(ierr);

  ierr = KSPSetType(ksp, KSPGMRES);CHKERRQ(ierr);
  //ierr = KSPSetType(ksp, KSPBCGS);CHKERRQ(ierr);

  ierr = KSPGetPC(ksp,&pc);CHKERRQ(ierr);
  ierr = PCSetType(pc,PCLU);CHKERRQ(ierr);
  ierr = PCFactorSetMatSolverPackage(pc,MATSOLVERMUMPS);CHKERRQ(ierr);

  ierr = KSPSetTolerances(ksp,1e-14,PETSC_DEFAULT,PETSC_DEFAULT,maxit);CHKERRQ(ierr);
  //ierr = KSPSetTolerances(ksp,1e-20,1e-20,PETSC_DEFAULT,maxit);CHKERRQ(ierr);

  ierr = PCSetFromOptions(pc);
  ierr = KSPSetFromOptions(ksp);CHKERRQ(ierr);

  *kspout=ksp;
  *pcout=pc;

  PetscFunctionReturn(0);

}


#undef _FUNCT_
#define _FUNCT_ "SolveMatrixDirect"
PetscErrorCode SolveMatrixDirect(MPI_Comm comm, KSP ksp, Mat M, Vec b, Vec x, int *its, int maxit)
{
  /*-----------------KSP Solving------------------*/
  PetscErrorCode ierr;
  PetscLogDouble t1,t2,tpast;
  ierr = PetscTime(&t1);CHKERRQ(ierr);

  if (*its>(maxit-5)){
    ierr = KSPSetOperators(ksp,M,M);CHKERRQ(ierr);}
  else{
    ierr = KSPSetOperators(ksp,M,M);CHKERRQ(ierr);
    ierr = KSPSetReusePreconditioner(ksp,PETSC_TRUE);CHKERRQ(ierr);}

  ierr = PetscPrintf(comm,"==> initial-its is %d. maxit is %d.----\n ",*its,maxit);CHKERRQ(ierr);
  ierr = KSPSolve(ksp,b,x);CHKERRQ(ierr);
  ierr = KSPGetIterationNumber(ksp,its);CHKERRQ(ierr);

  // if GMRES is stopped due to maxit, then redo it with sparse direct solve;
  if(*its>(maxit-2))
    {
      ierr = PetscPrintf(comm,"==> after-one-solve-its is %d. maxit is %d.----\n ",*its,maxit);CHKERRQ(ierr);
      ierr = PetscPrintf(comm,"==> Too Many Iterations. Re-solving with Sparse Direct Solver.\n");CHKERRQ(ierr);
      ierr = KSPSetOperators(ksp,M,M);CHKERRQ(ierr);
      ierr = KSPSetReusePreconditioner(ksp,PETSC_FALSE);CHKERRQ(ierr);
      ierr = KSPSolve(ksp,b,x);CHKERRQ(ierr);
      ierr = KSPGetIterationNumber(ksp,its);CHKERRQ(ierr);
    }

  //Print kspsolving information
  double norm;
  Vec xdiff;
  ierr=VecDuplicate(x,&xdiff);CHKERRQ(ierr);
  ierr = MatMult(M,x,xdiff);CHKERRQ(ierr);
  ierr = VecAXPY(xdiff,-1.0,b);CHKERRQ(ierr);
  ierr = VecNorm(xdiff,NORM_INFINITY,&norm);CHKERRQ(ierr);
  ierr = PetscPrintf(comm,"==> Matrix solution: norm of error %g, Kryolv Iterations %d----\n ",norm,*its);CHKERRQ(ierr);

  ierr = PetscTime(&t2);CHKERRQ(ierr);
  tpast = t2 - t1;

  PetscPrintf(comm,"==> Matrix solution: the runing time is %f s \n",tpast);
  /*--------------Finish KSP Solving---------------*/

  VecDestroy(&xdiff);
  PetscFunctionReturn(0);
}
