all: farfieldopt_exec farfieldoptsym_exec lasingopt_exec

export PETSC_DIR=${HOME}/petsc-3.6.4
export PETSC_ARCH=arch-mumps-opt
export FDOPT_PATH=../library
export NLOPT_PATH=/usr/local/lib

include ${PETSC_DIR}/lib/petsc/conf/variables
include ${PETSC_DIR}/lib/petsc/conf/rules
CLEANFILES = farfieldopt.o farfieldoptsym.o lasingopt.o

optimtemp: optimtemp.o
	$(CLINKER) -Wall -g -o $@ optimtemp.o

farfieldopt: farfieldopt.o
	$(CLINKER) -Wall -g -o $@ farfieldopt.o
farfieldopt_exec: farfieldopt.o optimtemp.o
	$(CLINKER) farfieldopt.o optimtemp.o -o farfieldopt_exec ${NLOPT_PATH}/libnlopt.a ${FDOPT_PATH}/libFDOPT.a $(PETSC_LIB)

farfieldoptsym: farfieldoptsym.o
	$(CLINKER) -Wall -g -o $@ farfieldoptsym.o
farfieldoptsym_exec: farfieldoptsym.o optimtemp.o
	$(CLINKER) farfieldoptsym.o optimtemp.o -o farfieldoptsym_exec ${NLOPT_PATH}/libnlopt.a ${FDOPT_PATH}/libFDOPT.a $(PETSC_LIB)

lasingopt: lasingopt.o
	$(CLINKER) -Wall -g -o $@ lasingopt.o
lasingopt_exec: lasingopt.o optimtemp.o
	$(CLINKER) lasingopt.o optimtemp.o -o lasingopt_exec ${NLOPT_PATH}/libnlopt.a ${FDOPT_PATH}/libFDOPT.a $(PETSC_LIB)

