all: topopt_exec 

export PETSC_DIR=${HOME}/petsc-3.6.4
export PETSC_ARCH=arch-mumps-opt
export FDOPT_PATH=../library_new
export NLOPT_PATH=/usr/local/lib

include ${PETSC_DIR}/lib/petsc/conf/variables
include ${PETSC_DIR}/lib/petsc/conf/rules
CLEANFILES = topopt.o optimtemp.o 

optimtemp: optimtemp.o
	$(CLINKER) -Wall -g -o $@ optimtemp.o

topopt: topopt.o
	$(CLINKER) -Wall -g -o $@ topopt.o
topopt_exec: topopt.o optimtemp.o
	$(CLINKER) topopt.o optimtemp.o -o topopt_exec ${NLOPT_PATH}/libnlopt.a ${FDOPT_PATH}/libFDOPT.a $(PETSC_LIB)




