all: logging.o  initialize.o  input.o  output.o  dof2domdirect.o  pml.o  vec.o  mat.o  solver.o  filters.o  farfield.o  ffopt.o  sym.o  ffoptpar.o  lasing.o  lasing2.o

export PETSC_DIR=${HOME}/petsc-3.6.4
export PETSC_ARCH=arch-mumps-opt

include ${PETSC_DIR}/lib/petsc/conf/variables
include ${PETSC_DIR}/lib/petsc/conf/rules
CLEANFILES = logging.o  initialize.o  input.o  output.o  dof2domdirect.o  pml.o  vec.o  mat.o  solver.o  filters.o  farfield.o  ffopt.o  sym.o  ffoptpar.o  lasing.o  lasing2.o

logging: logging.o
	$(CLINKER) -Wall -g -o $@ logging.o
initialize: initialize.o
	$(CLINKER) -Wall -g -o $@ initialize.o
input: input.o
	$(CLINKER) -Wall -g -o $@ input.o
output: output.o
	$(CLINKER) -Wall -g -o $@ output.o
dof2domdirect: dof2domdirect.o
	$(CLINKER) -Wall -g -o $@ dof2domdirect.o
pml: pml.o
	$(CLINKER) -Wall -g -o $@ pml.o
vec: vec.o
	$(CLINKER) -Wall -g -o $@ vec.o
mat: mat.o
	$(CLINKER) -Wall -g -o $@ mat.o
solver: solver.o
	$(CLINKER) -Wall -g -o $@ solver.o
filters: filters.o
	$(CLINKER) -Wall -g -o $@ filters.o
farfield: farfield.o
	$(CLINKER) -Wall -g -o $@ farfield.o
ffopt: ffopt.o
	$(CLINKER) -Wall -g -o $@ ffopt.o
sym: sym.o
	$(CLINKER) -Wall -g -o $@ sym.o
ffoptpar: ffoptpar.o
	$(CLINKER) -Wall -g -o $@ ffoptpar.o
lasing: lasing.o
	$(CLINKER) -Wall -g -o $@ lasing.o
lasing2: lasing2.o
	$(CLINKER) -Wall -g -o $@ lasing2.o







