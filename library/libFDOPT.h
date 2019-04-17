#include "type.h"
#include "logging.h"
#include "initialize.h"
#include "input.h"
#include "output.h"
#include "dof2domdirect.h"
#include "pml.h"
#include "vec.h"
#include "mat.h"
#include "solver.h"
#include "filters.h"
#include "farfield.h"
#include "ffopt.h"
#include "ffoptpar.h"
#include "lasing.h"

typedef double (*optfunc)(int DegFree,double *epsopt, double *grad, void *data);
