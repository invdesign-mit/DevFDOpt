#include "logging.h"

#undef __FUNCT__
#define __FUNCT__ "initTimeStamp"
/**
 * initTimeStamp
 * -------------
 * Initialize the time stamp.
 */
PetscErrorCode initTimeStamp(TimeStamp *ts)
{
	PetscFunctionBegin;
	PetscErrorCode ierr;

	ierr = PetscTime(&ts->start); CHKERRQ(ierr);
	ts->curr = ts->start;

	PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "updateTimeStamp"
/**
 * updateTimeStamp
 * ---------------
 * Update the time stamp, and print it.
 */
PetscErrorCode updateTimeStamp(MPI_Comm comm, TimeStamp *ts, const char *event_description)
{
	PetscFunctionBegin;
	PetscErrorCode ierr;

	ts->prev = ts->curr;
	ierr = PetscTime(&ts->curr); CHKERRQ(ierr);
	ierr = PetscFPrintf(comm, stdout, "time elapsed: %f, time %s: %f\n", ts->curr - ts->start, event_description, ts->curr - ts->prev); CHKERRQ(ierr);

	PetscFunctionReturn(0);
}
