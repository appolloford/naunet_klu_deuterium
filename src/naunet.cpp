#include <cvode/cvode.h>  // prototypes for CVODE fcts., consts.
/* */
#include <nvector/nvector_serial.h>      // access to serial N_Vector
#include <sunlinsol/sunlinsol_klu.h>     // access to KLU sparse direct solver
#include <sunmatrix/sunmatrix_sparse.h>  // access to sparse SUNMatrix
/* */
/*  */
#include "naunet.h"
/*  */
#include "naunet_ode.h"

// check_flag function is from the cvDiurnals_ky.c example from the CVODE
// package. Check function return value...
//   opt == 0 means SUNDIALS function allocates memory so check if
//            returned NULL pointer
//   opt == 1 means SUNDIALS function returns a flag so check if
//            flag >= 0
//   opt == 2 means function allocates memory so check if returned
//            NULL pointer
static int check_flag(void *flagvalue, const char *funcname, int opt) {
    int *errflag;

    /* Check if SUNDIALS function returned NULL pointer - no memory allocated */
    if (opt == 0 && flagvalue == NULL) {
        fprintf(stderr,
                "\nSUNDIALS_ERROR: %s() failed - returned NULL pointer\n\n",
                funcname);
        return 1;
    }

    /* Check if flag < 0 */
    else if (opt == 1) {
        errflag = (int *)flagvalue;
        if (*errflag < 0) {
            fprintf(stderr, "\nSUNDIALS_ERROR: %s() failed with flag = %d\n\n",
                    funcname, *errflag);
            return 1;
        }
    }

    /* Check if function returned NULL pointer - no memory allocated */
    else if (opt == 2 && flagvalue == NULL) {
        fprintf(stderr,
                "\nMEMORY_ERROR: %s() failed - returned NULL pointer\n\n",
                funcname);
        return 1;
    }

    return 0;
}

Naunet::Naunet(){};

Naunet::~Naunet(){};

int Naunet::Init(int nsystem, double atol, double rtol, int mxsteps) {
    n_system_ = nsystem;
    mxsteps_  = mxsteps;
    atol_     = atol;
    rtol_     = rtol;

    /* */
    if (nsystem != 1) {
        printf("This solver doesn't support nsystem > 1!");
        return NAUNET_FAIL;
    }

    cv_y_  = N_VNew_Serial((sunindextype)NEQUATIONS);
    cv_a_  = SUNSparseMatrix(NEQUATIONS, NEQUATIONS, NNZ, CSR_MAT);
    cv_ls_ = SUNLinSol_KLU(cv_y_, cv_a_);

    cv_mem_ = CVodeCreate(CV_BDF);

    int flag;

    flag = CVodeInit(cv_mem_, Fex, 0.0, cv_y_);
    if (check_flag(&flag, "CVodeInit", 1)) return 1;
    flag = CVodeSetMaxNumSteps(cv_mem_, mxsteps_);
    if (check_flag(&flag, "CVodeSetMaxNumSteps", 0)) return 1;
    flag = CVodeSStolerances(cv_mem_, rtol_, atol_);
    if (check_flag(&flag, "CVodeSStolerances", 1)) return 1;
    flag = CVodeSetLinearSolver(cv_mem_, cv_ls_, cv_a_);
    if (check_flag(&flag, "CVodeSetLinearSolver", 1)) return 1;
    flag = CVodeSetJacFn(cv_mem_, Jac);
    if (check_flag(&flag, "CVodeSetJacFn", 1)) return 1;

    /*  */

    // reset the n_vector to empty, maybe not necessary
    /* */

    // N_VDestroy(cv_y_);
    // cv_y_ = N_VNewEmpty_Serial((sunindextype)NEQUATIONS);

    /* */

    return NAUNET_SUCCESS;
};

int Naunet::DebugInfo() {
    long int nst, nfe, nsetups, nje, netf, nge, nni, ncfn;
    int flag;

    /* */

    flag = CVodeGetNumSteps(cv_mem_, &nst);
    check_flag(&flag, "CVodeGetNumSteps", 1);
    flag = CVodeGetNumRhsEvals(cv_mem_, &nfe);
    check_flag(&flag, "CVodeGetNumRhsEvals", 1);
    flag = CVodeGetNumLinSolvSetups(cv_mem_, &nsetups);
    check_flag(&flag, "CVodeGetNumLinSolvSetups", 1);
    flag = CVodeGetNumErrTestFails(cv_mem_, &netf);
    check_flag(&flag, "CVodeGetNumErrTestFails", 1);
    flag = CVodeGetNumNonlinSolvIters(cv_mem_, &nni);
    check_flag(&flag, "CVodeGetNumNonlinSolvIters", 1);
    flag = CVodeGetNumNonlinSolvConvFails(cv_mem_, &ncfn);
    check_flag(&flag, "CVodeGetNumNonlinSolvConvFails", 1);

    flag = CVodeGetNumJacEvals(cv_mem_, &nje);
    check_flag(&flag, "CVodeGetNumJacEvals", 1);

    flag = CVodeGetNumGEvals(cv_mem_, &nge);
    check_flag(&flag, "CVodeGetNumGEvals", 1);

    printf("\nFinal Statistics:\n");
    printf("nst = %-6ld nfe  = %-6ld nsetups = %-6ld nje = %ld\n", nst, nfe, nsetups, nje);
    printf("nni = %-6ld ncfn = %-6ld netf = %-6ld    nge = %ld\n \n", nni, ncfn, netf, nge);

    /*  */

    return NAUNET_SUCCESS;
}

int Naunet::Finalize() {

    /* */

    // N_VDestroy(cv_y_);
    N_VFreeEmpty(cv_y_);
    SUNMatDestroy(cv_a_);
    CVodeFree(&cv_mem_);
    SUNLinSolFree(cv_ls_);
    // delete m_data;

    /*  */

    return NAUNET_SUCCESS;
};

/*  */

int Naunet::Solve(realtype *ab, realtype dt, NaunetData *data) {

    int flag;

    /* */

    // realtype *ydata = N_VGetArrayPointer(cv_y_);
    // for (int i=0; i<NEQUATIONS; i++)
    // {
    //     ydata[i] = ab[i];
    // }
    N_VSetArrayPointer(ab, cv_y_);

    flag = CVodeReInit(cv_mem_, 0.0, cv_y_);
    if (check_flag(&flag, "CVodeReInit", 1)) return 1;
    flag = CVodeSetUserData(cv_mem_, data);
    if (check_flag(&flag, "CVodeSetUserData", 1)) return 1;

    realtype t0 = 0.0;
    flag = CVode(cv_mem_, dt, cv_y_, &t0, CV_NORMAL);

    ab   = N_VGetArrayPointer(cv_y_);

    /* */

    return NAUNET_SUCCESS;
};

#ifdef PYMODULE
py::array_t<realtype> Naunet::PyWrapSolve(py::array_t<realtype> arr,
                                          realtype dt, NaunetData *data) {
    py::buffer_info info = arr.request();
    realtype *ab         = static_cast<realtype *>(info.ptr);

    Solve(ab, dt, data);

    return py::array_t<realtype>(info.shape, ab);
}
#endif