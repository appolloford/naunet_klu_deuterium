#ifndef __NAUNET_H__
#define __NAUNET_H__

#include <sundials/sundials_linearsolver.h>
#include <sundials/sundials_math.h>   // contains the macros ABS, SUNSQR, EXP
#include <sundials/sundials_types.h>  // defs. of realtype, sunindextype
/* */

#include "naunet_data.h"
#include "naunet_macros.h"

#ifdef PYMODULE
#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>

namespace py = pybind11;
#endif

class Naunet {
   public:
    Naunet();
    ~Naunet();
    int Init(int nsystem = MAX_NSYSTEMS, double atol = 1e-20, double rtol = 1e-5, int mxsteps=500);
    int DebugInfo();
    int Finalize();
    /* */
    int Solve(realtype *ab, realtype dt, NaunetData *data);
#ifdef PYMODULE
    py::array_t<realtype> PyWrapSolve(py::array_t<realtype> arr, realtype dt,
                                      NaunetData *data);
#endif

   private:
    int n_system_;
    int mxsteps_;
    realtype atol_;
    realtype rtol_;

    /*  */

    N_Vector cv_y_;
    SUNMatrix cv_a_;
    void *cv_mem_;
    SUNLinearSolver cv_ls_;

    /*  */
};

#ifdef PYMODULE

PYBIND11_MODULE(PYMODNAME, m) {
    py::class_<Naunet>(m, "Naunet")
        .def(py::init())
        .def("Init", &Naunet::Init, py::arg("nsystem") = 1,
             py::arg("atol") = 1e-20, py::arg("rtol") = 1e-5,
             py::arg("mxsteps") = 500)
        .def("Finalize", &Naunet::Finalize)
#ifdef USE_CUDA
        .def("Reset", &Naunet::Reset, py::arg("nsystem") = 1,
             py::arg("atol") = 1e-20, py::arg("rtol") = 1e-5,
             py::arg("mxsteps") = 500)
#endif
        .def("Solve", &Naunet::PyWrapSolve);

    // clang-format off
    py::class_<NaunetData>(m, "NaunetData")
        .def(py::init())
        .def_readwrite("nH", &NaunetData::nH)
        .def_readwrite("Tgas", &NaunetData::Tgas)
        .def_readwrite("user_crflux", &NaunetData::user_crflux)
        .def_readwrite("user_Av", &NaunetData::user_Av)
        .def_readwrite("user_GtoDN", &NaunetData::user_GtoDN)
        ;
    // clang-format on
}

#endif

#endif