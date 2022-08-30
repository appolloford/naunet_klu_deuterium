#ifndef __NAUNET_DATA_H__
#define __NAUNET_DATA_H__

// 
// Struct for holding the nessesary additional variables for the problem.
struct NaunetData {
    // clang-format off
    double nH;
    double Tgas;
    double zeta = 1.300e-17;
    double Av = 1.000e+00;
    double omega = 5.000e-01;
    double user_crflux;
    double user_Av;
    double user_GtoDN;
    
    // clang-format on
};
#endif