// 
#include <math.h>
#include <stdio.h>

#include <algorithm>

#include "naunet_constants.h"
#include "naunet_macros.h"
#include "naunet_physics.h"
#include "naunet_utilities.h"

// clang-format off
double GetElementAbund(double *y, int elemidx) {
    if (elemidx == IDX_ELEM_GRAIN) {
        return 1.0*y[IDX_GRAINM] + 1.0*y[IDX_GRAIN0I] + 0.0;
    }
    if (elemidx == IDX_ELEM_He) {
        return 1.0*y[IDX_HeHII] + 1.0*y[IDX_HeDII] + 1.0*y[IDX_HeII] + 1.0*y[IDX_HeI] + 0.0;
    }
    if (elemidx == IDX_ELEM_C) {
        return 2.0*y[IDX_C2OII] + 1.0*y[IDX_NCOII] + 2.0*y[IDX_C2NI] + 2.0*y[IDX_CCOI] + 
               2.0*y[IDX_CNCII] + 3.0*y[IDX_C3I] + 3.0*y[IDX_C3II] + 1.0*y[IDX_OCNI] + 
               1.0*y[IDX_DNCI] + 1.0*y[IDX_HNCI] + 1.0*y[IDX_HOCII] + 1.0*y[IDX_DOCII] + 
               1.0*y[IDX_CNM] + 2.0*y[IDX_C2NII] + 1.0*y[IDX_DNCII] + 1.0*y[IDX_HNCII] + 
               1.0*y[IDX_CO2II] + 1.0*y[IDX_CM] + 2.0*y[IDX_C2DI] + 2.0*y[IDX_C2HI] + 
               1.0*y[IDX_CO2I] + 1.0*y[IDX_DCNI] + 1.0*y[IDX_HCNI] + 1.0*y[IDX_CD2I] + 
               1.0*y[IDX_CH2I] + 1.0*y[IDX_HCNII] + 1.0*y[IDX_DCNII] + 2.0*y[IDX_C2II] + 
               2.0*y[IDX_C2HII] + 2.0*y[IDX_C2DII] + 1.0*y[IDX_CHDI] + 1.0*y[IDX_CH2II] + 
               1.0*y[IDX_CD2II] + 1.0*y[IDX_CII] + 1.0*y[IDX_COII] + 1.0*y[IDX_CNII] + 
               1.0*y[IDX_CHDII] + 1.0*y[IDX_HCOI] + 1.0*y[IDX_DCOI] + 1.0*y[IDX_HCOII] + 
               1.0*y[IDX_DCOII] + 1.0*y[IDX_CHII] + 1.0*y[IDX_CDII] + 2.0*y[IDX_C2I] + 
               1.0*y[IDX_CHI] + 1.0*y[IDX_CDI] + 1.0*y[IDX_CNI] + 1.0*y[IDX_CI] + 
               1.0*y[IDX_COI] + 0.0;
    }
    if (elemidx == IDX_ELEM_N) {
        return 1.0*y[IDX_NCOII] + 1.0*y[IDX_C2NI] + 1.0*y[IDX_CNCII] + 2.0*y[IDX_N2OI] + 
               1.0*y[IDX_OCNI] + 1.0*y[IDX_DNCI] + 1.0*y[IDX_HNCI] + 1.0*y[IDX_CNM] + 
               1.0*y[IDX_NO2II] + 1.0*y[IDX_NO2I] + 1.0*y[IDX_DNOI] + 1.0*y[IDX_HNOI] + 
               1.0*y[IDX_C2NII] + 1.0*y[IDX_DNCII] + 1.0*y[IDX_HNCII] + 2.0*y[IDX_N2HII] + 
               2.0*y[IDX_N2DII] + 1.0*y[IDX_HNOII] + 1.0*y[IDX_DNOII] + 1.0*y[IDX_DCNI] + 
               1.0*y[IDX_HCNI] + 1.0*y[IDX_HCNII] + 1.0*y[IDX_ND2I] + 1.0*y[IDX_NH2I] + 
               1.0*y[IDX_DCNII] + 1.0*y[IDX_NHDI] + 1.0*y[IDX_NHII] + 1.0*y[IDX_NH2II] + 
               1.0*y[IDX_NII] + 2.0*y[IDX_N2II] + 1.0*y[IDX_NDII] + 1.0*y[IDX_ND2II] + 
               1.0*y[IDX_CNII] + 1.0*y[IDX_NHDII] + 1.0*y[IDX_NOII] + 1.0*y[IDX_NHI] + 
               2.0*y[IDX_N2I] + 1.0*y[IDX_NDI] + 1.0*y[IDX_NOI] + 1.0*y[IDX_CNI] + 
               1.0*y[IDX_NI] + 0.0;
    }
    if (elemidx == IDX_ELEM_O) {
        return 1.0*y[IDX_C2OII] + 1.0*y[IDX_NCOII] + 2.0*y[IDX_O2DI] + 2.0*y[IDX_O2HI] + 
               1.0*y[IDX_CCOI] + 1.0*y[IDX_N2OI] + 1.0*y[IDX_OCNI] + 1.0*y[IDX_HOCII] + 
               1.0*y[IDX_DOCII] + 1.0*y[IDX_ODM] + 1.0*y[IDX_OHM] + 2.0*y[IDX_NO2II] + 
               2.0*y[IDX_NO2I] + 1.0*y[IDX_DNOI] + 1.0*y[IDX_HNOI] + 1.0*y[IDX_OM] + 
               2.0*y[IDX_CO2II] + 2.0*y[IDX_O2HII] + 2.0*y[IDX_O2DII] + 1.0*y[IDX_HNOII] + 
               1.0*y[IDX_DNOII] + 1.0*y[IDX_H3OII] + 1.0*y[IDX_D3OII] + 2.0*y[IDX_CO2I] + 
               1.0*y[IDX_H2DOII] + 1.0*y[IDX_HD2OII] + 2.0*y[IDX_O2II] + 1.0*y[IDX_COII] + 
               1.0*y[IDX_OII] + 1.0*y[IDX_OHII] + 1.0*y[IDX_ODII] + 1.0*y[IDX_HCOI] + 
               1.0*y[IDX_DCOI] + 1.0*y[IDX_H2OII] + 1.0*y[IDX_HCOII] + 1.0*y[IDX_D2OII] + 
               1.0*y[IDX_DCOII] + 1.0*y[IDX_HDOII] + 1.0*y[IDX_NOII] + 2.0*y[IDX_O2I] + 
               1.0*y[IDX_D2OI] + 1.0*y[IDX_H2OI] + 1.0*y[IDX_NOI] + 1.0*y[IDX_HDOI] + 
               1.0*y[IDX_OHI] + 1.0*y[IDX_ODI] + 1.0*y[IDX_COI] + 1.0*y[IDX_OI] + 0.0;
    }
    if (elemidx == IDX_ELEM_D) {
        return 1.0*y[IDX_O2DI] + 1.0*y[IDX_HeDII] + 1.0*y[IDX_DNCI] + 1.0*y[IDX_DOCII] + 
               1.0*y[IDX_ODM] + 1.0*y[IDX_DNOI] + 1.0*y[IDX_DNCII] + 1.0*y[IDX_O2DII] + 
               1.0*y[IDX_N2DII] + 1.0*y[IDX_DNOII] + 1.0*y[IDX_C2DI] + 3.0*y[IDX_pD3II] + 
               3.0*y[IDX_D3OII] + 3.0*y[IDX_mD3II] + 3.0*y[IDX_oD3II] + 1.0*y[IDX_DM] + 
               1.0*y[IDX_DCNI] + 2.0*y[IDX_CD2I] + 2.0*y[IDX_ND2I] + 1.0*y[IDX_DCNII] + 
               1.0*y[IDX_C2DII] + 1.0*y[IDX_CHDI] + 1.0*y[IDX_NHDI] + 2.0*y[IDX_CD2II] + 
               1.0*y[IDX_H2DOII] + 2.0*y[IDX_HD2OII] + 1.0*y[IDX_NDII] + 2.0*y[IDX_ND2II] + 
               1.0*y[IDX_NHDII] + 1.0*y[IDX_ODII] + 1.0*y[IDX_CHDII] + 1.0*y[IDX_DCOI] + 
               2.0*y[IDX_oD2HII] + 1.0*y[IDX_oH2DII] + 2.0*y[IDX_pD2HII] + 1.0*y[IDX_pH2DII] + 
               2.0*y[IDX_D2OII] + 1.0*y[IDX_DCOII] + 2.0*y[IDX_oD2II] + 2.0*y[IDX_pD2II] + 
               1.0*y[IDX_CDII] + 1.0*y[IDX_HDOII] + 1.0*y[IDX_DII] + 1.0*y[IDX_HDII] + 
               1.0*y[IDX_NDI] + 1.0*y[IDX_CDI] + 2.0*y[IDX_D2OI] + 1.0*y[IDX_HDOI] + 
               2.0*y[IDX_oD2I] + 1.0*y[IDX_ODI] + 2.0*y[IDX_pD2I] + 1.0*y[IDX_HDI] + 
               1.0*y[IDX_DI] + 0.0;
    }
    if (elemidx == IDX_ELEM_H) {
        return 1.0*y[IDX_O2HI] + 1.0*y[IDX_HeHII] + 1.0*y[IDX_HNCI] + 1.0*y[IDX_HOCII] + 
               1.0*y[IDX_OHM] + 1.0*y[IDX_HNOI] + 1.0*y[IDX_HNCII] + 1.0*y[IDX_O2HII] + 
               1.0*y[IDX_N2HII] + 1.0*y[IDX_HNOII] + 1.0*y[IDX_C2HI] + 3.0*y[IDX_H3OII] + 
               1.0*y[IDX_HM] + 3.0*y[IDX_oH3II] + 3.0*y[IDX_pH3II] + 1.0*y[IDX_HCNI] + 
               2.0*y[IDX_CH2I] + 1.0*y[IDX_HCNII] + 2.0*y[IDX_NH2I] + 1.0*y[IDX_C2HII] + 
               1.0*y[IDX_CHDI] + 2.0*y[IDX_CH2II] + 1.0*y[IDX_NHDI] + 2.0*y[IDX_H2DOII] + 
               1.0*y[IDX_HD2OII] + 1.0*y[IDX_NHII] + 2.0*y[IDX_NH2II] + 1.0*y[IDX_NHDII] + 
               1.0*y[IDX_OHII] + 1.0*y[IDX_CHDII] + 1.0*y[IDX_HCOI] + 2.0*y[IDX_oH2II] + 
               1.0*y[IDX_oD2HII] + 2.0*y[IDX_oH2DII] + 1.0*y[IDX_pD2HII] + 2.0*y[IDX_pH2DII] + 
               2.0*y[IDX_pH2II] + 2.0*y[IDX_H2OII] + 1.0*y[IDX_HCOII] + 1.0*y[IDX_CHII] + 
               1.0*y[IDX_HDOII] + 1.0*y[IDX_HII] + 1.0*y[IDX_HDII] + 1.0*y[IDX_NHI] + 
               1.0*y[IDX_CHI] + 2.0*y[IDX_H2OI] + 1.0*y[IDX_HDOI] + 1.0*y[IDX_OHI] + 
               2.0*y[IDX_oH2I] + 2.0*y[IDX_pH2I] + 1.0*y[IDX_HDI] + 1.0*y[IDX_HI] + 0.0;
    }
    
}

double GetMantleDens(double *y) {
    return  + 0.0;
}

double GetHNuclei(double *y) {
#ifdef IDX_ELEM_H
    return GetElementAbund(y, IDX_ELEM_H);
#else
    return 0.0;
#endif
}

double GetMu(double *y) {
    // TODO: exclude electron, grain?
    double mass = 40.0*y[IDX_C2OII] + 42.0*y[IDX_NCOII] + 34.0*y[IDX_O2DI] + 33.0*y[IDX_O2HI] + 
                  38.0*y[IDX_C2NI] + 40.0*y[IDX_CCOI] + 38.0*y[IDX_CNCII] + 44.0*y[IDX_N2OI] + 
                  36.0*y[IDX_C3I] + 5.0*y[IDX_HeHII] + 36.0*y[IDX_C3II] + 6.0*y[IDX_HeDII] + 
                  42.0*y[IDX_OCNI] + 28.0*y[IDX_DNCI] + 27.0*y[IDX_HNCI] + 29.0*y[IDX_HOCII] + 
                  30.0*y[IDX_DOCII] + 18.0*y[IDX_ODM] + 17.0*y[IDX_OHM] + 26.0*y[IDX_CNM] + 
                  46.0*y[IDX_NO2II] + 46.0*y[IDX_NO2I] + 32.0*y[IDX_DNOI] + 31.0*y[IDX_HNOI] + 
                  38.0*y[IDX_C2NII] + 0.0*y[IDX_GRAINM] + 0.0*y[IDX_GRAIN0I] + 28.0*y[IDX_DNCII] + 
                  27.0*y[IDX_HNCII] + 16.0*y[IDX_OM] + 44.0*y[IDX_CO2II] + 12.0*y[IDX_CM] + 
                  33.0*y[IDX_O2HII] + 34.0*y[IDX_O2DII] + 29.0*y[IDX_N2HII] + 30.0*y[IDX_N2DII] + 
                  31.0*y[IDX_HNOII] + 32.0*y[IDX_DNOII] + 26.0*y[IDX_C2DI] + 25.0*y[IDX_C2HI] + 
                  19.0*y[IDX_H3OII] + 6.0*y[IDX_pD3II] + 22.0*y[IDX_D3OII] + 1.0*y[IDX_HM] + 
                  44.0*y[IDX_CO2I] + 6.0*y[IDX_mD3II] + 6.0*y[IDX_oD3II] + 2.0*y[IDX_DM] + 
                  3.0*y[IDX_oH3II] + 3.0*y[IDX_pH3II] + 28.0*y[IDX_DCNI] + 27.0*y[IDX_HCNI] + 
                  16.0*y[IDX_CD2I] + 14.0*y[IDX_CH2I] + 27.0*y[IDX_HCNII] + 18.0*y[IDX_ND2I] + 
                  16.0*y[IDX_NH2I] + 28.0*y[IDX_DCNII] + 24.0*y[IDX_C2II] + 25.0*y[IDX_C2HII] + 
                  26.0*y[IDX_C2DII] + 15.0*y[IDX_CHDI] + 14.0*y[IDX_CH2II] + 17.0*y[IDX_NHDI] + 
                  16.0*y[IDX_CD2II] + 20.0*y[IDX_H2DOII] + 21.0*y[IDX_HD2OII] + 15.0*y[IDX_NHII] + 
                  16.0*y[IDX_NH2II] + 12.0*y[IDX_CII] + 14.0*y[IDX_NII] + 28.0*y[IDX_N2II] + 
                  16.0*y[IDX_NDII] + 18.0*y[IDX_ND2II] + 32.0*y[IDX_O2II] + 28.0*y[IDX_COII] + 
                  26.0*y[IDX_CNII] + 4.0*y[IDX_HeII] + 17.0*y[IDX_NHDII] + 16.0*y[IDX_OII] + 
                  17.0*y[IDX_OHII] + 18.0*y[IDX_ODII] + 15.0*y[IDX_CHDII] + 29.0*y[IDX_HCOI] + 
                  2.0*y[IDX_oH2II] + 30.0*y[IDX_DCOI] + 5.0*y[IDX_oD2HII] + 4.0*y[IDX_oH2DII] + 
                  5.0*y[IDX_pD2HII] + 4.0*y[IDX_pH2DII] + 2.0*y[IDX_pH2II] + 18.0*y[IDX_H2OII] + 
                  29.0*y[IDX_HCOII] + 20.0*y[IDX_D2OII] + 30.0*y[IDX_DCOII] + 4.0*y[IDX_oD2II] + 
                  4.0*y[IDX_pD2II] + 13.0*y[IDX_CHII] + 4.0*y[IDX_HeI] + 14.0*y[IDX_CDII] + 
                  19.0*y[IDX_HDOII] + 1.0*y[IDX_HII] + 30.0*y[IDX_NOII] + 2.0*y[IDX_DII] + 
                  3.0*y[IDX_HDII] + 15.0*y[IDX_NHI] + 28.0*y[IDX_N2I] + 16.0*y[IDX_NDI] + 
                  24.0*y[IDX_C2I] + 13.0*y[IDX_CHI] + 14.0*y[IDX_CDI] + 32.0*y[IDX_O2I] + 
                  20.0*y[IDX_D2OI] + 18.0*y[IDX_H2OI] + 30.0*y[IDX_NOI] + 19.0*y[IDX_HDOI] + 
                  17.0*y[IDX_OHI] + 4.0*y[IDX_oD2I] + 2.0*y[IDX_oH2I] + 26.0*y[IDX_CNI] + 
                  18.0*y[IDX_ODI] + 12.0*y[IDX_CI] + 28.0*y[IDX_COI] + 14.0*y[IDX_NI] + 
                  16.0*y[IDX_OI] + 4.0*y[IDX_pD2I] + 2.0*y[IDX_pH2I] + 3.0*y[IDX_HDI] + 
                  0.0*y[IDX_eM] + 2.0*y[IDX_DI] + 1.0*y[IDX_HI] + 0.0;
    double num = y[IDX_C2OII] + y[IDX_NCOII] + y[IDX_O2DI] + y[IDX_O2HI] +
                 y[IDX_C2NI] + y[IDX_CCOI] + y[IDX_CNCII] + y[IDX_N2OI] +
                 y[IDX_C3I] + y[IDX_HeHII] + y[IDX_C3II] + y[IDX_HeDII] +
                 y[IDX_OCNI] + y[IDX_DNCI] + y[IDX_HNCI] + y[IDX_HOCII] +
                 y[IDX_DOCII] + y[IDX_ODM] + y[IDX_OHM] + y[IDX_CNM] +
                 y[IDX_NO2II] + y[IDX_NO2I] + y[IDX_DNOI] + y[IDX_HNOI] +
                 y[IDX_C2NII] + y[IDX_GRAINM] + y[IDX_GRAIN0I] + y[IDX_DNCII] +
                 y[IDX_HNCII] + y[IDX_OM] + y[IDX_CO2II] + y[IDX_CM] +
                 y[IDX_O2HII] + y[IDX_O2DII] + y[IDX_N2HII] + y[IDX_N2DII] +
                 y[IDX_HNOII] + y[IDX_DNOII] + y[IDX_C2DI] + y[IDX_C2HI] +
                 y[IDX_H3OII] + y[IDX_pD3II] + y[IDX_D3OII] + y[IDX_HM] +
                 y[IDX_CO2I] + y[IDX_mD3II] + y[IDX_oD3II] + y[IDX_DM] +
                 y[IDX_oH3II] + y[IDX_pH3II] + y[IDX_DCNI] + y[IDX_HCNI] +
                 y[IDX_CD2I] + y[IDX_CH2I] + y[IDX_HCNII] + y[IDX_ND2I] +
                 y[IDX_NH2I] + y[IDX_DCNII] + y[IDX_C2II] + y[IDX_C2HII] +
                 y[IDX_C2DII] + y[IDX_CHDI] + y[IDX_CH2II] + y[IDX_NHDI] +
                 y[IDX_CD2II] + y[IDX_H2DOII] + y[IDX_HD2OII] + y[IDX_NHII] +
                 y[IDX_NH2II] + y[IDX_CII] + y[IDX_NII] + y[IDX_N2II] +
                 y[IDX_NDII] + y[IDX_ND2II] + y[IDX_O2II] + y[IDX_COII] +
                 y[IDX_CNII] + y[IDX_HeII] + y[IDX_NHDII] + y[IDX_OII] +
                 y[IDX_OHII] + y[IDX_ODII] + y[IDX_CHDII] + y[IDX_HCOI] +
                 y[IDX_oH2II] + y[IDX_DCOI] + y[IDX_oD2HII] + y[IDX_oH2DII] +
                 y[IDX_pD2HII] + y[IDX_pH2DII] + y[IDX_pH2II] + y[IDX_H2OII] +
                 y[IDX_HCOII] + y[IDX_D2OII] + y[IDX_DCOII] + y[IDX_oD2II] +
                 y[IDX_pD2II] + y[IDX_CHII] + y[IDX_HeI] + y[IDX_CDII] +
                 y[IDX_HDOII] + y[IDX_HII] + y[IDX_NOII] + y[IDX_DII] +
                 y[IDX_HDII] + y[IDX_NHI] + y[IDX_N2I] + y[IDX_NDI] + y[IDX_C2I]
                 + y[IDX_CHI] + y[IDX_CDI] + y[IDX_O2I] + y[IDX_D2OI] +
                 y[IDX_H2OI] + y[IDX_NOI] + y[IDX_HDOI] + y[IDX_OHI] +
                 y[IDX_oD2I] + y[IDX_oH2I] + y[IDX_CNI] + y[IDX_ODI] + y[IDX_CI]
                 + y[IDX_COI] + y[IDX_NI] + y[IDX_OI] + y[IDX_pD2I] +
                 y[IDX_pH2I] + y[IDX_HDI] + y[IDX_eM] + y[IDX_DI] + y[IDX_HI] + 0.0;

    return mass / num;
}

double GetGamma(double *y) {
    // TODO: different ways to get adiabatic index
    return 5.0 / 3.0;
}

double GetNumDens(double *y) {
    double numdens = 0.0;

    for (int i = 0; i < NSPECIES; i++) numdens += y[i];
    return numdens;
}
// clang-format on

// clang-format off
double GetShieldingFactor(int specidx, double h2coldens, double spcoldens,
                          double tgas, int method) {
    // clang-format on
    double factor;
#ifdef IDX_H2I
    if (specidx == IDX_H2I) {
        factor = GetH2shielding(h2coldens, method);
    }
#endif
#ifdef IDX_COI
    if (specidx == IDX_COI) {
        factor = GetCOshielding(tgas, h2coldens, spcoldens, method);
    }
#endif
#ifdef IDX_N2I
    if (specidx == IDX_N2I) {
        factor = GetN2shielding(tgas, h2coldens, spcoldens, method);
    }
#endif

    return factor;
}

// clang-format off
double GetH2shielding(double coldens, int method) {
    // clang-format on
    double shielding = -1.0;
    switch (method) {
        case 0:
            shielding = GetH2shieldingInt(coldens);
            break;
        case 1:
            shielding = GetH2shieldingFGK(coldens);
            break;
        default:
            break;
    }
    return shielding;
}

// clang-format off
double GetCOshielding(double tgas, double h2col, double coldens, int method) {
    // clang-format on
    double shielding = -1.0;
    switch (method) {
        case 0:
            shielding = GetCOshieldingInt(tgas, h2col, coldens);
            break;
        case 1:
            shielding = GetCOshieldingInt1(h2col, coldens);
            break;
        default:
            break;
    }
    return shielding;
}

// clang-format off
double GetN2shielding(double tgas, double h2col, double coldens, int method) {
    // clang-format on
    double shielding = -1.0;
    switch (method) {
        case 0:
            shielding = GetN2shieldingInt(tgas, h2col, coldens);
            break;
        default:
            break;
    }
    return shielding;
}

// Interpolate/Extropolate from table (must be rendered in naunet constants)
// clang-format off
double GetH2shieldingInt(double coldens) {
    // clang-format on

    double shielding = -1.0;

    /* */

    printf("WARNING!! Not Implemented! Return H2 shielding = -1.\n");

    /* */

    return shielding;
}

// Calculates the line self shielding function
// Ref: Federman et al. apj vol.227 p.466.
// Originally implemented in UCLCHEM
// clang-format off
double GetH2shieldingFGK(double coldens) {
    // clang-format on

    const double dopplerwidth       = 3.0e10;
    const double radiativewidth     = 8.0e7;
    const double oscillatorstrength = 1.0e-2;

    double shielding                = -1.0;

    double taud = 0.5 * coldens * 1.5e-2 * oscillatorstrength / dopplerwidth;

    // Calculate wing contribution of self shielding function sr
    if (taud < 0.0) taud = 0.0;

    double sr = 0.0;
    if (radiativewidth != 0.0) {
        double r = radiativewidth / (1.7724539 * dopplerwidth);
        double t = 3.02 * pow(1000.0 * r, -0.064);
        double u = pow(taud * r, 0.5) / t;
        sr       = pow((u * u + 0.78539816), -0.5) * r / t;
    }

    // Calculate doppler contribution of self shielding function sj
    double sj = 0.0;
    if (taud == 0.0) {
        sj = 1.0;
    } else if (taud < 2.0) {
        sj = exp(-0.6666667 * taud);
    } else if (taud < 10.0) {
        sj = 0.638 * pow(taud, -1.25);
    } else if (taud < 100.0) {
        sj = 0.505 * pow(taud, -1.15);
    } else {
        sj = 0.344 * pow(taud, -1.0667);
    }

    shielding = sj + sr;

    return shielding;
}

// Interpolate/Extropolate from table (must be rendered in naunet constants)
// clang-format off
double GetCOshieldingInt(double tgas, double h2col, double coldens) {
    // clang-format on
    double shielding = -1.0;

    /* */

    printf("WARNING!! Not Implemented! Return CO shielding = -1.\n");

    /* */

    return shielding;
}

// clang-format off
double GetCOshieldingInt1(double h2col, double coldens) {
    // clang-format on
    double shielding = -1.0;

    /* */

    printf("WARNING!! Not Implemented! Return CO shielding = -1.\n");

    /* */

    return shielding;
}

// Interpolate/Extropolate from table (must be rendered in naunet constants)
// clang-format off
double GetN2shieldingInt(double tgas, double h2col, double coldens) {
    // clang-format on

    double shielding = -1.0;

    /* */

    printf("WARNING!! Not Implemented! Return N2 shielding = -1.\n");

    /* */

    return shielding;
}

// Calculate xlamda := tau(lambda) / tau(visual)
// tau(lambda) is the opt. depth for dust extinction at
// wavelength x (cf. b.d.savage and j.s.mathis, annual
// review of astronomy and astrophysics vol.17(1979),p.84)
// clang-format off
double xlamda(double wavelength) {
    // clang-format on
    double x[29] = {910.0,  950.0,  1000.0,  1050.0,  1110.0, 1180.0,
                    1250.0, 1390.0, 1490.0,  1600.0,  1700.0, 1800.0,
                    1900.0, 2000.0, 2100.0,  2190.0,  2300.0, 2400.0,
                    2500.0, 2740.0, 3440.0,  4000.0,  4400.0, 5500.0,
                    7000.0, 9000.0, 12500.0, 22000.0, 34000.0};

    double y[29] = {5.76, 5.18, 4.65, 4.16, 3.73, 3.4,  3.11, 2.74, 2.63, 2.62,
                    2.54, 2.5,  2.58, 2.78, 3.01, 3.12, 2.86, 2.58, 2.35, 2.0,
                    1.58, 1.42, 1.32, 1.0,  0.75, 0.48, 0.28, 0.12, 0.05};

    if (wavelength < x[0]) {
        return 5.76;
    }

    else if (wavelength >= x[28]) {
        return 0.05 - 5.16e-11 * (wavelength - x[28]);
    }

    for (int i = 0; i < 28; i++) {
        if (wavelength >= x[i] && wavelength < x[i + 1]) {
            return y[i] +
                   (y[i + 1] - y[i]) * (wavelength - x[i]) / (x[i + 1] - x[i]);
        }
    }

    return 0.0;
}

// Calculate the influence of dust extinction (g=0.8, omega=0.3)
// Ref: Wagenblast & Hartquist, mnras237, 1019 (1989)
// Adapted from UCLCHEM
// clang-format off
double GetGrainScattering(double av, double wavelength) {
    // clang-format on
    double c[6] = {1.0e0, 2.006e0, -1.438e0, 7.364e-1, -5.076e-1, -5.920e-2};
    double k[6] = {7.514e-1, 8.490e-1, 1.013e0, 1.282e0, 2.005e0, 5.832e0};

    double tv   = av / 1.086;
    double tl   = tv * xlamda(wavelength);

    double scat = 0.0;
    double expo;
    if (tl < 1.0) {
        expo = k[0] * tl;
        if (expo < 35.0) {
            scat = c[0] * exp(-expo);
        }
    } else {
        for (int i = 1; i < 6; i++) {
            expo = k[i] * tl;
            if (expo < 35.0) {
                scat = scat + c[i] * exp(-expo);
            }
        }
    }

    return scat;
}

// Calculate lambda bar (in a) according to equ. 4 of van dishoeck
// and black, apj 334, p771 (1988)
// Adapted from UCLCHEM
// clang-format off
double GetCharactWavelength(double h2col, double cocol) {
    // clang-format on
    double logco = log10(abs(cocol) + 1.0);
    double logh2 = log10(abs(h2col) + 1.0);

    double lbar  = (5675.0 - 200.6 * logh2) - (571.6 - 24.09 * logh2) * logco +
                  (18.22 - 0.7664 * logh2) * pow(logco, 2.0);

    // lbar represents the mean of the wavelengths of the 33
    // dissociating bands weighted by their fractional contribution
    // to the total rate of each depth. lbar cannot be larger than
    // the wavelength of band 33 (1076.1a) and not be smaller than
    // the wavelength of band 1 (913.6a).

    /* */
    lbar = std::min(1076.0, std::max(913.0, lbar));
    /* */
    return lbar;
}