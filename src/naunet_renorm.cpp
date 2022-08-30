#include "naunet_macros.h"
#include "naunet_physics.h"
#include "naunet_renorm.h"

// clang-format off
int InitRenorm(realtype *ab, SUNMatrix A) {
    // clang-format on
    realtype Hnuclei = GetHNuclei(ab);

    // clang-format off
            
    IJth(A, IDX_ELEM_GRAIN, IDX_ELEM_GRAIN) = 0.0 + 0.0 * ab[IDX_GRAINM] / 0.0 / Hnuclei +
                                    0.0 * ab[IDX_GRAIN0I] / 0.0 / Hnuclei;
    IJth(A, IDX_ELEM_GRAIN, IDX_ELEM_He) = 0.0;
    IJth(A, IDX_ELEM_GRAIN, IDX_ELEM_C) = 0.0;
    IJth(A, IDX_ELEM_GRAIN, IDX_ELEM_N) = 0.0;
    IJth(A, IDX_ELEM_GRAIN, IDX_ELEM_O) = 0.0;
    IJth(A, IDX_ELEM_GRAIN, IDX_ELEM_D) = 0.0;
    IJth(A, IDX_ELEM_GRAIN, IDX_ELEM_H) = 0.0;
    IJth(A, IDX_ELEM_He, IDX_ELEM_GRAIN) = 0.0;
    IJth(A, IDX_ELEM_He, IDX_ELEM_He) = 0.0 + 4.0 * ab[IDX_HeHII] / 5.0 / Hnuclei +
                                    4.0 * ab[IDX_HeDII] / 6.0 / Hnuclei + 4.0 *
                                    ab[IDX_HeII] / 4.0 / Hnuclei + 4.0 *
                                    ab[IDX_HeI] / 4.0 / Hnuclei;
    IJth(A, IDX_ELEM_He, IDX_ELEM_C) = 0.0;
    IJth(A, IDX_ELEM_He, IDX_ELEM_N) = 0.0;
    IJth(A, IDX_ELEM_He, IDX_ELEM_O) = 0.0;
    IJth(A, IDX_ELEM_He, IDX_ELEM_D) = 0.0 + 2.0 * ab[IDX_HeDII] / 6.0 / Hnuclei;
    IJth(A, IDX_ELEM_He, IDX_ELEM_H) = 0.0 + 1.0 * ab[IDX_HeHII] / 5.0 / Hnuclei;
    IJth(A, IDX_ELEM_C, IDX_ELEM_GRAIN) = 0.0;
    IJth(A, IDX_ELEM_C, IDX_ELEM_He) = 0.0;
    IJth(A, IDX_ELEM_C, IDX_ELEM_C) = 0.0 + 48.0 * ab[IDX_C2OII] / 40.0 / Hnuclei
                                    + 12.0 * ab[IDX_NCOII] / 42.0 / Hnuclei +
                                    48.0 * ab[IDX_C2NI] / 38.0 / Hnuclei + 48.0
                                    * ab[IDX_CCOI] / 40.0 / Hnuclei + 48.0 *
                                    ab[IDX_CNCII] / 38.0 / Hnuclei + 108.0 *
                                    ab[IDX_C3I] / 36.0 / Hnuclei + 108.0 *
                                    ab[IDX_C3II] / 36.0 / Hnuclei + 12.0 *
                                    ab[IDX_OCNI] / 42.0 / Hnuclei + 12.0 *
                                    ab[IDX_DNCI] / 28.0 / Hnuclei + 12.0 *
                                    ab[IDX_HNCI] / 27.0 / Hnuclei + 12.0 *
                                    ab[IDX_HOCII] / 29.0 / Hnuclei + 12.0 *
                                    ab[IDX_DOCII] / 30.0 / Hnuclei + 12.0 *
                                    ab[IDX_CNM] / 26.0 / Hnuclei + 48.0 *
                                    ab[IDX_C2NII] / 38.0 / Hnuclei + 12.0 *
                                    ab[IDX_DNCII] / 28.0 / Hnuclei + 12.0 *
                                    ab[IDX_HNCII] / 27.0 / Hnuclei + 12.0 *
                                    ab[IDX_CO2II] / 44.0 / Hnuclei + 12.0 *
                                    ab[IDX_CM] / 12.0 / Hnuclei + 48.0 *
                                    ab[IDX_C2DI] / 26.0 / Hnuclei + 48.0 *
                                    ab[IDX_C2HI] / 25.0 / Hnuclei + 12.0 *
                                    ab[IDX_CO2I] / 44.0 / Hnuclei + 12.0 *
                                    ab[IDX_DCNI] / 28.0 / Hnuclei + 12.0 *
                                    ab[IDX_HCNI] / 27.0 / Hnuclei + 12.0 *
                                    ab[IDX_CD2I] / 16.0 / Hnuclei + 12.0 *
                                    ab[IDX_CH2I] / 14.0 / Hnuclei + 12.0 *
                                    ab[IDX_HCNII] / 27.0 / Hnuclei + 12.0 *
                                    ab[IDX_DCNII] / 28.0 / Hnuclei + 48.0 *
                                    ab[IDX_C2II] / 24.0 / Hnuclei + 48.0 *
                                    ab[IDX_C2HII] / 25.0 / Hnuclei + 48.0 *
                                    ab[IDX_C2DII] / 26.0 / Hnuclei + 12.0 *
                                    ab[IDX_CHDI] / 15.0 / Hnuclei + 12.0 *
                                    ab[IDX_CH2II] / 14.0 / Hnuclei + 12.0 *
                                    ab[IDX_CD2II] / 16.0 / Hnuclei + 12.0 *
                                    ab[IDX_CII] / 12.0 / Hnuclei + 12.0 *
                                    ab[IDX_COII] / 28.0 / Hnuclei + 12.0 *
                                    ab[IDX_CNII] / 26.0 / Hnuclei + 12.0 *
                                    ab[IDX_CHDII] / 15.0 / Hnuclei + 12.0 *
                                    ab[IDX_HCOI] / 29.0 / Hnuclei + 12.0 *
                                    ab[IDX_DCOI] / 30.0 / Hnuclei + 12.0 *
                                    ab[IDX_HCOII] / 29.0 / Hnuclei + 12.0 *
                                    ab[IDX_DCOII] / 30.0 / Hnuclei + 12.0 *
                                    ab[IDX_CHII] / 13.0 / Hnuclei + 12.0 *
                                    ab[IDX_CDII] / 14.0 / Hnuclei + 48.0 *
                                    ab[IDX_C2I] / 24.0 / Hnuclei + 12.0 *
                                    ab[IDX_CHI] / 13.0 / Hnuclei + 12.0 *
                                    ab[IDX_CDI] / 14.0 / Hnuclei + 12.0 *
                                    ab[IDX_CNI] / 26.0 / Hnuclei + 12.0 *
                                    ab[IDX_CI] / 12.0 / Hnuclei + 12.0 *
                                    ab[IDX_COI] / 28.0 / Hnuclei;
    IJth(A, IDX_ELEM_C, IDX_ELEM_N) = 0.0 + 14.0 * ab[IDX_NCOII] / 42.0 / Hnuclei
                                    + 28.0 * ab[IDX_C2NI] / 38.0 / Hnuclei +
                                    28.0 * ab[IDX_CNCII] / 38.0 / Hnuclei + 14.0
                                    * ab[IDX_OCNI] / 42.0 / Hnuclei + 14.0 *
                                    ab[IDX_DNCI] / 28.0 / Hnuclei + 14.0 *
                                    ab[IDX_HNCI] / 27.0 / Hnuclei + 14.0 *
                                    ab[IDX_CNM] / 26.0 / Hnuclei + 28.0 *
                                    ab[IDX_C2NII] / 38.0 / Hnuclei + 14.0 *
                                    ab[IDX_DNCII] / 28.0 / Hnuclei + 14.0 *
                                    ab[IDX_HNCII] / 27.0 / Hnuclei + 14.0 *
                                    ab[IDX_DCNI] / 28.0 / Hnuclei + 14.0 *
                                    ab[IDX_HCNI] / 27.0 / Hnuclei + 14.0 *
                                    ab[IDX_HCNII] / 27.0 / Hnuclei + 14.0 *
                                    ab[IDX_DCNII] / 28.0 / Hnuclei + 14.0 *
                                    ab[IDX_CNII] / 26.0 / Hnuclei + 14.0 *
                                    ab[IDX_CNI] / 26.0 / Hnuclei;
    IJth(A, IDX_ELEM_C, IDX_ELEM_O) = 0.0 + 32.0 * ab[IDX_C2OII] / 40.0 / Hnuclei
                                    + 16.0 * ab[IDX_NCOII] / 42.0 / Hnuclei +
                                    32.0 * ab[IDX_CCOI] / 40.0 / Hnuclei + 16.0
                                    * ab[IDX_OCNI] / 42.0 / Hnuclei + 16.0 *
                                    ab[IDX_HOCII] / 29.0 / Hnuclei + 16.0 *
                                    ab[IDX_DOCII] / 30.0 / Hnuclei + 32.0 *
                                    ab[IDX_CO2II] / 44.0 / Hnuclei + 32.0 *
                                    ab[IDX_CO2I] / 44.0 / Hnuclei + 16.0 *
                                    ab[IDX_COII] / 28.0 / Hnuclei + 16.0 *
                                    ab[IDX_HCOI] / 29.0 / Hnuclei + 16.0 *
                                    ab[IDX_DCOI] / 30.0 / Hnuclei + 16.0 *
                                    ab[IDX_HCOII] / 29.0 / Hnuclei + 16.0 *
                                    ab[IDX_DCOII] / 30.0 / Hnuclei + 16.0 *
                                    ab[IDX_COI] / 28.0 / Hnuclei;
    IJth(A, IDX_ELEM_C, IDX_ELEM_D) = 0.0 + 2.0 * ab[IDX_DNCI] / 28.0 / Hnuclei +
                                    2.0 * ab[IDX_DOCII] / 30.0 / Hnuclei + 2.0 *
                                    ab[IDX_DNCII] / 28.0 / Hnuclei + 4.0 *
                                    ab[IDX_C2DI] / 26.0 / Hnuclei + 2.0 *
                                    ab[IDX_DCNI] / 28.0 / Hnuclei + 4.0 *
                                    ab[IDX_CD2I] / 16.0 / Hnuclei + 2.0 *
                                    ab[IDX_DCNII] / 28.0 / Hnuclei + 4.0 *
                                    ab[IDX_C2DII] / 26.0 / Hnuclei + 2.0 *
                                    ab[IDX_CHDI] / 15.0 / Hnuclei + 4.0 *
                                    ab[IDX_CD2II] / 16.0 / Hnuclei + 2.0 *
                                    ab[IDX_CHDII] / 15.0 / Hnuclei + 2.0 *
                                    ab[IDX_DCOI] / 30.0 / Hnuclei + 2.0 *
                                    ab[IDX_DCOII] / 30.0 / Hnuclei + 2.0 *
                                    ab[IDX_CDII] / 14.0 / Hnuclei + 2.0 *
                                    ab[IDX_CDI] / 14.0 / Hnuclei;
    IJth(A, IDX_ELEM_C, IDX_ELEM_H) = 0.0 + 1.0 * ab[IDX_HNCI] / 27.0 / Hnuclei +
                                    1.0 * ab[IDX_HOCII] / 29.0 / Hnuclei + 1.0 *
                                    ab[IDX_HNCII] / 27.0 / Hnuclei + 2.0 *
                                    ab[IDX_C2HI] / 25.0 / Hnuclei + 1.0 *
                                    ab[IDX_HCNI] / 27.0 / Hnuclei + 2.0 *
                                    ab[IDX_CH2I] / 14.0 / Hnuclei + 1.0 *
                                    ab[IDX_HCNII] / 27.0 / Hnuclei + 2.0 *
                                    ab[IDX_C2HII] / 25.0 / Hnuclei + 1.0 *
                                    ab[IDX_CHDI] / 15.0 / Hnuclei + 2.0 *
                                    ab[IDX_CH2II] / 14.0 / Hnuclei + 1.0 *
                                    ab[IDX_CHDII] / 15.0 / Hnuclei + 1.0 *
                                    ab[IDX_HCOI] / 29.0 / Hnuclei + 1.0 *
                                    ab[IDX_HCOII] / 29.0 / Hnuclei + 1.0 *
                                    ab[IDX_CHII] / 13.0 / Hnuclei + 1.0 *
                                    ab[IDX_CHI] / 13.0 / Hnuclei;
    IJth(A, IDX_ELEM_N, IDX_ELEM_GRAIN) = 0.0;
    IJth(A, IDX_ELEM_N, IDX_ELEM_He) = 0.0;
    IJth(A, IDX_ELEM_N, IDX_ELEM_C) = 0.0 + 12.0 * ab[IDX_NCOII] / 42.0 / Hnuclei
                                    + 24.0 * ab[IDX_C2NI] / 38.0 / Hnuclei +
                                    24.0 * ab[IDX_CNCII] / 38.0 / Hnuclei + 12.0
                                    * ab[IDX_OCNI] / 42.0 / Hnuclei + 12.0 *
                                    ab[IDX_DNCI] / 28.0 / Hnuclei + 12.0 *
                                    ab[IDX_HNCI] / 27.0 / Hnuclei + 12.0 *
                                    ab[IDX_CNM] / 26.0 / Hnuclei + 24.0 *
                                    ab[IDX_C2NII] / 38.0 / Hnuclei + 12.0 *
                                    ab[IDX_DNCII] / 28.0 / Hnuclei + 12.0 *
                                    ab[IDX_HNCII] / 27.0 / Hnuclei + 12.0 *
                                    ab[IDX_DCNI] / 28.0 / Hnuclei + 12.0 *
                                    ab[IDX_HCNI] / 27.0 / Hnuclei + 12.0 *
                                    ab[IDX_HCNII] / 27.0 / Hnuclei + 12.0 *
                                    ab[IDX_DCNII] / 28.0 / Hnuclei + 12.0 *
                                    ab[IDX_CNII] / 26.0 / Hnuclei + 12.0 *
                                    ab[IDX_CNI] / 26.0 / Hnuclei;
    IJth(A, IDX_ELEM_N, IDX_ELEM_N) = 0.0 + 14.0 * ab[IDX_NCOII] / 42.0 / Hnuclei
                                    + 14.0 * ab[IDX_C2NI] / 38.0 / Hnuclei +
                                    14.0 * ab[IDX_CNCII] / 38.0 / Hnuclei + 56.0
                                    * ab[IDX_N2OI] / 44.0 / Hnuclei + 14.0 *
                                    ab[IDX_OCNI] / 42.0 / Hnuclei + 14.0 *
                                    ab[IDX_DNCI] / 28.0 / Hnuclei + 14.0 *
                                    ab[IDX_HNCI] / 27.0 / Hnuclei + 14.0 *
                                    ab[IDX_CNM] / 26.0 / Hnuclei + 14.0 *
                                    ab[IDX_NO2II] / 46.0 / Hnuclei + 14.0 *
                                    ab[IDX_NO2I] / 46.0 / Hnuclei + 14.0 *
                                    ab[IDX_DNOI] / 32.0 / Hnuclei + 14.0 *
                                    ab[IDX_HNOI] / 31.0 / Hnuclei + 14.0 *
                                    ab[IDX_C2NII] / 38.0 / Hnuclei + 14.0 *
                                    ab[IDX_DNCII] / 28.0 / Hnuclei + 14.0 *
                                    ab[IDX_HNCII] / 27.0 / Hnuclei + 56.0 *
                                    ab[IDX_N2HII] / 29.0 / Hnuclei + 56.0 *
                                    ab[IDX_N2DII] / 30.0 / Hnuclei + 14.0 *
                                    ab[IDX_HNOII] / 31.0 / Hnuclei + 14.0 *
                                    ab[IDX_DNOII] / 32.0 / Hnuclei + 14.0 *
                                    ab[IDX_DCNI] / 28.0 / Hnuclei + 14.0 *
                                    ab[IDX_HCNI] / 27.0 / Hnuclei + 14.0 *
                                    ab[IDX_HCNII] / 27.0 / Hnuclei + 14.0 *
                                    ab[IDX_ND2I] / 18.0 / Hnuclei + 14.0 *
                                    ab[IDX_NH2I] / 16.0 / Hnuclei + 14.0 *
                                    ab[IDX_DCNII] / 28.0 / Hnuclei + 14.0 *
                                    ab[IDX_NHDI] / 17.0 / Hnuclei + 14.0 *
                                    ab[IDX_NHII] / 15.0 / Hnuclei + 14.0 *
                                    ab[IDX_NH2II] / 16.0 / Hnuclei + 14.0 *
                                    ab[IDX_NII] / 14.0 / Hnuclei + 56.0 *
                                    ab[IDX_N2II] / 28.0 / Hnuclei + 14.0 *
                                    ab[IDX_NDII] / 16.0 / Hnuclei + 14.0 *
                                    ab[IDX_ND2II] / 18.0 / Hnuclei + 14.0 *
                                    ab[IDX_CNII] / 26.0 / Hnuclei + 14.0 *
                                    ab[IDX_NHDII] / 17.0 / Hnuclei + 14.0 *
                                    ab[IDX_NOII] / 30.0 / Hnuclei + 14.0 *
                                    ab[IDX_NHI] / 15.0 / Hnuclei + 56.0 *
                                    ab[IDX_N2I] / 28.0 / Hnuclei + 14.0 *
                                    ab[IDX_NDI] / 16.0 / Hnuclei + 14.0 *
                                    ab[IDX_NOI] / 30.0 / Hnuclei + 14.0 *
                                    ab[IDX_CNI] / 26.0 / Hnuclei + 14.0 *
                                    ab[IDX_NI] / 14.0 / Hnuclei;
    IJth(A, IDX_ELEM_N, IDX_ELEM_O) = 0.0 + 16.0 * ab[IDX_NCOII] / 42.0 / Hnuclei
                                    + 32.0 * ab[IDX_N2OI] / 44.0 / Hnuclei +
                                    16.0 * ab[IDX_OCNI] / 42.0 / Hnuclei + 32.0
                                    * ab[IDX_NO2II] / 46.0 / Hnuclei + 32.0 *
                                    ab[IDX_NO2I] / 46.0 / Hnuclei + 16.0 *
                                    ab[IDX_DNOI] / 32.0 / Hnuclei + 16.0 *
                                    ab[IDX_HNOI] / 31.0 / Hnuclei + 16.0 *
                                    ab[IDX_HNOII] / 31.0 / Hnuclei + 16.0 *
                                    ab[IDX_DNOII] / 32.0 / Hnuclei + 16.0 *
                                    ab[IDX_NOII] / 30.0 / Hnuclei + 16.0 *
                                    ab[IDX_NOI] / 30.0 / Hnuclei;
    IJth(A, IDX_ELEM_N, IDX_ELEM_D) = 0.0 + 2.0 * ab[IDX_DNCI] / 28.0 / Hnuclei +
                                    2.0 * ab[IDX_DNOI] / 32.0 / Hnuclei + 2.0 *
                                    ab[IDX_DNCII] / 28.0 / Hnuclei + 4.0 *
                                    ab[IDX_N2DII] / 30.0 / Hnuclei + 2.0 *
                                    ab[IDX_DNOII] / 32.0 / Hnuclei + 2.0 *
                                    ab[IDX_DCNI] / 28.0 / Hnuclei + 4.0 *
                                    ab[IDX_ND2I] / 18.0 / Hnuclei + 2.0 *
                                    ab[IDX_DCNII] / 28.0 / Hnuclei + 2.0 *
                                    ab[IDX_NHDI] / 17.0 / Hnuclei + 2.0 *
                                    ab[IDX_NDII] / 16.0 / Hnuclei + 4.0 *
                                    ab[IDX_ND2II] / 18.0 / Hnuclei + 2.0 *
                                    ab[IDX_NHDII] / 17.0 / Hnuclei + 2.0 *
                                    ab[IDX_NDI] / 16.0 / Hnuclei;
    IJth(A, IDX_ELEM_N, IDX_ELEM_H) = 0.0 + 1.0 * ab[IDX_HNCI] / 27.0 / Hnuclei +
                                    1.0 * ab[IDX_HNOI] / 31.0 / Hnuclei + 1.0 *
                                    ab[IDX_HNCII] / 27.0 / Hnuclei + 2.0 *
                                    ab[IDX_N2HII] / 29.0 / Hnuclei + 1.0 *
                                    ab[IDX_HNOII] / 31.0 / Hnuclei + 1.0 *
                                    ab[IDX_HCNI] / 27.0 / Hnuclei + 1.0 *
                                    ab[IDX_HCNII] / 27.0 / Hnuclei + 2.0 *
                                    ab[IDX_NH2I] / 16.0 / Hnuclei + 1.0 *
                                    ab[IDX_NHDI] / 17.0 / Hnuclei + 1.0 *
                                    ab[IDX_NHII] / 15.0 / Hnuclei + 2.0 *
                                    ab[IDX_NH2II] / 16.0 / Hnuclei + 1.0 *
                                    ab[IDX_NHDII] / 17.0 / Hnuclei + 1.0 *
                                    ab[IDX_NHI] / 15.0 / Hnuclei;
    IJth(A, IDX_ELEM_O, IDX_ELEM_GRAIN) = 0.0;
    IJth(A, IDX_ELEM_O, IDX_ELEM_He) = 0.0;
    IJth(A, IDX_ELEM_O, IDX_ELEM_C) = 0.0 + 24.0 * ab[IDX_C2OII] / 40.0 / Hnuclei
                                    + 12.0 * ab[IDX_NCOII] / 42.0 / Hnuclei +
                                    24.0 * ab[IDX_CCOI] / 40.0 / Hnuclei + 12.0
                                    * ab[IDX_OCNI] / 42.0 / Hnuclei + 12.0 *
                                    ab[IDX_HOCII] / 29.0 / Hnuclei + 12.0 *
                                    ab[IDX_DOCII] / 30.0 / Hnuclei + 24.0 *
                                    ab[IDX_CO2II] / 44.0 / Hnuclei + 24.0 *
                                    ab[IDX_CO2I] / 44.0 / Hnuclei + 12.0 *
                                    ab[IDX_COII] / 28.0 / Hnuclei + 12.0 *
                                    ab[IDX_HCOI] / 29.0 / Hnuclei + 12.0 *
                                    ab[IDX_DCOI] / 30.0 / Hnuclei + 12.0 *
                                    ab[IDX_HCOII] / 29.0 / Hnuclei + 12.0 *
                                    ab[IDX_DCOII] / 30.0 / Hnuclei + 12.0 *
                                    ab[IDX_COI] / 28.0 / Hnuclei;
    IJth(A, IDX_ELEM_O, IDX_ELEM_N) = 0.0 + 14.0 * ab[IDX_NCOII] / 42.0 / Hnuclei
                                    + 28.0 * ab[IDX_N2OI] / 44.0 / Hnuclei +
                                    14.0 * ab[IDX_OCNI] / 42.0 / Hnuclei + 28.0
                                    * ab[IDX_NO2II] / 46.0 / Hnuclei + 28.0 *
                                    ab[IDX_NO2I] / 46.0 / Hnuclei + 14.0 *
                                    ab[IDX_DNOI] / 32.0 / Hnuclei + 14.0 *
                                    ab[IDX_HNOI] / 31.0 / Hnuclei + 14.0 *
                                    ab[IDX_HNOII] / 31.0 / Hnuclei + 14.0 *
                                    ab[IDX_DNOII] / 32.0 / Hnuclei + 14.0 *
                                    ab[IDX_NOII] / 30.0 / Hnuclei + 14.0 *
                                    ab[IDX_NOI] / 30.0 / Hnuclei;
    IJth(A, IDX_ELEM_O, IDX_ELEM_O) = 0.0 + 16.0 * ab[IDX_C2OII] / 40.0 / Hnuclei
                                    + 16.0 * ab[IDX_NCOII] / 42.0 / Hnuclei +
                                    64.0 * ab[IDX_O2DI] / 34.0 / Hnuclei + 64.0
                                    * ab[IDX_O2HI] / 33.0 / Hnuclei + 16.0 *
                                    ab[IDX_CCOI] / 40.0 / Hnuclei + 16.0 *
                                    ab[IDX_N2OI] / 44.0 / Hnuclei + 16.0 *
                                    ab[IDX_OCNI] / 42.0 / Hnuclei + 16.0 *
                                    ab[IDX_HOCII] / 29.0 / Hnuclei + 16.0 *
                                    ab[IDX_DOCII] / 30.0 / Hnuclei + 16.0 *
                                    ab[IDX_ODM] / 18.0 / Hnuclei + 16.0 *
                                    ab[IDX_OHM] / 17.0 / Hnuclei + 64.0 *
                                    ab[IDX_NO2II] / 46.0 / Hnuclei + 64.0 *
                                    ab[IDX_NO2I] / 46.0 / Hnuclei + 16.0 *
                                    ab[IDX_DNOI] / 32.0 / Hnuclei + 16.0 *
                                    ab[IDX_HNOI] / 31.0 / Hnuclei + 16.0 *
                                    ab[IDX_OM] / 16.0 / Hnuclei + 64.0 *
                                    ab[IDX_CO2II] / 44.0 / Hnuclei + 64.0 *
                                    ab[IDX_O2HII] / 33.0 / Hnuclei + 64.0 *
                                    ab[IDX_O2DII] / 34.0 / Hnuclei + 16.0 *
                                    ab[IDX_HNOII] / 31.0 / Hnuclei + 16.0 *
                                    ab[IDX_DNOII] / 32.0 / Hnuclei + 16.0 *
                                    ab[IDX_H3OII] / 19.0 / Hnuclei + 16.0 *
                                    ab[IDX_D3OII] / 22.0 / Hnuclei + 64.0 *
                                    ab[IDX_CO2I] / 44.0 / Hnuclei + 16.0 *
                                    ab[IDX_H2DOII] / 20.0 / Hnuclei + 16.0 *
                                    ab[IDX_HD2OII] / 21.0 / Hnuclei + 64.0 *
                                    ab[IDX_O2II] / 32.0 / Hnuclei + 16.0 *
                                    ab[IDX_COII] / 28.0 / Hnuclei + 16.0 *
                                    ab[IDX_OII] / 16.0 / Hnuclei + 16.0 *
                                    ab[IDX_OHII] / 17.0 / Hnuclei + 16.0 *
                                    ab[IDX_ODII] / 18.0 / Hnuclei + 16.0 *
                                    ab[IDX_HCOI] / 29.0 / Hnuclei + 16.0 *
                                    ab[IDX_DCOI] / 30.0 / Hnuclei + 16.0 *
                                    ab[IDX_H2OII] / 18.0 / Hnuclei + 16.0 *
                                    ab[IDX_HCOII] / 29.0 / Hnuclei + 16.0 *
                                    ab[IDX_D2OII] / 20.0 / Hnuclei + 16.0 *
                                    ab[IDX_DCOII] / 30.0 / Hnuclei + 16.0 *
                                    ab[IDX_HDOII] / 19.0 / Hnuclei + 16.0 *
                                    ab[IDX_NOII] / 30.0 / Hnuclei + 64.0 *
                                    ab[IDX_O2I] / 32.0 / Hnuclei + 16.0 *
                                    ab[IDX_D2OI] / 20.0 / Hnuclei + 16.0 *
                                    ab[IDX_H2OI] / 18.0 / Hnuclei + 16.0 *
                                    ab[IDX_NOI] / 30.0 / Hnuclei + 16.0 *
                                    ab[IDX_HDOI] / 19.0 / Hnuclei + 16.0 *
                                    ab[IDX_OHI] / 17.0 / Hnuclei + 16.0 *
                                    ab[IDX_ODI] / 18.0 / Hnuclei + 16.0 *
                                    ab[IDX_COI] / 28.0 / Hnuclei + 16.0 *
                                    ab[IDX_OI] / 16.0 / Hnuclei;
    IJth(A, IDX_ELEM_O, IDX_ELEM_D) = 0.0 + 4.0 * ab[IDX_O2DI] / 34.0 / Hnuclei +
                                    2.0 * ab[IDX_DOCII] / 30.0 / Hnuclei + 2.0 *
                                    ab[IDX_ODM] / 18.0 / Hnuclei + 2.0 *
                                    ab[IDX_DNOI] / 32.0 / Hnuclei + 4.0 *
                                    ab[IDX_O2DII] / 34.0 / Hnuclei + 2.0 *
                                    ab[IDX_DNOII] / 32.0 / Hnuclei + 6.0 *
                                    ab[IDX_D3OII] / 22.0 / Hnuclei + 2.0 *
                                    ab[IDX_H2DOII] / 20.0 / Hnuclei + 4.0 *
                                    ab[IDX_HD2OII] / 21.0 / Hnuclei + 2.0 *
                                    ab[IDX_ODII] / 18.0 / Hnuclei + 2.0 *
                                    ab[IDX_DCOI] / 30.0 / Hnuclei + 4.0 *
                                    ab[IDX_D2OII] / 20.0 / Hnuclei + 2.0 *
                                    ab[IDX_DCOII] / 30.0 / Hnuclei + 2.0 *
                                    ab[IDX_HDOII] / 19.0 / Hnuclei + 4.0 *
                                    ab[IDX_D2OI] / 20.0 / Hnuclei + 2.0 *
                                    ab[IDX_HDOI] / 19.0 / Hnuclei + 2.0 *
                                    ab[IDX_ODI] / 18.0 / Hnuclei;
    IJth(A, IDX_ELEM_O, IDX_ELEM_H) = 0.0 + 2.0 * ab[IDX_O2HI] / 33.0 / Hnuclei +
                                    1.0 * ab[IDX_HOCII] / 29.0 / Hnuclei + 1.0 *
                                    ab[IDX_OHM] / 17.0 / Hnuclei + 1.0 *
                                    ab[IDX_HNOI] / 31.0 / Hnuclei + 2.0 *
                                    ab[IDX_O2HII] / 33.0 / Hnuclei + 1.0 *
                                    ab[IDX_HNOII] / 31.0 / Hnuclei + 3.0 *
                                    ab[IDX_H3OII] / 19.0 / Hnuclei + 2.0 *
                                    ab[IDX_H2DOII] / 20.0 / Hnuclei + 1.0 *
                                    ab[IDX_HD2OII] / 21.0 / Hnuclei + 1.0 *
                                    ab[IDX_OHII] / 17.0 / Hnuclei + 1.0 *
                                    ab[IDX_HCOI] / 29.0 / Hnuclei + 2.0 *
                                    ab[IDX_H2OII] / 18.0 / Hnuclei + 1.0 *
                                    ab[IDX_HCOII] / 29.0 / Hnuclei + 1.0 *
                                    ab[IDX_HDOII] / 19.0 / Hnuclei + 2.0 *
                                    ab[IDX_H2OI] / 18.0 / Hnuclei + 1.0 *
                                    ab[IDX_HDOI] / 19.0 / Hnuclei + 1.0 *
                                    ab[IDX_OHI] / 17.0 / Hnuclei;
    IJth(A, IDX_ELEM_D, IDX_ELEM_GRAIN) = 0.0;
    IJth(A, IDX_ELEM_D, IDX_ELEM_He) = 0.0 + 4.0 * ab[IDX_HeDII] / 6.0 / Hnuclei;
    IJth(A, IDX_ELEM_D, IDX_ELEM_C) = 0.0 + 12.0 * ab[IDX_DNCI] / 28.0 / Hnuclei +
                                    12.0 * ab[IDX_DOCII] / 30.0 / Hnuclei + 12.0
                                    * ab[IDX_DNCII] / 28.0 / Hnuclei + 24.0 *
                                    ab[IDX_C2DI] / 26.0 / Hnuclei + 12.0 *
                                    ab[IDX_DCNI] / 28.0 / Hnuclei + 24.0 *
                                    ab[IDX_CD2I] / 16.0 / Hnuclei + 12.0 *
                                    ab[IDX_DCNII] / 28.0 / Hnuclei + 24.0 *
                                    ab[IDX_C2DII] / 26.0 / Hnuclei + 12.0 *
                                    ab[IDX_CHDI] / 15.0 / Hnuclei + 24.0 *
                                    ab[IDX_CD2II] / 16.0 / Hnuclei + 12.0 *
                                    ab[IDX_CHDII] / 15.0 / Hnuclei + 12.0 *
                                    ab[IDX_DCOI] / 30.0 / Hnuclei + 12.0 *
                                    ab[IDX_DCOII] / 30.0 / Hnuclei + 12.0 *
                                    ab[IDX_CDII] / 14.0 / Hnuclei + 12.0 *
                                    ab[IDX_CDI] / 14.0 / Hnuclei;
    IJth(A, IDX_ELEM_D, IDX_ELEM_N) = 0.0 + 14.0 * ab[IDX_DNCI] / 28.0 / Hnuclei +
                                    14.0 * ab[IDX_DNOI] / 32.0 / Hnuclei + 14.0
                                    * ab[IDX_DNCII] / 28.0 / Hnuclei + 28.0 *
                                    ab[IDX_N2DII] / 30.0 / Hnuclei + 14.0 *
                                    ab[IDX_DNOII] / 32.0 / Hnuclei + 14.0 *
                                    ab[IDX_DCNI] / 28.0 / Hnuclei + 28.0 *
                                    ab[IDX_ND2I] / 18.0 / Hnuclei + 14.0 *
                                    ab[IDX_DCNII] / 28.0 / Hnuclei + 14.0 *
                                    ab[IDX_NHDI] / 17.0 / Hnuclei + 14.0 *
                                    ab[IDX_NDII] / 16.0 / Hnuclei + 28.0 *
                                    ab[IDX_ND2II] / 18.0 / Hnuclei + 14.0 *
                                    ab[IDX_NHDII] / 17.0 / Hnuclei + 14.0 *
                                    ab[IDX_NDI] / 16.0 / Hnuclei;
    IJth(A, IDX_ELEM_D, IDX_ELEM_O) = 0.0 + 32.0 * ab[IDX_O2DI] / 34.0 / Hnuclei +
                                    16.0 * ab[IDX_DOCII] / 30.0 / Hnuclei + 16.0
                                    * ab[IDX_ODM] / 18.0 / Hnuclei + 16.0 *
                                    ab[IDX_DNOI] / 32.0 / Hnuclei + 32.0 *
                                    ab[IDX_O2DII] / 34.0 / Hnuclei + 16.0 *
                                    ab[IDX_DNOII] / 32.0 / Hnuclei + 48.0 *
                                    ab[IDX_D3OII] / 22.0 / Hnuclei + 16.0 *
                                    ab[IDX_H2DOII] / 20.0 / Hnuclei + 32.0 *
                                    ab[IDX_HD2OII] / 21.0 / Hnuclei + 16.0 *
                                    ab[IDX_ODII] / 18.0 / Hnuclei + 16.0 *
                                    ab[IDX_DCOI] / 30.0 / Hnuclei + 32.0 *
                                    ab[IDX_D2OII] / 20.0 / Hnuclei + 16.0 *
                                    ab[IDX_DCOII] / 30.0 / Hnuclei + 16.0 *
                                    ab[IDX_HDOII] / 19.0 / Hnuclei + 32.0 *
                                    ab[IDX_D2OI] / 20.0 / Hnuclei + 16.0 *
                                    ab[IDX_HDOI] / 19.0 / Hnuclei + 16.0 *
                                    ab[IDX_ODI] / 18.0 / Hnuclei;
    IJth(A, IDX_ELEM_D, IDX_ELEM_D) = 0.0 + 2.0 * ab[IDX_O2DI] / 34.0 / Hnuclei +
                                    2.0 * ab[IDX_HeDII] / 6.0 / Hnuclei + 2.0 *
                                    ab[IDX_DNCI] / 28.0 / Hnuclei + 2.0 *
                                    ab[IDX_DOCII] / 30.0 / Hnuclei + 2.0 *
                                    ab[IDX_ODM] / 18.0 / Hnuclei + 2.0 *
                                    ab[IDX_DNOI] / 32.0 / Hnuclei + 2.0 *
                                    ab[IDX_DNCII] / 28.0 / Hnuclei + 2.0 *
                                    ab[IDX_O2DII] / 34.0 / Hnuclei + 2.0 *
                                    ab[IDX_N2DII] / 30.0 / Hnuclei + 2.0 *
                                    ab[IDX_DNOII] / 32.0 / Hnuclei + 2.0 *
                                    ab[IDX_C2DI] / 26.0 / Hnuclei + 18.0 *
                                    ab[IDX_pD3II] / 6.0 / Hnuclei + 18.0 *
                                    ab[IDX_D3OII] / 22.0 / Hnuclei + 18.0 *
                                    ab[IDX_mD3II] / 6.0 / Hnuclei + 18.0 *
                                    ab[IDX_oD3II] / 6.0 / Hnuclei + 2.0 *
                                    ab[IDX_DM] / 2.0 / Hnuclei + 2.0 *
                                    ab[IDX_DCNI] / 28.0 / Hnuclei + 8.0 *
                                    ab[IDX_CD2I] / 16.0 / Hnuclei + 8.0 *
                                    ab[IDX_ND2I] / 18.0 / Hnuclei + 2.0 *
                                    ab[IDX_DCNII] / 28.0 / Hnuclei + 2.0 *
                                    ab[IDX_C2DII] / 26.0 / Hnuclei + 2.0 *
                                    ab[IDX_CHDI] / 15.0 / Hnuclei + 2.0 *
                                    ab[IDX_NHDI] / 17.0 / Hnuclei + 8.0 *
                                    ab[IDX_CD2II] / 16.0 / Hnuclei + 2.0 *
                                    ab[IDX_H2DOII] / 20.0 / Hnuclei + 8.0 *
                                    ab[IDX_HD2OII] / 21.0 / Hnuclei + 2.0 *
                                    ab[IDX_NDII] / 16.0 / Hnuclei + 8.0 *
                                    ab[IDX_ND2II] / 18.0 / Hnuclei + 2.0 *
                                    ab[IDX_NHDII] / 17.0 / Hnuclei + 2.0 *
                                    ab[IDX_ODII] / 18.0 / Hnuclei + 2.0 *
                                    ab[IDX_CHDII] / 15.0 / Hnuclei + 2.0 *
                                    ab[IDX_DCOI] / 30.0 / Hnuclei + 8.0 *
                                    ab[IDX_oD2HII] / 5.0 / Hnuclei + 2.0 *
                                    ab[IDX_oH2DII] / 4.0 / Hnuclei + 8.0 *
                                    ab[IDX_pD2HII] / 5.0 / Hnuclei + 2.0 *
                                    ab[IDX_pH2DII] / 4.0 / Hnuclei + 8.0 *
                                    ab[IDX_D2OII] / 20.0 / Hnuclei + 2.0 *
                                    ab[IDX_DCOII] / 30.0 / Hnuclei + 8.0 *
                                    ab[IDX_oD2II] / 4.0 / Hnuclei + 8.0 *
                                    ab[IDX_pD2II] / 4.0 / Hnuclei + 2.0 *
                                    ab[IDX_CDII] / 14.0 / Hnuclei + 2.0 *
                                    ab[IDX_HDOII] / 19.0 / Hnuclei + 2.0 *
                                    ab[IDX_DII] / 2.0 / Hnuclei + 2.0 *
                                    ab[IDX_HDII] / 3.0 / Hnuclei + 2.0 *
                                    ab[IDX_NDI] / 16.0 / Hnuclei + 2.0 *
                                    ab[IDX_CDI] / 14.0 / Hnuclei + 8.0 *
                                    ab[IDX_D2OI] / 20.0 / Hnuclei + 2.0 *
                                    ab[IDX_HDOI] / 19.0 / Hnuclei + 8.0 *
                                    ab[IDX_oD2I] / 4.0 / Hnuclei + 2.0 *
                                    ab[IDX_ODI] / 18.0 / Hnuclei + 8.0 *
                                    ab[IDX_pD2I] / 4.0 / Hnuclei + 2.0 *
                                    ab[IDX_HDI] / 3.0 / Hnuclei + 2.0 *
                                    ab[IDX_DI] / 2.0 / Hnuclei;
    IJth(A, IDX_ELEM_D, IDX_ELEM_H) = 0.0 + 1.0 * ab[IDX_CHDI] / 15.0 / Hnuclei +
                                    1.0 * ab[IDX_NHDI] / 17.0 / Hnuclei + 2.0 *
                                    ab[IDX_H2DOII] / 20.0 / Hnuclei + 2.0 *
                                    ab[IDX_HD2OII] / 21.0 / Hnuclei + 1.0 *
                                    ab[IDX_NHDII] / 17.0 / Hnuclei + 1.0 *
                                    ab[IDX_CHDII] / 15.0 / Hnuclei + 2.0 *
                                    ab[IDX_oD2HII] / 5.0 / Hnuclei + 2.0 *
                                    ab[IDX_oH2DII] / 4.0 / Hnuclei + 2.0 *
                                    ab[IDX_pD2HII] / 5.0 / Hnuclei + 2.0 *
                                    ab[IDX_pH2DII] / 4.0 / Hnuclei + 1.0 *
                                    ab[IDX_HDOII] / 19.0 / Hnuclei + 1.0 *
                                    ab[IDX_HDII] / 3.0 / Hnuclei + 1.0 *
                                    ab[IDX_HDOI] / 19.0 / Hnuclei + 1.0 *
                                    ab[IDX_HDI] / 3.0 / Hnuclei;
    IJth(A, IDX_ELEM_H, IDX_ELEM_GRAIN) = 0.0;
    IJth(A, IDX_ELEM_H, IDX_ELEM_He) = 0.0 + 4.0 * ab[IDX_HeHII] / 5.0 / Hnuclei;
    IJth(A, IDX_ELEM_H, IDX_ELEM_C) = 0.0 + 12.0 * ab[IDX_HNCI] / 27.0 / Hnuclei +
                                    12.0 * ab[IDX_HOCII] / 29.0 / Hnuclei + 12.0
                                    * ab[IDX_HNCII] / 27.0 / Hnuclei + 24.0 *
                                    ab[IDX_C2HI] / 25.0 / Hnuclei + 12.0 *
                                    ab[IDX_HCNI] / 27.0 / Hnuclei + 24.0 *
                                    ab[IDX_CH2I] / 14.0 / Hnuclei + 12.0 *
                                    ab[IDX_HCNII] / 27.0 / Hnuclei + 24.0 *
                                    ab[IDX_C2HII] / 25.0 / Hnuclei + 12.0 *
                                    ab[IDX_CHDI] / 15.0 / Hnuclei + 24.0 *
                                    ab[IDX_CH2II] / 14.0 / Hnuclei + 12.0 *
                                    ab[IDX_CHDII] / 15.0 / Hnuclei + 12.0 *
                                    ab[IDX_HCOI] / 29.0 / Hnuclei + 12.0 *
                                    ab[IDX_HCOII] / 29.0 / Hnuclei + 12.0 *
                                    ab[IDX_CHII] / 13.0 / Hnuclei + 12.0 *
                                    ab[IDX_CHI] / 13.0 / Hnuclei;
    IJth(A, IDX_ELEM_H, IDX_ELEM_N) = 0.0 + 14.0 * ab[IDX_HNCI] / 27.0 / Hnuclei +
                                    14.0 * ab[IDX_HNOI] / 31.0 / Hnuclei + 14.0
                                    * ab[IDX_HNCII] / 27.0 / Hnuclei + 28.0 *
                                    ab[IDX_N2HII] / 29.0 / Hnuclei + 14.0 *
                                    ab[IDX_HNOII] / 31.0 / Hnuclei + 14.0 *
                                    ab[IDX_HCNI] / 27.0 / Hnuclei + 14.0 *
                                    ab[IDX_HCNII] / 27.0 / Hnuclei + 28.0 *
                                    ab[IDX_NH2I] / 16.0 / Hnuclei + 14.0 *
                                    ab[IDX_NHDI] / 17.0 / Hnuclei + 14.0 *
                                    ab[IDX_NHII] / 15.0 / Hnuclei + 28.0 *
                                    ab[IDX_NH2II] / 16.0 / Hnuclei + 14.0 *
                                    ab[IDX_NHDII] / 17.0 / Hnuclei + 14.0 *
                                    ab[IDX_NHI] / 15.0 / Hnuclei;
    IJth(A, IDX_ELEM_H, IDX_ELEM_O) = 0.0 + 32.0 * ab[IDX_O2HI] / 33.0 / Hnuclei +
                                    16.0 * ab[IDX_HOCII] / 29.0 / Hnuclei + 16.0
                                    * ab[IDX_OHM] / 17.0 / Hnuclei + 16.0 *
                                    ab[IDX_HNOI] / 31.0 / Hnuclei + 32.0 *
                                    ab[IDX_O2HII] / 33.0 / Hnuclei + 16.0 *
                                    ab[IDX_HNOII] / 31.0 / Hnuclei + 48.0 *
                                    ab[IDX_H3OII] / 19.0 / Hnuclei + 32.0 *
                                    ab[IDX_H2DOII] / 20.0 / Hnuclei + 16.0 *
                                    ab[IDX_HD2OII] / 21.0 / Hnuclei + 16.0 *
                                    ab[IDX_OHII] / 17.0 / Hnuclei + 16.0 *
                                    ab[IDX_HCOI] / 29.0 / Hnuclei + 32.0 *
                                    ab[IDX_H2OII] / 18.0 / Hnuclei + 16.0 *
                                    ab[IDX_HCOII] / 29.0 / Hnuclei + 16.0 *
                                    ab[IDX_HDOII] / 19.0 / Hnuclei + 32.0 *
                                    ab[IDX_H2OI] / 18.0 / Hnuclei + 16.0 *
                                    ab[IDX_HDOI] / 19.0 / Hnuclei + 16.0 *
                                    ab[IDX_OHI] / 17.0 / Hnuclei;
    IJth(A, IDX_ELEM_H, IDX_ELEM_D) = 0.0 + 2.0 * ab[IDX_CHDI] / 15.0 / Hnuclei +
                                    2.0 * ab[IDX_NHDI] / 17.0 / Hnuclei + 4.0 *
                                    ab[IDX_H2DOII] / 20.0 / Hnuclei + 4.0 *
                                    ab[IDX_HD2OII] / 21.0 / Hnuclei + 2.0 *
                                    ab[IDX_NHDII] / 17.0 / Hnuclei + 2.0 *
                                    ab[IDX_CHDII] / 15.0 / Hnuclei + 4.0 *
                                    ab[IDX_oD2HII] / 5.0 / Hnuclei + 4.0 *
                                    ab[IDX_oH2DII] / 4.0 / Hnuclei + 4.0 *
                                    ab[IDX_pD2HII] / 5.0 / Hnuclei + 4.0 *
                                    ab[IDX_pH2DII] / 4.0 / Hnuclei + 2.0 *
                                    ab[IDX_HDOII] / 19.0 / Hnuclei + 2.0 *
                                    ab[IDX_HDII] / 3.0 / Hnuclei + 2.0 *
                                    ab[IDX_HDOI] / 19.0 / Hnuclei + 2.0 *
                                    ab[IDX_HDI] / 3.0 / Hnuclei;
    IJth(A, IDX_ELEM_H, IDX_ELEM_H) = 0.0 + 1.0 * ab[IDX_O2HI] / 33.0 / Hnuclei +
                                    1.0 * ab[IDX_HeHII] / 5.0 / Hnuclei + 1.0 *
                                    ab[IDX_HNCI] / 27.0 / Hnuclei + 1.0 *
                                    ab[IDX_HOCII] / 29.0 / Hnuclei + 1.0 *
                                    ab[IDX_OHM] / 17.0 / Hnuclei + 1.0 *
                                    ab[IDX_HNOI] / 31.0 / Hnuclei + 1.0 *
                                    ab[IDX_HNCII] / 27.0 / Hnuclei + 1.0 *
                                    ab[IDX_O2HII] / 33.0 / Hnuclei + 1.0 *
                                    ab[IDX_N2HII] / 29.0 / Hnuclei + 1.0 *
                                    ab[IDX_HNOII] / 31.0 / Hnuclei + 1.0 *
                                    ab[IDX_C2HI] / 25.0 / Hnuclei + 9.0 *
                                    ab[IDX_H3OII] / 19.0 / Hnuclei + 1.0 *
                                    ab[IDX_HM] / 1.0 / Hnuclei + 9.0 *
                                    ab[IDX_oH3II] / 3.0 / Hnuclei + 9.0 *
                                    ab[IDX_pH3II] / 3.0 / Hnuclei + 1.0 *
                                    ab[IDX_HCNI] / 27.0 / Hnuclei + 4.0 *
                                    ab[IDX_CH2I] / 14.0 / Hnuclei + 1.0 *
                                    ab[IDX_HCNII] / 27.0 / Hnuclei + 4.0 *
                                    ab[IDX_NH2I] / 16.0 / Hnuclei + 1.0 *
                                    ab[IDX_C2HII] / 25.0 / Hnuclei + 1.0 *
                                    ab[IDX_CHDI] / 15.0 / Hnuclei + 4.0 *
                                    ab[IDX_CH2II] / 14.0 / Hnuclei + 1.0 *
                                    ab[IDX_NHDI] / 17.0 / Hnuclei + 4.0 *
                                    ab[IDX_H2DOII] / 20.0 / Hnuclei + 1.0 *
                                    ab[IDX_HD2OII] / 21.0 / Hnuclei + 1.0 *
                                    ab[IDX_NHII] / 15.0 / Hnuclei + 4.0 *
                                    ab[IDX_NH2II] / 16.0 / Hnuclei + 1.0 *
                                    ab[IDX_NHDII] / 17.0 / Hnuclei + 1.0 *
                                    ab[IDX_OHII] / 17.0 / Hnuclei + 1.0 *
                                    ab[IDX_CHDII] / 15.0 / Hnuclei + 1.0 *
                                    ab[IDX_HCOI] / 29.0 / Hnuclei + 4.0 *
                                    ab[IDX_oH2II] / 2.0 / Hnuclei + 1.0 *
                                    ab[IDX_oD2HII] / 5.0 / Hnuclei + 4.0 *
                                    ab[IDX_oH2DII] / 4.0 / Hnuclei + 1.0 *
                                    ab[IDX_pD2HII] / 5.0 / Hnuclei + 4.0 *
                                    ab[IDX_pH2DII] / 4.0 / Hnuclei + 4.0 *
                                    ab[IDX_pH2II] / 2.0 / Hnuclei + 4.0 *
                                    ab[IDX_H2OII] / 18.0 / Hnuclei + 1.0 *
                                    ab[IDX_HCOII] / 29.0 / Hnuclei + 1.0 *
                                    ab[IDX_CHII] / 13.0 / Hnuclei + 1.0 *
                                    ab[IDX_HDOII] / 19.0 / Hnuclei + 1.0 *
                                    ab[IDX_HII] / 1.0 / Hnuclei + 1.0 *
                                    ab[IDX_HDII] / 3.0 / Hnuclei + 1.0 *
                                    ab[IDX_NHI] / 15.0 / Hnuclei + 1.0 *
                                    ab[IDX_CHI] / 13.0 / Hnuclei + 4.0 *
                                    ab[IDX_H2OI] / 18.0 / Hnuclei + 1.0 *
                                    ab[IDX_HDOI] / 19.0 / Hnuclei + 1.0 *
                                    ab[IDX_OHI] / 17.0 / Hnuclei + 4.0 *
                                    ab[IDX_oH2I] / 2.0 / Hnuclei + 4.0 *
                                    ab[IDX_pH2I] / 2.0 / Hnuclei + 1.0 *
                                    ab[IDX_HDI] / 3.0 / Hnuclei + 1.0 *
                                    ab[IDX_HI] / 1.0 / Hnuclei;
        // clang-format on

    return NAUNET_SUCCESS;
}

// clang-format off
int RenormAbundance(realtype *rptr, realtype *ab) {
    
    ab[IDX_C2OII] = ab[IDX_C2OII] * (24.0 * rptr[IDX_ELEM_C] / 40.0 + 16.0 * rptr[IDX_ELEM_O] / 40.0);
    ab[IDX_NCOII] = ab[IDX_NCOII] * (12.0 * rptr[IDX_ELEM_C] / 42.0 + 14.0 * rptr[IDX_ELEM_N] / 42.0 + 16.0 * rptr[IDX_ELEM_O] / 42.0);
    ab[IDX_O2DI] = ab[IDX_O2DI] * (32.0 * rptr[IDX_ELEM_O] / 34.0 + 2.0 * rptr[IDX_ELEM_D] / 34.0);
    ab[IDX_O2HI] = ab[IDX_O2HI] * (32.0 * rptr[IDX_ELEM_O] / 33.0 + 1.0 * rptr[IDX_ELEM_H] / 33.0);
    ab[IDX_C2NI] = ab[IDX_C2NI] * (24.0 * rptr[IDX_ELEM_C] / 38.0 + 14.0 * rptr[IDX_ELEM_N] / 38.0);
    ab[IDX_CCOI] = ab[IDX_CCOI] * (24.0 * rptr[IDX_ELEM_C] / 40.0 + 16.0 * rptr[IDX_ELEM_O] / 40.0);
    ab[IDX_CNCII] = ab[IDX_CNCII] * (24.0 * rptr[IDX_ELEM_C] / 38.0 + 14.0 * rptr[IDX_ELEM_N] / 38.0);
    ab[IDX_N2OI] = ab[IDX_N2OI] * (28.0 * rptr[IDX_ELEM_N] / 44.0 + 16.0 * rptr[IDX_ELEM_O] / 44.0);
    ab[IDX_C3I] = ab[IDX_C3I] * (36.0 * rptr[IDX_ELEM_C] / 36.0);
    ab[IDX_HeHII] = ab[IDX_HeHII] * (4.0 * rptr[IDX_ELEM_He] / 5.0 + 1.0 * rptr[IDX_ELEM_H] / 5.0);
    ab[IDX_C3II] = ab[IDX_C3II] * (36.0 * rptr[IDX_ELEM_C] / 36.0);
    ab[IDX_HeDII] = ab[IDX_HeDII] * (4.0 * rptr[IDX_ELEM_He] / 6.0 + 2.0 * rptr[IDX_ELEM_D] / 6.0);
    ab[IDX_OCNI] = ab[IDX_OCNI] * (12.0 * rptr[IDX_ELEM_C] / 42.0 + 14.0 * rptr[IDX_ELEM_N] / 42.0 + 16.0 * rptr[IDX_ELEM_O] / 42.0);
    ab[IDX_DNCI] = ab[IDX_DNCI] * (12.0 * rptr[IDX_ELEM_C] / 28.0 + 14.0 * rptr[IDX_ELEM_N] / 28.0 + 2.0 * rptr[IDX_ELEM_D] / 28.0);
    ab[IDX_HNCI] = ab[IDX_HNCI] * (12.0 * rptr[IDX_ELEM_C] / 27.0 + 14.0 * rptr[IDX_ELEM_N] / 27.0 + 1.0 * rptr[IDX_ELEM_H] / 27.0);
    ab[IDX_HOCII] = ab[IDX_HOCII] * (12.0 * rptr[IDX_ELEM_C] / 29.0 + 16.0 * rptr[IDX_ELEM_O] / 29.0 + 1.0 * rptr[IDX_ELEM_H] / 29.0);
    ab[IDX_DOCII] = ab[IDX_DOCII] * (12.0 * rptr[IDX_ELEM_C] / 30.0 + 16.0 * rptr[IDX_ELEM_O] / 30.0 + 2.0 * rptr[IDX_ELEM_D] / 30.0);
    ab[IDX_ODM] = ab[IDX_ODM] * (16.0 * rptr[IDX_ELEM_O] / 18.0 + 2.0 * rptr[IDX_ELEM_D] / 18.0);
    ab[IDX_OHM] = ab[IDX_OHM] * (16.0 * rptr[IDX_ELEM_O] / 17.0 + 1.0 * rptr[IDX_ELEM_H] / 17.0);
    ab[IDX_CNM] = ab[IDX_CNM] * (12.0 * rptr[IDX_ELEM_C] / 26.0 + 14.0 * rptr[IDX_ELEM_N] / 26.0);
    ab[IDX_NO2II] = ab[IDX_NO2II] * (14.0 * rptr[IDX_ELEM_N] / 46.0 + 32.0 * rptr[IDX_ELEM_O] / 46.0);
    ab[IDX_NO2I] = ab[IDX_NO2I] * (14.0 * rptr[IDX_ELEM_N] / 46.0 + 32.0 * rptr[IDX_ELEM_O] / 46.0);
    ab[IDX_DNOI] = ab[IDX_DNOI] * (14.0 * rptr[IDX_ELEM_N] / 32.0 + 16.0 * rptr[IDX_ELEM_O] / 32.0 + 2.0 * rptr[IDX_ELEM_D] / 32.0);
    ab[IDX_HNOI] = ab[IDX_HNOI] * (14.0 * rptr[IDX_ELEM_N] / 31.0 + 16.0 * rptr[IDX_ELEM_O] / 31.0 + 1.0 * rptr[IDX_ELEM_H] / 31.0);
    ab[IDX_C2NII] = ab[IDX_C2NII] * (24.0 * rptr[IDX_ELEM_C] / 38.0 + 14.0 * rptr[IDX_ELEM_N] / 38.0);
    ab[IDX_GRAINM] = ab[IDX_GRAINM] * (0.0 * rptr[IDX_ELEM_GRAIN] / 0.0);
    ab[IDX_GRAIN0I] = ab[IDX_GRAIN0I] * (0.0 * rptr[IDX_ELEM_GRAIN] / 0.0);
    ab[IDX_DNCII] = ab[IDX_DNCII] * (12.0 * rptr[IDX_ELEM_C] / 28.0 + 14.0 * rptr[IDX_ELEM_N] / 28.0 + 2.0 * rptr[IDX_ELEM_D] / 28.0);
    ab[IDX_HNCII] = ab[IDX_HNCII] * (12.0 * rptr[IDX_ELEM_C] / 27.0 + 14.0 * rptr[IDX_ELEM_N] / 27.0 + 1.0 * rptr[IDX_ELEM_H] / 27.0);
    ab[IDX_OM] = ab[IDX_OM] * (16.0 * rptr[IDX_ELEM_O] / 16.0);
    ab[IDX_CO2II] = ab[IDX_CO2II] * (12.0 * rptr[IDX_ELEM_C] / 44.0 + 32.0 * rptr[IDX_ELEM_O] / 44.0);
    ab[IDX_CM] = ab[IDX_CM] * (12.0 * rptr[IDX_ELEM_C] / 12.0);
    ab[IDX_O2HII] = ab[IDX_O2HII] * (32.0 * rptr[IDX_ELEM_O] / 33.0 + 1.0 * rptr[IDX_ELEM_H] / 33.0);
    ab[IDX_O2DII] = ab[IDX_O2DII] * (32.0 * rptr[IDX_ELEM_O] / 34.0 + 2.0 * rptr[IDX_ELEM_D] / 34.0);
    ab[IDX_N2HII] = ab[IDX_N2HII] * (28.0 * rptr[IDX_ELEM_N] / 29.0 + 1.0 * rptr[IDX_ELEM_H] / 29.0);
    ab[IDX_N2DII] = ab[IDX_N2DII] * (28.0 * rptr[IDX_ELEM_N] / 30.0 + 2.0 * rptr[IDX_ELEM_D] / 30.0);
    ab[IDX_HNOII] = ab[IDX_HNOII] * (14.0 * rptr[IDX_ELEM_N] / 31.0 + 16.0 * rptr[IDX_ELEM_O] / 31.0 + 1.0 * rptr[IDX_ELEM_H] / 31.0);
    ab[IDX_DNOII] = ab[IDX_DNOII] * (14.0 * rptr[IDX_ELEM_N] / 32.0 + 16.0 * rptr[IDX_ELEM_O] / 32.0 + 2.0 * rptr[IDX_ELEM_D] / 32.0);
    ab[IDX_C2DI] = ab[IDX_C2DI] * (24.0 * rptr[IDX_ELEM_C] / 26.0 + 2.0 * rptr[IDX_ELEM_D] / 26.0);
    ab[IDX_C2HI] = ab[IDX_C2HI] * (24.0 * rptr[IDX_ELEM_C] / 25.0 + 1.0 * rptr[IDX_ELEM_H] / 25.0);
    ab[IDX_H3OII] = ab[IDX_H3OII] * (16.0 * rptr[IDX_ELEM_O] / 19.0 + 3.0 * rptr[IDX_ELEM_H] / 19.0);
    ab[IDX_pD3II] = ab[IDX_pD3II] * (6.0 * rptr[IDX_ELEM_D] / 6.0);
    ab[IDX_D3OII] = ab[IDX_D3OII] * (16.0 * rptr[IDX_ELEM_O] / 22.0 + 6.0 * rptr[IDX_ELEM_D] / 22.0);
    ab[IDX_HM] = ab[IDX_HM] * (1.0 * rptr[IDX_ELEM_H] / 1.0);
    ab[IDX_CO2I] = ab[IDX_CO2I] * (12.0 * rptr[IDX_ELEM_C] / 44.0 + 32.0 * rptr[IDX_ELEM_O] / 44.0);
    ab[IDX_mD3II] = ab[IDX_mD3II] * (6.0 * rptr[IDX_ELEM_D] / 6.0);
    ab[IDX_oD3II] = ab[IDX_oD3II] * (6.0 * rptr[IDX_ELEM_D] / 6.0);
    ab[IDX_DM] = ab[IDX_DM] * (2.0 * rptr[IDX_ELEM_D] / 2.0);
    ab[IDX_oH3II] = ab[IDX_oH3II] * (3.0 * rptr[IDX_ELEM_H] / 3.0);
    ab[IDX_pH3II] = ab[IDX_pH3II] * (3.0 * rptr[IDX_ELEM_H] / 3.0);
    ab[IDX_DCNI] = ab[IDX_DCNI] * (12.0 * rptr[IDX_ELEM_C] / 28.0 + 14.0 * rptr[IDX_ELEM_N] / 28.0 + 2.0 * rptr[IDX_ELEM_D] / 28.0);
    ab[IDX_HCNI] = ab[IDX_HCNI] * (12.0 * rptr[IDX_ELEM_C] / 27.0 + 14.0 * rptr[IDX_ELEM_N] / 27.0 + 1.0 * rptr[IDX_ELEM_H] / 27.0);
    ab[IDX_CD2I] = ab[IDX_CD2I] * (12.0 * rptr[IDX_ELEM_C] / 16.0 + 4.0 * rptr[IDX_ELEM_D] / 16.0);
    ab[IDX_CH2I] = ab[IDX_CH2I] * (12.0 * rptr[IDX_ELEM_C] / 14.0 + 2.0 * rptr[IDX_ELEM_H] / 14.0);
    ab[IDX_HCNII] = ab[IDX_HCNII] * (12.0 * rptr[IDX_ELEM_C] / 27.0 + 14.0 * rptr[IDX_ELEM_N] / 27.0 + 1.0 * rptr[IDX_ELEM_H] / 27.0);
    ab[IDX_ND2I] = ab[IDX_ND2I] * (14.0 * rptr[IDX_ELEM_N] / 18.0 + 4.0 * rptr[IDX_ELEM_D] / 18.0);
    ab[IDX_NH2I] = ab[IDX_NH2I] * (14.0 * rptr[IDX_ELEM_N] / 16.0 + 2.0 * rptr[IDX_ELEM_H] / 16.0);
    ab[IDX_DCNII] = ab[IDX_DCNII] * (12.0 * rptr[IDX_ELEM_C] / 28.0 + 14.0 * rptr[IDX_ELEM_N] / 28.0 + 2.0 * rptr[IDX_ELEM_D] / 28.0);
    ab[IDX_C2II] = ab[IDX_C2II] * (24.0 * rptr[IDX_ELEM_C] / 24.0);
    ab[IDX_C2HII] = ab[IDX_C2HII] * (24.0 * rptr[IDX_ELEM_C] / 25.0 + 1.0 * rptr[IDX_ELEM_H] / 25.0);
    ab[IDX_C2DII] = ab[IDX_C2DII] * (24.0 * rptr[IDX_ELEM_C] / 26.0 + 2.0 * rptr[IDX_ELEM_D] / 26.0);
    ab[IDX_CHDI] = ab[IDX_CHDI] * (12.0 * rptr[IDX_ELEM_C] / 15.0 + 2.0 * rptr[IDX_ELEM_D] / 15.0 + 1.0 * rptr[IDX_ELEM_H] / 15.0);
    ab[IDX_CH2II] = ab[IDX_CH2II] * (12.0 * rptr[IDX_ELEM_C] / 14.0 + 2.0 * rptr[IDX_ELEM_H] / 14.0);
    ab[IDX_NHDI] = ab[IDX_NHDI] * (14.0 * rptr[IDX_ELEM_N] / 17.0 + 2.0 * rptr[IDX_ELEM_D] / 17.0 + 1.0 * rptr[IDX_ELEM_H] / 17.0);
    ab[IDX_CD2II] = ab[IDX_CD2II] * (12.0 * rptr[IDX_ELEM_C] / 16.0 + 4.0 * rptr[IDX_ELEM_D] / 16.0);
    ab[IDX_H2DOII] = ab[IDX_H2DOII] * (16.0 * rptr[IDX_ELEM_O] / 20.0 + 2.0 * rptr[IDX_ELEM_D] / 20.0 + 2.0 * rptr[IDX_ELEM_H] / 20.0);
    ab[IDX_HD2OII] = ab[IDX_HD2OII] * (16.0 * rptr[IDX_ELEM_O] / 21.0 + 4.0 * rptr[IDX_ELEM_D] / 21.0 + 1.0 * rptr[IDX_ELEM_H] / 21.0);
    ab[IDX_NHII] = ab[IDX_NHII] * (14.0 * rptr[IDX_ELEM_N] / 15.0 + 1.0 * rptr[IDX_ELEM_H] / 15.0);
    ab[IDX_NH2II] = ab[IDX_NH2II] * (14.0 * rptr[IDX_ELEM_N] / 16.0 + 2.0 * rptr[IDX_ELEM_H] / 16.0);
    ab[IDX_CII] = ab[IDX_CII] * (12.0 * rptr[IDX_ELEM_C] / 12.0);
    ab[IDX_NII] = ab[IDX_NII] * (14.0 * rptr[IDX_ELEM_N] / 14.0);
    ab[IDX_N2II] = ab[IDX_N2II] * (28.0 * rptr[IDX_ELEM_N] / 28.0);
    ab[IDX_NDII] = ab[IDX_NDII] * (14.0 * rptr[IDX_ELEM_N] / 16.0 + 2.0 * rptr[IDX_ELEM_D] / 16.0);
    ab[IDX_ND2II] = ab[IDX_ND2II] * (14.0 * rptr[IDX_ELEM_N] / 18.0 + 4.0 * rptr[IDX_ELEM_D] / 18.0);
    ab[IDX_O2II] = ab[IDX_O2II] * (32.0 * rptr[IDX_ELEM_O] / 32.0);
    ab[IDX_COII] = ab[IDX_COII] * (12.0 * rptr[IDX_ELEM_C] / 28.0 + 16.0 * rptr[IDX_ELEM_O] / 28.0);
    ab[IDX_CNII] = ab[IDX_CNII] * (12.0 * rptr[IDX_ELEM_C] / 26.0 + 14.0 * rptr[IDX_ELEM_N] / 26.0);
    ab[IDX_HeII] = ab[IDX_HeII] * (4.0 * rptr[IDX_ELEM_He] / 4.0);
    ab[IDX_NHDII] = ab[IDX_NHDII] * (14.0 * rptr[IDX_ELEM_N] / 17.0 + 2.0 * rptr[IDX_ELEM_D] / 17.0 + 1.0 * rptr[IDX_ELEM_H] / 17.0);
    ab[IDX_OII] = ab[IDX_OII] * (16.0 * rptr[IDX_ELEM_O] / 16.0);
    ab[IDX_OHII] = ab[IDX_OHII] * (16.0 * rptr[IDX_ELEM_O] / 17.0 + 1.0 * rptr[IDX_ELEM_H] / 17.0);
    ab[IDX_ODII] = ab[IDX_ODII] * (16.0 * rptr[IDX_ELEM_O] / 18.0 + 2.0 * rptr[IDX_ELEM_D] / 18.0);
    ab[IDX_CHDII] = ab[IDX_CHDII] * (12.0 * rptr[IDX_ELEM_C] / 15.0 + 2.0 * rptr[IDX_ELEM_D] / 15.0 + 1.0 * rptr[IDX_ELEM_H] / 15.0);
    ab[IDX_HCOI] = ab[IDX_HCOI] * (12.0 * rptr[IDX_ELEM_C] / 29.0 + 16.0 * rptr[IDX_ELEM_O] / 29.0 + 1.0 * rptr[IDX_ELEM_H] / 29.0);
    ab[IDX_oH2II] = ab[IDX_oH2II] * (2.0 * rptr[IDX_ELEM_H] / 2.0);
    ab[IDX_DCOI] = ab[IDX_DCOI] * (12.0 * rptr[IDX_ELEM_C] / 30.0 + 16.0 * rptr[IDX_ELEM_O] / 30.0 + 2.0 * rptr[IDX_ELEM_D] / 30.0);
    ab[IDX_oD2HII] = ab[IDX_oD2HII] * (4.0 * rptr[IDX_ELEM_D] / 5.0 + 1.0 * rptr[IDX_ELEM_H] / 5.0);
    ab[IDX_oH2DII] = ab[IDX_oH2DII] * (2.0 * rptr[IDX_ELEM_D] / 4.0 + 2.0 * rptr[IDX_ELEM_H] / 4.0);
    ab[IDX_pD2HII] = ab[IDX_pD2HII] * (4.0 * rptr[IDX_ELEM_D] / 5.0 + 1.0 * rptr[IDX_ELEM_H] / 5.0);
    ab[IDX_pH2DII] = ab[IDX_pH2DII] * (2.0 * rptr[IDX_ELEM_D] / 4.0 + 2.0 * rptr[IDX_ELEM_H] / 4.0);
    ab[IDX_pH2II] = ab[IDX_pH2II] * (2.0 * rptr[IDX_ELEM_H] / 2.0);
    ab[IDX_H2OII] = ab[IDX_H2OII] * (16.0 * rptr[IDX_ELEM_O] / 18.0 + 2.0 * rptr[IDX_ELEM_H] / 18.0);
    ab[IDX_HCOII] = ab[IDX_HCOII] * (12.0 * rptr[IDX_ELEM_C] / 29.0 + 16.0 * rptr[IDX_ELEM_O] / 29.0 + 1.0 * rptr[IDX_ELEM_H] / 29.0);
    ab[IDX_D2OII] = ab[IDX_D2OII] * (16.0 * rptr[IDX_ELEM_O] / 20.0 + 4.0 * rptr[IDX_ELEM_D] / 20.0);
    ab[IDX_DCOII] = ab[IDX_DCOII] * (12.0 * rptr[IDX_ELEM_C] / 30.0 + 16.0 * rptr[IDX_ELEM_O] / 30.0 + 2.0 * rptr[IDX_ELEM_D] / 30.0);
    ab[IDX_oD2II] = ab[IDX_oD2II] * (4.0 * rptr[IDX_ELEM_D] / 4.0);
    ab[IDX_pD2II] = ab[IDX_pD2II] * (4.0 * rptr[IDX_ELEM_D] / 4.0);
    ab[IDX_CHII] = ab[IDX_CHII] * (12.0 * rptr[IDX_ELEM_C] / 13.0 + 1.0 * rptr[IDX_ELEM_H] / 13.0);
    ab[IDX_HeI] = ab[IDX_HeI] * (4.0 * rptr[IDX_ELEM_He] / 4.0);
    ab[IDX_CDII] = ab[IDX_CDII] * (12.0 * rptr[IDX_ELEM_C] / 14.0 + 2.0 * rptr[IDX_ELEM_D] / 14.0);
    ab[IDX_HDOII] = ab[IDX_HDOII] * (16.0 * rptr[IDX_ELEM_O] / 19.0 + 2.0 * rptr[IDX_ELEM_D] / 19.0 + 1.0 * rptr[IDX_ELEM_H] / 19.0);
    ab[IDX_HII] = ab[IDX_HII] * (1.0 * rptr[IDX_ELEM_H] / 1.0);
    ab[IDX_NOII] = ab[IDX_NOII] * (14.0 * rptr[IDX_ELEM_N] / 30.0 + 16.0 * rptr[IDX_ELEM_O] / 30.0);
    ab[IDX_DII] = ab[IDX_DII] * (2.0 * rptr[IDX_ELEM_D] / 2.0);
    ab[IDX_HDII] = ab[IDX_HDII] * (2.0 * rptr[IDX_ELEM_D] / 3.0 + 1.0 * rptr[IDX_ELEM_H] / 3.0);
    ab[IDX_NHI] = ab[IDX_NHI] * (14.0 * rptr[IDX_ELEM_N] / 15.0 + 1.0 * rptr[IDX_ELEM_H] / 15.0);
    ab[IDX_N2I] = ab[IDX_N2I] * (28.0 * rptr[IDX_ELEM_N] / 28.0);
    ab[IDX_NDI] = ab[IDX_NDI] * (14.0 * rptr[IDX_ELEM_N] / 16.0 + 2.0 * rptr[IDX_ELEM_D] / 16.0);
    ab[IDX_C2I] = ab[IDX_C2I] * (24.0 * rptr[IDX_ELEM_C] / 24.0);
    ab[IDX_CHI] = ab[IDX_CHI] * (12.0 * rptr[IDX_ELEM_C] / 13.0 + 1.0 * rptr[IDX_ELEM_H] / 13.0);
    ab[IDX_CDI] = ab[IDX_CDI] * (12.0 * rptr[IDX_ELEM_C] / 14.0 + 2.0 * rptr[IDX_ELEM_D] / 14.0);
    ab[IDX_O2I] = ab[IDX_O2I] * (32.0 * rptr[IDX_ELEM_O] / 32.0);
    ab[IDX_D2OI] = ab[IDX_D2OI] * (16.0 * rptr[IDX_ELEM_O] / 20.0 + 4.0 * rptr[IDX_ELEM_D] / 20.0);
    ab[IDX_H2OI] = ab[IDX_H2OI] * (16.0 * rptr[IDX_ELEM_O] / 18.0 + 2.0 * rptr[IDX_ELEM_H] / 18.0);
    ab[IDX_NOI] = ab[IDX_NOI] * (14.0 * rptr[IDX_ELEM_N] / 30.0 + 16.0 * rptr[IDX_ELEM_O] / 30.0);
    ab[IDX_HDOI] = ab[IDX_HDOI] * (16.0 * rptr[IDX_ELEM_O] / 19.0 + 2.0 * rptr[IDX_ELEM_D] / 19.0 + 1.0 * rptr[IDX_ELEM_H] / 19.0);
    ab[IDX_OHI] = ab[IDX_OHI] * (16.0 * rptr[IDX_ELEM_O] / 17.0 + 1.0 * rptr[IDX_ELEM_H] / 17.0);
    ab[IDX_oD2I] = ab[IDX_oD2I] * (4.0 * rptr[IDX_ELEM_D] / 4.0);
    ab[IDX_oH2I] = ab[IDX_oH2I] * (2.0 * rptr[IDX_ELEM_H] / 2.0);
    ab[IDX_CNI] = ab[IDX_CNI] * (12.0 * rptr[IDX_ELEM_C] / 26.0 + 14.0 * rptr[IDX_ELEM_N] / 26.0);
    ab[IDX_ODI] = ab[IDX_ODI] * (16.0 * rptr[IDX_ELEM_O] / 18.0 + 2.0 * rptr[IDX_ELEM_D] / 18.0);
    ab[IDX_CI] = ab[IDX_CI] * (12.0 * rptr[IDX_ELEM_C] / 12.0);
    ab[IDX_COI] = ab[IDX_COI] * (12.0 * rptr[IDX_ELEM_C] / 28.0 + 16.0 * rptr[IDX_ELEM_O] / 28.0);
    ab[IDX_NI] = ab[IDX_NI] * (14.0 * rptr[IDX_ELEM_N] / 14.0);
    ab[IDX_OI] = ab[IDX_OI] * (16.0 * rptr[IDX_ELEM_O] / 16.0);
    ab[IDX_pD2I] = ab[IDX_pD2I] * (4.0 * rptr[IDX_ELEM_D] / 4.0);
    ab[IDX_pH2I] = ab[IDX_pH2I] * (2.0 * rptr[IDX_ELEM_H] / 2.0);
    ab[IDX_HDI] = ab[IDX_HDI] * (2.0 * rptr[IDX_ELEM_D] / 3.0 + 1.0 * rptr[IDX_ELEM_H] / 3.0);
    ab[IDX_eM] = ab[IDX_eM] * (1.0);
    ab[IDX_DI] = ab[IDX_DI] * (2.0 * rptr[IDX_ELEM_D] / 2.0);
    ab[IDX_HI] = ab[IDX_HI] * (1.0 * rptr[IDX_ELEM_H] / 1.0);
        // clang-format on

    return NAUNET_SUCCESS;
}