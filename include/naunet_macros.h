#ifndef __NAUNET_MACROS_H__
#define __NAUNET_MACROS_H__

#include <sunmatrix/sunmatrix_dense.h>

// 
// clang-format off
#define NAUNET_SUCCESS 0
#define NAUNET_FAIL 1

#define MAX_NSYSTEMS 1

#define NELEMENTS 7
#define NSPECIES 131
#define NHEATPROCS 0
#define NCOOLPROCS 0
#define THERMAL (NHEATPROCS || NCOOLPROCS)
#if (NSPECIES + THERMAL)
#define NEQUATIONS (NSPECIES + THERMAL)
#else
#define NEQUATIONS 1
#endif
#define NREACTIONS 3466
// non-zero terms in jacobian matrix, used in sparse matrix
#define NNZ 6769

#define IDX_ELEM_GRAIN 0
#define IDX_ELEM_He 1
#define IDX_ELEM_C 2
#define IDX_ELEM_N 3
#define IDX_ELEM_O 4
#define IDX_ELEM_D 5
#define IDX_ELEM_H 6

#define IDX_C2OII 0
#define IDX_NCOII 1
#define IDX_O2DI 2
#define IDX_O2HI 3
#define IDX_C2NI 4
#define IDX_CCOI 5
#define IDX_CNCII 6
#define IDX_N2OI 7
#define IDX_C3I 8
#define IDX_HeHII 9
#define IDX_C3II 10
#define IDX_HeDII 11
#define IDX_OCNI 12
#define IDX_DNCI 13
#define IDX_HNCI 14
#define IDX_HOCII 15
#define IDX_DOCII 16
#define IDX_ODM 17
#define IDX_OHM 18
#define IDX_CNM 19
#define IDX_NO2II 20
#define IDX_NO2I 21
#define IDX_DNOI 22
#define IDX_HNOI 23
#define IDX_C2NII 24
#define IDX_GRAINM 25
#define IDX_GRAIN0I 26
#define IDX_DNCII 27
#define IDX_HNCII 28
#define IDX_OM 29
#define IDX_CO2II 30
#define IDX_CM 31
#define IDX_O2HII 32
#define IDX_O2DII 33
#define IDX_N2HII 34
#define IDX_N2DII 35
#define IDX_HNOII 36
#define IDX_DNOII 37
#define IDX_C2DI 38
#define IDX_C2HI 39
#define IDX_H3OII 40
#define IDX_pD3II 41
#define IDX_D3OII 42
#define IDX_HM 43
#define IDX_CO2I 44
#define IDX_mD3II 45
#define IDX_oD3II 46
#define IDX_DM 47
#define IDX_oH3II 48
#define IDX_pH3II 49
#define IDX_DCNI 50
#define IDX_HCNI 51
#define IDX_CD2I 52
#define IDX_CH2I 53
#define IDX_HCNII 54
#define IDX_ND2I 55
#define IDX_NH2I 56
#define IDX_DCNII 57
#define IDX_C2II 58
#define IDX_C2HII 59
#define IDX_C2DII 60
#define IDX_CHDI 61
#define IDX_CH2II 62
#define IDX_NHDI 63
#define IDX_CD2II 64
#define IDX_H2DOII 65
#define IDX_HD2OII 66
#define IDX_NHII 67
#define IDX_NH2II 68
#define IDX_CII 69
#define IDX_NII 70
#define IDX_N2II 71
#define IDX_NDII 72
#define IDX_ND2II 73
#define IDX_O2II 74
#define IDX_COII 75
#define IDX_CNII 76
#define IDX_HeII 77
#define IDX_NHDII 78
#define IDX_OII 79
#define IDX_OHII 80
#define IDX_ODII 81
#define IDX_CHDII 82
#define IDX_HCOI 83
#define IDX_oH2II 84
#define IDX_DCOI 85
#define IDX_oD2HII 86
#define IDX_oH2DII 87
#define IDX_pD2HII 88
#define IDX_pH2DII 89
#define IDX_pH2II 90
#define IDX_H2OII 91
#define IDX_HCOII 92
#define IDX_D2OII 93
#define IDX_DCOII 94
#define IDX_oD2II 95
#define IDX_pD2II 96
#define IDX_CHII 97
#define IDX_HeI 98
#define IDX_CDII 99
#define IDX_HDOII 100
#define IDX_HII 101
#define IDX_NOII 102
#define IDX_DII 103
#define IDX_HDII 104
#define IDX_NHI 105
#define IDX_N2I 106
#define IDX_NDI 107
#define IDX_C2I 108
#define IDX_CHI 109
#define IDX_CDI 110
#define IDX_O2I 111
#define IDX_D2OI 112
#define IDX_H2OI 113
#define IDX_NOI 114
#define IDX_HDOI 115
#define IDX_OHI 116
#define IDX_oD2I 117
#define IDX_oH2I 118
#define IDX_CNI 119
#define IDX_ODI 120
#define IDX_CI 121
#define IDX_COI 122
#define IDX_NI 123
#define IDX_OI 124
#define IDX_pD2I 125
#define IDX_pH2I 126
#define IDX_HDI 127
#define IDX_eM 128
#define IDX_DI 129
#define IDX_HI 130

#if THERMAL
#define IDX_TGAS NSPECIES
#endif
#define IJth(A, i, j) SM_ELEMENT_D(A, i, j)

#endif