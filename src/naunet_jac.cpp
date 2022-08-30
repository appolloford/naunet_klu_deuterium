#include <math.h>
/* */
#include <nvector/nvector_serial.h>
#include <sunmatrix/sunmatrix_sparse.h>  // access to sparse SUNMatrix
/* */
/*  */
#include "naunet_ode.h"
/*  */
#include "naunet_constants.h"
#include "naunet_macros.h"
#include "naunet_physics.h"

#define IJth(A, i, j)            SM_ELEMENT_D(A, i, j)
#define NVEC_CUDA_CONTENT(x)     ((N_VectorContent_Cuda)(x->content))
#define NVEC_CUDA_STREAM(x)      (NVEC_CUDA_CONTENT(x)->stream_exec_policy->stream())
#define NVEC_CUDA_BLOCKSIZE(x)   (NVEC_CUDA_CONTENT(x)->stream_exec_policy->blockSize())
#define NVEC_CUDA_GRIDSIZE(x, n) (NVEC_CUDA_CONTENT(x)->stream_exec_policy->gridSize(n))

/* */

int Jac(realtype t, N_Vector u, N_Vector fu, SUNMatrix jmatrix, void *user_data,
        N_Vector tmp1, N_Vector tmp2, N_Vector tmp3) {
    /* */
    realtype *y            = N_VGetArrayPointer(u);
    sunindextype *rowptrs  = SUNSparseMatrix_IndexPointers(jmatrix);
    sunindextype *colvals  = SUNSparseMatrix_IndexValues(jmatrix);
    realtype *data         = SUNSparseMatrix_Data(jmatrix);
    NaunetData *u_data     = (NaunetData *)user_data;

    // clang-format off
        
    // clang-format on

    realtype k[NREACTIONS] = {0.0};
    EvalRates(k, y, u_data);

    // clang-format off
    // number of non-zero elements in each row
    rowptrs[0] = 0;
    rowptrs[1] = 109;
    rowptrs[2] = 166;
    rowptrs[3] = 197;
    rowptrs[4] = 276;
    rowptrs[5] = 330;
    rowptrs[6] = 364;
    rowptrs[7] = 423;
    rowptrs[8] = 457;
    rowptrs[9] = 515;
    rowptrs[10] = 524;
    rowptrs[11] = 549;
    rowptrs[12] = 557;
    rowptrs[13] = 569;
    rowptrs[14] = 586;
    rowptrs[15] = 596;
    rowptrs[16] = 683;
    rowptrs[17] = 762;
    rowptrs[18] = 805;
    rowptrs[19] = 862;
    rowptrs[20] = 948;
    rowptrs[21] = 1026;
    rowptrs[22] = 1069;
    rowptrs[23] = 1125;
    rowptrs[24] = 1173;
    rowptrs[25] = 1241;
    rowptrs[26] = 1331;
    rowptrs[27] = 1384;
    rowptrs[28] = 1401;
    rowptrs[29] = 1412;
    rowptrs[30] = 1506;
    rowptrs[31] = 1562;
    rowptrs[32] = 1599;
    rowptrs[33] = 1620;
    rowptrs[34] = 1732;
    rowptrs[35] = 1801;
    rowptrs[36] = 1851;
    rowptrs[37] = 1939;
    rowptrs[38] = 2008;
    rowptrs[39] = 2057;
    rowptrs[40] = 2106;
    rowptrs[41] = 2161;
    rowptrs[42] = 2207;
    rowptrs[43] = 2288;
    rowptrs[44] = 2301;
    rowptrs[45] = 2320;
    rowptrs[46] = 2337;
    rowptrs[47] = 2376;
    rowptrs[48] = 2396;
    rowptrs[49] = 2419;
    rowptrs[50] = 2442;
    rowptrs[51] = 2553;
    rowptrs[52] = 2621;
    rowptrs[53] = 2668;
    rowptrs[54] = 2735;
    rowptrs[55] = 2823;
    rowptrs[56] = 2891;
    rowptrs[57] = 2939;
    rowptrs[58] = 2988;
    rowptrs[59] = 3042;
    rowptrs[60] = 3088;
    rowptrs[61] = 3168;
    rowptrs[62] = 3264;
    rowptrs[63] = 3319;
    rowptrs[64] = 3386;
    rowptrs[65] = 3480;
    rowptrs[66] = 3554;
    rowptrs[67] = 3567;
    rowptrs[68] = 3586;
    rowptrs[69] = 3603;
    rowptrs[70] = 3641;
    rowptrs[71] = 3660;
    rowptrs[72] = 3713;
    rowptrs[73] = 3764;
    rowptrs[74] = 3775;
    rowptrs[75] = 3786;
    rowptrs[76] = 3886;
    rowptrs[77] = 3929;
    rowptrs[78] = 3992;
    rowptrs[79] = 4037;
    rowptrs[80] = 4075;
    rowptrs[81] = 4112;
    rowptrs[82] = 4122;
    rowptrs[83] = 4131;
    rowptrs[84] = 4214;
    rowptrs[85] = 4257;
    rowptrs[86] = 4302;
    rowptrs[87] = 4359;
    rowptrs[88] = 4441;
    rowptrs[89] = 4484;
    rowptrs[90] = 4529;
    rowptrs[91] = 4585;
    rowptrs[92] = 4635;
    rowptrs[93] = 4699;
    rowptrs[94] = 4782;
    rowptrs[95] = 4859;
    rowptrs[96] = 4875;
    rowptrs[97] = 4895;
    rowptrs[98] = 5004;
    rowptrs[99] = 5061;
    rowptrs[100] = 5085;
    rowptrs[101] = 5172;
    rowptrs[102] = 5219;
    rowptrs[103] = 5226;
    rowptrs[104] = 5261;
    rowptrs[105] = 5268;
    rowptrs[106] = 5302;
    rowptrs[107] = 5316;
    rowptrs[108] = 5417;
    rowptrs[109] = 5474;
    rowptrs[110] = 5494;
    rowptrs[111] = 5594;
    rowptrs[112] = 5650;
    rowptrs[113] = 5670;
    rowptrs[114] = 5787;
    rowptrs[115] = 5830;
    rowptrs[116] = 5919;
    rowptrs[117] = 5970;
    rowptrs[118] = 6021;
    rowptrs[119] = 6064;
    rowptrs[120] = 6153;
    rowptrs[121] = 6201;
    rowptrs[122] = 6253;
    rowptrs[123] = 6296;
    rowptrs[124] = 6389;
    rowptrs[125] = 6441;
    rowptrs[126] = 6493;
    rowptrs[127] = 6531;
    rowptrs[128] = 6624;
    rowptrs[129] = 6674;
    rowptrs[130] = 6726;
    rowptrs[131] = 6769;
    
    // the column index of non-zero elements
    colvals[0] = 0;
    colvals[1] = 1;
    colvals[2] = 2;
    colvals[3] = 3;
    colvals[4] = 4;
    colvals[5] = 5;
    colvals[6] = 6;
    colvals[7] = 7;
    colvals[8] = 8;
    colvals[9] = 9;
    colvals[10] = 10;
    colvals[11] = 11;
    colvals[12] = 12;
    colvals[13] = 13;
    colvals[14] = 14;
    colvals[15] = 15;
    colvals[16] = 16;
    colvals[17] = 17;
    colvals[18] = 18;
    colvals[19] = 19;
    colvals[20] = 20;
    colvals[21] = 21;
    colvals[22] = 22;
    colvals[23] = 23;
    colvals[24] = 24;
    colvals[25] = 25;
    colvals[26] = 26;
    colvals[27] = 28;
    colvals[28] = 29;
    colvals[29] = 30;
    colvals[30] = 31;
    colvals[31] = 33;
    colvals[32] = 34;
    colvals[33] = 35;
    colvals[34] = 36;
    colvals[35] = 37;
    colvals[36] = 38;
    colvals[37] = 40;
    colvals[38] = 41;
    colvals[39] = 42;
    colvals[40] = 43;
    colvals[41] = 44;
    colvals[42] = 46;
    colvals[43] = 48;
    colvals[44] = 50;
    colvals[45] = 51;
    colvals[46] = 52;
    colvals[47] = 53;
    colvals[48] = 54;
    colvals[49] = 55;
    colvals[50] = 56;
    colvals[51] = 58;
    colvals[52] = 59;
    colvals[53] = 60;
    colvals[54] = 61;
    colvals[55] = 62;
    colvals[56] = 63;
    colvals[57] = 64;
    colvals[58] = 65;
    colvals[59] = 66;
    colvals[60] = 67;
    colvals[61] = 69;
    colvals[62] = 72;
    colvals[63] = 75;
    colvals[64] = 76;
    colvals[65] = 77;
    colvals[66] = 78;
    colvals[67] = 79;
    colvals[68] = 80;
    colvals[69] = 83;
    colvals[70] = 84;
    colvals[71] = 85;
    colvals[72] = 87;
    colvals[73] = 88;
    colvals[74] = 89;
    colvals[75] = 91;
    colvals[76] = 93;
    colvals[77] = 97;
    colvals[78] = 98;
    colvals[79] = 99;
    colvals[80] = 100;
    colvals[81] = 101;
    colvals[82] = 103;
    colvals[83] = 105;
    colvals[84] = 106;
    colvals[85] = 107;
    colvals[86] = 108;
    colvals[87] = 109;
    colvals[88] = 110;
    colvals[89] = 111;
    colvals[90] = 112;
    colvals[91] = 113;
    colvals[92] = 114;
    colvals[93] = 115;
    colvals[94] = 116;
    colvals[95] = 117;
    colvals[96] = 118;
    colvals[97] = 119;
    colvals[98] = 120;
    colvals[99] = 121;
    colvals[100] = 122;
    colvals[101] = 123;
    colvals[102] = 124;
    colvals[103] = 125;
    colvals[104] = 126;
    colvals[105] = 127;
    colvals[106] = 128;
    colvals[107] = 129;
    colvals[108] = 130;
    colvals[109] = 0;
    colvals[110] = 1;
    colvals[111] = 2;
    colvals[112] = 3;
    colvals[113] = 4;
    colvals[114] = 5;
    colvals[115] = 7;
    colvals[116] = 9;
    colvals[117] = 12;
    colvals[118] = 14;
    colvals[119] = 15;
    colvals[120] = 16;
    colvals[121] = 17;
    colvals[122] = 19;
    colvals[123] = 20;
    colvals[124] = 21;
    colvals[125] = 23;
    colvals[126] = 25;
    colvals[127] = 26;
    colvals[128] = 29;
    colvals[129] = 30;
    colvals[130] = 31;
    colvals[131] = 33;
    colvals[132] = 35;
    colvals[133] = 36;
    colvals[134] = 39;
    colvals[135] = 41;
    colvals[136] = 43;
    colvals[137] = 48;
    colvals[138] = 50;
    colvals[139] = 52;
    colvals[140] = 54;
    colvals[141] = 57;
    colvals[142] = 59;
    colvals[143] = 61;
    colvals[144] = 64;
    colvals[145] = 66;
    colvals[146] = 72;
    colvals[147] = 75;
    colvals[148] = 78;
    colvals[149] = 83;
    colvals[150] = 85;
    colvals[151] = 87;
    colvals[152] = 89;
    colvals[153] = 91;
    colvals[154] = 93;
    colvals[155] = 97;
    colvals[156] = 100;
    colvals[157] = 101;
    colvals[158] = 106;
    colvals[159] = 107;
    colvals[160] = 110;
    colvals[161] = 113;
    colvals[162] = 115;
    colvals[163] = 119;
    colvals[164] = 123;
    colvals[165] = 127;
    colvals[166] = 0;
    colvals[167] = 1;
    colvals[168] = 2;
    colvals[169] = 15;
    colvals[170] = 19;
    colvals[171] = 31;
    colvals[172] = 33;
    colvals[173] = 34;
    colvals[174] = 38;
    colvals[175] = 50;
    colvals[176] = 51;
    colvals[177] = 53;
    colvals[178] = 56;
    colvals[179] = 61;
    colvals[180] = 63;
    colvals[181] = 72;
    colvals[182] = 75;
    colvals[183] = 76;
    colvals[184] = 83;
    colvals[185] = 87;
    colvals[186] = 93;
    colvals[187] = 97;
    colvals[188] = 98;
    colvals[189] = 100;
    colvals[190] = 107;
    colvals[191] = 110;
    colvals[192] = 113;
    colvals[193] = 115;
    colvals[194] = 119;
    colvals[195] = 123;
    colvals[196] = 127;
    colvals[197] = 0;
    colvals[198] = 2;
    colvals[199] = 3;
    colvals[200] = 4;
    colvals[201] = 5;
    colvals[202] = 6;
    colvals[203] = 7;
    colvals[204] = 8;
    colvals[205] = 9;
    colvals[206] = 10;
    colvals[207] = 12;
    colvals[208] = 13;
    colvals[209] = 14;
    colvals[210] = 15;
    colvals[211] = 16;
    colvals[212] = 17;
    colvals[213] = 19;
    colvals[214] = 20;
    colvals[215] = 21;
    colvals[216] = 23;
    colvals[217] = 25;
    colvals[218] = 26;
    colvals[219] = 28;
    colvals[220] = 29;
    colvals[221] = 30;
    colvals[222] = 34;
    colvals[223] = 35;
    colvals[224] = 37;
    colvals[225] = 40;
    colvals[226] = 41;
    colvals[227] = 42;
    colvals[228] = 44;
    colvals[229] = 46;
    colvals[230] = 51;
    colvals[231] = 52;
    colvals[232] = 55;
    colvals[233] = 58;
    colvals[234] = 59;
    colvals[235] = 60;
    colvals[236] = 62;
    colvals[237] = 65;
    colvals[238] = 67;
    colvals[239] = 69;
    colvals[240] = 72;
    colvals[241] = 75;
    colvals[242] = 76;
    colvals[243] = 78;
    colvals[244] = 79;
    colvals[245] = 80;
    colvals[246] = 84;
    colvals[247] = 85;
    colvals[248] = 86;
    colvals[249] = 88;
    colvals[250] = 89;
    colvals[251] = 90;
    colvals[252] = 91;
    colvals[253] = 92;
    colvals[254] = 93;
    colvals[255] = 97;
    colvals[256] = 98;
    colvals[257] = 101;
    colvals[258] = 103;
    colvals[259] = 105;
    colvals[260] = 108;
    colvals[261] = 111;
    colvals[262] = 113;
    colvals[263] = 114;
    colvals[264] = 116;
    colvals[265] = 117;
    colvals[266] = 118;
    colvals[267] = 120;
    colvals[268] = 121;
    colvals[269] = 122;
    colvals[270] = 124;
    colvals[271] = 125;
    colvals[272] = 126;
    colvals[273] = 128;
    colvals[274] = 129;
    colvals[275] = 130;
    colvals[276] = 0;
    colvals[277] = 1;
    colvals[278] = 3;
    colvals[279] = 4;
    colvals[280] = 5;
    colvals[281] = 6;
    colvals[282] = 7;
    colvals[283] = 8;
    colvals[284] = 12;
    colvals[285] = 15;
    colvals[286] = 16;
    colvals[287] = 17;
    colvals[288] = 19;
    colvals[289] = 20;
    colvals[290] = 21;
    colvals[291] = 23;
    colvals[292] = 26;
    colvals[293] = 30;
    colvals[294] = 34;
    colvals[295] = 36;
    colvals[296] = 37;
    colvals[297] = 41;
    colvals[298] = 51;
    colvals[299] = 54;
    colvals[300] = 55;
    colvals[301] = 59;
    colvals[302] = 61;
    colvals[303] = 62;
    colvals[304] = 64;
    colvals[305] = 65;
    colvals[306] = 72;
    colvals[307] = 75;
    colvals[308] = 76;
    colvals[309] = 78;
    colvals[310] = 83;
    colvals[311] = 85;
    colvals[312] = 87;
    colvals[313] = 89;
    colvals[314] = 91;
    colvals[315] = 93;
    colvals[316] = 97;
    colvals[317] = 98;
    colvals[318] = 100;
    colvals[319] = 108;
    colvals[320] = 111;
    colvals[321] = 113;
    colvals[322] = 115;
    colvals[323] = 116;
    colvals[324] = 119;
    colvals[325] = 120;
    colvals[326] = 123;
    colvals[327] = 124;
    colvals[328] = 127;
    colvals[329] = 128;
    colvals[330] = 0;
    colvals[331] = 1;
    colvals[332] = 2;
    colvals[333] = 3;
    colvals[334] = 5;
    colvals[335] = 6;
    colvals[336] = 15;
    colvals[337] = 16;
    colvals[338] = 17;
    colvals[339] = 20;
    colvals[340] = 23;
    colvals[341] = 26;
    colvals[342] = 30;
    colvals[343] = 34;
    colvals[344] = 35;
    colvals[345] = 37;
    colvals[346] = 51;
    colvals[347] = 55;
    colvals[348] = 62;
    colvals[349] = 65;
    colvals[350] = 72;
    colvals[351] = 75;
    colvals[352] = 76;
    colvals[353] = 78;
    colvals[354] = 93;
    colvals[355] = 97;
    colvals[356] = 98;
    colvals[357] = 100;
    colvals[358] = 108;
    colvals[359] = 111;
    colvals[360] = 116;
    colvals[361] = 120;
    colvals[362] = 124;
    colvals[363] = 128;
    colvals[364] = 0;
    colvals[365] = 1;
    colvals[366] = 3;
    colvals[367] = 4;
    colvals[368] = 5;
    colvals[369] = 6;
    colvals[370] = 15;
    colvals[371] = 16;
    colvals[372] = 17;
    colvals[373] = 18;
    colvals[374] = 19;
    colvals[375] = 20;
    colvals[376] = 21;
    colvals[377] = 23;
    colvals[378] = 24;
    colvals[379] = 26;
    colvals[380] = 30;
    colvals[381] = 34;
    colvals[382] = 36;
    colvals[383] = 37;
    colvals[384] = 40;
    colvals[385] = 41;
    colvals[386] = 42;
    colvals[387] = 44;
    colvals[388] = 46;
    colvals[389] = 51;
    colvals[390] = 55;
    colvals[391] = 61;
    colvals[392] = 62;
    colvals[393] = 64;
    colvals[394] = 65;
    colvals[395] = 75;
    colvals[396] = 76;
    colvals[397] = 78;
    colvals[398] = 79;
    colvals[399] = 83;
    colvals[400] = 84;
    colvals[401] = 86;
    colvals[402] = 92;
    colvals[403] = 93;
    colvals[404] = 97;
    colvals[405] = 98;
    colvals[406] = 103;
    colvals[407] = 108;
    colvals[408] = 111;
    colvals[409] = 113;
    colvals[410] = 114;
    colvals[411] = 115;
    colvals[412] = 116;
    colvals[413] = 117;
    colvals[414] = 118;
    colvals[415] = 120;
    colvals[416] = 121;
    colvals[417] = 123;
    colvals[418] = 124;
    colvals[419] = 125;
    colvals[420] = 126;
    colvals[421] = 128;
    colvals[422] = 129;
    colvals[423] = 0;
    colvals[424] = 1;
    colvals[425] = 2;
    colvals[426] = 3;
    colvals[427] = 7;
    colvals[428] = 8;
    colvals[429] = 16;
    colvals[430] = 19;
    colvals[431] = 20;
    colvals[432] = 21;
    colvals[433] = 23;
    colvals[434] = 26;
    colvals[435] = 30;
    colvals[436] = 34;
    colvals[437] = 37;
    colvals[438] = 51;
    colvals[439] = 52;
    colvals[440] = 55;
    colvals[441] = 62;
    colvals[442] = 65;
    colvals[443] = 72;
    colvals[444] = 75;
    colvals[445] = 76;
    colvals[446] = 78;
    colvals[447] = 93;
    colvals[448] = 97;
    colvals[449] = 98;
    colvals[450] = 100;
    colvals[451] = 108;
    colvals[452] = 111;
    colvals[453] = 116;
    colvals[454] = 120;
    colvals[455] = 124;
    colvals[456] = 128;
    colvals[457] = 0;
    colvals[458] = 1;
    colvals[459] = 3;
    colvals[460] = 4;
    colvals[461] = 7;
    colvals[462] = 8;
    colvals[463] = 15;
    colvals[464] = 16;
    colvals[465] = 17;
    colvals[466] = 19;
    colvals[467] = 20;
    colvals[468] = 21;
    colvals[469] = 22;
    colvals[470] = 23;
    colvals[471] = 24;
    colvals[472] = 26;
    colvals[473] = 30;
    colvals[474] = 34;
    colvals[475] = 37;
    colvals[476] = 51;
    colvals[477] = 54;
    colvals[478] = 55;
    colvals[479] = 58;
    colvals[480] = 59;
    colvals[481] = 60;
    colvals[482] = 61;
    colvals[483] = 62;
    colvals[484] = 64;
    colvals[485] = 65;
    colvals[486] = 67;
    colvals[487] = 69;
    colvals[488] = 75;
    colvals[489] = 76;
    colvals[490] = 78;
    colvals[491] = 80;
    colvals[492] = 87;
    colvals[493] = 88;
    colvals[494] = 90;
    colvals[495] = 92;
    colvals[496] = 93;
    colvals[497] = 97;
    colvals[498] = 98;
    colvals[499] = 105;
    colvals[500] = 108;
    colvals[501] = 111;
    colvals[502] = 113;
    colvals[503] = 116;
    colvals[504] = 117;
    colvals[505] = 119;
    colvals[506] = 120;
    colvals[507] = 121;
    colvals[508] = 122;
    colvals[509] = 124;
    colvals[510] = 125;
    colvals[511] = 127;
    colvals[512] = 128;
    colvals[513] = 129;
    colvals[514] = 130;
    colvals[515] = 0;
    colvals[516] = 5;
    colvals[517] = 7;
    colvals[518] = 9;
    colvals[519] = 34;
    colvals[520] = 51;
    colvals[521] = 72;
    colvals[522] = 75;
    colvals[523] = 97;
    colvals[524] = 1;
    colvals[525] = 3;
    colvals[526] = 4;
    colvals[527] = 6;
    colvals[528] = 8;
    colvals[529] = 9;
    colvals[530] = 10;
    colvals[531] = 16;
    colvals[532] = 20;
    colvals[533] = 25;
    colvals[534] = 34;
    colvals[535] = 36;
    colvals[536] = 39;
    colvals[537] = 43;
    colvals[538] = 51;
    colvals[539] = 54;
    colvals[540] = 57;
    colvals[541] = 64;
    colvals[542] = 66;
    colvals[543] = 75;
    colvals[544] = 83;
    colvals[545] = 84;
    colvals[546] = 87;
    colvals[547] = 88;
    colvals[548] = 113;
    colvals[549] = 1;
    colvals[550] = 11;
    colvals[551] = 14;
    colvals[552] = 26;
    colvals[553] = 31;
    colvals[554] = 34;
    colvals[555] = 51;
    colvals[556] = 113;
    colvals[557] = 0;
    colvals[558] = 3;
    colvals[559] = 5;
    colvals[560] = 7;
    colvals[561] = 12;
    colvals[562] = 15;
    colvals[563] = 19;
    colvals[564] = 34;
    colvals[565] = 51;
    colvals[566] = 72;
    colvals[567] = 75;
    colvals[568] = 97;
    colvals[569] = 0;
    colvals[570] = 1;
    colvals[571] = 3;
    colvals[572] = 4;
    colvals[573] = 5;
    colvals[574] = 6;
    colvals[575] = 7;
    colvals[576] = 8;
    colvals[577] = 12;
    colvals[578] = 13;
    colvals[579] = 15;
    colvals[580] = 16;
    colvals[581] = 19;
    colvals[582] = 20;
    colvals[583] = 34;
    colvals[584] = 51;
    colvals[585] = 113;
    colvals[586] = 0;
    colvals[587] = 1;
    colvals[588] = 14;
    colvals[589] = 34;
    colvals[590] = 41;
    colvals[591] = 51;
    colvals[592] = 59;
    colvals[593] = 72;
    colvals[594] = 75;
    colvals[595] = 97;
    colvals[596] = 0;
    colvals[597] = 1;
    colvals[598] = 2;
    colvals[599] = 3;
    colvals[600] = 4;
    colvals[601] = 5;
    colvals[602] = 6;
    colvals[603] = 8;
    colvals[604] = 15;
    colvals[605] = 16;
    colvals[606] = 17;
    colvals[607] = 18;
    colvals[608] = 20;
    colvals[609] = 21;
    colvals[610] = 23;
    colvals[611] = 24;
    colvals[612] = 26;
    colvals[613] = 30;
    colvals[614] = 33;
    colvals[615] = 34;
    colvals[616] = 35;
    colvals[617] = 37;
    colvals[618] = 38;
    colvals[619] = 39;
    colvals[620] = 40;
    colvals[621] = 41;
    colvals[622] = 42;
    colvals[623] = 44;
    colvals[624] = 45;
    colvals[625] = 46;
    colvals[626] = 50;
    colvals[627] = 51;
    colvals[628] = 52;
    colvals[629] = 53;
    colvals[630] = 55;
    colvals[631] = 56;
    colvals[632] = 58;
    colvals[633] = 59;
    colvals[634] = 60;
    colvals[635] = 61;
    colvals[636] = 62;
    colvals[637] = 63;
    colvals[638] = 65;
    colvals[639] = 67;
    colvals[640] = 68;
    colvals[641] = 69;
    colvals[642] = 72;
    colvals[643] = 75;
    colvals[644] = 76;
    colvals[645] = 78;
    colvals[646] = 79;
    colvals[647] = 80;
    colvals[648] = 84;
    colvals[649] = 85;
    colvals[650] = 86;
    colvals[651] = 88;
    colvals[652] = 90;
    colvals[653] = 91;
    colvals[654] = 92;
    colvals[655] = 93;
    colvals[656] = 97;
    colvals[657] = 98;
    colvals[658] = 99;
    colvals[659] = 100;
    colvals[660] = 101;
    colvals[661] = 103;
    colvals[662] = 105;
    colvals[663] = 108;
    colvals[664] = 111;
    colvals[665] = 113;
    colvals[666] = 114;
    colvals[667] = 115;
    colvals[668] = 116;
    colvals[669] = 117;
    colvals[670] = 118;
    colvals[671] = 119;
    colvals[672] = 120;
    colvals[673] = 121;
    colvals[674] = 122;
    colvals[675] = 123;
    colvals[676] = 124;
    colvals[677] = 125;
    colvals[678] = 126;
    colvals[679] = 127;
    colvals[680] = 128;
    colvals[681] = 129;
    colvals[682] = 130;
    colvals[683] = 0;
    colvals[684] = 1;
    colvals[685] = 3;
    colvals[686] = 4;
    colvals[687] = 5;
    colvals[688] = 6;
    colvals[689] = 7;
    colvals[690] = 15;
    colvals[691] = 16;
    colvals[692] = 17;
    colvals[693] = 18;
    colvals[694] = 19;
    colvals[695] = 21;
    colvals[696] = 23;
    colvals[697] = 24;
    colvals[698] = 25;
    colvals[699] = 26;
    colvals[700] = 30;
    colvals[701] = 31;
    colvals[702] = 33;
    colvals[703] = 34;
    colvals[704] = 36;
    colvals[705] = 37;
    colvals[706] = 39;
    colvals[707] = 40;
    colvals[708] = 41;
    colvals[709] = 42;
    colvals[710] = 44;
    colvals[711] = 46;
    colvals[712] = 50;
    colvals[713] = 51;
    colvals[714] = 54;
    colvals[715] = 55;
    colvals[716] = 57;
    colvals[717] = 59;
    colvals[718] = 61;
    colvals[719] = 62;
    colvals[720] = 64;
    colvals[721] = 65;
    colvals[722] = 72;
    colvals[723] = 75;
    colvals[724] = 76;
    colvals[725] = 78;
    colvals[726] = 79;
    colvals[727] = 83;
    colvals[728] = 84;
    colvals[729] = 85;
    colvals[730] = 86;
    colvals[731] = 87;
    colvals[732] = 89;
    colvals[733] = 90;
    colvals[734] = 91;
    colvals[735] = 92;
    colvals[736] = 93;
    colvals[737] = 97;
    colvals[738] = 98;
    colvals[739] = 100;
    colvals[740] = 101;
    colvals[741] = 103;
    colvals[742] = 107;
    colvals[743] = 108;
    colvals[744] = 110;
    colvals[745] = 111;
    colvals[746] = 113;
    colvals[747] = 114;
    colvals[748] = 115;
    colvals[749] = 116;
    colvals[750] = 117;
    colvals[751] = 118;
    colvals[752] = 119;
    colvals[753] = 120;
    colvals[754] = 121;
    colvals[755] = 123;
    colvals[756] = 124;
    colvals[757] = 125;
    colvals[758] = 126;
    colvals[759] = 127;
    colvals[760] = 128;
    colvals[761] = 129;
    colvals[762] = 0;
    colvals[763] = 1;
    colvals[764] = 2;
    colvals[765] = 4;
    colvals[766] = 15;
    colvals[767] = 16;
    colvals[768] = 17;
    colvals[769] = 18;
    colvals[770] = 19;
    colvals[771] = 20;
    colvals[772] = 26;
    colvals[773] = 30;
    colvals[774] = 33;
    colvals[775] = 34;
    colvals[776] = 35;
    colvals[777] = 37;
    colvals[778] = 41;
    colvals[779] = 45;
    colvals[780] = 50;
    colvals[781] = 51;
    colvals[782] = 55;
    colvals[783] = 61;
    colvals[784] = 62;
    colvals[785] = 65;
    colvals[786] = 72;
    colvals[787] = 75;
    colvals[788] = 76;
    colvals[789] = 78;
    colvals[790] = 86;
    colvals[791] = 90;
    colvals[792] = 92;
    colvals[793] = 93;
    colvals[794] = 97;
    colvals[795] = 98;
    colvals[796] = 101;
    colvals[797] = 108;
    colvals[798] = 111;
    colvals[799] = 115;
    colvals[800] = 116;
    colvals[801] = 120;
    colvals[802] = 123;
    colvals[803] = 124;
    colvals[804] = 128;
    colvals[805] = 0;
    colvals[806] = 1;
    colvals[807] = 4;
    colvals[808] = 6;
    colvals[809] = 15;
    colvals[810] = 16;
    colvals[811] = 17;
    colvals[812] = 18;
    colvals[813] = 19;
    colvals[814] = 20;
    colvals[815] = 26;
    colvals[816] = 30;
    colvals[817] = 34;
    colvals[818] = 37;
    colvals[819] = 38;
    colvals[820] = 40;
    colvals[821] = 41;
    colvals[822] = 42;
    colvals[823] = 44;
    colvals[824] = 46;
    colvals[825] = 51;
    colvals[826] = 53;
    colvals[827] = 55;
    colvals[828] = 61;
    colvals[829] = 62;
    colvals[830] = 63;
    colvals[831] = 65;
    colvals[832] = 75;
    colvals[833] = 76;
    colvals[834] = 78;
    colvals[835] = 79;
    colvals[836] = 84;
    colvals[837] = 86;
    colvals[838] = 90;
    colvals[839] = 92;
    colvals[840] = 93;
    colvals[841] = 97;
    colvals[842] = 98;
    colvals[843] = 100;
    colvals[844] = 101;
    colvals[845] = 103;
    colvals[846] = 108;
    colvals[847] = 111;
    colvals[848] = 113;
    colvals[849] = 114;
    colvals[850] = 115;
    colvals[851] = 116;
    colvals[852] = 117;
    colvals[853] = 118;
    colvals[854] = 120;
    colvals[855] = 121;
    colvals[856] = 123;
    colvals[857] = 124;
    colvals[858] = 125;
    colvals[859] = 126;
    colvals[860] = 128;
    colvals[861] = 129;
    colvals[862] = 0;
    colvals[863] = 1;
    colvals[864] = 2;
    colvals[865] = 3;
    colvals[866] = 4;
    colvals[867] = 6;
    colvals[868] = 7;
    colvals[869] = 8;
    colvals[870] = 16;
    colvals[871] = 17;
    colvals[872] = 19;
    colvals[873] = 20;
    colvals[874] = 21;
    colvals[875] = 22;
    colvals[876] = 23;
    colvals[877] = 24;
    colvals[878] = 26;
    colvals[879] = 30;
    colvals[880] = 33;
    colvals[881] = 34;
    colvals[882] = 35;
    colvals[883] = 37;
    colvals[884] = 38;
    colvals[885] = 40;
    colvals[886] = 41;
    colvals[887] = 42;
    colvals[888] = 44;
    colvals[889] = 45;
    colvals[890] = 46;
    colvals[891] = 50;
    colvals[892] = 51;
    colvals[893] = 52;
    colvals[894] = 53;
    colvals[895] = 55;
    colvals[896] = 56;
    colvals[897] = 57;
    colvals[898] = 58;
    colvals[899] = 59;
    colvals[900] = 60;
    colvals[901] = 61;
    colvals[902] = 62;
    colvals[903] = 63;
    colvals[904] = 65;
    colvals[905] = 67;
    colvals[906] = 68;
    colvals[907] = 69;
    colvals[908] = 72;
    colvals[909] = 75;
    colvals[910] = 76;
    colvals[911] = 78;
    colvals[912] = 79;
    colvals[913] = 80;
    colvals[914] = 84;
    colvals[915] = 86;
    colvals[916] = 88;
    colvals[917] = 89;
    colvals[918] = 90;
    colvals[919] = 91;
    colvals[920] = 92;
    colvals[921] = 93;
    colvals[922] = 97;
    colvals[923] = 98;
    colvals[924] = 99;
    colvals[925] = 100;
    colvals[926] = 101;
    colvals[927] = 103;
    colvals[928] = 105;
    colvals[929] = 108;
    colvals[930] = 111;
    colvals[931] = 113;
    colvals[932] = 114;
    colvals[933] = 115;
    colvals[934] = 116;
    colvals[935] = 117;
    colvals[936] = 118;
    colvals[937] = 119;
    colvals[938] = 120;
    colvals[939] = 121;
    colvals[940] = 122;
    colvals[941] = 123;
    colvals[942] = 124;
    colvals[943] = 125;
    colvals[944] = 127;
    colvals[945] = 128;
    colvals[946] = 129;
    colvals[947] = 130;
    colvals[948] = 0;
    colvals[949] = 1;
    colvals[950] = 3;
    colvals[951] = 4;
    colvals[952] = 5;
    colvals[953] = 7;
    colvals[954] = 8;
    colvals[955] = 15;
    colvals[956] = 17;
    colvals[957] = 19;
    colvals[958] = 20;
    colvals[959] = 21;
    colvals[960] = 22;
    colvals[961] = 23;
    colvals[962] = 24;
    colvals[963] = 25;
    colvals[964] = 26;
    colvals[965] = 30;
    colvals[966] = 31;
    colvals[967] = 33;
    colvals[968] = 34;
    colvals[969] = 36;
    colvals[970] = 37;
    colvals[971] = 39;
    colvals[972] = 41;
    colvals[973] = 50;
    colvals[974] = 51;
    colvals[975] = 54;
    colvals[976] = 55;
    colvals[977] = 57;
    colvals[978] = 58;
    colvals[979] = 59;
    colvals[980] = 60;
    colvals[981] = 61;
    colvals[982] = 62;
    colvals[983] = 64;
    colvals[984] = 65;
    colvals[985] = 67;
    colvals[986] = 69;
    colvals[987] = 72;
    colvals[988] = 75;
    colvals[989] = 76;
    colvals[990] = 78;
    colvals[991] = 80;
    colvals[992] = 83;
    colvals[993] = 85;
    colvals[994] = 86;
    colvals[995] = 87;
    colvals[996] = 88;
    colvals[997] = 89;
    colvals[998] = 90;
    colvals[999] = 91;
    colvals[1000] = 92;
    colvals[1001] = 93;
    colvals[1002] = 97;
    colvals[1003] = 98;
    colvals[1004] = 100;
    colvals[1005] = 101;
    colvals[1006] = 105;
    colvals[1007] = 107;
    colvals[1008] = 108;
    colvals[1009] = 110;
    colvals[1010] = 111;
    colvals[1011] = 113;
    colvals[1012] = 115;
    colvals[1013] = 116;
    colvals[1014] = 117;
    colvals[1015] = 119;
    colvals[1016] = 120;
    colvals[1017] = 121;
    colvals[1018] = 122;
    colvals[1019] = 123;
    colvals[1020] = 124;
    colvals[1021] = 125;
    colvals[1022] = 127;
    colvals[1023] = 128;
    colvals[1024] = 129;
    colvals[1025] = 130;
    colvals[1026] = 0;
    colvals[1027] = 1;
    colvals[1028] = 2;
    colvals[1029] = 4;
    colvals[1030] = 15;
    colvals[1031] = 16;
    colvals[1032] = 19;
    colvals[1033] = 20;
    colvals[1034] = 21;
    colvals[1035] = 22;
    colvals[1036] = 26;
    colvals[1037] = 30;
    colvals[1038] = 33;
    colvals[1039] = 34;
    colvals[1040] = 37;
    colvals[1041] = 50;
    colvals[1042] = 51;
    colvals[1043] = 52;
    colvals[1044] = 55;
    colvals[1045] = 59;
    colvals[1046] = 61;
    colvals[1047] = 62;
    colvals[1048] = 65;
    colvals[1049] = 68;
    colvals[1050] = 72;
    colvals[1051] = 75;
    colvals[1052] = 76;
    colvals[1053] = 78;
    colvals[1054] = 86;
    colvals[1055] = 90;
    colvals[1056] = 92;
    colvals[1057] = 93;
    colvals[1058] = 97;
    colvals[1059] = 98;
    colvals[1060] = 101;
    colvals[1061] = 108;
    colvals[1062] = 111;
    colvals[1063] = 116;
    colvals[1064] = 119;
    colvals[1065] = 120;
    colvals[1066] = 124;
    colvals[1067] = 127;
    colvals[1068] = 128;
    colvals[1069] = 0;
    colvals[1070] = 1;
    colvals[1071] = 4;
    colvals[1072] = 8;
    colvals[1073] = 15;
    colvals[1074] = 16;
    colvals[1075] = 19;
    colvals[1076] = 20;
    colvals[1077] = 21;
    colvals[1078] = 22;
    colvals[1079] = 26;
    colvals[1080] = 30;
    colvals[1081] = 34;
    colvals[1082] = 37;
    colvals[1083] = 51;
    colvals[1084] = 53;
    colvals[1085] = 55;
    colvals[1086] = 56;
    colvals[1087] = 58;
    colvals[1088] = 59;
    colvals[1089] = 60;
    colvals[1090] = 61;
    colvals[1091] = 62;
    colvals[1092] = 63;
    colvals[1093] = 65;
    colvals[1094] = 67;
    colvals[1095] = 69;
    colvals[1096] = 75;
    colvals[1097] = 76;
    colvals[1098] = 78;
    colvals[1099] = 80;
    colvals[1100] = 86;
    colvals[1101] = 88;
    colvals[1102] = 90;
    colvals[1103] = 92;
    colvals[1104] = 93;
    colvals[1105] = 97;
    colvals[1106] = 98;
    colvals[1107] = 100;
    colvals[1108] = 101;
    colvals[1109] = 105;
    colvals[1110] = 108;
    colvals[1111] = 111;
    colvals[1112] = 113;
    colvals[1113] = 116;
    colvals[1114] = 117;
    colvals[1115] = 119;
    colvals[1116] = 120;
    colvals[1117] = 121;
    colvals[1118] = 122;
    colvals[1119] = 124;
    colvals[1120] = 125;
    colvals[1121] = 127;
    colvals[1122] = 128;
    colvals[1123] = 129;
    colvals[1124] = 130;
    colvals[1125] = 0;
    colvals[1126] = 1;
    colvals[1127] = 2;
    colvals[1128] = 4;
    colvals[1129] = 15;
    colvals[1130] = 16;
    colvals[1131] = 19;
    colvals[1132] = 20;
    colvals[1133] = 23;
    colvals[1134] = 24;
    colvals[1135] = 26;
    colvals[1136] = 30;
    colvals[1137] = 33;
    colvals[1138] = 34;
    colvals[1139] = 35;
    colvals[1140] = 37;
    colvals[1141] = 41;
    colvals[1142] = 45;
    colvals[1143] = 50;
    colvals[1144] = 51;
    colvals[1145] = 52;
    colvals[1146] = 55;
    colvals[1147] = 59;
    colvals[1148] = 61;
    colvals[1149] = 62;
    colvals[1150] = 65;
    colvals[1151] = 68;
    colvals[1152] = 72;
    colvals[1153] = 75;
    colvals[1154] = 76;
    colvals[1155] = 78;
    colvals[1156] = 86;
    colvals[1157] = 90;
    colvals[1158] = 92;
    colvals[1159] = 93;
    colvals[1160] = 97;
    colvals[1161] = 98;
    colvals[1162] = 101;
    colvals[1163] = 108;
    colvals[1164] = 111;
    colvals[1165] = 115;
    colvals[1166] = 116;
    colvals[1167] = 119;
    colvals[1168] = 120;
    colvals[1169] = 123;
    colvals[1170] = 124;
    colvals[1171] = 127;
    colvals[1172] = 128;
    colvals[1173] = 0;
    colvals[1174] = 1;
    colvals[1175] = 4;
    colvals[1176] = 6;
    colvals[1177] = 8;
    colvals[1178] = 15;
    colvals[1179] = 16;
    colvals[1180] = 19;
    colvals[1181] = 20;
    colvals[1182] = 23;
    colvals[1183] = 24;
    colvals[1184] = 26;
    colvals[1185] = 30;
    colvals[1186] = 34;
    colvals[1187] = 37;
    colvals[1188] = 38;
    colvals[1189] = 40;
    colvals[1190] = 41;
    colvals[1191] = 42;
    colvals[1192] = 44;
    colvals[1193] = 46;
    colvals[1194] = 51;
    colvals[1195] = 53;
    colvals[1196] = 55;
    colvals[1197] = 56;
    colvals[1198] = 58;
    colvals[1199] = 59;
    colvals[1200] = 60;
    colvals[1201] = 61;
    colvals[1202] = 62;
    colvals[1203] = 63;
    colvals[1204] = 65;
    colvals[1205] = 67;
    colvals[1206] = 69;
    colvals[1207] = 75;
    colvals[1208] = 76;
    colvals[1209] = 78;
    colvals[1210] = 79;
    colvals[1211] = 80;
    colvals[1212] = 84;
    colvals[1213] = 86;
    colvals[1214] = 88;
    colvals[1215] = 90;
    colvals[1216] = 92;
    colvals[1217] = 93;
    colvals[1218] = 97;
    colvals[1219] = 98;
    colvals[1220] = 100;
    colvals[1221] = 101;
    colvals[1222] = 103;
    colvals[1223] = 105;
    colvals[1224] = 108;
    colvals[1225] = 111;
    colvals[1226] = 113;
    colvals[1227] = 115;
    colvals[1228] = 116;
    colvals[1229] = 117;
    colvals[1230] = 119;
    colvals[1231] = 120;
    colvals[1232] = 121;
    colvals[1233] = 122;
    colvals[1234] = 123;
    colvals[1235] = 124;
    colvals[1236] = 125;
    colvals[1237] = 127;
    colvals[1238] = 128;
    colvals[1239] = 129;
    colvals[1240] = 130;
    colvals[1241] = 0;
    colvals[1242] = 1;
    colvals[1243] = 2;
    colvals[1244] = 3;
    colvals[1245] = 4;
    colvals[1246] = 5;
    colvals[1247] = 6;
    colvals[1248] = 7;
    colvals[1249] = 8;
    colvals[1250] = 9;
    colvals[1251] = 10;
    colvals[1252] = 12;
    colvals[1253] = 14;
    colvals[1254] = 15;
    colvals[1255] = 16;
    colvals[1256] = 17;
    colvals[1257] = 19;
    colvals[1258] = 20;
    colvals[1259] = 21;
    colvals[1260] = 23;
    colvals[1261] = 25;
    colvals[1262] = 26;
    colvals[1263] = 27;
    colvals[1264] = 28;
    colvals[1265] = 29;
    colvals[1266] = 31;
    colvals[1267] = 33;
    colvals[1268] = 35;
    colvals[1269] = 36;
    colvals[1270] = 38;
    colvals[1271] = 39;
    colvals[1272] = 40;
    colvals[1273] = 41;
    colvals[1274] = 43;
    colvals[1275] = 44;
    colvals[1276] = 45;
    colvals[1277] = 50;
    colvals[1278] = 52;
    colvals[1279] = 53;
    colvals[1280] = 54;
    colvals[1281] = 56;
    colvals[1282] = 57;
    colvals[1283] = 58;
    colvals[1284] = 59;
    colvals[1285] = 62;
    colvals[1286] = 63;
    colvals[1287] = 64;
    colvals[1288] = 66;
    colvals[1289] = 67;
    colvals[1290] = 68;
    colvals[1291] = 72;
    colvals[1292] = 75;
    colvals[1293] = 76;
    colvals[1294] = 77;
    colvals[1295] = 78;
    colvals[1296] = 83;
    colvals[1297] = 84;
    colvals[1298] = 85;
    colvals[1299] = 87;
    colvals[1300] = 88;
    colvals[1301] = 89;
    colvals[1302] = 91;
    colvals[1303] = 93;
    colvals[1304] = 97;
    colvals[1305] = 98;
    colvals[1306] = 99;
    colvals[1307] = 100;
    colvals[1308] = 103;
    colvals[1309] = 105;
    colvals[1310] = 106;
    colvals[1311] = 107;
    colvals[1312] = 108;
    colvals[1313] = 109;
    colvals[1314] = 110;
    colvals[1315] = 111;
    colvals[1316] = 112;
    colvals[1317] = 113;
    colvals[1318] = 114;
    colvals[1319] = 116;
    colvals[1320] = 117;
    colvals[1321] = 118;
    colvals[1322] = 120;
    colvals[1323] = 121;
    colvals[1324] = 122;
    colvals[1325] = 124;
    colvals[1326] = 125;
    colvals[1327] = 126;
    colvals[1328] = 128;
    colvals[1329] = 129;
    colvals[1330] = 130;
    colvals[1331] = 0;
    colvals[1332] = 1;
    colvals[1333] = 3;
    colvals[1334] = 5;
    colvals[1335] = 7;
    colvals[1336] = 15;
    colvals[1337] = 16;
    colvals[1338] = 17;
    colvals[1339] = 19;
    colvals[1340] = 20;
    colvals[1341] = 21;
    colvals[1342] = 23;
    colvals[1343] = 25;
    colvals[1344] = 26;
    colvals[1345] = 29;
    colvals[1346] = 31;
    colvals[1347] = 33;
    colvals[1348] = 36;
    colvals[1349] = 39;
    colvals[1350] = 41;
    colvals[1351] = 43;
    colvals[1352] = 50;
    colvals[1353] = 54;
    colvals[1354] = 57;
    colvals[1355] = 59;
    colvals[1356] = 61;
    colvals[1357] = 62;
    colvals[1358] = 64;
    colvals[1359] = 66;
    colvals[1360] = 72;
    colvals[1361] = 75;
    colvals[1362] = 76;
    colvals[1363] = 78;
    colvals[1364] = 83;
    colvals[1365] = 85;
    colvals[1366] = 87;
    colvals[1367] = 89;
    colvals[1368] = 91;
    colvals[1369] = 93;
    colvals[1370] = 97;
    colvals[1371] = 100;
    colvals[1372] = 106;
    colvals[1373] = 107;
    colvals[1374] = 110;
    colvals[1375] = 113;
    colvals[1376] = 115;
    colvals[1377] = 116;
    colvals[1378] = 119;
    colvals[1379] = 120;
    colvals[1380] = 123;
    colvals[1381] = 124;
    colvals[1382] = 127;
    colvals[1383] = 128;
    colvals[1384] = 2;
    colvals[1385] = 25;
    colvals[1386] = 27;
    colvals[1387] = 33;
    colvals[1388] = 35;
    colvals[1389] = 38;
    colvals[1390] = 39;
    colvals[1391] = 50;
    colvals[1392] = 52;
    colvals[1393] = 53;
    colvals[1394] = 56;
    colvals[1395] = 57;
    colvals[1396] = 63;
    colvals[1397] = 93;
    colvals[1398] = 99;
    colvals[1399] = 109;
    colvals[1400] = 112;
    colvals[1401] = 1;
    colvals[1402] = 16;
    colvals[1403] = 20;
    colvals[1404] = 25;
    colvals[1405] = 28;
    colvals[1406] = 36;
    colvals[1407] = 39;
    colvals[1408] = 54;
    colvals[1409] = 57;
    colvals[1410] = 64;
    colvals[1411] = 113;
    colvals[1412] = 0;
    colvals[1413] = 1;
    colvals[1414] = 2;
    colvals[1415] = 3;
    colvals[1416] = 4;
    colvals[1417] = 5;
    colvals[1418] = 7;
    colvals[1419] = 9;
    colvals[1420] = 11;
    colvals[1421] = 12;
    colvals[1422] = 14;
    colvals[1423] = 15;
    colvals[1424] = 16;
    colvals[1425] = 17;
    colvals[1426] = 19;
    colvals[1427] = 20;
    colvals[1428] = 21;
    colvals[1429] = 23;
    colvals[1430] = 25;
    colvals[1431] = 26;
    colvals[1432] = 29;
    colvals[1433] = 30;
    colvals[1434] = 31;
    colvals[1435] = 32;
    colvals[1436] = 33;
    colvals[1437] = 34;
    colvals[1438] = 35;
    colvals[1439] = 36;
    colvals[1440] = 37;
    colvals[1441] = 39;
    colvals[1442] = 40;
    colvals[1443] = 41;
    colvals[1444] = 42;
    colvals[1445] = 45;
    colvals[1446] = 46;
    colvals[1447] = 47;
    colvals[1448] = 48;
    colvals[1449] = 50;
    colvals[1450] = 51;
    colvals[1451] = 52;
    colvals[1452] = 54;
    colvals[1453] = 55;
    colvals[1454] = 57;
    colvals[1455] = 58;
    colvals[1456] = 59;
    colvals[1457] = 60;
    colvals[1458] = 62;
    colvals[1459] = 64;
    colvals[1460] = 65;
    colvals[1461] = 68;
    colvals[1462] = 69;
    colvals[1463] = 70;
    colvals[1464] = 72;
    colvals[1465] = 75;
    colvals[1466] = 76;
    colvals[1467] = 77;
    colvals[1468] = 78;
    colvals[1469] = 79;
    colvals[1470] = 80;
    colvals[1471] = 82;
    colvals[1472] = 83;
    colvals[1473] = 84;
    colvals[1474] = 85;
    colvals[1475] = 87;
    colvals[1476] = 88;
    colvals[1477] = 89;
    colvals[1478] = 91;
    colvals[1479] = 93;
    colvals[1480] = 97;
    colvals[1481] = 98;
    colvals[1482] = 99;
    colvals[1483] = 100;
    colvals[1484] = 101;
    colvals[1485] = 103;
    colvals[1486] = 105;
    colvals[1487] = 106;
    colvals[1488] = 107;
    colvals[1489] = 108;
    colvals[1490] = 110;
    colvals[1491] = 111;
    colvals[1492] = 113;
    colvals[1493] = 114;
    colvals[1494] = 116;
    colvals[1495] = 117;
    colvals[1496] = 118;
    colvals[1497] = 120;
    colvals[1498] = 121;
    colvals[1499] = 122;
    colvals[1500] = 124;
    colvals[1501] = 125;
    colvals[1502] = 126;
    colvals[1503] = 128;
    colvals[1504] = 129;
    colvals[1505] = 130;
    colvals[1506] = 0;
    colvals[1507] = 1;
    colvals[1508] = 3;
    colvals[1509] = 4;
    colvals[1510] = 5;
    colvals[1511] = 7;
    colvals[1512] = 15;
    colvals[1513] = 16;
    colvals[1514] = 17;
    colvals[1515] = 19;
    colvals[1516] = 20;
    colvals[1517] = 21;
    colvals[1518] = 23;
    colvals[1519] = 26;
    colvals[1520] = 29;
    colvals[1521] = 30;
    colvals[1522] = 31;
    colvals[1523] = 33;
    colvals[1524] = 34;
    colvals[1525] = 36;
    colvals[1526] = 39;
    colvals[1527] = 41;
    colvals[1528] = 50;
    colvals[1529] = 51;
    colvals[1530] = 54;
    colvals[1531] = 57;
    colvals[1532] = 59;
    colvals[1533] = 61;
    colvals[1534] = 62;
    colvals[1535] = 64;
    colvals[1536] = 72;
    colvals[1537] = 75;
    colvals[1538] = 76;
    colvals[1539] = 78;
    colvals[1540] = 83;
    colvals[1541] = 85;
    colvals[1542] = 87;
    colvals[1543] = 89;
    colvals[1544] = 91;
    colvals[1545] = 93;
    colvals[1546] = 97;
    colvals[1547] = 98;
    colvals[1548] = 100;
    colvals[1549] = 101;
    colvals[1550] = 106;
    colvals[1551] = 107;
    colvals[1552] = 110;
    colvals[1553] = 113;
    colvals[1554] = 115;
    colvals[1555] = 116;
    colvals[1556] = 119;
    colvals[1557] = 120;
    colvals[1558] = 123;
    colvals[1559] = 124;
    colvals[1560] = 127;
    colvals[1561] = 128;
    colvals[1562] = 1;
    colvals[1563] = 2;
    colvals[1564] = 16;
    colvals[1565] = 20;
    colvals[1566] = 26;
    colvals[1567] = 29;
    colvals[1568] = 30;
    colvals[1569] = 31;
    colvals[1570] = 32;
    colvals[1571] = 33;
    colvals[1572] = 34;
    colvals[1573] = 36;
    colvals[1574] = 41;
    colvals[1575] = 45;
    colvals[1576] = 50;
    colvals[1577] = 51;
    colvals[1578] = 54;
    colvals[1579] = 59;
    colvals[1580] = 62;
    colvals[1581] = 64;
    colvals[1582] = 68;
    colvals[1583] = 72;
    colvals[1584] = 76;
    colvals[1585] = 78;
    colvals[1586] = 84;
    colvals[1587] = 88;
    colvals[1588] = 93;
    colvals[1589] = 97;
    colvals[1590] = 98;
    colvals[1591] = 99;
    colvals[1592] = 100;
    colvals[1593] = 107;
    colvals[1594] = 110;
    colvals[1595] = 116;
    colvals[1596] = 120;
    colvals[1597] = 124;
    colvals[1598] = 128;
    colvals[1599] = 26;
    colvals[1600] = 30;
    colvals[1601] = 31;
    colvals[1602] = 32;
    colvals[1603] = 33;
    colvals[1604] = 36;
    colvals[1605] = 50;
    colvals[1606] = 54;
    colvals[1607] = 62;
    colvals[1608] = 64;
    colvals[1609] = 72;
    colvals[1610] = 76;
    colvals[1611] = 78;
    colvals[1612] = 93;
    colvals[1613] = 97;
    colvals[1614] = 100;
    colvals[1615] = 113;
    colvals[1616] = 116;
    colvals[1617] = 120;
    colvals[1618] = 124;
    colvals[1619] = 128;
    colvals[1620] = 0;
    colvals[1621] = 1;
    colvals[1622] = 2;
    colvals[1623] = 3;
    colvals[1624] = 4;
    colvals[1625] = 5;
    colvals[1626] = 6;
    colvals[1627] = 7;
    colvals[1628] = 9;
    colvals[1629] = 12;
    colvals[1630] = 14;
    colvals[1631] = 15;
    colvals[1632] = 16;
    colvals[1633] = 17;
    colvals[1634] = 18;
    colvals[1635] = 19;
    colvals[1636] = 20;
    colvals[1637] = 21;
    colvals[1638] = 23;
    colvals[1639] = 24;
    colvals[1640] = 25;
    colvals[1641] = 26;
    colvals[1642] = 27;
    colvals[1643] = 29;
    colvals[1644] = 30;
    colvals[1645] = 31;
    colvals[1646] = 32;
    colvals[1647] = 33;
    colvals[1648] = 34;
    colvals[1649] = 35;
    colvals[1650] = 36;
    colvals[1651] = 37;
    colvals[1652] = 38;
    colvals[1653] = 39;
    colvals[1654] = 40;
    colvals[1655] = 41;
    colvals[1656] = 42;
    colvals[1657] = 43;
    colvals[1658] = 44;
    colvals[1659] = 45;
    colvals[1660] = 46;
    colvals[1661] = 47;
    colvals[1662] = 48;
    colvals[1663] = 50;
    colvals[1664] = 51;
    colvals[1665] = 52;
    colvals[1666] = 53;
    colvals[1667] = 54;
    colvals[1668] = 55;
    colvals[1669] = 56;
    colvals[1670] = 57;
    colvals[1671] = 58;
    colvals[1672] = 59;
    colvals[1673] = 61;
    colvals[1674] = 62;
    colvals[1675] = 63;
    colvals[1676] = 64;
    colvals[1677] = 65;
    colvals[1678] = 68;
    colvals[1679] = 72;
    colvals[1680] = 73;
    colvals[1681] = 74;
    colvals[1682] = 75;
    colvals[1683] = 76;
    colvals[1684] = 77;
    colvals[1685] = 78;
    colvals[1686] = 79;
    colvals[1687] = 81;
    colvals[1688] = 83;
    colvals[1689] = 84;
    colvals[1690] = 85;
    colvals[1691] = 86;
    colvals[1692] = 87;
    colvals[1693] = 88;
    colvals[1694] = 89;
    colvals[1695] = 91;
    colvals[1696] = 92;
    colvals[1697] = 93;
    colvals[1698] = 95;
    colvals[1699] = 96;
    colvals[1700] = 97;
    colvals[1701] = 98;
    colvals[1702] = 99;
    colvals[1703] = 100;
    colvals[1704] = 101;
    colvals[1705] = 102;
    colvals[1706] = 103;
    colvals[1707] = 104;
    colvals[1708] = 107;
    colvals[1709] = 108;
    colvals[1710] = 109;
    colvals[1711] = 110;
    colvals[1712] = 111;
    colvals[1713] = 112;
    colvals[1714] = 113;
    colvals[1715] = 114;
    colvals[1716] = 115;
    colvals[1717] = 116;
    colvals[1718] = 117;
    colvals[1719] = 118;
    colvals[1720] = 119;
    colvals[1721] = 120;
    colvals[1722] = 121;
    colvals[1723] = 122;
    colvals[1724] = 123;
    colvals[1725] = 124;
    colvals[1726] = 125;
    colvals[1727] = 126;
    colvals[1728] = 127;
    colvals[1729] = 128;
    colvals[1730] = 129;
    colvals[1731] = 130;
    colvals[1732] = 2;
    colvals[1733] = 3;
    colvals[1734] = 5;
    colvals[1735] = 7;
    colvals[1736] = 9;
    colvals[1737] = 12;
    colvals[1738] = 14;
    colvals[1739] = 15;
    colvals[1740] = 16;
    colvals[1741] = 17;
    colvals[1742] = 19;
    colvals[1743] = 21;
    colvals[1744] = 23;
    colvals[1745] = 26;
    colvals[1746] = 30;
    colvals[1747] = 31;
    colvals[1748] = 32;
    colvals[1749] = 33;
    colvals[1750] = 34;
    colvals[1751] = 35;
    colvals[1752] = 36;
    colvals[1753] = 39;
    colvals[1754] = 40;
    colvals[1755] = 41;
    colvals[1756] = 43;
    colvals[1757] = 45;
    colvals[1758] = 48;
    colvals[1759] = 50;
    colvals[1760] = 51;
    colvals[1761] = 52;
    colvals[1762] = 54;
    colvals[1763] = 57;
    colvals[1764] = 58;
    colvals[1765] = 59;
    colvals[1766] = 61;
    colvals[1767] = 62;
    colvals[1768] = 64;
    colvals[1769] = 66;
    colvals[1770] = 68;
    colvals[1771] = 72;
    colvals[1772] = 78;
    colvals[1773] = 83;
    colvals[1774] = 85;
    colvals[1775] = 87;
    colvals[1776] = 89;
    colvals[1777] = 91;
    colvals[1778] = 93;
    colvals[1779] = 97;
    colvals[1780] = 98;
    colvals[1781] = 100;
    colvals[1782] = 107;
    colvals[1783] = 108;
    colvals[1784] = 110;
    colvals[1785] = 113;
    colvals[1786] = 114;
    colvals[1787] = 115;
    colvals[1788] = 116;
    colvals[1789] = 117;
    colvals[1790] = 118;
    colvals[1791] = 119;
    colvals[1792] = 120;
    colvals[1793] = 121;
    colvals[1794] = 123;
    colvals[1795] = 124;
    colvals[1796] = 125;
    colvals[1797] = 126;
    colvals[1798] = 127;
    colvals[1799] = 128;
    colvals[1800] = 129;
    colvals[1801] = 0;
    colvals[1802] = 1;
    colvals[1803] = 3;
    colvals[1804] = 15;
    colvals[1805] = 19;
    colvals[1806] = 25;
    colvals[1807] = 29;
    colvals[1808] = 33;
    colvals[1809] = 34;
    colvals[1810] = 35;
    colvals[1811] = 36;
    colvals[1812] = 38;
    colvals[1813] = 39;
    colvals[1814] = 42;
    colvals[1815] = 50;
    colvals[1816] = 51;
    colvals[1817] = 53;
    colvals[1818] = 54;
    colvals[1819] = 56;
    colvals[1820] = 57;
    colvals[1821] = 60;
    colvals[1822] = 61;
    colvals[1823] = 62;
    colvals[1824] = 63;
    colvals[1825] = 64;
    colvals[1826] = 72;
    colvals[1827] = 75;
    colvals[1828] = 76;
    colvals[1829] = 83;
    colvals[1830] = 87;
    colvals[1831] = 97;
    colvals[1832] = 98;
    colvals[1833] = 107;
    colvals[1834] = 110;
    colvals[1835] = 113;
    colvals[1836] = 114;
    colvals[1837] = 115;
    colvals[1838] = 116;
    colvals[1839] = 117;
    colvals[1840] = 118;
    colvals[1841] = 120;
    colvals[1842] = 121;
    colvals[1843] = 122;
    colvals[1844] = 123;
    colvals[1845] = 124;
    colvals[1846] = 125;
    colvals[1847] = 126;
    colvals[1848] = 128;
    colvals[1849] = 129;
    colvals[1850] = 130;
    colvals[1851] = 1;
    colvals[1852] = 2;
    colvals[1853] = 3;
    colvals[1854] = 4;
    colvals[1855] = 5;
    colvals[1856] = 7;
    colvals[1857] = 10;
    colvals[1858] = 15;
    colvals[1859] = 16;
    colvals[1860] = 17;
    colvals[1861] = 19;
    colvals[1862] = 20;
    colvals[1863] = 21;
    colvals[1864] = 23;
    colvals[1865] = 26;
    colvals[1866] = 27;
    colvals[1867] = 28;
    colvals[1868] = 30;
    colvals[1869] = 31;
    colvals[1870] = 32;
    colvals[1871] = 33;
    colvals[1872] = 34;
    colvals[1873] = 35;
    colvals[1874] = 36;
    colvals[1875] = 37;
    colvals[1876] = 38;
    colvals[1877] = 39;
    colvals[1878] = 40;
    colvals[1879] = 41;
    colvals[1880] = 42;
    colvals[1881] = 44;
    colvals[1882] = 45;
    colvals[1883] = 46;
    colvals[1884] = 50;
    colvals[1885] = 51;
    colvals[1886] = 52;
    colvals[1887] = 53;
    colvals[1888] = 55;
    colvals[1889] = 58;
    colvals[1890] = 59;
    colvals[1891] = 60;
    colvals[1892] = 61;
    colvals[1893] = 62;
    colvals[1894] = 63;
    colvals[1895] = 65;
    colvals[1896] = 67;
    colvals[1897] = 69;
    colvals[1898] = 72;
    colvals[1899] = 76;
    colvals[1900] = 78;
    colvals[1901] = 79;
    colvals[1902] = 80;
    colvals[1903] = 84;
    colvals[1904] = 85;
    colvals[1905] = 86;
    colvals[1906] = 88;
    colvals[1907] = 89;
    colvals[1908] = 90;
    colvals[1909] = 91;
    colvals[1910] = 92;
    colvals[1911] = 93;
    colvals[1912] = 96;
    colvals[1913] = 98;
    colvals[1914] = 99;
    colvals[1915] = 100;
    colvals[1916] = 102;
    colvals[1917] = 107;
    colvals[1918] = 108;
    colvals[1919] = 109;
    colvals[1920] = 110;
    colvals[1921] = 111;
    colvals[1922] = 112;
    colvals[1923] = 113;
    colvals[1924] = 114;
    colvals[1925] = 115;
    colvals[1926] = 116;
    colvals[1927] = 117;
    colvals[1928] = 118;
    colvals[1929] = 120;
    colvals[1930] = 121;
    colvals[1931] = 122;
    colvals[1932] = 123;
    colvals[1933] = 124;
    colvals[1934] = 125;
    colvals[1935] = 126;
    colvals[1936] = 128;
    colvals[1937] = 129;
    colvals[1938] = 130;
    colvals[1939] = 0;
    colvals[1940] = 3;
    colvals[1941] = 5;
    colvals[1942] = 7;
    colvals[1943] = 15;
    colvals[1944] = 17;
    colvals[1945] = 19;
    colvals[1946] = 21;
    colvals[1947] = 23;
    colvals[1948] = 29;
    colvals[1949] = 30;
    colvals[1950] = 32;
    colvals[1951] = 33;
    colvals[1952] = 34;
    colvals[1953] = 36;
    colvals[1954] = 37;
    colvals[1955] = 38;
    colvals[1956] = 40;
    colvals[1957] = 41;
    colvals[1958] = 42;
    colvals[1959] = 44;
    colvals[1960] = 46;
    colvals[1961] = 50;
    colvals[1962] = 51;
    colvals[1963] = 53;
    colvals[1964] = 54;
    colvals[1965] = 58;
    colvals[1966] = 59;
    colvals[1967] = 61;
    colvals[1968] = 62;
    colvals[1969] = 63;
    colvals[1970] = 64;
    colvals[1971] = 72;
    colvals[1972] = 75;
    colvals[1973] = 76;
    colvals[1974] = 78;
    colvals[1975] = 79;
    colvals[1976] = 83;
    colvals[1977] = 84;
    colvals[1978] = 85;
    colvals[1979] = 87;
    colvals[1980] = 88;
    colvals[1981] = 89;
    colvals[1982] = 91;
    colvals[1983] = 93;
    colvals[1984] = 97;
    colvals[1985] = 98;
    colvals[1986] = 100;
    colvals[1987] = 103;
    colvals[1988] = 107;
    colvals[1989] = 108;
    colvals[1990] = 110;
    colvals[1991] = 111;
    colvals[1992] = 113;
    colvals[1993] = 114;
    colvals[1994] = 115;
    colvals[1995] = 116;
    colvals[1996] = 117;
    colvals[1997] = 118;
    colvals[1998] = 119;
    colvals[1999] = 120;
    colvals[2000] = 121;
    colvals[2001] = 123;
    colvals[2002] = 124;
    colvals[2003] = 125;
    colvals[2004] = 126;
    colvals[2005] = 127;
    colvals[2006] = 128;
    colvals[2007] = 129;
    colvals[2008] = 0;
    colvals[2009] = 2;
    colvals[2010] = 15;
    colvals[2011] = 16;
    colvals[2012] = 19;
    colvals[2013] = 27;
    colvals[2014] = 33;
    colvals[2015] = 35;
    colvals[2016] = 36;
    colvals[2017] = 37;
    colvals[2018] = 38;
    colvals[2019] = 40;
    colvals[2020] = 41;
    colvals[2021] = 42;
    colvals[2022] = 44;
    colvals[2023] = 46;
    colvals[2024] = 50;
    colvals[2025] = 52;
    colvals[2026] = 53;
    colvals[2027] = 54;
    colvals[2028] = 61;
    colvals[2029] = 62;
    colvals[2030] = 63;
    colvals[2031] = 64;
    colvals[2032] = 65;
    colvals[2033] = 79;
    colvals[2034] = 83;
    colvals[2035] = 84;
    colvals[2036] = 86;
    colvals[2037] = 92;
    colvals[2038] = 99;
    colvals[2039] = 107;
    colvals[2040] = 108;
    colvals[2041] = 109;
    colvals[2042] = 112;
    colvals[2043] = 113;
    colvals[2044] = 114;
    colvals[2045] = 115;
    colvals[2046] = 116;
    colvals[2047] = 117;
    colvals[2048] = 118;
    colvals[2049] = 119;
    colvals[2050] = 121;
    colvals[2051] = 123;
    colvals[2052] = 124;
    colvals[2053] = 125;
    colvals[2054] = 126;
    colvals[2055] = 127;
    colvals[2056] = 129;
    colvals[2057] = 0;
    colvals[2058] = 1;
    colvals[2059] = 2;
    colvals[2060] = 10;
    colvals[2061] = 15;
    colvals[2062] = 16;
    colvals[2063] = 17;
    colvals[2064] = 20;
    colvals[2065] = 23;
    colvals[2066] = 25;
    colvals[2067] = 26;
    colvals[2068] = 27;
    colvals[2069] = 28;
    colvals[2070] = 30;
    colvals[2071] = 33;
    colvals[2072] = 34;
    colvals[2073] = 35;
    colvals[2074] = 36;
    colvals[2075] = 39;
    colvals[2076] = 40;
    colvals[2077] = 41;
    colvals[2078] = 43;
    colvals[2079] = 45;
    colvals[2080] = 50;
    colvals[2081] = 51;
    colvals[2082] = 52;
    colvals[2083] = 54;
    colvals[2084] = 62;
    colvals[2085] = 64;
    colvals[2086] = 66;
    colvals[2087] = 72;
    colvals[2088] = 75;
    colvals[2089] = 76;
    colvals[2090] = 78;
    colvals[2091] = 83;
    colvals[2092] = 85;
    colvals[2093] = 91;
    colvals[2094] = 93;
    colvals[2095] = 97;
    colvals[2096] = 98;
    colvals[2097] = 99;
    colvals[2098] = 100;
    colvals[2099] = 107;
    colvals[2100] = 109;
    colvals[2101] = 112;
    colvals[2102] = 116;
    colvals[2103] = 120;
    colvals[2104] = 124;
    colvals[2105] = 128;
    colvals[2106] = 0;
    colvals[2107] = 1;
    colvals[2108] = 3;
    colvals[2109] = 15;
    colvals[2110] = 16;
    colvals[2111] = 18;
    colvals[2112] = 19;
    colvals[2113] = 20;
    colvals[2114] = 24;
    colvals[2115] = 25;
    colvals[2116] = 26;
    colvals[2117] = 29;
    colvals[2118] = 30;
    colvals[2119] = 33;
    colvals[2120] = 34;
    colvals[2121] = 36;
    colvals[2122] = 39;
    colvals[2123] = 40;
    colvals[2124] = 41;
    colvals[2125] = 50;
    colvals[2126] = 51;
    colvals[2127] = 54;
    colvals[2128] = 61;
    colvals[2129] = 62;
    colvals[2130] = 64;
    colvals[2131] = 75;
    colvals[2132] = 76;
    colvals[2133] = 78;
    colvals[2134] = 83;
    colvals[2135] = 84;
    colvals[2136] = 85;
    colvals[2137] = 87;
    colvals[2138] = 89;
    colvals[2139] = 91;
    colvals[2140] = 93;
    colvals[2141] = 97;
    colvals[2142] = 100;
    colvals[2143] = 103;
    colvals[2144] = 107;
    colvals[2145] = 108;
    colvals[2146] = 110;
    colvals[2147] = 113;
    colvals[2148] = 114;
    colvals[2149] = 115;
    colvals[2150] = 116;
    colvals[2151] = 117;
    colvals[2152] = 118;
    colvals[2153] = 120;
    colvals[2154] = 121;
    colvals[2155] = 123;
    colvals[2156] = 124;
    colvals[2157] = 125;
    colvals[2158] = 126;
    colvals[2159] = 128;
    colvals[2160] = 129;
    colvals[2161] = 0;
    colvals[2162] = 1;
    colvals[2163] = 2;
    colvals[2164] = 4;
    colvals[2165] = 5;
    colvals[2166] = 15;
    colvals[2167] = 16;
    colvals[2168] = 20;
    colvals[2169] = 25;
    colvals[2170] = 26;
    colvals[2171] = 29;
    colvals[2172] = 30;
    colvals[2173] = 31;
    colvals[2174] = 33;
    colvals[2175] = 34;
    colvals[2176] = 35;
    colvals[2177] = 37;
    colvals[2178] = 41;
    colvals[2179] = 44;
    colvals[2180] = 50;
    colvals[2181] = 51;
    colvals[2182] = 55;
    colvals[2183] = 62;
    colvals[2184] = 65;
    colvals[2185] = 72;
    colvals[2186] = 75;
    colvals[2187] = 76;
    colvals[2188] = 78;
    colvals[2189] = 84;
    colvals[2190] = 86;
    colvals[2191] = 90;
    colvals[2192] = 92;
    colvals[2193] = 97;
    colvals[2194] = 98;
    colvals[2195] = 99;
    colvals[2196] = 100;
    colvals[2197] = 101;
    colvals[2198] = 107;
    colvals[2199] = 108;
    colvals[2200] = 109;
    colvals[2201] = 110;
    colvals[2202] = 111;
    colvals[2203] = 116;
    colvals[2204] = 120;
    colvals[2205] = 124;
    colvals[2206] = 128;
    colvals[2207] = 0;
    colvals[2208] = 1;
    colvals[2209] = 3;
    colvals[2210] = 4;
    colvals[2211] = 5;
    colvals[2212] = 6;
    colvals[2213] = 10;
    colvals[2214] = 15;
    colvals[2215] = 16;
    colvals[2216] = 17;
    colvals[2217] = 18;
    colvals[2218] = 19;
    colvals[2219] = 20;
    colvals[2220] = 23;
    colvals[2221] = 24;
    colvals[2222] = 26;
    colvals[2223] = 28;
    colvals[2224] = 29;
    colvals[2225] = 30;
    colvals[2226] = 31;
    colvals[2227] = 32;
    colvals[2228] = 33;
    colvals[2229] = 34;
    colvals[2230] = 35;
    colvals[2231] = 36;
    colvals[2232] = 37;
    colvals[2233] = 38;
    colvals[2234] = 39;
    colvals[2235] = 40;
    colvals[2236] = 41;
    colvals[2237] = 42;
    colvals[2238] = 46;
    colvals[2239] = 47;
    colvals[2240] = 48;
    colvals[2241] = 51;
    colvals[2242] = 52;
    colvals[2243] = 53;
    colvals[2244] = 54;
    colvals[2245] = 55;
    colvals[2246] = 61;
    colvals[2247] = 62;
    colvals[2248] = 63;
    colvals[2249] = 64;
    colvals[2250] = 65;
    colvals[2251] = 70;
    colvals[2252] = 76;
    colvals[2253] = 78;
    colvals[2254] = 79;
    colvals[2255] = 83;
    colvals[2256] = 84;
    colvals[2257] = 85;
    colvals[2258] = 86;
    colvals[2259] = 87;
    colvals[2260] = 90;
    colvals[2261] = 91;
    colvals[2262] = 92;
    colvals[2263] = 97;
    colvals[2264] = 98;
    colvals[2265] = 100;
    colvals[2266] = 101;
    colvals[2267] = 103;
    colvals[2268] = 107;
    colvals[2269] = 108;
    colvals[2270] = 110;
    colvals[2271] = 111;
    colvals[2272] = 113;
    colvals[2273] = 114;
    colvals[2274] = 115;
    colvals[2275] = 116;
    colvals[2276] = 117;
    colvals[2277] = 118;
    colvals[2278] = 119;
    colvals[2279] = 120;
    colvals[2280] = 121;
    colvals[2281] = 123;
    colvals[2282] = 124;
    colvals[2283] = 125;
    colvals[2284] = 126;
    colvals[2285] = 127;
    colvals[2286] = 128;
    colvals[2287] = 129;
    colvals[2288] = 0;
    colvals[2289] = 1;
    colvals[2290] = 17;
    colvals[2291] = 23;
    colvals[2292] = 34;
    colvals[2293] = 43;
    colvals[2294] = 44;
    colvals[2295] = 51;
    colvals[2296] = 72;
    colvals[2297] = 75;
    colvals[2298] = 85;
    colvals[2299] = 91;
    colvals[2300] = 93;
    colvals[2301] = 0;
    colvals[2302] = 3;
    colvals[2303] = 15;
    colvals[2304] = 19;
    colvals[2305] = 26;
    colvals[2306] = 36;
    colvals[2307] = 44;
    colvals[2308] = 54;
    colvals[2309] = 61;
    colvals[2310] = 64;
    colvals[2311] = 83;
    colvals[2312] = 87;
    colvals[2313] = 93;
    colvals[2314] = 100;
    colvals[2315] = 107;
    colvals[2316] = 110;
    colvals[2317] = 113;
    colvals[2318] = 115;
    colvals[2319] = 123;
    colvals[2320] = 15;
    colvals[2321] = 19;
    colvals[2322] = 25;
    colvals[2323] = 29;
    colvals[2324] = 33;
    colvals[2325] = 34;
    colvals[2326] = 45;
    colvals[2327] = 46;
    colvals[2328] = 50;
    colvals[2329] = 51;
    colvals[2330] = 72;
    colvals[2331] = 85;
    colvals[2332] = 91;
    colvals[2333] = 93;
    colvals[2334] = 97;
    colvals[2335] = 107;
    colvals[2336] = 110;
    colvals[2337] = 0;
    colvals[2338] = 3;
    colvals[2339] = 15;
    colvals[2340] = 19;
    colvals[2341] = 29;
    colvals[2342] = 31;
    colvals[2343] = 36;
    colvals[2344] = 37;
    colvals[2345] = 45;
    colvals[2346] = 46;
    colvals[2347] = 54;
    colvals[2348] = 62;
    colvals[2349] = 64;
    colvals[2350] = 65;
    colvals[2351] = 75;
    colvals[2352] = 83;
    colvals[2353] = 84;
    colvals[2354] = 86;
    colvals[2355] = 87;
    colvals[2356] = 88;
    colvals[2357] = 92;
    colvals[2358] = 93;
    colvals[2359] = 97;
    colvals[2360] = 100;
    colvals[2361] = 101;
    colvals[2362] = 103;
    colvals[2363] = 107;
    colvals[2364] = 108;
    colvals[2365] = 110;
    colvals[2366] = 113;
    colvals[2367] = 114;
    colvals[2368] = 116;
    colvals[2369] = 117;
    colvals[2370] = 118;
    colvals[2371] = 121;
    colvals[2372] = 124;
    colvals[2373] = 125;
    colvals[2374] = 126;
    colvals[2375] = 129;
    colvals[2376] = 1;
    colvals[2377] = 29;
    colvals[2378] = 30;
    colvals[2379] = 36;
    colvals[2380] = 47;
    colvals[2381] = 61;
    colvals[2382] = 64;
    colvals[2383] = 77;
    colvals[2384] = 113;
    colvals[2385] = 114;
    colvals[2386] = 115;
    colvals[2387] = 117;
    colvals[2388] = 118;
    colvals[2389] = 119;
    colvals[2390] = 121;
    colvals[2391] = 123;
    colvals[2392] = 125;
    colvals[2393] = 126;
    colvals[2394] = 127;
    colvals[2395] = 129;
    colvals[2396] = 1;
    colvals[2397] = 34;
    colvals[2398] = 42;
    colvals[2399] = 48;
    colvals[2400] = 49;
    colvals[2401] = 51;
    colvals[2402] = 60;
    colvals[2403] = 62;
    colvals[2404] = 72;
    colvals[2405] = 76;
    colvals[2406] = 98;
    colvals[2407] = 113;
    colvals[2408] = 114;
    colvals[2409] = 116;
    colvals[2410] = 117;
    colvals[2411] = 118;
    colvals[2412] = 121;
    colvals[2413] = 122;
    colvals[2414] = 124;
    colvals[2415] = 125;
    colvals[2416] = 126;
    colvals[2417] = 129;
    colvals[2418] = 130;
    colvals[2419] = 1;
    colvals[2420] = 34;
    colvals[2421] = 42;
    colvals[2422] = 48;
    colvals[2423] = 49;
    colvals[2424] = 51;
    colvals[2425] = 60;
    colvals[2426] = 62;
    colvals[2427] = 72;
    colvals[2428] = 76;
    colvals[2429] = 98;
    colvals[2430] = 113;
    colvals[2431] = 114;
    colvals[2432] = 116;
    colvals[2433] = 117;
    colvals[2434] = 118;
    colvals[2435] = 121;
    colvals[2436] = 122;
    colvals[2437] = 124;
    colvals[2438] = 125;
    colvals[2439] = 126;
    colvals[2440] = 129;
    colvals[2441] = 130;
    colvals[2442] = 0;
    colvals[2443] = 1;
    colvals[2444] = 2;
    colvals[2445] = 3;
    colvals[2446] = 4;
    colvals[2447] = 5;
    colvals[2448] = 7;
    colvals[2449] = 8;
    colvals[2450] = 9;
    colvals[2451] = 12;
    colvals[2452] = 14;
    colvals[2453] = 15;
    colvals[2454] = 16;
    colvals[2455] = 17;
    colvals[2456] = 19;
    colvals[2457] = 20;
    colvals[2458] = 21;
    colvals[2459] = 22;
    colvals[2460] = 23;
    colvals[2461] = 24;
    colvals[2462] = 25;
    colvals[2463] = 26;
    colvals[2464] = 27;
    colvals[2465] = 29;
    colvals[2466] = 30;
    colvals[2467] = 31;
    colvals[2468] = 32;
    colvals[2469] = 33;
    colvals[2470] = 34;
    colvals[2471] = 35;
    colvals[2472] = 36;
    colvals[2473] = 37;
    colvals[2474] = 38;
    colvals[2475] = 39;
    colvals[2476] = 40;
    colvals[2477] = 41;
    colvals[2478] = 45;
    colvals[2479] = 48;
    colvals[2480] = 50;
    colvals[2481] = 51;
    colvals[2482] = 52;
    colvals[2483] = 53;
    colvals[2484] = 54;
    colvals[2485] = 55;
    colvals[2486] = 56;
    colvals[2487] = 57;
    colvals[2488] = 58;
    colvals[2489] = 59;
    colvals[2490] = 60;
    colvals[2491] = 61;
    colvals[2492] = 62;
    colvals[2493] = 63;
    colvals[2494] = 64;
    colvals[2495] = 65;
    colvals[2496] = 66;
    colvals[2497] = 67;
    colvals[2498] = 68;
    colvals[2499] = 69;
    colvals[2500] = 70;
    colvals[2501] = 72;
    colvals[2502] = 73;
    colvals[2503] = 74;
    colvals[2504] = 75;
    colvals[2505] = 76;
    colvals[2506] = 77;
    colvals[2507] = 78;
    colvals[2508] = 80;
    colvals[2509] = 81;
    colvals[2510] = 83;
    colvals[2511] = 84;
    colvals[2512] = 85;
    colvals[2513] = 87;
    colvals[2514] = 88;
    colvals[2515] = 89;
    colvals[2516] = 90;
    colvals[2517] = 91;
    colvals[2518] = 92;
    colvals[2519] = 93;
    colvals[2520] = 95;
    colvals[2521] = 96;
    colvals[2522] = 97;
    colvals[2523] = 98;
    colvals[2524] = 99;
    colvals[2525] = 100;
    colvals[2526] = 101;
    colvals[2527] = 102;
    colvals[2528] = 104;
    colvals[2529] = 105;
    colvals[2530] = 107;
    colvals[2531] = 108;
    colvals[2532] = 109;
    colvals[2533] = 110;
    colvals[2534] = 111;
    colvals[2535] = 112;
    colvals[2536] = 113;
    colvals[2537] = 114;
    colvals[2538] = 115;
    colvals[2539] = 116;
    colvals[2540] = 117;
    colvals[2541] = 118;
    colvals[2542] = 119;
    colvals[2543] = 120;
    colvals[2544] = 121;
    colvals[2545] = 122;
    colvals[2546] = 123;
    colvals[2547] = 124;
    colvals[2548] = 125;
    colvals[2549] = 127;
    colvals[2550] = 128;
    colvals[2551] = 129;
    colvals[2552] = 130;
    colvals[2553] = 2;
    colvals[2554] = 3;
    colvals[2555] = 5;
    colvals[2556] = 7;
    colvals[2557] = 9;
    colvals[2558] = 12;
    colvals[2559] = 14;
    colvals[2560] = 15;
    colvals[2561] = 17;
    colvals[2562] = 19;
    colvals[2563] = 20;
    colvals[2564] = 21;
    colvals[2565] = 23;
    colvals[2566] = 26;
    colvals[2567] = 30;
    colvals[2568] = 31;
    colvals[2569] = 32;
    colvals[2570] = 33;
    colvals[2571] = 34;
    colvals[2572] = 35;
    colvals[2573] = 36;
    colvals[2574] = 39;
    colvals[2575] = 40;
    colvals[2576] = 41;
    colvals[2577] = 43;
    colvals[2578] = 45;
    colvals[2579] = 48;
    colvals[2580] = 50;
    colvals[2581] = 51;
    colvals[2582] = 52;
    colvals[2583] = 54;
    colvals[2584] = 57;
    colvals[2585] = 58;
    colvals[2586] = 59;
    colvals[2587] = 61;
    colvals[2588] = 62;
    colvals[2589] = 64;
    colvals[2590] = 66;
    colvals[2591] = 68;
    colvals[2592] = 72;
    colvals[2593] = 78;
    colvals[2594] = 83;
    colvals[2595] = 85;
    colvals[2596] = 87;
    colvals[2597] = 89;
    colvals[2598] = 91;
    colvals[2599] = 93;
    colvals[2600] = 97;
    colvals[2601] = 98;
    colvals[2602] = 100;
    colvals[2603] = 107;
    colvals[2604] = 110;
    colvals[2605] = 111;
    colvals[2606] = 113;
    colvals[2607] = 115;
    colvals[2608] = 116;
    colvals[2609] = 117;
    colvals[2610] = 119;
    colvals[2611] = 120;
    colvals[2612] = 121;
    colvals[2613] = 122;
    colvals[2614] = 123;
    colvals[2615] = 124;
    colvals[2616] = 125;
    colvals[2617] = 127;
    colvals[2618] = 128;
    colvals[2619] = 129;
    colvals[2620] = 130;
    colvals[2621] = 0;
    colvals[2622] = 1;
    colvals[2623] = 3;
    colvals[2624] = 15;
    colvals[2625] = 19;
    colvals[2626] = 25;
    colvals[2627] = 29;
    colvals[2628] = 33;
    colvals[2629] = 34;
    colvals[2630] = 36;
    colvals[2631] = 38;
    colvals[2632] = 39;
    colvals[2633] = 42;
    colvals[2634] = 50;
    colvals[2635] = 51;
    colvals[2636] = 52;
    colvals[2637] = 53;
    colvals[2638] = 54;
    colvals[2639] = 56;
    colvals[2640] = 57;
    colvals[2641] = 60;
    colvals[2642] = 61;
    colvals[2643] = 62;
    colvals[2644] = 63;
    colvals[2645] = 64;
    colvals[2646] = 72;
    colvals[2647] = 75;
    colvals[2648] = 76;
    colvals[2649] = 83;
    colvals[2650] = 87;
    colvals[2651] = 97;
    colvals[2652] = 98;
    colvals[2653] = 107;
    colvals[2654] = 110;
    colvals[2655] = 113;
    colvals[2656] = 116;
    colvals[2657] = 117;
    colvals[2658] = 119;
    colvals[2659] = 120;
    colvals[2660] = 121;
    colvals[2661] = 122;
    colvals[2662] = 124;
    colvals[2663] = 125;
    colvals[2664] = 127;
    colvals[2665] = 128;
    colvals[2666] = 129;
    colvals[2667] = 130;
    colvals[2668] = 0;
    colvals[2669] = 2;
    colvals[2670] = 15;
    colvals[2671] = 16;
    colvals[2672] = 19;
    colvals[2673] = 20;
    colvals[2674] = 27;
    colvals[2675] = 33;
    colvals[2676] = 35;
    colvals[2677] = 36;
    colvals[2678] = 37;
    colvals[2679] = 38;
    colvals[2680] = 40;
    colvals[2681] = 41;
    colvals[2682] = 42;
    colvals[2683] = 44;
    colvals[2684] = 46;
    colvals[2685] = 50;
    colvals[2686] = 52;
    colvals[2687] = 53;
    colvals[2688] = 54;
    colvals[2689] = 55;
    colvals[2690] = 56;
    colvals[2691] = 58;
    colvals[2692] = 59;
    colvals[2693] = 60;
    colvals[2694] = 61;
    colvals[2695] = 62;
    colvals[2696] = 63;
    colvals[2697] = 64;
    colvals[2698] = 65;
    colvals[2699] = 67;
    colvals[2700] = 69;
    colvals[2701] = 79;
    colvals[2702] = 80;
    colvals[2703] = 83;
    colvals[2704] = 84;
    colvals[2705] = 86;
    colvals[2706] = 87;
    colvals[2707] = 88;
    colvals[2708] = 90;
    colvals[2709] = 92;
    colvals[2710] = 99;
    colvals[2711] = 107;
    colvals[2712] = 108;
    colvals[2713] = 109;
    colvals[2714] = 110;
    colvals[2715] = 111;
    colvals[2716] = 112;
    colvals[2717] = 113;
    colvals[2718] = 114;
    colvals[2719] = 115;
    colvals[2720] = 116;
    colvals[2721] = 117;
    colvals[2722] = 118;
    colvals[2723] = 119;
    colvals[2724] = 120;
    colvals[2725] = 121;
    colvals[2726] = 122;
    colvals[2727] = 123;
    colvals[2728] = 124;
    colvals[2729] = 125;
    colvals[2730] = 126;
    colvals[2731] = 127;
    colvals[2732] = 128;
    colvals[2733] = 129;
    colvals[2734] = 130;
    colvals[2735] = 1;
    colvals[2736] = 2;
    colvals[2737] = 3;
    colvals[2738] = 4;
    colvals[2739] = 5;
    colvals[2740] = 7;
    colvals[2741] = 10;
    colvals[2742] = 15;
    colvals[2743] = 16;
    colvals[2744] = 17;
    colvals[2745] = 19;
    colvals[2746] = 20;
    colvals[2747] = 21;
    colvals[2748] = 23;
    colvals[2749] = 26;
    colvals[2750] = 27;
    colvals[2751] = 28;
    colvals[2752] = 30;
    colvals[2753] = 31;
    colvals[2754] = 32;
    colvals[2755] = 33;
    colvals[2756] = 34;
    colvals[2757] = 35;
    colvals[2758] = 37;
    colvals[2759] = 40;
    colvals[2760] = 41;
    colvals[2761] = 42;
    colvals[2762] = 44;
    colvals[2763] = 46;
    colvals[2764] = 50;
    colvals[2765] = 51;
    colvals[2766] = 52;
    colvals[2767] = 53;
    colvals[2768] = 54;
    colvals[2769] = 55;
    colvals[2770] = 56;
    colvals[2771] = 57;
    colvals[2772] = 58;
    colvals[2773] = 59;
    colvals[2774] = 60;
    colvals[2775] = 61;
    colvals[2776] = 62;
    colvals[2777] = 63;
    colvals[2778] = 65;
    colvals[2779] = 67;
    colvals[2780] = 68;
    colvals[2781] = 69;
    colvals[2782] = 72;
    colvals[2783] = 76;
    colvals[2784] = 78;
    colvals[2785] = 79;
    colvals[2786] = 80;
    colvals[2787] = 84;
    colvals[2788] = 85;
    colvals[2789] = 86;
    colvals[2790] = 88;
    colvals[2791] = 89;
    colvals[2792] = 90;
    colvals[2793] = 91;
    colvals[2794] = 92;
    colvals[2795] = 93;
    colvals[2796] = 96;
    colvals[2797] = 98;
    colvals[2798] = 99;
    colvals[2799] = 100;
    colvals[2800] = 104;
    colvals[2801] = 107;
    colvals[2802] = 108;
    colvals[2803] = 109;
    colvals[2804] = 110;
    colvals[2805] = 111;
    colvals[2806] = 112;
    colvals[2807] = 113;
    colvals[2808] = 114;
    colvals[2809] = 116;
    colvals[2810] = 117;
    colvals[2811] = 118;
    colvals[2812] = 119;
    colvals[2813] = 120;
    colvals[2814] = 121;
    colvals[2815] = 122;
    colvals[2816] = 124;
    colvals[2817] = 125;
    colvals[2818] = 126;
    colvals[2819] = 127;
    colvals[2820] = 128;
    colvals[2821] = 129;
    colvals[2822] = 130;
    colvals[2823] = 0;
    colvals[2824] = 3;
    colvals[2825] = 5;
    colvals[2826] = 7;
    colvals[2827] = 15;
    colvals[2828] = 17;
    colvals[2829] = 19;
    colvals[2830] = 21;
    colvals[2831] = 23;
    colvals[2832] = 29;
    colvals[2833] = 30;
    colvals[2834] = 32;
    colvals[2835] = 33;
    colvals[2836] = 34;
    colvals[2837] = 36;
    colvals[2838] = 40;
    colvals[2839] = 41;
    colvals[2840] = 50;
    colvals[2841] = 51;
    colvals[2842] = 53;
    colvals[2843] = 54;
    colvals[2844] = 55;
    colvals[2845] = 56;
    colvals[2846] = 58;
    colvals[2847] = 59;
    colvals[2848] = 60;
    colvals[2849] = 61;
    colvals[2850] = 62;
    colvals[2851] = 63;
    colvals[2852] = 64;
    colvals[2853] = 67;
    colvals[2854] = 69;
    colvals[2855] = 72;
    colvals[2856] = 75;
    colvals[2857] = 76;
    colvals[2858] = 78;
    colvals[2859] = 80;
    colvals[2860] = 83;
    colvals[2861] = 84;
    colvals[2862] = 85;
    colvals[2863] = 87;
    colvals[2864] = 88;
    colvals[2865] = 89;
    colvals[2866] = 91;
    colvals[2867] = 93;
    colvals[2868] = 97;
    colvals[2869] = 98;
    colvals[2870] = 100;
    colvals[2871] = 105;
    colvals[2872] = 107;
    colvals[2873] = 108;
    colvals[2874] = 110;
    colvals[2875] = 111;
    colvals[2876] = 113;
    colvals[2877] = 115;
    colvals[2878] = 116;
    colvals[2879] = 117;
    colvals[2880] = 119;
    colvals[2881] = 120;
    colvals[2882] = 121;
    colvals[2883] = 122;
    colvals[2884] = 123;
    colvals[2885] = 124;
    colvals[2886] = 125;
    colvals[2887] = 127;
    colvals[2888] = 128;
    colvals[2889] = 129;
    colvals[2890] = 130;
    colvals[2891] = 0;
    colvals[2892] = 2;
    colvals[2893] = 15;
    colvals[2894] = 19;
    colvals[2895] = 20;
    colvals[2896] = 27;
    colvals[2897] = 33;
    colvals[2898] = 35;
    colvals[2899] = 36;
    colvals[2900] = 50;
    colvals[2901] = 52;
    colvals[2902] = 53;
    colvals[2903] = 54;
    colvals[2904] = 55;
    colvals[2905] = 56;
    colvals[2906] = 58;
    colvals[2907] = 59;
    colvals[2908] = 60;
    colvals[2909] = 61;
    colvals[2910] = 62;
    colvals[2911] = 63;
    colvals[2912] = 64;
    colvals[2913] = 65;
    colvals[2914] = 67;
    colvals[2915] = 69;
    colvals[2916] = 80;
    colvals[2917] = 87;
    colvals[2918] = 88;
    colvals[2919] = 90;
    colvals[2920] = 92;
    colvals[2921] = 99;
    colvals[2922] = 109;
    colvals[2923] = 110;
    colvals[2924] = 111;
    colvals[2925] = 112;
    colvals[2926] = 113;
    colvals[2927] = 115;
    colvals[2928] = 117;
    colvals[2929] = 119;
    colvals[2930] = 120;
    colvals[2931] = 121;
    colvals[2932] = 122;
    colvals[2933] = 123;
    colvals[2934] = 125;
    colvals[2935] = 127;
    colvals[2936] = 128;
    colvals[2937] = 129;
    colvals[2938] = 130;
    colvals[2939] = 0;
    colvals[2940] = 1;
    colvals[2941] = 2;
    colvals[2942] = 10;
    colvals[2943] = 16;
    colvals[2944] = 19;
    colvals[2945] = 20;
    colvals[2946] = 21;
    colvals[2947] = 23;
    colvals[2948] = 25;
    colvals[2949] = 26;
    colvals[2950] = 27;
    colvals[2951] = 28;
    colvals[2952] = 30;
    colvals[2953] = 33;
    colvals[2954] = 34;
    colvals[2955] = 35;
    colvals[2956] = 36;
    colvals[2957] = 43;
    colvals[2958] = 50;
    colvals[2959] = 51;
    colvals[2960] = 52;
    colvals[2961] = 54;
    colvals[2962] = 57;
    colvals[2963] = 58;
    colvals[2964] = 59;
    colvals[2965] = 62;
    colvals[2966] = 64;
    colvals[2967] = 66;
    colvals[2968] = 68;
    colvals[2969] = 72;
    colvals[2970] = 75;
    colvals[2971] = 76;
    colvals[2972] = 78;
    colvals[2973] = 87;
    colvals[2974] = 89;
    colvals[2975] = 91;
    colvals[2976] = 93;
    colvals[2977] = 97;
    colvals[2978] = 98;
    colvals[2979] = 99;
    colvals[2980] = 100;
    colvals[2981] = 109;
    colvals[2982] = 110;
    colvals[2983] = 112;
    colvals[2984] = 116;
    colvals[2985] = 120;
    colvals[2986] = 124;
    colvals[2987] = 128;
    colvals[2988] = 0;
    colvals[2989] = 1;
    colvals[2990] = 3;
    colvals[2991] = 15;
    colvals[2992] = 16;
    colvals[2993] = 19;
    colvals[2994] = 20;
    colvals[2995] = 22;
    colvals[2996] = 24;
    colvals[2997] = 25;
    colvals[2998] = 26;
    colvals[2999] = 29;
    colvals[3000] = 30;
    colvals[3001] = 33;
    colvals[3002] = 34;
    colvals[3003] = 36;
    colvals[3004] = 50;
    colvals[3005] = 51;
    colvals[3006] = 54;
    colvals[3007] = 57;
    colvals[3008] = 58;
    colvals[3009] = 59;
    colvals[3010] = 61;
    colvals[3011] = 62;
    colvals[3012] = 64;
    colvals[3013] = 75;
    colvals[3014] = 76;
    colvals[3015] = 78;
    colvals[3016] = 83;
    colvals[3017] = 85;
    colvals[3018] = 87;
    colvals[3019] = 88;
    colvals[3020] = 89;
    colvals[3021] = 91;
    colvals[3022] = 93;
    colvals[3023] = 97;
    colvals[3024] = 100;
    colvals[3025] = 105;
    colvals[3026] = 107;
    colvals[3027] = 110;
    colvals[3028] = 111;
    colvals[3029] = 113;
    colvals[3030] = 116;
    colvals[3031] = 117;
    colvals[3032] = 119;
    colvals[3033] = 120;
    colvals[3034] = 121;
    colvals[3035] = 122;
    colvals[3036] = 124;
    colvals[3037] = 125;
    colvals[3038] = 127;
    colvals[3039] = 128;
    colvals[3040] = 129;
    colvals[3041] = 130;
    colvals[3042] = 0;
    colvals[3043] = 1;
    colvals[3044] = 2;
    colvals[3045] = 4;
    colvals[3046] = 7;
    colvals[3047] = 16;
    colvals[3048] = 19;
    colvals[3049] = 20;
    colvals[3050] = 25;
    colvals[3051] = 26;
    colvals[3052] = 29;
    colvals[3053] = 30;
    colvals[3054] = 31;
    colvals[3055] = 33;
    colvals[3056] = 34;
    colvals[3057] = 37;
    colvals[3058] = 50;
    colvals[3059] = 51;
    colvals[3060] = 52;
    colvals[3061] = 55;
    colvals[3062] = 59;
    colvals[3063] = 62;
    colvals[3064] = 65;
    colvals[3065] = 67;
    colvals[3066] = 72;
    colvals[3067] = 75;
    colvals[3068] = 76;
    colvals[3069] = 78;
    colvals[3070] = 86;
    colvals[3071] = 88;
    colvals[3072] = 90;
    colvals[3073] = 92;
    colvals[3074] = 97;
    colvals[3075] = 98;
    colvals[3076] = 99;
    colvals[3077] = 100;
    colvals[3078] = 101;
    colvals[3079] = 107;
    colvals[3080] = 108;
    colvals[3081] = 110;
    colvals[3082] = 111;
    colvals[3083] = 112;
    colvals[3084] = 116;
    colvals[3085] = 120;
    colvals[3086] = 124;
    colvals[3087] = 128;
    colvals[3088] = 0;
    colvals[3089] = 1;
    colvals[3090] = 3;
    colvals[3091] = 4;
    colvals[3092] = 7;
    colvals[3093] = 8;
    colvals[3094] = 10;
    colvals[3095] = 15;
    colvals[3096] = 16;
    colvals[3097] = 19;
    colvals[3098] = 20;
    colvals[3099] = 21;
    colvals[3100] = 22;
    colvals[3101] = 23;
    colvals[3102] = 24;
    colvals[3103] = 26;
    colvals[3104] = 28;
    colvals[3105] = 29;
    colvals[3106] = 30;
    colvals[3107] = 31;
    colvals[3108] = 32;
    colvals[3109] = 34;
    colvals[3110] = 35;
    colvals[3111] = 36;
    colvals[3112] = 37;
    colvals[3113] = 47;
    colvals[3114] = 48;
    colvals[3115] = 50;
    colvals[3116] = 51;
    colvals[3117] = 52;
    colvals[3118] = 53;
    colvals[3119] = 54;
    colvals[3120] = 55;
    colvals[3121] = 56;
    colvals[3122] = 57;
    colvals[3123] = 58;
    colvals[3124] = 59;
    colvals[3125] = 60;
    colvals[3126] = 61;
    colvals[3127] = 62;
    colvals[3128] = 63;
    colvals[3129] = 64;
    colvals[3130] = 65;
    colvals[3131] = 69;
    colvals[3132] = 70;
    colvals[3133] = 76;
    colvals[3134] = 78;
    colvals[3135] = 80;
    colvals[3136] = 83;
    colvals[3137] = 86;
    colvals[3138] = 87;
    colvals[3139] = 88;
    colvals[3140] = 89;
    colvals[3141] = 90;
    colvals[3142] = 91;
    colvals[3143] = 92;
    colvals[3144] = 97;
    colvals[3145] = 98;
    colvals[3146] = 100;
    colvals[3147] = 101;
    colvals[3148] = 105;
    colvals[3149] = 107;
    colvals[3150] = 108;
    colvals[3151] = 110;
    colvals[3152] = 111;
    colvals[3153] = 113;
    colvals[3154] = 115;
    colvals[3155] = 116;
    colvals[3156] = 117;
    colvals[3157] = 119;
    colvals[3158] = 120;
    colvals[3159] = 121;
    colvals[3160] = 122;
    colvals[3161] = 123;
    colvals[3162] = 124;
    colvals[3163] = 125;
    colvals[3164] = 127;
    colvals[3165] = 128;
    colvals[3166] = 129;
    colvals[3167] = 130;
    colvals[3168] = 0;
    colvals[3169] = 1;
    colvals[3170] = 2;
    colvals[3171] = 3;
    colvals[3172] = 4;
    colvals[3173] = 5;
    colvals[3174] = 7;
    colvals[3175] = 15;
    colvals[3176] = 16;
    colvals[3177] = 17;
    colvals[3178] = 19;
    colvals[3179] = 20;
    colvals[3180] = 21;
    colvals[3181] = 23;
    colvals[3182] = 24;
    colvals[3183] = 25;
    colvals[3184] = 26;
    colvals[3185] = 29;
    colvals[3186] = 30;
    colvals[3187] = 31;
    colvals[3188] = 33;
    colvals[3189] = 34;
    colvals[3190] = 35;
    colvals[3191] = 36;
    colvals[3192] = 37;
    colvals[3193] = 38;
    colvals[3194] = 39;
    colvals[3195] = 41;
    colvals[3196] = 42;
    colvals[3197] = 45;
    colvals[3198] = 47;
    colvals[3199] = 48;
    colvals[3200] = 50;
    colvals[3201] = 51;
    colvals[3202] = 52;
    colvals[3203] = 53;
    colvals[3204] = 54;
    colvals[3205] = 55;
    colvals[3206] = 56;
    colvals[3207] = 57;
    colvals[3208] = 59;
    colvals[3209] = 60;
    colvals[3210] = 61;
    colvals[3211] = 62;
    colvals[3212] = 63;
    colvals[3213] = 64;
    colvals[3214] = 65;
    colvals[3215] = 68;
    colvals[3216] = 70;
    colvals[3217] = 72;
    colvals[3218] = 73;
    colvals[3219] = 74;
    colvals[3220] = 75;
    colvals[3221] = 76;
    colvals[3222] = 77;
    colvals[3223] = 78;
    colvals[3224] = 83;
    colvals[3225] = 84;
    colvals[3226] = 85;
    colvals[3227] = 87;
    colvals[3228] = 88;
    colvals[3229] = 89;
    colvals[3230] = 91;
    colvals[3231] = 93;
    colvals[3232] = 95;
    colvals[3233] = 96;
    colvals[3234] = 97;
    colvals[3235] = 98;
    colvals[3236] = 99;
    colvals[3237] = 100;
    colvals[3238] = 102;
    colvals[3239] = 103;
    colvals[3240] = 104;
    colvals[3241] = 105;
    colvals[3242] = 107;
    colvals[3243] = 108;
    colvals[3244] = 110;
    colvals[3245] = 111;
    colvals[3246] = 113;
    colvals[3247] = 114;
    colvals[3248] = 115;
    colvals[3249] = 116;
    colvals[3250] = 117;
    colvals[3251] = 118;
    colvals[3252] = 119;
    colvals[3253] = 120;
    colvals[3254] = 121;
    colvals[3255] = 122;
    colvals[3256] = 123;
    colvals[3257] = 124;
    colvals[3258] = 125;
    colvals[3259] = 126;
    colvals[3260] = 127;
    colvals[3261] = 128;
    colvals[3262] = 129;
    colvals[3263] = 130;
    colvals[3264] = 0;
    colvals[3265] = 3;
    colvals[3266] = 5;
    colvals[3267] = 7;
    colvals[3268] = 15;
    colvals[3269] = 17;
    colvals[3270] = 19;
    colvals[3271] = 21;
    colvals[3272] = 23;
    colvals[3273] = 25;
    colvals[3274] = 29;
    colvals[3275] = 31;
    colvals[3276] = 33;
    colvals[3277] = 34;
    colvals[3278] = 35;
    colvals[3279] = 36;
    colvals[3280] = 39;
    colvals[3281] = 41;
    colvals[3282] = 48;
    colvals[3283] = 50;
    colvals[3284] = 51;
    colvals[3285] = 52;
    colvals[3286] = 54;
    colvals[3287] = 57;
    colvals[3288] = 59;
    colvals[3289] = 61;
    colvals[3290] = 62;
    colvals[3291] = 64;
    colvals[3292] = 72;
    colvals[3293] = 73;
    colvals[3294] = 74;
    colvals[3295] = 75;
    colvals[3296] = 77;
    colvals[3297] = 83;
    colvals[3298] = 85;
    colvals[3299] = 87;
    colvals[3300] = 89;
    colvals[3301] = 91;
    colvals[3302] = 93;
    colvals[3303] = 97;
    colvals[3304] = 100;
    colvals[3305] = 107;
    colvals[3306] = 110;
    colvals[3307] = 113;
    colvals[3308] = 115;
    colvals[3309] = 116;
    colvals[3310] = 117;
    colvals[3311] = 119;
    colvals[3312] = 121;
    colvals[3313] = 123;
    colvals[3314] = 124;
    colvals[3315] = 125;
    colvals[3316] = 127;
    colvals[3317] = 128;
    colvals[3318] = 129;
    colvals[3319] = 0;
    colvals[3320] = 2;
    colvals[3321] = 15;
    colvals[3322] = 16;
    colvals[3323] = 19;
    colvals[3324] = 20;
    colvals[3325] = 27;
    colvals[3326] = 33;
    colvals[3327] = 35;
    colvals[3328] = 36;
    colvals[3329] = 37;
    colvals[3330] = 38;
    colvals[3331] = 40;
    colvals[3332] = 41;
    colvals[3333] = 42;
    colvals[3334] = 44;
    colvals[3335] = 46;
    colvals[3336] = 50;
    colvals[3337] = 52;
    colvals[3338] = 53;
    colvals[3339] = 54;
    colvals[3340] = 55;
    colvals[3341] = 56;
    colvals[3342] = 58;
    colvals[3343] = 59;
    colvals[3344] = 60;
    colvals[3345] = 61;
    colvals[3346] = 62;
    colvals[3347] = 63;
    colvals[3348] = 64;
    colvals[3349] = 65;
    colvals[3350] = 67;
    colvals[3351] = 69;
    colvals[3352] = 79;
    colvals[3353] = 80;
    colvals[3354] = 83;
    colvals[3355] = 84;
    colvals[3356] = 86;
    colvals[3357] = 87;
    colvals[3358] = 88;
    colvals[3359] = 90;
    colvals[3360] = 92;
    colvals[3361] = 99;
    colvals[3362] = 107;
    colvals[3363] = 108;
    colvals[3364] = 109;
    colvals[3365] = 110;
    colvals[3366] = 111;
    colvals[3367] = 112;
    colvals[3368] = 113;
    colvals[3369] = 114;
    colvals[3370] = 115;
    colvals[3371] = 116;
    colvals[3372] = 117;
    colvals[3373] = 118;
    colvals[3374] = 119;
    colvals[3375] = 120;
    colvals[3376] = 121;
    colvals[3377] = 122;
    colvals[3378] = 123;
    colvals[3379] = 124;
    colvals[3380] = 125;
    colvals[3381] = 126;
    colvals[3382] = 127;
    colvals[3383] = 128;
    colvals[3384] = 129;
    colvals[3385] = 130;
    colvals[3386] = 1;
    colvals[3387] = 2;
    colvals[3388] = 3;
    colvals[3389] = 4;
    colvals[3390] = 5;
    colvals[3391] = 7;
    colvals[3392] = 10;
    colvals[3393] = 15;
    colvals[3394] = 16;
    colvals[3395] = 17;
    colvals[3396] = 19;
    colvals[3397] = 20;
    colvals[3398] = 21;
    colvals[3399] = 23;
    colvals[3400] = 26;
    colvals[3401] = 27;
    colvals[3402] = 28;
    colvals[3403] = 30;
    colvals[3404] = 31;
    colvals[3405] = 32;
    colvals[3406] = 33;
    colvals[3407] = 34;
    colvals[3408] = 35;
    colvals[3409] = 37;
    colvals[3410] = 38;
    colvals[3411] = 39;
    colvals[3412] = 40;
    colvals[3413] = 41;
    colvals[3414] = 42;
    colvals[3415] = 44;
    colvals[3416] = 45;
    colvals[3417] = 46;
    colvals[3418] = 50;
    colvals[3419] = 51;
    colvals[3420] = 52;
    colvals[3421] = 53;
    colvals[3422] = 55;
    colvals[3423] = 56;
    colvals[3424] = 57;
    colvals[3425] = 58;
    colvals[3426] = 59;
    colvals[3427] = 60;
    colvals[3428] = 61;
    colvals[3429] = 62;
    colvals[3430] = 63;
    colvals[3431] = 64;
    colvals[3432] = 65;
    colvals[3433] = 67;
    colvals[3434] = 68;
    colvals[3435] = 69;
    colvals[3436] = 72;
    colvals[3437] = 76;
    colvals[3438] = 78;
    colvals[3439] = 79;
    colvals[3440] = 80;
    colvals[3441] = 84;
    colvals[3442] = 85;
    colvals[3443] = 86;
    colvals[3444] = 88;
    colvals[3445] = 89;
    colvals[3446] = 90;
    colvals[3447] = 91;
    colvals[3448] = 92;
    colvals[3449] = 93;
    colvals[3450] = 96;
    colvals[3451] = 98;
    colvals[3452] = 99;
    colvals[3453] = 100;
    colvals[3454] = 102;
    colvals[3455] = 104;
    colvals[3456] = 107;
    colvals[3457] = 108;
    colvals[3458] = 109;
    colvals[3459] = 110;
    colvals[3460] = 111;
    colvals[3461] = 112;
    colvals[3462] = 113;
    colvals[3463] = 114;
    colvals[3464] = 115;
    colvals[3465] = 116;
    colvals[3466] = 117;
    colvals[3467] = 118;
    colvals[3468] = 119;
    colvals[3469] = 120;
    colvals[3470] = 121;
    colvals[3471] = 122;
    colvals[3472] = 123;
    colvals[3473] = 124;
    colvals[3474] = 125;
    colvals[3475] = 126;
    colvals[3476] = 127;
    colvals[3477] = 128;
    colvals[3478] = 129;
    colvals[3479] = 130;
    colvals[3480] = 0;
    colvals[3481] = 3;
    colvals[3482] = 5;
    colvals[3483] = 7;
    colvals[3484] = 15;
    colvals[3485] = 17;
    colvals[3486] = 19;
    colvals[3487] = 21;
    colvals[3488] = 23;
    colvals[3489] = 29;
    colvals[3490] = 30;
    colvals[3491] = 32;
    colvals[3492] = 33;
    colvals[3493] = 34;
    colvals[3494] = 36;
    colvals[3495] = 38;
    colvals[3496] = 40;
    colvals[3497] = 41;
    colvals[3498] = 42;
    colvals[3499] = 44;
    colvals[3500] = 46;
    colvals[3501] = 50;
    colvals[3502] = 51;
    colvals[3503] = 53;
    colvals[3504] = 54;
    colvals[3505] = 56;
    colvals[3506] = 58;
    colvals[3507] = 59;
    colvals[3508] = 60;
    colvals[3509] = 61;
    colvals[3510] = 62;
    colvals[3511] = 63;
    colvals[3512] = 64;
    colvals[3513] = 65;
    colvals[3514] = 67;
    colvals[3515] = 69;
    colvals[3516] = 72;
    colvals[3517] = 75;
    colvals[3518] = 76;
    colvals[3519] = 78;
    colvals[3520] = 79;
    colvals[3521] = 80;
    colvals[3522] = 83;
    colvals[3523] = 84;
    colvals[3524] = 85;
    colvals[3525] = 87;
    colvals[3526] = 88;
    colvals[3527] = 89;
    colvals[3528] = 91;
    colvals[3529] = 93;
    colvals[3530] = 97;
    colvals[3531] = 98;
    colvals[3532] = 100;
    colvals[3533] = 103;
    colvals[3534] = 105;
    colvals[3535] = 107;
    colvals[3536] = 108;
    colvals[3537] = 110;
    colvals[3538] = 111;
    colvals[3539] = 113;
    colvals[3540] = 115;
    colvals[3541] = 116;
    colvals[3542] = 117;
    colvals[3543] = 119;
    colvals[3544] = 120;
    colvals[3545] = 121;
    colvals[3546] = 122;
    colvals[3547] = 123;
    colvals[3548] = 124;
    colvals[3549] = 125;
    colvals[3550] = 127;
    colvals[3551] = 128;
    colvals[3552] = 129;
    colvals[3553] = 130;
    colvals[3554] = 0;
    colvals[3555] = 1;
    colvals[3556] = 21;
    colvals[3557] = 23;
    colvals[3558] = 34;
    colvals[3559] = 51;
    colvals[3560] = 66;
    colvals[3561] = 67;
    colvals[3562] = 72;
    colvals[3563] = 75;
    colvals[3564] = 89;
    colvals[3565] = 91;
    colvals[3566] = 93;
    colvals[3567] = 0;
    colvals[3568] = 3;
    colvals[3569] = 15;
    colvals[3570] = 19;
    colvals[3571] = 26;
    colvals[3572] = 36;
    colvals[3573] = 54;
    colvals[3574] = 61;
    colvals[3575] = 64;
    colvals[3576] = 67;
    colvals[3577] = 83;
    colvals[3578] = 87;
    colvals[3579] = 93;
    colvals[3580] = 100;
    colvals[3581] = 107;
    colvals[3582] = 110;
    colvals[3583] = 113;
    colvals[3584] = 119;
    colvals[3585] = 127;
    colvals[3586] = 15;
    colvals[3587] = 19;
    colvals[3588] = 25;
    colvals[3589] = 29;
    colvals[3590] = 33;
    colvals[3591] = 34;
    colvals[3592] = 50;
    colvals[3593] = 51;
    colvals[3594] = 68;
    colvals[3595] = 69;
    colvals[3596] = 72;
    colvals[3597] = 89;
    colvals[3598] = 91;
    colvals[3599] = 93;
    colvals[3600] = 97;
    colvals[3601] = 107;
    colvals[3602] = 110;
    colvals[3603] = 0;
    colvals[3604] = 3;
    colvals[3605] = 15;
    colvals[3606] = 19;
    colvals[3607] = 29;
    colvals[3608] = 31;
    colvals[3609] = 36;
    colvals[3610] = 54;
    colvals[3611] = 55;
    colvals[3612] = 62;
    colvals[3613] = 64;
    colvals[3614] = 65;
    colvals[3615] = 68;
    colvals[3616] = 69;
    colvals[3617] = 75;
    colvals[3618] = 83;
    colvals[3619] = 84;
    colvals[3620] = 87;
    colvals[3621] = 88;
    colvals[3622] = 90;
    colvals[3623] = 92;
    colvals[3624] = 93;
    colvals[3625] = 97;
    colvals[3626] = 100;
    colvals[3627] = 101;
    colvals[3628] = 105;
    colvals[3629] = 107;
    colvals[3630] = 110;
    colvals[3631] = 111;
    colvals[3632] = 113;
    colvals[3633] = 117;
    colvals[3634] = 120;
    colvals[3635] = 121;
    colvals[3636] = 122;
    colvals[3637] = 125;
    colvals[3638] = 128;
    colvals[3639] = 129;
    colvals[3640] = 130;
    colvals[3641] = 1;
    colvals[3642] = 29;
    colvals[3643] = 30;
    colvals[3644] = 54;
    colvals[3645] = 61;
    colvals[3646] = 64;
    colvals[3647] = 70;
    colvals[3648] = 77;
    colvals[3649] = 113;
    colvals[3650] = 115;
    colvals[3651] = 117;
    colvals[3652] = 119;
    colvals[3653] = 121;
    colvals[3654] = 122;
    colvals[3655] = 123;
    colvals[3656] = 125;
    colvals[3657] = 127;
    colvals[3658] = 129;
    colvals[3659] = 130;
    colvals[3660] = 2;
    colvals[3661] = 3;
    colvals[3662] = 5;
    colvals[3663] = 7;
    colvals[3664] = 9;
    colvals[3665] = 12;
    colvals[3666] = 14;
    colvals[3667] = 15;
    colvals[3668] = 17;
    colvals[3669] = 19;
    colvals[3670] = 21;
    colvals[3671] = 23;
    colvals[3672] = 25;
    colvals[3673] = 29;
    colvals[3674] = 31;
    colvals[3675] = 33;
    colvals[3676] = 35;
    colvals[3677] = 36;
    colvals[3678] = 39;
    colvals[3679] = 41;
    colvals[3680] = 43;
    colvals[3681] = 45;
    colvals[3682] = 48;
    colvals[3683] = 50;
    colvals[3684] = 52;
    colvals[3685] = 54;
    colvals[3686] = 57;
    colvals[3687] = 59;
    colvals[3688] = 61;
    colvals[3689] = 64;
    colvals[3690] = 66;
    colvals[3691] = 68;
    colvals[3692] = 71;
    colvals[3693] = 72;
    colvals[3694] = 73;
    colvals[3695] = 74;
    colvals[3696] = 77;
    colvals[3697] = 81;
    colvals[3698] = 83;
    colvals[3699] = 85;
    colvals[3700] = 87;
    colvals[3701] = 89;
    colvals[3702] = 91;
    colvals[3703] = 93;
    colvals[3704] = 100;
    colvals[3705] = 106;
    colvals[3706] = 107;
    colvals[3707] = 110;
    colvals[3708] = 113;
    colvals[3709] = 115;
    colvals[3710] = 119;
    colvals[3711] = 123;
    colvals[3712] = 127;
    colvals[3713] = 2;
    colvals[3714] = 3;
    colvals[3715] = 5;
    colvals[3716] = 7;
    colvals[3717] = 9;
    colvals[3718] = 12;
    colvals[3719] = 14;
    colvals[3720] = 15;
    colvals[3721] = 17;
    colvals[3722] = 19;
    colvals[3723] = 21;
    colvals[3724] = 23;
    colvals[3725] = 25;
    colvals[3726] = 29;
    colvals[3727] = 31;
    colvals[3728] = 33;
    colvals[3729] = 35;
    colvals[3730] = 36;
    colvals[3731] = 39;
    colvals[3732] = 41;
    colvals[3733] = 43;
    colvals[3734] = 45;
    colvals[3735] = 48;
    colvals[3736] = 50;
    colvals[3737] = 52;
    colvals[3738] = 54;
    colvals[3739] = 57;
    colvals[3740] = 59;
    colvals[3741] = 61;
    colvals[3742] = 64;
    colvals[3743] = 66;
    colvals[3744] = 68;
    colvals[3745] = 71;
    colvals[3746] = 72;
    colvals[3747] = 77;
    colvals[3748] = 81;
    colvals[3749] = 83;
    colvals[3750] = 85;
    colvals[3751] = 87;
    colvals[3752] = 89;
    colvals[3753] = 91;
    colvals[3754] = 93;
    colvals[3755] = 100;
    colvals[3756] = 106;
    colvals[3757] = 107;
    colvals[3758] = 110;
    colvals[3759] = 113;
    colvals[3760] = 115;
    colvals[3761] = 119;
    colvals[3762] = 123;
    colvals[3763] = 127;
    colvals[3764] = 33;
    colvals[3765] = 41;
    colvals[3766] = 50;
    colvals[3767] = 61;
    colvals[3768] = 72;
    colvals[3769] = 73;
    colvals[3770] = 113;
    colvals[3771] = 115;
    colvals[3772] = 119;
    colvals[3773] = 123;
    colvals[3774] = 127;
    colvals[3775] = 33;
    colvals[3776] = 50;
    colvals[3777] = 59;
    colvals[3778] = 61;
    colvals[3779] = 72;
    colvals[3780] = 74;
    colvals[3781] = 113;
    colvals[3782] = 115;
    colvals[3783] = 119;
    colvals[3784] = 123;
    colvals[3785] = 127;
    colvals[3786] = 0;
    colvals[3787] = 2;
    colvals[3788] = 3;
    colvals[3789] = 4;
    colvals[3790] = 5;
    colvals[3791] = 6;
    colvals[3792] = 7;
    colvals[3793] = 8;
    colvals[3794] = 9;
    colvals[3795] = 10;
    colvals[3796] = 12;
    colvals[3797] = 14;
    colvals[3798] = 15;
    colvals[3799] = 16;
    colvals[3800] = 17;
    colvals[3801] = 18;
    colvals[3802] = 19;
    colvals[3803] = 20;
    colvals[3804] = 21;
    colvals[3805] = 22;
    colvals[3806] = 23;
    colvals[3807] = 24;
    colvals[3808] = 25;
    colvals[3809] = 26;
    colvals[3810] = 28;
    colvals[3811] = 29;
    colvals[3812] = 30;
    colvals[3813] = 31;
    colvals[3814] = 33;
    colvals[3815] = 35;
    colvals[3816] = 36;
    colvals[3817] = 37;
    colvals[3818] = 39;
    colvals[3819] = 41;
    colvals[3820] = 43;
    colvals[3821] = 48;
    colvals[3822] = 50;
    colvals[3823] = 52;
    colvals[3824] = 54;
    colvals[3825] = 55;
    colvals[3826] = 57;
    colvals[3827] = 59;
    colvals[3828] = 61;
    colvals[3829] = 62;
    colvals[3830] = 64;
    colvals[3831] = 65;
    colvals[3832] = 66;
    colvals[3833] = 72;
    colvals[3834] = 75;
    colvals[3835] = 76;
    colvals[3836] = 77;
    colvals[3837] = 78;
    colvals[3838] = 79;
    colvals[3839] = 80;
    colvals[3840] = 81;
    colvals[3841] = 82;
    colvals[3842] = 83;
    colvals[3843] = 84;
    colvals[3844] = 85;
    colvals[3845] = 86;
    colvals[3846] = 87;
    colvals[3847] = 88;
    colvals[3848] = 89;
    colvals[3849] = 90;
    colvals[3850] = 91;
    colvals[3851] = 92;
    colvals[3852] = 93;
    colvals[3853] = 94;
    colvals[3854] = 95;
    colvals[3855] = 97;
    colvals[3856] = 98;
    colvals[3857] = 99;
    colvals[3858] = 100;
    colvals[3859] = 101;
    colvals[3860] = 102;
    colvals[3861] = 103;
    colvals[3862] = 104;
    colvals[3863] = 105;
    colvals[3864] = 107;
    colvals[3865] = 108;
    colvals[3866] = 110;
    colvals[3867] = 111;
    colvals[3868] = 113;
    colvals[3869] = 114;
    colvals[3870] = 115;
    colvals[3871] = 116;
    colvals[3872] = 117;
    colvals[3873] = 118;
    colvals[3874] = 119;
    colvals[3875] = 120;
    colvals[3876] = 121;
    colvals[3877] = 122;
    colvals[3878] = 123;
    colvals[3879] = 124;
    colvals[3880] = 125;
    colvals[3881] = 126;
    colvals[3882] = 127;
    colvals[3883] = 128;
    colvals[3884] = 129;
    colvals[3885] = 130;
    colvals[3886] = 2;
    colvals[3887] = 3;
    colvals[3888] = 5;
    colvals[3889] = 7;
    colvals[3890] = 15;
    colvals[3891] = 17;
    colvals[3892] = 19;
    colvals[3893] = 21;
    colvals[3894] = 23;
    colvals[3895] = 25;
    colvals[3896] = 29;
    colvals[3897] = 31;
    colvals[3898] = 35;
    colvals[3899] = 36;
    colvals[3900] = 39;
    colvals[3901] = 41;
    colvals[3902] = 48;
    colvals[3903] = 52;
    colvals[3904] = 54;
    colvals[3905] = 57;
    colvals[3906] = 59;
    colvals[3907] = 61;
    colvals[3908] = 64;
    colvals[3909] = 72;
    colvals[3910] = 75;
    colvals[3911] = 76;
    colvals[3912] = 77;
    colvals[3913] = 78;
    colvals[3914] = 81;
    colvals[3915] = 83;
    colvals[3916] = 85;
    colvals[3917] = 87;
    colvals[3918] = 89;
    colvals[3919] = 91;
    colvals[3920] = 93;
    colvals[3921] = 100;
    colvals[3922] = 107;
    colvals[3923] = 110;
    colvals[3924] = 113;
    colvals[3925] = 115;
    colvals[3926] = 119;
    colvals[3927] = 123;
    colvals[3928] = 127;
    colvals[3929] = 0;
    colvals[3930] = 3;
    colvals[3931] = 5;
    colvals[3932] = 7;
    colvals[3933] = 15;
    colvals[3934] = 17;
    colvals[3935] = 19;
    colvals[3936] = 21;
    colvals[3937] = 23;
    colvals[3938] = 25;
    colvals[3939] = 29;
    colvals[3940] = 31;
    colvals[3941] = 33;
    colvals[3942] = 36;
    colvals[3943] = 39;
    colvals[3944] = 41;
    colvals[3945] = 47;
    colvals[3946] = 50;
    colvals[3947] = 54;
    colvals[3948] = 57;
    colvals[3949] = 59;
    colvals[3950] = 62;
    colvals[3951] = 64;
    colvals[3952] = 70;
    colvals[3953] = 72;
    colvals[3954] = 75;
    colvals[3955] = 77;
    colvals[3956] = 78;
    colvals[3957] = 79;
    colvals[3958] = 80;
    colvals[3959] = 81;
    colvals[3960] = 83;
    colvals[3961] = 84;
    colvals[3962] = 85;
    colvals[3963] = 87;
    colvals[3964] = 88;
    colvals[3965] = 89;
    colvals[3966] = 91;
    colvals[3967] = 93;
    colvals[3968] = 95;
    colvals[3969] = 97;
    colvals[3970] = 98;
    colvals[3971] = 100;
    colvals[3972] = 103;
    colvals[3973] = 105;
    colvals[3974] = 107;
    colvals[3975] = 108;
    colvals[3976] = 110;
    colvals[3977] = 111;
    colvals[3978] = 113;
    colvals[3979] = 114;
    colvals[3980] = 116;
    colvals[3981] = 117;
    colvals[3982] = 118;
    colvals[3983] = 120;
    colvals[3984] = 121;
    colvals[3985] = 122;
    colvals[3986] = 124;
    colvals[3987] = 125;
    colvals[3988] = 126;
    colvals[3989] = 128;
    colvals[3990] = 129;
    colvals[3991] = 130;
    colvals[3992] = 0;
    colvals[3993] = 3;
    colvals[3994] = 5;
    colvals[3995] = 7;
    colvals[3996] = 15;
    colvals[3997] = 17;
    colvals[3998] = 19;
    colvals[3999] = 21;
    colvals[4000] = 23;
    colvals[4001] = 25;
    colvals[4002] = 29;
    colvals[4003] = 31;
    colvals[4004] = 33;
    colvals[4005] = 36;
    colvals[4006] = 39;
    colvals[4007] = 41;
    colvals[4008] = 50;
    colvals[4009] = 54;
    colvals[4010] = 57;
    colvals[4011] = 59;
    colvals[4012] = 61;
    colvals[4013] = 64;
    colvals[4014] = 72;
    colvals[4015] = 75;
    colvals[4016] = 76;
    colvals[4017] = 77;
    colvals[4018] = 78;
    colvals[4019] = 81;
    colvals[4020] = 83;
    colvals[4021] = 84;
    colvals[4022] = 85;
    colvals[4023] = 87;
    colvals[4024] = 88;
    colvals[4025] = 89;
    colvals[4026] = 91;
    colvals[4027] = 93;
    colvals[4028] = 97;
    colvals[4029] = 100;
    colvals[4030] = 107;
    colvals[4031] = 110;
    colvals[4032] = 113;
    colvals[4033] = 115;
    colvals[4034] = 119;
    colvals[4035] = 123;
    colvals[4036] = 127;
    colvals[4037] = 0;
    colvals[4038] = 3;
    colvals[4039] = 15;
    colvals[4040] = 19;
    colvals[4041] = 29;
    colvals[4042] = 36;
    colvals[4043] = 41;
    colvals[4044] = 47;
    colvals[4045] = 54;
    colvals[4046] = 61;
    colvals[4047] = 62;
    colvals[4048] = 64;
    colvals[4049] = 75;
    colvals[4050] = 77;
    colvals[4051] = 78;
    colvals[4052] = 79;
    colvals[4053] = 83;
    colvals[4054] = 84;
    colvals[4055] = 86;
    colvals[4056] = 87;
    colvals[4057] = 92;
    colvals[4058] = 93;
    colvals[4059] = 103;
    colvals[4060] = 107;
    colvals[4061] = 108;
    colvals[4062] = 110;
    colvals[4063] = 113;
    colvals[4064] = 114;
    colvals[4065] = 115;
    colvals[4066] = 116;
    colvals[4067] = 117;
    colvals[4068] = 118;
    colvals[4069] = 121;
    colvals[4070] = 123;
    colvals[4071] = 124;
    colvals[4072] = 125;
    colvals[4073] = 126;
    colvals[4074] = 129;
    colvals[4075] = 0;
    colvals[4076] = 3;
    colvals[4077] = 15;
    colvals[4078] = 19;
    colvals[4079] = 29;
    colvals[4080] = 36;
    colvals[4081] = 54;
    colvals[4082] = 59;
    colvals[4083] = 61;
    colvals[4084] = 62;
    colvals[4085] = 64;
    colvals[4086] = 70;
    colvals[4087] = 75;
    colvals[4088] = 77;
    colvals[4089] = 78;
    colvals[4090] = 80;
    colvals[4091] = 83;
    colvals[4092] = 87;
    colvals[4093] = 88;
    colvals[4094] = 90;
    colvals[4095] = 92;
    colvals[4096] = 93;
    colvals[4097] = 105;
    colvals[4098] = 107;
    colvals[4099] = 110;
    colvals[4100] = 111;
    colvals[4101] = 113;
    colvals[4102] = 117;
    colvals[4103] = 119;
    colvals[4104] = 120;
    colvals[4105] = 121;
    colvals[4106] = 122;
    colvals[4107] = 125;
    colvals[4108] = 127;
    colvals[4109] = 128;
    colvals[4110] = 129;
    colvals[4111] = 130;
    colvals[4112] = 33;
    colvals[4113] = 50;
    colvals[4114] = 72;
    colvals[4115] = 75;
    colvals[4116] = 81;
    colvals[4117] = 83;
    colvals[4118] = 87;
    colvals[4119] = 93;
    colvals[4120] = 95;
    colvals[4121] = 98;
    colvals[4122] = 26;
    colvals[4123] = 29;
    colvals[4124] = 31;
    colvals[4125] = 82;
    colvals[4126] = 84;
    colvals[4127] = 88;
    colvals[4128] = 93;
    colvals[4129] = 100;
    colvals[4130] = 113;
    colvals[4131] = 0;
    colvals[4132] = 1;
    colvals[4133] = 2;
    colvals[4134] = 3;
    colvals[4135] = 4;
    colvals[4136] = 15;
    colvals[4137] = 16;
    colvals[4138] = 19;
    colvals[4139] = 20;
    colvals[4140] = 26;
    colvals[4141] = 29;
    colvals[4142] = 30;
    colvals[4143] = 33;
    colvals[4144] = 34;
    colvals[4145] = 35;
    colvals[4146] = 36;
    colvals[4147] = 37;
    colvals[4148] = 39;
    colvals[4149] = 40;
    colvals[4150] = 42;
    colvals[4151] = 44;
    colvals[4152] = 45;
    colvals[4153] = 46;
    colvals[4154] = 50;
    colvals[4155] = 51;
    colvals[4156] = 52;
    colvals[4157] = 54;
    colvals[4158] = 55;
    colvals[4159] = 58;
    colvals[4160] = 60;
    colvals[4161] = 61;
    colvals[4162] = 62;
    colvals[4163] = 64;
    colvals[4164] = 65;
    colvals[4165] = 67;
    colvals[4166] = 68;
    colvals[4167] = 69;
    colvals[4168] = 72;
    colvals[4169] = 75;
    colvals[4170] = 76;
    colvals[4171] = 78;
    colvals[4172] = 79;
    colvals[4173] = 80;
    colvals[4174] = 83;
    colvals[4175] = 84;
    colvals[4176] = 85;
    colvals[4177] = 86;
    colvals[4178] = 87;
    colvals[4179] = 88;
    colvals[4180] = 89;
    colvals[4181] = 90;
    colvals[4182] = 91;
    colvals[4183] = 92;
    colvals[4184] = 93;
    colvals[4185] = 97;
    colvals[4186] = 98;
    colvals[4187] = 100;
    colvals[4188] = 101;
    colvals[4189] = 102;
    colvals[4190] = 103;
    colvals[4191] = 105;
    colvals[4192] = 107;
    colvals[4193] = 108;
    colvals[4194] = 110;
    colvals[4195] = 111;
    colvals[4196] = 113;
    colvals[4197] = 114;
    colvals[4198] = 115;
    colvals[4199] = 116;
    colvals[4200] = 117;
    colvals[4201] = 118;
    colvals[4202] = 119;
    colvals[4203] = 120;
    colvals[4204] = 121;
    colvals[4205] = 122;
    colvals[4206] = 123;
    colvals[4207] = 124;
    colvals[4208] = 125;
    colvals[4209] = 126;
    colvals[4210] = 127;
    colvals[4211] = 128;
    colvals[4212] = 129;
    colvals[4213] = 130;
    colvals[4214] = 0;
    colvals[4215] = 3;
    colvals[4216] = 15;
    colvals[4217] = 19;
    colvals[4218] = 25;
    colvals[4219] = 26;
    colvals[4220] = 29;
    colvals[4221] = 30;
    colvals[4222] = 31;
    colvals[4223] = 34;
    colvals[4224] = 36;
    colvals[4225] = 41;
    colvals[4226] = 43;
    colvals[4227] = 51;
    colvals[4228] = 54;
    colvals[4229] = 61;
    colvals[4230] = 62;
    colvals[4231] = 64;
    colvals[4232] = 72;
    colvals[4233] = 75;
    colvals[4234] = 76;
    colvals[4235] = 77;
    colvals[4236] = 78;
    colvals[4237] = 83;
    colvals[4238] = 84;
    colvals[4239] = 85;
    colvals[4240] = 87;
    colvals[4241] = 91;
    colvals[4242] = 93;
    colvals[4243] = 97;
    colvals[4244] = 98;
    colvals[4245] = 100;
    colvals[4246] = 107;
    colvals[4247] = 110;
    colvals[4248] = 113;
    colvals[4249] = 115;
    colvals[4250] = 116;
    colvals[4251] = 119;
    colvals[4252] = 120;
    colvals[4253] = 123;
    colvals[4254] = 124;
    colvals[4255] = 127;
    colvals[4256] = 128;
    colvals[4257] = 0;
    colvals[4258] = 1;
    colvals[4259] = 4;
    colvals[4260] = 15;
    colvals[4261] = 16;
    colvals[4262] = 17;
    colvals[4263] = 19;
    colvals[4264] = 20;
    colvals[4265] = 21;
    colvals[4266] = 23;
    colvals[4267] = 26;
    colvals[4268] = 30;
    colvals[4269] = 33;
    colvals[4270] = 34;
    colvals[4271] = 35;
    colvals[4272] = 37;
    colvals[4273] = 41;
    colvals[4274] = 45;
    colvals[4275] = 51;
    colvals[4276] = 55;
    colvals[4277] = 59;
    colvals[4278] = 61;
    colvals[4279] = 62;
    colvals[4280] = 65;
    colvals[4281] = 72;
    colvals[4282] = 76;
    colvals[4283] = 78;
    colvals[4284] = 83;
    colvals[4285] = 85;
    colvals[4286] = 86;
    colvals[4287] = 87;
    colvals[4288] = 93;
    colvals[4289] = 97;
    colvals[4290] = 98;
    colvals[4291] = 101;
    colvals[4292] = 107;
    colvals[4293] = 108;
    colvals[4294] = 110;
    colvals[4295] = 111;
    colvals[4296] = 115;
    colvals[4297] = 116;
    colvals[4298] = 120;
    colvals[4299] = 123;
    colvals[4300] = 124;
    colvals[4301] = 128;
    colvals[4302] = 3;
    colvals[4303] = 4;
    colvals[4304] = 15;
    colvals[4305] = 17;
    colvals[4306] = 19;
    colvals[4307] = 21;
    colvals[4308] = 23;
    colvals[4309] = 26;
    colvals[4310] = 30;
    colvals[4311] = 34;
    colvals[4312] = 36;
    colvals[4313] = 37;
    colvals[4314] = 40;
    colvals[4315] = 41;
    colvals[4316] = 42;
    colvals[4317] = 44;
    colvals[4318] = 46;
    colvals[4319] = 51;
    colvals[4320] = 54;
    colvals[4321] = 55;
    colvals[4322] = 59;
    colvals[4323] = 61;
    colvals[4324] = 62;
    colvals[4325] = 64;
    colvals[4326] = 65;
    colvals[4327] = 75;
    colvals[4328] = 76;
    colvals[4329] = 78;
    colvals[4330] = 79;
    colvals[4331] = 83;
    colvals[4332] = 84;
    colvals[4333] = 85;
    colvals[4334] = 86;
    colvals[4335] = 87;
    colvals[4336] = 88;
    colvals[4337] = 93;
    colvals[4338] = 97;
    colvals[4339] = 98;
    colvals[4340] = 100;
    colvals[4341] = 101;
    colvals[4342] = 103;
    colvals[4343] = 108;
    colvals[4344] = 111;
    colvals[4345] = 113;
    colvals[4346] = 114;
    colvals[4347] = 115;
    colvals[4348] = 116;
    colvals[4349] = 117;
    colvals[4350] = 118;
    colvals[4351] = 120;
    colvals[4352] = 121;
    colvals[4353] = 123;
    colvals[4354] = 124;
    colvals[4355] = 125;
    colvals[4356] = 126;
    colvals[4357] = 128;
    colvals[4358] = 129;
    colvals[4359] = 0;
    colvals[4360] = 1;
    colvals[4361] = 2;
    colvals[4362] = 3;
    colvals[4363] = 4;
    colvals[4364] = 15;
    colvals[4365] = 16;
    colvals[4366] = 19;
    colvals[4367] = 20;
    colvals[4368] = 26;
    colvals[4369] = 29;
    colvals[4370] = 30;
    colvals[4371] = 33;
    colvals[4372] = 34;
    colvals[4373] = 35;
    colvals[4374] = 36;
    colvals[4375] = 37;
    colvals[4376] = 40;
    colvals[4377] = 42;
    colvals[4378] = 44;
    colvals[4379] = 45;
    colvals[4380] = 46;
    colvals[4381] = 50;
    colvals[4382] = 51;
    colvals[4383] = 52;
    colvals[4384] = 54;
    colvals[4385] = 55;
    colvals[4386] = 57;
    colvals[4387] = 58;
    colvals[4388] = 60;
    colvals[4389] = 61;
    colvals[4390] = 62;
    colvals[4391] = 64;
    colvals[4392] = 65;
    colvals[4393] = 67;
    colvals[4394] = 68;
    colvals[4395] = 69;
    colvals[4396] = 72;
    colvals[4397] = 75;
    colvals[4398] = 76;
    colvals[4399] = 78;
    colvals[4400] = 79;
    colvals[4401] = 80;
    colvals[4402] = 83;
    colvals[4403] = 84;
    colvals[4404] = 85;
    colvals[4405] = 86;
    colvals[4406] = 87;
    colvals[4407] = 88;
    colvals[4408] = 89;
    colvals[4409] = 90;
    colvals[4410] = 91;
    colvals[4411] = 92;
    colvals[4412] = 93;
    colvals[4413] = 97;
    colvals[4414] = 98;
    colvals[4415] = 100;
    colvals[4416] = 101;
    colvals[4417] = 103;
    colvals[4418] = 104;
    colvals[4419] = 105;
    colvals[4420] = 107;
    colvals[4421] = 108;
    colvals[4422] = 110;
    colvals[4423] = 111;
    colvals[4424] = 113;
    colvals[4425] = 114;
    colvals[4426] = 115;
    colvals[4427] = 116;
    colvals[4428] = 117;
    colvals[4429] = 118;
    colvals[4430] = 119;
    colvals[4431] = 120;
    colvals[4432] = 121;
    colvals[4433] = 122;
    colvals[4434] = 123;
    colvals[4435] = 124;
    colvals[4436] = 125;
    colvals[4437] = 127;
    colvals[4438] = 128;
    colvals[4439] = 129;
    colvals[4440] = 130;
    colvals[4441] = 0;
    colvals[4442] = 3;
    colvals[4443] = 15;
    colvals[4444] = 19;
    colvals[4445] = 25;
    colvals[4446] = 26;
    colvals[4447] = 29;
    colvals[4448] = 30;
    colvals[4449] = 31;
    colvals[4450] = 34;
    colvals[4451] = 36;
    colvals[4452] = 51;
    colvals[4453] = 54;
    colvals[4454] = 59;
    colvals[4455] = 61;
    colvals[4456] = 62;
    colvals[4457] = 64;
    colvals[4458] = 66;
    colvals[4459] = 72;
    colvals[4460] = 75;
    colvals[4461] = 76;
    colvals[4462] = 77;
    colvals[4463] = 78;
    colvals[4464] = 83;
    colvals[4465] = 87;
    colvals[4466] = 88;
    colvals[4467] = 89;
    colvals[4468] = 91;
    colvals[4469] = 93;
    colvals[4470] = 97;
    colvals[4471] = 98;
    colvals[4472] = 100;
    colvals[4473] = 107;
    colvals[4474] = 110;
    colvals[4475] = 113;
    colvals[4476] = 115;
    colvals[4477] = 116;
    colvals[4478] = 119;
    colvals[4479] = 120;
    colvals[4480] = 123;
    colvals[4481] = 124;
    colvals[4482] = 127;
    colvals[4483] = 128;
    colvals[4484] = 0;
    colvals[4485] = 1;
    colvals[4486] = 4;
    colvals[4487] = 15;
    colvals[4488] = 16;
    colvals[4489] = 17;
    colvals[4490] = 19;
    colvals[4491] = 20;
    colvals[4492] = 21;
    colvals[4493] = 23;
    colvals[4494] = 26;
    colvals[4495] = 30;
    colvals[4496] = 34;
    colvals[4497] = 37;
    colvals[4498] = 41;
    colvals[4499] = 50;
    colvals[4500] = 51;
    colvals[4501] = 52;
    colvals[4502] = 55;
    colvals[4503] = 59;
    colvals[4504] = 61;
    colvals[4505] = 62;
    colvals[4506] = 65;
    colvals[4507] = 68;
    colvals[4508] = 72;
    colvals[4509] = 76;
    colvals[4510] = 78;
    colvals[4511] = 83;
    colvals[4512] = 87;
    colvals[4513] = 89;
    colvals[4514] = 90;
    colvals[4515] = 93;
    colvals[4516] = 97;
    colvals[4517] = 98;
    colvals[4518] = 101;
    colvals[4519] = 107;
    colvals[4520] = 108;
    colvals[4521] = 110;
    colvals[4522] = 111;
    colvals[4523] = 116;
    colvals[4524] = 119;
    colvals[4525] = 120;
    colvals[4526] = 124;
    colvals[4527] = 127;
    colvals[4528] = 128;
    colvals[4529] = 3;
    colvals[4530] = 4;
    colvals[4531] = 15;
    colvals[4532] = 17;
    colvals[4533] = 19;
    colvals[4534] = 21;
    colvals[4535] = 23;
    colvals[4536] = 26;
    colvals[4537] = 30;
    colvals[4538] = 34;
    colvals[4539] = 36;
    colvals[4540] = 37;
    colvals[4541] = 41;
    colvals[4542] = 51;
    colvals[4543] = 54;
    colvals[4544] = 55;
    colvals[4545] = 58;
    colvals[4546] = 59;
    colvals[4547] = 60;
    colvals[4548] = 61;
    colvals[4549] = 62;
    colvals[4550] = 64;
    colvals[4551] = 65;
    colvals[4552] = 67;
    colvals[4553] = 69;
    colvals[4554] = 75;
    colvals[4555] = 76;
    colvals[4556] = 78;
    colvals[4557] = 80;
    colvals[4558] = 83;
    colvals[4559] = 84;
    colvals[4560] = 87;
    colvals[4561] = 88;
    colvals[4562] = 89;
    colvals[4563] = 90;
    colvals[4564] = 93;
    colvals[4565] = 97;
    colvals[4566] = 98;
    colvals[4567] = 100;
    colvals[4568] = 101;
    colvals[4569] = 105;
    colvals[4570] = 108;
    colvals[4571] = 111;
    colvals[4572] = 113;
    colvals[4573] = 116;
    colvals[4574] = 117;
    colvals[4575] = 119;
    colvals[4576] = 120;
    colvals[4577] = 121;
    colvals[4578] = 122;
    colvals[4579] = 124;
    colvals[4580] = 125;
    colvals[4581] = 127;
    colvals[4582] = 128;
    colvals[4583] = 129;
    colvals[4584] = 130;
    colvals[4585] = 0;
    colvals[4586] = 1;
    colvals[4587] = 4;
    colvals[4588] = 15;
    colvals[4589] = 16;
    colvals[4590] = 17;
    colvals[4591] = 19;
    colvals[4592] = 20;
    colvals[4593] = 21;
    colvals[4594] = 23;
    colvals[4595] = 26;
    colvals[4596] = 30;
    colvals[4597] = 33;
    colvals[4598] = 34;
    colvals[4599] = 35;
    colvals[4600] = 37;
    colvals[4601] = 41;
    colvals[4602] = 45;
    colvals[4603] = 50;
    colvals[4604] = 51;
    colvals[4605] = 52;
    colvals[4606] = 55;
    colvals[4607] = 59;
    colvals[4608] = 61;
    colvals[4609] = 62;
    colvals[4610] = 65;
    colvals[4611] = 68;
    colvals[4612] = 72;
    colvals[4613] = 76;
    colvals[4614] = 78;
    colvals[4615] = 83;
    colvals[4616] = 87;
    colvals[4617] = 91;
    colvals[4618] = 92;
    colvals[4619] = 93;
    colvals[4620] = 97;
    colvals[4621] = 98;
    colvals[4622] = 101;
    colvals[4623] = 107;
    colvals[4624] = 108;
    colvals[4625] = 110;
    colvals[4626] = 111;
    colvals[4627] = 115;
    colvals[4628] = 116;
    colvals[4629] = 119;
    colvals[4630] = 120;
    colvals[4631] = 123;
    colvals[4632] = 124;
    colvals[4633] = 127;
    colvals[4634] = 128;
    colvals[4635] = 3;
    colvals[4636] = 4;
    colvals[4637] = 15;
    colvals[4638] = 17;
    colvals[4639] = 19;
    colvals[4640] = 21;
    colvals[4641] = 23;
    colvals[4642] = 26;
    colvals[4643] = 30;
    colvals[4644] = 34;
    colvals[4645] = 36;
    colvals[4646] = 37;
    colvals[4647] = 40;
    colvals[4648] = 41;
    colvals[4649] = 42;
    colvals[4650] = 44;
    colvals[4651] = 46;
    colvals[4652] = 51;
    colvals[4653] = 54;
    colvals[4654] = 55;
    colvals[4655] = 58;
    colvals[4656] = 59;
    colvals[4657] = 60;
    colvals[4658] = 61;
    colvals[4659] = 62;
    colvals[4660] = 64;
    colvals[4661] = 65;
    colvals[4662] = 67;
    colvals[4663] = 69;
    colvals[4664] = 75;
    colvals[4665] = 76;
    colvals[4666] = 78;
    colvals[4667] = 79;
    colvals[4668] = 80;
    colvals[4669] = 83;
    colvals[4670] = 84;
    colvals[4671] = 87;
    colvals[4672] = 88;
    colvals[4673] = 91;
    colvals[4674] = 92;
    colvals[4675] = 93;
    colvals[4676] = 97;
    colvals[4677] = 98;
    colvals[4678] = 100;
    colvals[4679] = 101;
    colvals[4680] = 103;
    colvals[4681] = 105;
    colvals[4682] = 108;
    colvals[4683] = 111;
    colvals[4684] = 113;
    colvals[4685] = 115;
    colvals[4686] = 116;
    colvals[4687] = 117;
    colvals[4688] = 119;
    colvals[4689] = 120;
    colvals[4690] = 121;
    colvals[4691] = 122;
    colvals[4692] = 123;
    colvals[4693] = 124;
    colvals[4694] = 125;
    colvals[4695] = 127;
    colvals[4696] = 128;
    colvals[4697] = 129;
    colvals[4698] = 130;
    colvals[4699] = 0;
    colvals[4700] = 1;
    colvals[4701] = 2;
    colvals[4702] = 3;
    colvals[4703] = 4;
    colvals[4704] = 6;
    colvals[4705] = 8;
    colvals[4706] = 15;
    colvals[4707] = 16;
    colvals[4708] = 18;
    colvals[4709] = 19;
    colvals[4710] = 20;
    colvals[4711] = 22;
    colvals[4712] = 24;
    colvals[4713] = 25;
    colvals[4714] = 26;
    colvals[4715] = 29;
    colvals[4716] = 30;
    colvals[4717] = 31;
    colvals[4718] = 32;
    colvals[4719] = 33;
    colvals[4720] = 34;
    colvals[4721] = 36;
    colvals[4722] = 37;
    colvals[4723] = 40;
    colvals[4724] = 44;
    colvals[4725] = 45;
    colvals[4726] = 46;
    colvals[4727] = 50;
    colvals[4728] = 51;
    colvals[4729] = 54;
    colvals[4730] = 55;
    colvals[4731] = 58;
    colvals[4732] = 62;
    colvals[4733] = 64;
    colvals[4734] = 65;
    colvals[4735] = 67;
    colvals[4736] = 68;
    colvals[4737] = 69;
    colvals[4738] = 72;
    colvals[4739] = 75;
    colvals[4740] = 76;
    colvals[4741] = 78;
    colvals[4742] = 81;
    colvals[4743] = 83;
    colvals[4744] = 84;
    colvals[4745] = 85;
    colvals[4746] = 86;
    colvals[4747] = 87;
    colvals[4748] = 88;
    colvals[4749] = 89;
    colvals[4750] = 90;
    colvals[4751] = 91;
    colvals[4752] = 92;
    colvals[4753] = 93;
    colvals[4754] = 95;
    colvals[4755] = 96;
    colvals[4756] = 97;
    colvals[4757] = 98;
    colvals[4758] = 99;
    colvals[4759] = 100;
    colvals[4760] = 101;
    colvals[4761] = 103;
    colvals[4762] = 105;
    colvals[4763] = 106;
    colvals[4764] = 107;
    colvals[4765] = 108;
    colvals[4766] = 110;
    colvals[4767] = 111;
    colvals[4768] = 113;
    colvals[4769] = 114;
    colvals[4770] = 116;
    colvals[4771] = 117;
    colvals[4772] = 118;
    colvals[4773] = 120;
    colvals[4774] = 121;
    colvals[4775] = 122;
    colvals[4776] = 124;
    colvals[4777] = 125;
    colvals[4778] = 126;
    colvals[4779] = 128;
    colvals[4780] = 129;
    colvals[4781] = 130;
    colvals[4782] = 1;
    colvals[4783] = 4;
    colvals[4784] = 6;
    colvals[4785] = 8;
    colvals[4786] = 16;
    colvals[4787] = 18;
    colvals[4788] = 20;
    colvals[4789] = 22;
    colvals[4790] = 24;
    colvals[4791] = 25;
    colvals[4792] = 26;
    colvals[4793] = 29;
    colvals[4794] = 30;
    colvals[4795] = 31;
    colvals[4796] = 32;
    colvals[4797] = 33;
    colvals[4798] = 34;
    colvals[4799] = 37;
    colvals[4800] = 39;
    colvals[4801] = 40;
    colvals[4802] = 44;
    colvals[4803] = 45;
    colvals[4804] = 46;
    colvals[4805] = 50;
    colvals[4806] = 51;
    colvals[4807] = 55;
    colvals[4808] = 57;
    colvals[4809] = 58;
    colvals[4810] = 61;
    colvals[4811] = 62;
    colvals[4812] = 65;
    colvals[4813] = 67;
    colvals[4814] = 68;
    colvals[4815] = 69;
    colvals[4816] = 72;
    colvals[4817] = 75;
    colvals[4818] = 76;
    colvals[4819] = 77;
    colvals[4820] = 78;
    colvals[4821] = 81;
    colvals[4822] = 83;
    colvals[4823] = 84;
    colvals[4824] = 86;
    colvals[4825] = 87;
    colvals[4826] = 88;
    colvals[4827] = 90;
    colvals[4828] = 92;
    colvals[4829] = 93;
    colvals[4830] = 94;
    colvals[4831] = 95;
    colvals[4832] = 96;
    colvals[4833] = 97;
    colvals[4834] = 98;
    colvals[4835] = 100;
    colvals[4836] = 101;
    colvals[4837] = 107;
    colvals[4838] = 108;
    colvals[4839] = 110;
    colvals[4840] = 111;
    colvals[4841] = 113;
    colvals[4842] = 114;
    colvals[4843] = 115;
    colvals[4844] = 116;
    colvals[4845] = 117;
    colvals[4846] = 118;
    colvals[4847] = 119;
    colvals[4848] = 120;
    colvals[4849] = 121;
    colvals[4850] = 122;
    colvals[4851] = 123;
    colvals[4852] = 124;
    colvals[4853] = 125;
    colvals[4854] = 126;
    colvals[4855] = 127;
    colvals[4856] = 128;
    colvals[4857] = 129;
    colvals[4858] = 130;
    colvals[4859] = 33;
    colvals[4860] = 50;
    colvals[4861] = 75;
    colvals[4862] = 95;
    colvals[4863] = 97;
    colvals[4864] = 98;
    colvals[4865] = 101;
    colvals[4866] = 114;
    colvals[4867] = 117;
    colvals[4868] = 118;
    colvals[4869] = 121;
    colvals[4870] = 122;
    colvals[4871] = 125;
    colvals[4872] = 126;
    colvals[4873] = 129;
    colvals[4874] = 130;
    colvals[4875] = 33;
    colvals[4876] = 46;
    colvals[4877] = 50;
    colvals[4878] = 61;
    colvals[4879] = 69;
    colvals[4880] = 75;
    colvals[4881] = 83;
    colvals[4882] = 87;
    colvals[4883] = 95;
    colvals[4884] = 96;
    colvals[4885] = 97;
    colvals[4886] = 98;
    colvals[4887] = 101;
    colvals[4888] = 103;
    colvals[4889] = 105;
    colvals[4890] = 113;
    colvals[4891] = 115;
    colvals[4892] = 119;
    colvals[4893] = 123;
    colvals[4894] = 127;
    colvals[4895] = 0;
    colvals[4896] = 1;
    colvals[4897] = 2;
    colvals[4898] = 3;
    colvals[4899] = 4;
    colvals[4900] = 5;
    colvals[4901] = 6;
    colvals[4902] = 7;
    colvals[4903] = 8;
    colvals[4904] = 9;
    colvals[4905] = 12;
    colvals[4906] = 14;
    colvals[4907] = 15;
    colvals[4908] = 16;
    colvals[4909] = 17;
    colvals[4910] = 18;
    colvals[4911] = 19;
    colvals[4912] = 20;
    colvals[4913] = 21;
    colvals[4914] = 22;
    colvals[4915] = 23;
    colvals[4916] = 24;
    colvals[4917] = 25;
    colvals[4918] = 26;
    colvals[4919] = 29;
    colvals[4920] = 30;
    colvals[4921] = 31;
    colvals[4922] = 32;
    colvals[4923] = 33;
    colvals[4924] = 34;
    colvals[4925] = 35;
    colvals[4926] = 36;
    colvals[4927] = 37;
    colvals[4928] = 38;
    colvals[4929] = 39;
    colvals[4930] = 40;
    colvals[4931] = 41;
    colvals[4932] = 45;
    colvals[4933] = 46;
    colvals[4934] = 48;
    colvals[4935] = 50;
    colvals[4936] = 51;
    colvals[4937] = 52;
    colvals[4938] = 53;
    colvals[4939] = 54;
    colvals[4940] = 55;
    colvals[4941] = 56;
    colvals[4942] = 57;
    colvals[4943] = 58;
    colvals[4944] = 59;
    colvals[4945] = 61;
    colvals[4946] = 62;
    colvals[4947] = 63;
    colvals[4948] = 64;
    colvals[4949] = 65;
    colvals[4950] = 68;
    colvals[4951] = 69;
    colvals[4952] = 72;
    colvals[4953] = 75;
    colvals[4954] = 76;
    colvals[4955] = 77;
    colvals[4956] = 78;
    colvals[4957] = 81;
    colvals[4958] = 83;
    colvals[4959] = 84;
    colvals[4960] = 85;
    colvals[4961] = 86;
    colvals[4962] = 87;
    colvals[4963] = 88;
    colvals[4964] = 89;
    colvals[4965] = 90;
    colvals[4966] = 91;
    colvals[4967] = 92;
    colvals[4968] = 93;
    colvals[4969] = 94;
    colvals[4970] = 95;
    colvals[4971] = 96;
    colvals[4972] = 97;
    colvals[4973] = 98;
    colvals[4974] = 99;
    colvals[4975] = 100;
    colvals[4976] = 101;
    colvals[4977] = 102;
    colvals[4978] = 103;
    colvals[4979] = 104;
    colvals[4980] = 105;
    colvals[4981] = 106;
    colvals[4982] = 107;
    colvals[4983] = 108;
    colvals[4984] = 110;
    colvals[4985] = 111;
    colvals[4986] = 113;
    colvals[4987] = 114;
    colvals[4988] = 115;
    colvals[4989] = 116;
    colvals[4990] = 117;
    colvals[4991] = 118;
    colvals[4992] = 119;
    colvals[4993] = 120;
    colvals[4994] = 121;
    colvals[4995] = 122;
    colvals[4996] = 123;
    colvals[4997] = 124;
    colvals[4998] = 125;
    colvals[4999] = 126;
    colvals[5000] = 127;
    colvals[5001] = 128;
    colvals[5002] = 129;
    colvals[5003] = 130;
    colvals[5004] = 1;
    colvals[5005] = 2;
    colvals[5006] = 3;
    colvals[5007] = 5;
    colvals[5008] = 7;
    colvals[5009] = 15;
    colvals[5010] = 16;
    colvals[5011] = 17;
    colvals[5012] = 19;
    colvals[5013] = 20;
    colvals[5014] = 21;
    colvals[5015] = 23;
    colvals[5016] = 25;
    colvals[5017] = 26;
    colvals[5018] = 30;
    colvals[5019] = 31;
    colvals[5020] = 32;
    colvals[5021] = 33;
    colvals[5022] = 34;
    colvals[5023] = 35;
    colvals[5024] = 36;
    colvals[5025] = 39;
    colvals[5026] = 40;
    colvals[5027] = 41;
    colvals[5028] = 48;
    colvals[5029] = 50;
    colvals[5030] = 51;
    colvals[5031] = 52;
    colvals[5032] = 54;
    colvals[5033] = 57;
    colvals[5034] = 58;
    colvals[5035] = 59;
    colvals[5036] = 61;
    colvals[5037] = 64;
    colvals[5038] = 72;
    colvals[5039] = 76;
    colvals[5040] = 77;
    colvals[5041] = 78;
    colvals[5042] = 81;
    colvals[5043] = 83;
    colvals[5044] = 85;
    colvals[5045] = 87;
    colvals[5046] = 89;
    colvals[5047] = 91;
    colvals[5048] = 93;
    colvals[5049] = 95;
    colvals[5050] = 97;
    colvals[5051] = 98;
    colvals[5052] = 100;
    colvals[5053] = 106;
    colvals[5054] = 107;
    colvals[5055] = 110;
    colvals[5056] = 113;
    colvals[5057] = 115;
    colvals[5058] = 119;
    colvals[5059] = 123;
    colvals[5060] = 127;
    colvals[5061] = 0;
    colvals[5062] = 2;
    colvals[5063] = 15;
    colvals[5064] = 19;
    colvals[5065] = 25;
    colvals[5066] = 29;
    colvals[5067] = 33;
    colvals[5068] = 38;
    colvals[5069] = 39;
    colvals[5070] = 50;
    colvals[5071] = 53;
    colvals[5072] = 56;
    colvals[5073] = 57;
    colvals[5074] = 61;
    colvals[5075] = 63;
    colvals[5076] = 75;
    colvals[5077] = 97;
    colvals[5078] = 99;
    colvals[5079] = 100;
    colvals[5080] = 113;
    colvals[5081] = 115;
    colvals[5082] = 119;
    colvals[5083] = 123;
    colvals[5084] = 127;
    colvals[5085] = 0;
    colvals[5086] = 1;
    colvals[5087] = 2;
    colvals[5088] = 3;
    colvals[5089] = 4;
    colvals[5090] = 5;
    colvals[5091] = 7;
    colvals[5092] = 15;
    colvals[5093] = 16;
    colvals[5094] = 17;
    colvals[5095] = 18;
    colvals[5096] = 19;
    colvals[5097] = 20;
    colvals[5098] = 21;
    colvals[5099] = 22;
    colvals[5100] = 23;
    colvals[5101] = 24;
    colvals[5102] = 25;
    colvals[5103] = 26;
    colvals[5104] = 29;
    colvals[5105] = 30;
    colvals[5106] = 31;
    colvals[5107] = 32;
    colvals[5108] = 33;
    colvals[5109] = 34;
    colvals[5110] = 37;
    colvals[5111] = 40;
    colvals[5112] = 41;
    colvals[5113] = 44;
    colvals[5114] = 50;
    colvals[5115] = 51;
    colvals[5116] = 55;
    colvals[5117] = 58;
    colvals[5118] = 59;
    colvals[5119] = 61;
    colvals[5120] = 62;
    colvals[5121] = 65;
    colvals[5122] = 67;
    colvals[5123] = 72;
    colvals[5124] = 75;
    colvals[5125] = 76;
    colvals[5126] = 77;
    colvals[5127] = 78;
    colvals[5128] = 83;
    colvals[5129] = 84;
    colvals[5130] = 85;
    colvals[5131] = 86;
    colvals[5132] = 87;
    colvals[5133] = 88;
    colvals[5134] = 89;
    colvals[5135] = 90;
    colvals[5136] = 91;
    colvals[5137] = 92;
    colvals[5138] = 93;
    colvals[5139] = 95;
    colvals[5140] = 97;
    colvals[5141] = 98;
    colvals[5142] = 99;
    colvals[5143] = 100;
    colvals[5144] = 101;
    colvals[5145] = 102;
    colvals[5146] = 103;
    colvals[5147] = 104;
    colvals[5148] = 105;
    colvals[5149] = 106;
    colvals[5150] = 107;
    colvals[5151] = 108;
    colvals[5152] = 110;
    colvals[5153] = 111;
    colvals[5154] = 113;
    colvals[5155] = 114;
    colvals[5156] = 115;
    colvals[5157] = 116;
    colvals[5158] = 117;
    colvals[5159] = 118;
    colvals[5160] = 119;
    colvals[5161] = 120;
    colvals[5162] = 121;
    colvals[5163] = 122;
    colvals[5164] = 123;
    colvals[5165] = 124;
    colvals[5166] = 125;
    colvals[5167] = 126;
    colvals[5168] = 127;
    colvals[5169] = 128;
    colvals[5170] = 129;
    colvals[5171] = 130;
    colvals[5172] = 0;
    colvals[5173] = 3;
    colvals[5174] = 15;
    colvals[5175] = 17;
    colvals[5176] = 19;
    colvals[5177] = 21;
    colvals[5178] = 23;
    colvals[5179] = 26;
    colvals[5180] = 30;
    colvals[5181] = 31;
    colvals[5182] = 32;
    colvals[5183] = 34;
    colvals[5184] = 37;
    colvals[5185] = 40;
    colvals[5186] = 41;
    colvals[5187] = 51;
    colvals[5188] = 55;
    colvals[5189] = 58;
    colvals[5190] = 59;
    colvals[5191] = 62;
    colvals[5192] = 65;
    colvals[5193] = 72;
    colvals[5194] = 75;
    colvals[5195] = 76;
    colvals[5196] = 78;
    colvals[5197] = 83;
    colvals[5198] = 84;
    colvals[5199] = 85;
    colvals[5200] = 87;
    colvals[5201] = 88;
    colvals[5202] = 89;
    colvals[5203] = 91;
    colvals[5204] = 93;
    colvals[5205] = 95;
    colvals[5206] = 97;
    colvals[5207] = 98;
    colvals[5208] = 100;
    colvals[5209] = 101;
    colvals[5210] = 107;
    colvals[5211] = 108;
    colvals[5212] = 110;
    colvals[5213] = 111;
    colvals[5214] = 113;
    colvals[5215] = 116;
    colvals[5216] = 120;
    colvals[5217] = 124;
    colvals[5218] = 128;
    colvals[5219] = 33;
    colvals[5220] = 50;
    colvals[5221] = 75;
    colvals[5222] = 97;
    colvals[5223] = 102;
    colvals[5224] = 107;
    colvals[5225] = 110;
    colvals[5226] = 0;
    colvals[5227] = 3;
    colvals[5228] = 15;
    colvals[5229] = 19;
    colvals[5230] = 25;
    colvals[5231] = 29;
    colvals[5232] = 41;
    colvals[5233] = 61;
    colvals[5234] = 62;
    colvals[5235] = 75;
    colvals[5236] = 77;
    colvals[5237] = 83;
    colvals[5238] = 84;
    colvals[5239] = 87;
    colvals[5240] = 93;
    colvals[5241] = 97;
    colvals[5242] = 100;
    colvals[5243] = 101;
    colvals[5244] = 103;
    colvals[5245] = 107;
    colvals[5246] = 110;
    colvals[5247] = 113;
    colvals[5248] = 114;
    colvals[5249] = 115;
    colvals[5250] = 116;
    colvals[5251] = 117;
    colvals[5252] = 118;
    colvals[5253] = 119;
    colvals[5254] = 121;
    colvals[5255] = 123;
    colvals[5256] = 124;
    colvals[5257] = 125;
    colvals[5258] = 126;
    colvals[5259] = 127;
    colvals[5260] = 129;
    colvals[5261] = 33;
    colvals[5262] = 50;
    colvals[5263] = 75;
    colvals[5264] = 97;
    colvals[5265] = 104;
    colvals[5266] = 107;
    colvals[5267] = 110;
    colvals[5268] = 0;
    colvals[5269] = 3;
    colvals[5270] = 15;
    colvals[5271] = 19;
    colvals[5272] = 25;
    colvals[5273] = 29;
    colvals[5274] = 59;
    colvals[5275] = 61;
    colvals[5276] = 62;
    colvals[5277] = 75;
    colvals[5278] = 77;
    colvals[5279] = 83;
    colvals[5280] = 87;
    colvals[5281] = 88;
    colvals[5282] = 93;
    colvals[5283] = 97;
    colvals[5284] = 100;
    colvals[5285] = 101;
    colvals[5286] = 105;
    colvals[5287] = 107;
    colvals[5288] = 110;
    colvals[5289] = 113;
    colvals[5290] = 115;
    colvals[5291] = 117;
    colvals[5292] = 119;
    colvals[5293] = 120;
    colvals[5294] = 121;
    colvals[5295] = 122;
    colvals[5296] = 123;
    colvals[5297] = 125;
    colvals[5298] = 127;
    colvals[5299] = 128;
    colvals[5300] = 129;
    colvals[5301] = 130;
    colvals[5302] = 0;
    colvals[5303] = 1;
    colvals[5304] = 25;
    colvals[5305] = 39;
    colvals[5306] = 41;
    colvals[5307] = 57;
    colvals[5308] = 59;
    colvals[5309] = 72;
    colvals[5310] = 75;
    colvals[5311] = 97;
    colvals[5312] = 100;
    colvals[5313] = 106;
    colvals[5314] = 107;
    colvals[5315] = 110;
    colvals[5316] = 0;
    colvals[5317] = 1;
    colvals[5318] = 2;
    colvals[5319] = 3;
    colvals[5320] = 4;
    colvals[5321] = 5;
    colvals[5322] = 7;
    colvals[5323] = 15;
    colvals[5324] = 16;
    colvals[5325] = 17;
    colvals[5326] = 18;
    colvals[5327] = 19;
    colvals[5328] = 20;
    colvals[5329] = 21;
    colvals[5330] = 23;
    colvals[5331] = 24;
    colvals[5332] = 25;
    colvals[5333] = 26;
    colvals[5334] = 29;
    colvals[5335] = 30;
    colvals[5336] = 31;
    colvals[5337] = 33;
    colvals[5338] = 34;
    colvals[5339] = 35;
    colvals[5340] = 36;
    colvals[5341] = 37;
    colvals[5342] = 38;
    colvals[5343] = 39;
    colvals[5344] = 40;
    colvals[5345] = 41;
    colvals[5346] = 42;
    colvals[5347] = 44;
    colvals[5348] = 45;
    colvals[5349] = 46;
    colvals[5350] = 50;
    colvals[5351] = 51;
    colvals[5352] = 52;
    colvals[5353] = 53;
    colvals[5354] = 54;
    colvals[5355] = 55;
    colvals[5356] = 56;
    colvals[5357] = 58;
    colvals[5358] = 59;
    colvals[5359] = 60;
    colvals[5360] = 61;
    colvals[5361] = 62;
    colvals[5362] = 63;
    colvals[5363] = 64;
    colvals[5364] = 65;
    colvals[5365] = 67;
    colvals[5366] = 68;
    colvals[5367] = 69;
    colvals[5368] = 72;
    colvals[5369] = 75;
    colvals[5370] = 76;
    colvals[5371] = 78;
    colvals[5372] = 79;
    colvals[5373] = 80;
    colvals[5374] = 81;
    colvals[5375] = 84;
    colvals[5376] = 85;
    colvals[5377] = 86;
    colvals[5378] = 88;
    colvals[5379] = 89;
    colvals[5380] = 91;
    colvals[5381] = 92;
    colvals[5382] = 93;
    colvals[5383] = 95;
    colvals[5384] = 96;
    colvals[5385] = 97;
    colvals[5386] = 98;
    colvals[5387] = 99;
    colvals[5388] = 100;
    colvals[5389] = 102;
    colvals[5390] = 103;
    colvals[5391] = 104;
    colvals[5392] = 105;
    colvals[5393] = 107;
    colvals[5394] = 108;
    colvals[5395] = 109;
    colvals[5396] = 110;
    colvals[5397] = 111;
    colvals[5398] = 112;
    colvals[5399] = 113;
    colvals[5400] = 114;
    colvals[5401] = 115;
    colvals[5402] = 116;
    colvals[5403] = 117;
    colvals[5404] = 118;
    colvals[5405] = 119;
    colvals[5406] = 120;
    colvals[5407] = 121;
    colvals[5408] = 122;
    colvals[5409] = 123;
    colvals[5410] = 124;
    colvals[5411] = 125;
    colvals[5412] = 126;
    colvals[5413] = 127;
    colvals[5414] = 128;
    colvals[5415] = 129;
    colvals[5416] = 130;
    colvals[5417] = 0;
    colvals[5418] = 3;
    colvals[5419] = 5;
    colvals[5420] = 7;
    colvals[5421] = 15;
    colvals[5422] = 17;
    colvals[5423] = 19;
    colvals[5424] = 21;
    colvals[5425] = 23;
    colvals[5426] = 25;
    colvals[5427] = 26;
    colvals[5428] = 29;
    colvals[5429] = 30;
    colvals[5430] = 34;
    colvals[5431] = 36;
    colvals[5432] = 41;
    colvals[5433] = 51;
    colvals[5434] = 54;
    colvals[5435] = 59;
    colvals[5436] = 61;
    colvals[5437] = 62;
    colvals[5438] = 64;
    colvals[5439] = 72;
    colvals[5440] = 75;
    colvals[5441] = 76;
    colvals[5442] = 77;
    colvals[5443] = 78;
    colvals[5444] = 83;
    colvals[5445] = 84;
    colvals[5446] = 85;
    colvals[5447] = 87;
    colvals[5448] = 89;
    colvals[5449] = 91;
    colvals[5450] = 93;
    colvals[5451] = 97;
    colvals[5452] = 98;
    colvals[5453] = 100;
    colvals[5454] = 103;
    colvals[5455] = 107;
    colvals[5456] = 108;
    colvals[5457] = 110;
    colvals[5458] = 113;
    colvals[5459] = 114;
    colvals[5460] = 115;
    colvals[5461] = 116;
    colvals[5462] = 117;
    colvals[5463] = 118;
    colvals[5464] = 119;
    colvals[5465] = 120;
    colvals[5466] = 121;
    colvals[5467] = 123;
    colvals[5468] = 124;
    colvals[5469] = 125;
    colvals[5470] = 126;
    colvals[5471] = 127;
    colvals[5472] = 128;
    colvals[5473] = 129;
    colvals[5474] = 0;
    colvals[5475] = 25;
    colvals[5476] = 33;
    colvals[5477] = 35;
    colvals[5478] = 36;
    colvals[5479] = 38;
    colvals[5480] = 39;
    colvals[5481] = 50;
    colvals[5482] = 52;
    colvals[5483] = 53;
    colvals[5484] = 54;
    colvals[5485] = 56;
    colvals[5486] = 57;
    colvals[5487] = 61;
    colvals[5488] = 63;
    colvals[5489] = 64;
    colvals[5490] = 99;
    colvals[5491] = 109;
    colvals[5492] = 115;
    colvals[5493] = 123;
    colvals[5494] = 0;
    colvals[5495] = 1;
    colvals[5496] = 2;
    colvals[5497] = 3;
    colvals[5498] = 4;
    colvals[5499] = 5;
    colvals[5500] = 7;
    colvals[5501] = 15;
    colvals[5502] = 16;
    colvals[5503] = 17;
    colvals[5504] = 19;
    colvals[5505] = 20;
    colvals[5506] = 21;
    colvals[5507] = 22;
    colvals[5508] = 23;
    colvals[5509] = 24;
    colvals[5510] = 25;
    colvals[5511] = 26;
    colvals[5512] = 29;
    colvals[5513] = 30;
    colvals[5514] = 31;
    colvals[5515] = 33;
    colvals[5516] = 34;
    colvals[5517] = 35;
    colvals[5518] = 36;
    colvals[5519] = 37;
    colvals[5520] = 38;
    colvals[5521] = 40;
    colvals[5522] = 41;
    colvals[5523] = 42;
    colvals[5524] = 44;
    colvals[5525] = 45;
    colvals[5526] = 46;
    colvals[5527] = 50;
    colvals[5528] = 51;
    colvals[5529] = 52;
    colvals[5530] = 53;
    colvals[5531] = 54;
    colvals[5532] = 55;
    colvals[5533] = 56;
    colvals[5534] = 57;
    colvals[5535] = 58;
    colvals[5536] = 59;
    colvals[5537] = 60;
    colvals[5538] = 61;
    colvals[5539] = 62;
    colvals[5540] = 63;
    colvals[5541] = 64;
    colvals[5542] = 65;
    colvals[5543] = 67;
    colvals[5544] = 68;
    colvals[5545] = 69;
    colvals[5546] = 72;
    colvals[5547] = 75;
    colvals[5548] = 76;
    colvals[5549] = 78;
    colvals[5550] = 79;
    colvals[5551] = 80;
    colvals[5552] = 81;
    colvals[5553] = 84;
    colvals[5554] = 85;
    colvals[5555] = 88;
    colvals[5556] = 89;
    colvals[5557] = 90;
    colvals[5558] = 91;
    colvals[5559] = 92;
    colvals[5560] = 93;
    colvals[5561] = 95;
    colvals[5562] = 96;
    colvals[5563] = 97;
    colvals[5564] = 98;
    colvals[5565] = 99;
    colvals[5566] = 100;
    colvals[5567] = 102;
    colvals[5568] = 103;
    colvals[5569] = 104;
    colvals[5570] = 105;
    colvals[5571] = 107;
    colvals[5572] = 108;
    colvals[5573] = 109;
    colvals[5574] = 110;
    colvals[5575] = 111;
    colvals[5576] = 112;
    colvals[5577] = 113;
    colvals[5578] = 114;
    colvals[5579] = 115;
    colvals[5580] = 116;
    colvals[5581] = 117;
    colvals[5582] = 118;
    colvals[5583] = 119;
    colvals[5584] = 120;
    colvals[5585] = 121;
    colvals[5586] = 122;
    colvals[5587] = 123;
    colvals[5588] = 124;
    colvals[5589] = 125;
    colvals[5590] = 127;
    colvals[5591] = 128;
    colvals[5592] = 129;
    colvals[5593] = 130;
    colvals[5594] = 0;
    colvals[5595] = 3;
    colvals[5596] = 5;
    colvals[5597] = 7;
    colvals[5598] = 15;
    colvals[5599] = 17;
    colvals[5600] = 19;
    colvals[5601] = 21;
    colvals[5602] = 23;
    colvals[5603] = 25;
    colvals[5604] = 26;
    colvals[5605] = 29;
    colvals[5606] = 30;
    colvals[5607] = 34;
    colvals[5608] = 36;
    colvals[5609] = 41;
    colvals[5610] = 51;
    colvals[5611] = 54;
    colvals[5612] = 59;
    colvals[5613] = 61;
    colvals[5614] = 62;
    colvals[5615] = 64;
    colvals[5616] = 72;
    colvals[5617] = 75;
    colvals[5618] = 76;
    colvals[5619] = 77;
    colvals[5620] = 78;
    colvals[5621] = 83;
    colvals[5622] = 85;
    colvals[5623] = 87;
    colvals[5624] = 88;
    colvals[5625] = 89;
    colvals[5626] = 91;
    colvals[5627] = 93;
    colvals[5628] = 97;
    colvals[5629] = 98;
    colvals[5630] = 100;
    colvals[5631] = 105;
    colvals[5632] = 107;
    colvals[5633] = 110;
    colvals[5634] = 111;
    colvals[5635] = 113;
    colvals[5636] = 115;
    colvals[5637] = 116;
    colvals[5638] = 117;
    colvals[5639] = 119;
    colvals[5640] = 120;
    colvals[5641] = 121;
    colvals[5642] = 122;
    colvals[5643] = 123;
    colvals[5644] = 124;
    colvals[5645] = 125;
    colvals[5646] = 127;
    colvals[5647] = 128;
    colvals[5648] = 129;
    colvals[5649] = 130;
    colvals[5650] = 0;
    colvals[5651] = 25;
    colvals[5652] = 33;
    colvals[5653] = 35;
    colvals[5654] = 36;
    colvals[5655] = 38;
    colvals[5656] = 39;
    colvals[5657] = 50;
    colvals[5658] = 52;
    colvals[5659] = 53;
    colvals[5660] = 54;
    colvals[5661] = 56;
    colvals[5662] = 57;
    colvals[5663] = 61;
    colvals[5664] = 63;
    colvals[5665] = 64;
    colvals[5666] = 99;
    colvals[5667] = 112;
    colvals[5668] = 119;
    colvals[5669] = 127;
    colvals[5670] = 0;
    colvals[5671] = 1;
    colvals[5672] = 2;
    colvals[5673] = 3;
    colvals[5674] = 4;
    colvals[5675] = 5;
    colvals[5676] = 6;
    colvals[5677] = 7;
    colvals[5678] = 8;
    colvals[5679] = 10;
    colvals[5680] = 11;
    colvals[5681] = 13;
    colvals[5682] = 15;
    colvals[5683] = 16;
    colvals[5684] = 17;
    colvals[5685] = 18;
    colvals[5686] = 19;
    colvals[5687] = 20;
    colvals[5688] = 21;
    colvals[5689] = 22;
    colvals[5690] = 23;
    colvals[5691] = 24;
    colvals[5692] = 25;
    colvals[5693] = 26;
    colvals[5694] = 27;
    colvals[5695] = 28;
    colvals[5696] = 29;
    colvals[5697] = 30;
    colvals[5698] = 31;
    colvals[5699] = 32;
    colvals[5700] = 33;
    colvals[5701] = 34;
    colvals[5702] = 35;
    colvals[5703] = 36;
    colvals[5704] = 37;
    colvals[5705] = 38;
    colvals[5706] = 40;
    colvals[5707] = 41;
    colvals[5708] = 42;
    colvals[5709] = 44;
    colvals[5710] = 45;
    colvals[5711] = 46;
    colvals[5712] = 47;
    colvals[5713] = 49;
    colvals[5714] = 50;
    colvals[5715] = 51;
    colvals[5716] = 52;
    colvals[5717] = 53;
    colvals[5718] = 54;
    colvals[5719] = 55;
    colvals[5720] = 56;
    colvals[5721] = 58;
    colvals[5722] = 59;
    colvals[5723] = 60;
    colvals[5724] = 61;
    colvals[5725] = 62;
    colvals[5726] = 63;
    colvals[5727] = 64;
    colvals[5728] = 65;
    colvals[5729] = 67;
    colvals[5730] = 68;
    colvals[5731] = 69;
    colvals[5732] = 70;
    colvals[5733] = 71;
    colvals[5734] = 72;
    colvals[5735] = 73;
    colvals[5736] = 74;
    colvals[5737] = 75;
    colvals[5738] = 76;
    colvals[5739] = 78;
    colvals[5740] = 79;
    colvals[5741] = 80;
    colvals[5742] = 82;
    colvals[5743] = 83;
    colvals[5744] = 84;
    colvals[5745] = 85;
    colvals[5746] = 86;
    colvals[5747] = 87;
    colvals[5748] = 88;
    colvals[5749] = 89;
    colvals[5750] = 90;
    colvals[5751] = 91;
    colvals[5752] = 92;
    colvals[5753] = 93;
    colvals[5754] = 94;
    colvals[5755] = 96;
    colvals[5756] = 97;
    colvals[5757] = 98;
    colvals[5758] = 99;
    colvals[5759] = 100;
    colvals[5760] = 101;
    colvals[5761] = 103;
    colvals[5762] = 105;
    colvals[5763] = 107;
    colvals[5764] = 108;
    colvals[5765] = 109;
    colvals[5766] = 110;
    colvals[5767] = 111;
    colvals[5768] = 112;
    colvals[5769] = 113;
    colvals[5770] = 114;
    colvals[5771] = 115;
    colvals[5772] = 116;
    colvals[5773] = 117;
    colvals[5774] = 118;
    colvals[5775] = 119;
    colvals[5776] = 120;
    colvals[5777] = 121;
    colvals[5778] = 122;
    colvals[5779] = 123;
    colvals[5780] = 124;
    colvals[5781] = 125;
    colvals[5782] = 126;
    colvals[5783] = 127;
    colvals[5784] = 128;
    colvals[5785] = 129;
    colvals[5786] = 130;
    colvals[5787] = 0;
    colvals[5788] = 3;
    colvals[5789] = 15;
    colvals[5790] = 19;
    colvals[5791] = 25;
    colvals[5792] = 29;
    colvals[5793] = 33;
    colvals[5794] = 35;
    colvals[5795] = 36;
    colvals[5796] = 41;
    colvals[5797] = 48;
    colvals[5798] = 50;
    colvals[5799] = 54;
    colvals[5800] = 61;
    colvals[5801] = 62;
    colvals[5802] = 64;
    colvals[5803] = 73;
    colvals[5804] = 75;
    colvals[5805] = 77;
    colvals[5806] = 83;
    colvals[5807] = 84;
    colvals[5808] = 87;
    colvals[5809] = 93;
    colvals[5810] = 95;
    colvals[5811] = 97;
    colvals[5812] = 100;
    colvals[5813] = 103;
    colvals[5814] = 107;
    colvals[5815] = 110;
    colvals[5816] = 113;
    colvals[5817] = 114;
    colvals[5818] = 115;
    colvals[5819] = 116;
    colvals[5820] = 117;
    colvals[5821] = 118;
    colvals[5822] = 119;
    colvals[5823] = 121;
    colvals[5824] = 123;
    colvals[5825] = 124;
    colvals[5826] = 125;
    colvals[5827] = 126;
    colvals[5828] = 127;
    colvals[5829] = 129;
    colvals[5830] = 0;
    colvals[5831] = 1;
    colvals[5832] = 2;
    colvals[5833] = 3;
    colvals[5834] = 4;
    colvals[5835] = 5;
    colvals[5836] = 7;
    colvals[5837] = 15;
    colvals[5838] = 16;
    colvals[5839] = 17;
    colvals[5840] = 19;
    colvals[5841] = 20;
    colvals[5842] = 21;
    colvals[5843] = 23;
    colvals[5844] = 25;
    colvals[5845] = 26;
    colvals[5846] = 29;
    colvals[5847] = 30;
    colvals[5848] = 31;
    colvals[5849] = 33;
    colvals[5850] = 34;
    colvals[5851] = 35;
    colvals[5852] = 36;
    colvals[5853] = 37;
    colvals[5854] = 38;
    colvals[5855] = 39;
    colvals[5856] = 41;
    colvals[5857] = 47;
    colvals[5858] = 48;
    colvals[5859] = 50;
    colvals[5860] = 51;
    colvals[5861] = 52;
    colvals[5862] = 53;
    colvals[5863] = 54;
    colvals[5864] = 55;
    colvals[5865] = 56;
    colvals[5866] = 57;
    colvals[5867] = 59;
    colvals[5868] = 61;
    colvals[5869] = 62;
    colvals[5870] = 63;
    colvals[5871] = 64;
    colvals[5872] = 65;
    colvals[5873] = 70;
    colvals[5874] = 72;
    colvals[5875] = 73;
    colvals[5876] = 74;
    colvals[5877] = 75;
    colvals[5878] = 76;
    colvals[5879] = 77;
    colvals[5880] = 78;
    colvals[5881] = 83;
    colvals[5882] = 84;
    colvals[5883] = 85;
    colvals[5884] = 87;
    colvals[5885] = 88;
    colvals[5886] = 89;
    colvals[5887] = 91;
    colvals[5888] = 93;
    colvals[5889] = 95;
    colvals[5890] = 96;
    colvals[5891] = 97;
    colvals[5892] = 98;
    colvals[5893] = 99;
    colvals[5894] = 100;
    colvals[5895] = 103;
    colvals[5896] = 105;
    colvals[5897] = 107;
    colvals[5898] = 108;
    colvals[5899] = 110;
    colvals[5900] = 111;
    colvals[5901] = 113;
    colvals[5902] = 114;
    colvals[5903] = 115;
    colvals[5904] = 116;
    colvals[5905] = 117;
    colvals[5906] = 118;
    colvals[5907] = 119;
    colvals[5908] = 120;
    colvals[5909] = 121;
    colvals[5910] = 122;
    colvals[5911] = 123;
    colvals[5912] = 124;
    colvals[5913] = 125;
    colvals[5914] = 126;
    colvals[5915] = 127;
    colvals[5916] = 128;
    colvals[5917] = 129;
    colvals[5918] = 130;
    colvals[5919] = 0;
    colvals[5920] = 3;
    colvals[5921] = 5;
    colvals[5922] = 7;
    colvals[5923] = 15;
    colvals[5924] = 17;
    colvals[5925] = 19;
    colvals[5926] = 21;
    colvals[5927] = 23;
    colvals[5928] = 25;
    colvals[5929] = 29;
    colvals[5930] = 31;
    colvals[5931] = 33;
    colvals[5932] = 34;
    colvals[5933] = 35;
    colvals[5934] = 36;
    colvals[5935] = 39;
    colvals[5936] = 41;
    colvals[5937] = 48;
    colvals[5938] = 50;
    colvals[5939] = 52;
    colvals[5940] = 54;
    colvals[5941] = 57;
    colvals[5942] = 59;
    colvals[5943] = 61;
    colvals[5944] = 62;
    colvals[5945] = 64;
    colvals[5946] = 72;
    colvals[5947] = 75;
    colvals[5948] = 77;
    colvals[5949] = 83;
    colvals[5950] = 85;
    colvals[5951] = 87;
    colvals[5952] = 89;
    colvals[5953] = 91;
    colvals[5954] = 93;
    colvals[5955] = 97;
    colvals[5956] = 100;
    colvals[5957] = 107;
    colvals[5958] = 110;
    colvals[5959] = 113;
    colvals[5960] = 114;
    colvals[5961] = 115;
    colvals[5962] = 116;
    colvals[5963] = 117;
    colvals[5964] = 118;
    colvals[5965] = 119;
    colvals[5966] = 123;
    colvals[5967] = 125;
    colvals[5968] = 126;
    colvals[5969] = 127;
    colvals[5970] = 0;
    colvals[5971] = 3;
    colvals[5972] = 15;
    colvals[5973] = 19;
    colvals[5974] = 25;
    colvals[5975] = 29;
    colvals[5976] = 33;
    colvals[5977] = 35;
    colvals[5978] = 36;
    colvals[5979] = 41;
    colvals[5980] = 48;
    colvals[5981] = 50;
    colvals[5982] = 52;
    colvals[5983] = 54;
    colvals[5984] = 59;
    colvals[5985] = 61;
    colvals[5986] = 62;
    colvals[5987] = 64;
    colvals[5988] = 73;
    colvals[5989] = 74;
    colvals[5990] = 75;
    colvals[5991] = 77;
    colvals[5992] = 83;
    colvals[5993] = 84;
    colvals[5994] = 87;
    colvals[5995] = 88;
    colvals[5996] = 93;
    colvals[5997] = 95;
    colvals[5998] = 97;
    colvals[5999] = 100;
    colvals[6000] = 103;
    colvals[6001] = 105;
    colvals[6002] = 107;
    colvals[6003] = 110;
    colvals[6004] = 113;
    colvals[6005] = 114;
    colvals[6006] = 115;
    colvals[6007] = 116;
    colvals[6008] = 117;
    colvals[6009] = 118;
    colvals[6010] = 119;
    colvals[6011] = 120;
    colvals[6012] = 121;
    colvals[6013] = 122;
    colvals[6014] = 123;
    colvals[6015] = 125;
    colvals[6016] = 126;
    colvals[6017] = 127;
    colvals[6018] = 128;
    colvals[6019] = 129;
    colvals[6020] = 130;
    colvals[6021] = 0;
    colvals[6022] = 3;
    colvals[6023] = 15;
    colvals[6024] = 19;
    colvals[6025] = 25;
    colvals[6026] = 29;
    colvals[6027] = 33;
    colvals[6028] = 35;
    colvals[6029] = 36;
    colvals[6030] = 41;
    colvals[6031] = 48;
    colvals[6032] = 50;
    colvals[6033] = 54;
    colvals[6034] = 61;
    colvals[6035] = 62;
    colvals[6036] = 64;
    colvals[6037] = 73;
    colvals[6038] = 75;
    colvals[6039] = 77;
    colvals[6040] = 83;
    colvals[6041] = 84;
    colvals[6042] = 87;
    colvals[6043] = 93;
    colvals[6044] = 95;
    colvals[6045] = 97;
    colvals[6046] = 100;
    colvals[6047] = 103;
    colvals[6048] = 107;
    colvals[6049] = 110;
    colvals[6050] = 113;
    colvals[6051] = 114;
    colvals[6052] = 115;
    colvals[6053] = 116;
    colvals[6054] = 117;
    colvals[6055] = 118;
    colvals[6056] = 119;
    colvals[6057] = 121;
    colvals[6058] = 123;
    colvals[6059] = 124;
    colvals[6060] = 125;
    colvals[6061] = 126;
    colvals[6062] = 127;
    colvals[6063] = 129;
    colvals[6064] = 0;
    colvals[6065] = 1;
    colvals[6066] = 2;
    colvals[6067] = 3;
    colvals[6068] = 4;
    colvals[6069] = 5;
    colvals[6070] = 7;
    colvals[6071] = 15;
    colvals[6072] = 16;
    colvals[6073] = 17;
    colvals[6074] = 19;
    colvals[6075] = 20;
    colvals[6076] = 21;
    colvals[6077] = 23;
    colvals[6078] = 25;
    colvals[6079] = 26;
    colvals[6080] = 29;
    colvals[6081] = 30;
    colvals[6082] = 31;
    colvals[6083] = 33;
    colvals[6084] = 34;
    colvals[6085] = 35;
    colvals[6086] = 36;
    colvals[6087] = 37;
    colvals[6088] = 38;
    colvals[6089] = 39;
    colvals[6090] = 41;
    colvals[6091] = 47;
    colvals[6092] = 48;
    colvals[6093] = 50;
    colvals[6094] = 51;
    colvals[6095] = 52;
    colvals[6096] = 53;
    colvals[6097] = 54;
    colvals[6098] = 55;
    colvals[6099] = 56;
    colvals[6100] = 57;
    colvals[6101] = 59;
    colvals[6102] = 61;
    colvals[6103] = 62;
    colvals[6104] = 63;
    colvals[6105] = 64;
    colvals[6106] = 65;
    colvals[6107] = 70;
    colvals[6108] = 72;
    colvals[6109] = 73;
    colvals[6110] = 74;
    colvals[6111] = 75;
    colvals[6112] = 76;
    colvals[6113] = 77;
    colvals[6114] = 78;
    colvals[6115] = 83;
    colvals[6116] = 84;
    colvals[6117] = 85;
    colvals[6118] = 87;
    colvals[6119] = 88;
    colvals[6120] = 89;
    colvals[6121] = 91;
    colvals[6122] = 93;
    colvals[6123] = 95;
    colvals[6124] = 96;
    colvals[6125] = 97;
    colvals[6126] = 98;
    colvals[6127] = 99;
    colvals[6128] = 100;
    colvals[6129] = 103;
    colvals[6130] = 105;
    colvals[6131] = 107;
    colvals[6132] = 108;
    colvals[6133] = 110;
    colvals[6134] = 111;
    colvals[6135] = 113;
    colvals[6136] = 114;
    colvals[6137] = 115;
    colvals[6138] = 116;
    colvals[6139] = 117;
    colvals[6140] = 118;
    colvals[6141] = 119;
    colvals[6142] = 120;
    colvals[6143] = 121;
    colvals[6144] = 122;
    colvals[6145] = 123;
    colvals[6146] = 124;
    colvals[6147] = 125;
    colvals[6148] = 126;
    colvals[6149] = 127;
    colvals[6150] = 128;
    colvals[6151] = 129;
    colvals[6152] = 130;
    colvals[6153] = 0;
    colvals[6154] = 3;
    colvals[6155] = 5;
    colvals[6156] = 7;
    colvals[6157] = 15;
    colvals[6158] = 17;
    colvals[6159] = 19;
    colvals[6160] = 21;
    colvals[6161] = 23;
    colvals[6162] = 25;
    colvals[6163] = 29;
    colvals[6164] = 31;
    colvals[6165] = 33;
    colvals[6166] = 35;
    colvals[6167] = 36;
    colvals[6168] = 39;
    colvals[6169] = 41;
    colvals[6170] = 50;
    colvals[6171] = 51;
    colvals[6172] = 52;
    colvals[6173] = 54;
    colvals[6174] = 57;
    colvals[6175] = 59;
    colvals[6176] = 61;
    colvals[6177] = 64;
    colvals[6178] = 72;
    colvals[6179] = 75;
    colvals[6180] = 77;
    colvals[6181] = 83;
    colvals[6182] = 85;
    colvals[6183] = 87;
    colvals[6184] = 89;
    colvals[6185] = 91;
    colvals[6186] = 93;
    colvals[6187] = 97;
    colvals[6188] = 100;
    colvals[6189] = 107;
    colvals[6190] = 110;
    colvals[6191] = 113;
    colvals[6192] = 115;
    colvals[6193] = 119;
    colvals[6194] = 120;
    colvals[6195] = 121;
    colvals[6196] = 122;
    colvals[6197] = 123;
    colvals[6198] = 127;
    colvals[6199] = 129;
    colvals[6200] = 130;
    colvals[6201] = 0;
    colvals[6202] = 3;
    colvals[6203] = 15;
    colvals[6204] = 19;
    colvals[6205] = 25;
    colvals[6206] = 29;
    colvals[6207] = 33;
    colvals[6208] = 35;
    colvals[6209] = 36;
    colvals[6210] = 41;
    colvals[6211] = 48;
    colvals[6212] = 50;
    colvals[6213] = 52;
    colvals[6214] = 54;
    colvals[6215] = 59;
    colvals[6216] = 61;
    colvals[6217] = 62;
    colvals[6218] = 64;
    colvals[6219] = 73;
    colvals[6220] = 74;
    colvals[6221] = 75;
    colvals[6222] = 77;
    colvals[6223] = 83;
    colvals[6224] = 84;
    colvals[6225] = 87;
    colvals[6226] = 88;
    colvals[6227] = 93;
    colvals[6228] = 95;
    colvals[6229] = 97;
    colvals[6230] = 100;
    colvals[6231] = 103;
    colvals[6232] = 105;
    colvals[6233] = 107;
    colvals[6234] = 110;
    colvals[6235] = 113;
    colvals[6236] = 114;
    colvals[6237] = 115;
    colvals[6238] = 116;
    colvals[6239] = 117;
    colvals[6240] = 118;
    colvals[6241] = 119;
    colvals[6242] = 120;
    colvals[6243] = 121;
    colvals[6244] = 122;
    colvals[6245] = 123;
    colvals[6246] = 124;
    colvals[6247] = 125;
    colvals[6248] = 126;
    colvals[6249] = 127;
    colvals[6250] = 128;
    colvals[6251] = 129;
    colvals[6252] = 130;
    colvals[6253] = 0;
    colvals[6254] = 3;
    colvals[6255] = 15;
    colvals[6256] = 19;
    colvals[6257] = 25;
    colvals[6258] = 29;
    colvals[6259] = 33;
    colvals[6260] = 35;
    colvals[6261] = 36;
    colvals[6262] = 48;
    colvals[6263] = 50;
    colvals[6264] = 52;
    colvals[6265] = 54;
    colvals[6266] = 59;
    colvals[6267] = 61;
    colvals[6268] = 62;
    colvals[6269] = 64;
    colvals[6270] = 74;
    colvals[6271] = 75;
    colvals[6272] = 77;
    colvals[6273] = 83;
    colvals[6274] = 87;
    colvals[6275] = 88;
    colvals[6276] = 93;
    colvals[6277] = 95;
    colvals[6278] = 97;
    colvals[6279] = 100;
    colvals[6280] = 105;
    colvals[6281] = 107;
    colvals[6282] = 110;
    colvals[6283] = 113;
    colvals[6284] = 115;
    colvals[6285] = 117;
    colvals[6286] = 119;
    colvals[6287] = 120;
    colvals[6288] = 121;
    colvals[6289] = 122;
    colvals[6290] = 123;
    colvals[6291] = 125;
    colvals[6292] = 127;
    colvals[6293] = 128;
    colvals[6294] = 129;
    colvals[6295] = 130;
    colvals[6296] = 0;
    colvals[6297] = 1;
    colvals[6298] = 2;
    colvals[6299] = 3;
    colvals[6300] = 4;
    colvals[6301] = 5;
    colvals[6302] = 7;
    colvals[6303] = 15;
    colvals[6304] = 16;
    colvals[6305] = 17;
    colvals[6306] = 18;
    colvals[6307] = 19;
    colvals[6308] = 20;
    colvals[6309] = 21;
    colvals[6310] = 23;
    colvals[6311] = 25;
    colvals[6312] = 26;
    colvals[6313] = 29;
    colvals[6314] = 30;
    colvals[6315] = 31;
    colvals[6316] = 33;
    colvals[6317] = 34;
    colvals[6318] = 35;
    colvals[6319] = 36;
    colvals[6320] = 37;
    colvals[6321] = 38;
    colvals[6322] = 39;
    colvals[6323] = 41;
    colvals[6324] = 42;
    colvals[6325] = 45;
    colvals[6326] = 47;
    colvals[6327] = 48;
    colvals[6328] = 50;
    colvals[6329] = 51;
    colvals[6330] = 52;
    colvals[6331] = 53;
    colvals[6332] = 54;
    colvals[6333] = 55;
    colvals[6334] = 56;
    colvals[6335] = 57;
    colvals[6336] = 59;
    colvals[6337] = 61;
    colvals[6338] = 62;
    colvals[6339] = 63;
    colvals[6340] = 64;
    colvals[6341] = 65;
    colvals[6342] = 70;
    colvals[6343] = 72;
    colvals[6344] = 73;
    colvals[6345] = 74;
    colvals[6346] = 75;
    colvals[6347] = 76;
    colvals[6348] = 77;
    colvals[6349] = 78;
    colvals[6350] = 83;
    colvals[6351] = 84;
    colvals[6352] = 85;
    colvals[6353] = 87;
    colvals[6354] = 88;
    colvals[6355] = 89;
    colvals[6356] = 91;
    colvals[6357] = 93;
    colvals[6358] = 95;
    colvals[6359] = 96;
    colvals[6360] = 97;
    colvals[6361] = 98;
    colvals[6362] = 99;
    colvals[6363] = 100;
    colvals[6364] = 102;
    colvals[6365] = 103;
    colvals[6366] = 105;
    colvals[6367] = 107;
    colvals[6368] = 108;
    colvals[6369] = 110;
    colvals[6370] = 111;
    colvals[6371] = 113;
    colvals[6372] = 114;
    colvals[6373] = 115;
    colvals[6374] = 116;
    colvals[6375] = 117;
    colvals[6376] = 118;
    colvals[6377] = 119;
    colvals[6378] = 120;
    colvals[6379] = 121;
    colvals[6380] = 122;
    colvals[6381] = 123;
    colvals[6382] = 124;
    colvals[6383] = 125;
    colvals[6384] = 126;
    colvals[6385] = 127;
    colvals[6386] = 128;
    colvals[6387] = 129;
    colvals[6388] = 130;
    colvals[6389] = 0;
    colvals[6390] = 3;
    colvals[6391] = 5;
    colvals[6392] = 7;
    colvals[6393] = 15;
    colvals[6394] = 17;
    colvals[6395] = 19;
    colvals[6396] = 21;
    colvals[6397] = 23;
    colvals[6398] = 25;
    colvals[6399] = 29;
    colvals[6400] = 31;
    colvals[6401] = 33;
    colvals[6402] = 34;
    colvals[6403] = 35;
    colvals[6404] = 36;
    colvals[6405] = 39;
    colvals[6406] = 41;
    colvals[6407] = 48;
    colvals[6408] = 50;
    colvals[6409] = 52;
    colvals[6410] = 54;
    colvals[6411] = 57;
    colvals[6412] = 59;
    colvals[6413] = 61;
    colvals[6414] = 62;
    colvals[6415] = 64;
    colvals[6416] = 72;
    colvals[6417] = 73;
    colvals[6418] = 75;
    colvals[6419] = 77;
    colvals[6420] = 83;
    colvals[6421] = 85;
    colvals[6422] = 87;
    colvals[6423] = 89;
    colvals[6424] = 91;
    colvals[6425] = 93;
    colvals[6426] = 97;
    colvals[6427] = 100;
    colvals[6428] = 107;
    colvals[6429] = 110;
    colvals[6430] = 113;
    colvals[6431] = 114;
    colvals[6432] = 115;
    colvals[6433] = 117;
    colvals[6434] = 118;
    colvals[6435] = 119;
    colvals[6436] = 123;
    colvals[6437] = 124;
    colvals[6438] = 125;
    colvals[6439] = 126;
    colvals[6440] = 127;
    colvals[6441] = 0;
    colvals[6442] = 3;
    colvals[6443] = 15;
    colvals[6444] = 19;
    colvals[6445] = 25;
    colvals[6446] = 29;
    colvals[6447] = 33;
    colvals[6448] = 35;
    colvals[6449] = 36;
    colvals[6450] = 41;
    colvals[6451] = 48;
    colvals[6452] = 50;
    colvals[6453] = 52;
    colvals[6454] = 54;
    colvals[6455] = 59;
    colvals[6456] = 61;
    colvals[6457] = 62;
    colvals[6458] = 64;
    colvals[6459] = 73;
    colvals[6460] = 74;
    colvals[6461] = 75;
    colvals[6462] = 77;
    colvals[6463] = 83;
    colvals[6464] = 84;
    colvals[6465] = 87;
    colvals[6466] = 88;
    colvals[6467] = 93;
    colvals[6468] = 95;
    colvals[6469] = 97;
    colvals[6470] = 100;
    colvals[6471] = 103;
    colvals[6472] = 105;
    colvals[6473] = 107;
    colvals[6474] = 110;
    colvals[6475] = 113;
    colvals[6476] = 114;
    colvals[6477] = 115;
    colvals[6478] = 116;
    colvals[6479] = 117;
    colvals[6480] = 118;
    colvals[6481] = 119;
    colvals[6482] = 120;
    colvals[6483] = 121;
    colvals[6484] = 122;
    colvals[6485] = 123;
    colvals[6486] = 124;
    colvals[6487] = 125;
    colvals[6488] = 126;
    colvals[6489] = 127;
    colvals[6490] = 128;
    colvals[6491] = 129;
    colvals[6492] = 130;
    colvals[6493] = 0;
    colvals[6494] = 3;
    colvals[6495] = 15;
    colvals[6496] = 25;
    colvals[6497] = 29;
    colvals[6498] = 35;
    colvals[6499] = 36;
    colvals[6500] = 41;
    colvals[6501] = 48;
    colvals[6502] = 54;
    colvals[6503] = 61;
    colvals[6504] = 62;
    colvals[6505] = 64;
    colvals[6506] = 73;
    colvals[6507] = 75;
    colvals[6508] = 77;
    colvals[6509] = 83;
    colvals[6510] = 84;
    colvals[6511] = 93;
    colvals[6512] = 95;
    colvals[6513] = 97;
    colvals[6514] = 100;
    colvals[6515] = 103;
    colvals[6516] = 107;
    colvals[6517] = 113;
    colvals[6518] = 114;
    colvals[6519] = 115;
    colvals[6520] = 116;
    colvals[6521] = 117;
    colvals[6522] = 118;
    colvals[6523] = 119;
    colvals[6524] = 121;
    colvals[6525] = 123;
    colvals[6526] = 124;
    colvals[6527] = 125;
    colvals[6528] = 126;
    colvals[6529] = 127;
    colvals[6530] = 129;
    colvals[6531] = 0;
    colvals[6532] = 1;
    colvals[6533] = 2;
    colvals[6534] = 3;
    colvals[6535] = 4;
    colvals[6536] = 5;
    colvals[6537] = 7;
    colvals[6538] = 15;
    colvals[6539] = 16;
    colvals[6540] = 17;
    colvals[6541] = 19;
    colvals[6542] = 20;
    colvals[6543] = 21;
    colvals[6544] = 22;
    colvals[6545] = 23;
    colvals[6546] = 25;
    colvals[6547] = 26;
    colvals[6548] = 29;
    colvals[6549] = 30;
    colvals[6550] = 31;
    colvals[6551] = 33;
    colvals[6552] = 34;
    colvals[6553] = 35;
    colvals[6554] = 36;
    colvals[6555] = 37;
    colvals[6556] = 38;
    colvals[6557] = 39;
    colvals[6558] = 41;
    colvals[6559] = 47;
    colvals[6560] = 48;
    colvals[6561] = 50;
    colvals[6562] = 51;
    colvals[6563] = 52;
    colvals[6564] = 53;
    colvals[6565] = 54;
    colvals[6566] = 55;
    colvals[6567] = 56;
    colvals[6568] = 57;
    colvals[6569] = 59;
    colvals[6570] = 60;
    colvals[6571] = 61;
    colvals[6572] = 62;
    colvals[6573] = 63;
    colvals[6574] = 64;
    colvals[6575] = 65;
    colvals[6576] = 68;
    colvals[6577] = 70;
    colvals[6578] = 72;
    colvals[6579] = 73;
    colvals[6580] = 74;
    colvals[6581] = 75;
    colvals[6582] = 76;
    colvals[6583] = 77;
    colvals[6584] = 78;
    colvals[6585] = 83;
    colvals[6586] = 84;
    colvals[6587] = 85;
    colvals[6588] = 87;
    colvals[6589] = 88;
    colvals[6590] = 89;
    colvals[6591] = 91;
    colvals[6592] = 93;
    colvals[6593] = 95;
    colvals[6594] = 96;
    colvals[6595] = 97;
    colvals[6596] = 98;
    colvals[6597] = 99;
    colvals[6598] = 100;
    colvals[6599] = 103;
    colvals[6600] = 104;
    colvals[6601] = 105;
    colvals[6602] = 107;
    colvals[6603] = 108;
    colvals[6604] = 110;
    colvals[6605] = 111;
    colvals[6606] = 113;
    colvals[6607] = 114;
    colvals[6608] = 115;
    colvals[6609] = 116;
    colvals[6610] = 117;
    colvals[6611] = 118;
    colvals[6612] = 119;
    colvals[6613] = 120;
    colvals[6614] = 121;
    colvals[6615] = 122;
    colvals[6616] = 123;
    colvals[6617] = 124;
    colvals[6618] = 125;
    colvals[6619] = 126;
    colvals[6620] = 127;
    colvals[6621] = 128;
    colvals[6622] = 129;
    colvals[6623] = 130;
    colvals[6624] = 0;
    colvals[6625] = 3;
    colvals[6626] = 5;
    colvals[6627] = 7;
    colvals[6628] = 15;
    colvals[6629] = 17;
    colvals[6630] = 19;
    colvals[6631] = 21;
    colvals[6632] = 23;
    colvals[6633] = 25;
    colvals[6634] = 29;
    colvals[6635] = 31;
    colvals[6636] = 33;
    colvals[6637] = 35;
    colvals[6638] = 36;
    colvals[6639] = 39;
    colvals[6640] = 41;
    colvals[6641] = 50;
    colvals[6642] = 51;
    colvals[6643] = 52;
    colvals[6644] = 54;
    colvals[6645] = 57;
    colvals[6646] = 59;
    colvals[6647] = 61;
    colvals[6648] = 62;
    colvals[6649] = 64;
    colvals[6650] = 72;
    colvals[6651] = 74;
    colvals[6652] = 75;
    colvals[6653] = 77;
    colvals[6654] = 83;
    colvals[6655] = 85;
    colvals[6656] = 87;
    colvals[6657] = 89;
    colvals[6658] = 91;
    colvals[6659] = 93;
    colvals[6660] = 97;
    colvals[6661] = 100;
    colvals[6662] = 107;
    colvals[6663] = 110;
    colvals[6664] = 113;
    colvals[6665] = 115;
    colvals[6666] = 119;
    colvals[6667] = 121;
    colvals[6668] = 122;
    colvals[6669] = 123;
    colvals[6670] = 127;
    colvals[6671] = 128;
    colvals[6672] = 129;
    colvals[6673] = 130;
    colvals[6674] = 0;
    colvals[6675] = 3;
    colvals[6676] = 15;
    colvals[6677] = 19;
    colvals[6678] = 25;
    colvals[6679] = 29;
    colvals[6680] = 33;
    colvals[6681] = 35;
    colvals[6682] = 36;
    colvals[6683] = 41;
    colvals[6684] = 48;
    colvals[6685] = 50;
    colvals[6686] = 52;
    colvals[6687] = 54;
    colvals[6688] = 59;
    colvals[6689] = 61;
    colvals[6690] = 62;
    colvals[6691] = 64;
    colvals[6692] = 73;
    colvals[6693] = 74;
    colvals[6694] = 75;
    colvals[6695] = 77;
    colvals[6696] = 83;
    colvals[6697] = 84;
    colvals[6698] = 87;
    colvals[6699] = 88;
    colvals[6700] = 93;
    colvals[6701] = 95;
    colvals[6702] = 97;
    colvals[6703] = 100;
    colvals[6704] = 103;
    colvals[6705] = 105;
    colvals[6706] = 107;
    colvals[6707] = 110;
    colvals[6708] = 113;
    colvals[6709] = 114;
    colvals[6710] = 115;
    colvals[6711] = 116;
    colvals[6712] = 117;
    colvals[6713] = 118;
    colvals[6714] = 119;
    colvals[6715] = 120;
    colvals[6716] = 121;
    colvals[6717] = 122;
    colvals[6718] = 123;
    colvals[6719] = 124;
    colvals[6720] = 125;
    colvals[6721] = 126;
    colvals[6722] = 127;
    colvals[6723] = 128;
    colvals[6724] = 129;
    colvals[6725] = 130;
    colvals[6726] = 0;
    colvals[6727] = 3;
    colvals[6728] = 15;
    colvals[6729] = 19;
    colvals[6730] = 25;
    colvals[6731] = 29;
    colvals[6732] = 33;
    colvals[6733] = 35;
    colvals[6734] = 36;
    colvals[6735] = 48;
    colvals[6736] = 50;
    colvals[6737] = 52;
    colvals[6738] = 54;
    colvals[6739] = 59;
    colvals[6740] = 61;
    colvals[6741] = 62;
    colvals[6742] = 64;
    colvals[6743] = 74;
    colvals[6744] = 75;
    colvals[6745] = 77;
    colvals[6746] = 83;
    colvals[6747] = 87;
    colvals[6748] = 88;
    colvals[6749] = 93;
    colvals[6750] = 95;
    colvals[6751] = 97;
    colvals[6752] = 100;
    colvals[6753] = 105;
    colvals[6754] = 107;
    colvals[6755] = 110;
    colvals[6756] = 113;
    colvals[6757] = 115;
    colvals[6758] = 117;
    colvals[6759] = 119;
    colvals[6760] = 120;
    colvals[6761] = 121;
    colvals[6762] = 122;
    colvals[6763] = 123;
    colvals[6764] = 125;
    colvals[6765] = 127;
    colvals[6766] = 128;
    colvals[6767] = 129;
    colvals[6768] = 130;
    
    // value of each non-zero element
    data[0] = 0.0 - k[202] - k[351] - k[630]*y[IDX_pH2II] -
        k[631]*y[IDX_oH2II] - k[632]*y[IDX_pD2II] - k[633]*y[IDX_oD2II] -
        k[634]*y[IDX_HDII] - k[635]*y[IDX_HDII] - k[948]*y[IDX_C2HII] -
        k[949]*y[IDX_C2DII] - k[964]*y[IDX_CH2II] - k[965]*y[IDX_CD2II] -
        k[966]*y[IDX_CHDII] - k[967]*y[IDX_CHDII] - k[987]*y[IDX_H2OII] -
        k[988]*y[IDX_D2OII] - k[989]*y[IDX_HDOII] - k[990]*y[IDX_HDOII] -
        k[998]*y[IDX_HCNII] - k[999]*y[IDX_DCNII] - k[1016]*y[IDX_HCOII] -
        k[1017]*y[IDX_DCOII] - k[1034]*y[IDX_CH2I] - k[1035]*y[IDX_CD2I] -
        k[1036]*y[IDX_CHDI] - k[1037]*y[IDX_CH2I] - k[1038]*y[IDX_CD2I] -
        k[1039]*y[IDX_CHDI] - k[1040]*y[IDX_CHDI] - k[1041]*y[IDX_NH2I] -
        k[1042]*y[IDX_ND2I] - k[1043]*y[IDX_NHDI] - k[1044]*y[IDX_NHDI] -
        k[1045]*y[IDX_NH2I] - k[1046]*y[IDX_ND2I] - k[1047]*y[IDX_NHDI] -
        k[1048]*y[IDX_NHDI] - k[1049]*y[IDX_NH2I] - k[1050]*y[IDX_ND2I] -
        k[1051]*y[IDX_NHDI] - k[1052]*y[IDX_NHDI] - k[1053]*y[IDX_CNI] -
        k[1054]*y[IDX_N2I] - k[1055]*y[IDX_COI] - k[1130]*y[IDX_pH2I] -
        k[1131]*y[IDX_oH2I] - k[1132]*y[IDX_pD2I] - k[1133]*y[IDX_oD2I] -
        k[1134]*y[IDX_HDI] - k[1135]*y[IDX_HDI] - k[1256]*y[IDX_oH3II] -
        k[1257]*y[IDX_pH3II] - k[1258]*y[IDX_pH3II] - k[1259]*y[IDX_oD3II] -
        k[1260]*y[IDX_mD3II] - k[1261]*y[IDX_oH2DII] - k[1262]*y[IDX_pH2DII] -
        k[1263]*y[IDX_pH2DII] - k[1264]*y[IDX_oH2DII] - k[1265]*y[IDX_pH2DII] -
        k[1266]*y[IDX_oD2HII] - k[1267]*y[IDX_pD2HII] - k[1268]*y[IDX_pD2HII] -
        k[1269]*y[IDX_oD2HII] - k[1270]*y[IDX_pD2HII] - k[1552]*y[IDX_HNCII] -
        k[1553]*y[IDX_DNCII] - k[1570]*y[IDX_HNOII] - k[1571]*y[IDX_DNOII] -
        k[1616]*y[IDX_N2HII] - k[1617]*y[IDX_N2DII] - k[1711]*y[IDX_CHII] -
        k[1712]*y[IDX_CDII] - k[1818]*y[IDX_NHII] - k[1819]*y[IDX_NDII] -
        k[1910]*y[IDX_O2II] - k[1921]*y[IDX_OHII] - k[1922]*y[IDX_ODII] -
        k[2002]*y[IDX_O2HII] - k[2003]*y[IDX_O2DII] - k[2046]*y[IDX_CHI] -
        k[2047]*y[IDX_CDI] - k[2048]*y[IDX_NHI] - k[2049]*y[IDX_NDI] -
        k[2050]*y[IDX_NOI] - k[2051]*y[IDX_NOI] - k[2052]*y[IDX_O2I] -
        k[2053]*y[IDX_OHI] - k[2054]*y[IDX_ODI] - k[2055]*y[IDX_HCOI] -
        k[2056]*y[IDX_DCOI] - k[2057]*y[IDX_HCOI] - k[2058]*y[IDX_DCOI] -
        k[2059]*y[IDX_C2NI] - k[2060]*y[IDX_C2HI] - k[2061]*y[IDX_C2DI] -
        k[2062]*y[IDX_CCOI] - k[2063]*y[IDX_OCNI] - k[2081]*y[IDX_COII] -
        k[2128]*y[IDX_N2II] - k[2259]*y[IDX_C2II] - k[2263]*y[IDX_CNII] -
        k[2557]*y[IDX_O2II] - k[2652]*y[IDX_CI] - k[2652]*y[IDX_CI] -
        k[2652]*y[IDX_CI] - k[2652]*y[IDX_CI] - k[2653]*y[IDX_HI] -
        k[2654]*y[IDX_DI] - k[2655]*y[IDX_NI] - k[2656]*y[IDX_OI] -
        k[2657]*y[IDX_pH2I] - k[2658]*y[IDX_oH2I] - k[2659]*y[IDX_pD2I] -
        k[2660]*y[IDX_oD2I] - k[2661]*y[IDX_HDI] - k[2665]*y[IDX_C2I] -
        k[2670]*y[IDX_CM] - k[2687]*y[IDX_HM] - k[2688]*y[IDX_DM] -
        k[2715]*y[IDX_OM] - k[2730]*y[IDX_OHM] - k[2731]*y[IDX_ODM] -
        k[2736]*y[IDX_eM] - k[2956]*y[IDX_pD3II] - k[2957]*y[IDX_pD3II] -
        k[3010]*y[IDX_H3OII] - k[3011]*y[IDX_H3OII] - k[3283]*y[IDX_H2DOII] -
        k[3284]*y[IDX_H2DOII] - k[3285]*y[IDX_H2DOII] - k[3286]*y[IDX_HD2OII] -
        k[3287]*y[IDX_HD2OII] - k[3288]*y[IDX_HD2OII] - k[3289]*y[IDX_D3OII] -
        k[3290]*y[IDX_D3OII];
    data[1] = 0.0 + k[429]*y[IDX_CCOI] + k[2065]*y[IDX_NOI] +
        k[2132]*y[IDX_CM] + k[2132]*y[IDX_CM] + k[2133]*y[IDX_HM] +
        k[2134]*y[IDX_DM] + k[2309]*y[IDX_CHI] + k[2310]*y[IDX_CDI] +
        k[2311]*y[IDX_CH2I] + k[2312]*y[IDX_CD2I] + k[2313]*y[IDX_CHDI] +
        k[2314]*y[IDX_HCOI] + k[2315]*y[IDX_DCOI] + k[2845]*y[IDX_eM] +
        k[2918]*y[IDX_GRAINM];
    data[2] = 0.0 + k[344] + k[2132]*y[IDX_CII] + k[2132]*y[IDX_CII] +
        k[2135]*y[IDX_HII] + k[2136]*y[IDX_DII] + k[2141]*y[IDX_HeII] +
        k[2144]*y[IDX_NII] + k[2147]*y[IDX_OII] - k[2670]*y[IDX_CI] +
        k[3017]*y[IDX_H3OII] + k[3336]*y[IDX_H2DOII] + k[3337]*y[IDX_H2DOII] +
        k[3338]*y[IDX_HD2OII] + k[3339]*y[IDX_HD2OII] + k[3340]*y[IDX_D3OII];
    data[3] = 0.0 + k[281] + k[281] + k[352] + k[352] + k[862]*y[IDX_HeII] +
        k[1196]*y[IDX_NI] + k[1226]*y[IDX_OI] + k[1652]*y[IDX_OII] +
        k[1691]*y[IDX_C2II] + k[1828]*y[IDX_NHII] + k[1829]*y[IDX_NDII] -
        k[2665]*y[IDX_CI];
    data[4] = 0.0 + k[287] + k[1690]*y[IDX_OI] + k[1691]*y[IDX_C2I] -
        k[2259]*y[IDX_CI] + k[2740]*y[IDX_eM] + k[2740]*y[IDX_eM];
    data[5] = 0.0 + k[887]*y[IDX_HeII] - k[2061]*y[IDX_CI];
    data[6] = 0.0 - k[949]*y[IDX_CI] + k[955]*y[IDX_OI] + k[2768]*y[IDX_eM]
        + k[2770]*y[IDX_eM] + k[2770]*y[IDX_eM];
    data[7] = 0.0 + k[886]*y[IDX_HeII] - k[2060]*y[IDX_CI];
    data[8] = 0.0 - k[948]*y[IDX_CI] + k[954]*y[IDX_OI] + k[2767]*y[IDX_eM]
        + k[2769]*y[IDX_eM] + k[2769]*y[IDX_eM];
    data[9] = 0.0 + k[245] + k[383] - k[2059]*y[IDX_CI];
    data[10] = 0.0 + k[2771]*y[IDX_eM];
    data[11] = 0.0 + k[2773]*y[IDX_eM];
    data[12] = 0.0 + k[246] + k[384] + k[890]*y[IDX_HeII];
    data[13] = 0.0 + k[2774]*y[IDX_eM];
    data[14] = 0.0 + k[248] + k[429]*y[IDX_CII] - k[2062]*y[IDX_CI];
    data[15] = 0.0 + k[283] + k[355] + k[611]*y[IDX_COII] +
        k[1065]*y[IDX_HI] + k[1067]*y[IDX_DI] - k[2047]*y[IDX_CI] +
        k[2310]*y[IDX_CII];
    data[16] = 0.0 + k[231] - k[1712]*y[IDX_CI] + k[2742]*y[IDX_eM] +
        k[3306]*y[IDX_H2OI] + k[3308]*y[IDX_HDOI] + k[3310]*y[IDX_D2OI];
    data[17] = 0.0 - k[1035]*y[IDX_CI] - k[1038]*y[IDX_CI] +
        k[2312]*y[IDX_CII];
    data[18] = 0.0 - k[965]*y[IDX_CI] + k[2776]*y[IDX_eM] +
        k[2783]*y[IDX_eM];
    data[19] = 0.0 + k[282] + k[354] + k[610]*y[IDX_COII] +
        k[1064]*y[IDX_HI] + k[1066]*y[IDX_DI] - k[2046]*y[IDX_CI] +
        k[2309]*y[IDX_CII];
    data[20] = 0.0 + k[230] - k[1711]*y[IDX_CI] + k[2741]*y[IDX_eM] +
        k[3013]*y[IDX_H2OI] + k[3307]*y[IDX_HDOI] + k[3309]*y[IDX_D2OI];
    data[21] = 0.0 - k[1034]*y[IDX_CI] - k[1037]*y[IDX_CI] +
        k[2311]*y[IDX_CII];
    data[22] = 0.0 - k[964]*y[IDX_CI] + k[2775]*y[IDX_eM] +
        k[2782]*y[IDX_eM];
    data[23] = 0.0 - k[1036]*y[IDX_CI] - k[1039]*y[IDX_CI] -
        k[1040]*y[IDX_CI] + k[2313]*y[IDX_CII];
    data[24] = 0.0 - k[966]*y[IDX_CI] - k[967]*y[IDX_CI] + k[2777]*y[IDX_eM]
        + k[2784]*y[IDX_eM];
    data[25] = 0.0 + k[284] + k[358] + k[866]*y[IDX_HeII] -
        k[1053]*y[IDX_CI] + k[1199]*y[IDX_NI] + k[1655]*y[IDX_OII];
    data[26] = 0.0 - k[2263]*y[IDX_CI] + k[2743]*y[IDX_eM];
    data[27] = 0.0 + k[2785]*y[IDX_eM];
    data[28] = 0.0 + k[285] + k[359] - k[1055]*y[IDX_CI] +
        k[2064]*y[IDX_NII];
    data[29] = 0.0 + k[609]*y[IDX_NI] + k[610]*y[IDX_CHI] +
        k[611]*y[IDX_CDI] - k[2081]*y[IDX_CI] + k[2744]*y[IDX_eM];
    data[30] = 0.0 + k[902]*y[IDX_HeII];
    data[31] = 0.0 + k[1066]*y[IDX_CHI] + k[1067]*y[IDX_CDI] -
        k[2654]*y[IDX_CI];
    data[32] = 0.0 + k[2136]*y[IDX_CM];
    data[33] = 0.0 + k[2134]*y[IDX_CII] - k[2688]*y[IDX_CI];
    data[34] = 0.0 + k[3309]*y[IDX_CHII] + k[3310]*y[IDX_CDII];
    data[35] = 0.0 - k[988]*y[IDX_CI];
    data[36] = 0.0 - k[3289]*y[IDX_CI] - k[3290]*y[IDX_CI] +
        k[3340]*y[IDX_CM];
    data[37] = 0.0 - k[999]*y[IDX_CI];
    data[38] = 0.0 - k[2056]*y[IDX_CI] - k[2058]*y[IDX_CI] +
        k[2315]*y[IDX_CII];
    data[39] = 0.0 - k[1017]*y[IDX_CI];
    data[40] = 0.0 + k[930]*y[IDX_HeII];
    data[41] = 0.0 - k[1553]*y[IDX_CI];
    data[42] = 0.0 - k[1571]*y[IDX_CI];
    data[43] = 0.0 + k[2918]*y[IDX_CII];
    data[44] = 0.0 + k[1064]*y[IDX_CHI] + k[1065]*y[IDX_CDI] -
        k[2653]*y[IDX_CI];
    data[45] = 0.0 + k[2135]*y[IDX_CM];
    data[46] = 0.0 + k[2133]*y[IDX_CII] - k[2687]*y[IDX_CI];
    data[47] = 0.0 - k[3283]*y[IDX_CI] - k[3284]*y[IDX_CI] -
        k[3285]*y[IDX_CI] + k[3336]*y[IDX_CM] + k[3337]*y[IDX_CM];
    data[48] = 0.0 + k[3013]*y[IDX_CHII] + k[3306]*y[IDX_CDII];
    data[49] = 0.0 - k[987]*y[IDX_CI];
    data[50] = 0.0 - k[3010]*y[IDX_CI] - k[3011]*y[IDX_CI] +
        k[3017]*y[IDX_CM];
    data[51] = 0.0 - k[998]*y[IDX_CI];
    data[52] = 0.0 - k[2055]*y[IDX_CI] - k[2057]*y[IDX_CI] +
        k[2314]*y[IDX_CII];
    data[53] = 0.0 - k[1016]*y[IDX_CI];
    data[54] = 0.0 - k[1134]*y[IDX_CI] - k[1135]*y[IDX_CI] -
        k[2661]*y[IDX_CI];
    data[55] = 0.0 - k[634]*y[IDX_CI] - k[635]*y[IDX_CI];
    data[56] = 0.0 - k[3286]*y[IDX_CI] - k[3287]*y[IDX_CI] -
        k[3288]*y[IDX_CI] + k[3338]*y[IDX_CM] + k[3339]*y[IDX_CM];
    data[57] = 0.0 + k[3307]*y[IDX_CHII] + k[3308]*y[IDX_CDII];
    data[58] = 0.0 - k[989]*y[IDX_CI] - k[990]*y[IDX_CI];
    data[59] = 0.0 + k[929]*y[IDX_HeII];
    data[60] = 0.0 - k[1552]*y[IDX_CI];
    data[61] = 0.0 - k[1570]*y[IDX_CI];
    data[62] = 0.0 + k[862]*y[IDX_C2I] + k[866]*y[IDX_CNI] +
        k[886]*y[IDX_C2HI] + k[887]*y[IDX_C2DI] + k[890]*y[IDX_C3I] +
        k[902]*y[IDX_CO2I] + k[929]*y[IDX_HNCI] + k[930]*y[IDX_DNCI] +
        k[2141]*y[IDX_CM];
    data[63] = 0.0 + k[609]*y[IDX_COII] + k[1196]*y[IDX_C2I] +
        k[1199]*y[IDX_CNI] - k[2655]*y[IDX_CI];
    data[64] = 0.0 + k[2064]*y[IDX_COI] + k[2144]*y[IDX_CM];
    data[65] = 0.0 - k[1054]*y[IDX_CI];
    data[66] = 0.0 - k[2128]*y[IDX_CI];
    data[67] = 0.0 - k[1617]*y[IDX_CI];
    data[68] = 0.0 - k[1616]*y[IDX_CI];
    data[69] = 0.0 - k[2049]*y[IDX_CI];
    data[70] = 0.0 - k[1819]*y[IDX_CI] + k[1829]*y[IDX_C2I];
    data[71] = 0.0 - k[1042]*y[IDX_CI] - k[1046]*y[IDX_CI] -
        k[1050]*y[IDX_CI];
    data[72] = 0.0 - k[2048]*y[IDX_CI];
    data[73] = 0.0 - k[1818]*y[IDX_CI] + k[1828]*y[IDX_C2I];
    data[74] = 0.0 - k[1041]*y[IDX_CI] - k[1045]*y[IDX_CI] -
        k[1049]*y[IDX_CI];
    data[75] = 0.0 - k[1043]*y[IDX_CI] - k[1044]*y[IDX_CI] -
        k[1047]*y[IDX_CI] - k[1048]*y[IDX_CI] - k[1051]*y[IDX_CI] -
        k[1052]*y[IDX_CI];
    data[76] = 0.0 - k[2050]*y[IDX_CI] - k[2051]*y[IDX_CI] +
        k[2065]*y[IDX_CII];
    data[77] = 0.0 + k[954]*y[IDX_C2HII] + k[955]*y[IDX_C2DII] +
        k[1226]*y[IDX_C2I] + k[1690]*y[IDX_C2II] - k[2656]*y[IDX_CI];
    data[78] = 0.0 + k[1652]*y[IDX_C2I] + k[1655]*y[IDX_CNI] +
        k[2147]*y[IDX_CM];
    data[79] = 0.0 - k[2715]*y[IDX_CI];
    data[80] = 0.0 - k[2052]*y[IDX_CI];
    data[81] = 0.0 - k[1910]*y[IDX_CI] - k[2557]*y[IDX_CI];
    data[82] = 0.0 - k[2003]*y[IDX_CI];
    data[83] = 0.0 - k[2002]*y[IDX_CI];
    data[84] = 0.0 - k[2063]*y[IDX_CI];
    data[85] = 0.0 - k[2054]*y[IDX_CI];
    data[86] = 0.0 - k[1922]*y[IDX_CI];
    data[87] = 0.0 - k[2731]*y[IDX_CI];
    data[88] = 0.0 - k[2053]*y[IDX_CI];
    data[89] = 0.0 - k[1921]*y[IDX_CI];
    data[90] = 0.0 - k[2730]*y[IDX_CI];
    data[91] = 0.0 - k[2736]*y[IDX_CI] + k[2740]*y[IDX_C2II] +
        k[2740]*y[IDX_C2II] + k[2741]*y[IDX_CHII] + k[2742]*y[IDX_CDII] +
        k[2743]*y[IDX_CNII] + k[2744]*y[IDX_COII] + k[2767]*y[IDX_C2HII] +
        k[2768]*y[IDX_C2DII] + k[2769]*y[IDX_C2HII] + k[2769]*y[IDX_C2HII] +
        k[2770]*y[IDX_C2DII] + k[2770]*y[IDX_C2DII] + k[2771]*y[IDX_C2NII] +
        k[2773]*y[IDX_C2OII] + k[2774]*y[IDX_C3II] + k[2775]*y[IDX_CH2II] +
        k[2776]*y[IDX_CD2II] + k[2777]*y[IDX_CHDII] + k[2782]*y[IDX_CH2II] +
        k[2783]*y[IDX_CD2II] + k[2784]*y[IDX_CHDII] + k[2785]*y[IDX_CNCII] +
        k[2845]*y[IDX_CII];
    data[92] = 0.0 - k[1260]*y[IDX_CI];
    data[93] = 0.0 - k[1133]*y[IDX_CI] - k[2660]*y[IDX_CI];
    data[94] = 0.0 - k[633]*y[IDX_CI];
    data[95] = 0.0 - k[1266]*y[IDX_CI] - k[1269]*y[IDX_CI];
    data[96] = 0.0 - k[1259]*y[IDX_CI];
    data[97] = 0.0 - k[1131]*y[IDX_CI] - k[2658]*y[IDX_CI];
    data[98] = 0.0 - k[631]*y[IDX_CI];
    data[99] = 0.0 - k[1261]*y[IDX_CI] - k[1264]*y[IDX_CI];
    data[100] = 0.0 - k[1256]*y[IDX_CI];
    data[101] = 0.0 - k[1132]*y[IDX_CI] - k[2659]*y[IDX_CI];
    data[102] = 0.0 - k[632]*y[IDX_CI];
    data[103] = 0.0 - k[1267]*y[IDX_CI] - k[1268]*y[IDX_CI] -
        k[1270]*y[IDX_CI];
    data[104] = 0.0 - k[2956]*y[IDX_CI] - k[2957]*y[IDX_CI];
    data[105] = 0.0 - k[1130]*y[IDX_CI] - k[2657]*y[IDX_CI];
    data[106] = 0.0 - k[630]*y[IDX_CI];
    data[107] = 0.0 - k[1262]*y[IDX_CI] - k[1263]*y[IDX_CI] -
        k[1265]*y[IDX_CI];
    data[108] = 0.0 - k[1257]*y[IDX_CI] - k[1258]*y[IDX_CI];
    data[109] = 0.0 + k[202] + k[351] + k[2081]*y[IDX_COII] +
        k[2128]*y[IDX_N2II] + k[2259]*y[IDX_C2II] + k[2263]*y[IDX_CNII] +
        k[2557]*y[IDX_O2II];
    data[110] = 0.0 - k[419]*y[IDX_CHI] - k[420]*y[IDX_CDI] -
        k[421]*y[IDX_NHI] - k[422]*y[IDX_NDI] - k[423]*y[IDX_O2I] -
        k[424]*y[IDX_O2I] - k[425]*y[IDX_OHI] - k[426]*y[IDX_ODI] -
        k[427]*y[IDX_C2HI] - k[428]*y[IDX_C2DI] - k[429]*y[IDX_CCOI] -
        k[430]*y[IDX_CH2I] - k[431]*y[IDX_CD2I] - k[432]*y[IDX_CHDI] -
        k[433]*y[IDX_CHDI] - k[434]*y[IDX_CO2I] - k[435]*y[IDX_H2OI] -
        k[436]*y[IDX_D2OI] - k[437]*y[IDX_HDOI] - k[438]*y[IDX_HDOI] -
        k[439]*y[IDX_H2OI] - k[440]*y[IDX_D2OI] - k[441]*y[IDX_HDOI] -
        k[442]*y[IDX_HDOI] - k[443]*y[IDX_HCNI] - k[444]*y[IDX_DCNI] -
        k[445]*y[IDX_HCNI] - k[446]*y[IDX_DCNI] - k[447]*y[IDX_HCOI] -
        k[448]*y[IDX_DCOI] - k[449]*y[IDX_HNCI] - k[450]*y[IDX_DNCI] -
        k[451]*y[IDX_NH2I] - k[452]*y[IDX_ND2I] - k[453]*y[IDX_NHDI] -
        k[454]*y[IDX_NHDI] - k[455]*y[IDX_OCNI] - k[2065]*y[IDX_NOI] -
        k[2132]*y[IDX_CM] - k[2133]*y[IDX_HM] - k[2134]*y[IDX_DM] -
        k[2309]*y[IDX_CHI] - k[2310]*y[IDX_CDI] - k[2311]*y[IDX_CH2I] -
        k[2312]*y[IDX_CD2I] - k[2313]*y[IDX_CHDI] - k[2314]*y[IDX_HCOI] -
        k[2315]*y[IDX_DCOI] - k[2638]*y[IDX_HI] - k[2639]*y[IDX_DI] -
        k[2640]*y[IDX_OI] - k[2641]*y[IDX_pH2I] - k[2642]*y[IDX_oH2I] -
        k[2643]*y[IDX_pD2I] - k[2644]*y[IDX_oD2I] - k[2645]*y[IDX_HDI] -
        k[2845]*y[IDX_eM] - k[2918]*y[IDX_GRAINM];
    data[111] = 0.0 - k[2132]*y[IDX_CII];
    data[112] = 0.0 + k[862]*y[IDX_HeII];
    data[113] = 0.0 + k[287] + k[1689]*y[IDX_NI] + k[2259]*y[IDX_CI];
    data[114] = 0.0 - k[428]*y[IDX_CII] + k[883]*y[IDX_HeII];
    data[115] = 0.0 - k[427]*y[IDX_CII] + k[882]*y[IDX_HeII];
    data[116] = 0.0 + k[888]*y[IDX_HeII];
    data[117] = 0.0 + k[889]*y[IDX_HeII];
    data[118] = 0.0 - k[429]*y[IDX_CII] + k[891]*y[IDX_HeII];
    data[119] = 0.0 - k[420]*y[IDX_CII] + k[864]*y[IDX_HeII] -
        k[2310]*y[IDX_CII];
    data[120] = 0.0 + k[233] + k[289] + k[1714]*y[IDX_HI] +
        k[1716]*y[IDX_DI];
    data[121] = 0.0 - k[431]*y[IDX_CII] + k[893]*y[IDX_HeII] -
        k[2312]*y[IDX_CII];
    data[122] = 0.0 - k[419]*y[IDX_CII] + k[863]*y[IDX_HeII] -
        k[2309]*y[IDX_CII];
    data[123] = 0.0 + k[232] + k[288] + k[1713]*y[IDX_HI] +
        k[1715]*y[IDX_DI];
    data[124] = 0.0 - k[430]*y[IDX_CII] + k[892]*y[IDX_HeII] -
        k[2311]*y[IDX_CII];
    data[125] = 0.0 - k[432]*y[IDX_CII] - k[433]*y[IDX_CII] +
        k[894]*y[IDX_HeII] - k[2313]*y[IDX_CII];
    data[126] = 0.0 + k[865]*y[IDX_HeII];
    data[127] = 0.0 + k[2263]*y[IDX_CI];
    data[128] = 0.0 + k[867]*y[IDX_HeII];
    data[129] = 0.0 + k[2081]*y[IDX_CI];
    data[130] = 0.0 - k[434]*y[IDX_CII] + k[899]*y[IDX_HeII];
    data[131] = 0.0 + k[1715]*y[IDX_CHII] + k[1716]*y[IDX_CDII] -
        k[2639]*y[IDX_CII];
    data[132] = 0.0 - k[2134]*y[IDX_CII];
    data[133] = 0.0 - k[436]*y[IDX_CII] - k[440]*y[IDX_CII];
    data[134] = 0.0 - k[444]*y[IDX_CII] - k[446]*y[IDX_CII] +
        k[912]*y[IDX_HeII];
    data[135] = 0.0 - k[448]*y[IDX_CII] - k[2315]*y[IDX_CII];
    data[136] = 0.0 - k[450]*y[IDX_CII] + k[926]*y[IDX_HeII];
    data[137] = 0.0 - k[2918]*y[IDX_CII];
    data[138] = 0.0 + k[1713]*y[IDX_CHII] + k[1714]*y[IDX_CDII] -
        k[2638]*y[IDX_CII];
    data[139] = 0.0 - k[2133]*y[IDX_CII];
    data[140] = 0.0 - k[435]*y[IDX_CII] - k[439]*y[IDX_CII];
    data[141] = 0.0 - k[443]*y[IDX_CII] - k[445]*y[IDX_CII] +
        k[911]*y[IDX_HeII];
    data[142] = 0.0 - k[447]*y[IDX_CII] - k[2314]*y[IDX_CII];
    data[143] = 0.0 - k[2645]*y[IDX_CII];
    data[144] = 0.0 - k[437]*y[IDX_CII] - k[438]*y[IDX_CII] -
        k[441]*y[IDX_CII] - k[442]*y[IDX_CII];
    data[145] = 0.0 - k[449]*y[IDX_CII] + k[925]*y[IDX_HeII];
    data[146] = 0.0 + k[862]*y[IDX_C2I] + k[863]*y[IDX_CHI] +
        k[864]*y[IDX_CDI] + k[865]*y[IDX_CNI] + k[867]*y[IDX_COI] +
        k[882]*y[IDX_C2HI] + k[883]*y[IDX_C2DI] + k[888]*y[IDX_C2NI] +
        k[889]*y[IDX_C3I] + k[891]*y[IDX_CCOI] + k[892]*y[IDX_CH2I] +
        k[893]*y[IDX_CD2I] + k[894]*y[IDX_CHDI] + k[899]*y[IDX_CO2I] +
        k[911]*y[IDX_HCNI] + k[912]*y[IDX_DCNI] + k[925]*y[IDX_HNCI] +
        k[926]*y[IDX_DNCI];
    data[147] = 0.0 + k[1689]*y[IDX_C2II];
    data[148] = 0.0 + k[2128]*y[IDX_CI];
    data[149] = 0.0 - k[422]*y[IDX_CII];
    data[150] = 0.0 - k[452]*y[IDX_CII];
    data[151] = 0.0 - k[421]*y[IDX_CII];
    data[152] = 0.0 - k[451]*y[IDX_CII];
    data[153] = 0.0 - k[453]*y[IDX_CII] - k[454]*y[IDX_CII];
    data[154] = 0.0 - k[2065]*y[IDX_CII];
    data[155] = 0.0 - k[2640]*y[IDX_CII];
    data[156] = 0.0 - k[423]*y[IDX_CII] - k[424]*y[IDX_CII];
    data[157] = 0.0 + k[2557]*y[IDX_CI];
    data[158] = 0.0 - k[455]*y[IDX_CII];
    data[159] = 0.0 - k[426]*y[IDX_CII];
    data[160] = 0.0 - k[425]*y[IDX_CII];
    data[161] = 0.0 - k[2845]*y[IDX_CII];
    data[162] = 0.0 - k[2644]*y[IDX_CII];
    data[163] = 0.0 - k[2642]*y[IDX_CII];
    data[164] = 0.0 - k[2643]*y[IDX_CII];
    data[165] = 0.0 - k[2641]*y[IDX_CII];
    data[166] = 0.0 - k[2670]*y[IDX_CM] + k[2736]*y[IDX_eM];
    data[167] = 0.0 - k[2132]*y[IDX_CM];
    data[168] = 0.0 - k[344] - k[480]*y[IDX_NOI] - k[481]*y[IDX_O2I] -
        k[482]*y[IDX_CO2I] - k[2132]*y[IDX_CII] - k[2135]*y[IDX_HII] -
        k[2136]*y[IDX_DII] - k[2141]*y[IDX_HeII] - k[2144]*y[IDX_NII] -
        k[2147]*y[IDX_OII] - k[2670]*y[IDX_CI] - k[2671]*y[IDX_HI] -
        k[2672]*y[IDX_DI] - k[2673]*y[IDX_NI] - k[2674]*y[IDX_OI] -
        k[2675]*y[IDX_CHI] - k[2676]*y[IDX_CDI] - k[2677]*y[IDX_pH2I] -
        k[2678]*y[IDX_oH2I] - k[2679]*y[IDX_pD2I] - k[2680]*y[IDX_oD2I] -
        k[2681]*y[IDX_HDI] - k[2682]*y[IDX_NHI] - k[2683]*y[IDX_NDI] -
        k[2684]*y[IDX_O2I] - k[2685]*y[IDX_OHI] - k[2686]*y[IDX_ODI] -
        k[3017]*y[IDX_H3OII] - k[3336]*y[IDX_H2DOII] - k[3337]*y[IDX_H2DOII] -
        k[3338]*y[IDX_HD2OII] - k[3339]*y[IDX_HD2OII] - k[3340]*y[IDX_D3OII];
    data[169] = 0.0 - k[2676]*y[IDX_CM];
    data[170] = 0.0 - k[2675]*y[IDX_CM];
    data[171] = 0.0 - k[482]*y[IDX_CM];
    data[172] = 0.0 - k[2672]*y[IDX_CM];
    data[173] = 0.0 - k[2136]*y[IDX_CM];
    data[174] = 0.0 - k[3340]*y[IDX_CM];
    data[175] = 0.0 - k[2671]*y[IDX_CM];
    data[176] = 0.0 - k[2135]*y[IDX_CM];
    data[177] = 0.0 - k[3336]*y[IDX_CM] - k[3337]*y[IDX_CM];
    data[178] = 0.0 - k[3017]*y[IDX_CM];
    data[179] = 0.0 - k[2681]*y[IDX_CM];
    data[180] = 0.0 - k[3338]*y[IDX_CM] - k[3339]*y[IDX_CM];
    data[181] = 0.0 - k[2141]*y[IDX_CM];
    data[182] = 0.0 - k[2673]*y[IDX_CM];
    data[183] = 0.0 - k[2144]*y[IDX_CM];
    data[184] = 0.0 - k[2683]*y[IDX_CM];
    data[185] = 0.0 - k[2682]*y[IDX_CM];
    data[186] = 0.0 - k[480]*y[IDX_CM];
    data[187] = 0.0 - k[2674]*y[IDX_CM];
    data[188] = 0.0 - k[2147]*y[IDX_CM];
    data[189] = 0.0 - k[481]*y[IDX_CM] - k[2684]*y[IDX_CM];
    data[190] = 0.0 - k[2686]*y[IDX_CM];
    data[191] = 0.0 - k[2685]*y[IDX_CM];
    data[192] = 0.0 + k[2736]*y[IDX_CI];
    data[193] = 0.0 - k[2680]*y[IDX_CM];
    data[194] = 0.0 - k[2678]*y[IDX_CM];
    data[195] = 0.0 - k[2679]*y[IDX_CM];
    data[196] = 0.0 - k[2677]*y[IDX_CM];
    data[197] = 0.0 + k[1053]*y[IDX_CNI] + k[1055]*y[IDX_COI] +
        k[2046]*y[IDX_CHI] + k[2047]*y[IDX_CDI] + k[2059]*y[IDX_C2NI] +
        k[2062]*y[IDX_CCOI] + k[2259]*y[IDX_C2II] + k[2652]*y[IDX_CI] +
        k[2652]*y[IDX_CI] - k[2665]*y[IDX_C2I] + k[2670]*y[IDX_CM];
    data[198] = 0.0 + k[2670]*y[IDX_CI];
    data[199] = 0.0 - k[281] - k[352] - k[353] - k[545]*y[IDX_CHI] -
        k[546]*y[IDX_CDI] - k[648]*y[IDX_pH2II] - k[649]*y[IDX_oH2II] -
        k[650]*y[IDX_pD2II] - k[651]*y[IDX_oD2II] - k[652]*y[IDX_HDII] -
        k[653]*y[IDX_HDII] - k[862]*y[IDX_HeII] - k[1000]*y[IDX_HCNII] -
        k[1001]*y[IDX_DCNII] - k[1018]*y[IDX_HCOII] - k[1019]*y[IDX_DCOII] -
        k[1196]*y[IDX_NI] - k[1226]*y[IDX_OI] - k[1238]*y[IDX_H2OII] -
        k[1239]*y[IDX_D2OII] - k[1240]*y[IDX_HDOII] - k[1241]*y[IDX_HDOII] -
        k[1310]*y[IDX_oH3II] - k[1311]*y[IDX_pH3II] - k[1312]*y[IDX_pH3II] -
        k[1313]*y[IDX_oD3II] - k[1314]*y[IDX_mD3II] - k[1315]*y[IDX_oH2DII] -
        k[1316]*y[IDX_pH2DII] - k[1317]*y[IDX_pH2DII] - k[1318]*y[IDX_oH2DII] -
        k[1319]*y[IDX_pH2DII] - k[1320]*y[IDX_oD2HII] - k[1321]*y[IDX_pD2HII] -
        k[1322]*y[IDX_pD2HII] - k[1323]*y[IDX_oD2HII] - k[1324]*y[IDX_pD2HII] -
        k[1554]*y[IDX_HNCII] - k[1555]*y[IDX_DNCII] - k[1574]*y[IDX_HNOII] -
        k[1575]*y[IDX_DNOII] - k[1618]*y[IDX_N2HII] - k[1619]*y[IDX_N2DII] -
        k[1652]*y[IDX_OII] - k[1691]*y[IDX_C2II] - k[1721]*y[IDX_CHII] -
        k[1722]*y[IDX_CDII] - k[1824]*y[IDX_NHII] - k[1825]*y[IDX_NDII] -
        k[1826]*y[IDX_NHII] - k[1827]*y[IDX_NDII] - k[1828]*y[IDX_NHII] -
        k[1829]*y[IDX_NDII] - k[1912]*y[IDX_O2II] - k[1927]*y[IDX_OHII] -
        k[1928]*y[IDX_ODII] - k[1977]*y[IDX_NH2II] - k[1978]*y[IDX_ND2II] -
        k[1979]*y[IDX_NHDII] - k[1980]*y[IDX_NHDII] - k[2008]*y[IDX_O2HII] -
        k[2009]*y[IDX_O2DII] - k[2085]*y[IDX_COII] - k[2187]*y[IDX_HII] -
        k[2188]*y[IDX_DII] - k[2223]*y[IDX_NII] - k[2267]*y[IDX_CNII] -
        k[2271]*y[IDX_N2II] - k[2340]*y[IDX_pH2II] - k[2341]*y[IDX_oH2II] -
        k[2342]*y[IDX_pD2II] - k[2343]*y[IDX_oD2II] - k[2344]*y[IDX_HDII] -
        k[2468]*y[IDX_HeII] - k[2479]*y[IDX_H2OII] - k[2480]*y[IDX_D2OII] -
        k[2481]*y[IDX_HDOII] - k[2522]*y[IDX_OII] - k[2576]*y[IDX_OHII] -
        k[2577]*y[IDX_ODII] - k[2665]*y[IDX_CI] - k[2697]*y[IDX_HM] -
        k[2698]*y[IDX_DM] - k[2961]*y[IDX_pD3II] - k[2962]*y[IDX_pD3II];
    data[200] = 0.0 - k[1691]*y[IDX_C2I] + k[2259]*y[IDX_CI] +
        k[2260]*y[IDX_NOI] + k[2543]*y[IDX_NH2I] + k[2544]*y[IDX_ND2I] +
        k[2545]*y[IDX_NHDI] + k[2614]*y[IDX_CHI] + k[2615]*y[IDX_CDI] +
        k[2616]*y[IDX_CH2I] + k[2617]*y[IDX_CD2I] + k[2618]*y[IDX_CHDI] +
        k[2619]*y[IDX_HCOI] + k[2620]*y[IDX_DCOI];
    data[201] = 0.0 + k[244] + k[379] + k[617]*y[IDX_COII];
    data[202] = 0.0 + k[957]*y[IDX_CHI] + k[959]*y[IDX_CDI] +
        k[2766]*y[IDX_eM];
    data[203] = 0.0 + k[243] + k[378] + k[616]*y[IDX_COII];
    data[204] = 0.0 + k[956]*y[IDX_CHI] + k[958]*y[IDX_CDI] +
        k[2765]*y[IDX_eM];
    data[205] = 0.0 + k[382] + k[2059]*y[IDX_CI];
    data[206] = 0.0 + k[2772]*y[IDX_eM];
    data[207] = 0.0 + k[246] + k[384] + k[536]*y[IDX_OI] +
        k[889]*y[IDX_HeII] + k[1225]*y[IDX_NI];
    data[208] = 0.0 + k[2774]*y[IDX_eM];
    data[209] = 0.0 + k[247] + k[2062]*y[IDX_CI];
    data[210] = 0.0 - k[546]*y[IDX_C2I] + k[958]*y[IDX_C2HII] +
        k[959]*y[IDX_C2DII] + k[2047]*y[IDX_CI] + k[2615]*y[IDX_C2II];
    data[211] = 0.0 - k[1722]*y[IDX_C2I];
    data[212] = 0.0 + k[2617]*y[IDX_C2II];
    data[213] = 0.0 - k[545]*y[IDX_C2I] + k[956]*y[IDX_C2HII] +
        k[957]*y[IDX_C2DII] + k[2046]*y[IDX_CI] + k[2614]*y[IDX_C2II];
    data[214] = 0.0 - k[1721]*y[IDX_C2I];
    data[215] = 0.0 + k[2616]*y[IDX_C2II];
    data[216] = 0.0 + k[2618]*y[IDX_C2II];
    data[217] = 0.0 + k[1053]*y[IDX_CI];
    data[218] = 0.0 - k[2267]*y[IDX_C2I];
    data[219] = 0.0 + k[2786]*y[IDX_eM];
    data[220] = 0.0 + k[1055]*y[IDX_CI];
    data[221] = 0.0 + k[616]*y[IDX_C2HI] + k[617]*y[IDX_C2DI] -
        k[2085]*y[IDX_C2I];
    data[222] = 0.0 - k[2188]*y[IDX_C2I];
    data[223] = 0.0 - k[2698]*y[IDX_C2I];
    data[224] = 0.0 - k[1239]*y[IDX_C2I] - k[2480]*y[IDX_C2I];
    data[225] = 0.0 - k[1001]*y[IDX_C2I];
    data[226] = 0.0 + k[2620]*y[IDX_C2II];
    data[227] = 0.0 - k[1019]*y[IDX_C2I];
    data[228] = 0.0 - k[1555]*y[IDX_C2I];
    data[229] = 0.0 - k[1575]*y[IDX_C2I];
    data[230] = 0.0 - k[2187]*y[IDX_C2I];
    data[231] = 0.0 - k[2697]*y[IDX_C2I];
    data[232] = 0.0 - k[1238]*y[IDX_C2I] - k[2479]*y[IDX_C2I];
    data[233] = 0.0 - k[1000]*y[IDX_C2I];
    data[234] = 0.0 + k[2619]*y[IDX_C2II];
    data[235] = 0.0 - k[1018]*y[IDX_C2I];
    data[236] = 0.0 - k[652]*y[IDX_C2I] - k[653]*y[IDX_C2I] -
        k[2344]*y[IDX_C2I];
    data[237] = 0.0 - k[1240]*y[IDX_C2I] - k[1241]*y[IDX_C2I] -
        k[2481]*y[IDX_C2I];
    data[238] = 0.0 - k[1554]*y[IDX_C2I];
    data[239] = 0.0 - k[1574]*y[IDX_C2I];
    data[240] = 0.0 - k[862]*y[IDX_C2I] + k[889]*y[IDX_C3I] -
        k[2468]*y[IDX_C2I];
    data[241] = 0.0 - k[1196]*y[IDX_C2I] + k[1225]*y[IDX_C3I];
    data[242] = 0.0 - k[2223]*y[IDX_C2I];
    data[243] = 0.0 - k[2271]*y[IDX_C2I];
    data[244] = 0.0 - k[1619]*y[IDX_C2I];
    data[245] = 0.0 - k[1618]*y[IDX_C2I];
    data[246] = 0.0 - k[1825]*y[IDX_C2I] - k[1827]*y[IDX_C2I] -
        k[1829]*y[IDX_C2I];
    data[247] = 0.0 + k[2544]*y[IDX_C2II];
    data[248] = 0.0 - k[1978]*y[IDX_C2I];
    data[249] = 0.0 - k[1824]*y[IDX_C2I] - k[1826]*y[IDX_C2I] -
        k[1828]*y[IDX_C2I];
    data[250] = 0.0 + k[2543]*y[IDX_C2II];
    data[251] = 0.0 - k[1977]*y[IDX_C2I];
    data[252] = 0.0 + k[2545]*y[IDX_C2II];
    data[253] = 0.0 - k[1979]*y[IDX_C2I] - k[1980]*y[IDX_C2I];
    data[254] = 0.0 + k[2260]*y[IDX_C2II];
    data[255] = 0.0 + k[536]*y[IDX_C3I] - k[1226]*y[IDX_C2I];
    data[256] = 0.0 - k[1652]*y[IDX_C2I] - k[2522]*y[IDX_C2I];
    data[257] = 0.0 - k[1912]*y[IDX_C2I];
    data[258] = 0.0 - k[2009]*y[IDX_C2I];
    data[259] = 0.0 - k[2008]*y[IDX_C2I];
    data[260] = 0.0 - k[1928]*y[IDX_C2I] - k[2577]*y[IDX_C2I];
    data[261] = 0.0 - k[1927]*y[IDX_C2I] - k[2576]*y[IDX_C2I];
    data[262] = 0.0 + k[2765]*y[IDX_C2HII] + k[2766]*y[IDX_C2DII] +
        k[2772]*y[IDX_C2NII] + k[2774]*y[IDX_C3II] + k[2786]*y[IDX_CNCII];
    data[263] = 0.0 - k[1314]*y[IDX_C2I];
    data[264] = 0.0 - k[651]*y[IDX_C2I] - k[2343]*y[IDX_C2I];
    data[265] = 0.0 - k[1320]*y[IDX_C2I] - k[1323]*y[IDX_C2I];
    data[266] = 0.0 - k[1313]*y[IDX_C2I];
    data[267] = 0.0 - k[649]*y[IDX_C2I] - k[2341]*y[IDX_C2I];
    data[268] = 0.0 - k[1315]*y[IDX_C2I] - k[1318]*y[IDX_C2I];
    data[269] = 0.0 - k[1310]*y[IDX_C2I];
    data[270] = 0.0 - k[650]*y[IDX_C2I] - k[2342]*y[IDX_C2I];
    data[271] = 0.0 - k[1321]*y[IDX_C2I] - k[1322]*y[IDX_C2I] -
        k[1324]*y[IDX_C2I];
    data[272] = 0.0 - k[2961]*y[IDX_C2I] - k[2962]*y[IDX_C2I];
    data[273] = 0.0 - k[648]*y[IDX_C2I] - k[2340]*y[IDX_C2I];
    data[274] = 0.0 - k[1316]*y[IDX_C2I] - k[1317]*y[IDX_C2I] -
        k[1319]*y[IDX_C2I];
    data[275] = 0.0 - k[1311]*y[IDX_C2I] - k[1312]*y[IDX_C2I];
    data[276] = 0.0 + k[1711]*y[IDX_CHII] + k[1712]*y[IDX_CDII] -
        k[2259]*y[IDX_C2II];
    data[277] = 0.0 + k[419]*y[IDX_CHI] + k[420]*y[IDX_CDI];
    data[278] = 0.0 + k[353] - k[1691]*y[IDX_C2II] + k[2085]*y[IDX_COII] +
        k[2187]*y[IDX_HII] + k[2188]*y[IDX_DII] + k[2223]*y[IDX_NII] +
        k[2267]*y[IDX_CNII] + k[2271]*y[IDX_N2II] + k[2340]*y[IDX_pH2II] +
        k[2341]*y[IDX_oH2II] + k[2342]*y[IDX_pD2II] + k[2343]*y[IDX_oD2II] +
        k[2344]*y[IDX_HDII] + k[2468]*y[IDX_HeII] + k[2479]*y[IDX_H2OII] +
        k[2480]*y[IDX_D2OII] + k[2481]*y[IDX_HDOII] + k[2522]*y[IDX_OII] +
        k[2576]*y[IDX_OHII] + k[2577]*y[IDX_ODII];
    data[279] = 0.0 - k[287] - k[1689]*y[IDX_NI] - k[1690]*y[IDX_OI] -
        k[1691]*y[IDX_C2I] - k[1692]*y[IDX_CHI] - k[1693]*y[IDX_CDI] -
        k[1694]*y[IDX_pH2I] - k[1695]*y[IDX_oH2I] - k[1696]*y[IDX_pD2I] -
        k[1697]*y[IDX_oD2I] - k[1698]*y[IDX_HDI] - k[1699]*y[IDX_HDI] -
        k[1700]*y[IDX_NHI] - k[1701]*y[IDX_NDI] - k[1702]*y[IDX_NHI] -
        k[1703]*y[IDX_NDI] - k[1704]*y[IDX_O2I] - k[1705]*y[IDX_H2OI] -
        k[1706]*y[IDX_D2OI] - k[1707]*y[IDX_HDOI] - k[1708]*y[IDX_HDOI] -
        k[1709]*y[IDX_HCOI] - k[1710]*y[IDX_DCOI] - k[2259]*y[IDX_CI] -
        k[2260]*y[IDX_NOI] - k[2543]*y[IDX_NH2I] - k[2544]*y[IDX_ND2I] -
        k[2545]*y[IDX_NHDI] - k[2614]*y[IDX_CHI] - k[2615]*y[IDX_CDI] -
        k[2616]*y[IDX_CH2I] - k[2617]*y[IDX_CD2I] - k[2618]*y[IDX_CHDI] -
        k[2619]*y[IDX_HCOI] - k[2620]*y[IDX_DCOI] - k[2740]*y[IDX_eM];
    data[280] = 0.0 + k[885]*y[IDX_HeII] + k[1528]*y[IDX_HII] +
        k[1529]*y[IDX_DII];
    data[281] = 0.0 + k[299];
    data[282] = 0.0 + k[884]*y[IDX_HeII] + k[1526]*y[IDX_HII] +
        k[1527]*y[IDX_DII];
    data[283] = 0.0 + k[298];
    data[284] = 0.0 + k[890]*y[IDX_HeII];
    data[285] = 0.0 + k[420]*y[IDX_CII] - k[1693]*y[IDX_C2II] +
        k[1725]*y[IDX_CHII] + k[1726]*y[IDX_CDII] - k[2615]*y[IDX_C2II];
    data[286] = 0.0 + k[1712]*y[IDX_CI] + k[1724]*y[IDX_CHI] +
        k[1726]*y[IDX_CDI];
    data[287] = 0.0 - k[2617]*y[IDX_C2II];
    data[288] = 0.0 + k[419]*y[IDX_CII] - k[1692]*y[IDX_C2II] +
        k[1723]*y[IDX_CHII] + k[1724]*y[IDX_CDII] - k[2614]*y[IDX_C2II];
    data[289] = 0.0 + k[1711]*y[IDX_CI] + k[1723]*y[IDX_CHI] +
        k[1725]*y[IDX_CDI];
    data[290] = 0.0 - k[2616]*y[IDX_C2II];
    data[291] = 0.0 - k[2618]*y[IDX_C2II];
    data[292] = 0.0 + k[2267]*y[IDX_C2I];
    data[293] = 0.0 + k[2085]*y[IDX_C2I];
    data[294] = 0.0 + k[1527]*y[IDX_C2HI] + k[1529]*y[IDX_C2DI] +
        k[2188]*y[IDX_C2I];
    data[295] = 0.0 - k[1706]*y[IDX_C2II];
    data[296] = 0.0 + k[2480]*y[IDX_C2I];
    data[297] = 0.0 - k[1710]*y[IDX_C2II] - k[2620]*y[IDX_C2II];
    data[298] = 0.0 + k[1526]*y[IDX_C2HI] + k[1528]*y[IDX_C2DI] +
        k[2187]*y[IDX_C2I];
    data[299] = 0.0 - k[1705]*y[IDX_C2II];
    data[300] = 0.0 + k[2479]*y[IDX_C2I];
    data[301] = 0.0 - k[1709]*y[IDX_C2II] - k[2619]*y[IDX_C2II];
    data[302] = 0.0 - k[1698]*y[IDX_C2II] - k[1699]*y[IDX_C2II];
    data[303] = 0.0 + k[2344]*y[IDX_C2I];
    data[304] = 0.0 - k[1707]*y[IDX_C2II] - k[1708]*y[IDX_C2II];
    data[305] = 0.0 + k[2481]*y[IDX_C2I];
    data[306] = 0.0 + k[884]*y[IDX_C2HI] + k[885]*y[IDX_C2DI] +
        k[890]*y[IDX_C3I] + k[2468]*y[IDX_C2I];
    data[307] = 0.0 - k[1689]*y[IDX_C2II];
    data[308] = 0.0 + k[2223]*y[IDX_C2I];
    data[309] = 0.0 + k[2271]*y[IDX_C2I];
    data[310] = 0.0 - k[1701]*y[IDX_C2II] - k[1703]*y[IDX_C2II];
    data[311] = 0.0 - k[2544]*y[IDX_C2II];
    data[312] = 0.0 - k[1700]*y[IDX_C2II] - k[1702]*y[IDX_C2II];
    data[313] = 0.0 - k[2543]*y[IDX_C2II];
    data[314] = 0.0 - k[2545]*y[IDX_C2II];
    data[315] = 0.0 - k[2260]*y[IDX_C2II];
    data[316] = 0.0 - k[1690]*y[IDX_C2II];
    data[317] = 0.0 + k[2522]*y[IDX_C2I];
    data[318] = 0.0 - k[1704]*y[IDX_C2II];
    data[319] = 0.0 + k[2577]*y[IDX_C2I];
    data[320] = 0.0 + k[2576]*y[IDX_C2I];
    data[321] = 0.0 - k[2740]*y[IDX_C2II];
    data[322] = 0.0 - k[1697]*y[IDX_C2II];
    data[323] = 0.0 + k[2343]*y[IDX_C2I];
    data[324] = 0.0 - k[1695]*y[IDX_C2II];
    data[325] = 0.0 + k[2341]*y[IDX_C2I];
    data[326] = 0.0 - k[1696]*y[IDX_C2II];
    data[327] = 0.0 + k[2342]*y[IDX_C2I];
    data[328] = 0.0 - k[1694]*y[IDX_C2II];
    data[329] = 0.0 + k[2340]*y[IDX_C2I];
    data[330] = 0.0 + k[1038]*y[IDX_CD2I] + k[1039]*y[IDX_CHDI] -
        k[2061]*y[IDX_C2DI];
    data[331] = 0.0 - k[428]*y[IDX_C2DI];
    data[332] = 0.0 + k[2676]*y[IDX_CDI];
    data[333] = 0.0 + k[2698]*y[IDX_DM];
    data[334] = 0.0 - k[244] - k[379] - k[381] - k[428]*y[IDX_CII] -
        k[617]*y[IDX_COII] - k[883]*y[IDX_HeII] - k[885]*y[IDX_HeII] -
        k[887]*y[IDX_HeII] - k[1033]*y[IDX_O2I] - k[1222]*y[IDX_NI] -
        k[1235]*y[IDX_OI] - k[1528]*y[IDX_HII] - k[1529]*y[IDX_DII] -
        k[1668]*y[IDX_OII] - k[1763]*y[IDX_CHII] - k[1764]*y[IDX_CDII] -
        k[2061]*y[IDX_CI] - k[2074]*y[IDX_CNII] - k[2227]*y[IDX_NII] -
        k[2283]*y[IDX_N2II] - k[2330]*y[IDX_COII] - k[2423]*y[IDX_pH2II] -
        k[2424]*y[IDX_oH2II] - k[2425]*y[IDX_pD2II] - k[2426]*y[IDX_oD2II] -
        k[2427]*y[IDX_HDII] - k[2491]*y[IDX_H2OII] - k[2492]*y[IDX_D2OII] -
        k[2493]*y[IDX_HDOII] - k[2525]*y[IDX_HII] - k[2526]*y[IDX_DII] -
        k[2569]*y[IDX_OII] - k[2584]*y[IDX_OHII] - k[2585]*y[IDX_ODII];
    data[335] = 0.0 + k[2163]*y[IDX_NOI];
    data[336] = 0.0 + k[2676]*y[IDX_CM];
    data[337] = 0.0 - k[1764]*y[IDX_C2DI];
    data[338] = 0.0 + k[1038]*y[IDX_CI];
    data[339] = 0.0 - k[1763]*y[IDX_C2DI];
    data[340] = 0.0 + k[1039]*y[IDX_CI];
    data[341] = 0.0 - k[2074]*y[IDX_C2DI];
    data[342] = 0.0 - k[617]*y[IDX_C2DI] - k[2330]*y[IDX_C2DI];
    data[343] = 0.0 - k[1529]*y[IDX_C2DI] - k[2526]*y[IDX_C2DI];
    data[344] = 0.0 + k[2698]*y[IDX_C2I];
    data[345] = 0.0 - k[2492]*y[IDX_C2DI];
    data[346] = 0.0 - k[1528]*y[IDX_C2DI] - k[2525]*y[IDX_C2DI];
    data[347] = 0.0 - k[2491]*y[IDX_C2DI];
    data[348] = 0.0 - k[2427]*y[IDX_C2DI];
    data[349] = 0.0 - k[2493]*y[IDX_C2DI];
    data[350] = 0.0 - k[883]*y[IDX_C2DI] - k[885]*y[IDX_C2DI] -
        k[887]*y[IDX_C2DI];
    data[351] = 0.0 - k[1222]*y[IDX_C2DI];
    data[352] = 0.0 - k[2227]*y[IDX_C2DI];
    data[353] = 0.0 - k[2283]*y[IDX_C2DI];
    data[354] = 0.0 + k[2163]*y[IDX_C2DII];
    data[355] = 0.0 - k[1235]*y[IDX_C2DI];
    data[356] = 0.0 - k[1668]*y[IDX_C2DI] - k[2569]*y[IDX_C2DI];
    data[357] = 0.0 - k[1033]*y[IDX_C2DI];
    data[358] = 0.0 - k[2585]*y[IDX_C2DI];
    data[359] = 0.0 - k[2584]*y[IDX_C2DI];
    data[360] = 0.0 - k[2426]*y[IDX_C2DI];
    data[361] = 0.0 - k[2424]*y[IDX_C2DI];
    data[362] = 0.0 - k[2425]*y[IDX_C2DI];
    data[363] = 0.0 - k[2423]*y[IDX_C2DI];
    data[364] = 0.0 - k[949]*y[IDX_C2DII] + k[965]*y[IDX_CD2II] +
        k[966]*y[IDX_CHDII];
    data[365] = 0.0 + k[431]*y[IDX_CD2I] + k[432]*y[IDX_CHDI];
    data[366] = 0.0 + k[650]*y[IDX_pD2II] + k[651]*y[IDX_oD2II] +
        k[652]*y[IDX_HDII] + k[1001]*y[IDX_DCNII] + k[1019]*y[IDX_DCOII] +
        k[1239]*y[IDX_D2OII] + k[1240]*y[IDX_HDOII] + k[1313]*y[IDX_oD3II] +
        k[1314]*y[IDX_mD3II] + k[1315]*y[IDX_oH2DII] + k[1316]*y[IDX_pH2DII] +
        k[1317]*y[IDX_pH2DII] + k[1323]*y[IDX_oD2HII] + k[1324]*y[IDX_pD2HII] +
        k[1555]*y[IDX_DNCII] + k[1575]*y[IDX_DNOII] + k[1619]*y[IDX_N2DII] +
        k[1825]*y[IDX_NDII] + k[1928]*y[IDX_ODII] + k[1978]*y[IDX_ND2II] +
        k[1979]*y[IDX_NHDII] + k[2009]*y[IDX_O2DII] + k[2961]*y[IDX_pD3II] +
        k[2962]*y[IDX_pD3II];
    data[367] = 0.0 + k[1696]*y[IDX_pD2I] + k[1697]*y[IDX_oD2I] +
        k[1698]*y[IDX_HDI] + k[1701]*y[IDX_NDI] + k[1706]*y[IDX_D2OI] +
        k[1707]*y[IDX_HDOI] + k[1710]*y[IDX_DCOI];
    data[368] = 0.0 + k[381] + k[2074]*y[IDX_CNII] + k[2227]*y[IDX_NII] +
        k[2283]*y[IDX_N2II] + k[2330]*y[IDX_COII] + k[2423]*y[IDX_pH2II] +
        k[2424]*y[IDX_oH2II] + k[2425]*y[IDX_pD2II] + k[2426]*y[IDX_oD2II] +
        k[2427]*y[IDX_HDII] + k[2491]*y[IDX_H2OII] + k[2492]*y[IDX_D2OII] +
        k[2493]*y[IDX_HDOII] + k[2525]*y[IDX_HII] + k[2526]*y[IDX_DII] +
        k[2569]*y[IDX_OII] + k[2584]*y[IDX_OHII] + k[2585]*y[IDX_ODII];
    data[369] = 0.0 - k[299] - k[949]*y[IDX_CI] - k[951]*y[IDX_NI] -
        k[953]*y[IDX_NI] - k[955]*y[IDX_OI] - k[957]*y[IDX_CHI] -
        k[959]*y[IDX_CDI] - k[2163]*y[IDX_NOI] - k[2766]*y[IDX_eM] -
        k[2768]*y[IDX_eM] - k[2770]*y[IDX_eM];
    data[370] = 0.0 - k[959]*y[IDX_C2DII];
    data[371] = 0.0 + k[1766]*y[IDX_CH2I] + k[1770]*y[IDX_CD2I] +
        k[1774]*y[IDX_CHDI];
    data[372] = 0.0 + k[431]*y[IDX_CII] + k[1769]*y[IDX_CHII] +
        k[1770]*y[IDX_CDII];
    data[373] = 0.0 + k[965]*y[IDX_CI];
    data[374] = 0.0 - k[957]*y[IDX_C2DII];
    data[375] = 0.0 + k[1769]*y[IDX_CD2I] + k[1771]*y[IDX_CHDI];
    data[376] = 0.0 + k[1766]*y[IDX_CDII];
    data[377] = 0.0 + k[432]*y[IDX_CII] + k[1771]*y[IDX_CHII] +
        k[1774]*y[IDX_CDII];
    data[378] = 0.0 + k[966]*y[IDX_CI];
    data[379] = 0.0 + k[2074]*y[IDX_C2DI];
    data[380] = 0.0 + k[2330]*y[IDX_C2DI];
    data[381] = 0.0 + k[2526]*y[IDX_C2DI];
    data[382] = 0.0 + k[1706]*y[IDX_C2II];
    data[383] = 0.0 + k[1239]*y[IDX_C2I] + k[2492]*y[IDX_C2DI];
    data[384] = 0.0 + k[1001]*y[IDX_C2I];
    data[385] = 0.0 + k[1710]*y[IDX_C2II];
    data[386] = 0.0 + k[1019]*y[IDX_C2I];
    data[387] = 0.0 + k[1555]*y[IDX_C2I];
    data[388] = 0.0 + k[1575]*y[IDX_C2I];
    data[389] = 0.0 + k[2525]*y[IDX_C2DI];
    data[390] = 0.0 + k[2491]*y[IDX_C2DI];
    data[391] = 0.0 + k[1698]*y[IDX_C2II];
    data[392] = 0.0 + k[652]*y[IDX_C2I] + k[2427]*y[IDX_C2DI];
    data[393] = 0.0 + k[1707]*y[IDX_C2II];
    data[394] = 0.0 + k[1240]*y[IDX_C2I] + k[2493]*y[IDX_C2DI];
    data[395] = 0.0 - k[951]*y[IDX_C2DII] - k[953]*y[IDX_C2DII];
    data[396] = 0.0 + k[2227]*y[IDX_C2DI];
    data[397] = 0.0 + k[2283]*y[IDX_C2DI];
    data[398] = 0.0 + k[1619]*y[IDX_C2I];
    data[399] = 0.0 + k[1701]*y[IDX_C2II];
    data[400] = 0.0 + k[1825]*y[IDX_C2I];
    data[401] = 0.0 + k[1978]*y[IDX_C2I];
    data[402] = 0.0 + k[1979]*y[IDX_C2I];
    data[403] = 0.0 - k[2163]*y[IDX_C2DII];
    data[404] = 0.0 - k[955]*y[IDX_C2DII];
    data[405] = 0.0 + k[2569]*y[IDX_C2DI];
    data[406] = 0.0 + k[2009]*y[IDX_C2I];
    data[407] = 0.0 + k[1928]*y[IDX_C2I] + k[2585]*y[IDX_C2DI];
    data[408] = 0.0 + k[2584]*y[IDX_C2DI];
    data[409] = 0.0 - k[2766]*y[IDX_C2DII] - k[2768]*y[IDX_C2DII] -
        k[2770]*y[IDX_C2DII];
    data[410] = 0.0 + k[1314]*y[IDX_C2I];
    data[411] = 0.0 + k[1697]*y[IDX_C2II];
    data[412] = 0.0 + k[651]*y[IDX_C2I] + k[2426]*y[IDX_C2DI];
    data[413] = 0.0 + k[1323]*y[IDX_C2I];
    data[414] = 0.0 + k[1313]*y[IDX_C2I];
    data[415] = 0.0 + k[2424]*y[IDX_C2DI];
    data[416] = 0.0 + k[1315]*y[IDX_C2I];
    data[417] = 0.0 + k[1696]*y[IDX_C2II];
    data[418] = 0.0 + k[650]*y[IDX_C2I] + k[2425]*y[IDX_C2DI];
    data[419] = 0.0 + k[1324]*y[IDX_C2I];
    data[420] = 0.0 + k[2961]*y[IDX_C2I] + k[2962]*y[IDX_C2I];
    data[421] = 0.0 + k[2423]*y[IDX_C2DI];
    data[422] = 0.0 + k[1316]*y[IDX_C2I] + k[1317]*y[IDX_C2I];
    data[423] = 0.0 + k[1037]*y[IDX_CH2I] + k[1040]*y[IDX_CHDI] -
        k[2060]*y[IDX_C2HI];
    data[424] = 0.0 - k[427]*y[IDX_C2HI];
    data[425] = 0.0 + k[2675]*y[IDX_CHI];
    data[426] = 0.0 + k[2697]*y[IDX_HM];
    data[427] = 0.0 - k[243] - k[378] - k[380] - k[427]*y[IDX_CII] -
        k[616]*y[IDX_COII] - k[882]*y[IDX_HeII] - k[884]*y[IDX_HeII] -
        k[886]*y[IDX_HeII] - k[1032]*y[IDX_O2I] - k[1221]*y[IDX_NI] -
        k[1234]*y[IDX_OI] - k[1526]*y[IDX_HII] - k[1527]*y[IDX_DII] -
        k[1667]*y[IDX_OII] - k[1761]*y[IDX_CHII] - k[1762]*y[IDX_CDII] -
        k[2060]*y[IDX_CI] - k[2073]*y[IDX_CNII] - k[2226]*y[IDX_NII] -
        k[2282]*y[IDX_N2II] - k[2329]*y[IDX_COII] - k[2418]*y[IDX_pH2II] -
        k[2419]*y[IDX_oH2II] - k[2420]*y[IDX_pD2II] - k[2421]*y[IDX_oD2II] -
        k[2422]*y[IDX_HDII] - k[2488]*y[IDX_H2OII] - k[2489]*y[IDX_D2OII] -
        k[2490]*y[IDX_HDOII] - k[2523]*y[IDX_HII] - k[2524]*y[IDX_DII] -
        k[2568]*y[IDX_OII] - k[2582]*y[IDX_OHII] - k[2583]*y[IDX_ODII];
    data[428] = 0.0 + k[2162]*y[IDX_NOI];
    data[429] = 0.0 - k[1762]*y[IDX_C2HI];
    data[430] = 0.0 + k[2675]*y[IDX_CM];
    data[431] = 0.0 - k[1761]*y[IDX_C2HI];
    data[432] = 0.0 + k[1037]*y[IDX_CI];
    data[433] = 0.0 + k[1040]*y[IDX_CI];
    data[434] = 0.0 - k[2073]*y[IDX_C2HI];
    data[435] = 0.0 - k[616]*y[IDX_C2HI] - k[2329]*y[IDX_C2HI];
    data[436] = 0.0 - k[1527]*y[IDX_C2HI] - k[2524]*y[IDX_C2HI];
    data[437] = 0.0 - k[2489]*y[IDX_C2HI];
    data[438] = 0.0 - k[1526]*y[IDX_C2HI] - k[2523]*y[IDX_C2HI];
    data[439] = 0.0 + k[2697]*y[IDX_C2I];
    data[440] = 0.0 - k[2488]*y[IDX_C2HI];
    data[441] = 0.0 - k[2422]*y[IDX_C2HI];
    data[442] = 0.0 - k[2490]*y[IDX_C2HI];
    data[443] = 0.0 - k[882]*y[IDX_C2HI] - k[884]*y[IDX_C2HI] -
        k[886]*y[IDX_C2HI];
    data[444] = 0.0 - k[1221]*y[IDX_C2HI];
    data[445] = 0.0 - k[2226]*y[IDX_C2HI];
    data[446] = 0.0 - k[2282]*y[IDX_C2HI];
    data[447] = 0.0 + k[2162]*y[IDX_C2HII];
    data[448] = 0.0 - k[1234]*y[IDX_C2HI];
    data[449] = 0.0 - k[1667]*y[IDX_C2HI] - k[2568]*y[IDX_C2HI];
    data[450] = 0.0 - k[1032]*y[IDX_C2HI];
    data[451] = 0.0 - k[2583]*y[IDX_C2HI];
    data[452] = 0.0 - k[2582]*y[IDX_C2HI];
    data[453] = 0.0 - k[2421]*y[IDX_C2HI];
    data[454] = 0.0 - k[2419]*y[IDX_C2HI];
    data[455] = 0.0 - k[2420]*y[IDX_C2HI];
    data[456] = 0.0 - k[2418]*y[IDX_C2HI];
    data[457] = 0.0 - k[948]*y[IDX_C2HII] + k[964]*y[IDX_CH2II] +
        k[967]*y[IDX_CHDII];
    data[458] = 0.0 + k[430]*y[IDX_CH2I] + k[433]*y[IDX_CHDI];
    data[459] = 0.0 + k[648]*y[IDX_pH2II] + k[649]*y[IDX_oH2II] +
        k[653]*y[IDX_HDII] + k[1000]*y[IDX_HCNII] + k[1018]*y[IDX_HCOII] +
        k[1238]*y[IDX_H2OII] + k[1241]*y[IDX_HDOII] + k[1310]*y[IDX_oH3II] +
        k[1311]*y[IDX_pH3II] + k[1312]*y[IDX_pH3II] + k[1318]*y[IDX_oH2DII] +
        k[1319]*y[IDX_pH2DII] + k[1320]*y[IDX_oD2HII] + k[1321]*y[IDX_pD2HII] +
        k[1322]*y[IDX_pD2HII] + k[1554]*y[IDX_HNCII] + k[1574]*y[IDX_HNOII] +
        k[1618]*y[IDX_N2HII] + k[1824]*y[IDX_NHII] + k[1927]*y[IDX_OHII] +
        k[1977]*y[IDX_NH2II] + k[1980]*y[IDX_NHDII] + k[2008]*y[IDX_O2HII];
    data[460] = 0.0 + k[1694]*y[IDX_pH2I] + k[1695]*y[IDX_oH2I] +
        k[1699]*y[IDX_HDI] + k[1700]*y[IDX_NHI] + k[1705]*y[IDX_H2OI] +
        k[1708]*y[IDX_HDOI] + k[1709]*y[IDX_HCOI];
    data[461] = 0.0 + k[380] + k[2073]*y[IDX_CNII] + k[2226]*y[IDX_NII] +
        k[2282]*y[IDX_N2II] + k[2329]*y[IDX_COII] + k[2418]*y[IDX_pH2II] +
        k[2419]*y[IDX_oH2II] + k[2420]*y[IDX_pD2II] + k[2421]*y[IDX_oD2II] +
        k[2422]*y[IDX_HDII] + k[2488]*y[IDX_H2OII] + k[2489]*y[IDX_D2OII] +
        k[2490]*y[IDX_HDOII] + k[2523]*y[IDX_HII] + k[2524]*y[IDX_DII] +
        k[2568]*y[IDX_OII] + k[2582]*y[IDX_OHII] + k[2583]*y[IDX_ODII];
    data[462] = 0.0 - k[298] - k[948]*y[IDX_CI] - k[950]*y[IDX_NI] -
        k[952]*y[IDX_NI] - k[954]*y[IDX_OI] - k[956]*y[IDX_CHI] -
        k[958]*y[IDX_CDI] - k[2162]*y[IDX_NOI] - k[2765]*y[IDX_eM] -
        k[2767]*y[IDX_eM] - k[2769]*y[IDX_eM];
    data[463] = 0.0 - k[958]*y[IDX_C2HII];
    data[464] = 0.0 + k[1767]*y[IDX_CH2I] + k[1773]*y[IDX_CHDI];
    data[465] = 0.0 + k[1768]*y[IDX_CHII];
    data[466] = 0.0 - k[956]*y[IDX_C2HII];
    data[467] = 0.0 + k[1765]*y[IDX_CH2I] + k[1768]*y[IDX_CD2I] +
        k[1772]*y[IDX_CHDI];
    data[468] = 0.0 + k[430]*y[IDX_CII] + k[1765]*y[IDX_CHII] +
        k[1767]*y[IDX_CDII];
    data[469] = 0.0 + k[964]*y[IDX_CI];
    data[470] = 0.0 + k[433]*y[IDX_CII] + k[1772]*y[IDX_CHII] +
        k[1773]*y[IDX_CDII];
    data[471] = 0.0 + k[967]*y[IDX_CI];
    data[472] = 0.0 + k[2073]*y[IDX_C2HI];
    data[473] = 0.0 + k[2329]*y[IDX_C2HI];
    data[474] = 0.0 + k[2524]*y[IDX_C2HI];
    data[475] = 0.0 + k[2489]*y[IDX_C2HI];
    data[476] = 0.0 + k[2523]*y[IDX_C2HI];
    data[477] = 0.0 + k[1705]*y[IDX_C2II];
    data[478] = 0.0 + k[1238]*y[IDX_C2I] + k[2488]*y[IDX_C2HI];
    data[479] = 0.0 + k[1000]*y[IDX_C2I];
    data[480] = 0.0 + k[1709]*y[IDX_C2II];
    data[481] = 0.0 + k[1018]*y[IDX_C2I];
    data[482] = 0.0 + k[1699]*y[IDX_C2II];
    data[483] = 0.0 + k[653]*y[IDX_C2I] + k[2422]*y[IDX_C2HI];
    data[484] = 0.0 + k[1708]*y[IDX_C2II];
    data[485] = 0.0 + k[1241]*y[IDX_C2I] + k[2490]*y[IDX_C2HI];
    data[486] = 0.0 + k[1554]*y[IDX_C2I];
    data[487] = 0.0 + k[1574]*y[IDX_C2I];
    data[488] = 0.0 - k[950]*y[IDX_C2HII] - k[952]*y[IDX_C2HII];
    data[489] = 0.0 + k[2226]*y[IDX_C2HI];
    data[490] = 0.0 + k[2282]*y[IDX_C2HI];
    data[491] = 0.0 + k[1618]*y[IDX_C2I];
    data[492] = 0.0 + k[1700]*y[IDX_C2II];
    data[493] = 0.0 + k[1824]*y[IDX_C2I];
    data[494] = 0.0 + k[1977]*y[IDX_C2I];
    data[495] = 0.0 + k[1980]*y[IDX_C2I];
    data[496] = 0.0 - k[2162]*y[IDX_C2HII];
    data[497] = 0.0 - k[954]*y[IDX_C2HII];
    data[498] = 0.0 + k[2568]*y[IDX_C2HI];
    data[499] = 0.0 + k[2008]*y[IDX_C2I];
    data[500] = 0.0 + k[2583]*y[IDX_C2HI];
    data[501] = 0.0 + k[1927]*y[IDX_C2I] + k[2582]*y[IDX_C2HI];
    data[502] = 0.0 - k[2765]*y[IDX_C2HII] - k[2767]*y[IDX_C2HII] -
        k[2769]*y[IDX_C2HII];
    data[503] = 0.0 + k[2421]*y[IDX_C2HI];
    data[504] = 0.0 + k[1320]*y[IDX_C2I];
    data[505] = 0.0 + k[1695]*y[IDX_C2II];
    data[506] = 0.0 + k[649]*y[IDX_C2I] + k[2419]*y[IDX_C2HI];
    data[507] = 0.0 + k[1318]*y[IDX_C2I];
    data[508] = 0.0 + k[1310]*y[IDX_C2I];
    data[509] = 0.0 + k[2420]*y[IDX_C2HI];
    data[510] = 0.0 + k[1321]*y[IDX_C2I] + k[1322]*y[IDX_C2I];
    data[511] = 0.0 + k[1694]*y[IDX_C2II];
    data[512] = 0.0 + k[648]*y[IDX_C2I] + k[2418]*y[IDX_C2HI];
    data[513] = 0.0 + k[1319]*y[IDX_C2I];
    data[514] = 0.0 + k[1311]*y[IDX_C2I] + k[1312]*y[IDX_C2I];
    data[515] = 0.0 - k[2059]*y[IDX_C2NI];
    data[516] = 0.0 + k[1222]*y[IDX_NI];
    data[517] = 0.0 + k[1221]*y[IDX_NI];
    data[518] = 0.0 - k[245] - k[382] - k[383] - k[888]*y[IDX_HeII] -
        k[1223]*y[IDX_NI] - k[1236]*y[IDX_OI] - k[2059]*y[IDX_CI] -
        k[2205]*y[IDX_HII] - k[2206]*y[IDX_DII];
    data[519] = 0.0 - k[2206]*y[IDX_C2NI];
    data[520] = 0.0 - k[2205]*y[IDX_C2NI];
    data[521] = 0.0 - k[888]*y[IDX_C2NI];
    data[522] = 0.0 + k[1221]*y[IDX_C2HI] + k[1222]*y[IDX_C2DI] -
        k[1223]*y[IDX_C2NI];
    data[523] = 0.0 - k[1236]*y[IDX_C2NI];
    data[524] = 0.0 + k[443]*y[IDX_HCNI] + k[444]*y[IDX_DCNI] +
        k[449]*y[IDX_HNCI] + k[450]*y[IDX_DNCI];
    data[525] = 0.0 + k[1826]*y[IDX_NHII] + k[1827]*y[IDX_NDII];
    data[526] = 0.0 + k[1702]*y[IDX_NHI] + k[1703]*y[IDX_NDI];
    data[527] = 0.0 + k[953]*y[IDX_NI];
    data[528] = 0.0 + k[952]*y[IDX_NI];
    data[529] = 0.0 + k[2205]*y[IDX_HII] + k[2206]*y[IDX_DII];
    data[530] = 0.0 - k[960]*y[IDX_H2OI] - k[961]*y[IDX_D2OI] -
        k[962]*y[IDX_HDOI] - k[963]*y[IDX_HDOI] - k[2771]*y[IDX_eM] -
        k[2772]*y[IDX_eM];
    data[531] = 0.0 + k[1728]*y[IDX_CNI] + k[1788]*y[IDX_HCNI] +
        k[1790]*y[IDX_DCNI];
    data[532] = 0.0 + k[1727]*y[IDX_CNI] + k[1787]*y[IDX_HCNI] +
        k[1789]*y[IDX_DCNI];
    data[533] = 0.0 + k[1727]*y[IDX_CHII] + k[1728]*y[IDX_CDII];
    data[534] = 0.0 + k[2206]*y[IDX_C2NI];
    data[535] = 0.0 - k[961]*y[IDX_C2NII];
    data[536] = 0.0 + k[444]*y[IDX_CII] + k[1789]*y[IDX_CHII] +
        k[1790]*y[IDX_CDII];
    data[537] = 0.0 + k[450]*y[IDX_CII];
    data[538] = 0.0 + k[2205]*y[IDX_C2NI];
    data[539] = 0.0 - k[960]*y[IDX_C2NII];
    data[540] = 0.0 + k[443]*y[IDX_CII] + k[1787]*y[IDX_CHII] +
        k[1788]*y[IDX_CDII];
    data[541] = 0.0 - k[962]*y[IDX_C2NII] - k[963]*y[IDX_C2NII];
    data[542] = 0.0 + k[449]*y[IDX_CII];
    data[543] = 0.0 + k[952]*y[IDX_C2HII] + k[953]*y[IDX_C2DII];
    data[544] = 0.0 + k[1703]*y[IDX_C2II];
    data[545] = 0.0 + k[1827]*y[IDX_C2I];
    data[546] = 0.0 + k[1702]*y[IDX_C2II];
    data[547] = 0.0 + k[1826]*y[IDX_C2I];
    data[548] = 0.0 - k[2771]*y[IDX_C2NII] - k[2772]*y[IDX_C2NII];
    data[549] = 0.0 + k[429]*y[IDX_CCOI];
    data[550] = 0.0 - k[2773]*y[IDX_eM];
    data[551] = 0.0 + k[429]*y[IDX_CII] + k[1530]*y[IDX_HII] +
        k[1531]*y[IDX_DII];
    data[552] = 0.0 + k[597]*y[IDX_CO2I];
    data[553] = 0.0 + k[597]*y[IDX_CNII];
    data[554] = 0.0 + k[1531]*y[IDX_CCOI];
    data[555] = 0.0 + k[1530]*y[IDX_CCOI];
    data[556] = 0.0 - k[2773]*y[IDX_C2OII];
    data[557] = 0.0 + k[2060]*y[IDX_C2HI] + k[2061]*y[IDX_C2DI] +
        k[2665]*y[IDX_C2I];
    data[558] = 0.0 + k[545]*y[IDX_CHI] + k[546]*y[IDX_CDI] +
        k[2665]*y[IDX_CI];
    data[559] = 0.0 + k[2061]*y[IDX_CI];
    data[560] = 0.0 + k[2060]*y[IDX_CI];
    data[561] = 0.0 - k[246] - k[384] - k[536]*y[IDX_OI] -
        k[889]*y[IDX_HeII] - k[890]*y[IDX_HeII] - k[1225]*y[IDX_NI] -
        k[2207]*y[IDX_HII] - k[2208]*y[IDX_DII];
    data[562] = 0.0 + k[546]*y[IDX_C2I];
    data[563] = 0.0 + k[545]*y[IDX_C2I];
    data[564] = 0.0 - k[2208]*y[IDX_C3I];
    data[565] = 0.0 - k[2207]*y[IDX_C3I];
    data[566] = 0.0 - k[889]*y[IDX_C3I] - k[890]*y[IDX_C3I];
    data[567] = 0.0 - k[1225]*y[IDX_C3I];
    data[568] = 0.0 - k[536]*y[IDX_C3I];
    data[569] = 0.0 + k[948]*y[IDX_C2HII] + k[949]*y[IDX_C2DII];
    data[570] = 0.0 + k[427]*y[IDX_C2HI] + k[428]*y[IDX_C2DI];
    data[571] = 0.0 + k[1691]*y[IDX_C2II] + k[1721]*y[IDX_CHII] +
        k[1722]*y[IDX_CDII];
    data[572] = 0.0 + k[1691]*y[IDX_C2I] + k[1692]*y[IDX_CHI] +
        k[1693]*y[IDX_CDI];
    data[573] = 0.0 + k[428]*y[IDX_CII] + k[1763]*y[IDX_CHII] +
        k[1764]*y[IDX_CDII];
    data[574] = 0.0 + k[949]*y[IDX_CI];
    data[575] = 0.0 + k[427]*y[IDX_CII] + k[1761]*y[IDX_CHII] +
        k[1762]*y[IDX_CDII];
    data[576] = 0.0 + k[948]*y[IDX_CI];
    data[577] = 0.0 + k[2207]*y[IDX_HII] + k[2208]*y[IDX_DII];
    data[578] = 0.0 - k[2774]*y[IDX_eM];
    data[579] = 0.0 + k[1693]*y[IDX_C2II];
    data[580] = 0.0 + k[1722]*y[IDX_C2I] + k[1762]*y[IDX_C2HI] +
        k[1764]*y[IDX_C2DI];
    data[581] = 0.0 + k[1692]*y[IDX_C2II];
    data[582] = 0.0 + k[1721]*y[IDX_C2I] + k[1761]*y[IDX_C2HI] +
        k[1763]*y[IDX_C2DI];
    data[583] = 0.0 + k[2208]*y[IDX_C3I];
    data[584] = 0.0 + k[2207]*y[IDX_C3I];
    data[585] = 0.0 - k[2774]*y[IDX_C3II];
    data[586] = 0.0 + k[2057]*y[IDX_HCOI] + k[2058]*y[IDX_DCOI] -
        k[2062]*y[IDX_CCOI];
    data[587] = 0.0 - k[429]*y[IDX_CCOI];
    data[588] = 0.0 - k[247] - k[248] - k[429]*y[IDX_CII] -
        k[891]*y[IDX_HeII] - k[1224]*y[IDX_NI] - k[1237]*y[IDX_OI] -
        k[1530]*y[IDX_HII] - k[1531]*y[IDX_DII] - k[2062]*y[IDX_CI];
    data[589] = 0.0 - k[1531]*y[IDX_CCOI];
    data[590] = 0.0 + k[2058]*y[IDX_CI];
    data[591] = 0.0 - k[1530]*y[IDX_CCOI];
    data[592] = 0.0 + k[2057]*y[IDX_CI];
    data[593] = 0.0 - k[891]*y[IDX_CCOI];
    data[594] = 0.0 - k[1224]*y[IDX_CCOI];
    data[595] = 0.0 - k[1237]*y[IDX_CCOI];
    data[596] = 0.0 + k[1035]*y[IDX_CD2I] + k[1035]*y[IDX_CD2I] +
        k[1036]*y[IDX_CHDI] + k[1042]*y[IDX_ND2I] + k[1044]*y[IDX_NHDI] +
        k[1132]*y[IDX_pD2I] + k[1133]*y[IDX_oD2I] + k[1134]*y[IDX_HDI] -
        k[2047]*y[IDX_CDI] + k[2056]*y[IDX_DCOI] + k[2654]*y[IDX_DI] +
        k[2688]*y[IDX_DM];
    data[597] = 0.0 - k[420]*y[IDX_CDI] - k[2310]*y[IDX_CDI];
    data[598] = 0.0 + k[2672]*y[IDX_DI] - k[2676]*y[IDX_CDI];
    data[599] = 0.0 - k[546]*y[IDX_CDI];
    data[600] = 0.0 - k[1693]*y[IDX_CDI] - k[2615]*y[IDX_CDI];
    data[601] = 0.0 + k[883]*y[IDX_HeII] + k[1235]*y[IDX_OI] +
        k[1668]*y[IDX_OII];
    data[602] = 0.0 - k[959]*y[IDX_CDI] + k[2768]*y[IDX_eM];
    data[603] = 0.0 - k[958]*y[IDX_CDI];
    data[604] = 0.0 - k[283] - k[355] - k[357] - k[420]*y[IDX_CII] -
        k[538]*y[IDX_NOI] - k[540]*y[IDX_O2I] - k[543]*y[IDX_HNOI] -
        k[544]*y[IDX_DNOI] - k[546]*y[IDX_C2I] - k[611]*y[IDX_COII] -
        k[662]*y[IDX_pH2II] - k[663]*y[IDX_oH2II] - k[664]*y[IDX_pH2II] -
        k[665]*y[IDX_oH2II] - k[666]*y[IDX_pD2II] - k[667]*y[IDX_oD2II] -
        k[668]*y[IDX_HDII] - k[669]*y[IDX_HDII] - k[864]*y[IDX_HeII] -
        k[958]*y[IDX_C2HII] - k[959]*y[IDX_C2DII] - k[1004]*y[IDX_HCNII] -
        k[1005]*y[IDX_DCNII] - k[1022]*y[IDX_HCOII] - k[1023]*y[IDX_DCOII] -
        k[1065]*y[IDX_HI] - k[1067]*y[IDX_DI] - k[1144]*y[IDX_pH2I] -
        k[1145]*y[IDX_oH2I] - k[1146]*y[IDX_pH2I] - k[1147]*y[IDX_oH2I] -
        k[1148]*y[IDX_pD2I] - k[1149]*y[IDX_oD2I] - k[1150]*y[IDX_HDI] -
        k[1151]*y[IDX_HDI] - k[1198]*y[IDX_NI] - k[1228]*y[IDX_OI] -
        k[1247]*y[IDX_H2OII] - k[1248]*y[IDX_H2OII] - k[1249]*y[IDX_D2OII] -
        k[1250]*y[IDX_HDOII] - k[1251]*y[IDX_HDOII] - k[1340]*y[IDX_oH3II] -
        k[1341]*y[IDX_pH3II] - k[1342]*y[IDX_pH3II] - k[1343]*y[IDX_oH3II] -
        k[1344]*y[IDX_pH3II] - k[1345]*y[IDX_oD3II] - k[1346]*y[IDX_mD3II] -
        k[1347]*y[IDX_oH2DII] - k[1348]*y[IDX_pH2DII] - k[1349]*y[IDX_pH2DII] -
        k[1350]*y[IDX_oH2DII] - k[1351]*y[IDX_pH2DII] - k[1352]*y[IDX_oD2HII] -
        k[1353]*y[IDX_pD2HII] - k[1354]*y[IDX_pD2HII] - k[1355]*y[IDX_oD2HII] -
        k[1356]*y[IDX_pD2HII] - k[1558]*y[IDX_HNCII] - k[1559]*y[IDX_DNCII] -
        k[1578]*y[IDX_HNOII] - k[1579]*y[IDX_DNOII] - k[1622]*y[IDX_N2HII] -
        k[1623]*y[IDX_N2DII] - k[1635]*y[IDX_NII] - k[1654]*y[IDX_OII] -
        k[1693]*y[IDX_C2II] - k[1725]*y[IDX_CHII] - k[1726]*y[IDX_CDII] -
        k[1832]*y[IDX_NHII] - k[1833]*y[IDX_NDII] - k[1914]*y[IDX_O2II] -
        k[1931]*y[IDX_OHII] - k[1932]*y[IDX_ODII] - k[1986]*y[IDX_NH2II] -
        k[1987]*y[IDX_NH2II] - k[1988]*y[IDX_ND2II] - k[1989]*y[IDX_NHDII] -
        k[1990]*y[IDX_NHDII] - k[2012]*y[IDX_O2HII] - k[2013]*y[IDX_O2DII] -
        k[2045]*y[IDX_OI] - k[2047]*y[IDX_CI] - k[2191]*y[IDX_HII] -
        k[2192]*y[IDX_DII] - k[2269]*y[IDX_CNII] - k[2273]*y[IDX_N2II] -
        k[2310]*y[IDX_CII] - k[2324]*y[IDX_COII] - k[2350]*y[IDX_pH2II] -
        k[2351]*y[IDX_oH2II] - k[2352]*y[IDX_pD2II] - k[2353]*y[IDX_oD2II] -
        k[2354]*y[IDX_HDII] - k[2366]*y[IDX_NII] - k[2470]*y[IDX_HeII] -
        k[2485]*y[IDX_H2OII] - k[2486]*y[IDX_D2OII] - k[2487]*y[IDX_HDOII] -
        k[2565]*y[IDX_OII] - k[2580]*y[IDX_OHII] - k[2581]*y[IDX_ODII] -
        k[2611]*y[IDX_NH2II] - k[2612]*y[IDX_ND2II] - k[2613]*y[IDX_NHDII] -
        k[2615]*y[IDX_C2II] - k[2632]*y[IDX_O2II] - k[2676]*y[IDX_CM] -
        k[2701]*y[IDX_HM] - k[2702]*y[IDX_DM] - k[2721]*y[IDX_OM] -
        k[2963]*y[IDX_pD3II] - k[2964]*y[IDX_pD3II] - k[3298]*y[IDX_H3OII] -
        k[3299]*y[IDX_H3OII] - k[3300]*y[IDX_H2DOII] - k[3301]*y[IDX_H2DOII] -
        k[3302]*y[IDX_H2DOII] - k[3303]*y[IDX_HD2OII] - k[3304]*y[IDX_HD2OII] -
        k[3305]*y[IDX_D3OII];
    data[605] = 0.0 - k[1726]*y[IDX_CDI] + k[2262]*y[IDX_NOI] +
        k[2571]*y[IDX_HCOI] + k[2573]*y[IDX_DCOI];
    data[606] = 0.0 + k[386] + k[619]*y[IDX_COII] + k[1035]*y[IDX_CI] +
        k[1035]*y[IDX_CI] + k[1070]*y[IDX_HI] + k[1075]*y[IDX_DI];
    data[607] = 0.0 + k[2779]*y[IDX_eM];
    data[608] = 0.0 - k[1725]*y[IDX_CDI];
    data[609] = 0.0 + k[1074]*y[IDX_DI];
    data[610] = 0.0 + k[387] + k[621]*y[IDX_COII] + k[1036]*y[IDX_CI] +
        k[1072]*y[IDX_HI] + k[1077]*y[IDX_DI];
    data[611] = 0.0 + k[2780]*y[IDX_eM];
    data[612] = 0.0 - k[2269]*y[IDX_CDI];
    data[613] = 0.0 - k[611]*y[IDX_CDI] + k[619]*y[IDX_CD2I] +
        k[621]*y[IDX_CHDI] - k[2324]*y[IDX_CDI];
    data[614] = 0.0 - k[1067]*y[IDX_CDI] + k[1074]*y[IDX_CH2I] +
        k[1075]*y[IDX_CD2I] + k[1077]*y[IDX_CHDI] + k[2654]*y[IDX_CI] +
        k[2672]*y[IDX_CM];
    data[615] = 0.0 - k[2192]*y[IDX_CDI];
    data[616] = 0.0 + k[2688]*y[IDX_CI] - k[2702]*y[IDX_CDI];
    data[617] = 0.0 - k[1249]*y[IDX_CDI] - k[2486]*y[IDX_CDI];
    data[618] = 0.0 - k[3305]*y[IDX_CDI];
    data[619] = 0.0 + k[914]*y[IDX_HeII] + k[1673]*y[IDX_OII];
    data[620] = 0.0 - k[1005]*y[IDX_CDI];
    data[621] = 0.0 + k[2056]*y[IDX_CI] + k[2573]*y[IDX_CDII];
    data[622] = 0.0 - k[1023]*y[IDX_CDI];
    data[623] = 0.0 - k[1559]*y[IDX_CDI];
    data[624] = 0.0 - k[544]*y[IDX_CDI];
    data[625] = 0.0 - k[1579]*y[IDX_CDI];
    data[626] = 0.0 - k[1065]*y[IDX_CDI] + k[1070]*y[IDX_CD2I] +
        k[1072]*y[IDX_CHDI];
    data[627] = 0.0 - k[2191]*y[IDX_CDI];
    data[628] = 0.0 - k[2701]*y[IDX_CDI];
    data[629] = 0.0 - k[3300]*y[IDX_CDI] - k[3301]*y[IDX_CDI] -
        k[3302]*y[IDX_CDI];
    data[630] = 0.0 - k[1247]*y[IDX_CDI] - k[1248]*y[IDX_CDI] -
        k[2485]*y[IDX_CDI];
    data[631] = 0.0 - k[3298]*y[IDX_CDI] - k[3299]*y[IDX_CDI];
    data[632] = 0.0 - k[1004]*y[IDX_CDI];
    data[633] = 0.0 + k[2571]*y[IDX_CDII];
    data[634] = 0.0 - k[1022]*y[IDX_CDI];
    data[635] = 0.0 + k[1134]*y[IDX_CI] - k[1150]*y[IDX_CDI] -
        k[1151]*y[IDX_CDI];
    data[636] = 0.0 - k[668]*y[IDX_CDI] - k[669]*y[IDX_CDI] -
        k[2354]*y[IDX_CDI];
    data[637] = 0.0 - k[3303]*y[IDX_CDI] - k[3304]*y[IDX_CDI];
    data[638] = 0.0 - k[1250]*y[IDX_CDI] - k[1251]*y[IDX_CDI] -
        k[2487]*y[IDX_CDI];
    data[639] = 0.0 - k[1558]*y[IDX_CDI];
    data[640] = 0.0 - k[543]*y[IDX_CDI];
    data[641] = 0.0 - k[1578]*y[IDX_CDI];
    data[642] = 0.0 - k[864]*y[IDX_CDI] + k[883]*y[IDX_C2DI] +
        k[914]*y[IDX_DCNI] - k[2470]*y[IDX_CDI];
    data[643] = 0.0 - k[1198]*y[IDX_CDI];
    data[644] = 0.0 - k[1635]*y[IDX_CDI] - k[2366]*y[IDX_CDI];
    data[645] = 0.0 - k[2273]*y[IDX_CDI];
    data[646] = 0.0 - k[1623]*y[IDX_CDI];
    data[647] = 0.0 - k[1622]*y[IDX_CDI];
    data[648] = 0.0 - k[1833]*y[IDX_CDI];
    data[649] = 0.0 + k[1042]*y[IDX_CI];
    data[650] = 0.0 - k[1988]*y[IDX_CDI] - k[2612]*y[IDX_CDI];
    data[651] = 0.0 - k[1832]*y[IDX_CDI];
    data[652] = 0.0 - k[1986]*y[IDX_CDI] - k[1987]*y[IDX_CDI] -
        k[2611]*y[IDX_CDI];
    data[653] = 0.0 + k[1044]*y[IDX_CI];
    data[654] = 0.0 - k[1989]*y[IDX_CDI] - k[1990]*y[IDX_CDI] -
        k[2613]*y[IDX_CDI];
    data[655] = 0.0 - k[538]*y[IDX_CDI] + k[2262]*y[IDX_CDII];
    data[656] = 0.0 - k[1228]*y[IDX_CDI] + k[1235]*y[IDX_C2DI] -
        k[2045]*y[IDX_CDI];
    data[657] = 0.0 - k[1654]*y[IDX_CDI] + k[1668]*y[IDX_C2DI] +
        k[1673]*y[IDX_DCNI] - k[2565]*y[IDX_CDI];
    data[658] = 0.0 - k[2721]*y[IDX_CDI];
    data[659] = 0.0 - k[540]*y[IDX_CDI];
    data[660] = 0.0 - k[1914]*y[IDX_CDI] - k[2632]*y[IDX_CDI];
    data[661] = 0.0 - k[2013]*y[IDX_CDI];
    data[662] = 0.0 - k[2012]*y[IDX_CDI];
    data[663] = 0.0 - k[1932]*y[IDX_CDI] - k[2581]*y[IDX_CDI];
    data[664] = 0.0 - k[1931]*y[IDX_CDI] - k[2580]*y[IDX_CDI];
    data[665] = 0.0 + k[2768]*y[IDX_C2DII] + k[2779]*y[IDX_CD2II] +
        k[2780]*y[IDX_CHDII];
    data[666] = 0.0 - k[1346]*y[IDX_CDI];
    data[667] = 0.0 + k[1133]*y[IDX_CI] - k[1149]*y[IDX_CDI];
    data[668] = 0.0 - k[667]*y[IDX_CDI] - k[2353]*y[IDX_CDI];
    data[669] = 0.0 - k[1352]*y[IDX_CDI] - k[1355]*y[IDX_CDI];
    data[670] = 0.0 - k[1345]*y[IDX_CDI];
    data[671] = 0.0 - k[1145]*y[IDX_CDI] - k[1147]*y[IDX_CDI];
    data[672] = 0.0 - k[663]*y[IDX_CDI] - k[665]*y[IDX_CDI] -
        k[2351]*y[IDX_CDI];
    data[673] = 0.0 - k[1347]*y[IDX_CDI] - k[1350]*y[IDX_CDI];
    data[674] = 0.0 - k[1340]*y[IDX_CDI] - k[1343]*y[IDX_CDI];
    data[675] = 0.0 + k[1132]*y[IDX_CI] - k[1148]*y[IDX_CDI];
    data[676] = 0.0 - k[666]*y[IDX_CDI] - k[2352]*y[IDX_CDI];
    data[677] = 0.0 - k[1353]*y[IDX_CDI] - k[1354]*y[IDX_CDI] -
        k[1356]*y[IDX_CDI];
    data[678] = 0.0 - k[2963]*y[IDX_CDI] - k[2964]*y[IDX_CDI];
    data[679] = 0.0 - k[1144]*y[IDX_CDI] - k[1146]*y[IDX_CDI];
    data[680] = 0.0 - k[662]*y[IDX_CDI] - k[664]*y[IDX_CDI] -
        k[2350]*y[IDX_CDI];
    data[681] = 0.0 - k[1348]*y[IDX_CDI] - k[1349]*y[IDX_CDI] -
        k[1351]*y[IDX_CDI];
    data[682] = 0.0 - k[1341]*y[IDX_CDI] - k[1342]*y[IDX_CDI] -
        k[1344]*y[IDX_CDI];
    data[683] = 0.0 + k[632]*y[IDX_pD2II] + k[633]*y[IDX_oD2II] +
        k[634]*y[IDX_HDII] + k[988]*y[IDX_D2OII] + k[989]*y[IDX_HDOII] +
        k[999]*y[IDX_DCNII] + k[1017]*y[IDX_DCOII] + k[1259]*y[IDX_oD3II] +
        k[1260]*y[IDX_mD3II] + k[1261]*y[IDX_oH2DII] + k[1262]*y[IDX_pH2DII] +
        k[1263]*y[IDX_pH2DII] + k[1269]*y[IDX_oD2HII] + k[1270]*y[IDX_pD2HII] +
        k[1553]*y[IDX_DNCII] + k[1571]*y[IDX_DNOII] + k[1617]*y[IDX_N2DII] -
        k[1712]*y[IDX_CDII] + k[1819]*y[IDX_NDII] + k[1922]*y[IDX_ODII] +
        k[2003]*y[IDX_O2DII] + k[2956]*y[IDX_pD3II] + k[2957]*y[IDX_pD3II];
    data[684] = 0.0 + k[448]*y[IDX_DCOI] + k[2310]*y[IDX_CDI] +
        k[2639]*y[IDX_DI];
    data[685] = 0.0 - k[1722]*y[IDX_CDII];
    data[686] = 0.0 + k[2615]*y[IDX_CDI];
    data[687] = 0.0 + k[887]*y[IDX_HeII] - k[1764]*y[IDX_CDII];
    data[688] = 0.0 + k[951]*y[IDX_NI];
    data[689] = 0.0 - k[1762]*y[IDX_CDII];
    data[690] = 0.0 + k[357] - k[1726]*y[IDX_CDII] + k[2191]*y[IDX_HII] +
        k[2192]*y[IDX_DII] + k[2269]*y[IDX_CNII] + k[2273]*y[IDX_N2II] +
        k[2310]*y[IDX_CII] + k[2324]*y[IDX_COII] + k[2350]*y[IDX_pH2II] +
        k[2351]*y[IDX_oH2II] + k[2352]*y[IDX_pD2II] + k[2353]*y[IDX_oD2II] +
        k[2354]*y[IDX_HDII] + k[2366]*y[IDX_NII] + k[2470]*y[IDX_HeII] +
        k[2485]*y[IDX_H2OII] + k[2486]*y[IDX_D2OII] + k[2487]*y[IDX_HDOII] +
        k[2565]*y[IDX_OII] + k[2580]*y[IDX_OHII] + k[2581]*y[IDX_ODII] +
        k[2611]*y[IDX_NH2II] + k[2612]*y[IDX_ND2II] + k[2613]*y[IDX_NHDII] +
        k[2615]*y[IDX_C2II] + k[2632]*y[IDX_O2II];
    data[691] = 0.0 - k[231] - k[233] - k[289] - k[1712]*y[IDX_CI] -
        k[1714]*y[IDX_HI] - k[1716]*y[IDX_DI] - k[1718]*y[IDX_NI] -
        k[1720]*y[IDX_OI] - k[1722]*y[IDX_C2I] - k[1724]*y[IDX_CHI] -
        k[1726]*y[IDX_CDI] - k[1728]*y[IDX_CNI] - k[1730]*y[IDX_CNI] -
        k[1733]*y[IDX_pH2I] - k[1734]*y[IDX_oH2I] - k[1735]*y[IDX_pH2I] -
        k[1736]*y[IDX_oH2I] - k[1741]*y[IDX_pD2I] - k[1742]*y[IDX_oD2I] -
        k[1745]*y[IDX_HDI] - k[1746]*y[IDX_HDI] - k[1748]*y[IDX_NHI] -
        k[1750]*y[IDX_NDI] - k[1752]*y[IDX_O2I] - k[1754]*y[IDX_O2I] -
        k[1756]*y[IDX_O2I] - k[1758]*y[IDX_OHI] - k[1760]*y[IDX_ODI] -
        k[1762]*y[IDX_C2HI] - k[1764]*y[IDX_C2DI] - k[1766]*y[IDX_CH2I] -
        k[1767]*y[IDX_CH2I] - k[1770]*y[IDX_CD2I] - k[1773]*y[IDX_CHDI] -
        k[1774]*y[IDX_CHDI] - k[1776]*y[IDX_CO2I] - k[1778]*y[IDX_H2OI] -
        k[1779]*y[IDX_H2OI] - k[1782]*y[IDX_D2OI] - k[1785]*y[IDX_HDOI] -
        k[1786]*y[IDX_HDOI] - k[1788]*y[IDX_HCNI] - k[1790]*y[IDX_DCNI] -
        k[1792]*y[IDX_HCOI] - k[1794]*y[IDX_DCOI] - k[1796]*y[IDX_NH2I] -
        k[1797]*y[IDX_NH2I] - k[1800]*y[IDX_ND2I] - k[1803]*y[IDX_NHDI] -
        k[1804]*y[IDX_NHDI] - k[2262]*y[IDX_NOI] - k[2571]*y[IDX_HCOI] -
        k[2573]*y[IDX_DCOI] - k[2742]*y[IDX_eM] - k[3306]*y[IDX_H2OI] -
        k[3308]*y[IDX_HDOI] - k[3310]*y[IDX_D2OI];
    data[692] = 0.0 + k[896]*y[IDX_HeII] + k[1536]*y[IDX_HII] +
        k[1537]*y[IDX_DII] - k[1770]*y[IDX_CDII];
    data[693] = 0.0 + k[301];
    data[694] = 0.0 - k[1724]*y[IDX_CDII];
    data[695] = 0.0 + k[1533]*y[IDX_DII] - k[1766]*y[IDX_CDII] -
        k[1767]*y[IDX_CDII];
    data[696] = 0.0 + k[897]*y[IDX_HeII] + k[1538]*y[IDX_HII] +
        k[1541]*y[IDX_DII] - k[1773]*y[IDX_CDII] - k[1774]*y[IDX_CDII];
    data[697] = 0.0 + k[302];
    data[698] = 0.0 - k[1728]*y[IDX_CDII] - k[1730]*y[IDX_CDII];
    data[699] = 0.0 + k[2269]*y[IDX_CDI];
    data[700] = 0.0 + k[2324]*y[IDX_CDI];
    data[701] = 0.0 - k[1776]*y[IDX_CDII];
    data[702] = 0.0 - k[1716]*y[IDX_CDII] + k[2639]*y[IDX_CII];
    data[703] = 0.0 + k[1533]*y[IDX_CH2I] + k[1537]*y[IDX_CD2I] +
        k[1541]*y[IDX_CHDI] + k[2192]*y[IDX_CDI];
    data[704] = 0.0 - k[1782]*y[IDX_CDII] - k[3310]*y[IDX_CDII];
    data[705] = 0.0 + k[988]*y[IDX_CI] + k[2486]*y[IDX_CDI];
    data[706] = 0.0 + k[916]*y[IDX_HeII] - k[1790]*y[IDX_CDII];
    data[707] = 0.0 + k[999]*y[IDX_CI];
    data[708] = 0.0 + k[448]*y[IDX_CII] + k[920]*y[IDX_HeII] -
        k[1794]*y[IDX_CDII] - k[2573]*y[IDX_CDII];
    data[709] = 0.0 + k[1017]*y[IDX_CI];
    data[710] = 0.0 + k[1553]*y[IDX_CI];
    data[711] = 0.0 + k[1571]*y[IDX_CI];
    data[712] = 0.0 - k[1714]*y[IDX_CDII];
    data[713] = 0.0 + k[1536]*y[IDX_CD2I] + k[1538]*y[IDX_CHDI] +
        k[2191]*y[IDX_CDI];
    data[714] = 0.0 - k[1778]*y[IDX_CDII] - k[1779]*y[IDX_CDII] -
        k[3306]*y[IDX_CDII];
    data[715] = 0.0 + k[2485]*y[IDX_CDI];
    data[716] = 0.0 - k[1788]*y[IDX_CDII];
    data[717] = 0.0 - k[1792]*y[IDX_CDII] - k[2571]*y[IDX_CDII];
    data[718] = 0.0 - k[1745]*y[IDX_CDII] - k[1746]*y[IDX_CDII];
    data[719] = 0.0 + k[634]*y[IDX_CI] + k[2354]*y[IDX_CDI];
    data[720] = 0.0 - k[1785]*y[IDX_CDII] - k[1786]*y[IDX_CDII] -
        k[3308]*y[IDX_CDII];
    data[721] = 0.0 + k[989]*y[IDX_CI] + k[2487]*y[IDX_CDI];
    data[722] = 0.0 + k[887]*y[IDX_C2DI] + k[896]*y[IDX_CD2I] +
        k[897]*y[IDX_CHDI] + k[916]*y[IDX_DCNI] + k[920]*y[IDX_DCOI] +
        k[2470]*y[IDX_CDI];
    data[723] = 0.0 + k[951]*y[IDX_C2DII] - k[1718]*y[IDX_CDII];
    data[724] = 0.0 + k[2366]*y[IDX_CDI];
    data[725] = 0.0 + k[2273]*y[IDX_CDI];
    data[726] = 0.0 + k[1617]*y[IDX_CI];
    data[727] = 0.0 - k[1750]*y[IDX_CDII];
    data[728] = 0.0 + k[1819]*y[IDX_CI];
    data[729] = 0.0 - k[1800]*y[IDX_CDII];
    data[730] = 0.0 + k[2612]*y[IDX_CDI];
    data[731] = 0.0 - k[1748]*y[IDX_CDII];
    data[732] = 0.0 - k[1796]*y[IDX_CDII] - k[1797]*y[IDX_CDII];
    data[733] = 0.0 + k[2611]*y[IDX_CDI];
    data[734] = 0.0 - k[1803]*y[IDX_CDII] - k[1804]*y[IDX_CDII];
    data[735] = 0.0 + k[2613]*y[IDX_CDI];
    data[736] = 0.0 - k[2262]*y[IDX_CDII];
    data[737] = 0.0 - k[1720]*y[IDX_CDII];
    data[738] = 0.0 + k[2565]*y[IDX_CDI];
    data[739] = 0.0 - k[1752]*y[IDX_CDII] - k[1754]*y[IDX_CDII] -
        k[1756]*y[IDX_CDII];
    data[740] = 0.0 + k[2632]*y[IDX_CDI];
    data[741] = 0.0 + k[2003]*y[IDX_CI];
    data[742] = 0.0 - k[1760]*y[IDX_CDII];
    data[743] = 0.0 + k[1922]*y[IDX_CI] + k[2581]*y[IDX_CDI];
    data[744] = 0.0 - k[1758]*y[IDX_CDII];
    data[745] = 0.0 + k[2580]*y[IDX_CDI];
    data[746] = 0.0 - k[2742]*y[IDX_CDII];
    data[747] = 0.0 + k[1260]*y[IDX_CI];
    data[748] = 0.0 - k[1742]*y[IDX_CDII];
    data[749] = 0.0 + k[633]*y[IDX_CI] + k[2353]*y[IDX_CDI];
    data[750] = 0.0 + k[1269]*y[IDX_CI];
    data[751] = 0.0 + k[1259]*y[IDX_CI];
    data[752] = 0.0 - k[1734]*y[IDX_CDII] - k[1736]*y[IDX_CDII];
    data[753] = 0.0 + k[2351]*y[IDX_CDI];
    data[754] = 0.0 + k[1261]*y[IDX_CI];
    data[755] = 0.0 - k[1741]*y[IDX_CDII];
    data[756] = 0.0 + k[632]*y[IDX_CI] + k[2352]*y[IDX_CDI];
    data[757] = 0.0 + k[1270]*y[IDX_CI];
    data[758] = 0.0 + k[2956]*y[IDX_CI] + k[2957]*y[IDX_CI];
    data[759] = 0.0 - k[1733]*y[IDX_CDII] - k[1735]*y[IDX_CDII];
    data[760] = 0.0 + k[2350]*y[IDX_CDI];
    data[761] = 0.0 + k[1262]*y[IDX_CI] + k[1263]*y[IDX_CI];
    data[762] = 0.0 - k[1035]*y[IDX_CD2I] - k[1038]*y[IDX_CD2I] +
        k[2659]*y[IDX_pD2I] + k[2660]*y[IDX_oD2I];
    data[763] = 0.0 - k[431]*y[IDX_CD2I] - k[2312]*y[IDX_CD2I];
    data[764] = 0.0 + k[2679]*y[IDX_pD2I] + k[2680]*y[IDX_oD2I];
    data[765] = 0.0 - k[2617]*y[IDX_CD2I];
    data[766] = 0.0 + k[544]*y[IDX_DNOI] + k[1148]*y[IDX_pD2I] +
        k[1149]*y[IDX_oD2I] + k[1150]*y[IDX_HDI] + k[2702]*y[IDX_DM];
    data[767] = 0.0 - k[1770]*y[IDX_CD2I];
    data[768] = 0.0 - k[250] - k[386] - k[390] - k[431]*y[IDX_CII] -
        k[510]*y[IDX_OI] - k[513]*y[IDX_OI] - k[619]*y[IDX_COII] -
        k[893]*y[IDX_HeII] - k[896]*y[IDX_HeII] - k[1035]*y[IDX_CI] -
        k[1038]*y[IDX_CI] - k[1069]*y[IDX_HI] - k[1070]*y[IDX_HI] -
        k[1075]*y[IDX_DI] - k[1207]*y[IDX_NI] - k[1211]*y[IDX_NI] -
        k[1535]*y[IDX_HII] - k[1536]*y[IDX_HII] - k[1537]*y[IDX_DII] -
        k[1768]*y[IDX_CHII] - k[1769]*y[IDX_CHII] - k[1770]*y[IDX_CDII] -
        k[2076]*y[IDX_CNII] - k[2229]*y[IDX_NII] - k[2244]*y[IDX_OII] -
        k[2285]*y[IDX_N2II] - k[2312]*y[IDX_CII] - k[2332]*y[IDX_COII] -
        k[2433]*y[IDX_pH2II] - k[2434]*y[IDX_oH2II] - k[2435]*y[IDX_pD2II] -
        k[2436]*y[IDX_oD2II] - k[2437]*y[IDX_HDII] - k[2497]*y[IDX_H2OII] -
        k[2498]*y[IDX_D2OII] - k[2499]*y[IDX_HDOII] - k[2529]*y[IDX_HII] -
        k[2530]*y[IDX_DII] - k[2549]*y[IDX_NH2II] - k[2550]*y[IDX_ND2II] -
        k[2551]*y[IDX_NHDII] - k[2588]*y[IDX_OHII] - k[2589]*y[IDX_ODII] -
        k[2617]*y[IDX_C2II] - k[2634]*y[IDX_O2II];
    data[769] = 0.0 + k[2165]*y[IDX_NOI];
    data[770] = 0.0 + k[1138]*y[IDX_pD2I] + k[1139]*y[IDX_oD2I];
    data[771] = 0.0 - k[1768]*y[IDX_CD2I] - k[1769]*y[IDX_CD2I];
    data[772] = 0.0 - k[2076]*y[IDX_CD2I];
    data[773] = 0.0 - k[619]*y[IDX_CD2I] - k[2332]*y[IDX_CD2I];
    data[774] = 0.0 + k[1063]*y[IDX_DCOI] - k[1075]*y[IDX_CD2I];
    data[775] = 0.0 - k[1537]*y[IDX_CD2I] - k[2530]*y[IDX_CD2I];
    data[776] = 0.0 + k[2702]*y[IDX_CDI];
    data[777] = 0.0 - k[2498]*y[IDX_CD2I];
    data[778] = 0.0 + k[1063]*y[IDX_DI];
    data[779] = 0.0 + k[544]*y[IDX_CDI];
    data[780] = 0.0 - k[1069]*y[IDX_CD2I] - k[1070]*y[IDX_CD2I];
    data[781] = 0.0 - k[1535]*y[IDX_CD2I] - k[1536]*y[IDX_CD2I] -
        k[2529]*y[IDX_CD2I];
    data[782] = 0.0 - k[2497]*y[IDX_CD2I];
    data[783] = 0.0 + k[1150]*y[IDX_CDI];
    data[784] = 0.0 - k[2437]*y[IDX_CD2I];
    data[785] = 0.0 - k[2499]*y[IDX_CD2I];
    data[786] = 0.0 - k[893]*y[IDX_CD2I] - k[896]*y[IDX_CD2I];
    data[787] = 0.0 - k[1207]*y[IDX_CD2I] - k[1211]*y[IDX_CD2I];
    data[788] = 0.0 - k[2229]*y[IDX_CD2I];
    data[789] = 0.0 - k[2285]*y[IDX_CD2I];
    data[790] = 0.0 - k[2550]*y[IDX_CD2I];
    data[791] = 0.0 - k[2549]*y[IDX_CD2I];
    data[792] = 0.0 - k[2551]*y[IDX_CD2I];
    data[793] = 0.0 + k[2165]*y[IDX_CD2II];
    data[794] = 0.0 - k[510]*y[IDX_CD2I] - k[513]*y[IDX_CD2I];
    data[795] = 0.0 - k[2244]*y[IDX_CD2I];
    data[796] = 0.0 - k[2634]*y[IDX_CD2I];
    data[797] = 0.0 - k[2589]*y[IDX_CD2I];
    data[798] = 0.0 - k[2588]*y[IDX_CD2I];
    data[799] = 0.0 + k[1139]*y[IDX_CHI] + k[1149]*y[IDX_CDI] +
        k[2660]*y[IDX_CI] + k[2680]*y[IDX_CM];
    data[800] = 0.0 - k[2436]*y[IDX_CD2I];
    data[801] = 0.0 - k[2434]*y[IDX_CD2I];
    data[802] = 0.0 + k[1138]*y[IDX_CHI] + k[1148]*y[IDX_CDI] +
        k[2659]*y[IDX_CI] + k[2679]*y[IDX_CM];
    data[803] = 0.0 - k[2435]*y[IDX_CD2I];
    data[804] = 0.0 - k[2433]*y[IDX_CD2I];
    data[805] = 0.0 - k[965]*y[IDX_CD2II];
    data[806] = 0.0 + k[2312]*y[IDX_CD2I] + k[2643]*y[IDX_pD2I] +
        k[2644]*y[IDX_oD2I];
    data[807] = 0.0 + k[2617]*y[IDX_CD2I];
    data[808] = 0.0 + k[959]*y[IDX_CDI];
    data[809] = 0.0 + k[666]*y[IDX_pD2II] + k[667]*y[IDX_oD2II] +
        k[668]*y[IDX_HDII] + k[959]*y[IDX_C2DII] + k[1005]*y[IDX_DCNII] +
        k[1023]*y[IDX_DCOII] + k[1249]*y[IDX_D2OII] + k[1250]*y[IDX_HDOII] +
        k[1345]*y[IDX_oD3II] + k[1346]*y[IDX_mD3II] + k[1347]*y[IDX_oH2DII] +
        k[1348]*y[IDX_pH2DII] + k[1349]*y[IDX_pH2DII] + k[1355]*y[IDX_oD2HII] +
        k[1356]*y[IDX_pD2HII] + k[1559]*y[IDX_DNCII] + k[1579]*y[IDX_DNOII] +
        k[1623]*y[IDX_N2DII] + k[1833]*y[IDX_NDII] + k[1932]*y[IDX_ODII] +
        k[1988]*y[IDX_ND2II] + k[1989]*y[IDX_NHDII] + k[2013]*y[IDX_O2DII] +
        k[2963]*y[IDX_pD3II] + k[2964]*y[IDX_pD3II] + k[3300]*y[IDX_H2DOII] +
        k[3303]*y[IDX_HD2OII] + k[3305]*y[IDX_D3OII];
    data[810] = 0.0 + k[1741]*y[IDX_pD2I] + k[1742]*y[IDX_oD2I] +
        k[1745]*y[IDX_HDI] + k[1794]*y[IDX_DCOI];
    data[811] = 0.0 + k[250] + k[390] + k[2076]*y[IDX_CNII] +
        k[2229]*y[IDX_NII] + k[2244]*y[IDX_OII] + k[2285]*y[IDX_N2II] +
        k[2312]*y[IDX_CII] + k[2332]*y[IDX_COII] + k[2433]*y[IDX_pH2II] +
        k[2434]*y[IDX_oH2II] + k[2435]*y[IDX_pD2II] + k[2436]*y[IDX_oD2II] +
        k[2437]*y[IDX_HDII] + k[2497]*y[IDX_H2OII] + k[2498]*y[IDX_D2OII] +
        k[2499]*y[IDX_HDOII] + k[2529]*y[IDX_HII] + k[2530]*y[IDX_DII] +
        k[2549]*y[IDX_NH2II] + k[2550]*y[IDX_ND2II] + k[2551]*y[IDX_NHDII] +
        k[2588]*y[IDX_OHII] + k[2589]*y[IDX_ODII] + k[2617]*y[IDX_C2II] +
        k[2634]*y[IDX_O2II];
    data[812] = 0.0 - k[301] - k[965]*y[IDX_CI] - k[969]*y[IDX_NI] -
        k[973]*y[IDX_OI] - k[977]*y[IDX_O2I] - k[2165]*y[IDX_NOI] -
        k[2776]*y[IDX_eM] - k[2779]*y[IDX_eM] - k[2783]*y[IDX_eM];
    data[813] = 0.0 + k[656]*y[IDX_pD2II] + k[657]*y[IDX_oD2II] +
        k[1243]*y[IDX_D2OII] + k[1328]*y[IDX_oD3II] + k[1329]*y[IDX_mD3II] +
        k[1982]*y[IDX_ND2II] + k[3293]*y[IDX_HD2OII] + k[3296]*y[IDX_D3OII];
    data[814] = 0.0 + k[1737]*y[IDX_pD2I] + k[1738]*y[IDX_oD2I];
    data[815] = 0.0 + k[2076]*y[IDX_CD2I];
    data[816] = 0.0 + k[2332]*y[IDX_CD2I];
    data[817] = 0.0 + k[2530]*y[IDX_CD2I];
    data[818] = 0.0 + k[1243]*y[IDX_CHI] + k[1249]*y[IDX_CDI] +
        k[2498]*y[IDX_CD2I];
    data[819] = 0.0 + k[3296]*y[IDX_CHI] + k[3305]*y[IDX_CDI];
    data[820] = 0.0 + k[1005]*y[IDX_CDI];
    data[821] = 0.0 + k[1794]*y[IDX_CDII];
    data[822] = 0.0 + k[1023]*y[IDX_CDI];
    data[823] = 0.0 + k[1559]*y[IDX_CDI];
    data[824] = 0.0 + k[1579]*y[IDX_CDI];
    data[825] = 0.0 + k[2529]*y[IDX_CD2I];
    data[826] = 0.0 + k[3300]*y[IDX_CDI];
    data[827] = 0.0 + k[2497]*y[IDX_CD2I];
    data[828] = 0.0 + k[1745]*y[IDX_CDII];
    data[829] = 0.0 + k[668]*y[IDX_CDI] + k[2437]*y[IDX_CD2I];
    data[830] = 0.0 + k[3293]*y[IDX_CHI] + k[3303]*y[IDX_CDI];
    data[831] = 0.0 + k[1250]*y[IDX_CDI] + k[2499]*y[IDX_CD2I];
    data[832] = 0.0 - k[969]*y[IDX_CD2II];
    data[833] = 0.0 + k[2229]*y[IDX_CD2I];
    data[834] = 0.0 + k[2285]*y[IDX_CD2I];
    data[835] = 0.0 + k[1623]*y[IDX_CDI];
    data[836] = 0.0 + k[1833]*y[IDX_CDI];
    data[837] = 0.0 + k[1982]*y[IDX_CHI] + k[1988]*y[IDX_CDI] +
        k[2550]*y[IDX_CD2I];
    data[838] = 0.0 + k[2549]*y[IDX_CD2I];
    data[839] = 0.0 + k[1989]*y[IDX_CDI] + k[2551]*y[IDX_CD2I];
    data[840] = 0.0 - k[2165]*y[IDX_CD2II];
    data[841] = 0.0 - k[973]*y[IDX_CD2II];
    data[842] = 0.0 + k[2244]*y[IDX_CD2I];
    data[843] = 0.0 - k[977]*y[IDX_CD2II];
    data[844] = 0.0 + k[2634]*y[IDX_CD2I];
    data[845] = 0.0 + k[2013]*y[IDX_CDI];
    data[846] = 0.0 + k[1932]*y[IDX_CDI] + k[2589]*y[IDX_CD2I];
    data[847] = 0.0 + k[2588]*y[IDX_CD2I];
    data[848] = 0.0 - k[2776]*y[IDX_CD2II] - k[2779]*y[IDX_CD2II] -
        k[2783]*y[IDX_CD2II];
    data[849] = 0.0 + k[1329]*y[IDX_CHI] + k[1346]*y[IDX_CDI];
    data[850] = 0.0 + k[1738]*y[IDX_CHII] + k[1742]*y[IDX_CDII] +
        k[2644]*y[IDX_CII];
    data[851] = 0.0 + k[657]*y[IDX_CHI] + k[667]*y[IDX_CDI] +
        k[2436]*y[IDX_CD2I];
    data[852] = 0.0 + k[1355]*y[IDX_CDI];
    data[853] = 0.0 + k[1328]*y[IDX_CHI] + k[1345]*y[IDX_CDI];
    data[854] = 0.0 + k[2434]*y[IDX_CD2I];
    data[855] = 0.0 + k[1347]*y[IDX_CDI];
    data[856] = 0.0 + k[1737]*y[IDX_CHII] + k[1741]*y[IDX_CDII] +
        k[2643]*y[IDX_CII];
    data[857] = 0.0 + k[656]*y[IDX_CHI] + k[666]*y[IDX_CDI] +
        k[2435]*y[IDX_CD2I];
    data[858] = 0.0 + k[1356]*y[IDX_CDI];
    data[859] = 0.0 + k[2963]*y[IDX_CDI] + k[2964]*y[IDX_CDI];
    data[860] = 0.0 + k[2433]*y[IDX_CD2I];
    data[861] = 0.0 + k[1348]*y[IDX_CDI] + k[1349]*y[IDX_CDI];
    data[862] = 0.0 + k[1034]*y[IDX_CH2I] + k[1034]*y[IDX_CH2I] +
        k[1036]*y[IDX_CHDI] + k[1041]*y[IDX_NH2I] + k[1043]*y[IDX_NHDI] +
        k[1130]*y[IDX_pH2I] + k[1131]*y[IDX_oH2I] + k[1135]*y[IDX_HDI] -
        k[2046]*y[IDX_CHI] + k[2055]*y[IDX_HCOI] + k[2653]*y[IDX_HI] +
        k[2687]*y[IDX_HM];
    data[863] = 0.0 - k[419]*y[IDX_CHI] - k[2309]*y[IDX_CHI];
    data[864] = 0.0 + k[2671]*y[IDX_HI] - k[2675]*y[IDX_CHI];
    data[865] = 0.0 - k[545]*y[IDX_CHI];
    data[866] = 0.0 - k[1692]*y[IDX_CHI] - k[2614]*y[IDX_CHI];
    data[867] = 0.0 - k[957]*y[IDX_CHI];
    data[868] = 0.0 + k[882]*y[IDX_HeII] + k[1234]*y[IDX_OI] +
        k[1667]*y[IDX_OII];
    data[869] = 0.0 - k[956]*y[IDX_CHI] + k[2767]*y[IDX_eM];
    data[870] = 0.0 - k[1724]*y[IDX_CHI];
    data[871] = 0.0 + k[1069]*y[IDX_HI];
    data[872] = 0.0 - k[282] - k[354] - k[356] - k[419]*y[IDX_CII] -
        k[537]*y[IDX_NOI] - k[539]*y[IDX_O2I] - k[541]*y[IDX_HNOI] -
        k[542]*y[IDX_DNOI] - k[545]*y[IDX_C2I] - k[610]*y[IDX_COII] -
        k[654]*y[IDX_pH2II] - k[655]*y[IDX_oH2II] - k[656]*y[IDX_pD2II] -
        k[657]*y[IDX_oD2II] - k[658]*y[IDX_pD2II] - k[659]*y[IDX_oD2II] -
        k[660]*y[IDX_HDII] - k[661]*y[IDX_HDII] - k[863]*y[IDX_HeII] -
        k[956]*y[IDX_C2HII] - k[957]*y[IDX_C2DII] - k[1002]*y[IDX_HCNII] -
        k[1003]*y[IDX_DCNII] - k[1020]*y[IDX_HCOII] - k[1021]*y[IDX_DCOII] -
        k[1064]*y[IDX_HI] - k[1066]*y[IDX_DI] - k[1136]*y[IDX_pH2I] -
        k[1137]*y[IDX_oH2I] - k[1138]*y[IDX_pD2I] - k[1139]*y[IDX_oD2I] -
        k[1140]*y[IDX_pD2I] - k[1141]*y[IDX_oD2I] - k[1142]*y[IDX_HDI] -
        k[1143]*y[IDX_HDI] - k[1197]*y[IDX_NI] - k[1227]*y[IDX_OI] -
        k[1242]*y[IDX_H2OII] - k[1243]*y[IDX_D2OII] - k[1244]*y[IDX_D2OII] -
        k[1245]*y[IDX_HDOII] - k[1246]*y[IDX_HDOII] - k[1325]*y[IDX_oH3II] -
        k[1326]*y[IDX_pH3II] - k[1327]*y[IDX_pH3II] - k[1328]*y[IDX_oD3II] -
        k[1329]*y[IDX_mD3II] - k[1330]*y[IDX_oH2DII] - k[1331]*y[IDX_pH2DII] -
        k[1332]*y[IDX_pH2DII] - k[1333]*y[IDX_oH2DII] - k[1334]*y[IDX_pH2DII] -
        k[1335]*y[IDX_oD2HII] - k[1336]*y[IDX_pD2HII] - k[1337]*y[IDX_pD2HII] -
        k[1338]*y[IDX_oD2HII] - k[1339]*y[IDX_pD2HII] - k[1556]*y[IDX_HNCII] -
        k[1557]*y[IDX_DNCII] - k[1576]*y[IDX_HNOII] - k[1577]*y[IDX_DNOII] -
        k[1620]*y[IDX_N2HII] - k[1621]*y[IDX_N2DII] - k[1634]*y[IDX_NII] -
        k[1653]*y[IDX_OII] - k[1692]*y[IDX_C2II] - k[1723]*y[IDX_CHII] -
        k[1724]*y[IDX_CDII] - k[1830]*y[IDX_NHII] - k[1831]*y[IDX_NDII] -
        k[1913]*y[IDX_O2II] - k[1929]*y[IDX_OHII] - k[1930]*y[IDX_ODII] -
        k[1981]*y[IDX_NH2II] - k[1982]*y[IDX_ND2II] - k[1983]*y[IDX_ND2II] -
        k[1984]*y[IDX_NHDII] - k[1985]*y[IDX_NHDII] - k[2010]*y[IDX_O2HII] -
        k[2011]*y[IDX_O2DII] - k[2044]*y[IDX_OI] - k[2046]*y[IDX_CI] -
        k[2189]*y[IDX_HII] - k[2190]*y[IDX_DII] - k[2268]*y[IDX_CNII] -
        k[2272]*y[IDX_N2II] - k[2309]*y[IDX_CII] - k[2323]*y[IDX_COII] -
        k[2345]*y[IDX_pH2II] - k[2346]*y[IDX_oH2II] - k[2347]*y[IDX_pD2II] -
        k[2348]*y[IDX_oD2II] - k[2349]*y[IDX_HDII] - k[2365]*y[IDX_NII] -
        k[2469]*y[IDX_HeII] - k[2482]*y[IDX_H2OII] - k[2483]*y[IDX_D2OII] -
        k[2484]*y[IDX_HDOII] - k[2564]*y[IDX_OII] - k[2578]*y[IDX_OHII] -
        k[2579]*y[IDX_ODII] - k[2608]*y[IDX_NH2II] - k[2609]*y[IDX_ND2II] -
        k[2610]*y[IDX_NHDII] - k[2614]*y[IDX_C2II] - k[2631]*y[IDX_O2II] -
        k[2675]*y[IDX_CM] - k[2699]*y[IDX_HM] - k[2700]*y[IDX_DM] -
        k[2720]*y[IDX_OM] - k[3012]*y[IDX_H3OII] - k[3291]*y[IDX_H2DOII] -
        k[3292]*y[IDX_H2DOII] - k[3293]*y[IDX_HD2OII] - k[3294]*y[IDX_HD2OII] -
        k[3295]*y[IDX_HD2OII] - k[3296]*y[IDX_D3OII] - k[3297]*y[IDX_D3OII];
    data[873] = 0.0 - k[1723]*y[IDX_CHI] + k[2261]*y[IDX_NOI] +
        k[2570]*y[IDX_HCOI] + k[2572]*y[IDX_DCOI];
    data[874] = 0.0 + k[385] + k[618]*y[IDX_COII] + k[1034]*y[IDX_CI] +
        k[1034]*y[IDX_CI] + k[1068]*y[IDX_HI] + k[1073]*y[IDX_DI];
    data[875] = 0.0 + k[2778]*y[IDX_eM];
    data[876] = 0.0 + k[388] + k[620]*y[IDX_COII] + k[1036]*y[IDX_CI] +
        k[1071]*y[IDX_HI] + k[1076]*y[IDX_DI];
    data[877] = 0.0 + k[2781]*y[IDX_eM];
    data[878] = 0.0 - k[2268]*y[IDX_CHI];
    data[879] = 0.0 - k[610]*y[IDX_CHI] + k[618]*y[IDX_CH2I] +
        k[620]*y[IDX_CHDI] - k[2323]*y[IDX_CHI];
    data[880] = 0.0 - k[1066]*y[IDX_CHI] + k[1073]*y[IDX_CH2I] +
        k[1076]*y[IDX_CHDI];
    data[881] = 0.0 - k[2190]*y[IDX_CHI];
    data[882] = 0.0 - k[2700]*y[IDX_CHI];
    data[883] = 0.0 - k[1243]*y[IDX_CHI] - k[1244]*y[IDX_CHI] -
        k[2483]*y[IDX_CHI];
    data[884] = 0.0 - k[3296]*y[IDX_CHI] - k[3297]*y[IDX_CHI];
    data[885] = 0.0 - k[1003]*y[IDX_CHI];
    data[886] = 0.0 + k[2572]*y[IDX_CHII];
    data[887] = 0.0 - k[1021]*y[IDX_CHI];
    data[888] = 0.0 - k[1557]*y[IDX_CHI];
    data[889] = 0.0 - k[542]*y[IDX_CHI];
    data[890] = 0.0 - k[1577]*y[IDX_CHI];
    data[891] = 0.0 - k[1064]*y[IDX_CHI] + k[1068]*y[IDX_CH2I] +
        k[1069]*y[IDX_CD2I] + k[1071]*y[IDX_CHDI] + k[2653]*y[IDX_CI] +
        k[2671]*y[IDX_CM];
    data[892] = 0.0 - k[2189]*y[IDX_CHI];
    data[893] = 0.0 + k[2687]*y[IDX_CI] - k[2699]*y[IDX_CHI];
    data[894] = 0.0 - k[3291]*y[IDX_CHI] - k[3292]*y[IDX_CHI];
    data[895] = 0.0 - k[1242]*y[IDX_CHI] - k[2482]*y[IDX_CHI];
    data[896] = 0.0 - k[3012]*y[IDX_CHI];
    data[897] = 0.0 + k[913]*y[IDX_HeII] + k[1672]*y[IDX_OII];
    data[898] = 0.0 - k[1002]*y[IDX_CHI];
    data[899] = 0.0 + k[2055]*y[IDX_CI] + k[2570]*y[IDX_CHII];
    data[900] = 0.0 - k[1020]*y[IDX_CHI];
    data[901] = 0.0 + k[1135]*y[IDX_CI] - k[1142]*y[IDX_CHI] -
        k[1143]*y[IDX_CHI];
    data[902] = 0.0 - k[660]*y[IDX_CHI] - k[661]*y[IDX_CHI] -
        k[2349]*y[IDX_CHI];
    data[903] = 0.0 - k[3293]*y[IDX_CHI] - k[3294]*y[IDX_CHI] -
        k[3295]*y[IDX_CHI];
    data[904] = 0.0 - k[1245]*y[IDX_CHI] - k[1246]*y[IDX_CHI] -
        k[2484]*y[IDX_CHI];
    data[905] = 0.0 - k[1556]*y[IDX_CHI];
    data[906] = 0.0 - k[541]*y[IDX_CHI];
    data[907] = 0.0 - k[1576]*y[IDX_CHI];
    data[908] = 0.0 - k[863]*y[IDX_CHI] + k[882]*y[IDX_C2HI] +
        k[913]*y[IDX_HCNI] - k[2469]*y[IDX_CHI];
    data[909] = 0.0 - k[1197]*y[IDX_CHI];
    data[910] = 0.0 - k[1634]*y[IDX_CHI] - k[2365]*y[IDX_CHI];
    data[911] = 0.0 - k[2272]*y[IDX_CHI];
    data[912] = 0.0 - k[1621]*y[IDX_CHI];
    data[913] = 0.0 - k[1620]*y[IDX_CHI];
    data[914] = 0.0 - k[1831]*y[IDX_CHI];
    data[915] = 0.0 - k[1982]*y[IDX_CHI] - k[1983]*y[IDX_CHI] -
        k[2609]*y[IDX_CHI];
    data[916] = 0.0 - k[1830]*y[IDX_CHI];
    data[917] = 0.0 + k[1041]*y[IDX_CI];
    data[918] = 0.0 - k[1981]*y[IDX_CHI] - k[2608]*y[IDX_CHI];
    data[919] = 0.0 + k[1043]*y[IDX_CI];
    data[920] = 0.0 - k[1984]*y[IDX_CHI] - k[1985]*y[IDX_CHI] -
        k[2610]*y[IDX_CHI];
    data[921] = 0.0 - k[537]*y[IDX_CHI] + k[2261]*y[IDX_CHII];
    data[922] = 0.0 - k[1227]*y[IDX_CHI] + k[1234]*y[IDX_C2HI] -
        k[2044]*y[IDX_CHI];
    data[923] = 0.0 - k[1653]*y[IDX_CHI] + k[1667]*y[IDX_C2HI] +
        k[1672]*y[IDX_HCNI] - k[2564]*y[IDX_CHI];
    data[924] = 0.0 - k[2720]*y[IDX_CHI];
    data[925] = 0.0 - k[539]*y[IDX_CHI];
    data[926] = 0.0 - k[1913]*y[IDX_CHI] - k[2631]*y[IDX_CHI];
    data[927] = 0.0 - k[2011]*y[IDX_CHI];
    data[928] = 0.0 - k[2010]*y[IDX_CHI];
    data[929] = 0.0 - k[1930]*y[IDX_CHI] - k[2579]*y[IDX_CHI];
    data[930] = 0.0 - k[1929]*y[IDX_CHI] - k[2578]*y[IDX_CHI];
    data[931] = 0.0 + k[2767]*y[IDX_C2HII] + k[2778]*y[IDX_CH2II] +
        k[2781]*y[IDX_CHDII];
    data[932] = 0.0 - k[1329]*y[IDX_CHI];
    data[933] = 0.0 - k[1139]*y[IDX_CHI] - k[1141]*y[IDX_CHI];
    data[934] = 0.0 - k[657]*y[IDX_CHI] - k[659]*y[IDX_CHI] -
        k[2348]*y[IDX_CHI];
    data[935] = 0.0 - k[1335]*y[IDX_CHI] - k[1338]*y[IDX_CHI];
    data[936] = 0.0 - k[1328]*y[IDX_CHI];
    data[937] = 0.0 + k[1131]*y[IDX_CI] - k[1137]*y[IDX_CHI];
    data[938] = 0.0 - k[655]*y[IDX_CHI] - k[2346]*y[IDX_CHI];
    data[939] = 0.0 - k[1330]*y[IDX_CHI] - k[1333]*y[IDX_CHI];
    data[940] = 0.0 - k[1325]*y[IDX_CHI];
    data[941] = 0.0 - k[1138]*y[IDX_CHI] - k[1140]*y[IDX_CHI];
    data[942] = 0.0 - k[656]*y[IDX_CHI] - k[658]*y[IDX_CHI] -
        k[2347]*y[IDX_CHI];
    data[943] = 0.0 - k[1336]*y[IDX_CHI] - k[1337]*y[IDX_CHI] -
        k[1339]*y[IDX_CHI];
    data[944] = 0.0 + k[1130]*y[IDX_CI] - k[1136]*y[IDX_CHI];
    data[945] = 0.0 - k[654]*y[IDX_CHI] - k[2345]*y[IDX_CHI];
    data[946] = 0.0 - k[1331]*y[IDX_CHI] - k[1332]*y[IDX_CHI] -
        k[1334]*y[IDX_CHI];
    data[947] = 0.0 - k[1326]*y[IDX_CHI] - k[1327]*y[IDX_CHI];
    data[948] = 0.0 + k[630]*y[IDX_pH2II] + k[631]*y[IDX_oH2II] +
        k[635]*y[IDX_HDII] + k[987]*y[IDX_H2OII] + k[990]*y[IDX_HDOII] +
        k[998]*y[IDX_HCNII] + k[1016]*y[IDX_HCOII] + k[1256]*y[IDX_oH3II] +
        k[1257]*y[IDX_pH3II] + k[1258]*y[IDX_pH3II] + k[1264]*y[IDX_oH2DII] +
        k[1265]*y[IDX_pH2DII] + k[1266]*y[IDX_oD2HII] + k[1267]*y[IDX_pD2HII] +
        k[1268]*y[IDX_pD2HII] + k[1552]*y[IDX_HNCII] + k[1570]*y[IDX_HNOII] +
        k[1616]*y[IDX_N2HII] - k[1711]*y[IDX_CHII] + k[1818]*y[IDX_NHII] +
        k[1921]*y[IDX_OHII] + k[2002]*y[IDX_O2HII];
    data[949] = 0.0 + k[447]*y[IDX_HCOI] + k[2309]*y[IDX_CHI] +
        k[2638]*y[IDX_HI];
    data[950] = 0.0 - k[1721]*y[IDX_CHII];
    data[951] = 0.0 + k[2614]*y[IDX_CHI];
    data[952] = 0.0 - k[1763]*y[IDX_CHII];
    data[953] = 0.0 + k[886]*y[IDX_HeII] - k[1761]*y[IDX_CHII];
    data[954] = 0.0 + k[950]*y[IDX_NI];
    data[955] = 0.0 - k[1725]*y[IDX_CHII];
    data[956] = 0.0 + k[1535]*y[IDX_HII] - k[1768]*y[IDX_CHII] -
        k[1769]*y[IDX_CHII];
    data[957] = 0.0 + k[356] - k[1723]*y[IDX_CHII] + k[2189]*y[IDX_HII] +
        k[2190]*y[IDX_DII] + k[2268]*y[IDX_CNII] + k[2272]*y[IDX_N2II] +
        k[2309]*y[IDX_CII] + k[2323]*y[IDX_COII] + k[2345]*y[IDX_pH2II] +
        k[2346]*y[IDX_oH2II] + k[2347]*y[IDX_pD2II] + k[2348]*y[IDX_oD2II] +
        k[2349]*y[IDX_HDII] + k[2365]*y[IDX_NII] + k[2469]*y[IDX_HeII] +
        k[2482]*y[IDX_H2OII] + k[2483]*y[IDX_D2OII] + k[2484]*y[IDX_HDOII] +
        k[2564]*y[IDX_OII] + k[2578]*y[IDX_OHII] + k[2579]*y[IDX_ODII] +
        k[2608]*y[IDX_NH2II] + k[2609]*y[IDX_ND2II] + k[2610]*y[IDX_NHDII] +
        k[2614]*y[IDX_C2II] + k[2631]*y[IDX_O2II];
    data[958] = 0.0 - k[230] - k[232] - k[288] - k[1711]*y[IDX_CI] -
        k[1713]*y[IDX_HI] - k[1715]*y[IDX_DI] - k[1717]*y[IDX_NI] -
        k[1719]*y[IDX_OI] - k[1721]*y[IDX_C2I] - k[1723]*y[IDX_CHI] -
        k[1725]*y[IDX_CDI] - k[1727]*y[IDX_CNI] - k[1729]*y[IDX_CNI] -
        k[1731]*y[IDX_pH2I] - k[1732]*y[IDX_oH2I] - k[1737]*y[IDX_pD2I] -
        k[1738]*y[IDX_oD2I] - k[1739]*y[IDX_pD2I] - k[1740]*y[IDX_oD2I] -
        k[1743]*y[IDX_HDI] - k[1744]*y[IDX_HDI] - k[1747]*y[IDX_NHI] -
        k[1749]*y[IDX_NDI] - k[1751]*y[IDX_O2I] - k[1753]*y[IDX_O2I] -
        k[1755]*y[IDX_O2I] - k[1757]*y[IDX_OHI] - k[1759]*y[IDX_ODI] -
        k[1761]*y[IDX_C2HI] - k[1763]*y[IDX_C2DI] - k[1765]*y[IDX_CH2I] -
        k[1768]*y[IDX_CD2I] - k[1769]*y[IDX_CD2I] - k[1771]*y[IDX_CHDI] -
        k[1772]*y[IDX_CHDI] - k[1775]*y[IDX_CO2I] - k[1777]*y[IDX_H2OI] -
        k[1780]*y[IDX_D2OI] - k[1781]*y[IDX_D2OI] - k[1783]*y[IDX_HDOI] -
        k[1784]*y[IDX_HDOI] - k[1787]*y[IDX_HCNI] - k[1789]*y[IDX_DCNI] -
        k[1791]*y[IDX_HCOI] - k[1793]*y[IDX_DCOI] - k[1795]*y[IDX_NH2I] -
        k[1798]*y[IDX_ND2I] - k[1799]*y[IDX_ND2I] - k[1801]*y[IDX_NHDI] -
        k[1802]*y[IDX_NHDI] - k[2261]*y[IDX_NOI] - k[2570]*y[IDX_HCOI] -
        k[2572]*y[IDX_DCOI] - k[2741]*y[IDX_eM] - k[3013]*y[IDX_H2OI] -
        k[3307]*y[IDX_HDOI] - k[3309]*y[IDX_D2OI];
    data[959] = 0.0 + k[895]*y[IDX_HeII] + k[1532]*y[IDX_HII] +
        k[1534]*y[IDX_DII] - k[1765]*y[IDX_CHII];
    data[960] = 0.0 + k[300];
    data[961] = 0.0 + k[898]*y[IDX_HeII] + k[1539]*y[IDX_HII] +
        k[1540]*y[IDX_DII] - k[1771]*y[IDX_CHII] - k[1772]*y[IDX_CHII];
    data[962] = 0.0 + k[303];
    data[963] = 0.0 - k[1727]*y[IDX_CHII] - k[1729]*y[IDX_CHII];
    data[964] = 0.0 + k[2268]*y[IDX_CHI];
    data[965] = 0.0 + k[2323]*y[IDX_CHI];
    data[966] = 0.0 - k[1775]*y[IDX_CHII];
    data[967] = 0.0 - k[1715]*y[IDX_CHII];
    data[968] = 0.0 + k[1534]*y[IDX_CH2I] + k[1540]*y[IDX_CHDI] +
        k[2190]*y[IDX_CHI];
    data[969] = 0.0 - k[1780]*y[IDX_CHII] - k[1781]*y[IDX_CHII] -
        k[3309]*y[IDX_CHII];
    data[970] = 0.0 + k[2483]*y[IDX_CHI];
    data[971] = 0.0 - k[1789]*y[IDX_CHII];
    data[972] = 0.0 - k[1793]*y[IDX_CHII] - k[2572]*y[IDX_CHII];
    data[973] = 0.0 - k[1713]*y[IDX_CHII] + k[2638]*y[IDX_CII];
    data[974] = 0.0 + k[1532]*y[IDX_CH2I] + k[1535]*y[IDX_CD2I] +
        k[1539]*y[IDX_CHDI] + k[2189]*y[IDX_CHI];
    data[975] = 0.0 - k[1777]*y[IDX_CHII] - k[3013]*y[IDX_CHII];
    data[976] = 0.0 + k[987]*y[IDX_CI] + k[2482]*y[IDX_CHI];
    data[977] = 0.0 + k[915]*y[IDX_HeII] - k[1787]*y[IDX_CHII];
    data[978] = 0.0 + k[998]*y[IDX_CI];
    data[979] = 0.0 + k[447]*y[IDX_CII] + k[919]*y[IDX_HeII] -
        k[1791]*y[IDX_CHII] - k[2570]*y[IDX_CHII];
    data[980] = 0.0 + k[1016]*y[IDX_CI];
    data[981] = 0.0 - k[1743]*y[IDX_CHII] - k[1744]*y[IDX_CHII];
    data[982] = 0.0 + k[635]*y[IDX_CI] + k[2349]*y[IDX_CHI];
    data[983] = 0.0 - k[1783]*y[IDX_CHII] - k[1784]*y[IDX_CHII] -
        k[3307]*y[IDX_CHII];
    data[984] = 0.0 + k[990]*y[IDX_CI] + k[2484]*y[IDX_CHI];
    data[985] = 0.0 + k[1552]*y[IDX_CI];
    data[986] = 0.0 + k[1570]*y[IDX_CI];
    data[987] = 0.0 + k[886]*y[IDX_C2HI] + k[895]*y[IDX_CH2I] +
        k[898]*y[IDX_CHDI] + k[915]*y[IDX_HCNI] + k[919]*y[IDX_HCOI] +
        k[2469]*y[IDX_CHI];
    data[988] = 0.0 + k[950]*y[IDX_C2HII] - k[1717]*y[IDX_CHII];
    data[989] = 0.0 + k[2365]*y[IDX_CHI];
    data[990] = 0.0 + k[2272]*y[IDX_CHI];
    data[991] = 0.0 + k[1616]*y[IDX_CI];
    data[992] = 0.0 - k[1749]*y[IDX_CHII];
    data[993] = 0.0 - k[1798]*y[IDX_CHII] - k[1799]*y[IDX_CHII];
    data[994] = 0.0 + k[2609]*y[IDX_CHI];
    data[995] = 0.0 - k[1747]*y[IDX_CHII];
    data[996] = 0.0 + k[1818]*y[IDX_CI];
    data[997] = 0.0 - k[1795]*y[IDX_CHII];
    data[998] = 0.0 + k[2608]*y[IDX_CHI];
    data[999] = 0.0 - k[1801]*y[IDX_CHII] - k[1802]*y[IDX_CHII];
    data[1000] = 0.0 + k[2610]*y[IDX_CHI];
    data[1001] = 0.0 - k[2261]*y[IDX_CHII];
    data[1002] = 0.0 - k[1719]*y[IDX_CHII];
    data[1003] = 0.0 + k[2564]*y[IDX_CHI];
    data[1004] = 0.0 - k[1751]*y[IDX_CHII] - k[1753]*y[IDX_CHII] -
        k[1755]*y[IDX_CHII];
    data[1005] = 0.0 + k[2631]*y[IDX_CHI];
    data[1006] = 0.0 + k[2002]*y[IDX_CI];
    data[1007] = 0.0 - k[1759]*y[IDX_CHII];
    data[1008] = 0.0 + k[2579]*y[IDX_CHI];
    data[1009] = 0.0 - k[1757]*y[IDX_CHII];
    data[1010] = 0.0 + k[1921]*y[IDX_CI] + k[2578]*y[IDX_CHI];
    data[1011] = 0.0 - k[2741]*y[IDX_CHII];
    data[1012] = 0.0 - k[1738]*y[IDX_CHII] - k[1740]*y[IDX_CHII];
    data[1013] = 0.0 + k[2348]*y[IDX_CHI];
    data[1014] = 0.0 + k[1266]*y[IDX_CI];
    data[1015] = 0.0 - k[1732]*y[IDX_CHII];
    data[1016] = 0.0 + k[631]*y[IDX_CI] + k[2346]*y[IDX_CHI];
    data[1017] = 0.0 + k[1264]*y[IDX_CI];
    data[1018] = 0.0 + k[1256]*y[IDX_CI];
    data[1019] = 0.0 - k[1737]*y[IDX_CHII] - k[1739]*y[IDX_CHII];
    data[1020] = 0.0 + k[2347]*y[IDX_CHI];
    data[1021] = 0.0 + k[1267]*y[IDX_CI] + k[1268]*y[IDX_CI];
    data[1022] = 0.0 - k[1731]*y[IDX_CHII];
    data[1023] = 0.0 + k[630]*y[IDX_CI] + k[2345]*y[IDX_CHI];
    data[1024] = 0.0 + k[1265]*y[IDX_CI];
    data[1025] = 0.0 + k[1257]*y[IDX_CI] + k[1258]*y[IDX_CI];
    data[1026] = 0.0 - k[1034]*y[IDX_CH2I] - k[1037]*y[IDX_CH2I] +
        k[2657]*y[IDX_pH2I] + k[2658]*y[IDX_oH2I];
    data[1027] = 0.0 - k[430]*y[IDX_CH2I] - k[2311]*y[IDX_CH2I];
    data[1028] = 0.0 + k[2677]*y[IDX_pH2I] + k[2678]*y[IDX_oH2I];
    data[1029] = 0.0 - k[2616]*y[IDX_CH2I];
    data[1030] = 0.0 + k[1146]*y[IDX_pH2I] + k[1147]*y[IDX_oH2I];
    data[1031] = 0.0 - k[1766]*y[IDX_CH2I] - k[1767]*y[IDX_CH2I];
    data[1032] = 0.0 + k[541]*y[IDX_HNOI] + k[1136]*y[IDX_pH2I] +
        k[1137]*y[IDX_oH2I] + k[1143]*y[IDX_HDI] + k[2699]*y[IDX_HM];
    data[1033] = 0.0 - k[1765]*y[IDX_CH2I];
    data[1034] = 0.0 - k[249] - k[385] - k[389] - k[430]*y[IDX_CII] -
        k[509]*y[IDX_OI] - k[512]*y[IDX_OI] - k[618]*y[IDX_COII] -
        k[892]*y[IDX_HeII] - k[895]*y[IDX_HeII] - k[1034]*y[IDX_CI] -
        k[1037]*y[IDX_CI] - k[1068]*y[IDX_HI] - k[1073]*y[IDX_DI] -
        k[1074]*y[IDX_DI] - k[1206]*y[IDX_NI] - k[1210]*y[IDX_NI] -
        k[1532]*y[IDX_HII] - k[1533]*y[IDX_DII] - k[1534]*y[IDX_DII] -
        k[1765]*y[IDX_CHII] - k[1766]*y[IDX_CDII] - k[1767]*y[IDX_CDII] -
        k[2075]*y[IDX_CNII] - k[2228]*y[IDX_NII] - k[2243]*y[IDX_OII] -
        k[2284]*y[IDX_N2II] - k[2311]*y[IDX_CII] - k[2331]*y[IDX_COII] -
        k[2428]*y[IDX_pH2II] - k[2429]*y[IDX_oH2II] - k[2430]*y[IDX_pD2II] -
        k[2431]*y[IDX_oD2II] - k[2432]*y[IDX_HDII] - k[2494]*y[IDX_H2OII] -
        k[2495]*y[IDX_D2OII] - k[2496]*y[IDX_HDOII] - k[2527]*y[IDX_HII] -
        k[2528]*y[IDX_DII] - k[2546]*y[IDX_NH2II] - k[2547]*y[IDX_ND2II] -
        k[2548]*y[IDX_NHDII] - k[2586]*y[IDX_OHII] - k[2587]*y[IDX_ODII] -
        k[2616]*y[IDX_C2II] - k[2633]*y[IDX_O2II];
    data[1035] = 0.0 + k[2164]*y[IDX_NOI];
    data[1036] = 0.0 - k[2075]*y[IDX_CH2I];
    data[1037] = 0.0 - k[618]*y[IDX_CH2I] - k[2331]*y[IDX_CH2I];
    data[1038] = 0.0 - k[1073]*y[IDX_CH2I] - k[1074]*y[IDX_CH2I];
    data[1039] = 0.0 - k[1533]*y[IDX_CH2I] - k[1534]*y[IDX_CH2I] -
        k[2528]*y[IDX_CH2I];
    data[1040] = 0.0 - k[2495]*y[IDX_CH2I];
    data[1041] = 0.0 + k[1060]*y[IDX_HCOI] - k[1068]*y[IDX_CH2I];
    data[1042] = 0.0 - k[1532]*y[IDX_CH2I] - k[2527]*y[IDX_CH2I];
    data[1043] = 0.0 + k[2699]*y[IDX_CHI];
    data[1044] = 0.0 - k[2494]*y[IDX_CH2I];
    data[1045] = 0.0 + k[1060]*y[IDX_HI];
    data[1046] = 0.0 + k[1143]*y[IDX_CHI];
    data[1047] = 0.0 - k[2432]*y[IDX_CH2I];
    data[1048] = 0.0 - k[2496]*y[IDX_CH2I];
    data[1049] = 0.0 + k[541]*y[IDX_CHI];
    data[1050] = 0.0 - k[892]*y[IDX_CH2I] - k[895]*y[IDX_CH2I];
    data[1051] = 0.0 - k[1206]*y[IDX_CH2I] - k[1210]*y[IDX_CH2I];
    data[1052] = 0.0 - k[2228]*y[IDX_CH2I];
    data[1053] = 0.0 - k[2284]*y[IDX_CH2I];
    data[1054] = 0.0 - k[2547]*y[IDX_CH2I];
    data[1055] = 0.0 - k[2546]*y[IDX_CH2I];
    data[1056] = 0.0 - k[2548]*y[IDX_CH2I];
    data[1057] = 0.0 + k[2164]*y[IDX_CH2II];
    data[1058] = 0.0 - k[509]*y[IDX_CH2I] - k[512]*y[IDX_CH2I];
    data[1059] = 0.0 - k[2243]*y[IDX_CH2I];
    data[1060] = 0.0 - k[2633]*y[IDX_CH2I];
    data[1061] = 0.0 - k[2587]*y[IDX_CH2I];
    data[1062] = 0.0 - k[2586]*y[IDX_CH2I];
    data[1063] = 0.0 - k[2431]*y[IDX_CH2I];
    data[1064] = 0.0 + k[1137]*y[IDX_CHI] + k[1147]*y[IDX_CDI] +
        k[2658]*y[IDX_CI] + k[2678]*y[IDX_CM];
    data[1065] = 0.0 - k[2429]*y[IDX_CH2I];
    data[1066] = 0.0 - k[2430]*y[IDX_CH2I];
    data[1067] = 0.0 + k[1136]*y[IDX_CHI] + k[1146]*y[IDX_CDI] +
        k[2657]*y[IDX_CI] + k[2677]*y[IDX_CM];
    data[1068] = 0.0 - k[2428]*y[IDX_CH2I];
    data[1069] = 0.0 - k[964]*y[IDX_CH2II];
    data[1070] = 0.0 + k[2311]*y[IDX_CH2I] + k[2641]*y[IDX_pH2I] +
        k[2642]*y[IDX_oH2I];
    data[1071] = 0.0 + k[2616]*y[IDX_CH2I];
    data[1072] = 0.0 + k[956]*y[IDX_CHI];
    data[1073] = 0.0 + k[664]*y[IDX_pH2II] + k[665]*y[IDX_oH2II] +
        k[1248]*y[IDX_H2OII] + k[1343]*y[IDX_oH3II] + k[1344]*y[IDX_pH3II] +
        k[1987]*y[IDX_NH2II] + k[3299]*y[IDX_H3OII] + k[3302]*y[IDX_H2DOII];
    data[1074] = 0.0 + k[1735]*y[IDX_pH2I] + k[1736]*y[IDX_oH2I];
    data[1075] = 0.0 + k[654]*y[IDX_pH2II] + k[655]*y[IDX_oH2II] +
        k[661]*y[IDX_HDII] + k[956]*y[IDX_C2HII] + k[1002]*y[IDX_HCNII] +
        k[1020]*y[IDX_HCOII] + k[1242]*y[IDX_H2OII] + k[1246]*y[IDX_HDOII] +
        k[1325]*y[IDX_oH3II] + k[1326]*y[IDX_pH3II] + k[1327]*y[IDX_pH3II] +
        k[1333]*y[IDX_oH2DII] + k[1334]*y[IDX_pH2DII] + k[1335]*y[IDX_oD2HII] +
        k[1336]*y[IDX_pD2HII] + k[1337]*y[IDX_pD2HII] + k[1556]*y[IDX_HNCII] +
        k[1576]*y[IDX_HNOII] + k[1620]*y[IDX_N2HII] + k[1830]*y[IDX_NHII] +
        k[1929]*y[IDX_OHII] + k[1981]*y[IDX_NH2II] + k[1985]*y[IDX_NHDII] +
        k[2010]*y[IDX_O2HII] + k[3012]*y[IDX_H3OII] + k[3292]*y[IDX_H2DOII] +
        k[3295]*y[IDX_HD2OII];
    data[1076] = 0.0 + k[1731]*y[IDX_pH2I] + k[1732]*y[IDX_oH2I] +
        k[1744]*y[IDX_HDI] + k[1791]*y[IDX_HCOI];
    data[1077] = 0.0 + k[249] + k[389] + k[2075]*y[IDX_CNII] +
        k[2228]*y[IDX_NII] + k[2243]*y[IDX_OII] + k[2284]*y[IDX_N2II] +
        k[2311]*y[IDX_CII] + k[2331]*y[IDX_COII] + k[2428]*y[IDX_pH2II] +
        k[2429]*y[IDX_oH2II] + k[2430]*y[IDX_pD2II] + k[2431]*y[IDX_oD2II] +
        k[2432]*y[IDX_HDII] + k[2494]*y[IDX_H2OII] + k[2495]*y[IDX_D2OII] +
        k[2496]*y[IDX_HDOII] + k[2527]*y[IDX_HII] + k[2528]*y[IDX_DII] +
        k[2546]*y[IDX_NH2II] + k[2547]*y[IDX_ND2II] + k[2548]*y[IDX_NHDII] +
        k[2586]*y[IDX_OHII] + k[2587]*y[IDX_ODII] + k[2616]*y[IDX_C2II] +
        k[2633]*y[IDX_O2II];
    data[1078] = 0.0 - k[300] - k[964]*y[IDX_CI] - k[968]*y[IDX_NI] -
        k[972]*y[IDX_OI] - k[976]*y[IDX_O2I] - k[2164]*y[IDX_NOI] -
        k[2775]*y[IDX_eM] - k[2778]*y[IDX_eM] - k[2782]*y[IDX_eM];
    data[1079] = 0.0 + k[2075]*y[IDX_CH2I];
    data[1080] = 0.0 + k[2331]*y[IDX_CH2I];
    data[1081] = 0.0 + k[2528]*y[IDX_CH2I];
    data[1082] = 0.0 + k[2495]*y[IDX_CH2I];
    data[1083] = 0.0 + k[2527]*y[IDX_CH2I];
    data[1084] = 0.0 + k[3292]*y[IDX_CHI] + k[3302]*y[IDX_CDI];
    data[1085] = 0.0 + k[1242]*y[IDX_CHI] + k[1248]*y[IDX_CDI] +
        k[2494]*y[IDX_CH2I];
    data[1086] = 0.0 + k[3012]*y[IDX_CHI] + k[3299]*y[IDX_CDI];
    data[1087] = 0.0 + k[1002]*y[IDX_CHI];
    data[1088] = 0.0 + k[1791]*y[IDX_CHII];
    data[1089] = 0.0 + k[1020]*y[IDX_CHI];
    data[1090] = 0.0 + k[1744]*y[IDX_CHII];
    data[1091] = 0.0 + k[661]*y[IDX_CHI] + k[2432]*y[IDX_CH2I];
    data[1092] = 0.0 + k[3295]*y[IDX_CHI];
    data[1093] = 0.0 + k[1246]*y[IDX_CHI] + k[2496]*y[IDX_CH2I];
    data[1094] = 0.0 + k[1556]*y[IDX_CHI];
    data[1095] = 0.0 + k[1576]*y[IDX_CHI];
    data[1096] = 0.0 - k[968]*y[IDX_CH2II];
    data[1097] = 0.0 + k[2228]*y[IDX_CH2I];
    data[1098] = 0.0 + k[2284]*y[IDX_CH2I];
    data[1099] = 0.0 + k[1620]*y[IDX_CHI];
    data[1100] = 0.0 + k[2547]*y[IDX_CH2I];
    data[1101] = 0.0 + k[1830]*y[IDX_CHI];
    data[1102] = 0.0 + k[1981]*y[IDX_CHI] + k[1987]*y[IDX_CDI] +
        k[2546]*y[IDX_CH2I];
    data[1103] = 0.0 + k[1985]*y[IDX_CHI] + k[2548]*y[IDX_CH2I];
    data[1104] = 0.0 - k[2164]*y[IDX_CH2II];
    data[1105] = 0.0 - k[972]*y[IDX_CH2II];
    data[1106] = 0.0 + k[2243]*y[IDX_CH2I];
    data[1107] = 0.0 - k[976]*y[IDX_CH2II];
    data[1108] = 0.0 + k[2633]*y[IDX_CH2I];
    data[1109] = 0.0 + k[2010]*y[IDX_CHI];
    data[1110] = 0.0 + k[2587]*y[IDX_CH2I];
    data[1111] = 0.0 + k[1929]*y[IDX_CHI] + k[2586]*y[IDX_CH2I];
    data[1112] = 0.0 - k[2775]*y[IDX_CH2II] - k[2778]*y[IDX_CH2II] -
        k[2782]*y[IDX_CH2II];
    data[1113] = 0.0 + k[2431]*y[IDX_CH2I];
    data[1114] = 0.0 + k[1335]*y[IDX_CHI];
    data[1115] = 0.0 + k[1732]*y[IDX_CHII] + k[1736]*y[IDX_CDII] +
        k[2642]*y[IDX_CII];
    data[1116] = 0.0 + k[655]*y[IDX_CHI] + k[665]*y[IDX_CDI] +
        k[2429]*y[IDX_CH2I];
    data[1117] = 0.0 + k[1333]*y[IDX_CHI];
    data[1118] = 0.0 + k[1325]*y[IDX_CHI] + k[1343]*y[IDX_CDI];
    data[1119] = 0.0 + k[2430]*y[IDX_CH2I];
    data[1120] = 0.0 + k[1336]*y[IDX_CHI] + k[1337]*y[IDX_CHI];
    data[1121] = 0.0 + k[1731]*y[IDX_CHII] + k[1735]*y[IDX_CDII] +
        k[2641]*y[IDX_CII];
    data[1122] = 0.0 + k[654]*y[IDX_CHI] + k[664]*y[IDX_CDI] +
        k[2428]*y[IDX_CH2I];
    data[1123] = 0.0 + k[1334]*y[IDX_CHI];
    data[1124] = 0.0 + k[1326]*y[IDX_CHI] + k[1327]*y[IDX_CHI] +
        k[1344]*y[IDX_CDI];
    data[1125] = 0.0 - k[1036]*y[IDX_CHDI] - k[1039]*y[IDX_CHDI] -
        k[1040]*y[IDX_CHDI] + k[2661]*y[IDX_HDI];
    data[1126] = 0.0 - k[432]*y[IDX_CHDI] - k[433]*y[IDX_CHDI] -
        k[2313]*y[IDX_CHDI];
    data[1127] = 0.0 + k[2681]*y[IDX_HDI];
    data[1128] = 0.0 - k[2618]*y[IDX_CHDI];
    data[1129] = 0.0 + k[543]*y[IDX_HNOI] + k[1144]*y[IDX_pH2I] +
        k[1145]*y[IDX_oH2I] + k[1151]*y[IDX_HDI] + k[2701]*y[IDX_HM];
    data[1130] = 0.0 - k[1773]*y[IDX_CHDI] - k[1774]*y[IDX_CHDI];
    data[1131] = 0.0 + k[542]*y[IDX_DNOI] + k[1140]*y[IDX_pD2I] +
        k[1141]*y[IDX_oD2I] + k[1142]*y[IDX_HDI] + k[2700]*y[IDX_DM];
    data[1132] = 0.0 - k[1771]*y[IDX_CHDI] - k[1772]*y[IDX_CHDI];
    data[1133] = 0.0 - k[251] - k[387] - k[388] - k[391] - k[432]*y[IDX_CII]
        - k[433]*y[IDX_CII] - k[511]*y[IDX_OI] - k[514]*y[IDX_OI] -
        k[620]*y[IDX_COII] - k[621]*y[IDX_COII] - k[894]*y[IDX_HeII] -
        k[897]*y[IDX_HeII] - k[898]*y[IDX_HeII] - k[1036]*y[IDX_CI] -
        k[1039]*y[IDX_CI] - k[1040]*y[IDX_CI] - k[1071]*y[IDX_HI] -
        k[1072]*y[IDX_HI] - k[1076]*y[IDX_DI] - k[1077]*y[IDX_DI] -
        k[1208]*y[IDX_NI] - k[1209]*y[IDX_NI] - k[1212]*y[IDX_NI] -
        k[1213]*y[IDX_NI] - k[1538]*y[IDX_HII] - k[1539]*y[IDX_HII] -
        k[1540]*y[IDX_DII] - k[1541]*y[IDX_DII] - k[1771]*y[IDX_CHII] -
        k[1772]*y[IDX_CHII] - k[1773]*y[IDX_CDII] - k[1774]*y[IDX_CDII] -
        k[2077]*y[IDX_CNII] - k[2230]*y[IDX_NII] - k[2245]*y[IDX_OII] -
        k[2286]*y[IDX_N2II] - k[2313]*y[IDX_CII] - k[2333]*y[IDX_COII] -
        k[2438]*y[IDX_pH2II] - k[2439]*y[IDX_oH2II] - k[2440]*y[IDX_pD2II] -
        k[2441]*y[IDX_oD2II] - k[2442]*y[IDX_HDII] - k[2500]*y[IDX_H2OII] -
        k[2501]*y[IDX_D2OII] - k[2502]*y[IDX_HDOII] - k[2531]*y[IDX_HII] -
        k[2532]*y[IDX_DII] - k[2552]*y[IDX_NH2II] - k[2553]*y[IDX_ND2II] -
        k[2554]*y[IDX_NHDII] - k[2590]*y[IDX_OHII] - k[2591]*y[IDX_ODII] -
        k[2618]*y[IDX_C2II] - k[2635]*y[IDX_O2II];
    data[1134] = 0.0 + k[2166]*y[IDX_NOI];
    data[1135] = 0.0 - k[2077]*y[IDX_CHDI];
    data[1136] = 0.0 - k[620]*y[IDX_CHDI] - k[621]*y[IDX_CHDI] -
        k[2333]*y[IDX_CHDI];
    data[1137] = 0.0 + k[1062]*y[IDX_HCOI] - k[1076]*y[IDX_CHDI] -
        k[1077]*y[IDX_CHDI];
    data[1138] = 0.0 - k[1540]*y[IDX_CHDI] - k[1541]*y[IDX_CHDI] -
        k[2532]*y[IDX_CHDI];
    data[1139] = 0.0 + k[2700]*y[IDX_CHI];
    data[1140] = 0.0 - k[2501]*y[IDX_CHDI];
    data[1141] = 0.0 + k[1061]*y[IDX_HI];
    data[1142] = 0.0 + k[542]*y[IDX_CHI];
    data[1143] = 0.0 + k[1061]*y[IDX_DCOI] - k[1071]*y[IDX_CHDI] -
        k[1072]*y[IDX_CHDI];
    data[1144] = 0.0 - k[1538]*y[IDX_CHDI] - k[1539]*y[IDX_CHDI] -
        k[2531]*y[IDX_CHDI];
    data[1145] = 0.0 + k[2701]*y[IDX_CDI];
    data[1146] = 0.0 - k[2500]*y[IDX_CHDI];
    data[1147] = 0.0 + k[1062]*y[IDX_DI];
    data[1148] = 0.0 + k[1142]*y[IDX_CHI] + k[1151]*y[IDX_CDI] +
        k[2661]*y[IDX_CI] + k[2681]*y[IDX_CM];
    data[1149] = 0.0 - k[2442]*y[IDX_CHDI];
    data[1150] = 0.0 - k[2502]*y[IDX_CHDI];
    data[1151] = 0.0 + k[543]*y[IDX_CDI];
    data[1152] = 0.0 - k[894]*y[IDX_CHDI] - k[897]*y[IDX_CHDI] -
        k[898]*y[IDX_CHDI];
    data[1153] = 0.0 - k[1208]*y[IDX_CHDI] - k[1209]*y[IDX_CHDI] -
        k[1212]*y[IDX_CHDI] - k[1213]*y[IDX_CHDI];
    data[1154] = 0.0 - k[2230]*y[IDX_CHDI];
    data[1155] = 0.0 - k[2286]*y[IDX_CHDI];
    data[1156] = 0.0 - k[2553]*y[IDX_CHDI];
    data[1157] = 0.0 - k[2552]*y[IDX_CHDI];
    data[1158] = 0.0 - k[2554]*y[IDX_CHDI];
    data[1159] = 0.0 + k[2166]*y[IDX_CHDII];
    data[1160] = 0.0 - k[511]*y[IDX_CHDI] - k[514]*y[IDX_CHDI];
    data[1161] = 0.0 - k[2245]*y[IDX_CHDI];
    data[1162] = 0.0 - k[2635]*y[IDX_CHDI];
    data[1163] = 0.0 - k[2591]*y[IDX_CHDI];
    data[1164] = 0.0 - k[2590]*y[IDX_CHDI];
    data[1165] = 0.0 + k[1141]*y[IDX_CHI];
    data[1166] = 0.0 - k[2441]*y[IDX_CHDI];
    data[1167] = 0.0 + k[1145]*y[IDX_CDI];
    data[1168] = 0.0 - k[2439]*y[IDX_CHDI];
    data[1169] = 0.0 + k[1140]*y[IDX_CHI];
    data[1170] = 0.0 - k[2440]*y[IDX_CHDI];
    data[1171] = 0.0 + k[1144]*y[IDX_CDI];
    data[1172] = 0.0 - k[2438]*y[IDX_CHDI];
    data[1173] = 0.0 - k[966]*y[IDX_CHDII] - k[967]*y[IDX_CHDII];
    data[1174] = 0.0 + k[2313]*y[IDX_CHDI] + k[2645]*y[IDX_HDI];
    data[1175] = 0.0 + k[2618]*y[IDX_CHDI];
    data[1176] = 0.0 + k[957]*y[IDX_CHI];
    data[1177] = 0.0 + k[958]*y[IDX_CDI];
    data[1178] = 0.0 + k[662]*y[IDX_pH2II] + k[663]*y[IDX_oH2II] +
        k[669]*y[IDX_HDII] + k[958]*y[IDX_C2HII] + k[1004]*y[IDX_HCNII] +
        k[1022]*y[IDX_HCOII] + k[1247]*y[IDX_H2OII] + k[1251]*y[IDX_HDOII] +
        k[1340]*y[IDX_oH3II] + k[1341]*y[IDX_pH3II] + k[1342]*y[IDX_pH3II] +
        k[1350]*y[IDX_oH2DII] + k[1351]*y[IDX_pH2DII] + k[1352]*y[IDX_oD2HII] +
        k[1353]*y[IDX_pD2HII] + k[1354]*y[IDX_pD2HII] + k[1558]*y[IDX_HNCII] +
        k[1578]*y[IDX_HNOII] + k[1622]*y[IDX_N2HII] + k[1832]*y[IDX_NHII] +
        k[1931]*y[IDX_OHII] + k[1986]*y[IDX_NH2II] + k[1990]*y[IDX_NHDII] +
        k[2012]*y[IDX_O2HII] + k[3298]*y[IDX_H3OII] + k[3301]*y[IDX_H2DOII] +
        k[3304]*y[IDX_HD2OII];
    data[1179] = 0.0 + k[1733]*y[IDX_pH2I] + k[1734]*y[IDX_oH2I] +
        k[1746]*y[IDX_HDI] + k[1792]*y[IDX_HCOI];
    data[1180] = 0.0 + k[658]*y[IDX_pD2II] + k[659]*y[IDX_oD2II] +
        k[660]*y[IDX_HDII] + k[957]*y[IDX_C2DII] + k[1003]*y[IDX_DCNII] +
        k[1021]*y[IDX_DCOII] + k[1244]*y[IDX_D2OII] + k[1245]*y[IDX_HDOII] +
        k[1330]*y[IDX_oH2DII] + k[1331]*y[IDX_pH2DII] + k[1332]*y[IDX_pH2DII] +
        k[1338]*y[IDX_oD2HII] + k[1339]*y[IDX_pD2HII] + k[1557]*y[IDX_DNCII] +
        k[1577]*y[IDX_DNOII] + k[1621]*y[IDX_N2DII] + k[1831]*y[IDX_NDII] +
        k[1930]*y[IDX_ODII] + k[1983]*y[IDX_ND2II] + k[1984]*y[IDX_NHDII] +
        k[2011]*y[IDX_O2DII] + k[3291]*y[IDX_H2DOII] + k[3294]*y[IDX_HD2OII] +
        k[3297]*y[IDX_D3OII];
    data[1181] = 0.0 + k[1739]*y[IDX_pD2I] + k[1740]*y[IDX_oD2I] +
        k[1743]*y[IDX_HDI] + k[1793]*y[IDX_DCOI];
    data[1182] = 0.0 + k[251] + k[391] + k[2077]*y[IDX_CNII] +
        k[2230]*y[IDX_NII] + k[2245]*y[IDX_OII] + k[2286]*y[IDX_N2II] +
        k[2313]*y[IDX_CII] + k[2333]*y[IDX_COII] + k[2438]*y[IDX_pH2II] +
        k[2439]*y[IDX_oH2II] + k[2440]*y[IDX_pD2II] + k[2441]*y[IDX_oD2II] +
        k[2442]*y[IDX_HDII] + k[2500]*y[IDX_H2OII] + k[2501]*y[IDX_D2OII] +
        k[2502]*y[IDX_HDOII] + k[2531]*y[IDX_HII] + k[2532]*y[IDX_DII] +
        k[2552]*y[IDX_NH2II] + k[2553]*y[IDX_ND2II] + k[2554]*y[IDX_NHDII] +
        k[2590]*y[IDX_OHII] + k[2591]*y[IDX_ODII] + k[2618]*y[IDX_C2II] +
        k[2635]*y[IDX_O2II];
    data[1183] = 0.0 - k[302] - k[303] - k[966]*y[IDX_CI] - k[967]*y[IDX_CI]
        - k[970]*y[IDX_NI] - k[971]*y[IDX_NI] - k[974]*y[IDX_OI] -
        k[975]*y[IDX_OI] - k[978]*y[IDX_O2I] - k[979]*y[IDX_O2I] -
        k[2166]*y[IDX_NOI] - k[2777]*y[IDX_eM] - k[2780]*y[IDX_eM] -
        k[2781]*y[IDX_eM] - k[2784]*y[IDX_eM];
    data[1184] = 0.0 + k[2077]*y[IDX_CHDI];
    data[1185] = 0.0 + k[2333]*y[IDX_CHDI];
    data[1186] = 0.0 + k[2532]*y[IDX_CHDI];
    data[1187] = 0.0 + k[1244]*y[IDX_CHI] + k[2501]*y[IDX_CHDI];
    data[1188] = 0.0 + k[3297]*y[IDX_CHI];
    data[1189] = 0.0 + k[1003]*y[IDX_CHI];
    data[1190] = 0.0 + k[1793]*y[IDX_CHII];
    data[1191] = 0.0 + k[1021]*y[IDX_CHI];
    data[1192] = 0.0 + k[1557]*y[IDX_CHI];
    data[1193] = 0.0 + k[1577]*y[IDX_CHI];
    data[1194] = 0.0 + k[2531]*y[IDX_CHDI];
    data[1195] = 0.0 + k[3291]*y[IDX_CHI] + k[3301]*y[IDX_CDI];
    data[1196] = 0.0 + k[1247]*y[IDX_CDI] + k[2500]*y[IDX_CHDI];
    data[1197] = 0.0 + k[3298]*y[IDX_CDI];
    data[1198] = 0.0 + k[1004]*y[IDX_CDI];
    data[1199] = 0.0 + k[1792]*y[IDX_CDII];
    data[1200] = 0.0 + k[1022]*y[IDX_CDI];
    data[1201] = 0.0 + k[1743]*y[IDX_CHII] + k[1746]*y[IDX_CDII] +
        k[2645]*y[IDX_CII];
    data[1202] = 0.0 + k[660]*y[IDX_CHI] + k[669]*y[IDX_CDI] +
        k[2442]*y[IDX_CHDI];
    data[1203] = 0.0 + k[3294]*y[IDX_CHI] + k[3304]*y[IDX_CDI];
    data[1204] = 0.0 + k[1245]*y[IDX_CHI] + k[1251]*y[IDX_CDI] +
        k[2502]*y[IDX_CHDI];
    data[1205] = 0.0 + k[1558]*y[IDX_CDI];
    data[1206] = 0.0 + k[1578]*y[IDX_CDI];
    data[1207] = 0.0 - k[970]*y[IDX_CHDII] - k[971]*y[IDX_CHDII];
    data[1208] = 0.0 + k[2230]*y[IDX_CHDI];
    data[1209] = 0.0 + k[2286]*y[IDX_CHDI];
    data[1210] = 0.0 + k[1621]*y[IDX_CHI];
    data[1211] = 0.0 + k[1622]*y[IDX_CDI];
    data[1212] = 0.0 + k[1831]*y[IDX_CHI];
    data[1213] = 0.0 + k[1983]*y[IDX_CHI] + k[2553]*y[IDX_CHDI];
    data[1214] = 0.0 + k[1832]*y[IDX_CDI];
    data[1215] = 0.0 + k[1986]*y[IDX_CDI] + k[2552]*y[IDX_CHDI];
    data[1216] = 0.0 + k[1984]*y[IDX_CHI] + k[1990]*y[IDX_CDI] +
        k[2554]*y[IDX_CHDI];
    data[1217] = 0.0 - k[2166]*y[IDX_CHDII];
    data[1218] = 0.0 - k[974]*y[IDX_CHDII] - k[975]*y[IDX_CHDII];
    data[1219] = 0.0 + k[2245]*y[IDX_CHDI];
    data[1220] = 0.0 - k[978]*y[IDX_CHDII] - k[979]*y[IDX_CHDII];
    data[1221] = 0.0 + k[2635]*y[IDX_CHDI];
    data[1222] = 0.0 + k[2011]*y[IDX_CHI];
    data[1223] = 0.0 + k[2012]*y[IDX_CDI];
    data[1224] = 0.0 + k[1930]*y[IDX_CHI] + k[2591]*y[IDX_CHDI];
    data[1225] = 0.0 + k[1931]*y[IDX_CDI] + k[2590]*y[IDX_CHDI];
    data[1226] = 0.0 - k[2777]*y[IDX_CHDII] - k[2780]*y[IDX_CHDII] -
        k[2781]*y[IDX_CHDII] - k[2784]*y[IDX_CHDII];
    data[1227] = 0.0 + k[1740]*y[IDX_CHII];
    data[1228] = 0.0 + k[659]*y[IDX_CHI] + k[2441]*y[IDX_CHDI];
    data[1229] = 0.0 + k[1338]*y[IDX_CHI] + k[1352]*y[IDX_CDI];
    data[1230] = 0.0 + k[1734]*y[IDX_CDII];
    data[1231] = 0.0 + k[663]*y[IDX_CDI] + k[2439]*y[IDX_CHDI];
    data[1232] = 0.0 + k[1330]*y[IDX_CHI] + k[1350]*y[IDX_CDI];
    data[1233] = 0.0 + k[1340]*y[IDX_CDI];
    data[1234] = 0.0 + k[1739]*y[IDX_CHII];
    data[1235] = 0.0 + k[658]*y[IDX_CHI] + k[2440]*y[IDX_CHDI];
    data[1236] = 0.0 + k[1339]*y[IDX_CHI] + k[1353]*y[IDX_CDI] +
        k[1354]*y[IDX_CDI];
    data[1237] = 0.0 + k[1733]*y[IDX_CDII];
    data[1238] = 0.0 + k[662]*y[IDX_CDI] + k[2438]*y[IDX_CHDI];
    data[1239] = 0.0 + k[1331]*y[IDX_CHI] + k[1332]*y[IDX_CHI] +
        k[1351]*y[IDX_CDI];
    data[1240] = 0.0 + k[1341]*y[IDX_CDI] + k[1342]*y[IDX_CDI];
    data[1241] = 0.0 + k[998]*y[IDX_HCNII] + k[999]*y[IDX_DCNII] -
        k[1053]*y[IDX_CNI] + k[1054]*y[IDX_N2I] + k[1552]*y[IDX_HNCII] +
        k[1553]*y[IDX_DNCII] + k[2048]*y[IDX_NHI] + k[2049]*y[IDX_NDI] +
        k[2050]*y[IDX_NOI] + k[2059]*y[IDX_C2NI] + k[2063]*y[IDX_OCNI] +
        k[2263]*y[IDX_CNII] + k[2655]*y[IDX_NI];
    data[1242] = 0.0 + k[455]*y[IDX_OCNI];
    data[1243] = 0.0 + k[2673]*y[IDX_NI];
    data[1244] = 0.0 + k[1000]*y[IDX_HCNII] + k[1001]*y[IDX_DCNII] +
        k[1196]*y[IDX_NI] + k[1554]*y[IDX_HNCII] + k[1555]*y[IDX_DNCII] +
        k[2267]*y[IDX_CNII];
    data[1245] = 0.0 + k[1689]*y[IDX_NI];
    data[1246] = 0.0 + k[2074]*y[IDX_CNII];
    data[1247] = 0.0 + k[951]*y[IDX_NI];
    data[1248] = 0.0 + k[2073]*y[IDX_CNII];
    data[1249] = 0.0 + k[950]*y[IDX_NI];
    data[1250] = 0.0 + k[245] + k[383] + k[888]*y[IDX_HeII] +
        k[1223]*y[IDX_NI] + k[1223]*y[IDX_NI] + k[1236]*y[IDX_OI] +
        k[2059]*y[IDX_CI];
    data[1251] = 0.0 + k[2771]*y[IDX_eM];
    data[1252] = 0.0 + k[1225]*y[IDX_NI];
    data[1253] = 0.0 + k[1224]*y[IDX_NI];
    data[1254] = 0.0 + k[1004]*y[IDX_HCNII] + k[1005]*y[IDX_DCNII] +
        k[1198]*y[IDX_NI] + k[1558]*y[IDX_HNCII] + k[1559]*y[IDX_DNCII] +
        k[2269]*y[IDX_CNII];
    data[1255] = 0.0 - k[1728]*y[IDX_CNI] - k[1730]*y[IDX_CNI];
    data[1256] = 0.0 + k[2076]*y[IDX_CNII];
    data[1257] = 0.0 + k[1002]*y[IDX_HCNII] + k[1003]*y[IDX_DCNII] +
        k[1197]*y[IDX_NI] + k[1556]*y[IDX_HNCII] + k[1557]*y[IDX_DNCII] +
        k[2268]*y[IDX_CNII];
    data[1258] = 0.0 - k[1727]*y[IDX_CNI] - k[1729]*y[IDX_CNI];
    data[1259] = 0.0 + k[2075]*y[IDX_CNII];
    data[1260] = 0.0 + k[2077]*y[IDX_CNII];
    data[1261] = 0.0 - k[284] - k[358] - k[547]*y[IDX_NOI] -
        k[548]*y[IDX_O2I] - k[549]*y[IDX_HCOI] - k[550]*y[IDX_DCOI] -
        k[551]*y[IDX_OHI] - k[552]*y[IDX_ODI] - k[553]*y[IDX_OHI] -
        k[554]*y[IDX_ODI] - k[555]*y[IDX_HNOI] - k[556]*y[IDX_DNOI] -
        k[670]*y[IDX_pH2II] - k[671]*y[IDX_oH2II] - k[672]*y[IDX_pD2II] -
        k[673]*y[IDX_oD2II] - k[674]*y[IDX_HDII] - k[675]*y[IDX_HDII] -
        k[865]*y[IDX_HeII] - k[866]*y[IDX_HeII] - k[1053]*y[IDX_CI] -
        k[1199]*y[IDX_NI] - k[1229]*y[IDX_OI] - k[1357]*y[IDX_oH3II] -
        k[1358]*y[IDX_pH3II] - k[1359]*y[IDX_pH3II] - k[1360]*y[IDX_oD3II] -
        k[1361]*y[IDX_mD3II] - k[1362]*y[IDX_oH2DII] - k[1363]*y[IDX_pH2DII] -
        k[1364]*y[IDX_pH2DII] - k[1365]*y[IDX_oH2DII] - k[1366]*y[IDX_pH2DII] -
        k[1367]*y[IDX_oD2HII] - k[1368]*y[IDX_pD2HII] - k[1369]*y[IDX_pD2HII] -
        k[1370]*y[IDX_oD2HII] - k[1371]*y[IDX_pD2HII] - k[1655]*y[IDX_OII] -
        k[1727]*y[IDX_CHII] - k[1728]*y[IDX_CDII] - k[1729]*y[IDX_CHII] -
        k[1730]*y[IDX_CDII] - k[1834]*y[IDX_NHII] - k[1835]*y[IDX_NDII] -
        k[1933]*y[IDX_OHII] - k[1934]*y[IDX_ODII] - k[2014]*y[IDX_O2HII] -
        k[2015]*y[IDX_O2DII] - k[2066]*y[IDX_OM] - k[2067]*y[IDX_OHM] -
        k[2068]*y[IDX_ODM] - k[2224]*y[IDX_NII] - k[2274]*y[IDX_N2II] -
        k[2355]*y[IDX_pH2II] - k[2356]*y[IDX_oH2II] - k[2357]*y[IDX_pD2II] -
        k[2358]*y[IDX_oD2II] - k[2359]*y[IDX_HDII] - k[2703]*y[IDX_HM] -
        k[2704]*y[IDX_DM] - k[2965]*y[IDX_pD3II] - k[2966]*y[IDX_pD3II];
    data[1262] = 0.0 + k[2069]*y[IDX_NHI] + k[2070]*y[IDX_NDI] +
        k[2071]*y[IDX_OHI] + k[2072]*y[IDX_ODI] + k[2073]*y[IDX_C2HI] +
        k[2074]*y[IDX_C2DI] + k[2075]*y[IDX_CH2I] + k[2076]*y[IDX_CD2I] +
        k[2077]*y[IDX_CHDI] + k[2078]*y[IDX_NH2I] + k[2079]*y[IDX_ND2I] +
        k[2080]*y[IDX_NHDI] + k[2263]*y[IDX_CI] + k[2264]*y[IDX_HI] +
        k[2265]*y[IDX_DI] + k[2266]*y[IDX_OI] + k[2267]*y[IDX_C2I] +
        k[2268]*y[IDX_CHI] + k[2269]*y[IDX_CDI] + k[2270]*y[IDX_COI] +
        k[2316]*y[IDX_NOI] + k[2317]*y[IDX_O2I] + k[2318]*y[IDX_CO2I] +
        k[2319]*y[IDX_HCNI] + k[2320]*y[IDX_DCNI] + k[2321]*y[IDX_HCOI] +
        k[2322]*y[IDX_DCOI];
    data[1263] = 0.0 + k[348] + k[3019]*y[IDX_H3OII] + k[3346]*y[IDX_H2DOII]
        + k[3347]*y[IDX_H2DOII] + k[3348]*y[IDX_HD2OII] + k[3349]*y[IDX_HD2OII]
        + k[3350]*y[IDX_D3OII];
    data[1264] = 0.0 + k[2785]*y[IDX_eM];
    data[1265] = 0.0 + k[1006]*y[IDX_HCNII] + k[1007]*y[IDX_DCNII] +
        k[2270]*y[IDX_CNII];
    data[1266] = 0.0 + k[2318]*y[IDX_CNII];
    data[1267] = 0.0 + k[1094]*y[IDX_HCNI] + k[1095]*y[IDX_DCNI] +
        k[2265]*y[IDX_CNII];
    data[1268] = 0.0 - k[2704]*y[IDX_CNI];
    data[1269] = 0.0 + k[3116]*y[IDX_HCNII] + k[3117]*y[IDX_DCNII] +
        k[3271]*y[IDX_HNCII] + k[3272]*y[IDX_DNCII];
    data[1270] = 0.0 + k[3350]*y[IDX_CNM];
    data[1271] = 0.0 + k[258] + k[401] + k[1093]*y[IDX_HI] +
        k[1095]*y[IDX_DI] + k[2320]*y[IDX_CNII];
    data[1272] = 0.0 + k[999]*y[IDX_CI] + k[1001]*y[IDX_C2I] +
        k[1003]*y[IDX_CHI] + k[1005]*y[IDX_CDI] + k[1007]*y[IDX_COI] +
        k[1009]*y[IDX_NHI] + k[1011]*y[IDX_NDI] + k[1013]*y[IDX_OHI] +
        k[1015]*y[IDX_ODI] + k[2821]*y[IDX_eM] + k[3113]*y[IDX_H2OI] +
        k[3115]*y[IDX_HDOI] + k[3117]*y[IDX_D2OI];
    data[1273] = 0.0 - k[550]*y[IDX_CNI] + k[2322]*y[IDX_CNII];
    data[1274] = 0.0 + k[264] + k[407];
    data[1275] = 0.0 + k[1553]*y[IDX_CI] + k[1555]*y[IDX_C2I] +
        k[1557]*y[IDX_CHI] + k[1559]*y[IDX_CDI] + k[1561]*y[IDX_NHI] +
        k[1563]*y[IDX_NDI] + k[1567]*y[IDX_OHI] + k[1569]*y[IDX_ODI] +
        k[2825]*y[IDX_eM] + k[3268]*y[IDX_H2OI] + k[3270]*y[IDX_HDOI] +
        k[3272]*y[IDX_D2OI];
    data[1276] = 0.0 - k[556]*y[IDX_CNI];
    data[1277] = 0.0 + k[1092]*y[IDX_HCNI] + k[1093]*y[IDX_DCNI] +
        k[2264]*y[IDX_CNII];
    data[1278] = 0.0 - k[2703]*y[IDX_CNI];
    data[1279] = 0.0 + k[3346]*y[IDX_CNM] + k[3347]*y[IDX_CNM];
    data[1280] = 0.0 + k[2995]*y[IDX_HCNII] + k[3007]*y[IDX_HNCII] +
        k[3113]*y[IDX_DCNII] + k[3268]*y[IDX_DNCII];
    data[1281] = 0.0 + k[3019]*y[IDX_CNM];
    data[1282] = 0.0 + k[257] + k[400] + k[1092]*y[IDX_HI] +
        k[1094]*y[IDX_DI] + k[2319]*y[IDX_CNII];
    data[1283] = 0.0 + k[998]*y[IDX_CI] + k[1000]*y[IDX_C2I] +
        k[1002]*y[IDX_CHI] + k[1004]*y[IDX_CDI] + k[1006]*y[IDX_COI] +
        k[1008]*y[IDX_NHI] + k[1010]*y[IDX_NDI] + k[1012]*y[IDX_OHI] +
        k[1014]*y[IDX_ODI] + k[2820]*y[IDX_eM] + k[2995]*y[IDX_H2OI] +
        k[3114]*y[IDX_HDOI] + k[3116]*y[IDX_D2OI];
    data[1284] = 0.0 - k[549]*y[IDX_CNI] + k[2321]*y[IDX_CNII];
    data[1285] = 0.0 - k[674]*y[IDX_CNI] - k[675]*y[IDX_CNI] -
        k[2359]*y[IDX_CNI];
    data[1286] = 0.0 + k[3348]*y[IDX_CNM] + k[3349]*y[IDX_CNM];
    data[1287] = 0.0 + k[3114]*y[IDX_HCNII] + k[3115]*y[IDX_DCNII] +
        k[3269]*y[IDX_HNCII] + k[3270]*y[IDX_DNCII];
    data[1288] = 0.0 + k[263] + k[406];
    data[1289] = 0.0 + k[1552]*y[IDX_CI] + k[1554]*y[IDX_C2I] +
        k[1556]*y[IDX_CHI] + k[1558]*y[IDX_CDI] + k[1560]*y[IDX_NHI] +
        k[1562]*y[IDX_NDI] + k[1566]*y[IDX_OHI] + k[1568]*y[IDX_ODI] +
        k[2824]*y[IDX_eM] + k[3007]*y[IDX_H2OI] + k[3269]*y[IDX_HDOI] +
        k[3271]*y[IDX_D2OI];
    data[1290] = 0.0 - k[555]*y[IDX_CNI];
    data[1291] = 0.0 - k[865]*y[IDX_CNI] - k[866]*y[IDX_CNI] +
        k[888]*y[IDX_C2NI] + k[946]*y[IDX_OCNI];
    data[1292] = 0.0 + k[950]*y[IDX_C2HII] + k[951]*y[IDX_C2DII] +
        k[1196]*y[IDX_C2I] + k[1197]*y[IDX_CHI] + k[1198]*y[IDX_CDI] -
        k[1199]*y[IDX_CNI] + k[1223]*y[IDX_C2NI] + k[1223]*y[IDX_C2NI] +
        k[1224]*y[IDX_CCOI] + k[1225]*y[IDX_C3I] + k[1689]*y[IDX_C2II] +
        k[2655]*y[IDX_CI] + k[2673]*y[IDX_CM];
    data[1293] = 0.0 - k[2224]*y[IDX_CNI];
    data[1294] = 0.0 + k[1054]*y[IDX_CI];
    data[1295] = 0.0 - k[2274]*y[IDX_CNI];
    data[1296] = 0.0 + k[1010]*y[IDX_HCNII] + k[1011]*y[IDX_DCNII] +
        k[1562]*y[IDX_HNCII] + k[1563]*y[IDX_DNCII] + k[2049]*y[IDX_CI] +
        k[2070]*y[IDX_CNII];
    data[1297] = 0.0 - k[1835]*y[IDX_CNI];
    data[1298] = 0.0 + k[2079]*y[IDX_CNII];
    data[1299] = 0.0 + k[1008]*y[IDX_HCNII] + k[1009]*y[IDX_DCNII] +
        k[1560]*y[IDX_HNCII] + k[1561]*y[IDX_DNCII] + k[2048]*y[IDX_CI] +
        k[2069]*y[IDX_CNII];
    data[1300] = 0.0 - k[1834]*y[IDX_CNI];
    data[1301] = 0.0 + k[2078]*y[IDX_CNII];
    data[1302] = 0.0 + k[2080]*y[IDX_CNII];
    data[1303] = 0.0 - k[547]*y[IDX_CNI] + k[2050]*y[IDX_CI] +
        k[2316]*y[IDX_CNII];
    data[1304] = 0.0 + k[535]*y[IDX_OCNI] - k[1229]*y[IDX_CNI] +
        k[1236]*y[IDX_C2NI] + k[2266]*y[IDX_CNII];
    data[1305] = 0.0 - k[1655]*y[IDX_CNI];
    data[1306] = 0.0 - k[2066]*y[IDX_CNI];
    data[1307] = 0.0 - k[548]*y[IDX_CNI] + k[2317]*y[IDX_CNII];
    data[1308] = 0.0 - k[2015]*y[IDX_CNI];
    data[1309] = 0.0 - k[2014]*y[IDX_CNI];
    data[1310] = 0.0 + k[280] + k[418] + k[455]*y[IDX_CII] +
        k[535]*y[IDX_OI] + k[946]*y[IDX_HeII] + k[2063]*y[IDX_CI];
    data[1311] = 0.0 - k[552]*y[IDX_CNI] - k[554]*y[IDX_CNI] +
        k[1014]*y[IDX_HCNII] + k[1015]*y[IDX_DCNII] + k[1568]*y[IDX_HNCII] +
        k[1569]*y[IDX_DNCII] + k[2072]*y[IDX_CNII];
    data[1312] = 0.0 - k[1934]*y[IDX_CNI];
    data[1313] = 0.0 - k[2068]*y[IDX_CNI];
    data[1314] = 0.0 - k[551]*y[IDX_CNI] - k[553]*y[IDX_CNI] +
        k[1012]*y[IDX_HCNII] + k[1013]*y[IDX_DCNII] + k[1566]*y[IDX_HNCII] +
        k[1567]*y[IDX_DNCII] + k[2071]*y[IDX_CNII];
    data[1315] = 0.0 - k[1933]*y[IDX_CNI];
    data[1316] = 0.0 - k[2067]*y[IDX_CNI];
    data[1317] = 0.0 + k[2771]*y[IDX_C2NII] + k[2785]*y[IDX_CNCII] +
        k[2820]*y[IDX_HCNII] + k[2821]*y[IDX_DCNII] + k[2824]*y[IDX_HNCII] +
        k[2825]*y[IDX_DNCII];
    data[1318] = 0.0 - k[1361]*y[IDX_CNI];
    data[1319] = 0.0 - k[673]*y[IDX_CNI] - k[2358]*y[IDX_CNI];
    data[1320] = 0.0 - k[1367]*y[IDX_CNI] - k[1370]*y[IDX_CNI];
    data[1321] = 0.0 - k[1360]*y[IDX_CNI];
    data[1322] = 0.0 - k[671]*y[IDX_CNI] - k[2356]*y[IDX_CNI];
    data[1323] = 0.0 - k[1362]*y[IDX_CNI] - k[1365]*y[IDX_CNI];
    data[1324] = 0.0 - k[1357]*y[IDX_CNI];
    data[1325] = 0.0 - k[672]*y[IDX_CNI] - k[2357]*y[IDX_CNI];
    data[1326] = 0.0 - k[1368]*y[IDX_CNI] - k[1369]*y[IDX_CNI] -
        k[1371]*y[IDX_CNI];
    data[1327] = 0.0 - k[2965]*y[IDX_CNI] - k[2966]*y[IDX_CNI];
    data[1328] = 0.0 - k[670]*y[IDX_CNI] - k[2355]*y[IDX_CNI];
    data[1329] = 0.0 - k[1363]*y[IDX_CNI] - k[1364]*y[IDX_CNI] -
        k[1366]*y[IDX_CNI];
    data[1330] = 0.0 - k[1358]*y[IDX_CNI] - k[1359]*y[IDX_CNI];
    data[1331] = 0.0 - k[2263]*y[IDX_CNII];
    data[1332] = 0.0 + k[421]*y[IDX_NHI] + k[422]*y[IDX_NDI];
    data[1333] = 0.0 - k[2267]*y[IDX_CNII];
    data[1334] = 0.0 - k[2074]*y[IDX_CNII];
    data[1335] = 0.0 - k[2073]*y[IDX_CNII];
    data[1336] = 0.0 + k[1635]*y[IDX_NII] - k[2269]*y[IDX_CNII];
    data[1337] = 0.0 + k[1718]*y[IDX_NI] + k[1748]*y[IDX_NHI] +
        k[1750]*y[IDX_NDI];
    data[1338] = 0.0 - k[2076]*y[IDX_CNII];
    data[1339] = 0.0 + k[1634]*y[IDX_NII] - k[2268]*y[IDX_CNII];
    data[1340] = 0.0 + k[1717]*y[IDX_NI] + k[1747]*y[IDX_NHI] +
        k[1749]*y[IDX_NDI];
    data[1341] = 0.0 - k[2075]*y[IDX_CNII];
    data[1342] = 0.0 - k[2077]*y[IDX_CNII];
    data[1343] = 0.0 + k[2224]*y[IDX_NII] + k[2274]*y[IDX_N2II] +
        k[2355]*y[IDX_pH2II] + k[2356]*y[IDX_oH2II] + k[2357]*y[IDX_pD2II] +
        k[2358]*y[IDX_oD2II] + k[2359]*y[IDX_HDII];
    data[1344] = 0.0 - k[468]*y[IDX_pH2I] - k[469]*y[IDX_oH2I] -
        k[470]*y[IDX_pD2I] - k[471]*y[IDX_oD2I] - k[472]*y[IDX_HDI] -
        k[473]*y[IDX_HDI] - k[474]*y[IDX_pH2I] - k[475]*y[IDX_oH2I] -
        k[476]*y[IDX_pD2I] - k[477]*y[IDX_oD2I] - k[478]*y[IDX_HDI] -
        k[479]*y[IDX_HDI] - k[594]*y[IDX_NOI] - k[595]*y[IDX_O2I] -
        k[596]*y[IDX_O2I] - k[597]*y[IDX_CO2I] - k[598]*y[IDX_CO2I] -
        k[599]*y[IDX_H2OI] - k[600]*y[IDX_D2OI] - k[601]*y[IDX_HDOI] -
        k[602]*y[IDX_HDOI] - k[603]*y[IDX_H2OI] - k[604]*y[IDX_D2OI] -
        k[605]*y[IDX_HDOI] - k[606]*y[IDX_HDOI] - k[607]*y[IDX_HCOI] -
        k[608]*y[IDX_DCOI] - k[2069]*y[IDX_NHI] - k[2070]*y[IDX_NDI] -
        k[2071]*y[IDX_OHI] - k[2072]*y[IDX_ODI] - k[2073]*y[IDX_C2HI] -
        k[2074]*y[IDX_C2DI] - k[2075]*y[IDX_CH2I] - k[2076]*y[IDX_CD2I] -
        k[2077]*y[IDX_CHDI] - k[2078]*y[IDX_NH2I] - k[2079]*y[IDX_ND2I] -
        k[2080]*y[IDX_NHDI] - k[2263]*y[IDX_CI] - k[2264]*y[IDX_HI] -
        k[2265]*y[IDX_DI] - k[2266]*y[IDX_OI] - k[2267]*y[IDX_C2I] -
        k[2268]*y[IDX_CHI] - k[2269]*y[IDX_CDI] - k[2270]*y[IDX_COI] -
        k[2316]*y[IDX_NOI] - k[2317]*y[IDX_O2I] - k[2318]*y[IDX_CO2I] -
        k[2319]*y[IDX_HCNI] - k[2320]*y[IDX_DCNI] - k[2321]*y[IDX_HCOI] -
        k[2322]*y[IDX_DCOI] - k[2743]*y[IDX_eM];
    data[1345] = 0.0 - k[2270]*y[IDX_CNII];
    data[1346] = 0.0 - k[597]*y[IDX_CNII] - k[598]*y[IDX_CNII] -
        k[2318]*y[IDX_CNII];
    data[1347] = 0.0 - k[2265]*y[IDX_CNII];
    data[1348] = 0.0 - k[600]*y[IDX_CNII] - k[604]*y[IDX_CNII];
    data[1349] = 0.0 + k[918]*y[IDX_HeII] - k[2320]*y[IDX_CNII];
    data[1350] = 0.0 - k[608]*y[IDX_CNII] - k[2322]*y[IDX_CNII];
    data[1351] = 0.0 + k[928]*y[IDX_HeII];
    data[1352] = 0.0 - k[2264]*y[IDX_CNII];
    data[1353] = 0.0 - k[599]*y[IDX_CNII] - k[603]*y[IDX_CNII];
    data[1354] = 0.0 + k[917]*y[IDX_HeII] - k[2319]*y[IDX_CNII];
    data[1355] = 0.0 - k[607]*y[IDX_CNII] - k[2321]*y[IDX_CNII];
    data[1356] = 0.0 - k[472]*y[IDX_CNII] - k[473]*y[IDX_CNII] -
        k[478]*y[IDX_CNII] - k[479]*y[IDX_CNII];
    data[1357] = 0.0 + k[2359]*y[IDX_CNI];
    data[1358] = 0.0 - k[601]*y[IDX_CNII] - k[602]*y[IDX_CNII] -
        k[605]*y[IDX_CNII] - k[606]*y[IDX_CNII];
    data[1359] = 0.0 + k[927]*y[IDX_HeII];
    data[1360] = 0.0 + k[917]*y[IDX_HCNI] + k[918]*y[IDX_DCNI] +
        k[927]*y[IDX_HNCI] + k[928]*y[IDX_DNCI] + k[947]*y[IDX_OCNI];
    data[1361] = 0.0 + k[1717]*y[IDX_CHII] + k[1718]*y[IDX_CDII];
    data[1362] = 0.0 + k[1634]*y[IDX_CHI] + k[1635]*y[IDX_CDI] +
        k[2224]*y[IDX_CNI];
    data[1363] = 0.0 + k[2274]*y[IDX_CNI];
    data[1364] = 0.0 + k[422]*y[IDX_CII] + k[1749]*y[IDX_CHII] +
        k[1750]*y[IDX_CDII] - k[2070]*y[IDX_CNII];
    data[1365] = 0.0 - k[2079]*y[IDX_CNII];
    data[1366] = 0.0 + k[421]*y[IDX_CII] + k[1747]*y[IDX_CHII] +
        k[1748]*y[IDX_CDII] - k[2069]*y[IDX_CNII];
    data[1367] = 0.0 - k[2078]*y[IDX_CNII];
    data[1368] = 0.0 - k[2080]*y[IDX_CNII];
    data[1369] = 0.0 - k[594]*y[IDX_CNII] - k[2316]*y[IDX_CNII];
    data[1370] = 0.0 - k[2266]*y[IDX_CNII];
    data[1371] = 0.0 - k[595]*y[IDX_CNII] - k[596]*y[IDX_CNII] -
        k[2317]*y[IDX_CNII];
    data[1372] = 0.0 + k[947]*y[IDX_HeII];
    data[1373] = 0.0 - k[2072]*y[IDX_CNII];
    data[1374] = 0.0 - k[2071]*y[IDX_CNII];
    data[1375] = 0.0 - k[2743]*y[IDX_CNII];
    data[1376] = 0.0 - k[471]*y[IDX_CNII] - k[477]*y[IDX_CNII];
    data[1377] = 0.0 + k[2358]*y[IDX_CNI];
    data[1378] = 0.0 - k[469]*y[IDX_CNII] - k[475]*y[IDX_CNII];
    data[1379] = 0.0 + k[2356]*y[IDX_CNI];
    data[1380] = 0.0 - k[470]*y[IDX_CNII] - k[476]*y[IDX_CNII];
    data[1381] = 0.0 + k[2357]*y[IDX_CNI];
    data[1382] = 0.0 - k[468]*y[IDX_CNII] - k[474]*y[IDX_CNII];
    data[1383] = 0.0 + k[2355]*y[IDX_CNI];
    data[1384] = 0.0 + k[480]*y[IDX_NOI];
    data[1385] = 0.0 + k[2066]*y[IDX_OM] + k[2067]*y[IDX_OHM] +
        k[2068]*y[IDX_ODM];
    data[1386] = 0.0 - k[348] - k[2728]*y[IDX_HI] - k[2729]*y[IDX_DI] -
        k[3019]*y[IDX_H3OII] - k[3346]*y[IDX_H2DOII] - k[3347]*y[IDX_H2DOII] -
        k[3348]*y[IDX_HD2OII] - k[3349]*y[IDX_HD2OII] - k[3350]*y[IDX_D3OII];
    data[1387] = 0.0 - k[2729]*y[IDX_CNM];
    data[1388] = 0.0 + k[494]*y[IDX_HCNI] + k[496]*y[IDX_DCNI];
    data[1389] = 0.0 - k[3350]*y[IDX_CNM];
    data[1390] = 0.0 + k[495]*y[IDX_HM] + k[496]*y[IDX_DM] +
        k[504]*y[IDX_OM] + k[507]*y[IDX_OHM] + k[508]*y[IDX_ODM];
    data[1391] = 0.0 - k[2728]*y[IDX_CNM];
    data[1392] = 0.0 + k[493]*y[IDX_HCNI] + k[495]*y[IDX_DCNI];
    data[1393] = 0.0 - k[3346]*y[IDX_CNM] - k[3347]*y[IDX_CNM];
    data[1394] = 0.0 - k[3019]*y[IDX_CNM];
    data[1395] = 0.0 + k[493]*y[IDX_HM] + k[494]*y[IDX_DM] +
        k[503]*y[IDX_OM] + k[505]*y[IDX_OHM] + k[506]*y[IDX_ODM];
    data[1396] = 0.0 - k[3348]*y[IDX_CNM] - k[3349]*y[IDX_CNM];
    data[1397] = 0.0 + k[480]*y[IDX_CM];
    data[1398] = 0.0 + k[503]*y[IDX_HCNI] + k[504]*y[IDX_DCNI] +
        k[2066]*y[IDX_CNI];
    data[1399] = 0.0 + k[506]*y[IDX_HCNI] + k[508]*y[IDX_DCNI] +
        k[2068]*y[IDX_CNI];
    data[1400] = 0.0 + k[505]*y[IDX_HCNI] + k[507]*y[IDX_DCNI] +
        k[2067]*y[IDX_CNI];
    data[1401] = 0.0 + k[445]*y[IDX_HCNI] + k[446]*y[IDX_DCNI];
    data[1402] = 0.0 + k[1730]*y[IDX_CNI];
    data[1403] = 0.0 + k[1729]*y[IDX_CNI];
    data[1404] = 0.0 + k[1729]*y[IDX_CHII] + k[1730]*y[IDX_CDII];
    data[1405] = 0.0 - k[980]*y[IDX_H2OI] - k[981]*y[IDX_D2OI] -
        k[982]*y[IDX_HDOI] - k[983]*y[IDX_HDOI] - k[2785]*y[IDX_eM] -
        k[2786]*y[IDX_eM];
    data[1406] = 0.0 - k[981]*y[IDX_CNCII];
    data[1407] = 0.0 + k[446]*y[IDX_CII];
    data[1408] = 0.0 - k[980]*y[IDX_CNCII];
    data[1409] = 0.0 + k[445]*y[IDX_CII];
    data[1410] = 0.0 - k[982]*y[IDX_CNCII] - k[983]*y[IDX_CNCII];
    data[1411] = 0.0 - k[2785]*y[IDX_CNCII] - k[2786]*y[IDX_CNCII];
    data[1412] = 0.0 + k[1016]*y[IDX_HCOII] + k[1017]*y[IDX_DCOII] -
        k[1055]*y[IDX_COI] + k[2051]*y[IDX_NOI] + k[2052]*y[IDX_O2I] +
        k[2053]*y[IDX_OHI] + k[2054]*y[IDX_ODI] + k[2055]*y[IDX_HCOI] +
        k[2056]*y[IDX_DCOI] + k[2062]*y[IDX_CCOI] + k[2063]*y[IDX_OCNI] +
        k[2081]*y[IDX_COII] + k[2656]*y[IDX_OI] + k[2715]*y[IDX_OM];
    data[1413] = 0.0 + k[423]*y[IDX_O2I] + k[434]*y[IDX_CO2I] +
        k[447]*y[IDX_HCOI] + k[448]*y[IDX_DCOI];
    data[1414] = 0.0 + k[481]*y[IDX_O2I] + k[482]*y[IDX_CO2I] +
        k[482]*y[IDX_CO2I] + k[2674]*y[IDX_OI];
    data[1415] = 0.0 + k[1018]*y[IDX_HCOII] + k[1019]*y[IDX_DCOII] +
        k[1226]*y[IDX_OI] + k[1912]*y[IDX_O2II] + k[2085]*y[IDX_COII];
    data[1416] = 0.0 + k[1704]*y[IDX_O2I] + k[1709]*y[IDX_HCOI] +
        k[1710]*y[IDX_DCOI];
    data[1417] = 0.0 + k[1033]*y[IDX_O2I] + k[1235]*y[IDX_OI] +
        k[2330]*y[IDX_COII];
    data[1418] = 0.0 + k[1032]*y[IDX_O2I] + k[1234]*y[IDX_OI] +
        k[2329]*y[IDX_COII];
    data[1419] = 0.0 + k[1236]*y[IDX_OI];
    data[1420] = 0.0 + k[2773]*y[IDX_eM];
    data[1421] = 0.0 + k[536]*y[IDX_OI];
    data[1422] = 0.0 + k[248] + k[891]*y[IDX_HeII] + k[1224]*y[IDX_NI] +
        k[1237]*y[IDX_OI] + k[1237]*y[IDX_OI] + k[2062]*y[IDX_CI];
    data[1423] = 0.0 + k[540]*y[IDX_O2I] + k[1022]*y[IDX_HCOII] +
        k[1023]*y[IDX_DCOII] + k[1228]*y[IDX_OI] + k[2324]*y[IDX_COII];
    data[1424] = 0.0 + k[1776]*y[IDX_CO2I] + k[1792]*y[IDX_HCOI] +
        k[1794]*y[IDX_DCOI];
    data[1425] = 0.0 + k[510]*y[IDX_OI] + k[513]*y[IDX_OI] +
        k[2332]*y[IDX_COII];
    data[1426] = 0.0 + k[539]*y[IDX_O2I] + k[1020]*y[IDX_HCOII] +
        k[1021]*y[IDX_DCOII] + k[1227]*y[IDX_OI] + k[2323]*y[IDX_COII];
    data[1427] = 0.0 + k[1775]*y[IDX_CO2I] + k[1791]*y[IDX_HCOI] +
        k[1793]*y[IDX_DCOI];
    data[1428] = 0.0 + k[509]*y[IDX_OI] + k[512]*y[IDX_OI] +
        k[2331]*y[IDX_COII];
    data[1429] = 0.0 + k[511]*y[IDX_OI] + k[514]*y[IDX_OI] +
        k[2333]*y[IDX_COII];
    data[1430] = 0.0 + k[547]*y[IDX_NOI] + k[549]*y[IDX_HCOI] +
        k[550]*y[IDX_DCOI] + k[1229]*y[IDX_OI];
    data[1431] = 0.0 + k[595]*y[IDX_O2I] + k[598]*y[IDX_CO2I] +
        k[607]*y[IDX_HCOI] + k[608]*y[IDX_DCOI] - k[2270]*y[IDX_COI];
    data[1432] = 0.0 - k[285] - k[286] - k[359] - k[557]*y[IDX_OHI] -
        k[558]*y[IDX_ODI] - k[559]*y[IDX_HNOI] - k[560]*y[IDX_DNOI] -
        k[676]*y[IDX_pH2II] - k[677]*y[IDX_oH2II] - k[678]*y[IDX_pD2II] -
        k[679]*y[IDX_oD2II] - k[680]*y[IDX_HDII] - k[681]*y[IDX_HDII] -
        k[867]*y[IDX_HeII] - k[1006]*y[IDX_HCNII] - k[1007]*y[IDX_DCNII] -
        k[1055]*y[IDX_CI] - k[1252]*y[IDX_H2OII] - k[1253]*y[IDX_D2OII] -
        k[1254]*y[IDX_HDOII] - k[1255]*y[IDX_HDOII] - k[1372]*y[IDX_oH3II] -
        k[1373]*y[IDX_pH3II] - k[1374]*y[IDX_pH3II] - k[1375]*y[IDX_oD3II] -
        k[1376]*y[IDX_mD3II] - k[1377]*y[IDX_oH2DII] - k[1378]*y[IDX_pH2DII] -
        k[1379]*y[IDX_pH2DII] - k[1380]*y[IDX_oH2DII] - k[1381]*y[IDX_pH2DII] -
        k[1382]*y[IDX_oD2HII] - k[1383]*y[IDX_pD2HII] - k[1384]*y[IDX_pD2HII] -
        k[1385]*y[IDX_oD2HII] - k[1386]*y[IDX_pD2HII] - k[1387]*y[IDX_oH3II] -
        k[1388]*y[IDX_pH3II] - k[1389]*y[IDX_pH3II] - k[1390]*y[IDX_oD3II] -
        k[1391]*y[IDX_mD3II] - k[1392]*y[IDX_oH2DII] - k[1393]*y[IDX_pH2DII] -
        k[1394]*y[IDX_pH2DII] - k[1395]*y[IDX_oH2DII] - k[1396]*y[IDX_pH2DII] -
        k[1397]*y[IDX_oD2HII] - k[1398]*y[IDX_pD2HII] - k[1399]*y[IDX_pD2HII] -
        k[1400]*y[IDX_oD2HII] - k[1401]*y[IDX_pD2HII] - k[1580]*y[IDX_HNOII] -
        k[1581]*y[IDX_DNOII] - k[1590]*y[IDX_HOCII] + k[1590]*y[IDX_HOCII] -
        k[1591]*y[IDX_DOCII] + k[1591]*y[IDX_DOCII] - k[1624]*y[IDX_N2HII] -
        k[1625]*y[IDX_N2DII] - k[1836]*y[IDX_NHII] - k[1837]*y[IDX_NDII] -
        k[1838]*y[IDX_NHII] - k[1839]*y[IDX_NDII] - k[1935]*y[IDX_OHII] -
        k[1936]*y[IDX_ODII] - k[2016]*y[IDX_O2HII] - k[2017]*y[IDX_O2DII] -
        k[2064]*y[IDX_NII] - k[2225]*y[IDX_NII] - k[2270]*y[IDX_CNII] -
        k[2275]*y[IDX_N2II] - k[2360]*y[IDX_pH2II] - k[2361]*y[IDX_oH2II] -
        k[2362]*y[IDX_pD2II] - k[2363]*y[IDX_oD2II] - k[2364]*y[IDX_HDII] -
        k[2705]*y[IDX_HM] - k[2706]*y[IDX_DM] - k[2722]*y[IDX_OM] -
        k[2912]*y[IDX_pD3II] - k[2967]*y[IDX_pD3II] - k[2968]*y[IDX_pD3II] -
        k[2969]*y[IDX_pD3II];
    data[1433] = 0.0 + k[2081]*y[IDX_CI] + k[2082]*y[IDX_HI] +
        k[2083]*y[IDX_DI] + k[2084]*y[IDX_OI] + k[2085]*y[IDX_C2I] +
        k[2086]*y[IDX_NOI] + k[2087]*y[IDX_O2I] + k[2088]*y[IDX_CO2I] +
        k[2089]*y[IDX_HCNI] + k[2090]*y[IDX_DCNI] + k[2091]*y[IDX_HCOI] +
        k[2092]*y[IDX_DCOI] + k[2323]*y[IDX_CHI] + k[2324]*y[IDX_CDI] +
        k[2325]*y[IDX_NHI] + k[2326]*y[IDX_NDI] + k[2327]*y[IDX_OHI] +
        k[2328]*y[IDX_ODI] + k[2329]*y[IDX_C2HI] + k[2330]*y[IDX_C2DI] +
        k[2331]*y[IDX_CH2I] + k[2332]*y[IDX_CD2I] + k[2333]*y[IDX_CHDI] +
        k[2334]*y[IDX_H2OI] + k[2335]*y[IDX_D2OI] + k[2336]*y[IDX_HDOI] +
        k[2337]*y[IDX_NH2I] + k[2338]*y[IDX_ND2I] + k[2339]*y[IDX_NHDI];
    data[1434] = 0.0 + k[252] + k[392] + k[434]*y[IDX_CII] +
        k[482]*y[IDX_CM] + k[482]*y[IDX_CM] + k[598]*y[IDX_CNII] +
        k[900]*y[IDX_HeII] + k[1124]*y[IDX_HI] + k[1125]*y[IDX_DI] +
        k[1669]*y[IDX_OII] + k[1775]*y[IDX_CHII] + k[1776]*y[IDX_CDII] +
        k[1888]*y[IDX_NHII] + k[1889]*y[IDX_NDII] + k[2088]*y[IDX_COII];
    data[1435] = 0.0 + k[986]*y[IDX_OI] + k[2787]*y[IDX_eM];
    data[1436] = 0.0 + k[1058]*y[IDX_HCOI] + k[1059]*y[IDX_DCOI] +
        k[1125]*y[IDX_CO2I] + k[2083]*y[IDX_COII];
    data[1437] = 0.0 + k[1549]*y[IDX_HCOI] + k[1551]*y[IDX_DCOI];
    data[1438] = 0.0 + k[859]*y[IDX_HCOII] + k[861]*y[IDX_DCOII] -
        k[2706]*y[IDX_COI];
    data[1439] = 0.0 + k[2335]*y[IDX_COII] + k[3121]*y[IDX_HCOII] +
        k[3122]*y[IDX_DCOII];
    data[1440] = 0.0 - k[1253]*y[IDX_COI] + k[3173]*y[IDX_HCOI] +
        k[3176]*y[IDX_DCOI];
    data[1441] = 0.0 + k[2090]*y[IDX_COII];
    data[1442] = 0.0 - k[1007]*y[IDX_COI];
    data[1443] = 0.0 + k[260] + k[403] + k[448]*y[IDX_CII] +
        k[518]*y[IDX_OI] + k[550]*y[IDX_CNI] + k[573]*y[IDX_OHI] +
        k[575]*y[IDX_ODI] + k[608]*y[IDX_CNII] + k[801]*y[IDX_pH2II] +
        k[802]*y[IDX_oH2II] + k[803]*y[IDX_oH2II] + k[804]*y[IDX_pD2II] +
        k[805]*y[IDX_oD2II] + k[806]*y[IDX_oD2II] + k[807]*y[IDX_HDII] +
        k[808]*y[IDX_HDII] + k[924]*y[IDX_HeII] + k[1057]*y[IDX_HI] +
        k[1059]*y[IDX_DI] + k[1550]*y[IDX_HII] + k[1551]*y[IDX_DII] +
        k[1651]*y[IDX_NII] + k[1677]*y[IDX_OII] + k[1710]*y[IDX_C2II] +
        k[1793]*y[IDX_CHII] + k[1794]*y[IDX_CDII] + k[1817]*y[IDX_N2II] +
        k[1920]*y[IDX_O2II] + k[1967]*y[IDX_OHII] + k[1968]*y[IDX_ODII] +
        k[2056]*y[IDX_CI] + k[2092]*y[IDX_COII] + k[2950]*y[IDX_oD2II] +
        k[2951]*y[IDX_pD2II] + k[3174]*y[IDX_H2OII] + k[3175]*y[IDX_HDOII] +
        k[3176]*y[IDX_D2OII];
    data[1444] = 0.0 + k[201]*y[IDX_GRAINM] + k[860]*y[IDX_HM] +
        k[861]*y[IDX_DM] + k[1017]*y[IDX_CI] + k[1019]*y[IDX_C2I] +
        k[1021]*y[IDX_CHI] + k[1023]*y[IDX_CDI] + k[1025]*y[IDX_NHI] +
        k[1027]*y[IDX_NDI] + k[1029]*y[IDX_OHI] + k[1031]*y[IDX_ODI] +
        k[2823]*y[IDX_eM] + k[3118]*y[IDX_H2OI] + k[3120]*y[IDX_HDOI] +
        k[3122]*y[IDX_D2OI];
    data[1445] = 0.0 - k[560]*y[IDX_COI];
    data[1446] = 0.0 - k[1581]*y[IDX_COI];
    data[1447] = 0.0 - k[1591]*y[IDX_COI] + k[1591]*y[IDX_COI] +
        k[1615]*y[IDX_N2I] + k[2829]*y[IDX_eM];
    data[1448] = 0.0 + k[200]*y[IDX_HCOII] + k[201]*y[IDX_DCOII];
    data[1449] = 0.0 + k[1056]*y[IDX_HCOI] + k[1057]*y[IDX_DCOI] +
        k[1124]*y[IDX_CO2I] + k[2082]*y[IDX_COII];
    data[1450] = 0.0 + k[1548]*y[IDX_HCOI] + k[1550]*y[IDX_DCOI];
    data[1451] = 0.0 + k[858]*y[IDX_HCOII] + k[860]*y[IDX_DCOII] -
        k[2705]*y[IDX_COI];
    data[1452] = 0.0 + k[2334]*y[IDX_COII] + k[2996]*y[IDX_HCOII] +
        k[3118]*y[IDX_DCOII];
    data[1453] = 0.0 - k[1252]*y[IDX_COI] + k[3002]*y[IDX_HCOI] +
        k[3174]*y[IDX_DCOI];
    data[1454] = 0.0 + k[2089]*y[IDX_COII];
    data[1455] = 0.0 - k[1006]*y[IDX_COI];
    data[1456] = 0.0 + k[259] + k[402] + k[447]*y[IDX_CII] +
        k[517]*y[IDX_OI] + k[549]*y[IDX_CNI] + k[572]*y[IDX_OHI] +
        k[574]*y[IDX_ODI] + k[607]*y[IDX_CNII] + k[795]*y[IDX_oH2II] +
        k[796]*y[IDX_pD2II] + k[797]*y[IDX_oD2II] + k[798]*y[IDX_oD2II] +
        k[799]*y[IDX_HDII] + k[800]*y[IDX_HDII] + k[923]*y[IDX_HeII] +
        k[1056]*y[IDX_HI] + k[1058]*y[IDX_DI] + k[1548]*y[IDX_HII] +
        k[1549]*y[IDX_DII] + k[1650]*y[IDX_NII] + k[1676]*y[IDX_OII] +
        k[1709]*y[IDX_C2II] + k[1791]*y[IDX_CHII] + k[1792]*y[IDX_CDII] +
        k[1816]*y[IDX_N2II] + k[1919]*y[IDX_O2II] + k[1965]*y[IDX_OHII] +
        k[1966]*y[IDX_ODII] + k[2055]*y[IDX_CI] + k[2091]*y[IDX_COII] +
        k[2921]*y[IDX_oH2II] + k[2922]*y[IDX_pH2II] + k[3002]*y[IDX_H2OII] +
        k[3172]*y[IDX_HDOII] + k[3173]*y[IDX_D2OII];
    data[1457] = 0.0 + k[200]*y[IDX_GRAINM] + k[858]*y[IDX_HM] +
        k[859]*y[IDX_DM] + k[1016]*y[IDX_CI] + k[1018]*y[IDX_C2I] +
        k[1020]*y[IDX_CHI] + k[1022]*y[IDX_CDI] + k[1024]*y[IDX_NHI] +
        k[1026]*y[IDX_NDI] + k[1028]*y[IDX_OHI] + k[1030]*y[IDX_ODI] +
        k[2822]*y[IDX_eM] + k[2996]*y[IDX_H2OI] + k[3119]*y[IDX_HDOI] +
        k[3121]*y[IDX_D2OI];
    data[1458] = 0.0 - k[680]*y[IDX_COI] - k[681]*y[IDX_COI] +
        k[799]*y[IDX_HCOI] + k[800]*y[IDX_HCOI] + k[807]*y[IDX_DCOI] +
        k[808]*y[IDX_DCOI] - k[2364]*y[IDX_COI];
    data[1459] = 0.0 + k[2336]*y[IDX_COII] + k[3119]*y[IDX_HCOII] +
        k[3120]*y[IDX_DCOII];
    data[1460] = 0.0 - k[1254]*y[IDX_COI] - k[1255]*y[IDX_COI] +
        k[3172]*y[IDX_HCOI] + k[3175]*y[IDX_DCOI];
    data[1461] = 0.0 - k[559]*y[IDX_COI];
    data[1462] = 0.0 - k[1580]*y[IDX_COI];
    data[1463] = 0.0 - k[1590]*y[IDX_COI] + k[1590]*y[IDX_COI] +
        k[1614]*y[IDX_N2I] + k[2828]*y[IDX_eM];
    data[1464] = 0.0 - k[867]*y[IDX_COI] + k[891]*y[IDX_CCOI] +
        k[900]*y[IDX_CO2I] + k[923]*y[IDX_HCOI] + k[924]*y[IDX_DCOI];
    data[1465] = 0.0 + k[1224]*y[IDX_CCOI];
    data[1466] = 0.0 + k[1650]*y[IDX_HCOI] + k[1651]*y[IDX_DCOI] -
        k[2064]*y[IDX_COI] - k[2225]*y[IDX_COI];
    data[1467] = 0.0 + k[1614]*y[IDX_HOCII] + k[1615]*y[IDX_DOCII];
    data[1468] = 0.0 + k[1816]*y[IDX_HCOI] + k[1817]*y[IDX_DCOI] -
        k[2275]*y[IDX_COI];
    data[1469] = 0.0 - k[1625]*y[IDX_COI];
    data[1470] = 0.0 - k[1624]*y[IDX_COI];
    data[1471] = 0.0 + k[2834]*y[IDX_eM];
    data[1472] = 0.0 + k[1026]*y[IDX_HCOII] + k[1027]*y[IDX_DCOII] +
        k[2326]*y[IDX_COII];
    data[1473] = 0.0 - k[1837]*y[IDX_COI] - k[1839]*y[IDX_COI] +
        k[1889]*y[IDX_CO2I];
    data[1474] = 0.0 + k[2338]*y[IDX_COII];
    data[1475] = 0.0 + k[1024]*y[IDX_HCOII] + k[1025]*y[IDX_DCOII] +
        k[2325]*y[IDX_COII];
    data[1476] = 0.0 - k[1836]*y[IDX_COI] - k[1838]*y[IDX_COI] +
        k[1888]*y[IDX_CO2I];
    data[1477] = 0.0 + k[2337]*y[IDX_COII];
    data[1478] = 0.0 + k[2339]*y[IDX_COII];
    data[1479] = 0.0 + k[547]*y[IDX_CNI] + k[2051]*y[IDX_CI] +
        k[2086]*y[IDX_COII];
    data[1480] = 0.0 + k[509]*y[IDX_CH2I] + k[510]*y[IDX_CD2I] +
        k[511]*y[IDX_CHDI] + k[512]*y[IDX_CH2I] + k[513]*y[IDX_CD2I] +
        k[514]*y[IDX_CHDI] + k[517]*y[IDX_HCOI] + k[518]*y[IDX_DCOI] +
        k[534]*y[IDX_OCNI] + k[536]*y[IDX_C3I] + k[986]*y[IDX_CO2II] +
        k[1226]*y[IDX_C2I] + k[1227]*y[IDX_CHI] + k[1228]*y[IDX_CDI] +
        k[1229]*y[IDX_CNI] + k[1234]*y[IDX_C2HI] + k[1235]*y[IDX_C2DI] +
        k[1236]*y[IDX_C2NI] + k[1237]*y[IDX_CCOI] + k[1237]*y[IDX_CCOI] +
        k[2084]*y[IDX_COII] + k[2656]*y[IDX_CI] + k[2674]*y[IDX_CM];
    data[1481] = 0.0 + k[1669]*y[IDX_CO2I] + k[1676]*y[IDX_HCOI] +
        k[1677]*y[IDX_DCOI];
    data[1482] = 0.0 + k[2715]*y[IDX_CI] - k[2722]*y[IDX_COI];
    data[1483] = 0.0 + k[423]*y[IDX_CII] + k[481]*y[IDX_CM] +
        k[539]*y[IDX_CHI] + k[540]*y[IDX_CDI] + k[595]*y[IDX_CNII] +
        k[1032]*y[IDX_C2HI] + k[1033]*y[IDX_C2DI] + k[1704]*y[IDX_C2II] +
        k[2052]*y[IDX_CI] + k[2087]*y[IDX_COII];
    data[1484] = 0.0 + k[1912]*y[IDX_C2I] + k[1919]*y[IDX_HCOI] +
        k[1920]*y[IDX_DCOI];
    data[1485] = 0.0 - k[2017]*y[IDX_COI];
    data[1486] = 0.0 - k[2016]*y[IDX_COI];
    data[1487] = 0.0 + k[534]*y[IDX_OI] + k[2063]*y[IDX_CI];
    data[1488] = 0.0 - k[558]*y[IDX_COI] + k[574]*y[IDX_HCOI] +
        k[575]*y[IDX_DCOI] + k[1030]*y[IDX_HCOII] + k[1031]*y[IDX_DCOII] +
        k[2054]*y[IDX_CI] + k[2328]*y[IDX_COII];
    data[1489] = 0.0 - k[1936]*y[IDX_COI] + k[1966]*y[IDX_HCOI] +
        k[1968]*y[IDX_DCOI];
    data[1490] = 0.0 - k[557]*y[IDX_COI] + k[572]*y[IDX_HCOI] +
        k[573]*y[IDX_DCOI] + k[1028]*y[IDX_HCOII] + k[1029]*y[IDX_DCOII] +
        k[2053]*y[IDX_CI] + k[2327]*y[IDX_COII];
    data[1491] = 0.0 - k[1935]*y[IDX_COI] + k[1965]*y[IDX_HCOI] +
        k[1967]*y[IDX_DCOI];
    data[1492] = 0.0 + k[2773]*y[IDX_C2OII] + k[2787]*y[IDX_CO2II] +
        k[2822]*y[IDX_HCOII] + k[2823]*y[IDX_DCOII] + k[2828]*y[IDX_HOCII] +
        k[2829]*y[IDX_DOCII] + k[2834]*y[IDX_NCOII];
    data[1493] = 0.0 - k[1376]*y[IDX_COI] - k[1391]*y[IDX_COI];
    data[1494] = 0.0 - k[679]*y[IDX_COI] + k[797]*y[IDX_HCOI] +
        k[798]*y[IDX_HCOI] + k[805]*y[IDX_DCOI] + k[806]*y[IDX_DCOI] -
        k[2363]*y[IDX_COI] + k[2950]*y[IDX_DCOI];
    data[1495] = 0.0 - k[1382]*y[IDX_COI] - k[1385]*y[IDX_COI] -
        k[1397]*y[IDX_COI] - k[1400]*y[IDX_COI];
    data[1496] = 0.0 - k[1375]*y[IDX_COI] - k[1390]*y[IDX_COI];
    data[1497] = 0.0 - k[677]*y[IDX_COI] + k[795]*y[IDX_HCOI] +
        k[802]*y[IDX_DCOI] + k[803]*y[IDX_DCOI] - k[2361]*y[IDX_COI] +
        k[2921]*y[IDX_HCOI];
    data[1498] = 0.0 - k[1377]*y[IDX_COI] - k[1380]*y[IDX_COI] -
        k[1392]*y[IDX_COI] - k[1395]*y[IDX_COI];
    data[1499] = 0.0 - k[1372]*y[IDX_COI] - k[1387]*y[IDX_COI];
    data[1500] = 0.0 - k[678]*y[IDX_COI] + k[796]*y[IDX_HCOI] +
        k[804]*y[IDX_DCOI] - k[2362]*y[IDX_COI] + k[2951]*y[IDX_DCOI];
    data[1501] = 0.0 - k[1383]*y[IDX_COI] - k[1384]*y[IDX_COI] -
        k[1386]*y[IDX_COI] - k[1398]*y[IDX_COI] - k[1399]*y[IDX_COI] -
        k[1401]*y[IDX_COI];
    data[1502] = 0.0 - k[2912]*y[IDX_COI] - k[2967]*y[IDX_COI] -
        k[2968]*y[IDX_COI] - k[2969]*y[IDX_COI];
    data[1503] = 0.0 - k[676]*y[IDX_COI] + k[801]*y[IDX_DCOI] -
        k[2360]*y[IDX_COI] + k[2922]*y[IDX_HCOI];
    data[1504] = 0.0 - k[1378]*y[IDX_COI] - k[1379]*y[IDX_COI] -
        k[1381]*y[IDX_COI] - k[1393]*y[IDX_COI] - k[1394]*y[IDX_COI] -
        k[1396]*y[IDX_COI];
    data[1505] = 0.0 - k[1373]*y[IDX_COI] - k[1374]*y[IDX_COI] -
        k[1388]*y[IDX_COI] - k[1389]*y[IDX_COI];
    data[1506] = 0.0 + k[1910]*y[IDX_O2II] - k[2081]*y[IDX_COII];
    data[1507] = 0.0 + k[424]*y[IDX_O2I] + k[425]*y[IDX_OHI] +
        k[426]*y[IDX_ODI] + k[434]*y[IDX_CO2I] + k[455]*y[IDX_OCNI] +
        k[2640]*y[IDX_OI];
    data[1508] = 0.0 + k[1652]*y[IDX_OII] + k[1912]*y[IDX_O2II] -
        k[2085]*y[IDX_COII];
    data[1509] = 0.0 + k[1690]*y[IDX_OI] + k[1704]*y[IDX_O2I];
    data[1510] = 0.0 - k[617]*y[IDX_COII] + k[1668]*y[IDX_OII] -
        k[2330]*y[IDX_COII];
    data[1511] = 0.0 - k[616]*y[IDX_COII] + k[1667]*y[IDX_OII] -
        k[2329]*y[IDX_COII];
    data[1512] = 0.0 - k[611]*y[IDX_COII] + k[1654]*y[IDX_OII] -
        k[2324]*y[IDX_COII];
    data[1513] = 0.0 + k[1720]*y[IDX_OI] + k[1754]*y[IDX_O2I] +
        k[1758]*y[IDX_OHI] + k[1760]*y[IDX_ODI];
    data[1514] = 0.0 - k[619]*y[IDX_COII] - k[2332]*y[IDX_COII];
    data[1515] = 0.0 - k[610]*y[IDX_COII] + k[1653]*y[IDX_OII] -
        k[2323]*y[IDX_COII];
    data[1516] = 0.0 + k[1719]*y[IDX_OI] + k[1753]*y[IDX_O2I] +
        k[1757]*y[IDX_OHI] + k[1759]*y[IDX_ODI];
    data[1517] = 0.0 - k[618]*y[IDX_COII] - k[2331]*y[IDX_COII];
    data[1518] = 0.0 - k[620]*y[IDX_COII] - k[621]*y[IDX_COII] -
        k[2333]*y[IDX_COII];
    data[1519] = 0.0 + k[2270]*y[IDX_COI];
    data[1520] = 0.0 + k[286] + k[2225]*y[IDX_NII] + k[2270]*y[IDX_CNII] +
        k[2275]*y[IDX_N2II] + k[2360]*y[IDX_pH2II] + k[2361]*y[IDX_oH2II] +
        k[2362]*y[IDX_pD2II] + k[2363]*y[IDX_oD2II] + k[2364]*y[IDX_HDII];
    data[1521] = 0.0 - k[456]*y[IDX_pH2I] - k[457]*y[IDX_oH2I] -
        k[458]*y[IDX_pD2I] - k[459]*y[IDX_oD2I] - k[460]*y[IDX_HDI] -
        k[461]*y[IDX_HDI] - k[462]*y[IDX_pH2I] - k[463]*y[IDX_oH2I] -
        k[464]*y[IDX_pD2I] - k[465]*y[IDX_oD2I] - k[466]*y[IDX_HDI] -
        k[467]*y[IDX_HDI] - k[609]*y[IDX_NI] - k[610]*y[IDX_CHI] -
        k[611]*y[IDX_CDI] - k[612]*y[IDX_NHI] - k[613]*y[IDX_NDI] -
        k[614]*y[IDX_OHI] - k[615]*y[IDX_ODI] - k[616]*y[IDX_C2HI] -
        k[617]*y[IDX_C2DI] - k[618]*y[IDX_CH2I] - k[619]*y[IDX_CD2I] -
        k[620]*y[IDX_CHDI] - k[621]*y[IDX_CHDI] - k[622]*y[IDX_H2OI] -
        k[623]*y[IDX_D2OI] - k[624]*y[IDX_HDOI] - k[625]*y[IDX_HDOI] -
        k[626]*y[IDX_NH2I] - k[627]*y[IDX_ND2I] - k[628]*y[IDX_NHDI] -
        k[629]*y[IDX_NHDI] - k[2081]*y[IDX_CI] - k[2082]*y[IDX_HI] -
        k[2083]*y[IDX_DI] - k[2084]*y[IDX_OI] - k[2085]*y[IDX_C2I] -
        k[2086]*y[IDX_NOI] - k[2087]*y[IDX_O2I] - k[2088]*y[IDX_CO2I] -
        k[2089]*y[IDX_HCNI] - k[2090]*y[IDX_DCNI] - k[2091]*y[IDX_HCOI] -
        k[2092]*y[IDX_DCOI] - k[2323]*y[IDX_CHI] - k[2324]*y[IDX_CDI] -
        k[2325]*y[IDX_NHI] - k[2326]*y[IDX_NDI] - k[2327]*y[IDX_OHI] -
        k[2328]*y[IDX_ODI] - k[2329]*y[IDX_C2HI] - k[2330]*y[IDX_C2DI] -
        k[2331]*y[IDX_CH2I] - k[2332]*y[IDX_CD2I] - k[2333]*y[IDX_CHDI] -
        k[2334]*y[IDX_H2OI] - k[2335]*y[IDX_D2OI] - k[2336]*y[IDX_HDOI] -
        k[2337]*y[IDX_NH2I] - k[2338]*y[IDX_ND2I] - k[2339]*y[IDX_NHDI] -
        k[2744]*y[IDX_eM];
    data[1522] = 0.0 + k[434]*y[IDX_CII] + k[790]*y[IDX_pH2II] +
        k[791]*y[IDX_oH2II] + k[792]*y[IDX_pD2II] + k[793]*y[IDX_oD2II] +
        k[794]*y[IDX_HDII] + k[901]*y[IDX_HeII] + k[1649]*y[IDX_NII] -
        k[2088]*y[IDX_COII];
    data[1523] = 0.0 - k[2083]*y[IDX_COII];
    data[1524] = 0.0 + k[1545]*y[IDX_HCOI] + k[1547]*y[IDX_DCOI];
    data[1525] = 0.0 - k[623]*y[IDX_COII] - k[2335]*y[IDX_COII];
    data[1526] = 0.0 + k[1671]*y[IDX_OII] - k[2090]*y[IDX_COII];
    data[1527] = 0.0 + k[922]*y[IDX_HeII] + k[1546]*y[IDX_HII] +
        k[1547]*y[IDX_DII] - k[2092]*y[IDX_COII];
    data[1528] = 0.0 - k[2082]*y[IDX_COII];
    data[1529] = 0.0 + k[1544]*y[IDX_HCOI] + k[1546]*y[IDX_DCOI];
    data[1530] = 0.0 - k[622]*y[IDX_COII] - k[2334]*y[IDX_COII];
    data[1531] = 0.0 + k[1670]*y[IDX_OII] - k[2089]*y[IDX_COII];
    data[1532] = 0.0 + k[921]*y[IDX_HeII] + k[1544]*y[IDX_HII] +
        k[1545]*y[IDX_DII] - k[2091]*y[IDX_COII];
    data[1533] = 0.0 - k[460]*y[IDX_COII] - k[461]*y[IDX_COII] -
        k[466]*y[IDX_COII] - k[467]*y[IDX_COII];
    data[1534] = 0.0 + k[794]*y[IDX_CO2I] + k[2364]*y[IDX_COI];
    data[1535] = 0.0 - k[624]*y[IDX_COII] - k[625]*y[IDX_COII] -
        k[2336]*y[IDX_COII];
    data[1536] = 0.0 + k[901]*y[IDX_CO2I] + k[921]*y[IDX_HCOI] +
        k[922]*y[IDX_DCOI];
    data[1537] = 0.0 - k[609]*y[IDX_COII];
    data[1538] = 0.0 + k[1649]*y[IDX_CO2I] + k[2225]*y[IDX_COI];
    data[1539] = 0.0 + k[2275]*y[IDX_COI];
    data[1540] = 0.0 - k[613]*y[IDX_COII] - k[2326]*y[IDX_COII];
    data[1541] = 0.0 - k[627]*y[IDX_COII] - k[2338]*y[IDX_COII];
    data[1542] = 0.0 - k[612]*y[IDX_COII] - k[2325]*y[IDX_COII];
    data[1543] = 0.0 - k[626]*y[IDX_COII] - k[2337]*y[IDX_COII];
    data[1544] = 0.0 - k[628]*y[IDX_COII] - k[629]*y[IDX_COII] -
        k[2339]*y[IDX_COII];
    data[1545] = 0.0 - k[2086]*y[IDX_COII];
    data[1546] = 0.0 + k[1690]*y[IDX_C2II] + k[1719]*y[IDX_CHII] +
        k[1720]*y[IDX_CDII] - k[2084]*y[IDX_COII] + k[2640]*y[IDX_CII];
    data[1547] = 0.0 + k[1652]*y[IDX_C2I] + k[1653]*y[IDX_CHI] +
        k[1654]*y[IDX_CDI] + k[1667]*y[IDX_C2HI] + k[1668]*y[IDX_C2DI] +
        k[1670]*y[IDX_HCNI] + k[1671]*y[IDX_DCNI];
    data[1548] = 0.0 + k[424]*y[IDX_CII] + k[1704]*y[IDX_C2II] +
        k[1753]*y[IDX_CHII] + k[1754]*y[IDX_CDII] - k[2087]*y[IDX_COII];
    data[1549] = 0.0 + k[1910]*y[IDX_CI] + k[1912]*y[IDX_C2I];
    data[1550] = 0.0 + k[455]*y[IDX_CII];
    data[1551] = 0.0 + k[426]*y[IDX_CII] - k[615]*y[IDX_COII] +
        k[1759]*y[IDX_CHII] + k[1760]*y[IDX_CDII] - k[2328]*y[IDX_COII];
    data[1552] = 0.0 + k[425]*y[IDX_CII] - k[614]*y[IDX_COII] +
        k[1757]*y[IDX_CHII] + k[1758]*y[IDX_CDII] - k[2327]*y[IDX_COII];
    data[1553] = 0.0 - k[2744]*y[IDX_COII];
    data[1554] = 0.0 - k[459]*y[IDX_COII] - k[465]*y[IDX_COII];
    data[1555] = 0.0 + k[793]*y[IDX_CO2I] + k[2363]*y[IDX_COI];
    data[1556] = 0.0 - k[457]*y[IDX_COII] - k[463]*y[IDX_COII];
    data[1557] = 0.0 + k[791]*y[IDX_CO2I] + k[2361]*y[IDX_COI];
    data[1558] = 0.0 - k[458]*y[IDX_COII] - k[464]*y[IDX_COII];
    data[1559] = 0.0 + k[792]*y[IDX_CO2I] + k[2362]*y[IDX_COI];
    data[1560] = 0.0 - k[456]*y[IDX_COII] - k[462]*y[IDX_COII];
    data[1561] = 0.0 + k[790]*y[IDX_CO2I] + k[2360]*y[IDX_COI];
    data[1562] = 0.0 - k[434]*y[IDX_CO2I];
    data[1563] = 0.0 - k[482]*y[IDX_CO2I] + k[2684]*y[IDX_O2I];
    data[1564] = 0.0 - k[1776]*y[IDX_CO2I];
    data[1565] = 0.0 - k[1775]*y[IDX_CO2I];
    data[1566] = 0.0 - k[597]*y[IDX_CO2I] - k[598]*y[IDX_CO2I] -
        k[2318]*y[IDX_CO2I];
    data[1567] = 0.0 + k[557]*y[IDX_OHI] + k[558]*y[IDX_ODI] +
        k[559]*y[IDX_HNOI] + k[560]*y[IDX_DNOI] + k[2722]*y[IDX_OM];
    data[1568] = 0.0 - k[2088]*y[IDX_CO2I];
    data[1569] = 0.0 - k[252] - k[392] - k[434]*y[IDX_CII] -
        k[482]*y[IDX_CM] - k[597]*y[IDX_CNII] - k[598]*y[IDX_CNII] -
        k[790]*y[IDX_pH2II] - k[791]*y[IDX_oH2II] - k[792]*y[IDX_pD2II] -
        k[793]*y[IDX_oD2II] - k[794]*y[IDX_HDII] - k[899]*y[IDX_HeII] -
        k[900]*y[IDX_HeII] - k[901]*y[IDX_HeII] - k[902]*y[IDX_HeII] -
        k[1124]*y[IDX_HI] - k[1125]*y[IDX_DI] - k[1542]*y[IDX_HII] -
        k[1543]*y[IDX_DII] - k[1649]*y[IDX_NII] - k[1669]*y[IDX_OII] -
        k[1775]*y[IDX_CHII] - k[1776]*y[IDX_CDII] - k[1886]*y[IDX_NHII] -
        k[1887]*y[IDX_NDII] - k[1888]*y[IDX_NHII] - k[1889]*y[IDX_NDII] -
        k[2088]*y[IDX_COII] - k[2287]*y[IDX_N2II] - k[2318]*y[IDX_CNII] -
        k[2478]*y[IDX_HeII] - k[2521]*y[IDX_NII] - k[2538]*y[IDX_pH2II] -
        k[2539]*y[IDX_oH2II] - k[2540]*y[IDX_pD2II] - k[2541]*y[IDX_oD2II] -
        k[2542]*y[IDX_HDII];
    data[1570] = 0.0 + k[2167]*y[IDX_NOI] + k[2168]*y[IDX_O2I] +
        k[2374]*y[IDX_HI] + k[2375]*y[IDX_DI] + k[2376]*y[IDX_OI] +
        k[2377]*y[IDX_H2OI] + k[2378]*y[IDX_D2OI] + k[2379]*y[IDX_HDOI];
    data[1571] = 0.0 - k[1125]*y[IDX_CO2I] + k[2375]*y[IDX_CO2II];
    data[1572] = 0.0 - k[1543]*y[IDX_CO2I];
    data[1573] = 0.0 + k[2378]*y[IDX_CO2II];
    data[1574] = 0.0 + k[516]*y[IDX_OI];
    data[1575] = 0.0 + k[560]*y[IDX_COI];
    data[1576] = 0.0 - k[1124]*y[IDX_CO2I] + k[2374]*y[IDX_CO2II];
    data[1577] = 0.0 - k[1542]*y[IDX_CO2I];
    data[1578] = 0.0 + k[2377]*y[IDX_CO2II];
    data[1579] = 0.0 + k[515]*y[IDX_OI];
    data[1580] = 0.0 - k[794]*y[IDX_CO2I] - k[2542]*y[IDX_CO2I];
    data[1581] = 0.0 + k[2379]*y[IDX_CO2II];
    data[1582] = 0.0 + k[559]*y[IDX_COI];
    data[1583] = 0.0 - k[899]*y[IDX_CO2I] - k[900]*y[IDX_CO2I] -
        k[901]*y[IDX_CO2I] - k[902]*y[IDX_CO2I] - k[2478]*y[IDX_CO2I];
    data[1584] = 0.0 - k[1649]*y[IDX_CO2I] - k[2521]*y[IDX_CO2I];
    data[1585] = 0.0 - k[2287]*y[IDX_CO2I];
    data[1586] = 0.0 - k[1887]*y[IDX_CO2I] - k[1889]*y[IDX_CO2I];
    data[1587] = 0.0 - k[1886]*y[IDX_CO2I] - k[1888]*y[IDX_CO2I];
    data[1588] = 0.0 + k[2167]*y[IDX_CO2II];
    data[1589] = 0.0 + k[515]*y[IDX_HCOI] + k[516]*y[IDX_DCOI] +
        k[2376]*y[IDX_CO2II];
    data[1590] = 0.0 - k[1669]*y[IDX_CO2I];
    data[1591] = 0.0 + k[2722]*y[IDX_COI];
    data[1592] = 0.0 + k[2168]*y[IDX_CO2II] + k[2684]*y[IDX_CM];
    data[1593] = 0.0 + k[558]*y[IDX_COI];
    data[1594] = 0.0 + k[557]*y[IDX_COI];
    data[1595] = 0.0 - k[793]*y[IDX_CO2I] - k[2541]*y[IDX_CO2I];
    data[1596] = 0.0 - k[791]*y[IDX_CO2I] - k[2539]*y[IDX_CO2I];
    data[1597] = 0.0 - k[792]*y[IDX_CO2I] - k[2540]*y[IDX_CO2I];
    data[1598] = 0.0 - k[790]*y[IDX_CO2I] - k[2538]*y[IDX_CO2I];
    data[1599] = 0.0 + k[2318]*y[IDX_CO2I];
    data[1600] = 0.0 + k[2088]*y[IDX_CO2I];
    data[1601] = 0.0 + k[2088]*y[IDX_COII] + k[2287]*y[IDX_N2II] +
        k[2318]*y[IDX_CNII] + k[2478]*y[IDX_HeII] + k[2521]*y[IDX_NII] +
        k[2538]*y[IDX_pH2II] + k[2539]*y[IDX_oH2II] + k[2540]*y[IDX_pD2II] +
        k[2541]*y[IDX_oD2II] + k[2542]*y[IDX_HDII];
    data[1602] = 0.0 - k[984]*y[IDX_HI] - k[985]*y[IDX_DI] -
        k[986]*y[IDX_OI] - k[2167]*y[IDX_NOI] - k[2168]*y[IDX_O2I] -
        k[2374]*y[IDX_HI] - k[2375]*y[IDX_DI] - k[2376]*y[IDX_OI] -
        k[2377]*y[IDX_H2OI] - k[2378]*y[IDX_D2OI] - k[2379]*y[IDX_HDOI] -
        k[2787]*y[IDX_eM];
    data[1603] = 0.0 - k[985]*y[IDX_CO2II] - k[2375]*y[IDX_CO2II];
    data[1604] = 0.0 - k[2378]*y[IDX_CO2II];
    data[1605] = 0.0 - k[984]*y[IDX_CO2II] - k[2374]*y[IDX_CO2II];
    data[1606] = 0.0 - k[2377]*y[IDX_CO2II];
    data[1607] = 0.0 + k[2542]*y[IDX_CO2I];
    data[1608] = 0.0 - k[2379]*y[IDX_CO2II];
    data[1609] = 0.0 + k[2478]*y[IDX_CO2I];
    data[1610] = 0.0 + k[2521]*y[IDX_CO2I];
    data[1611] = 0.0 + k[2287]*y[IDX_CO2I];
    data[1612] = 0.0 - k[2167]*y[IDX_CO2II];
    data[1613] = 0.0 - k[986]*y[IDX_CO2II] - k[2376]*y[IDX_CO2II];
    data[1614] = 0.0 - k[2168]*y[IDX_CO2II];
    data[1615] = 0.0 - k[2787]*y[IDX_CO2II];
    data[1616] = 0.0 + k[2541]*y[IDX_CO2I];
    data[1617] = 0.0 + k[2539]*y[IDX_CO2I];
    data[1618] = 0.0 + k[2540]*y[IDX_CO2I];
    data[1619] = 0.0 + k[2538]*y[IDX_CO2I];
    data[1620] = 0.0 + k[632]*y[IDX_pD2II] + k[633]*y[IDX_oD2II] +
        k[635]*y[IDX_HDII] + k[949]*y[IDX_C2DII] + k[965]*y[IDX_CD2II] +
        k[967]*y[IDX_CHDII] + k[1038]*y[IDX_CD2I] + k[1040]*y[IDX_CHDI] +
        k[1046]*y[IDX_ND2I] + k[1048]*y[IDX_NHDI] + k[1050]*y[IDX_ND2I] +
        k[1052]*y[IDX_NHDI] + k[1132]*y[IDX_pD2I] + k[1133]*y[IDX_oD2I] +
        k[1135]*y[IDX_HDI] + k[1712]*y[IDX_CDII] + k[2047]*y[IDX_CDI] +
        k[2049]*y[IDX_NDI] + k[2054]*y[IDX_ODI] + k[2058]*y[IDX_DCOI] +
        k[2061]*y[IDX_C2DI] - k[2654]*y[IDX_DI];
    data[1621] = 0.0 + k[420]*y[IDX_CDI] + k[422]*y[IDX_NDI] +
        k[426]*y[IDX_ODI] + k[428]*y[IDX_C2DI] + k[431]*y[IDX_CD2I] +
        k[433]*y[IDX_CHDI] + k[436]*y[IDX_D2OI] + k[438]*y[IDX_HDOI] +
        k[440]*y[IDX_D2OI] + k[442]*y[IDX_HDOI] + k[444]*y[IDX_DCNI] +
        k[446]*y[IDX_DCNI] + k[450]*y[IDX_DNCI] + k[452]*y[IDX_ND2I] +
        k[454]*y[IDX_NHDI] + k[2134]*y[IDX_DM] - k[2639]*y[IDX_DI];
    data[1622] = 0.0 + k[2136]*y[IDX_DII] - k[2672]*y[IDX_DI] +
        k[3337]*y[IDX_H2DOII] + k[3339]*y[IDX_HD2OII] + k[3340]*y[IDX_D3OII];
    data[1623] = 0.0 + k[546]*y[IDX_CDI] + k[650]*y[IDX_pD2II] +
        k[651]*y[IDX_oD2II] + k[653]*y[IDX_HDII] + k[1722]*y[IDX_CDII] +
        k[1827]*y[IDX_NDII] + k[2188]*y[IDX_DII];
    data[1624] = 0.0 + k[1693]*y[IDX_CDI] + k[1696]*y[IDX_pD2I] +
        k[1697]*y[IDX_oD2I] + k[1699]*y[IDX_HDI] + k[1703]*y[IDX_NDI];
    data[1625] = 0.0 + k[244] + k[379] + k[428]*y[IDX_CII] +
        k[885]*y[IDX_HeII] + k[1222]*y[IDX_NI] + k[2061]*y[IDX_CI] +
        k[2526]*y[IDX_DII];
    data[1626] = 0.0 + k[299] + k[949]*y[IDX_CI] + k[953]*y[IDX_NI] +
        k[2766]*y[IDX_eM] + k[2770]*y[IDX_eM];
    data[1627] = 0.0 + k[2524]*y[IDX_DII];
    data[1628] = 0.0 + k[2206]*y[IDX_DII];
    data[1629] = 0.0 + k[2208]*y[IDX_DII];
    data[1630] = 0.0 + k[1531]*y[IDX_DII];
    data[1631] = 0.0 + k[283] + k[355] + k[420]*y[IDX_CII] +
        k[546]*y[IDX_C2I] + k[664]*y[IDX_pH2II] + k[665]*y[IDX_oH2II] +
        k[666]*y[IDX_pD2II] + k[667]*y[IDX_oD2II] + k[669]*y[IDX_HDII] +
        k[864]*y[IDX_HeII] - k[1067]*y[IDX_DI] + k[1146]*y[IDX_pH2I] +
        k[1147]*y[IDX_oH2I] + k[1148]*y[IDX_pD2I] + k[1149]*y[IDX_oD2I] +
        k[1151]*y[IDX_HDI] + k[1198]*y[IDX_NI] + k[1228]*y[IDX_OI] +
        k[1635]*y[IDX_NII] + k[1654]*y[IDX_OII] + k[1693]*y[IDX_C2II] +
        k[2047]*y[IDX_CI] + k[2192]*y[IDX_DII];
    data[1632] = 0.0 + k[233] + k[289] + k[1712]*y[IDX_CI] -
        k[1716]*y[IDX_DI] + k[1718]*y[IDX_NI] + k[1720]*y[IDX_OI] +
        k[1722]*y[IDX_C2I] + k[1728]*y[IDX_CNI] + k[1730]*y[IDX_CNI] +
        k[1735]*y[IDX_pH2I] + k[1736]*y[IDX_oH2I] + k[1741]*y[IDX_pD2I] +
        k[1742]*y[IDX_oD2I] + k[1746]*y[IDX_HDI] + k[2742]*y[IDX_eM];
    data[1633] = 0.0 + k[386] + k[431]*y[IDX_CII] + k[510]*y[IDX_OI] +
        k[510]*y[IDX_OI] + k[896]*y[IDX_HeII] + k[1038]*y[IDX_CI] -
        k[1075]*y[IDX_DI] + k[1207]*y[IDX_NI] + k[1211]*y[IDX_NI] +
        k[2530]*y[IDX_DII];
    data[1634] = 0.0 + k[301] + k[965]*y[IDX_CI] + k[969]*y[IDX_NI] +
        k[973]*y[IDX_OI] + k[2779]*y[IDX_eM] + k[2783]*y[IDX_eM] +
        k[2783]*y[IDX_eM];
    data[1635] = 0.0 + k[658]*y[IDX_pD2II] + k[659]*y[IDX_oD2II] +
        k[661]*y[IDX_HDII] - k[1066]*y[IDX_DI] + k[1140]*y[IDX_pD2I] +
        k[1141]*y[IDX_oD2I] + k[1143]*y[IDX_HDI] + k[2190]*y[IDX_DII];
    data[1636] = 0.0 - k[1715]*y[IDX_DI] + k[1739]*y[IDX_pD2I] +
        k[1740]*y[IDX_oD2I] + k[1744]*y[IDX_HDI];
    data[1637] = 0.0 - k[1073]*y[IDX_DI] - k[1074]*y[IDX_DI] +
        k[2528]*y[IDX_DII];
    data[1638] = 0.0 + k[388] + k[433]*y[IDX_CII] + k[511]*y[IDX_OI] +
        k[898]*y[IDX_HeII] + k[1040]*y[IDX_CI] - k[1076]*y[IDX_DI] -
        k[1077]*y[IDX_DI] + k[1209]*y[IDX_NI] + k[1213]*y[IDX_NI] +
        k[2532]*y[IDX_DII];
    data[1639] = 0.0 + k[303] + k[967]*y[IDX_CI] + k[971]*y[IDX_NI] +
        k[975]*y[IDX_OI] + k[2781]*y[IDX_eM] + k[2784]*y[IDX_eM];
    data[1640] = 0.0 + k[552]*y[IDX_ODI] + k[672]*y[IDX_pD2II] +
        k[673]*y[IDX_oD2II] + k[675]*y[IDX_HDII] + k[1728]*y[IDX_CDII] +
        k[1730]*y[IDX_CDII];
    data[1641] = 0.0 + k[470]*y[IDX_pD2I] + k[471]*y[IDX_oD2I] +
        k[473]*y[IDX_HDI] + k[476]*y[IDX_pD2I] + k[477]*y[IDX_oD2I] +
        k[479]*y[IDX_HDI] - k[2265]*y[IDX_DI];
    data[1642] = 0.0 - k[2729]*y[IDX_DI] + k[3347]*y[IDX_H2DOII] +
        k[3349]*y[IDX_HD2OII] + k[3350]*y[IDX_D3OII];
    data[1643] = 0.0 + k[558]*y[IDX_ODI] + k[678]*y[IDX_pD2II] +
        k[679]*y[IDX_oD2II] + k[681]*y[IDX_HDII] + k[1839]*y[IDX_NDII];
    data[1644] = 0.0 + k[458]*y[IDX_pD2I] + k[459]*y[IDX_oD2I] +
        k[461]*y[IDX_HDI] + k[464]*y[IDX_pD2I] + k[465]*y[IDX_oD2I] +
        k[467]*y[IDX_HDI] - k[2083]*y[IDX_DI];
    data[1645] = 0.0 - k[1125]*y[IDX_DI];
    data[1646] = 0.0 - k[985]*y[IDX_DI] - k[2375]*y[IDX_DI];
    data[1647] = 0.0 - k[166]*y[IDX_HI] - k[167]*y[IDX_DI] -
        k[167]*y[IDX_DI] - k[167]*y[IDX_DI] - k[167]*y[IDX_DI] -
        k[168]*y[IDX_DI] - k[168]*y[IDX_DI] - k[168]*y[IDX_DI] -
        k[168]*y[IDX_DI] - k[204] - k[811]*y[IDX_HeHII] - k[812]*y[IDX_HeDII] -
        k[985]*y[IDX_CO2II] - k[1058]*y[IDX_HCOI] - k[1059]*y[IDX_DCOI] -
        k[1062]*y[IDX_HCOI] - k[1063]*y[IDX_DCOI] - k[1066]*y[IDX_CHI] -
        k[1067]*y[IDX_CDI] - k[1073]*y[IDX_CH2I] - k[1074]*y[IDX_CH2I] -
        k[1075]*y[IDX_CD2I] - k[1076]*y[IDX_CHDI] - k[1077]*y[IDX_CHDI] -
        k[1080]*y[IDX_OHI] - k[1081]*y[IDX_ODI] - k[1087]*y[IDX_H2OI] -
        k[1088]*y[IDX_H2OI] - k[1089]*y[IDX_D2OI] - k[1090]*y[IDX_HDOI] -
        k[1091]*y[IDX_HDOI] - k[1094]*y[IDX_HCNI] - k[1095]*y[IDX_DCNI] -
        k[1097]*y[IDX_NOI] - k[1099]*y[IDX_NOI] - k[1102]*y[IDX_HNOI] -
        k[1103]*y[IDX_DNOI] - k[1107]*y[IDX_HNOI] - k[1108]*y[IDX_HNOI] -
        k[1109]*y[IDX_DNOI] - k[1111]*y[IDX_O2I] - k[1114]*y[IDX_O2HI] -
        k[1115]*y[IDX_O2DI] - k[1118]*y[IDX_O2HI] - k[1119]*y[IDX_O2DI] -
        k[1122]*y[IDX_O2HI] - k[1123]*y[IDX_O2DI] - k[1125]*y[IDX_CO2I] -
        k[1127]*y[IDX_N2OI] - k[1129]*y[IDX_NO2I] - k[1715]*y[IDX_CHII] -
        k[1716]*y[IDX_CDII] - k[1996]*y[IDX_NO2II] - k[2083]*y[IDX_COII] -
        k[2098]*y[IDX_pH2II] - k[2099]*y[IDX_oH2II] - k[2100]*y[IDX_pD2II] -
        k[2101]*y[IDX_oD2II] - k[2102]*y[IDX_HDII] - k[2130]*y[IDX_N2II] -
        k[2161]*y[IDX_HeII] - k[2171]*y[IDX_HCNII] - k[2172]*y[IDX_DCNII] -
        k[2240]*y[IDX_OII] - k[2265]*y[IDX_CNII] - k[2375]*y[IDX_CO2II] -
        k[2639]*y[IDX_CII] - k[2649]*y[IDX_HII] - k[2650]*y[IDX_DII] -
        k[2651]*y[IDX_DII] - k[2654]*y[IDX_CI] - k[2663]*y[IDX_OI] -
        k[2668]*y[IDX_OHI] - k[2669]*y[IDX_ODI] - k[2672]*y[IDX_CM] -
        k[2691]*y[IDX_HM] - k[2692]*y[IDX_DM] - k[2717]*y[IDX_OM] -
        k[2729]*y[IDX_CNM] - k[2734]*y[IDX_OHM] - k[2735]*y[IDX_ODM] -
        k[2738]*y[IDX_eM] - k[2855]*y[IDX_pH2DII] - k[2856]*y[IDX_pH2DII] -
        k[2857]*y[IDX_oH2DII] - k[2858]*y[IDX_oH2DII] - k[2863]*y[IDX_HDII] -
        k[2864]*y[IDX_HDII] - k[2867]*y[IDX_oD2HII] - k[2868]*y[IDX_oD2HII] -
        k[2869]*y[IDX_pD2HII] - k[2881]*y[IDX_HII] - k[2890]*y[IDX_pH3II] -
        k[2891]*y[IDX_pH3II] - k[2892]*y[IDX_oH3II] - k[2896]*y[IDX_pH2II] -
        k[2899]*y[IDX_pH2II] - k[3383]*y[IDX_H3OII] - k[3384]*y[IDX_H3OII] -
        k[3385]*y[IDX_H3OII] - k[3386]*y[IDX_H2DOII] - k[3387]*y[IDX_H2DOII] -
        k[3388]*y[IDX_H2DOII] - k[3389]*y[IDX_H2DOII] - k[3390]*y[IDX_H2DOII] -
        k[3391]*y[IDX_HD2OII] - k[3392]*y[IDX_HD2OII] - k[3393]*y[IDX_HD2OII] -
        k[3394]*y[IDX_D3OII] - k[3395]*y[IDX_D3OII];
    data[1648] = 0.0 + k[1531]*y[IDX_CCOI] + k[2136]*y[IDX_CM] +
        k[2139]*y[IDX_HM] + k[2140]*y[IDX_DM] + k[2140]*y[IDX_DM] +
        k[2186]*y[IDX_OI] + k[2188]*y[IDX_C2I] + k[2190]*y[IDX_CHI] +
        k[2192]*y[IDX_CDI] + k[2194]*y[IDX_NHI] + k[2196]*y[IDX_NDI] +
        k[2198]*y[IDX_NOI] + k[2200]*y[IDX_O2I] + k[2202]*y[IDX_OHI] +
        k[2204]*y[IDX_ODI] + k[2206]*y[IDX_C2NI] + k[2208]*y[IDX_C3I] +
        k[2210]*y[IDX_H2OI] + k[2212]*y[IDX_D2OI] + k[2214]*y[IDX_HDOI] +
        k[2216]*y[IDX_HCNI] + k[2218]*y[IDX_DCNI] + k[2254]*y[IDX_NH2I] +
        k[2256]*y[IDX_ND2I] + k[2258]*y[IDX_NHDI] + k[2524]*y[IDX_C2HI] +
        k[2526]*y[IDX_C2DI] + k[2528]*y[IDX_CH2I] + k[2530]*y[IDX_CD2I] +
        k[2532]*y[IDX_CHDI] + k[2561]*y[IDX_HCOI] + k[2563]*y[IDX_DCOI] -
        k[2650]*y[IDX_DI] - k[2651]*y[IDX_DI] + k[2847]*y[IDX_eM] +
        k[2880]*y[IDX_HI] + k[2929]*y[IDX_GRAINM];
    data[1649] = 0.0 + k[346] + k[2134]*y[IDX_CII] + k[2138]*y[IDX_HII] +
        k[2140]*y[IDX_DII] + k[2140]*y[IDX_DII] + k[2143]*y[IDX_HeII] +
        k[2146]*y[IDX_NII] + k[2149]*y[IDX_OII] + k[2152]*y[IDX_pH2II] +
        k[2153]*y[IDX_oH2II] + k[2156]*y[IDX_pD2II] + k[2157]*y[IDX_oD2II] +
        k[2159]*y[IDX_HDII] - k[2692]*y[IDX_DI] + k[3058]*y[IDX_H3OII] +
        k[3059]*y[IDX_H3OII] + k[3068]*y[IDX_H2DOII] + k[3069]*y[IDX_H2DOII] +
        k[3070]*y[IDX_H2DOII] + k[3079]*y[IDX_HD2OII] + k[3080]*y[IDX_HD2OII] +
        k[3081]*y[IDX_HD2OII] + k[3087]*y[IDX_D3OII] + k[3088]*y[IDX_D3OII];
    data[1650] = 0.0 + k[254] + k[394] + k[436]*y[IDX_CII] +
        k[440]*y[IDX_CII] + k[908]*y[IDX_HeII] - k[1089]*y[IDX_DI] +
        k[2212]*y[IDX_DII] + k[3049]*y[IDX_pH2II] + k[3050]*y[IDX_oH2II] +
        k[3052]*y[IDX_HDII] + k[3053]*y[IDX_oD2II] + k[3054]*y[IDX_pD2II];
    data[1651] = 0.0 + k[992]*y[IDX_NI] + k[2792]*y[IDX_eM] +
        k[2796]*y[IDX_eM] + k[2796]*y[IDX_eM] + k[3129]*y[IDX_pH2I] +
        k[3130]*y[IDX_oH2I] + k[3136]*y[IDX_HDI] + k[3145]*y[IDX_oD2I] +
        k[3146]*y[IDX_pD2I];
    data[1652] = 0.0 + k[3084]*y[IDX_HM] + k[3085]*y[IDX_HM] +
        k[3086]*y[IDX_HM] + k[3087]*y[IDX_DM] + k[3088]*y[IDX_DM] +
        k[3340]*y[IDX_CM] + k[3345]*y[IDX_OM] + k[3350]*y[IDX_CNM] +
        k[3369]*y[IDX_OHM] + k[3370]*y[IDX_OHM] + k[3371]*y[IDX_ODM] -
        k[3394]*y[IDX_DI] - k[3395]*y[IDX_DI] + k[3408]*y[IDX_oH2I] +
        k[3409]*y[IDX_pH2I] + k[3410]*y[IDX_oH2I] + k[3410]*y[IDX_oH2I] +
        k[3411]*y[IDX_pH2I] + k[3411]*y[IDX_pH2I] + k[3420]*y[IDX_HDI] +
        k[3421]*y[IDX_HDI] + k[3421]*y[IDX_HDI] + k[3438]*y[IDX_oD2I] +
        k[3438]*y[IDX_oD2I] + k[3439]*y[IDX_pD2I] + k[3439]*y[IDX_pD2I] +
        k[3444]*y[IDX_eM] + k[3444]*y[IDX_eM] + k[3449]*y[IDX_eM] +
        k[3464]*y[IDX_eM] + k[3465]*y[IDX_eM];
    data[1653] = 0.0 + k[258] + k[401] + k[444]*y[IDX_CII] +
        k[446]*y[IDX_CII] + k[520]*y[IDX_OI] + k[912]*y[IDX_HeII] +
        k[918]*y[IDX_HeII] - k[1095]*y[IDX_DI] + k[2218]*y[IDX_DII];
    data[1654] = 0.0 - k[2172]*y[IDX_DI] + k[2821]*y[IDX_eM];
    data[1655] = 0.0 + k[260] + k[403] + k[516]*y[IDX_OI] +
        k[922]*y[IDX_HeII] - k[1059]*y[IDX_DI] - k[1063]*y[IDX_DI] +
        k[1215]*y[IDX_NI] + k[2058]*y[IDX_CI] + k[2563]*y[IDX_DII];
    data[1656] = 0.0 + k[201]*y[IDX_GRAINM] + k[2823]*y[IDX_eM];
    data[1657] = 0.0 + k[264] + k[407] + k[450]*y[IDX_CII] +
        k[926]*y[IDX_HeII] + k[928]*y[IDX_HeII];
    data[1658] = 0.0 + k[2825]*y[IDX_eM];
    data[1659] = 0.0 + k[409] + k[934]*y[IDX_HeII] - k[1103]*y[IDX_DI] -
        k[1109]*y[IDX_DI];
    data[1660] = 0.0 + k[1573]*y[IDX_OI] + k[2827]*y[IDX_eM];
    data[1661] = 0.0 + k[2829]*y[IDX_eM];
    data[1662] = 0.0 + k[171]*y[IDX_HDII] + k[173]*y[IDX_oD2II] +
        k[173]*y[IDX_oD2II] + k[175]*y[IDX_pD2II] + k[175]*y[IDX_pD2II] +
        k[183]*y[IDX_pH2DII] + k[185]*y[IDX_pH2DII] + k[186]*y[IDX_oH2DII] +
        k[187]*y[IDX_oH2DII] + k[189]*y[IDX_oD2HII] + k[189]*y[IDX_oD2HII] +
        k[191]*y[IDX_oD2HII] + k[193]*y[IDX_pD2HII] + k[194]*y[IDX_pD2HII] +
        k[194]*y[IDX_pD2HII] + k[195]*y[IDX_mD3II] + k[195]*y[IDX_mD3II] +
        k[195]*y[IDX_mD3II] + k[196]*y[IDX_mD3II] + k[197]*y[IDX_oD3II] +
        k[197]*y[IDX_oD3II] + k[197]*y[IDX_oD3II] + k[198]*y[IDX_oD3II] +
        k[199]*y[IDX_oD3II] + k[201]*y[IDX_DCOII] + k[2916]*y[IDX_pD3II] +
        k[2916]*y[IDX_pD3II] + k[2916]*y[IDX_pD3II] + k[2917]*y[IDX_pD3II] +
        k[2929]*y[IDX_DII];
    data[1663] = 0.0 - k[166]*y[IDX_DI] + k[2859]*y[IDX_pD2HII] +
        k[2860]*y[IDX_oD2HII] + k[2861]*y[IDX_pD2HII] + k[2862]*y[IDX_oD2HII] +
        k[2865]*y[IDX_pD2II] + k[2866]*y[IDX_oD2II] + k[2870]*y[IDX_oD3II] +
        k[2871]*y[IDX_mD3II] + k[2880]*y[IDX_DII] + k[2893]*y[IDX_pH2DII] +
        k[2894]*y[IDX_oH2DII] + k[2895]*y[IDX_oH2DII] + k[2898]*y[IDX_HDII];
    data[1664] = 0.0 + k[2138]*y[IDX_DM] - k[2649]*y[IDX_DI] -
        k[2881]*y[IDX_DI];
    data[1665] = 0.0 + k[2139]*y[IDX_DII] - k[2691]*y[IDX_DI] +
        k[3063]*y[IDX_H2DOII] + k[3064]*y[IDX_H2DOII] + k[3074]*y[IDX_HD2OII] +
        k[3075]*y[IDX_HD2OII] + k[3076]*y[IDX_HD2OII] + k[3084]*y[IDX_D3OII] +
        k[3085]*y[IDX_D3OII] + k[3086]*y[IDX_D3OII];
    data[1666] = 0.0 + k[3063]*y[IDX_HM] + k[3064]*y[IDX_HM] +
        k[3068]*y[IDX_DM] + k[3069]*y[IDX_DM] + k[3070]*y[IDX_DM] +
        k[3337]*y[IDX_CM] + k[3342]*y[IDX_OM] + k[3347]*y[IDX_CNM] +
        k[3356]*y[IDX_OHM] + k[3359]*y[IDX_ODM] + k[3360]*y[IDX_ODM] -
        k[3386]*y[IDX_DI] - k[3387]*y[IDX_DI] - k[3388]*y[IDX_DI] -
        k[3389]*y[IDX_DI] - k[3390]*y[IDX_DI] + k[3398]*y[IDX_pH2I] +
        k[3399]*y[IDX_oH2I] + k[3415]*y[IDX_HDI] + k[3416]*y[IDX_HDI] +
        k[3416]*y[IDX_HDI] + k[3430]*y[IDX_oD2I] + k[3431]*y[IDX_pD2I] +
        k[3432]*y[IDX_oD2I] + k[3432]*y[IDX_oD2I] + k[3433]*y[IDX_pD2I] +
        k[3433]*y[IDX_pD2I] + k[3441]*y[IDX_eM] + k[3446]*y[IDX_eM] +
        k[3459]*y[IDX_eM] + k[3460]*y[IDX_eM];
    data[1667] = 0.0 - k[1087]*y[IDX_DI] - k[1088]*y[IDX_DI] +
        k[2210]*y[IDX_DII] + k[3032]*y[IDX_HDII] + k[3035]*y[IDX_oD2II] +
        k[3036]*y[IDX_pD2II];
    data[1668] = 0.0 + k[3132]*y[IDX_HDI] + k[3139]*y[IDX_oD2I] +
        k[3140]*y[IDX_pD2I];
    data[1669] = 0.0 + k[3058]*y[IDX_DM] + k[3059]*y[IDX_DM] +
        k[3353]*y[IDX_ODM] - k[3383]*y[IDX_DI] - k[3384]*y[IDX_DI] -
        k[3385]*y[IDX_DI] + k[3413]*y[IDX_HDI] + k[3424]*y[IDX_pD2I] +
        k[3425]*y[IDX_oD2I] + k[3426]*y[IDX_pD2I] + k[3426]*y[IDX_pD2I] +
        k[3427]*y[IDX_oD2I] + k[3427]*y[IDX_oD2I];
    data[1670] = 0.0 - k[1094]*y[IDX_DI] + k[2216]*y[IDX_DII];
    data[1671] = 0.0 - k[2171]*y[IDX_DI];
    data[1672] = 0.0 - k[1058]*y[IDX_DI] - k[1062]*y[IDX_DI] +
        k[2561]*y[IDX_DII];
    data[1673] = 0.0 + k[212] + k[218] + k[364] + k[461]*y[IDX_COII] +
        k[467]*y[IDX_COII] + k[473]*y[IDX_CNII] + k[479]*y[IDX_CNII] +
        k[502]*y[IDX_OM] + k[730]*y[IDX_oH2II] + k[734]*y[IDX_pD2II] +
        k[735]*y[IDX_oD2II] + k[738]*y[IDX_HDII] + k[739]*y[IDX_HDII] +
        k[873]*y[IDX_HeII] + k[1135]*y[IDX_CI] + k[1143]*y[IDX_CHI] +
        k[1151]*y[IDX_CDI] + k[1157]*y[IDX_OI] + k[1171]*y[IDX_OHI] +
        k[1173]*y[IDX_ODI] + k[1179]*y[IDX_NI] + k[1193]*y[IDX_NHI] +
        k[1195]*y[IDX_NDI] + k[1641]*y[IDX_NII] + k[1661]*y[IDX_OII] +
        k[1699]*y[IDX_C2II] + k[1744]*y[IDX_CHII] + k[1746]*y[IDX_CDII] +
        k[1811]*y[IDX_N2II] + k[1867]*y[IDX_NHII] + k[1869]*y[IDX_NDII] +
        k[1950]*y[IDX_OHII] + k[1952]*y[IDX_ODII] + k[2888]*y[IDX_pH2II] +
        k[2889]*y[IDX_oH2II] + k[3132]*y[IDX_H2OII] + k[3134]*y[IDX_HDOII] +
        k[3136]*y[IDX_D2OII] + k[3413]*y[IDX_H3OII] + k[3415]*y[IDX_H2DOII] +
        k[3416]*y[IDX_H2DOII] + k[3416]*y[IDX_H2DOII] + k[3418]*y[IDX_HD2OII] +
        k[3419]*y[IDX_HD2OII] + k[3419]*y[IDX_HD2OII] + k[3420]*y[IDX_D3OII] +
        k[3421]*y[IDX_D3OII] + k[3421]*y[IDX_D3OII];
    data[1674] = 0.0 + k[171]*y[IDX_GRAINM] + k[295] + k[635]*y[IDX_CI] +
        k[641]*y[IDX_NI] + k[647]*y[IDX_OI] + k[653]*y[IDX_C2I] +
        k[661]*y[IDX_CHI] + k[669]*y[IDX_CDI] + k[675]*y[IDX_CNI] +
        k[681]*y[IDX_COI] + k[697]*y[IDX_pH2I] + k[698]*y[IDX_oH2I] +
        k[722]*y[IDX_pD2I] + k[723]*y[IDX_pD2I] + k[724]*y[IDX_oD2I] +
        k[725]*y[IDX_oD2I] + k[738]*y[IDX_HDI] + k[739]*y[IDX_HDI] +
        k[745]*y[IDX_N2I] + k[753]*y[IDX_NHI] + k[761]*y[IDX_NDI] +
        k[767]*y[IDX_NOI] + k[773]*y[IDX_O2I] + k[781]*y[IDX_OHI] +
        k[789]*y[IDX_ODI] - k[2102]*y[IDX_DI] + k[2159]*y[IDX_DM] +
        k[2749]*y[IDX_eM] - k[2863]*y[IDX_DI] - k[2864]*y[IDX_DI] +
        k[2886]*y[IDX_pH2I] + k[2887]*y[IDX_oH2I] + k[2898]*y[IDX_HI] +
        k[3032]*y[IDX_H2OI] + k[3042]*y[IDX_HDOI] + k[3052]*y[IDX_D2OI];
    data[1675] = 0.0 + k[3074]*y[IDX_HM] + k[3075]*y[IDX_HM] +
        k[3076]*y[IDX_HM] + k[3079]*y[IDX_DM] + k[3080]*y[IDX_DM] +
        k[3081]*y[IDX_DM] + k[3339]*y[IDX_CM] + k[3344]*y[IDX_OM] +
        k[3349]*y[IDX_CNM] + k[3363]*y[IDX_OHM] + k[3364]*y[IDX_OHM] +
        k[3366]*y[IDX_ODM] + k[3367]*y[IDX_ODM] - k[3391]*y[IDX_DI] -
        k[3392]*y[IDX_DI] - k[3393]*y[IDX_DI] + k[3402]*y[IDX_pH2I] +
        k[3403]*y[IDX_oH2I] + k[3404]*y[IDX_pH2I] + k[3404]*y[IDX_pH2I] +
        k[3405]*y[IDX_oH2I] + k[3405]*y[IDX_oH2I] + k[3418]*y[IDX_HDI] +
        k[3419]*y[IDX_HDI] + k[3419]*y[IDX_HDI] + k[3434]*y[IDX_oD2I] +
        k[3435]*y[IDX_pD2I] + k[3436]*y[IDX_oD2I] + k[3436]*y[IDX_oD2I] +
        k[3437]*y[IDX_pD2I] + k[3437]*y[IDX_pD2I] + k[3442]*y[IDX_eM] +
        k[3443]*y[IDX_eM] + k[3443]*y[IDX_eM] + k[3448]*y[IDX_eM] +
        k[3463]*y[IDX_eM];
    data[1676] = 0.0 + k[256] + k[396] + k[438]*y[IDX_CII] +
        k[442]*y[IDX_CII] + k[910]*y[IDX_HeII] - k[1090]*y[IDX_DI] -
        k[1091]*y[IDX_DI] + k[2214]*y[IDX_DII] + k[3039]*y[IDX_pH2II] +
        k[3040]*y[IDX_oH2II] + k[3042]*y[IDX_HDII] + k[3045]*y[IDX_pD2II] +
        k[3046]*y[IDX_oD2II];
    data[1677] = 0.0 + k[994]*y[IDX_NI] + k[2794]*y[IDX_eM] +
        k[2797]*y[IDX_eM] + k[3125]*y[IDX_pH2I] + k[3126]*y[IDX_oH2I] +
        k[3134]*y[IDX_HDI] + k[3143]*y[IDX_pD2I] + k[3144]*y[IDX_oD2I];
    data[1678] = 0.0 - k[1102]*y[IDX_DI] - k[1107]*y[IDX_DI] -
        k[1108]*y[IDX_DI];
    data[1679] = 0.0 + k[864]*y[IDX_CDI] + k[870]*y[IDX_pD2I] +
        k[871]*y[IDX_oD2I] + k[873]*y[IDX_HDI] + k[876]*y[IDX_NDI] +
        k[881]*y[IDX_ODI] + k[885]*y[IDX_C2DI] + k[896]*y[IDX_CD2I] +
        k[898]*y[IDX_CHDI] + k[908]*y[IDX_D2OI] + k[910]*y[IDX_HDOI] +
        k[912]*y[IDX_DCNI] + k[918]*y[IDX_DCNI] + k[922]*y[IDX_DCOI] +
        k[926]*y[IDX_DNCI] + k[928]*y[IDX_DNCI] + k[934]*y[IDX_DNOI] +
        k[943]*y[IDX_ND2I] + k[945]*y[IDX_NHDI] + k[2143]*y[IDX_DM] -
        k[2161]*y[IDX_DI];
    data[1680] = 0.0 - k[812]*y[IDX_DI] + k[2757]*y[IDX_eM];
    data[1681] = 0.0 - k[811]*y[IDX_DI];
    data[1682] = 0.0 + k[638]*y[IDX_pD2II] + k[639]*y[IDX_oD2II] +
        k[641]*y[IDX_HDII] + k[953]*y[IDX_C2DII] + k[969]*y[IDX_CD2II] +
        k[971]*y[IDX_CHDII] + k[992]*y[IDX_D2OII] + k[994]*y[IDX_HDOII] +
        k[1176]*y[IDX_pD2I] + k[1177]*y[IDX_oD2I] + k[1179]*y[IDX_HDI] +
        k[1198]*y[IDX_CDI] + k[1201]*y[IDX_NDI] + k[1204]*y[IDX_ODI] +
        k[1207]*y[IDX_CD2I] + k[1209]*y[IDX_CHDI] + k[1211]*y[IDX_CD2I] +
        k[1213]*y[IDX_CHDI] + k[1215]*y[IDX_DCOI] + k[1222]*y[IDX_C2DI] +
        k[1273]*y[IDX_oD3II] + k[1274]*y[IDX_mD3II] + k[1277]*y[IDX_oH2DII] +
        k[1278]*y[IDX_pH2DII] + k[1281]*y[IDX_oD2HII] + k[1282]*y[IDX_pD2HII] +
        k[1718]*y[IDX_CDII] + k[1821]*y[IDX_NDII] + k[1924]*y[IDX_ODII] +
        k[1970]*y[IDX_ND2II] + k[1972]*y[IDX_NHDII] + k[2005]*y[IDX_O2DII] +
        k[2958]*y[IDX_pD3II];
    data[1683] = 0.0 + k[1635]*y[IDX_CDI] + k[1638]*y[IDX_pD2I] +
        k[1639]*y[IDX_oD2I] + k[1641]*y[IDX_HDI] + k[1643]*y[IDX_NDI] +
        k[1648]*y[IDX_ODI] + k[2146]*y[IDX_DM];
    data[1684] = 0.0 + k[742]*y[IDX_pD2II] + k[743]*y[IDX_oD2II] +
        k[745]*y[IDX_HDII];
    data[1685] = 0.0 + k[1808]*y[IDX_pD2I] + k[1809]*y[IDX_oD2I] +
        k[1811]*y[IDX_HDI] - k[2130]*y[IDX_DI];
    data[1686] = 0.0 + k[2831]*y[IDX_eM];
    data[1687] = 0.0 - k[1127]*y[IDX_DI];
    data[1688] = 0.0 + k[236] + k[367] + k[422]*y[IDX_CII] +
        k[562]*y[IDX_NHI] + k[563]*y[IDX_NDI] + k[563]*y[IDX_NDI] +
        k[563]*y[IDX_NDI] + k[563]*y[IDX_NDI] + k[565]*y[IDX_NOI] +
        k[756]*y[IDX_pH2II] + k[757]*y[IDX_oH2II] + k[758]*y[IDX_pD2II] +
        k[759]*y[IDX_oD2II] + k[761]*y[IDX_HDII] + k[876]*y[IDX_HeII] +
        k[1184]*y[IDX_pH2I] + k[1185]*y[IDX_oH2I] + k[1190]*y[IDX_pD2I] +
        k[1191]*y[IDX_oD2I] + k[1195]*y[IDX_HDI] + k[1201]*y[IDX_NI] +
        k[1231]*y[IDX_OI] + k[1643]*y[IDX_NII] + k[1664]*y[IDX_OII] +
        k[1703]*y[IDX_C2II] + k[1918]*y[IDX_O2II] + k[2049]*y[IDX_CI] +
        k[2196]*y[IDX_DII];
    data[1689] = 0.0 + k[1821]*y[IDX_NI] + k[1827]*y[IDX_C2I] +
        k[1839]*y[IDX_COI] + k[1858]*y[IDX_pH2I] + k[1859]*y[IDX_oH2I] +
        k[1864]*y[IDX_pD2I] + k[1865]*y[IDX_oD2I] + k[1869]*y[IDX_HDI] +
        k[2760]*y[IDX_eM];
    data[1690] = 0.0 + k[269] + k[411] + k[452]*y[IDX_CII] +
        k[528]*y[IDX_OI] + k[943]*y[IDX_HeII] + k[1046]*y[IDX_CI] +
        k[1050]*y[IDX_CI] + k[2256]*y[IDX_DII];
    data[1691] = 0.0 + k[1970]*y[IDX_NI] + k[1974]*y[IDX_OI] +
        k[2836]*y[IDX_eM] + k[2836]*y[IDX_eM] + k[2839]*y[IDX_eM];
    data[1692] = 0.0 + k[562]*y[IDX_NDI] + k[750]*y[IDX_pD2II] +
        k[751]*y[IDX_oD2II] + k[753]*y[IDX_HDII] + k[1188]*y[IDX_pD2I] +
        k[1189]*y[IDX_oD2I] + k[1193]*y[IDX_HDI] + k[2194]*y[IDX_DII];
    data[1693] = 0.0 + k[1862]*y[IDX_pD2I] + k[1863]*y[IDX_oD2I] +
        k[1867]*y[IDX_HDI];
    data[1694] = 0.0 + k[2254]*y[IDX_DII];
    data[1695] = 0.0 + k[271] + k[413] + k[454]*y[IDX_CII] +
        k[530]*y[IDX_OI] + k[945]*y[IDX_HeII] + k[1048]*y[IDX_CI] +
        k[1052]*y[IDX_CI] + k[2258]*y[IDX_DII];
    data[1696] = 0.0 + k[1972]*y[IDX_NI] + k[1976]*y[IDX_OI] +
        k[2837]*y[IDX_eM] + k[2841]*y[IDX_eM];
    data[1697] = 0.0 + k[565]*y[IDX_NDI] + k[764]*y[IDX_pD2II] +
        k[765]*y[IDX_oD2II] + k[767]*y[IDX_HDII] - k[1097]*y[IDX_DI] -
        k[1099]*y[IDX_DI] + k[2198]*y[IDX_DII];
    data[1698] = 0.0 - k[1129]*y[IDX_DI];
    data[1699] = 0.0 - k[1996]*y[IDX_DI];
    data[1700] = 0.0 + k[510]*y[IDX_CD2I] + k[510]*y[IDX_CD2I] +
        k[511]*y[IDX_CHDI] + k[516]*y[IDX_DCOI] + k[520]*y[IDX_DCNI] +
        k[528]*y[IDX_ND2I] + k[530]*y[IDX_NHDI] + k[644]*y[IDX_pD2II] +
        k[645]*y[IDX_oD2II] + k[647]*y[IDX_HDII] + k[973]*y[IDX_CD2II] +
        k[975]*y[IDX_CHDII] + k[1154]*y[IDX_pD2I] + k[1155]*y[IDX_oD2I] +
        k[1157]*y[IDX_HDI] + k[1228]*y[IDX_CDI] + k[1231]*y[IDX_NDI] +
        k[1233]*y[IDX_ODI] + k[1300]*y[IDX_oD3II] + k[1301]*y[IDX_mD3II] +
        k[1304]*y[IDX_oH2DII] + k[1305]*y[IDX_pH2DII] + k[1308]*y[IDX_oD2HII] +
        k[1309]*y[IDX_pD2HII] + k[1573]*y[IDX_DNOII] + k[1720]*y[IDX_CDII] +
        k[1926]*y[IDX_ODII] + k[1974]*y[IDX_ND2II] + k[1976]*y[IDX_NHDII] +
        k[2186]*y[IDX_DII] - k[2663]*y[IDX_DI];
    data[1701] = 0.0 + k[1654]*y[IDX_CDI] + k[1658]*y[IDX_pD2I] +
        k[1659]*y[IDX_oD2I] + k[1661]*y[IDX_HDI] + k[1664]*y[IDX_NDI] +
        k[1666]*y[IDX_ODI] + k[2149]*y[IDX_DM] - k[2240]*y[IDX_DI];
    data[1702] = 0.0 + k[499]*y[IDX_pD2I] + k[500]*y[IDX_oD2I] +
        k[502]*y[IDX_HDI] - k[2717]*y[IDX_DI] + k[3342]*y[IDX_H2DOII] +
        k[3344]*y[IDX_HD2OII] + k[3345]*y[IDX_D3OII];
    data[1703] = 0.0 + k[770]*y[IDX_pD2II] + k[771]*y[IDX_oD2II] +
        k[773]*y[IDX_HDII] - k[1111]*y[IDX_DI] + k[2200]*y[IDX_DII];
    data[1704] = 0.0 + k[1918]*y[IDX_NDI];
    data[1705] = 0.0 + k[279] - k[1115]*y[IDX_DI] - k[1119]*y[IDX_DI] -
        k[1123]*y[IDX_DI];
    data[1706] = 0.0 + k[2005]*y[IDX_NI] + k[2844]*y[IDX_eM];
    data[1707] = 0.0 - k[1114]*y[IDX_DI] - k[1118]*y[IDX_DI] -
        k[1122]*y[IDX_DI];
    data[1708] = 0.0 + k[242] + k[375] + k[426]*y[IDX_CII] +
        k[552]*y[IDX_CNI] + k[558]*y[IDX_COI] + k[784]*y[IDX_pH2II] +
        k[785]*y[IDX_oH2II] + k[786]*y[IDX_pD2II] + k[787]*y[IDX_oD2II] +
        k[789]*y[IDX_HDII] + k[881]*y[IDX_HeII] - k[1081]*y[IDX_DI] +
        k[1162]*y[IDX_pH2I] + k[1163]*y[IDX_oH2I] + k[1168]*y[IDX_pD2I] +
        k[1169]*y[IDX_oD2I] + k[1173]*y[IDX_HDI] + k[1204]*y[IDX_NI] +
        k[1233]*y[IDX_OI] + k[1648]*y[IDX_NII] + k[1666]*y[IDX_OII] +
        k[2054]*y[IDX_CI] + k[2204]*y[IDX_DII] - k[2669]*y[IDX_DI];
    data[1709] = 0.0 + k[1924]*y[IDX_NI] + k[1926]*y[IDX_OI] +
        k[1941]*y[IDX_pH2I] + k[1942]*y[IDX_oH2I] + k[1947]*y[IDX_pD2I] +
        k[1948]*y[IDX_oD2I] + k[1952]*y[IDX_HDI] + k[2764]*y[IDX_eM];
    data[1710] = 0.0 - k[2735]*y[IDX_DI] + k[3353]*y[IDX_H3OII] +
        k[3359]*y[IDX_H2DOII] + k[3360]*y[IDX_H2DOII] + k[3366]*y[IDX_HD2OII] +
        k[3367]*y[IDX_HD2OII] + k[3371]*y[IDX_D3OII];
    data[1711] = 0.0 + k[778]*y[IDX_pD2II] + k[779]*y[IDX_oD2II] +
        k[781]*y[IDX_HDII] - k[1080]*y[IDX_DI] + k[1166]*y[IDX_pD2I] +
        k[1167]*y[IDX_oD2I] + k[1171]*y[IDX_HDI] + k[2202]*y[IDX_DII] -
        k[2668]*y[IDX_DI];
    data[1712] = 0.0 + k[1945]*y[IDX_pD2I] + k[1946]*y[IDX_oD2I] +
        k[1950]*y[IDX_HDI];
    data[1713] = 0.0 - k[2734]*y[IDX_DI] + k[3356]*y[IDX_H2DOII] +
        k[3363]*y[IDX_HD2OII] + k[3364]*y[IDX_HD2OII] + k[3369]*y[IDX_D3OII] +
        k[3370]*y[IDX_D3OII];
    data[1714] = 0.0 - k[2738]*y[IDX_DI] + k[2742]*y[IDX_CDII] +
        k[2747]*y[IDX_pD2II] + k[2747]*y[IDX_pD2II] + k[2748]*y[IDX_oD2II] +
        k[2748]*y[IDX_oD2II] + k[2749]*y[IDX_HDII] + k[2757]*y[IDX_HeDII] +
        k[2760]*y[IDX_NDII] + k[2764]*y[IDX_ODII] + k[2766]*y[IDX_C2DII] +
        k[2770]*y[IDX_C2DII] + k[2779]*y[IDX_CD2II] + k[2781]*y[IDX_CHDII] +
        k[2783]*y[IDX_CD2II] + k[2783]*y[IDX_CD2II] + k[2784]*y[IDX_CHDII] +
        k[2792]*y[IDX_D2OII] + k[2794]*y[IDX_HDOII] + k[2796]*y[IDX_D2OII] +
        k[2796]*y[IDX_D2OII] + k[2797]*y[IDX_HDOII] + k[2800]*y[IDX_mD3II] +
        k[2800]*y[IDX_mD3II] + k[2800]*y[IDX_mD3II] + k[2801]*y[IDX_oD3II] +
        k[2801]*y[IDX_oD3II] + k[2801]*y[IDX_oD3II] + k[2802]*y[IDX_oH2DII] +
        k[2803]*y[IDX_pH2DII] + k[2804]*y[IDX_oD2HII] + k[2804]*y[IDX_oD2HII] +
        k[2805]*y[IDX_pD2HII] + k[2805]*y[IDX_pD2HII] + k[2809]*y[IDX_mD3II] +
        k[2810]*y[IDX_oD3II] + k[2811]*y[IDX_oD3II] + k[2814]*y[IDX_oH2DII] +
        k[2815]*y[IDX_pH2DII] + k[2818]*y[IDX_oD2HII] + k[2819]*y[IDX_pD2HII] +
        k[2821]*y[IDX_DCNII] + k[2823]*y[IDX_DCOII] + k[2825]*y[IDX_DNCII] +
        k[2827]*y[IDX_DNOII] + k[2829]*y[IDX_DOCII] + k[2831]*y[IDX_N2DII] +
        k[2836]*y[IDX_ND2II] + k[2836]*y[IDX_ND2II] + k[2837]*y[IDX_NHDII] +
        k[2839]*y[IDX_ND2II] + k[2841]*y[IDX_NHDII] + k[2844]*y[IDX_O2DII] +
        k[2847]*y[IDX_DII] + k[2914]*y[IDX_pD3II] + k[2914]*y[IDX_pD3II] +
        k[2914]*y[IDX_pD3II] + k[2915]*y[IDX_pD3II] + k[3441]*y[IDX_H2DOII] +
        k[3442]*y[IDX_HD2OII] + k[3443]*y[IDX_HD2OII] + k[3443]*y[IDX_HD2OII] +
        k[3444]*y[IDX_D3OII] + k[3444]*y[IDX_D3OII] + k[3446]*y[IDX_H2DOII] +
        k[3448]*y[IDX_HD2OII] + k[3449]*y[IDX_D3OII] + k[3459]*y[IDX_H2DOII] +
        k[3460]*y[IDX_H2DOII] + k[3463]*y[IDX_HD2OII] + k[3464]*y[IDX_D3OII] +
        k[3465]*y[IDX_D3OII];
    data[1715] = 0.0 + k[195]*y[IDX_GRAINM] + k[195]*y[IDX_GRAINM] +
        k[195]*y[IDX_GRAINM] + k[196]*y[IDX_GRAINM] + k[330] + k[331] +
        k[1274]*y[IDX_NI] + k[1301]*y[IDX_OI] + k[2800]*y[IDX_eM] +
        k[2800]*y[IDX_eM] + k[2800]*y[IDX_eM] + k[2809]*y[IDX_eM] +
        k[2871]*y[IDX_HI];
    data[1716] = 0.0 + k[211] + k[211] + k[216] + k[363] + k[363] +
        k[459]*y[IDX_COII] + k[465]*y[IDX_COII] + k[471]*y[IDX_CNII] +
        k[477]*y[IDX_CNII] + k[500]*y[IDX_OM] + k[708]*y[IDX_pH2II] +
        k[709]*y[IDX_oH2II] + k[714]*y[IDX_pD2II] + k[715]*y[IDX_pD2II] +
        k[716]*y[IDX_oD2II] + k[717]*y[IDX_oD2II] + k[724]*y[IDX_HDII] +
        k[725]*y[IDX_HDII] + k[871]*y[IDX_HeII] + k[1133]*y[IDX_CI] +
        k[1141]*y[IDX_CHI] + k[1149]*y[IDX_CDI] + k[1155]*y[IDX_OI] +
        k[1167]*y[IDX_OHI] + k[1169]*y[IDX_ODI] + k[1177]*y[IDX_NI] +
        k[1189]*y[IDX_NHI] + k[1191]*y[IDX_NDI] + k[1639]*y[IDX_NII] +
        k[1659]*y[IDX_OII] + k[1697]*y[IDX_C2II] + k[1740]*y[IDX_CHII] +
        k[1742]*y[IDX_CDII] + k[1809]*y[IDX_N2II] + k[1863]*y[IDX_NHII] +
        k[1865]*y[IDX_NDII] + k[1946]*y[IDX_OHII] + k[1948]*y[IDX_ODII] +
        k[2931]*y[IDX_oD2II] + k[2934]*y[IDX_pD2II] + k[3139]*y[IDX_H2OII] +
        k[3144]*y[IDX_HDOII] + k[3145]*y[IDX_D2OII] + k[3425]*y[IDX_H3OII] +
        k[3427]*y[IDX_H3OII] + k[3427]*y[IDX_H3OII] + k[3430]*y[IDX_H2DOII] +
        k[3432]*y[IDX_H2DOII] + k[3432]*y[IDX_H2DOII] + k[3434]*y[IDX_HD2OII] +
        k[3436]*y[IDX_HD2OII] + k[3436]*y[IDX_HD2OII] + k[3438]*y[IDX_D3OII] +
        k[3438]*y[IDX_D3OII];
    data[1717] = 0.0 + k[173]*y[IDX_GRAINM] + k[173]*y[IDX_GRAINM] + k[293]
        + k[633]*y[IDX_CI] + k[639]*y[IDX_NI] + k[645]*y[IDX_OI] +
        k[651]*y[IDX_C2I] + k[659]*y[IDX_CHI] + k[667]*y[IDX_CDI] +
        k[673]*y[IDX_CNI] + k[679]*y[IDX_COI] + k[690]*y[IDX_pH2I] +
        k[692]*y[IDX_oH2I] + k[712]*y[IDX_pD2I] + k[713]*y[IDX_pD2I] +
        k[716]*y[IDX_oD2I] + k[717]*y[IDX_oD2I] + k[735]*y[IDX_HDI] +
        k[743]*y[IDX_N2I] + k[751]*y[IDX_NHI] + k[759]*y[IDX_NDI] +
        k[765]*y[IDX_NOI] + k[771]*y[IDX_O2I] + k[779]*y[IDX_OHI] +
        k[787]*y[IDX_ODI] - k[2101]*y[IDX_DI] + k[2157]*y[IDX_DM] +
        k[2748]*y[IDX_eM] + k[2748]*y[IDX_eM] + k[2866]*y[IDX_HI] +
        k[2931]*y[IDX_oD2I] + k[2933]*y[IDX_pD2I] + k[3035]*y[IDX_H2OI] +
        k[3046]*y[IDX_HDOI] + k[3053]*y[IDX_D2OI];
    data[1718] = 0.0 + k[189]*y[IDX_GRAINM] + k[189]*y[IDX_GRAINM] +
        k[191]*y[IDX_GRAINM] + k[342] + k[1281]*y[IDX_NI] + k[1308]*y[IDX_OI] +
        k[2804]*y[IDX_eM] + k[2804]*y[IDX_eM] + k[2818]*y[IDX_eM] +
        k[2860]*y[IDX_HI] + k[2862]*y[IDX_HI] - k[2867]*y[IDX_DI] -
        k[2868]*y[IDX_DI];
    data[1719] = 0.0 + k[197]*y[IDX_GRAINM] + k[197]*y[IDX_GRAINM] +
        k[197]*y[IDX_GRAINM] + k[198]*y[IDX_GRAINM] + k[199]*y[IDX_GRAINM] +
        k[328] + k[329] + k[1273]*y[IDX_NI] + k[1300]*y[IDX_OI] +
        k[2801]*y[IDX_eM] + k[2801]*y[IDX_eM] + k[2801]*y[IDX_eM] +
        k[2810]*y[IDX_eM] + k[2811]*y[IDX_eM] + k[2870]*y[IDX_HI];
    data[1720] = 0.0 + k[691]*y[IDX_pD2II] + k[692]*y[IDX_oD2II] +
        k[698]*y[IDX_HDII] + k[1147]*y[IDX_CDI] + k[1163]*y[IDX_ODI] +
        k[1185]*y[IDX_NDI] + k[1736]*y[IDX_CDII] + k[1859]*y[IDX_NDII] +
        k[1942]*y[IDX_ODII] + k[2887]*y[IDX_HDII] + k[3126]*y[IDX_HDOII] +
        k[3130]*y[IDX_D2OII] + k[3399]*y[IDX_H2DOII] + k[3403]*y[IDX_HD2OII] +
        k[3405]*y[IDX_HD2OII] + k[3405]*y[IDX_HD2OII] + k[3408]*y[IDX_D3OII] +
        k[3410]*y[IDX_D3OII] + k[3410]*y[IDX_D3OII];
    data[1721] = 0.0 + k[665]*y[IDX_CDI] + k[707]*y[IDX_pD2I] +
        k[709]*y[IDX_oD2I] + k[730]*y[IDX_HDI] + k[757]*y[IDX_NDI] +
        k[785]*y[IDX_ODI] - k[2099]*y[IDX_DI] + k[2153]*y[IDX_DM] +
        k[2889]*y[IDX_HDI] + k[3040]*y[IDX_HDOI] + k[3050]*y[IDX_D2OI];
    data[1722] = 0.0 + k[186]*y[IDX_GRAINM] + k[187]*y[IDX_GRAINM] + k[334]
        + k[335] + k[1277]*y[IDX_NI] + k[1304]*y[IDX_OI] + k[2802]*y[IDX_eM] +
        k[2814]*y[IDX_eM] - k[2857]*y[IDX_DI] - k[2858]*y[IDX_DI] +
        k[2894]*y[IDX_HI] + k[2895]*y[IDX_HI];
    data[1723] = 0.0 - k[2892]*y[IDX_DI];
    data[1724] = 0.0 + k[210] + k[210] + k[215] + k[362] + k[362] +
        k[458]*y[IDX_COII] + k[464]*y[IDX_COII] + k[470]*y[IDX_CNII] +
        k[476]*y[IDX_CNII] + k[499]*y[IDX_OM] + k[706]*y[IDX_pH2II] +
        k[707]*y[IDX_oH2II] + k[710]*y[IDX_pD2II] + k[711]*y[IDX_pD2II] +
        k[712]*y[IDX_oD2II] + k[713]*y[IDX_oD2II] + k[722]*y[IDX_HDII] +
        k[723]*y[IDX_HDII] + k[870]*y[IDX_HeII] + k[1132]*y[IDX_CI] +
        k[1140]*y[IDX_CHI] + k[1148]*y[IDX_CDI] + k[1154]*y[IDX_OI] +
        k[1166]*y[IDX_OHI] + k[1168]*y[IDX_ODI] + k[1176]*y[IDX_NI] +
        k[1188]*y[IDX_NHI] + k[1190]*y[IDX_NDI] + k[1638]*y[IDX_NII] +
        k[1658]*y[IDX_OII] + k[1696]*y[IDX_C2II] + k[1739]*y[IDX_CHII] +
        k[1741]*y[IDX_CDII] + k[1808]*y[IDX_N2II] + k[1862]*y[IDX_NHII] +
        k[1864]*y[IDX_NDII] + k[1945]*y[IDX_OHII] + k[1947]*y[IDX_ODII] +
        k[2933]*y[IDX_oD2II] + k[2935]*y[IDX_pD2II] + k[3140]*y[IDX_H2OII] +
        k[3143]*y[IDX_HDOII] + k[3146]*y[IDX_D2OII] + k[3424]*y[IDX_H3OII] +
        k[3426]*y[IDX_H3OII] + k[3426]*y[IDX_H3OII] + k[3431]*y[IDX_H2DOII] +
        k[3433]*y[IDX_H2DOII] + k[3433]*y[IDX_H2DOII] + k[3435]*y[IDX_HD2OII] +
        k[3437]*y[IDX_HD2OII] + k[3437]*y[IDX_HD2OII] + k[3439]*y[IDX_D3OII] +
        k[3439]*y[IDX_D3OII];
    data[1725] = 0.0 + k[175]*y[IDX_GRAINM] + k[175]*y[IDX_GRAINM] + k[292]
        + k[632]*y[IDX_CI] + k[638]*y[IDX_NI] + k[644]*y[IDX_OI] +
        k[650]*y[IDX_C2I] + k[658]*y[IDX_CHI] + k[666]*y[IDX_CDI] +
        k[672]*y[IDX_CNI] + k[678]*y[IDX_COI] + k[689]*y[IDX_pH2I] +
        k[691]*y[IDX_oH2I] + k[710]*y[IDX_pD2I] + k[711]*y[IDX_pD2I] +
        k[714]*y[IDX_oD2I] + k[715]*y[IDX_oD2I] + k[734]*y[IDX_HDI] +
        k[742]*y[IDX_N2I] + k[750]*y[IDX_NHI] + k[758]*y[IDX_NDI] +
        k[764]*y[IDX_NOI] + k[770]*y[IDX_O2I] + k[778]*y[IDX_OHI] +
        k[786]*y[IDX_ODI] - k[2100]*y[IDX_DI] + k[2156]*y[IDX_DM] +
        k[2747]*y[IDX_eM] + k[2747]*y[IDX_eM] + k[2865]*y[IDX_HI] +
        k[2934]*y[IDX_oD2I] + k[2935]*y[IDX_pD2I] + k[3036]*y[IDX_H2OI] +
        k[3045]*y[IDX_HDOI] + k[3054]*y[IDX_D2OI];
    data[1726] = 0.0 + k[193]*y[IDX_GRAINM] + k[194]*y[IDX_GRAINM] +
        k[194]*y[IDX_GRAINM] + k[343] + k[1282]*y[IDX_NI] + k[1309]*y[IDX_OI] +
        k[2805]*y[IDX_eM] + k[2805]*y[IDX_eM] + k[2819]*y[IDX_eM] +
        k[2859]*y[IDX_HI] + k[2861]*y[IDX_HI] - k[2869]*y[IDX_DI];
    data[1727] = 0.0 + k[2914]*y[IDX_eM] + k[2914]*y[IDX_eM] +
        k[2914]*y[IDX_eM] + k[2915]*y[IDX_eM] + k[2916]*y[IDX_GRAINM] +
        k[2916]*y[IDX_GRAINM] + k[2916]*y[IDX_GRAINM] + k[2917]*y[IDX_GRAINM] +
        k[2958]*y[IDX_NI] + k[2987] + k[2988];
    data[1728] = 0.0 + k[689]*y[IDX_pD2II] + k[690]*y[IDX_oD2II] +
        k[697]*y[IDX_HDII] + k[1146]*y[IDX_CDI] + k[1162]*y[IDX_ODI] +
        k[1184]*y[IDX_NDI] + k[1735]*y[IDX_CDII] + k[1858]*y[IDX_NDII] +
        k[1941]*y[IDX_ODII] + k[2886]*y[IDX_HDII] + k[3125]*y[IDX_HDOII] +
        k[3129]*y[IDX_D2OII] + k[3398]*y[IDX_H2DOII] + k[3402]*y[IDX_HD2OII] +
        k[3404]*y[IDX_HD2OII] + k[3404]*y[IDX_HD2OII] + k[3409]*y[IDX_D3OII] +
        k[3411]*y[IDX_D3OII] + k[3411]*y[IDX_D3OII];
    data[1729] = 0.0 + k[664]*y[IDX_CDI] + k[706]*y[IDX_pD2I] +
        k[708]*y[IDX_oD2I] + k[756]*y[IDX_NDI] + k[784]*y[IDX_ODI] -
        k[2098]*y[IDX_DI] + k[2152]*y[IDX_DM] + k[2888]*y[IDX_HDI] -
        k[2896]*y[IDX_DI] - k[2899]*y[IDX_DI] + k[3039]*y[IDX_HDOI] +
        k[3049]*y[IDX_D2OI];
    data[1730] = 0.0 + k[183]*y[IDX_GRAINM] + k[185]*y[IDX_GRAINM] + k[336]
        + k[337] + k[1278]*y[IDX_NI] + k[1305]*y[IDX_OI] + k[2803]*y[IDX_eM] +
        k[2815]*y[IDX_eM] - k[2855]*y[IDX_DI] - k[2856]*y[IDX_DI] +
        k[2893]*y[IDX_HI];
    data[1731] = 0.0 - k[2890]*y[IDX_DI] - k[2891]*y[IDX_DI];
    data[1732] = 0.0 - k[2136]*y[IDX_DII];
    data[1733] = 0.0 - k[2188]*y[IDX_DII];
    data[1734] = 0.0 - k[1529]*y[IDX_DII] - k[2526]*y[IDX_DII];
    data[1735] = 0.0 - k[1527]*y[IDX_DII] - k[2524]*y[IDX_DII];
    data[1736] = 0.0 - k[2206]*y[IDX_DII];
    data[1737] = 0.0 - k[2208]*y[IDX_DII];
    data[1738] = 0.0 - k[1531]*y[IDX_DII];
    data[1739] = 0.0 - k[2192]*y[IDX_DII];
    data[1740] = 0.0 + k[231];
    data[1741] = 0.0 - k[1537]*y[IDX_DII] - k[2530]*y[IDX_DII];
    data[1742] = 0.0 - k[2190]*y[IDX_DII];
    data[1743] = 0.0 - k[1533]*y[IDX_DII] - k[1534]*y[IDX_DII] -
        k[2528]*y[IDX_DII];
    data[1744] = 0.0 - k[1540]*y[IDX_DII] - k[1541]*y[IDX_DII] -
        k[2532]*y[IDX_DII];
    data[1745] = 0.0 + k[2265]*y[IDX_DI];
    data[1746] = 0.0 + k[2083]*y[IDX_DI];
    data[1747] = 0.0 - k[1543]*y[IDX_DII];
    data[1748] = 0.0 + k[2375]*y[IDX_DI];
    data[1749] = 0.0 + k[204] + k[2083]*y[IDX_COII] + k[2098]*y[IDX_pH2II] +
        k[2099]*y[IDX_oH2II] + k[2100]*y[IDX_pD2II] + k[2101]*y[IDX_oD2II] +
        k[2102]*y[IDX_HDII] + k[2130]*y[IDX_N2II] + k[2161]*y[IDX_HeII] +
        k[2171]*y[IDX_HCNII] + k[2172]*y[IDX_DCNII] + k[2240]*y[IDX_OII] +
        k[2265]*y[IDX_CNII] + k[2375]*y[IDX_CO2II] - k[2650]*y[IDX_DII] -
        k[2651]*y[IDX_DII] + k[2881]*y[IDX_HII];
    data[1750] = 0.0 - k[1527]*y[IDX_C2HI] - k[1529]*y[IDX_C2DI] -
        k[1531]*y[IDX_CCOI] - k[1533]*y[IDX_CH2I] - k[1534]*y[IDX_CH2I] -
        k[1537]*y[IDX_CD2I] - k[1540]*y[IDX_CHDI] - k[1541]*y[IDX_CHDI] -
        k[1543]*y[IDX_CO2I] - k[1545]*y[IDX_HCOI] - k[1547]*y[IDX_DCOI] -
        k[1549]*y[IDX_HCOI] - k[1551]*y[IDX_DCOI] - k[1680]*y[IDX_HNCI] +
        k[1680]*y[IDX_HNCI] - k[1681]*y[IDX_HNCI] - k[1684]*y[IDX_DNCI] +
        k[1684]*y[IDX_DNCI] - k[1686]*y[IDX_HNOI] - k[1688]*y[IDX_DNOI] -
        k[2136]*y[IDX_CM] - k[2139]*y[IDX_HM] - k[2140]*y[IDX_DM] -
        k[2186]*y[IDX_OI] - k[2188]*y[IDX_C2I] - k[2190]*y[IDX_CHI] -
        k[2192]*y[IDX_CDI] - k[2194]*y[IDX_NHI] - k[2196]*y[IDX_NDI] -
        k[2198]*y[IDX_NOI] - k[2200]*y[IDX_O2I] - k[2202]*y[IDX_OHI] -
        k[2204]*y[IDX_ODI] - k[2206]*y[IDX_C2NI] - k[2208]*y[IDX_C3I] -
        k[2210]*y[IDX_H2OI] - k[2212]*y[IDX_D2OI] - k[2214]*y[IDX_HDOI] -
        k[2216]*y[IDX_HCNI] - k[2218]*y[IDX_DCNI] - k[2254]*y[IDX_NH2I] -
        k[2256]*y[IDX_ND2I] - k[2258]*y[IDX_NHDI] - k[2524]*y[IDX_C2HI] -
        k[2526]*y[IDX_C2DI] - k[2528]*y[IDX_CH2I] - k[2530]*y[IDX_CD2I] -
        k[2532]*y[IDX_CHDI] - k[2561]*y[IDX_HCOI] - k[2563]*y[IDX_DCOI] -
        k[2648]*y[IDX_HI] - k[2650]*y[IDX_DI] - k[2651]*y[IDX_DI] -
        k[2847]*y[IDX_eM] - k[2853]*y[IDX_HDI] - k[2854]*y[IDX_HDI] -
        k[2872]*y[IDX_oD2I] + k[2872]*y[IDX_oD2I] - k[2873]*y[IDX_pD2I] +
        k[2873]*y[IDX_pD2I] - k[2880]*y[IDX_HI] - k[2882]*y[IDX_pH2I] -
        k[2883]*y[IDX_oH2I] - k[2929]*y[IDX_GRAINM];
    data[1751] = 0.0 - k[2140]*y[IDX_DII];
    data[1752] = 0.0 + k[904]*y[IDX_HeII] - k[2212]*y[IDX_DII];
    data[1753] = 0.0 - k[2218]*y[IDX_DII];
    data[1754] = 0.0 + k[2172]*y[IDX_DI];
    data[1755] = 0.0 - k[1547]*y[IDX_DII] - k[1551]*y[IDX_DII] -
        k[2563]*y[IDX_DII];
    data[1756] = 0.0 + k[1682]*y[IDX_HII] - k[1684]*y[IDX_DII] +
        k[1684]*y[IDX_DII];
    data[1757] = 0.0 + k[932]*y[IDX_HeII] - k[1688]*y[IDX_DII];
    data[1758] = 0.0 - k[2929]*y[IDX_DII];
    data[1759] = 0.0 - k[2648]*y[IDX_DII] - k[2880]*y[IDX_DII];
    data[1760] = 0.0 + k[1682]*y[IDX_DNCI] + k[2851]*y[IDX_pD2I] +
        k[2852]*y[IDX_oD2I] + k[2881]*y[IDX_DI] + k[2884]*y[IDX_HDI] +
        k[2885]*y[IDX_HDI];
    data[1761] = 0.0 - k[2139]*y[IDX_DII];
    data[1762] = 0.0 - k[2210]*y[IDX_DII];
    data[1763] = 0.0 - k[2216]*y[IDX_DII];
    data[1764] = 0.0 + k[2171]*y[IDX_DI];
    data[1765] = 0.0 - k[1545]*y[IDX_DII] - k[1549]*y[IDX_DII] -
        k[2561]*y[IDX_DII];
    data[1766] = 0.0 + k[217] + k[224] + k[872]*y[IDX_HeII] -
        k[2853]*y[IDX_DII] - k[2854]*y[IDX_DII] + k[2884]*y[IDX_HII] +
        k[2885]*y[IDX_HII];
    data[1767] = 0.0 + k[294] + k[2102]*y[IDX_DI];
    data[1768] = 0.0 + k[905]*y[IDX_HeII] - k[2214]*y[IDX_DII];
    data[1769] = 0.0 - k[1680]*y[IDX_DII] + k[1680]*y[IDX_DII] -
        k[1681]*y[IDX_DII];
    data[1770] = 0.0 - k[1686]*y[IDX_DII];
    data[1771] = 0.0 + k[870]*y[IDX_pD2I] + k[871]*y[IDX_oD2I] +
        k[872]*y[IDX_HDI] + k[904]*y[IDX_D2OI] + k[905]*y[IDX_HDOI] +
        k[932]*y[IDX_DNOI] + k[2161]*y[IDX_DI];
    data[1772] = 0.0 + k[2130]*y[IDX_DI];
    data[1773] = 0.0 - k[2196]*y[IDX_DII];
    data[1774] = 0.0 - k[2256]*y[IDX_DII];
    data[1775] = 0.0 - k[2194]*y[IDX_DII];
    data[1776] = 0.0 - k[2254]*y[IDX_DII];
    data[1777] = 0.0 - k[2258]*y[IDX_DII];
    data[1778] = 0.0 - k[2198]*y[IDX_DII];
    data[1779] = 0.0 - k[2186]*y[IDX_DII];
    data[1780] = 0.0 + k[2240]*y[IDX_DI];
    data[1781] = 0.0 - k[2200]*y[IDX_DII];
    data[1782] = 0.0 - k[2204]*y[IDX_DII];
    data[1783] = 0.0 + k[297];
    data[1784] = 0.0 - k[2202]*y[IDX_DII];
    data[1785] = 0.0 - k[2847]*y[IDX_DII];
    data[1786] = 0.0 + k[310] + k[311];
    data[1787] = 0.0 + k[216] + k[222] + k[871]*y[IDX_HeII] +
        k[2852]*y[IDX_HII] - k[2872]*y[IDX_DII] + k[2872]*y[IDX_DII];
    data[1788] = 0.0 + k[293] + k[2101]*y[IDX_DI];
    data[1789] = 0.0 + k[322];
    data[1790] = 0.0 + k[308] + k[309];
    data[1791] = 0.0 - k[2883]*y[IDX_DII];
    data[1792] = 0.0 + k[2099]*y[IDX_DI];
    data[1793] = 0.0 + k[312] + k[313];
    data[1794] = 0.0 + k[215] + k[221] + k[870]*y[IDX_HeII] +
        k[2851]*y[IDX_HII] - k[2873]*y[IDX_DII] + k[2873]*y[IDX_DII];
    data[1795] = 0.0 + k[292] + k[2100]*y[IDX_DI];
    data[1796] = 0.0 + k[323];
    data[1797] = 0.0 + k[2985] + k[2986];
    data[1798] = 0.0 - k[2882]*y[IDX_DII];
    data[1799] = 0.0 + k[2098]*y[IDX_DI];
    data[1800] = 0.0 + k[314] + k[315];
    data[1801] = 0.0 - k[2688]*y[IDX_DM];
    data[1802] = 0.0 - k[2134]*y[IDX_DM];
    data[1803] = 0.0 - k[2698]*y[IDX_DM];
    data[1804] = 0.0 - k[2702]*y[IDX_DM];
    data[1805] = 0.0 - k[2700]*y[IDX_DM];
    data[1806] = 0.0 - k[2704]*y[IDX_DM];
    data[1807] = 0.0 - k[2706]*y[IDX_DM];
    data[1808] = 0.0 - k[2692]*y[IDX_DM] + k[2738]*y[IDX_eM];
    data[1809] = 0.0 - k[2140]*y[IDX_DM];
    data[1810] = 0.0 - k[346] - k[484]*y[IDX_H2OI] - k[485]*y[IDX_H2OI] -
        k[488]*y[IDX_D2OI] - k[491]*y[IDX_HDOI] - k[492]*y[IDX_HDOI] -
        k[494]*y[IDX_HCNI] - k[496]*y[IDX_DCNI] - k[830]*y[IDX_oH3II] -
        k[831]*y[IDX_oH3II] - k[832]*y[IDX_pH3II] - k[833]*y[IDX_pH3II] -
        k[834]*y[IDX_oD3II] - k[835]*y[IDX_mD3II] - k[840]*y[IDX_oH2DII] -
        k[841]*y[IDX_oH2DII] - k[842]*y[IDX_pH2DII] - k[843]*y[IDX_pH2DII] -
        k[844]*y[IDX_pH2DII] - k[845]*y[IDX_oH2DII] - k[846]*y[IDX_pH2DII] -
        k[854]*y[IDX_oD2HII] - k[855]*y[IDX_oD2HII] - k[856]*y[IDX_pD2HII] -
        k[857]*y[IDX_pD2HII] - k[859]*y[IDX_HCOII] - k[861]*y[IDX_DCOII] -
        k[2134]*y[IDX_CII] - k[2138]*y[IDX_HII] - k[2140]*y[IDX_DII] -
        k[2143]*y[IDX_HeII] - k[2146]*y[IDX_NII] - k[2149]*y[IDX_OII] -
        k[2152]*y[IDX_pH2II] - k[2153]*y[IDX_oH2II] - k[2156]*y[IDX_pD2II] -
        k[2157]*y[IDX_oD2II] - k[2159]*y[IDX_HDII] - k[2688]*y[IDX_CI] -
        k[2690]*y[IDX_HI] - k[2692]*y[IDX_DI] - k[2694]*y[IDX_NI] -
        k[2696]*y[IDX_OI] - k[2698]*y[IDX_C2I] - k[2700]*y[IDX_CHI] -
        k[2702]*y[IDX_CDI] - k[2704]*y[IDX_CNI] - k[2706]*y[IDX_COI] -
        k[2708]*y[IDX_NHI] - k[2710]*y[IDX_NDI] - k[2712]*y[IDX_OHI] -
        k[2714]*y[IDX_ODI] - k[2983]*y[IDX_pD3II] - k[2984]*y[IDX_pD3II] -
        k[3055]*y[IDX_H3OII] - k[3056]*y[IDX_H3OII] - k[3057]*y[IDX_H3OII] -
        k[3058]*y[IDX_H3OII] - k[3059]*y[IDX_H3OII] - k[3065]*y[IDX_H2DOII] -
        k[3066]*y[IDX_H2DOII] - k[3067]*y[IDX_H2DOII] - k[3068]*y[IDX_H2DOII] -
        k[3069]*y[IDX_H2DOII] - k[3070]*y[IDX_H2DOII] - k[3077]*y[IDX_HD2OII] -
        k[3078]*y[IDX_HD2OII] - k[3079]*y[IDX_HD2OII] - k[3080]*y[IDX_HD2OII] -
        k[3081]*y[IDX_HD2OII] - k[3087]*y[IDX_D3OII] - k[3088]*y[IDX_D3OII] -
        k[3089]*y[IDX_H3OII] - k[3090]*y[IDX_H3OII] - k[3091]*y[IDX_H3OII] -
        k[3095]*y[IDX_H2DOII] - k[3096]*y[IDX_H2DOII] - k[3097]*y[IDX_H2DOII] -
        k[3098]*y[IDX_H2DOII] - k[3099]*y[IDX_H2DOII] - k[3105]*y[IDX_HD2OII] -
        k[3106]*y[IDX_HD2OII] - k[3107]*y[IDX_HD2OII] - k[3111]*y[IDX_D3OII] -
        k[3112]*y[IDX_D3OII];
    data[1811] = 0.0 - k[488]*y[IDX_DM];
    data[1812] = 0.0 - k[3087]*y[IDX_DM] - k[3088]*y[IDX_DM] -
        k[3111]*y[IDX_DM] - k[3112]*y[IDX_DM];
    data[1813] = 0.0 - k[496]*y[IDX_DM];
    data[1814] = 0.0 - k[861]*y[IDX_DM];
    data[1815] = 0.0 - k[2690]*y[IDX_DM];
    data[1816] = 0.0 - k[2138]*y[IDX_DM];
    data[1817] = 0.0 - k[3065]*y[IDX_DM] - k[3066]*y[IDX_DM] -
        k[3067]*y[IDX_DM] - k[3068]*y[IDX_DM] - k[3069]*y[IDX_DM] -
        k[3070]*y[IDX_DM] - k[3095]*y[IDX_DM] - k[3096]*y[IDX_DM] -
        k[3097]*y[IDX_DM] - k[3098]*y[IDX_DM] - k[3099]*y[IDX_DM];
    data[1818] = 0.0 - k[484]*y[IDX_DM] - k[485]*y[IDX_DM];
    data[1819] = 0.0 - k[3055]*y[IDX_DM] - k[3056]*y[IDX_DM] -
        k[3057]*y[IDX_DM] - k[3058]*y[IDX_DM] - k[3059]*y[IDX_DM] -
        k[3089]*y[IDX_DM] - k[3090]*y[IDX_DM] - k[3091]*y[IDX_DM];
    data[1820] = 0.0 - k[494]*y[IDX_DM];
    data[1821] = 0.0 - k[859]*y[IDX_DM];
    data[1822] = 0.0 + k[223];
    data[1823] = 0.0 - k[2159]*y[IDX_DM];
    data[1824] = 0.0 - k[3077]*y[IDX_DM] - k[3078]*y[IDX_DM] -
        k[3079]*y[IDX_DM] - k[3080]*y[IDX_DM] - k[3081]*y[IDX_DM] -
        k[3105]*y[IDX_DM] - k[3106]*y[IDX_DM] - k[3107]*y[IDX_DM];
    data[1825] = 0.0 - k[491]*y[IDX_DM] - k[492]*y[IDX_DM];
    data[1826] = 0.0 - k[2143]*y[IDX_DM];
    data[1827] = 0.0 - k[2694]*y[IDX_DM];
    data[1828] = 0.0 - k[2146]*y[IDX_DM];
    data[1829] = 0.0 - k[2710]*y[IDX_DM];
    data[1830] = 0.0 - k[2708]*y[IDX_DM];
    data[1831] = 0.0 - k[2696]*y[IDX_DM];
    data[1832] = 0.0 - k[2149]*y[IDX_DM];
    data[1833] = 0.0 - k[2714]*y[IDX_DM];
    data[1834] = 0.0 - k[2712]*y[IDX_DM];
    data[1835] = 0.0 + k[2738]*y[IDX_DI];
    data[1836] = 0.0 - k[835]*y[IDX_DM];
    data[1837] = 0.0 + k[222];
    data[1838] = 0.0 - k[2157]*y[IDX_DM];
    data[1839] = 0.0 - k[854]*y[IDX_DM] - k[855]*y[IDX_DM];
    data[1840] = 0.0 - k[834]*y[IDX_DM];
    data[1841] = 0.0 - k[2153]*y[IDX_DM];
    data[1842] = 0.0 - k[840]*y[IDX_DM] - k[841]*y[IDX_DM] -
        k[845]*y[IDX_DM];
    data[1843] = 0.0 - k[830]*y[IDX_DM] - k[831]*y[IDX_DM];
    data[1844] = 0.0 + k[221];
    data[1845] = 0.0 - k[2156]*y[IDX_DM];
    data[1846] = 0.0 - k[856]*y[IDX_DM] - k[857]*y[IDX_DM];
    data[1847] = 0.0 - k[2983]*y[IDX_DM] - k[2984]*y[IDX_DM];
    data[1848] = 0.0 - k[2152]*y[IDX_DM];
    data[1849] = 0.0 - k[842]*y[IDX_DM] - k[843]*y[IDX_DM] -
        k[844]*y[IDX_DM] - k[846]*y[IDX_DM];
    data[1850] = 0.0 - k[832]*y[IDX_DM] - k[833]*y[IDX_DM];
    data[1851] = 0.0 - k[436]*y[IDX_D2OI] - k[440]*y[IDX_D2OI];
    data[1852] = 0.0 + k[3338]*y[IDX_HD2OII] + k[3340]*y[IDX_D3OII];
    data[1853] = 0.0 + k[2480]*y[IDX_D2OII];
    data[1854] = 0.0 - k[1706]*y[IDX_D2OI];
    data[1855] = 0.0 + k[2492]*y[IDX_D2OII];
    data[1856] = 0.0 + k[2489]*y[IDX_D2OII];
    data[1857] = 0.0 - k[961]*y[IDX_D2OI];
    data[1858] = 0.0 + k[2486]*y[IDX_D2OII] + k[3302]*y[IDX_H2DOII] +
        k[3304]*y[IDX_HD2OII] + k[3305]*y[IDX_D3OII];
    data[1859] = 0.0 - k[1782]*y[IDX_D2OI] - k[3310]*y[IDX_D2OI];
    data[1860] = 0.0 + k[2498]*y[IDX_D2OII];
    data[1861] = 0.0 + k[2483]*y[IDX_D2OII] + k[3295]*y[IDX_HD2OII] +
        k[3297]*y[IDX_D3OII];
    data[1862] = 0.0 - k[1780]*y[IDX_D2OI] - k[1781]*y[IDX_D2OI] -
        k[3309]*y[IDX_D2OI];
    data[1863] = 0.0 + k[2495]*y[IDX_D2OII];
    data[1864] = 0.0 + k[2501]*y[IDX_D2OII];
    data[1865] = 0.0 - k[600]*y[IDX_D2OI] - k[604]*y[IDX_D2OI];
    data[1866] = 0.0 + k[3348]*y[IDX_HD2OII] + k[3350]*y[IDX_D3OII];
    data[1867] = 0.0 - k[981]*y[IDX_D2OI];
    data[1868] = 0.0 - k[623]*y[IDX_D2OI] - k[2335]*y[IDX_D2OI];
    data[1869] = 0.0 + k[792]*y[IDX_pD2II] + k[793]*y[IDX_oD2II];
    data[1870] = 0.0 - k[2378]*y[IDX_D2OI];
    data[1871] = 0.0 - k[1089]*y[IDX_D2OI] + k[1123]*y[IDX_O2DI] +
        k[2669]*y[IDX_ODI] + k[2735]*y[IDX_ODM];
    data[1872] = 0.0 - k[2212]*y[IDX_D2OI];
    data[1873] = 0.0 - k[488]*y[IDX_D2OI] + k[2714]*y[IDX_ODI] +
        k[3095]*y[IDX_H2DOII] + k[3096]*y[IDX_H2DOII] + k[3105]*y[IDX_HD2OII] +
        k[3111]*y[IDX_D3OII] + k[3112]*y[IDX_D3OII];
    data[1874] = 0.0 - k[254] - k[394] - k[398] - k[436]*y[IDX_CII] -
        k[440]*y[IDX_CII] - k[486]*y[IDX_HM] - k[487]*y[IDX_HM] -
        k[488]*y[IDX_DM] - k[600]*y[IDX_CNII] - k[604]*y[IDX_CNII] -
        k[623]*y[IDX_COII] - k[904]*y[IDX_HeII] - k[908]*y[IDX_HeII] -
        k[961]*y[IDX_C2NII] - k[981]*y[IDX_CNCII] - k[1083]*y[IDX_HI] -
        k[1084]*y[IDX_HI] - k[1089]*y[IDX_DI] - k[1706]*y[IDX_C2II] -
        k[1780]*y[IDX_CHII] - k[1781]*y[IDX_CHII] - k[1782]*y[IDX_CDII] -
        k[1813]*y[IDX_N2II] - k[1893]*y[IDX_NHII] - k[1894]*y[IDX_NHII] -
        k[1895]*y[IDX_NDII] - k[1903]*y[IDX_NHII] - k[1904]*y[IDX_NHII] -
        k[1905]*y[IDX_NDII] - k[2211]*y[IDX_HII] - k[2212]*y[IDX_DII] -
        k[2232]*y[IDX_NII] - k[2247]*y[IDX_OII] - k[2335]*y[IDX_COII] -
        k[2372]*y[IDX_HeII] - k[2378]*y[IDX_CO2II] - k[2382]*y[IDX_HCNII] -
        k[2383]*y[IDX_DCNII] - k[2448]*y[IDX_pH2II] - k[2449]*y[IDX_oH2II] -
        k[2450]*y[IDX_pD2II] - k[2451]*y[IDX_oD2II] - k[2452]*y[IDX_HDII] -
        k[2594]*y[IDX_OHII] - k[2595]*y[IDX_ODII] - k[2623]*y[IDX_N2II] -
        k[2627]*y[IDX_NHII] - k[2628]*y[IDX_NDII] - k[3047]*y[IDX_pH2II] -
        k[3048]*y[IDX_oH2II] - k[3049]*y[IDX_pH2II] - k[3050]*y[IDX_oH2II] -
        k[3051]*y[IDX_HDII] - k[3052]*y[IDX_HDII] - k[3053]*y[IDX_oD2II] -
        k[3054]*y[IDX_pD2II] - k[3116]*y[IDX_HCNII] - k[3117]*y[IDX_DCNII] -
        k[3121]*y[IDX_HCOII] - k[3122]*y[IDX_DCOII] - k[3167]*y[IDX_H2OII] -
        k[3168]*y[IDX_H2OII] - k[3169]*y[IDX_HDOII] - k[3170]*y[IDX_HDOII] -
        k[3171]*y[IDX_D2OII] - k[3239]*y[IDX_pH3II] - k[3240]*y[IDX_pH3II] -
        k[3241]*y[IDX_oH3II] - k[3242]*y[IDX_pH3II] - k[3243]*y[IDX_oH3II] -
        k[3244]*y[IDX_pH3II] - k[3245]*y[IDX_oH3II] - k[3246]*y[IDX_pH3II] -
        k[3247]*y[IDX_oH3II] - k[3248]*y[IDX_pH2DII] - k[3249]*y[IDX_oH2DII] -
        k[3250]*y[IDX_pH2DII] - k[3251]*y[IDX_oH2DII] - k[3252]*y[IDX_pH2DII] -
        k[3253]*y[IDX_pH2DII] - k[3254]*y[IDX_oH2DII] - k[3255]*y[IDX_oH2DII] -
        k[3256]*y[IDX_pD2HII] - k[3257]*y[IDX_oD2HII] - k[3258]*y[IDX_pD2HII] -
        k[3259]*y[IDX_pD2HII] - k[3260]*y[IDX_oD2HII] - k[3261]*y[IDX_oD2HII] -
        k[3262]*y[IDX_pD3II] - k[3263]*y[IDX_pD3II] - k[3264]*y[IDX_mD3II] -
        k[3265]*y[IDX_mD3II] - k[3266]*y[IDX_oD3II] - k[3267]*y[IDX_oD3II] -
        k[3271]*y[IDX_HNCII] - k[3272]*y[IDX_DNCII] - k[3276]*y[IDX_HNOII] -
        k[3277]*y[IDX_DNOII] - k[3281]*y[IDX_N2HII] - k[3282]*y[IDX_N2DII] -
        k[3309]*y[IDX_CHII] - k[3310]*y[IDX_CDII] - k[3314]*y[IDX_NHII] -
        k[3315]*y[IDX_NDII] - k[3319]*y[IDX_OHII] - k[3320]*y[IDX_ODII] -
        k[3331]*y[IDX_NH2II] - k[3332]*y[IDX_NH2II] - k[3333]*y[IDX_NHDII] -
        k[3334]*y[IDX_NHDII] - k[3335]*y[IDX_ND2II];
    data[1875] = 0.0 + k[2180]*y[IDX_NOI] + k[2183]*y[IDX_O2I] +
        k[2480]*y[IDX_C2I] + k[2483]*y[IDX_CHI] + k[2486]*y[IDX_CDI] +
        k[2489]*y[IDX_C2HI] + k[2492]*y[IDX_C2DI] + k[2495]*y[IDX_CH2I] +
        k[2498]*y[IDX_CD2I] + k[2501]*y[IDX_CHDI] + k[2504]*y[IDX_HCOI] +
        k[2507]*y[IDX_DCOI] + k[2512]*y[IDX_NH2I] + k[2515]*y[IDX_ND2I] +
        k[2518]*y[IDX_NHDI] - k[3171]*y[IDX_D2OI];
    data[1876] = 0.0 + k[3108]*y[IDX_HM] + k[3111]*y[IDX_DM] +
        k[3112]*y[IDX_DM] + k[3297]*y[IDX_CHI] + k[3305]*y[IDX_CDI] +
        k[3340]*y[IDX_CM] + k[3345]*y[IDX_OM] + k[3350]*y[IDX_CNM] +
        k[3368]*y[IDX_OHM] + k[3369]*y[IDX_OHM] + k[3371]*y[IDX_ODM] +
        k[3449]*y[IDX_eM];
    data[1877] = 0.0 + k[508]*y[IDX_ODM];
    data[1878] = 0.0 - k[2383]*y[IDX_D2OI] - k[3117]*y[IDX_D2OI];
    data[1879] = 0.0 + k[575]*y[IDX_ODI] + k[2507]*y[IDX_D2OII];
    data[1880] = 0.0 - k[3122]*y[IDX_D2OI];
    data[1881] = 0.0 - k[3272]*y[IDX_D2OI];
    data[1882] = 0.0 + k[579]*y[IDX_ODI];
    data[1883] = 0.0 - k[3277]*y[IDX_D2OI];
    data[1884] = 0.0 - k[1083]*y[IDX_D2OI] - k[1084]*y[IDX_D2OI];
    data[1885] = 0.0 - k[2211]*y[IDX_D2OI];
    data[1886] = 0.0 - k[486]*y[IDX_D2OI] - k[487]*y[IDX_D2OI] +
        k[3100]*y[IDX_HD2OII] + k[3101]*y[IDX_HD2OII] + k[3108]*y[IDX_D3OII];
    data[1887] = 0.0 + k[3095]*y[IDX_DM] + k[3096]*y[IDX_DM] +
        k[3302]*y[IDX_CDI] + k[3357]*y[IDX_ODM];
    data[1888] = 0.0 - k[3167]*y[IDX_D2OI] - k[3168]*y[IDX_D2OI];
    data[1889] = 0.0 - k[2382]*y[IDX_D2OI] - k[3116]*y[IDX_D2OI];
    data[1890] = 0.0 + k[2504]*y[IDX_D2OII];
    data[1891] = 0.0 - k[3121]*y[IDX_D2OI];
    data[1892] = 0.0 + k[1172]*y[IDX_ODI];
    data[1893] = 0.0 - k[2452]*y[IDX_D2OI] - k[3051]*y[IDX_D2OI] -
        k[3052]*y[IDX_D2OI];
    data[1894] = 0.0 + k[3100]*y[IDX_HM] + k[3101]*y[IDX_HM] +
        k[3105]*y[IDX_DM] + k[3295]*y[IDX_CHI] + k[3304]*y[IDX_CDI] +
        k[3338]*y[IDX_CM] + k[3343]*y[IDX_OM] + k[3348]*y[IDX_CNM] +
        k[3361]*y[IDX_OHM] + k[3365]*y[IDX_ODM] + k[3366]*y[IDX_ODM] +
        k[3447]*y[IDX_eM];
    data[1895] = 0.0 - k[3169]*y[IDX_D2OI] - k[3170]*y[IDX_D2OI];
    data[1896] = 0.0 - k[3271]*y[IDX_D2OI];
    data[1897] = 0.0 - k[3276]*y[IDX_D2OI];
    data[1898] = 0.0 - k[904]*y[IDX_D2OI] - k[908]*y[IDX_D2OI] -
        k[2372]*y[IDX_D2OI];
    data[1899] = 0.0 - k[2232]*y[IDX_D2OI];
    data[1900] = 0.0 - k[1813]*y[IDX_D2OI] - k[2623]*y[IDX_D2OI];
    data[1901] = 0.0 - k[3282]*y[IDX_D2OI];
    data[1902] = 0.0 - k[3281]*y[IDX_D2OI];
    data[1903] = 0.0 - k[1895]*y[IDX_D2OI] - k[1905]*y[IDX_D2OI] -
        k[2628]*y[IDX_D2OI] - k[3315]*y[IDX_D2OI];
    data[1904] = 0.0 + k[567]*y[IDX_NOI] + k[581]*y[IDX_OHI] +
        k[587]*y[IDX_ODI] + k[2515]*y[IDX_D2OII];
    data[1905] = 0.0 - k[3335]*y[IDX_D2OI];
    data[1906] = 0.0 - k[1893]*y[IDX_D2OI] - k[1894]*y[IDX_D2OI] -
        k[1903]*y[IDX_D2OI] - k[1904]*y[IDX_D2OI] - k[2627]*y[IDX_D2OI] -
        k[3314]*y[IDX_D2OI];
    data[1907] = 0.0 + k[2512]*y[IDX_D2OII];
    data[1908] = 0.0 - k[3331]*y[IDX_D2OI] - k[3332]*y[IDX_D2OI];
    data[1909] = 0.0 + k[588]*y[IDX_ODI] + k[2518]*y[IDX_D2OII];
    data[1910] = 0.0 - k[3333]*y[IDX_D2OI] - k[3334]*y[IDX_D2OI];
    data[1911] = 0.0 + k[567]*y[IDX_ND2I] + k[2180]*y[IDX_D2OII];
    data[1912] = 0.0 + k[1999]*y[IDX_pD2I] + k[2000]*y[IDX_oD2I];
    data[1913] = 0.0 - k[2247]*y[IDX_D2OI];
    data[1914] = 0.0 + k[2725]*y[IDX_pD2I] + k[2726]*y[IDX_oD2I] +
        k[3343]*y[IDX_HD2OII] + k[3345]*y[IDX_D3OII];
    data[1915] = 0.0 + k[2183]*y[IDX_D2OII];
    data[1916] = 0.0 + k[593]*y[IDX_ODI] + k[1123]*y[IDX_DI];
    data[1917] = 0.0 + k[571]*y[IDX_ODI] + k[571]*y[IDX_ODI] +
        k[575]*y[IDX_DCOI] + k[579]*y[IDX_DNOI] + k[587]*y[IDX_ND2I] +
        k[588]*y[IDX_NHDI] + k[593]*y[IDX_O2DI] + k[1168]*y[IDX_pD2I] +
        k[1169]*y[IDX_oD2I] + k[1172]*y[IDX_HDI] + k[2669]*y[IDX_DI] +
        k[2714]*y[IDX_DM];
    data[1918] = 0.0 - k[2595]*y[IDX_D2OI] - k[3320]*y[IDX_D2OI];
    data[1919] = 0.0 + k[508]*y[IDX_DCNI] + k[2735]*y[IDX_DI] +
        k[3357]*y[IDX_H2DOII] + k[3365]*y[IDX_HD2OII] + k[3366]*y[IDX_HD2OII] +
        k[3371]*y[IDX_D3OII];
    data[1920] = 0.0 + k[581]*y[IDX_ND2I] + k[1164]*y[IDX_pD2I] +
        k[1165]*y[IDX_oD2I];
    data[1921] = 0.0 - k[2594]*y[IDX_D2OI] - k[3319]*y[IDX_D2OI];
    data[1922] = 0.0 + k[3361]*y[IDX_HD2OII] + k[3368]*y[IDX_D3OII] +
        k[3369]*y[IDX_D3OII];
    data[1923] = 0.0 + k[3447]*y[IDX_HD2OII] + k[3449]*y[IDX_D3OII];
    data[1924] = 0.0 - k[3264]*y[IDX_D2OI] - k[3265]*y[IDX_D2OI];
    data[1925] = 0.0 + k[1165]*y[IDX_OHI] + k[1169]*y[IDX_ODI] +
        k[2000]*y[IDX_NO2II] + k[2726]*y[IDX_OM];
    data[1926] = 0.0 + k[793]*y[IDX_CO2I] - k[2451]*y[IDX_D2OI] -
        k[3053]*y[IDX_D2OI];
    data[1927] = 0.0 - k[3257]*y[IDX_D2OI] - k[3260]*y[IDX_D2OI] -
        k[3261]*y[IDX_D2OI];
    data[1928] = 0.0 - k[3266]*y[IDX_D2OI] - k[3267]*y[IDX_D2OI];
    data[1929] = 0.0 - k[2449]*y[IDX_D2OI] - k[3048]*y[IDX_D2OI] -
        k[3050]*y[IDX_D2OI];
    data[1930] = 0.0 - k[3249]*y[IDX_D2OI] - k[3251]*y[IDX_D2OI] -
        k[3254]*y[IDX_D2OI] - k[3255]*y[IDX_D2OI];
    data[1931] = 0.0 - k[3241]*y[IDX_D2OI] - k[3243]*y[IDX_D2OI] -
        k[3245]*y[IDX_D2OI] - k[3247]*y[IDX_D2OI];
    data[1932] = 0.0 + k[1164]*y[IDX_OHI] + k[1168]*y[IDX_ODI] +
        k[1999]*y[IDX_NO2II] + k[2725]*y[IDX_OM];
    data[1933] = 0.0 + k[792]*y[IDX_CO2I] - k[2450]*y[IDX_D2OI] -
        k[3054]*y[IDX_D2OI];
    data[1934] = 0.0 - k[3256]*y[IDX_D2OI] - k[3258]*y[IDX_D2OI] -
        k[3259]*y[IDX_D2OI];
    data[1935] = 0.0 - k[3262]*y[IDX_D2OI] - k[3263]*y[IDX_D2OI];
    data[1936] = 0.0 - k[2448]*y[IDX_D2OI] - k[3047]*y[IDX_D2OI] -
        k[3049]*y[IDX_D2OI];
    data[1937] = 0.0 - k[3248]*y[IDX_D2OI] - k[3250]*y[IDX_D2OI] -
        k[3252]*y[IDX_D2OI] - k[3253]*y[IDX_D2OI];
    data[1938] = 0.0 - k[3239]*y[IDX_D2OI] - k[3240]*y[IDX_D2OI] -
        k[3242]*y[IDX_D2OI] - k[3244]*y[IDX_D2OI] - k[3246]*y[IDX_D2OI];
    data[1939] = 0.0 - k[988]*y[IDX_D2OII];
    data[1940] = 0.0 - k[1239]*y[IDX_D2OII] - k[2480]*y[IDX_D2OII];
    data[1941] = 0.0 - k[2492]*y[IDX_D2OII];
    data[1942] = 0.0 - k[2489]*y[IDX_D2OII];
    data[1943] = 0.0 - k[1249]*y[IDX_D2OII] - k[2486]*y[IDX_D2OII];
    data[1944] = 0.0 - k[2498]*y[IDX_D2OII];
    data[1945] = 0.0 - k[1243]*y[IDX_D2OII] - k[1244]*y[IDX_D2OII] -
        k[2483]*y[IDX_D2OII];
    data[1946] = 0.0 - k[2495]*y[IDX_D2OII];
    data[1947] = 0.0 - k[2501]*y[IDX_D2OII];
    data[1948] = 0.0 - k[1253]*y[IDX_D2OII];
    data[1949] = 0.0 + k[2335]*y[IDX_D2OI];
    data[1950] = 0.0 + k[2378]*y[IDX_D2OI];
    data[1951] = 0.0 + k[3386]*y[IDX_H2DOII] + k[3387]*y[IDX_H2DOII] +
        k[3391]*y[IDX_HD2OII] + k[3394]*y[IDX_D3OII] + k[3395]*y[IDX_D3OII];
    data[1952] = 0.0 + k[2212]*y[IDX_D2OI];
    data[1953] = 0.0 + k[398] + k[2211]*y[IDX_HII] + k[2212]*y[IDX_DII] +
        k[2232]*y[IDX_NII] + k[2247]*y[IDX_OII] + k[2335]*y[IDX_COII] +
        k[2372]*y[IDX_HeII] + k[2378]*y[IDX_CO2II] + k[2382]*y[IDX_HCNII] +
        k[2383]*y[IDX_DCNII] + k[2448]*y[IDX_pH2II] + k[2449]*y[IDX_oH2II] +
        k[2450]*y[IDX_pD2II] + k[2451]*y[IDX_oD2II] + k[2452]*y[IDX_HDII] +
        k[2594]*y[IDX_OHII] + k[2595]*y[IDX_ODII] + k[2623]*y[IDX_N2II] +
        k[2627]*y[IDX_NHII] + k[2628]*y[IDX_NDII] - k[3171]*y[IDX_D2OII];
    data[1954] = 0.0 - k[988]*y[IDX_CI] - k[992]*y[IDX_NI] -
        k[996]*y[IDX_OI] - k[1239]*y[IDX_C2I] - k[1243]*y[IDX_CHI] -
        k[1244]*y[IDX_CHI] - k[1249]*y[IDX_CDI] - k[1253]*y[IDX_COI] -
        k[2180]*y[IDX_NOI] - k[2183]*y[IDX_O2I] - k[2480]*y[IDX_C2I] -
        k[2483]*y[IDX_CHI] - k[2486]*y[IDX_CDI] - k[2489]*y[IDX_C2HI] -
        k[2492]*y[IDX_C2DI] - k[2495]*y[IDX_CH2I] - k[2498]*y[IDX_CD2I] -
        k[2501]*y[IDX_CHDI] - k[2504]*y[IDX_HCOI] - k[2507]*y[IDX_DCOI] -
        k[2512]*y[IDX_NH2I] - k[2515]*y[IDX_ND2I] - k[2518]*y[IDX_NHDI] -
        k[2789]*y[IDX_eM] - k[2792]*y[IDX_eM] - k[2796]*y[IDX_eM] -
        k[3127]*y[IDX_pH2I] - k[3128]*y[IDX_oH2I] - k[3129]*y[IDX_pH2I] -
        k[3130]*y[IDX_oH2I] - k[3135]*y[IDX_HDI] - k[3136]*y[IDX_HDI] -
        k[3145]*y[IDX_oD2I] - k[3146]*y[IDX_pD2I] - k[3148]*y[IDX_NHI] -
        k[3151]*y[IDX_NDI] - k[3153]*y[IDX_OHI] - k[3156]*y[IDX_ODI] -
        k[3159]*y[IDX_H2OI] - k[3160]*y[IDX_H2OI] - k[3165]*y[IDX_HDOI] -
        k[3166]*y[IDX_HDOI] - k[3171]*y[IDX_D2OI] - k[3173]*y[IDX_HCOI] -
        k[3176]*y[IDX_DCOI];
    data[1955] = 0.0 + k[3380]*y[IDX_HI] + k[3394]*y[IDX_DI] +
        k[3395]*y[IDX_DI];
    data[1956] = 0.0 + k[1015]*y[IDX_ODI] + k[2383]*y[IDX_D2OI];
    data[1957] = 0.0 + k[1968]*y[IDX_ODII] - k[2507]*y[IDX_D2OII] -
        k[3176]*y[IDX_D2OII];
    data[1958] = 0.0 + k[1031]*y[IDX_ODI];
    data[1959] = 0.0 + k[1569]*y[IDX_ODI];
    data[1960] = 0.0 + k[1589]*y[IDX_ODI];
    data[1961] = 0.0 + k[3375]*y[IDX_HD2OII] + k[3376]*y[IDX_HD2OII] +
        k[3380]*y[IDX_D3OII];
    data[1962] = 0.0 + k[2211]*y[IDX_D2OI];
    data[1963] = 0.0 + k[3386]*y[IDX_DI] + k[3387]*y[IDX_DI];
    data[1964] = 0.0 - k[3159]*y[IDX_D2OII] - k[3160]*y[IDX_D2OII];
    data[1965] = 0.0 + k[2382]*y[IDX_D2OI];
    data[1966] = 0.0 - k[2504]*y[IDX_D2OII] - k[3173]*y[IDX_D2OII];
    data[1967] = 0.0 + k[1951]*y[IDX_ODII] - k[3135]*y[IDX_D2OII] -
        k[3136]*y[IDX_D2OII];
    data[1968] = 0.0 + k[788]*y[IDX_ODI] + k[2452]*y[IDX_D2OI];
    data[1969] = 0.0 + k[3375]*y[IDX_HI] + k[3376]*y[IDX_HI] +
        k[3391]*y[IDX_DI];
    data[1970] = 0.0 - k[3165]*y[IDX_D2OII] - k[3166]*y[IDX_D2OII];
    data[1971] = 0.0 + k[2372]*y[IDX_D2OI];
    data[1972] = 0.0 - k[992]*y[IDX_D2OII];
    data[1973] = 0.0 + k[2232]*y[IDX_D2OI];
    data[1974] = 0.0 + k[2623]*y[IDX_D2OI];
    data[1975] = 0.0 + k[1633]*y[IDX_ODI];
    data[1976] = 0.0 - k[3151]*y[IDX_D2OII];
    data[1977] = 0.0 + k[1885]*y[IDX_ODI] + k[2628]*y[IDX_D2OI];
    data[1978] = 0.0 - k[2515]*y[IDX_D2OII];
    data[1979] = 0.0 - k[3148]*y[IDX_D2OII];
    data[1980] = 0.0 + k[2627]*y[IDX_D2OI];
    data[1981] = 0.0 - k[2512]*y[IDX_D2OII];
    data[1982] = 0.0 - k[2518]*y[IDX_D2OII];
    data[1983] = 0.0 - k[2180]*y[IDX_D2OII];
    data[1984] = 0.0 - k[996]*y[IDX_D2OII] + k[1300]*y[IDX_oD3II] +
        k[1301]*y[IDX_mD3II] + k[1306]*y[IDX_oD2HII] + k[1307]*y[IDX_pD2HII];
    data[1985] = 0.0 + k[2247]*y[IDX_D2OI];
    data[1986] = 0.0 - k[2183]*y[IDX_D2OII];
    data[1987] = 0.0 + k[2043]*y[IDX_ODI];
    data[1988] = 0.0 + k[786]*y[IDX_pD2II] + k[787]*y[IDX_oD2II] +
        k[788]*y[IDX_HDII] + k[1015]*y[IDX_DCNII] + k[1031]*y[IDX_DCOII] +
        k[1499]*y[IDX_oD3II] + k[1500]*y[IDX_mD3II] + k[1501]*y[IDX_oH2DII] +
        k[1502]*y[IDX_pH2DII] + k[1503]*y[IDX_pH2DII] + k[1509]*y[IDX_oD2HII] +
        k[1510]*y[IDX_pD2HII] + k[1569]*y[IDX_DNCII] + k[1589]*y[IDX_DNOII] +
        k[1633]*y[IDX_N2DII] + k[1885]*y[IDX_NDII] + k[1964]*y[IDX_ODII] +
        k[2043]*y[IDX_O2DII] + k[2977]*y[IDX_pD3II] + k[2978]*y[IDX_pD3II] -
        k[3156]*y[IDX_D2OII];
    data[1989] = 0.0 + k[1947]*y[IDX_pD2I] + k[1948]*y[IDX_oD2I] +
        k[1951]*y[IDX_HDI] + k[1964]*y[IDX_ODI] + k[1968]*y[IDX_DCOI] +
        k[2595]*y[IDX_D2OI];
    data[1990] = 0.0 + k[776]*y[IDX_pD2II] + k[777]*y[IDX_oD2II] +
        k[1482]*y[IDX_oD3II] + k[1483]*y[IDX_mD3II] - k[3153]*y[IDX_D2OII];
    data[1991] = 0.0 + k[1943]*y[IDX_pD2I] + k[1944]*y[IDX_oD2I] +
        k[2594]*y[IDX_D2OI];
    data[1992] = 0.0 - k[2789]*y[IDX_D2OII] - k[2792]*y[IDX_D2OII] -
        k[2796]*y[IDX_D2OII];
    data[1993] = 0.0 + k[1301]*y[IDX_OI] + k[1483]*y[IDX_OHI] +
        k[1500]*y[IDX_ODI];
    data[1994] = 0.0 + k[1944]*y[IDX_OHII] + k[1948]*y[IDX_ODII] -
        k[3145]*y[IDX_D2OII];
    data[1995] = 0.0 + k[777]*y[IDX_OHI] + k[787]*y[IDX_ODI] +
        k[2451]*y[IDX_D2OI];
    data[1996] = 0.0 + k[1306]*y[IDX_OI] + k[1509]*y[IDX_ODI];
    data[1997] = 0.0 + k[1300]*y[IDX_OI] + k[1482]*y[IDX_OHI] +
        k[1499]*y[IDX_ODI];
    data[1998] = 0.0 - k[3128]*y[IDX_D2OII] - k[3130]*y[IDX_D2OII];
    data[1999] = 0.0 + k[2449]*y[IDX_D2OI];
    data[2000] = 0.0 + k[1501]*y[IDX_ODI];
    data[2001] = 0.0 + k[1943]*y[IDX_OHII] + k[1947]*y[IDX_ODII] -
        k[3146]*y[IDX_D2OII];
    data[2002] = 0.0 + k[776]*y[IDX_OHI] + k[786]*y[IDX_ODI] +
        k[2450]*y[IDX_D2OI];
    data[2003] = 0.0 + k[1307]*y[IDX_OI] + k[1510]*y[IDX_ODI];
    data[2004] = 0.0 + k[2977]*y[IDX_ODI] + k[2978]*y[IDX_ODI];
    data[2005] = 0.0 - k[3127]*y[IDX_D2OII] - k[3129]*y[IDX_D2OII];
    data[2006] = 0.0 + k[2448]*y[IDX_D2OI];
    data[2007] = 0.0 + k[1502]*y[IDX_ODI] + k[1503]*y[IDX_ODI];
    data[2008] = 0.0 - k[3289]*y[IDX_D3OII] - k[3290]*y[IDX_D3OII];
    data[2009] = 0.0 - k[3340]*y[IDX_D3OII];
    data[2010] = 0.0 - k[3305]*y[IDX_D3OII];
    data[2011] = 0.0 + k[3310]*y[IDX_D2OI];
    data[2012] = 0.0 - k[3296]*y[IDX_D3OII] - k[3297]*y[IDX_D3OII];
    data[2013] = 0.0 - k[3350]*y[IDX_D3OII];
    data[2014] = 0.0 - k[3394]*y[IDX_D3OII] - k[3395]*y[IDX_D3OII];
    data[2015] = 0.0 - k[3087]*y[IDX_D3OII] - k[3088]*y[IDX_D3OII] -
        k[3111]*y[IDX_D3OII] - k[3112]*y[IDX_D3OII];
    data[2016] = 0.0 + k[3051]*y[IDX_HDII] + k[3053]*y[IDX_oD2II] +
        k[3054]*y[IDX_pD2II] + k[3117]*y[IDX_DCNII] + k[3122]*y[IDX_DCOII] +
        k[3169]*y[IDX_HDOII] + k[3171]*y[IDX_D2OII] + k[3248]*y[IDX_pH2DII] +
        k[3249]*y[IDX_oH2DII] + k[3256]*y[IDX_pD2HII] + k[3257]*y[IDX_oD2HII] +
        k[3262]*y[IDX_pD3II] + k[3263]*y[IDX_pD3II] + k[3264]*y[IDX_mD3II] +
        k[3265]*y[IDX_mD3II] + k[3266]*y[IDX_oD3II] + k[3267]*y[IDX_oD3II] +
        k[3272]*y[IDX_DNCII] + k[3277]*y[IDX_DNOII] + k[3282]*y[IDX_N2DII] +
        k[3310]*y[IDX_CDII] + k[3315]*y[IDX_NDII] + k[3320]*y[IDX_ODII] +
        k[3333]*y[IDX_NHDII] + k[3335]*y[IDX_ND2II];
    data[2017] = 0.0 + k[3135]*y[IDX_HDI] + k[3145]*y[IDX_oD2I] +
        k[3146]*y[IDX_pD2I] + k[3151]*y[IDX_NDI] + k[3156]*y[IDX_ODI] +
        k[3165]*y[IDX_HDOI] + k[3171]*y[IDX_D2OI] + k[3176]*y[IDX_DCOI];
    data[2018] = 0.0 - k[3082]*y[IDX_HM] - k[3083]*y[IDX_HM] -
        k[3084]*y[IDX_HM] - k[3085]*y[IDX_HM] - k[3086]*y[IDX_HM] -
        k[3087]*y[IDX_DM] - k[3088]*y[IDX_DM] - k[3108]*y[IDX_HM] -
        k[3109]*y[IDX_HM] - k[3110]*y[IDX_HM] - k[3111]*y[IDX_DM] -
        k[3112]*y[IDX_DM] - k[3289]*y[IDX_CI] - k[3290]*y[IDX_CI] -
        k[3296]*y[IDX_CHI] - k[3297]*y[IDX_CHI] - k[3305]*y[IDX_CDI] -
        k[3340]*y[IDX_CM] - k[3345]*y[IDX_OM] - k[3350]*y[IDX_CNM] -
        k[3368]*y[IDX_OHM] - k[3369]*y[IDX_OHM] - k[3370]*y[IDX_OHM] -
        k[3371]*y[IDX_ODM] - k[3380]*y[IDX_HI] - k[3381]*y[IDX_HI] -
        k[3382]*y[IDX_HI] - k[3394]*y[IDX_DI] - k[3395]*y[IDX_DI] -
        k[3406]*y[IDX_oH2I] + k[3406]*y[IDX_oH2I] - k[3407]*y[IDX_pH2I] +
        k[3407]*y[IDX_pH2I] - k[3408]*y[IDX_oH2I] - k[3409]*y[IDX_pH2I] -
        k[3410]*y[IDX_oH2I] - k[3411]*y[IDX_pH2I] - k[3420]*y[IDX_HDI] +
        k[3420]*y[IDX_HDI] - k[3421]*y[IDX_HDI] - k[3438]*y[IDX_oD2I] +
        k[3438]*y[IDX_oD2I] - k[3439]*y[IDX_pD2I] + k[3439]*y[IDX_pD2I] -
        k[3444]*y[IDX_eM] - k[3449]*y[IDX_eM] - k[3456]*y[IDX_eM] -
        k[3457]*y[IDX_eM] - k[3464]*y[IDX_eM] - k[3465]*y[IDX_eM];
    data[2019] = 0.0 + k[3117]*y[IDX_D2OI];
    data[2020] = 0.0 + k[3176]*y[IDX_D2OII];
    data[2021] = 0.0 + k[3122]*y[IDX_D2OI];
    data[2022] = 0.0 + k[3272]*y[IDX_D2OI];
    data[2023] = 0.0 + k[3277]*y[IDX_D2OI];
    data[2024] = 0.0 - k[3380]*y[IDX_D3OII] - k[3381]*y[IDX_D3OII] -
        k[3382]*y[IDX_D3OII];
    data[2025] = 0.0 - k[3082]*y[IDX_D3OII] - k[3083]*y[IDX_D3OII] -
        k[3084]*y[IDX_D3OII] - k[3085]*y[IDX_D3OII] - k[3086]*y[IDX_D3OII] -
        k[3108]*y[IDX_D3OII] - k[3109]*y[IDX_D3OII] - k[3110]*y[IDX_D3OII];
    data[2026] = 0.0 + k[3428]*y[IDX_oD2I] + k[3429]*y[IDX_pD2I];
    data[2027] = 0.0 + k[3191]*y[IDX_pD3II] + k[3192]*y[IDX_mD3II] +
        k[3193]*y[IDX_oD3II] + k[3194]*y[IDX_pD3II] + k[3195]*y[IDX_oD3II] +
        k[3196]*y[IDX_mD3II];
    data[2028] = 0.0 + k[3135]*y[IDX_D2OII] + k[3417]*y[IDX_HD2OII] -
        k[3420]*y[IDX_D3OII] + k[3420]*y[IDX_D3OII] - k[3421]*y[IDX_D3OII];
    data[2029] = 0.0 + k[3051]*y[IDX_D2OI];
    data[2030] = 0.0 + k[3417]*y[IDX_HDI] + k[3434]*y[IDX_oD2I] +
        k[3435]*y[IDX_pD2I];
    data[2031] = 0.0 + k[3043]*y[IDX_pD2II] + k[3044]*y[IDX_oD2II] +
        k[3165]*y[IDX_D2OII] + k[3220]*y[IDX_pD2HII] + k[3221]*y[IDX_pD2HII] +
        k[3222]*y[IDX_oD2HII] + k[3223]*y[IDX_oD2HII] + k[3230]*y[IDX_mD3II] +
        k[3231]*y[IDX_pD3II] + k[3232]*y[IDX_oD3II] + k[3329]*y[IDX_ND2II];
    data[2032] = 0.0 + k[3141]*y[IDX_pD2I] + k[3142]*y[IDX_oD2I] +
        k[3169]*y[IDX_D2OI];
    data[2033] = 0.0 + k[3282]*y[IDX_D2OI];
    data[2034] = 0.0 + k[3151]*y[IDX_D2OII];
    data[2035] = 0.0 + k[3315]*y[IDX_D2OI];
    data[2036] = 0.0 + k[3329]*y[IDX_HDOI] + k[3335]*y[IDX_D2OI];
    data[2037] = 0.0 + k[3333]*y[IDX_D2OI];
    data[2038] = 0.0 - k[3345]*y[IDX_D3OII];
    data[2039] = 0.0 + k[3156]*y[IDX_D2OII];
    data[2040] = 0.0 + k[3320]*y[IDX_D2OI];
    data[2041] = 0.0 - k[3371]*y[IDX_D3OII];
    data[2042] = 0.0 - k[3368]*y[IDX_D3OII] - k[3369]*y[IDX_D3OII] -
        k[3370]*y[IDX_D3OII];
    data[2043] = 0.0 - k[3444]*y[IDX_D3OII] - k[3449]*y[IDX_D3OII] -
        k[3456]*y[IDX_D3OII] - k[3457]*y[IDX_D3OII] - k[3464]*y[IDX_D3OII] -
        k[3465]*y[IDX_D3OII];
    data[2044] = 0.0 + k[3192]*y[IDX_H2OI] + k[3196]*y[IDX_H2OI] +
        k[3230]*y[IDX_HDOI] + k[3264]*y[IDX_D2OI] + k[3265]*y[IDX_D2OI];
    data[2045] = 0.0 + k[3142]*y[IDX_HDOII] + k[3145]*y[IDX_D2OII] +
        k[3428]*y[IDX_H2DOII] + k[3434]*y[IDX_HD2OII] - k[3438]*y[IDX_D3OII] +
        k[3438]*y[IDX_D3OII];
    data[2046] = 0.0 + k[3044]*y[IDX_HDOI] + k[3053]*y[IDX_D2OI];
    data[2047] = 0.0 + k[3222]*y[IDX_HDOI] + k[3223]*y[IDX_HDOI] +
        k[3257]*y[IDX_D2OI];
    data[2048] = 0.0 + k[3193]*y[IDX_H2OI] + k[3195]*y[IDX_H2OI] +
        k[3232]*y[IDX_HDOI] + k[3266]*y[IDX_D2OI] + k[3267]*y[IDX_D2OI];
    data[2049] = 0.0 - k[3406]*y[IDX_D3OII] + k[3406]*y[IDX_D3OII] -
        k[3408]*y[IDX_D3OII] - k[3410]*y[IDX_D3OII];
    data[2050] = 0.0 + k[3249]*y[IDX_D2OI];
    data[2051] = 0.0 + k[3141]*y[IDX_HDOII] + k[3146]*y[IDX_D2OII] +
        k[3429]*y[IDX_H2DOII] + k[3435]*y[IDX_HD2OII] - k[3439]*y[IDX_D3OII] +
        k[3439]*y[IDX_D3OII];
    data[2052] = 0.0 + k[3043]*y[IDX_HDOI] + k[3054]*y[IDX_D2OI];
    data[2053] = 0.0 + k[3220]*y[IDX_HDOI] + k[3221]*y[IDX_HDOI] +
        k[3256]*y[IDX_D2OI];
    data[2054] = 0.0 + k[3191]*y[IDX_H2OI] + k[3194]*y[IDX_H2OI] +
        k[3231]*y[IDX_HDOI] + k[3262]*y[IDX_D2OI] + k[3263]*y[IDX_D2OI];
    data[2055] = 0.0 - k[3407]*y[IDX_D3OII] + k[3407]*y[IDX_D3OII] -
        k[3409]*y[IDX_D3OII] - k[3411]*y[IDX_D3OII];
    data[2056] = 0.0 + k[3248]*y[IDX_D2OI];
    data[2057] = 0.0 + k[1050]*y[IDX_ND2I] + k[1051]*y[IDX_NHDI];
    data[2058] = 0.0 - k[444]*y[IDX_DCNI] - k[446]*y[IDX_DCNI];
    data[2059] = 0.0 + k[2683]*y[IDX_NDI];
    data[2060] = 0.0 + k[961]*y[IDX_D2OI] + k[963]*y[IDX_HDOI];
    data[2061] = 0.0 + k[538]*y[IDX_NOI];
    data[2062] = 0.0 - k[1790]*y[IDX_DCNI];
    data[2063] = 0.0 + k[1207]*y[IDX_NI];
    data[2064] = 0.0 - k[1789]*y[IDX_DCNI];
    data[2065] = 0.0 + k[1208]*y[IDX_NI];
    data[2066] = 0.0 + k[550]*y[IDX_DCOI] + k[554]*y[IDX_ODI] +
        k[556]*y[IDX_DNOI] + k[2704]*y[IDX_DM];
    data[2067] = 0.0 - k[2320]*y[IDX_DCNI];
    data[2068] = 0.0 + k[2729]*y[IDX_DI];
    data[2069] = 0.0 + k[981]*y[IDX_D2OI] + k[983]*y[IDX_HDOI];
    data[2070] = 0.0 - k[2090]*y[IDX_DCNI];
    data[2071] = 0.0 - k[1095]*y[IDX_DCNI] + k[2172]*y[IDX_DCNII] +
        k[2729]*y[IDX_CNM];
    data[2072] = 0.0 + k[1681]*y[IDX_HNCI] + k[1684]*y[IDX_DNCI] -
        k[2218]*y[IDX_DCNI];
    data[2073] = 0.0 - k[496]*y[IDX_DCNI] + k[2704]*y[IDX_CNI];
    data[2074] = 0.0 + k[961]*y[IDX_C2NII] + k[981]*y[IDX_CNCII] +
        k[2383]*y[IDX_DCNII];
    data[2075] = 0.0 - k[258] - k[401] - k[444]*y[IDX_CII] -
        k[446]*y[IDX_CII] - k[495]*y[IDX_HM] - k[496]*y[IDX_DM] -
        k[504]*y[IDX_OM] - k[507]*y[IDX_OHM] - k[508]*y[IDX_ODM] -
        k[520]*y[IDX_OI] - k[912]*y[IDX_HeII] - k[914]*y[IDX_HeII] -
        k[916]*y[IDX_HeII] - k[918]*y[IDX_HeII] - k[1093]*y[IDX_HI] -
        k[1095]*y[IDX_DI] - k[1671]*y[IDX_OII] - k[1673]*y[IDX_OII] -
        k[1675]*y[IDX_OII] - k[1789]*y[IDX_CHII] - k[1790]*y[IDX_CDII] -
        k[2090]*y[IDX_COII] - k[2108]*y[IDX_pH2II] - k[2109]*y[IDX_oH2II] -
        k[2110]*y[IDX_pD2II] - k[2111]*y[IDX_oD2II] - k[2112]*y[IDX_HDII] -
        k[2217]*y[IDX_HII] - k[2218]*y[IDX_DII] - k[2235]*y[IDX_NII] -
        k[2289]*y[IDX_N2II] - k[2320]*y[IDX_CNII];
    data[2076] = 0.0 + k[2170]*y[IDX_HI] + k[2172]*y[IDX_DI] +
        k[2174]*y[IDX_OI] + k[2176]*y[IDX_NOI] + k[2178]*y[IDX_O2I] +
        k[2381]*y[IDX_H2OI] + k[2383]*y[IDX_D2OI] + k[2385]*y[IDX_HDOI];
    data[2077] = 0.0 + k[550]*y[IDX_CNI];
    data[2078] = 0.0 + k[1683]*y[IDX_HII] + k[1684]*y[IDX_DII];
    data[2079] = 0.0 + k[556]*y[IDX_CNI];
    data[2080] = 0.0 - k[1093]*y[IDX_DCNI] + k[2170]*y[IDX_DCNII];
    data[2081] = 0.0 + k[1683]*y[IDX_DNCI] - k[2217]*y[IDX_DCNI];
    data[2082] = 0.0 - k[495]*y[IDX_DCNI];
    data[2083] = 0.0 + k[2381]*y[IDX_DCNII];
    data[2084] = 0.0 - k[2112]*y[IDX_DCNI];
    data[2085] = 0.0 + k[963]*y[IDX_C2NII] + k[983]*y[IDX_CNCII] +
        k[2385]*y[IDX_DCNII];
    data[2086] = 0.0 + k[1681]*y[IDX_DII];
    data[2087] = 0.0 - k[912]*y[IDX_DCNI] - k[914]*y[IDX_DCNI] -
        k[916]*y[IDX_DCNI] - k[918]*y[IDX_DCNI];
    data[2088] = 0.0 + k[1207]*y[IDX_CD2I] + k[1208]*y[IDX_CHDI];
    data[2089] = 0.0 - k[2235]*y[IDX_DCNI];
    data[2090] = 0.0 - k[2289]*y[IDX_DCNI];
    data[2091] = 0.0 + k[2683]*y[IDX_CM];
    data[2092] = 0.0 + k[1050]*y[IDX_CI];
    data[2093] = 0.0 + k[1051]*y[IDX_CI];
    data[2094] = 0.0 + k[538]*y[IDX_CDI] + k[2176]*y[IDX_DCNII];
    data[2095] = 0.0 - k[520]*y[IDX_DCNI] + k[2174]*y[IDX_DCNII];
    data[2096] = 0.0 - k[1671]*y[IDX_DCNI] - k[1673]*y[IDX_DCNI] -
        k[1675]*y[IDX_DCNI];
    data[2097] = 0.0 - k[504]*y[IDX_DCNI];
    data[2098] = 0.0 + k[2178]*y[IDX_DCNII];
    data[2099] = 0.0 + k[554]*y[IDX_CNI];
    data[2100] = 0.0 - k[508]*y[IDX_DCNI];
    data[2101] = 0.0 - k[507]*y[IDX_DCNI];
    data[2102] = 0.0 - k[2111]*y[IDX_DCNI];
    data[2103] = 0.0 - k[2109]*y[IDX_DCNI];
    data[2104] = 0.0 - k[2110]*y[IDX_DCNI];
    data[2105] = 0.0 - k[2108]*y[IDX_DCNI];
    data[2106] = 0.0 - k[999]*y[IDX_DCNII];
    data[2107] = 0.0 + k[452]*y[IDX_ND2I] + k[453]*y[IDX_NHDI];
    data[2108] = 0.0 - k[1001]*y[IDX_DCNII] + k[1829]*y[IDX_NDII];
    data[2109] = 0.0 - k[1005]*y[IDX_DCNII];
    data[2110] = 0.0 + k[1796]*y[IDX_NH2I] + k[1800]*y[IDX_ND2I] +
        k[1804]*y[IDX_NHDI];
    data[2111] = 0.0 + k[969]*y[IDX_NI];
    data[2112] = 0.0 - k[1003]*y[IDX_DCNII];
    data[2113] = 0.0 + k[1799]*y[IDX_ND2I] + k[1801]*y[IDX_NHDI];
    data[2114] = 0.0 + k[970]*y[IDX_NI];
    data[2115] = 0.0 + k[672]*y[IDX_pD2II] + k[673]*y[IDX_oD2II] +
        k[674]*y[IDX_HDII] + k[1360]*y[IDX_oD3II] + k[1361]*y[IDX_mD3II] +
        k[1362]*y[IDX_oH2DII] + k[1363]*y[IDX_pH2DII] + k[1364]*y[IDX_pH2DII] +
        k[1370]*y[IDX_oD2HII] + k[1371]*y[IDX_pD2HII] + k[1835]*y[IDX_NDII] +
        k[1934]*y[IDX_ODII] + k[2015]*y[IDX_O2DII] + k[2965]*y[IDX_pD3II] +
        k[2966]*y[IDX_pD3II];
    data[2116] = 0.0 + k[476]*y[IDX_pD2I] + k[477]*y[IDX_oD2I] +
        k[478]*y[IDX_HDI] + k[600]*y[IDX_D2OI] + k[601]*y[IDX_HDOI] +
        k[608]*y[IDX_DCOI] + k[2320]*y[IDX_DCNI];
    data[2117] = 0.0 - k[1007]*y[IDX_DCNII];
    data[2118] = 0.0 + k[2090]*y[IDX_DCNI];
    data[2119] = 0.0 - k[2172]*y[IDX_DCNII];
    data[2120] = 0.0 + k[2218]*y[IDX_DCNI];
    data[2121] = 0.0 + k[600]*y[IDX_CNII] - k[2383]*y[IDX_DCNII] -
        k[3117]*y[IDX_DCNII];
    data[2122] = 0.0 + k[2090]*y[IDX_COII] + k[2108]*y[IDX_pH2II] +
        k[2109]*y[IDX_oH2II] + k[2110]*y[IDX_pD2II] + k[2111]*y[IDX_oD2II] +
        k[2112]*y[IDX_HDII] + k[2217]*y[IDX_HII] + k[2218]*y[IDX_DII] +
        k[2235]*y[IDX_NII] + k[2289]*y[IDX_N2II] + k[2320]*y[IDX_CNII];
    data[2123] = 0.0 - k[999]*y[IDX_CI] - k[1001]*y[IDX_C2I] -
        k[1003]*y[IDX_CHI] - k[1005]*y[IDX_CDI] - k[1007]*y[IDX_COI] -
        k[1009]*y[IDX_NHI] - k[1011]*y[IDX_NDI] - k[1013]*y[IDX_OHI] -
        k[1015]*y[IDX_ODI] - k[2170]*y[IDX_HI] - k[2172]*y[IDX_DI] -
        k[2174]*y[IDX_OI] - k[2176]*y[IDX_NOI] - k[2178]*y[IDX_O2I] -
        k[2381]*y[IDX_H2OI] - k[2383]*y[IDX_D2OI] - k[2385]*y[IDX_HDOI] -
        k[2821]*y[IDX_eM] - k[3113]*y[IDX_H2OI] - k[3115]*y[IDX_HDOI] -
        k[3117]*y[IDX_D2OI];
    data[2124] = 0.0 + k[608]*y[IDX_CNII];
    data[2125] = 0.0 - k[2170]*y[IDX_DCNII];
    data[2126] = 0.0 + k[2217]*y[IDX_DCNI];
    data[2127] = 0.0 - k[2381]*y[IDX_DCNII] - k[3113]*y[IDX_DCNII];
    data[2128] = 0.0 + k[478]*y[IDX_CNII];
    data[2129] = 0.0 + k[674]*y[IDX_CNI] + k[2112]*y[IDX_DCNI];
    data[2130] = 0.0 + k[601]*y[IDX_CNII] - k[2385]*y[IDX_DCNII] -
        k[3115]*y[IDX_DCNII];
    data[2131] = 0.0 + k[969]*y[IDX_CD2II] + k[970]*y[IDX_CHDII];
    data[2132] = 0.0 + k[2235]*y[IDX_DCNI];
    data[2133] = 0.0 + k[2289]*y[IDX_DCNI];
    data[2134] = 0.0 - k[1011]*y[IDX_DCNII];
    data[2135] = 0.0 + k[1829]*y[IDX_C2I] + k[1835]*y[IDX_CNI];
    data[2136] = 0.0 + k[452]*y[IDX_CII] + k[1799]*y[IDX_CHII] +
        k[1800]*y[IDX_CDII];
    data[2137] = 0.0 - k[1009]*y[IDX_DCNII];
    data[2138] = 0.0 + k[1796]*y[IDX_CDII];
    data[2139] = 0.0 + k[453]*y[IDX_CII] + k[1801]*y[IDX_CHII] +
        k[1804]*y[IDX_CDII];
    data[2140] = 0.0 - k[2176]*y[IDX_DCNII];
    data[2141] = 0.0 - k[2174]*y[IDX_DCNII];
    data[2142] = 0.0 - k[2178]*y[IDX_DCNII];
    data[2143] = 0.0 + k[2015]*y[IDX_CNI];
    data[2144] = 0.0 - k[1015]*y[IDX_DCNII];
    data[2145] = 0.0 + k[1934]*y[IDX_CNI];
    data[2146] = 0.0 - k[1013]*y[IDX_DCNII];
    data[2147] = 0.0 - k[2821]*y[IDX_DCNII];
    data[2148] = 0.0 + k[1361]*y[IDX_CNI];
    data[2149] = 0.0 + k[477]*y[IDX_CNII];
    data[2150] = 0.0 + k[673]*y[IDX_CNI] + k[2111]*y[IDX_DCNI];
    data[2151] = 0.0 + k[1370]*y[IDX_CNI];
    data[2152] = 0.0 + k[1360]*y[IDX_CNI];
    data[2153] = 0.0 + k[2109]*y[IDX_DCNI];
    data[2154] = 0.0 + k[1362]*y[IDX_CNI];
    data[2155] = 0.0 + k[476]*y[IDX_CNII];
    data[2156] = 0.0 + k[672]*y[IDX_CNI] + k[2110]*y[IDX_DCNI];
    data[2157] = 0.0 + k[1371]*y[IDX_CNI];
    data[2158] = 0.0 + k[2965]*y[IDX_CNI] + k[2966]*y[IDX_CNI];
    data[2159] = 0.0 + k[2108]*y[IDX_DCNI];
    data[2160] = 0.0 + k[1363]*y[IDX_CNI] + k[1364]*y[IDX_CNI];
    data[2161] = 0.0 - k[2056]*y[IDX_DCOI] - k[2058]*y[IDX_DCOI] +
        k[2731]*y[IDX_ODM];
    data[2162] = 0.0 - k[448]*y[IDX_DCOI] - k[2315]*y[IDX_DCOI];
    data[2163] = 0.0 + k[2686]*y[IDX_ODI];
    data[2164] = 0.0 - k[1710]*y[IDX_DCOI] - k[2620]*y[IDX_DCOI];
    data[2165] = 0.0 + k[1033]*y[IDX_O2I];
    data[2166] = 0.0 + k[2721]*y[IDX_OM];
    data[2167] = 0.0 + k[1752]*y[IDX_O2I] - k[1794]*y[IDX_DCOI] -
        k[2573]*y[IDX_DCOI];
    data[2168] = 0.0 - k[1793]*y[IDX_DCOI] - k[2572]*y[IDX_DCOI];
    data[2169] = 0.0 - k[550]*y[IDX_DCOI];
    data[2170] = 0.0 - k[608]*y[IDX_DCOI] - k[2322]*y[IDX_DCOI];
    data[2171] = 0.0 + k[2706]*y[IDX_DM];
    data[2172] = 0.0 - k[2092]*y[IDX_DCOI];
    data[2173] = 0.0 + k[1887]*y[IDX_NDII];
    data[2174] = 0.0 - k[1059]*y[IDX_DCOI] - k[1063]*y[IDX_DCOI];
    data[2175] = 0.0 - k[1547]*y[IDX_DCOI] - k[1551]*y[IDX_DCOI] -
        k[2563]*y[IDX_DCOI];
    data[2176] = 0.0 + k[2706]*y[IDX_COI];
    data[2177] = 0.0 - k[2507]*y[IDX_DCOI] - k[3176]*y[IDX_DCOI];
    data[2178] = 0.0 - k[260] - k[262] - k[403] - k[405] - k[448]*y[IDX_CII]
        - k[516]*y[IDX_OI] - k[518]*y[IDX_OI] - k[550]*y[IDX_CNI] -
        k[573]*y[IDX_OHI] - k[575]*y[IDX_ODI] - k[608]*y[IDX_CNII] -
        k[801]*y[IDX_pH2II] - k[802]*y[IDX_oH2II] - k[803]*y[IDX_oH2II] -
        k[804]*y[IDX_pD2II] - k[805]*y[IDX_oD2II] - k[806]*y[IDX_oD2II] -
        k[807]*y[IDX_HDII] - k[808]*y[IDX_HDII] - k[920]*y[IDX_HeII] -
        k[922]*y[IDX_HeII] - k[924]*y[IDX_HeII] - k[1057]*y[IDX_HI] -
        k[1059]*y[IDX_DI] - k[1061]*y[IDX_HI] - k[1063]*y[IDX_DI] -
        k[1215]*y[IDX_NI] - k[1546]*y[IDX_HII] - k[1547]*y[IDX_DII] -
        k[1550]*y[IDX_HII] - k[1551]*y[IDX_DII] - k[1651]*y[IDX_NII] -
        k[1677]*y[IDX_OII] - k[1710]*y[IDX_C2II] - k[1793]*y[IDX_CHII] -
        k[1794]*y[IDX_CDII] - k[1817]*y[IDX_N2II] - k[1920]*y[IDX_O2II] -
        k[1967]*y[IDX_OHII] - k[1968]*y[IDX_ODII] - k[2056]*y[IDX_CI] -
        k[2058]*y[IDX_CI] - k[2092]*y[IDX_COII] - k[2306]*y[IDX_NH2II] -
        k[2307]*y[IDX_ND2II] - k[2308]*y[IDX_NHDII] - k[2315]*y[IDX_CII] -
        k[2322]*y[IDX_CNII] - k[2370]*y[IDX_OII] - k[2463]*y[IDX_pH2II] -
        k[2464]*y[IDX_oH2II] - k[2465]*y[IDX_pD2II] - k[2466]*y[IDX_oD2II] -
        k[2467]*y[IDX_HDII] - k[2506]*y[IDX_H2OII] - k[2507]*y[IDX_D2OII] -
        k[2508]*y[IDX_HDOII] - k[2535]*y[IDX_NII] - k[2562]*y[IDX_HII] -
        k[2563]*y[IDX_DII] - k[2572]*y[IDX_CHII] - k[2573]*y[IDX_CDII] -
        k[2575]*y[IDX_N2II] - k[2600]*y[IDX_OHII] - k[2601]*y[IDX_ODII] -
        k[2620]*y[IDX_C2II] - k[2637]*y[IDX_O2II] - k[2950]*y[IDX_oD2II] -
        k[2951]*y[IDX_pD2II] - k[3174]*y[IDX_H2OII] - k[3175]*y[IDX_HDOII] -
        k[3176]*y[IDX_D2OII];
    data[2179] = 0.0 + k[1565]*y[IDX_O2I];
    data[2180] = 0.0 - k[1057]*y[IDX_DCOI] - k[1061]*y[IDX_DCOI];
    data[2181] = 0.0 - k[1546]*y[IDX_DCOI] - k[1550]*y[IDX_DCOI] -
        k[2562]*y[IDX_DCOI];
    data[2182] = 0.0 - k[2506]*y[IDX_DCOI] - k[3174]*y[IDX_DCOI];
    data[2183] = 0.0 - k[807]*y[IDX_DCOI] - k[808]*y[IDX_DCOI] -
        k[2467]*y[IDX_DCOI];
    data[2184] = 0.0 - k[2508]*y[IDX_DCOI] - k[3175]*y[IDX_DCOI];
    data[2185] = 0.0 - k[920]*y[IDX_DCOI] - k[922]*y[IDX_DCOI] -
        k[924]*y[IDX_DCOI];
    data[2186] = 0.0 - k[1215]*y[IDX_DCOI];
    data[2187] = 0.0 - k[1651]*y[IDX_DCOI] - k[2535]*y[IDX_DCOI];
    data[2188] = 0.0 - k[1817]*y[IDX_DCOI] - k[2575]*y[IDX_DCOI];
    data[2189] = 0.0 + k[1887]*y[IDX_CO2I];
    data[2190] = 0.0 - k[2307]*y[IDX_DCOI];
    data[2191] = 0.0 - k[2306]*y[IDX_DCOI];
    data[2192] = 0.0 - k[2308]*y[IDX_DCOI];
    data[2193] = 0.0 - k[516]*y[IDX_DCOI] - k[518]*y[IDX_DCOI];
    data[2194] = 0.0 - k[1677]*y[IDX_DCOI] - k[2370]*y[IDX_DCOI];
    data[2195] = 0.0 + k[2721]*y[IDX_CDI];
    data[2196] = 0.0 + k[1033]*y[IDX_C2DI] + k[1565]*y[IDX_DNCII] +
        k[1752]*y[IDX_CDII];
    data[2197] = 0.0 - k[1920]*y[IDX_DCOI] - k[2637]*y[IDX_DCOI];
    data[2198] = 0.0 - k[575]*y[IDX_DCOI] + k[2686]*y[IDX_CM];
    data[2199] = 0.0 - k[1968]*y[IDX_DCOI] - k[2601]*y[IDX_DCOI];
    data[2200] = 0.0 + k[2731]*y[IDX_CI];
    data[2201] = 0.0 - k[573]*y[IDX_DCOI];
    data[2202] = 0.0 - k[1967]*y[IDX_DCOI] - k[2600]*y[IDX_DCOI];
    data[2203] = 0.0 - k[805]*y[IDX_DCOI] - k[806]*y[IDX_DCOI] -
        k[2466]*y[IDX_DCOI] - k[2950]*y[IDX_DCOI];
    data[2204] = 0.0 - k[802]*y[IDX_DCOI] - k[803]*y[IDX_DCOI] -
        k[2464]*y[IDX_DCOI];
    data[2205] = 0.0 - k[804]*y[IDX_DCOI] - k[2465]*y[IDX_DCOI] -
        k[2951]*y[IDX_DCOI];
    data[2206] = 0.0 - k[801]*y[IDX_DCOI] - k[2463]*y[IDX_DCOI];
    data[2207] = 0.0 - k[1017]*y[IDX_DCOII] + k[3283]*y[IDX_H2DOII] +
        k[3284]*y[IDX_H2DOII] + k[3286]*y[IDX_HD2OII] + k[3289]*y[IDX_D3OII] +
        k[3290]*y[IDX_D3OII];
    data[2208] = 0.0 + k[436]*y[IDX_D2OI] + k[437]*y[IDX_HDOI] +
        k[2315]*y[IDX_DCOI];
    data[2209] = 0.0 - k[1019]*y[IDX_DCOII];
    data[2210] = 0.0 + k[2620]*y[IDX_DCOI];
    data[2211] = 0.0 + k[617]*y[IDX_COII];
    data[2212] = 0.0 + k[955]*y[IDX_OI];
    data[2213] = 0.0 + k[961]*y[IDX_D2OI] + k[962]*y[IDX_HDOI];
    data[2214] = 0.0 + k[611]*y[IDX_COII] - k[1023]*y[IDX_DCOII] +
        k[1914]*y[IDX_O2II] + k[2045]*y[IDX_OI];
    data[2215] = 0.0 + k[1756]*y[IDX_O2I] + k[1776]*y[IDX_CO2I] +
        k[1778]*y[IDX_H2OI] + k[1782]*y[IDX_D2OI] + k[1786]*y[IDX_HDOI] +
        k[2573]*y[IDX_DCOI];
    data[2216] = 0.0 + k[619]*y[IDX_COII];
    data[2217] = 0.0 + k[973]*y[IDX_OI] + k[977]*y[IDX_O2I];
    data[2218] = 0.0 - k[1021]*y[IDX_DCOII];
    data[2219] = 0.0 + k[1781]*y[IDX_D2OI] + k[1783]*y[IDX_HDOI] +
        k[2572]*y[IDX_DCOI];
    data[2220] = 0.0 + k[620]*y[IDX_COII];
    data[2221] = 0.0 + k[974]*y[IDX_OI] + k[978]*y[IDX_O2I];
    data[2222] = 0.0 + k[604]*y[IDX_D2OI] + k[605]*y[IDX_HDOI] +
        k[2322]*y[IDX_DCOI];
    data[2223] = 0.0 + k[981]*y[IDX_D2OI] + k[982]*y[IDX_HDOI];
    data[2224] = 0.0 + k[678]*y[IDX_pD2II] + k[679]*y[IDX_oD2II] +
        k[680]*y[IDX_HDII] + k[1007]*y[IDX_DCNII] + k[1253]*y[IDX_D2OII] +
        k[1254]*y[IDX_HDOII] + k[1375]*y[IDX_oD3II] + k[1376]*y[IDX_mD3II] +
        k[1377]*y[IDX_oH2DII] + k[1378]*y[IDX_pH2DII] + k[1379]*y[IDX_pH2DII] +
        k[1385]*y[IDX_oD2HII] + k[1386]*y[IDX_pD2HII] + k[1581]*y[IDX_DNOII] +
        k[1591]*y[IDX_DOCII] + k[1625]*y[IDX_N2DII] + k[1837]*y[IDX_NDII] +
        k[1936]*y[IDX_ODII] + k[2017]*y[IDX_O2DII] + k[2912]*y[IDX_pD3II] +
        k[2967]*y[IDX_pD3II];
    data[2225] = 0.0 + k[464]*y[IDX_pD2I] + k[465]*y[IDX_oD2I] +
        k[466]*y[IDX_HDI] + k[611]*y[IDX_CDI] + k[613]*y[IDX_NDI] +
        k[615]*y[IDX_ODI] + k[617]*y[IDX_C2DI] + k[619]*y[IDX_CD2I] +
        k[620]*y[IDX_CHDI] + k[623]*y[IDX_D2OI] + k[624]*y[IDX_HDOI] +
        k[627]*y[IDX_ND2I] + k[628]*y[IDX_NHDI] + k[2092]*y[IDX_DCOI];
    data[2226] = 0.0 + k[1543]*y[IDX_DII] + k[1776]*y[IDX_CDII];
    data[2227] = 0.0 + k[985]*y[IDX_DI];
    data[2228] = 0.0 + k[985]*y[IDX_CO2II];
    data[2229] = 0.0 + k[1543]*y[IDX_CO2I] + k[2563]*y[IDX_DCOI];
    data[2230] = 0.0 - k[861]*y[IDX_DCOII];
    data[2231] = 0.0 + k[436]*y[IDX_CII] + k[604]*y[IDX_CNII] +
        k[623]*y[IDX_COII] + k[961]*y[IDX_C2NII] + k[981]*y[IDX_CNCII] +
        k[1781]*y[IDX_CHII] + k[1782]*y[IDX_CDII] - k[3122]*y[IDX_DCOII];
    data[2232] = 0.0 + k[1253]*y[IDX_COI] + k[2507]*y[IDX_DCOI];
    data[2233] = 0.0 + k[3289]*y[IDX_CI] + k[3290]*y[IDX_CI];
    data[2234] = 0.0 + k[1675]*y[IDX_OII];
    data[2235] = 0.0 + k[1007]*y[IDX_COI];
    data[2236] = 0.0 + k[262] + k[405] + k[2092]*y[IDX_COII] +
        k[2306]*y[IDX_NH2II] + k[2307]*y[IDX_ND2II] + k[2308]*y[IDX_NHDII] +
        k[2315]*y[IDX_CII] + k[2322]*y[IDX_CNII] + k[2370]*y[IDX_OII] +
        k[2463]*y[IDX_pH2II] + k[2464]*y[IDX_oH2II] + k[2465]*y[IDX_pD2II] +
        k[2466]*y[IDX_oD2II] + k[2467]*y[IDX_HDII] + k[2506]*y[IDX_H2OII] +
        k[2507]*y[IDX_D2OII] + k[2508]*y[IDX_HDOII] + k[2535]*y[IDX_NII] +
        k[2562]*y[IDX_HII] + k[2563]*y[IDX_DII] + k[2572]*y[IDX_CHII] +
        k[2573]*y[IDX_CDII] + k[2575]*y[IDX_N2II] + k[2600]*y[IDX_OHII] +
        k[2601]*y[IDX_ODII] + k[2620]*y[IDX_C2II] + k[2637]*y[IDX_O2II];
    data[2237] = 0.0 - k[201]*y[IDX_GRAINM] - k[860]*y[IDX_HM] -
        k[861]*y[IDX_DM] - k[1017]*y[IDX_CI] - k[1019]*y[IDX_C2I] -
        k[1021]*y[IDX_CHI] - k[1023]*y[IDX_CDI] - k[1025]*y[IDX_NHI] -
        k[1027]*y[IDX_NDI] - k[1029]*y[IDX_OHI] - k[1031]*y[IDX_ODI] -
        k[2823]*y[IDX_eM] - k[3118]*y[IDX_H2OI] - k[3120]*y[IDX_HDOI] -
        k[3122]*y[IDX_D2OI];
    data[2238] = 0.0 + k[1581]*y[IDX_COI];
    data[2239] = 0.0 + k[1591]*y[IDX_COI] + k[1595]*y[IDX_pH2I] +
        k[1596]*y[IDX_oH2I] + k[1597]*y[IDX_oH2I] + k[1605]*y[IDX_pD2I] +
        k[1606]*y[IDX_oD2I] + k[1607]*y[IDX_oD2I] + k[1613]*y[IDX_HDI];
    data[2240] = 0.0 - k[201]*y[IDX_DCOII];
    data[2241] = 0.0 + k[2562]*y[IDX_DCOI];
    data[2242] = 0.0 - k[860]*y[IDX_DCOII];
    data[2243] = 0.0 + k[3283]*y[IDX_CI] + k[3284]*y[IDX_CI];
    data[2244] = 0.0 + k[1778]*y[IDX_CDII] - k[3118]*y[IDX_DCOII];
    data[2245] = 0.0 + k[2506]*y[IDX_DCOI];
    data[2246] = 0.0 + k[466]*y[IDX_COII] + k[1608]*y[IDX_HOCII] +
        k[1609]*y[IDX_HOCII] + k[1613]*y[IDX_DOCII];
    data[2247] = 0.0 + k[680]*y[IDX_COI] + k[2467]*y[IDX_DCOI];
    data[2248] = 0.0 + k[3286]*y[IDX_CI];
    data[2249] = 0.0 + k[437]*y[IDX_CII] + k[605]*y[IDX_CNII] +
        k[624]*y[IDX_COII] + k[962]*y[IDX_C2NII] + k[982]*y[IDX_CNCII] +
        k[1783]*y[IDX_CHII] + k[1786]*y[IDX_CDII] - k[3120]*y[IDX_DCOII];
    data[2250] = 0.0 + k[1254]*y[IDX_COI] + k[2508]*y[IDX_DCOI];
    data[2251] = 0.0 + k[1603]*y[IDX_pD2I] + k[1604]*y[IDX_oD2I] +
        k[1608]*y[IDX_HDI] + k[1609]*y[IDX_HDI];
    data[2252] = 0.0 + k[2535]*y[IDX_DCOI];
    data[2253] = 0.0 + k[2575]*y[IDX_DCOI];
    data[2254] = 0.0 + k[1625]*y[IDX_COI];
    data[2255] = 0.0 + k[613]*y[IDX_COII] - k[1027]*y[IDX_DCOII];
    data[2256] = 0.0 + k[1837]*y[IDX_COI];
    data[2257] = 0.0 + k[627]*y[IDX_COII];
    data[2258] = 0.0 + k[2307]*y[IDX_DCOI];
    data[2259] = 0.0 - k[1025]*y[IDX_DCOII];
    data[2260] = 0.0 + k[2306]*y[IDX_DCOI];
    data[2261] = 0.0 + k[628]*y[IDX_COII];
    data[2262] = 0.0 + k[2308]*y[IDX_DCOI];
    data[2263] = 0.0 + k[955]*y[IDX_C2DII] + k[973]*y[IDX_CD2II] +
        k[974]*y[IDX_CHDII] + k[2045]*y[IDX_CDI];
    data[2264] = 0.0 + k[1675]*y[IDX_DCNI] + k[2370]*y[IDX_DCOI];
    data[2265] = 0.0 + k[977]*y[IDX_CD2II] + k[978]*y[IDX_CHDII] +
        k[1756]*y[IDX_CDII];
    data[2266] = 0.0 + k[1914]*y[IDX_CDI] + k[2637]*y[IDX_DCOI];
    data[2267] = 0.0 + k[2017]*y[IDX_COI];
    data[2268] = 0.0 + k[615]*y[IDX_COII] - k[1031]*y[IDX_DCOII];
    data[2269] = 0.0 + k[1936]*y[IDX_COI] + k[2601]*y[IDX_DCOI];
    data[2270] = 0.0 - k[1029]*y[IDX_DCOII];
    data[2271] = 0.0 + k[2600]*y[IDX_DCOI];
    data[2272] = 0.0 - k[2823]*y[IDX_DCOII];
    data[2273] = 0.0 + k[1376]*y[IDX_COI];
    data[2274] = 0.0 + k[465]*y[IDX_COII] + k[1604]*y[IDX_HOCII] +
        k[1606]*y[IDX_DOCII] + k[1607]*y[IDX_DOCII];
    data[2275] = 0.0 + k[679]*y[IDX_COI] + k[2466]*y[IDX_DCOI];
    data[2276] = 0.0 + k[1385]*y[IDX_COI];
    data[2277] = 0.0 + k[1375]*y[IDX_COI];
    data[2278] = 0.0 + k[1596]*y[IDX_DOCII] + k[1597]*y[IDX_DOCII];
    data[2279] = 0.0 + k[2464]*y[IDX_DCOI];
    data[2280] = 0.0 + k[1377]*y[IDX_COI];
    data[2281] = 0.0 + k[464]*y[IDX_COII] + k[1603]*y[IDX_HOCII] +
        k[1605]*y[IDX_DOCII];
    data[2282] = 0.0 + k[678]*y[IDX_COI] + k[2465]*y[IDX_DCOI];
    data[2283] = 0.0 + k[1386]*y[IDX_COI];
    data[2284] = 0.0 + k[2912]*y[IDX_COI] + k[2967]*y[IDX_COI];
    data[2285] = 0.0 + k[1595]*y[IDX_DOCII];
    data[2286] = 0.0 + k[2463]*y[IDX_DCOI];
    data[2287] = 0.0 + k[1378]*y[IDX_COI] + k[1379]*y[IDX_COI];
    data[2288] = 0.0 + k[1046]*y[IDX_ND2I] + k[1047]*y[IDX_NHDI];
    data[2289] = 0.0 - k[450]*y[IDX_DNCI];
    data[2290] = 0.0 + k[1211]*y[IDX_NI];
    data[2291] = 0.0 + k[1212]*y[IDX_NI];
    data[2292] = 0.0 - k[1684]*y[IDX_DNCI];
    data[2293] = 0.0 - k[264] - k[407] - k[450]*y[IDX_CII] -
        k[926]*y[IDX_HeII] - k[928]*y[IDX_HeII] - k[930]*y[IDX_HeII] -
        k[1682]*y[IDX_HII] - k[1683]*y[IDX_HII] - k[1684]*y[IDX_DII];
    data[2294] = 0.0 + k[2220]*y[IDX_NOI];
    data[2295] = 0.0 - k[1682]*y[IDX_DNCI] - k[1683]*y[IDX_DNCI];
    data[2296] = 0.0 - k[926]*y[IDX_DNCI] - k[928]*y[IDX_DNCI] -
        k[930]*y[IDX_DNCI];
    data[2297] = 0.0 + k[1211]*y[IDX_CD2I] + k[1212]*y[IDX_CHDI];
    data[2298] = 0.0 + k[1046]*y[IDX_CI];
    data[2299] = 0.0 + k[1047]*y[IDX_CI];
    data[2300] = 0.0 + k[2220]*y[IDX_DNCII];
    data[2301] = 0.0 - k[1553]*y[IDX_DNCII];
    data[2302] = 0.0 - k[1555]*y[IDX_DNCII];
    data[2303] = 0.0 - k[1559]*y[IDX_DNCII];
    data[2304] = 0.0 - k[1557]*y[IDX_DNCII];
    data[2305] = 0.0 + k[470]*y[IDX_pD2I] + k[471]*y[IDX_oD2I] +
        k[472]*y[IDX_HDI];
    data[2306] = 0.0 - k[3272]*y[IDX_DNCII];
    data[2307] = 0.0 - k[1553]*y[IDX_CI] - k[1555]*y[IDX_C2I] -
        k[1557]*y[IDX_CHI] - k[1559]*y[IDX_CDI] - k[1561]*y[IDX_NHI] -
        k[1563]*y[IDX_NDI] - k[1565]*y[IDX_O2I] - k[1567]*y[IDX_OHI] -
        k[1569]*y[IDX_ODI] - k[2220]*y[IDX_NOI] - k[2825]*y[IDX_eM] -
        k[3268]*y[IDX_H2OI] - k[3270]*y[IDX_HDOI] - k[3272]*y[IDX_D2OI];
    data[2308] = 0.0 - k[3268]*y[IDX_DNCII];
    data[2309] = 0.0 + k[472]*y[IDX_CNII];
    data[2310] = 0.0 - k[3270]*y[IDX_DNCII];
    data[2311] = 0.0 - k[1563]*y[IDX_DNCII];
    data[2312] = 0.0 - k[1561]*y[IDX_DNCII];
    data[2313] = 0.0 - k[2220]*y[IDX_DNCII];
    data[2314] = 0.0 - k[1565]*y[IDX_DNCII];
    data[2315] = 0.0 - k[1569]*y[IDX_DNCII];
    data[2316] = 0.0 - k[1567]*y[IDX_DNCII];
    data[2317] = 0.0 - k[2825]*y[IDX_DNCII];
    data[2318] = 0.0 + k[471]*y[IDX_CNII];
    data[2319] = 0.0 + k[470]*y[IDX_CNII];
    data[2320] = 0.0 - k[544]*y[IDX_DNOI];
    data[2321] = 0.0 - k[542]*y[IDX_DNOI];
    data[2322] = 0.0 - k[556]*y[IDX_DNOI];
    data[2323] = 0.0 - k[560]*y[IDX_DNOI];
    data[2324] = 0.0 - k[1103]*y[IDX_DNOI] - k[1109]*y[IDX_DNOI];
    data[2325] = 0.0 - k[1688]*y[IDX_DNOI];
    data[2326] = 0.0 - k[266] - k[409] - k[522]*y[IDX_OI] -
        k[542]*y[IDX_CHI] - k[544]*y[IDX_CDI] - k[556]*y[IDX_CNI] -
        k[560]*y[IDX_COI] - k[577]*y[IDX_OHI] - k[579]*y[IDX_ODI] -
        k[932]*y[IDX_HeII] - k[934]*y[IDX_HeII] - k[1101]*y[IDX_HI] -
        k[1103]*y[IDX_DI] - k[1105]*y[IDX_HI] - k[1106]*y[IDX_HI] -
        k[1109]*y[IDX_DI] - k[1687]*y[IDX_HII] - k[1688]*y[IDX_DII];
    data[2327] = 0.0 + k[2222]*y[IDX_NOI];
    data[2328] = 0.0 - k[1101]*y[IDX_DNOI] - k[1105]*y[IDX_DNOI] -
        k[1106]*y[IDX_DNOI];
    data[2329] = 0.0 - k[1687]*y[IDX_DNOI];
    data[2330] = 0.0 - k[932]*y[IDX_DNOI] - k[934]*y[IDX_DNOI];
    data[2331] = 0.0 + k[528]*y[IDX_OI];
    data[2332] = 0.0 + k[529]*y[IDX_OI];
    data[2333] = 0.0 + k[2222]*y[IDX_DNOII];
    data[2334] = 0.0 - k[522]*y[IDX_DNOI] + k[528]*y[IDX_ND2I] +
        k[529]*y[IDX_NHDI];
    data[2335] = 0.0 - k[579]*y[IDX_DNOI];
    data[2336] = 0.0 - k[577]*y[IDX_DNOI];
    data[2337] = 0.0 - k[1571]*y[IDX_DNOII];
    data[2338] = 0.0 - k[1575]*y[IDX_DNOII];
    data[2339] = 0.0 - k[1579]*y[IDX_DNOII];
    data[2340] = 0.0 - k[1577]*y[IDX_DNOII];
    data[2341] = 0.0 - k[1581]*y[IDX_DNOII];
    data[2342] = 0.0 + k[1889]*y[IDX_NDII];
    data[2343] = 0.0 + k[1894]*y[IDX_NHII] + k[1895]*y[IDX_NDII] -
        k[3277]*y[IDX_DNOII];
    data[2344] = 0.0 + k[992]*y[IDX_NI];
    data[2345] = 0.0 + k[266];
    data[2346] = 0.0 - k[1571]*y[IDX_CI] - k[1573]*y[IDX_OI] -
        k[1575]*y[IDX_C2I] - k[1577]*y[IDX_CHI] - k[1579]*y[IDX_CDI] -
        k[1581]*y[IDX_COI] - k[1583]*y[IDX_NHI] - k[1585]*y[IDX_NDI] -
        k[1587]*y[IDX_OHI] - k[1589]*y[IDX_ODI] - k[2222]*y[IDX_NOI] -
        k[2827]*y[IDX_eM] - k[3273]*y[IDX_H2OI] - k[3275]*y[IDX_HDOI] -
        k[3277]*y[IDX_D2OI];
    data[2347] = 0.0 + k[1891]*y[IDX_NDII] - k[3273]*y[IDX_DNOII];
    data[2348] = 0.0 + k[766]*y[IDX_NOI];
    data[2349] = 0.0 + k[1896]*y[IDX_NHII] + k[1899]*y[IDX_NDII] -
        k[3275]*y[IDX_DNOII];
    data[2350] = 0.0 + k[993]*y[IDX_NI];
    data[2351] = 0.0 + k[992]*y[IDX_D2OII] + k[993]*y[IDX_HDOII];
    data[2352] = 0.0 - k[1585]*y[IDX_DNOII] + k[1916]*y[IDX_O2II];
    data[2353] = 0.0 + k[1889]*y[IDX_CO2I] + k[1891]*y[IDX_H2OI] +
        k[1895]*y[IDX_D2OI] + k[1899]*y[IDX_HDOI];
    data[2354] = 0.0 + k[1974]*y[IDX_OI] + k[1992]*y[IDX_O2I];
    data[2355] = 0.0 - k[1583]*y[IDX_DNOII];
    data[2356] = 0.0 + k[1894]*y[IDX_D2OI] + k[1896]*y[IDX_HDOI];
    data[2357] = 0.0 + k[1975]*y[IDX_OI] + k[1993]*y[IDX_O2I];
    data[2358] = 0.0 + k[764]*y[IDX_pD2II] + k[765]*y[IDX_oD2II] +
        k[766]*y[IDX_HDII] + k[1452]*y[IDX_oD3II] + k[1453]*y[IDX_mD3II] +
        k[1454]*y[IDX_oH2DII] + k[1455]*y[IDX_pH2DII] + k[1456]*y[IDX_pH2DII] +
        k[1462]*y[IDX_oD2HII] + k[1463]*y[IDX_pD2HII] + k[1960]*y[IDX_ODII] +
        k[2039]*y[IDX_O2DII] - k[2222]*y[IDX_DNOII] + k[2973]*y[IDX_pD3II] +
        k[2974]*y[IDX_pD3II];
    data[2359] = 0.0 - k[1573]*y[IDX_DNOII] + k[1974]*y[IDX_ND2II] +
        k[1975]*y[IDX_NHDII];
    data[2360] = 0.0 + k[1992]*y[IDX_ND2II] + k[1993]*y[IDX_NHDII];
    data[2361] = 0.0 + k[1916]*y[IDX_NDI];
    data[2362] = 0.0 + k[2039]*y[IDX_NOI];
    data[2363] = 0.0 - k[1589]*y[IDX_DNOII];
    data[2364] = 0.0 + k[1960]*y[IDX_NOI];
    data[2365] = 0.0 - k[1587]*y[IDX_DNOII];
    data[2366] = 0.0 - k[2827]*y[IDX_DNOII];
    data[2367] = 0.0 + k[1453]*y[IDX_NOI];
    data[2368] = 0.0 + k[765]*y[IDX_NOI];
    data[2369] = 0.0 + k[1462]*y[IDX_NOI];
    data[2370] = 0.0 + k[1452]*y[IDX_NOI];
    data[2371] = 0.0 + k[1454]*y[IDX_NOI];
    data[2372] = 0.0 + k[764]*y[IDX_NOI];
    data[2373] = 0.0 + k[1463]*y[IDX_NOI];
    data[2374] = 0.0 + k[2973]*y[IDX_NOI] + k[2974]*y[IDX_NOI];
    data[2375] = 0.0 + k[1455]*y[IDX_NOI] + k[1456]*y[IDX_NOI];
    data[2376] = 0.0 + k[440]*y[IDX_D2OI] + k[441]*y[IDX_HDOI];
    data[2377] = 0.0 + k[1390]*y[IDX_oD3II] + k[1391]*y[IDX_mD3II] +
        k[1392]*y[IDX_oH2DII] + k[1393]*y[IDX_pH2DII] + k[1394]*y[IDX_pH2DII] +
        k[1400]*y[IDX_oD2HII] + k[1401]*y[IDX_pD2HII] - k[1591]*y[IDX_DOCII] +
        k[2968]*y[IDX_pD3II] + k[2969]*y[IDX_pD3II];
    data[2378] = 0.0 + k[458]*y[IDX_pD2I] + k[459]*y[IDX_oD2I] +
        k[460]*y[IDX_HDI];
    data[2379] = 0.0 + k[440]*y[IDX_CII];
    data[2380] = 0.0 - k[1591]*y[IDX_COI] - k[1595]*y[IDX_pH2I] -
        k[1596]*y[IDX_oH2I] - k[1597]*y[IDX_oH2I] - k[1598]*y[IDX_pH2I] -
        k[1599]*y[IDX_oH2I] - k[1605]*y[IDX_pD2I] - k[1606]*y[IDX_oD2I] -
        k[1607]*y[IDX_oD2I] - k[1611]*y[IDX_HDI] - k[1612]*y[IDX_HDI] -
        k[1613]*y[IDX_HDI] - k[1615]*y[IDX_N2I] - k[2829]*y[IDX_eM];
    data[2381] = 0.0 + k[460]*y[IDX_COII] - k[1611]*y[IDX_DOCII] -
        k[1612]*y[IDX_DOCII] - k[1613]*y[IDX_DOCII];
    data[2382] = 0.0 + k[441]*y[IDX_CII];
    data[2383] = 0.0 - k[1615]*y[IDX_DOCII];
    data[2384] = 0.0 - k[2829]*y[IDX_DOCII];
    data[2385] = 0.0 + k[1391]*y[IDX_COI];
    data[2386] = 0.0 + k[459]*y[IDX_COII] - k[1606]*y[IDX_DOCII] -
        k[1607]*y[IDX_DOCII];
    data[2387] = 0.0 + k[1400]*y[IDX_COI];
    data[2388] = 0.0 + k[1390]*y[IDX_COI];
    data[2389] = 0.0 - k[1596]*y[IDX_DOCII] - k[1597]*y[IDX_DOCII] -
        k[1599]*y[IDX_DOCII];
    data[2390] = 0.0 + k[1392]*y[IDX_COI];
    data[2391] = 0.0 + k[458]*y[IDX_COII] - k[1605]*y[IDX_DOCII];
    data[2392] = 0.0 + k[1401]*y[IDX_COI];
    data[2393] = 0.0 + k[2968]*y[IDX_COI] + k[2969]*y[IDX_COI];
    data[2394] = 0.0 - k[1595]*y[IDX_DOCII] - k[1598]*y[IDX_DOCII];
    data[2395] = 0.0 + k[1393]*y[IDX_COI] + k[1394]*y[IDX_COI];
    data[2396] = 0.0 - k[2918]*y[IDX_GRAINM];
    data[2397] = 0.0 - k[2929]*y[IDX_GRAINM];
    data[2398] = 0.0 - k[201]*y[IDX_GRAINM];
    data[2399] = 0.0 - k[170]*y[IDX_HII] - k[171]*y[IDX_HDII] -
        k[172]*y[IDX_HDII] - k[173]*y[IDX_oD2II] - k[174]*y[IDX_oD2II] -
        k[175]*y[IDX_pD2II] - k[176]*y[IDX_pD2II] - k[177]*y[IDX_HeII] -
        k[178]*y[IDX_pH3II] - k[179]*y[IDX_pH3II] - k[180]*y[IDX_pH3II] -
        k[181]*y[IDX_oH3II] - k[182]*y[IDX_oH3II] - k[183]*y[IDX_pH2DII] -
        k[184]*y[IDX_pH2DII] - k[185]*y[IDX_pH2DII] - k[186]*y[IDX_oH2DII] -
        k[187]*y[IDX_oH2DII] - k[188]*y[IDX_oH2DII] - k[189]*y[IDX_oD2HII] -
        k[190]*y[IDX_oD2HII] - k[191]*y[IDX_oD2HII] - k[192]*y[IDX_pD2HII] -
        k[193]*y[IDX_pD2HII] - k[194]*y[IDX_pD2HII] - k[195]*y[IDX_mD3II] -
        k[196]*y[IDX_mD3II] - k[197]*y[IDX_oD3II] - k[198]*y[IDX_oD3II] -
        k[199]*y[IDX_oD3II] - k[200]*y[IDX_HCOII] - k[201]*y[IDX_DCOII] -
        k[2916]*y[IDX_pD3II] - k[2917]*y[IDX_pD3II] - k[2918]*y[IDX_CII] -
        k[2919]*y[IDX_NII] - k[2920]*y[IDX_OII] - k[2929]*y[IDX_DII];
    data[2400] = 0.0 + k[169]*y[IDX_eM];
    data[2401] = 0.0 - k[170]*y[IDX_GRAINM];
    data[2402] = 0.0 - k[200]*y[IDX_GRAINM];
    data[2403] = 0.0 - k[171]*y[IDX_GRAINM] - k[172]*y[IDX_GRAINM];
    data[2404] = 0.0 - k[177]*y[IDX_GRAINM];
    data[2405] = 0.0 - k[2919]*y[IDX_GRAINM];
    data[2406] = 0.0 - k[2920]*y[IDX_GRAINM];
    data[2407] = 0.0 + k[169]*y[IDX_GRAIN0I];
    data[2408] = 0.0 - k[195]*y[IDX_GRAINM] - k[196]*y[IDX_GRAINM];
    data[2409] = 0.0 - k[173]*y[IDX_GRAINM] - k[174]*y[IDX_GRAINM];
    data[2410] = 0.0 - k[189]*y[IDX_GRAINM] - k[190]*y[IDX_GRAINM] -
        k[191]*y[IDX_GRAINM];
    data[2411] = 0.0 - k[197]*y[IDX_GRAINM] - k[198]*y[IDX_GRAINM] -
        k[199]*y[IDX_GRAINM];
    data[2412] = 0.0 - k[186]*y[IDX_GRAINM] - k[187]*y[IDX_GRAINM] -
        k[188]*y[IDX_GRAINM];
    data[2413] = 0.0 - k[181]*y[IDX_GRAINM] - k[182]*y[IDX_GRAINM];
    data[2414] = 0.0 - k[175]*y[IDX_GRAINM] - k[176]*y[IDX_GRAINM];
    data[2415] = 0.0 - k[192]*y[IDX_GRAINM] - k[193]*y[IDX_GRAINM] -
        k[194]*y[IDX_GRAINM];
    data[2416] = 0.0 - k[2916]*y[IDX_GRAINM] - k[2917]*y[IDX_GRAINM];
    data[2417] = 0.0 - k[183]*y[IDX_GRAINM] - k[184]*y[IDX_GRAINM] -
        k[185]*y[IDX_GRAINM];
    data[2418] = 0.0 - k[178]*y[IDX_GRAINM] - k[179]*y[IDX_GRAINM] -
        k[180]*y[IDX_GRAINM];
    data[2419] = 0.0 + k[2918]*y[IDX_GRAINM];
    data[2420] = 0.0 + k[2929]*y[IDX_GRAINM];
    data[2421] = 0.0 + k[201]*y[IDX_GRAINM];
    data[2422] = 0.0 + k[170]*y[IDX_HII] + k[171]*y[IDX_HDII] +
        k[172]*y[IDX_HDII] + k[173]*y[IDX_oD2II] + k[174]*y[IDX_oD2II] +
        k[175]*y[IDX_pD2II] + k[176]*y[IDX_pD2II] + k[177]*y[IDX_HeII] +
        k[178]*y[IDX_pH3II] + k[179]*y[IDX_pH3II] + k[180]*y[IDX_pH3II] +
        k[181]*y[IDX_oH3II] + k[182]*y[IDX_oH3II] + k[183]*y[IDX_pH2DII] +
        k[184]*y[IDX_pH2DII] + k[185]*y[IDX_pH2DII] + k[186]*y[IDX_oH2DII] +
        k[187]*y[IDX_oH2DII] + k[188]*y[IDX_oH2DII] + k[189]*y[IDX_oD2HII] +
        k[190]*y[IDX_oD2HII] + k[191]*y[IDX_oD2HII] + k[192]*y[IDX_pD2HII] +
        k[193]*y[IDX_pD2HII] + k[194]*y[IDX_pD2HII] + k[195]*y[IDX_mD3II] +
        k[196]*y[IDX_mD3II] + k[197]*y[IDX_oD3II] + k[198]*y[IDX_oD3II] +
        k[199]*y[IDX_oD3II] + k[200]*y[IDX_HCOII] + k[201]*y[IDX_DCOII] +
        k[2916]*y[IDX_pD3II] + k[2917]*y[IDX_pD3II] + k[2918]*y[IDX_CII] +
        k[2919]*y[IDX_NII] + k[2920]*y[IDX_OII] + k[2929]*y[IDX_DII];
    data[2423] = 0.0 - k[169]*y[IDX_eM];
    data[2424] = 0.0 + k[170]*y[IDX_GRAINM];
    data[2425] = 0.0 + k[200]*y[IDX_GRAINM];
    data[2426] = 0.0 + k[171]*y[IDX_GRAINM] + k[172]*y[IDX_GRAINM];
    data[2427] = 0.0 + k[177]*y[IDX_GRAINM];
    data[2428] = 0.0 + k[2919]*y[IDX_GRAINM];
    data[2429] = 0.0 + k[2920]*y[IDX_GRAINM];
    data[2430] = 0.0 - k[169]*y[IDX_GRAIN0I];
    data[2431] = 0.0 + k[195]*y[IDX_GRAINM] + k[196]*y[IDX_GRAINM];
    data[2432] = 0.0 + k[173]*y[IDX_GRAINM] + k[174]*y[IDX_GRAINM];
    data[2433] = 0.0 + k[189]*y[IDX_GRAINM] + k[190]*y[IDX_GRAINM] +
        k[191]*y[IDX_GRAINM];
    data[2434] = 0.0 + k[197]*y[IDX_GRAINM] + k[198]*y[IDX_GRAINM] +
        k[199]*y[IDX_GRAINM];
    data[2435] = 0.0 + k[186]*y[IDX_GRAINM] + k[187]*y[IDX_GRAINM] +
        k[188]*y[IDX_GRAINM];
    data[2436] = 0.0 + k[181]*y[IDX_GRAINM] + k[182]*y[IDX_GRAINM];
    data[2437] = 0.0 + k[175]*y[IDX_GRAINM] + k[176]*y[IDX_GRAINM];
    data[2438] = 0.0 + k[192]*y[IDX_GRAINM] + k[193]*y[IDX_GRAINM] +
        k[194]*y[IDX_GRAINM];
    data[2439] = 0.0 + k[2916]*y[IDX_GRAINM] + k[2917]*y[IDX_GRAINM];
    data[2440] = 0.0 + k[183]*y[IDX_GRAINM] + k[184]*y[IDX_GRAINM] +
        k[185]*y[IDX_GRAINM];
    data[2441] = 0.0 + k[178]*y[IDX_GRAINM] + k[179]*y[IDX_GRAINM] +
        k[180]*y[IDX_GRAINM];
    data[2442] = 0.0 + k[630]*y[IDX_pH2II] + k[631]*y[IDX_oH2II] +
        k[634]*y[IDX_HDII] + k[948]*y[IDX_C2HII] + k[964]*y[IDX_CH2II] +
        k[966]*y[IDX_CHDII] + k[1037]*y[IDX_CH2I] + k[1039]*y[IDX_CHDI] +
        k[1045]*y[IDX_NH2I] + k[1047]*y[IDX_NHDI] + k[1049]*y[IDX_NH2I] +
        k[1051]*y[IDX_NHDI] + k[1130]*y[IDX_pH2I] + k[1131]*y[IDX_oH2I] +
        k[1134]*y[IDX_HDI] + k[1711]*y[IDX_CHII] + k[2046]*y[IDX_CHI] +
        k[2048]*y[IDX_NHI] + k[2053]*y[IDX_OHI] + k[2057]*y[IDX_HCOI] +
        k[2060]*y[IDX_C2HI] - k[2653]*y[IDX_HI];
    data[2443] = 0.0 + k[419]*y[IDX_CHI] + k[421]*y[IDX_NHI] +
        k[425]*y[IDX_OHI] + k[427]*y[IDX_C2HI] + k[430]*y[IDX_CH2I] +
        k[432]*y[IDX_CHDI] + k[435]*y[IDX_H2OI] + k[437]*y[IDX_HDOI] +
        k[439]*y[IDX_H2OI] + k[441]*y[IDX_HDOI] + k[443]*y[IDX_HCNI] +
        k[445]*y[IDX_HCNI] + k[449]*y[IDX_HNCI] + k[451]*y[IDX_NH2I] +
        k[453]*y[IDX_NHDI] + k[2133]*y[IDX_HM] - k[2638]*y[IDX_HI];
    data[2444] = 0.0 + k[2135]*y[IDX_HII] - k[2671]*y[IDX_HI] +
        k[3017]*y[IDX_H3OII] + k[3336]*y[IDX_H2DOII] + k[3338]*y[IDX_HD2OII];
    data[2445] = 0.0 + k[545]*y[IDX_CHI] + k[648]*y[IDX_pH2II] +
        k[649]*y[IDX_oH2II] + k[652]*y[IDX_HDII] + k[1721]*y[IDX_CHII] +
        k[1826]*y[IDX_NHII] + k[2187]*y[IDX_HII];
    data[2446] = 0.0 + k[1692]*y[IDX_CHI] + k[1694]*y[IDX_pH2I] +
        k[1695]*y[IDX_oH2I] + k[1698]*y[IDX_HDI] + k[1702]*y[IDX_NHI];
    data[2447] = 0.0 + k[2525]*y[IDX_HII];
    data[2448] = 0.0 + k[243] + k[378] + k[427]*y[IDX_CII] +
        k[884]*y[IDX_HeII] + k[1221]*y[IDX_NI] + k[2060]*y[IDX_CI] +
        k[2523]*y[IDX_HII];
    data[2449] = 0.0 + k[298] + k[948]*y[IDX_CI] + k[952]*y[IDX_NI] +
        k[2765]*y[IDX_eM] + k[2769]*y[IDX_eM];
    data[2450] = 0.0 + k[2205]*y[IDX_HII];
    data[2451] = 0.0 + k[2207]*y[IDX_HII];
    data[2452] = 0.0 + k[1530]*y[IDX_HII];
    data[2453] = 0.0 + k[662]*y[IDX_pH2II] + k[663]*y[IDX_oH2II] +
        k[668]*y[IDX_HDII] - k[1065]*y[IDX_HI] + k[1144]*y[IDX_pH2I] +
        k[1145]*y[IDX_oH2I] + k[1150]*y[IDX_HDI] + k[2191]*y[IDX_HII];
    data[2454] = 0.0 - k[1714]*y[IDX_HI] + k[1733]*y[IDX_pH2I] +
        k[1734]*y[IDX_oH2I] + k[1745]*y[IDX_HDI];
    data[2455] = 0.0 - k[1069]*y[IDX_HI] - k[1070]*y[IDX_HI] +
        k[2529]*y[IDX_HII];
    data[2456] = 0.0 + k[282] + k[354] + k[419]*y[IDX_CII] +
        k[545]*y[IDX_C2I] + k[654]*y[IDX_pH2II] + k[655]*y[IDX_oH2II] +
        k[656]*y[IDX_pD2II] + k[657]*y[IDX_oD2II] + k[660]*y[IDX_HDII] +
        k[863]*y[IDX_HeII] - k[1064]*y[IDX_HI] + k[1136]*y[IDX_pH2I] +
        k[1137]*y[IDX_oH2I] + k[1138]*y[IDX_pD2I] + k[1139]*y[IDX_oD2I] +
        k[1142]*y[IDX_HDI] + k[1197]*y[IDX_NI] + k[1227]*y[IDX_OI] +
        k[1634]*y[IDX_NII] + k[1653]*y[IDX_OII] + k[1692]*y[IDX_C2II] +
        k[2046]*y[IDX_CI] + k[2189]*y[IDX_HII];
    data[2457] = 0.0 + k[232] + k[288] + k[1711]*y[IDX_CI] -
        k[1713]*y[IDX_HI] + k[1717]*y[IDX_NI] + k[1719]*y[IDX_OI] +
        k[1721]*y[IDX_C2I] + k[1727]*y[IDX_CNI] + k[1729]*y[IDX_CNI] +
        k[1731]*y[IDX_pH2I] + k[1732]*y[IDX_oH2I] + k[1737]*y[IDX_pD2I] +
        k[1738]*y[IDX_oD2I] + k[1743]*y[IDX_HDI] + k[2741]*y[IDX_eM];
    data[2458] = 0.0 + k[385] + k[430]*y[IDX_CII] + k[509]*y[IDX_OI] +
        k[509]*y[IDX_OI] + k[895]*y[IDX_HeII] + k[1037]*y[IDX_CI] -
        k[1068]*y[IDX_HI] + k[1206]*y[IDX_NI] + k[1210]*y[IDX_NI] +
        k[2527]*y[IDX_HII];
    data[2459] = 0.0 + k[300] + k[964]*y[IDX_CI] + k[968]*y[IDX_NI] +
        k[972]*y[IDX_OI] + k[2778]*y[IDX_eM] + k[2782]*y[IDX_eM] +
        k[2782]*y[IDX_eM];
    data[2460] = 0.0 + k[387] + k[432]*y[IDX_CII] + k[511]*y[IDX_OI] +
        k[897]*y[IDX_HeII] + k[1039]*y[IDX_CI] - k[1071]*y[IDX_HI] -
        k[1072]*y[IDX_HI] + k[1208]*y[IDX_NI] + k[1212]*y[IDX_NI] +
        k[2531]*y[IDX_HII];
    data[2461] = 0.0 + k[302] + k[966]*y[IDX_CI] + k[970]*y[IDX_NI] +
        k[974]*y[IDX_OI] + k[2780]*y[IDX_eM] + k[2784]*y[IDX_eM];
    data[2462] = 0.0 + k[551]*y[IDX_OHI] + k[670]*y[IDX_pH2II] +
        k[671]*y[IDX_oH2II] + k[674]*y[IDX_HDII] + k[1727]*y[IDX_CHII] +
        k[1729]*y[IDX_CHII];
    data[2463] = 0.0 + k[468]*y[IDX_pH2I] + k[469]*y[IDX_oH2I] +
        k[472]*y[IDX_HDI] + k[474]*y[IDX_pH2I] + k[475]*y[IDX_oH2I] +
        k[478]*y[IDX_HDI] - k[2264]*y[IDX_HI];
    data[2464] = 0.0 - k[2728]*y[IDX_HI] + k[3019]*y[IDX_H3OII] +
        k[3346]*y[IDX_H2DOII] + k[3348]*y[IDX_HD2OII];
    data[2465] = 0.0 + k[557]*y[IDX_OHI] + k[676]*y[IDX_pH2II] +
        k[677]*y[IDX_oH2II] + k[680]*y[IDX_HDII] + k[1838]*y[IDX_NHII];
    data[2466] = 0.0 + k[456]*y[IDX_pH2I] + k[457]*y[IDX_oH2I] +
        k[460]*y[IDX_HDI] + k[462]*y[IDX_pH2I] + k[463]*y[IDX_oH2I] +
        k[466]*y[IDX_HDI] - k[2082]*y[IDX_HI];
    data[2467] = 0.0 - k[1124]*y[IDX_HI];
    data[2468] = 0.0 - k[984]*y[IDX_HI] - k[2374]*y[IDX_HI];
    data[2469] = 0.0 - k[166]*y[IDX_HI] + k[2855]*y[IDX_pH2DII] +
        k[2856]*y[IDX_pH2DII] + k[2857]*y[IDX_oH2DII] + k[2858]*y[IDX_oH2DII] +
        k[2863]*y[IDX_HDII] + k[2864]*y[IDX_HDII] + k[2867]*y[IDX_oD2HII] +
        k[2868]*y[IDX_oD2HII] + k[2869]*y[IDX_pD2HII] + k[2881]*y[IDX_HII] +
        k[2890]*y[IDX_pH3II] + k[2891]*y[IDX_pH3II] + k[2892]*y[IDX_oH3II] +
        k[2899]*y[IDX_pH2II];
    data[2470] = 0.0 + k[2139]*y[IDX_HM] - k[2648]*y[IDX_HI] -
        k[2880]*y[IDX_HI];
    data[2471] = 0.0 + k[2138]*y[IDX_HII] - k[2690]*y[IDX_HI] +
        k[3055]*y[IDX_H3OII] + k[3056]*y[IDX_H3OII] + k[3057]*y[IDX_H3OII] +
        k[3065]*y[IDX_H2DOII] + k[3066]*y[IDX_H2DOII] + k[3067]*y[IDX_H2DOII] +
        k[3077]*y[IDX_HD2OII] + k[3078]*y[IDX_HD2OII];
    data[2472] = 0.0 - k[1083]*y[IDX_HI] - k[1084]*y[IDX_HI] +
        k[2211]*y[IDX_HII] + k[3047]*y[IDX_pH2II] + k[3048]*y[IDX_oH2II] +
        k[3051]*y[IDX_HDII];
    data[2473] = 0.0 + k[3127]*y[IDX_pH2I] + k[3128]*y[IDX_oH2I] +
        k[3135]*y[IDX_HDI];
    data[2474] = 0.0 + k[3082]*y[IDX_HM] + k[3083]*y[IDX_HM] +
        k[3368]*y[IDX_OHM] - k[3380]*y[IDX_HI] - k[3381]*y[IDX_HI] -
        k[3382]*y[IDX_HI] + k[3406]*y[IDX_oH2I] + k[3406]*y[IDX_oH2I] +
        k[3407]*y[IDX_pH2I] + k[3407]*y[IDX_pH2I] + k[3408]*y[IDX_oH2I] +
        k[3409]*y[IDX_pH2I] + k[3420]*y[IDX_HDI];
    data[2475] = 0.0 - k[1093]*y[IDX_HI] + k[2217]*y[IDX_HII];
    data[2476] = 0.0 - k[2170]*y[IDX_HI];
    data[2477] = 0.0 - k[1057]*y[IDX_HI] - k[1061]*y[IDX_HI] +
        k[2562]*y[IDX_HII];
    data[2478] = 0.0 - k[1101]*y[IDX_HI] - k[1105]*y[IDX_HI] -
        k[1106]*y[IDX_HI];
    data[2479] = 0.0 + k[170]*y[IDX_HII] + k[171]*y[IDX_HDII] +
        k[178]*y[IDX_pH3II] + k[179]*y[IDX_pH3II] + k[180]*y[IDX_pH3II] +
        k[180]*y[IDX_pH3II] + k[180]*y[IDX_pH3II] + k[181]*y[IDX_oH3II] +
        k[182]*y[IDX_oH3II] + k[182]*y[IDX_oH3II] + k[182]*y[IDX_oH3II] +
        k[184]*y[IDX_pH2DII] + k[185]*y[IDX_pH2DII] + k[185]*y[IDX_pH2DII] +
        k[186]*y[IDX_oH2DII] + k[186]*y[IDX_oH2DII] + k[188]*y[IDX_oH2DII] +
        k[189]*y[IDX_oD2HII] + k[190]*y[IDX_oD2HII] + k[192]*y[IDX_pD2HII] +
        k[194]*y[IDX_pD2HII] + k[200]*y[IDX_HCOII];
    data[2480] = 0.0 - k[164]*y[IDX_HI] - k[164]*y[IDX_HI] -
        k[164]*y[IDX_HI] - k[164]*y[IDX_HI] - k[165]*y[IDX_HI] -
        k[165]*y[IDX_HI] - k[165]*y[IDX_HI] - k[165]*y[IDX_HI] -
        k[166]*y[IDX_DI] - k[203] - k[809]*y[IDX_HeHII] - k[810]*y[IDX_HeDII] -
        k[984]*y[IDX_CO2II] - k[1056]*y[IDX_HCOI] - k[1057]*y[IDX_DCOI] -
        k[1060]*y[IDX_HCOI] - k[1061]*y[IDX_DCOI] - k[1064]*y[IDX_CHI] -
        k[1065]*y[IDX_CDI] - k[1068]*y[IDX_CH2I] - k[1069]*y[IDX_CD2I] -
        k[1070]*y[IDX_CD2I] - k[1071]*y[IDX_CHDI] - k[1072]*y[IDX_CHDI] -
        k[1078]*y[IDX_OHI] - k[1079]*y[IDX_ODI] - k[1082]*y[IDX_H2OI] -
        k[1083]*y[IDX_D2OI] - k[1084]*y[IDX_D2OI] - k[1085]*y[IDX_HDOI] -
        k[1086]*y[IDX_HDOI] - k[1092]*y[IDX_HCNI] - k[1093]*y[IDX_DCNI] -
        k[1096]*y[IDX_NOI] - k[1098]*y[IDX_NOI] - k[1100]*y[IDX_HNOI] -
        k[1101]*y[IDX_DNOI] - k[1104]*y[IDX_HNOI] - k[1105]*y[IDX_DNOI] -
        k[1106]*y[IDX_DNOI] - k[1110]*y[IDX_O2I] - k[1112]*y[IDX_O2HI] -
        k[1113]*y[IDX_O2DI] - k[1116]*y[IDX_O2HI] - k[1117]*y[IDX_O2DI] -
        k[1120]*y[IDX_O2HI] - k[1121]*y[IDX_O2DI] - k[1124]*y[IDX_CO2I] -
        k[1126]*y[IDX_N2OI] - k[1128]*y[IDX_NO2I] - k[1713]*y[IDX_CHII] -
        k[1714]*y[IDX_CDII] - k[1995]*y[IDX_NO2II] - k[2082]*y[IDX_COII] -
        k[2093]*y[IDX_pH2II] - k[2094]*y[IDX_oH2II] - k[2095]*y[IDX_pD2II] -
        k[2096]*y[IDX_oD2II] - k[2097]*y[IDX_HDII] - k[2129]*y[IDX_N2II] -
        k[2160]*y[IDX_HeII] - k[2169]*y[IDX_HCNII] - k[2170]*y[IDX_DCNII] -
        k[2239]*y[IDX_OII] - k[2264]*y[IDX_CNII] - k[2374]*y[IDX_CO2II] -
        k[2638]*y[IDX_CII] - k[2646]*y[IDX_HII] - k[2647]*y[IDX_HII] -
        k[2648]*y[IDX_DII] - k[2653]*y[IDX_CI] - k[2662]*y[IDX_OI] -
        k[2666]*y[IDX_OHI] - k[2667]*y[IDX_ODI] - k[2671]*y[IDX_CM] -
        k[2689]*y[IDX_HM] - k[2690]*y[IDX_DM] - k[2716]*y[IDX_OM] -
        k[2728]*y[IDX_CNM] - k[2732]*y[IDX_OHM] - k[2733]*y[IDX_ODM] -
        k[2737]*y[IDX_eM] - k[2859]*y[IDX_pD2HII] - k[2860]*y[IDX_oD2HII] -
        k[2861]*y[IDX_pD2HII] - k[2862]*y[IDX_oD2HII] - k[2865]*y[IDX_pD2II] -
        k[2866]*y[IDX_oD2II] - k[2870]*y[IDX_oD3II] - k[2871]*y[IDX_mD3II] -
        k[2880]*y[IDX_DII] - k[2893]*y[IDX_pH2DII] - k[2894]*y[IDX_oH2DII] -
        k[2895]*y[IDX_oH2DII] - k[2897]*y[IDX_HDII] - k[2898]*y[IDX_HDII] -
        k[3021]*y[IDX_H3OII] - k[3022]*y[IDX_H3OII] - k[3372]*y[IDX_H2DOII] -
        k[3373]*y[IDX_H2DOII] - k[3374]*y[IDX_H2DOII] - k[3375]*y[IDX_HD2OII] -
        k[3376]*y[IDX_HD2OII] - k[3377]*y[IDX_HD2OII] - k[3378]*y[IDX_HD2OII] -
        k[3379]*y[IDX_HD2OII] - k[3380]*y[IDX_D3OII] - k[3381]*y[IDX_D3OII] -
        k[3382]*y[IDX_D3OII];
    data[2481] = 0.0 + k[170]*y[IDX_GRAINM] + k[1530]*y[IDX_CCOI] +
        k[2135]*y[IDX_CM] + k[2137]*y[IDX_HM] + k[2137]*y[IDX_HM] +
        k[2138]*y[IDX_DM] + k[2185]*y[IDX_OI] + k[2187]*y[IDX_C2I] +
        k[2189]*y[IDX_CHI] + k[2191]*y[IDX_CDI] + k[2193]*y[IDX_NHI] +
        k[2195]*y[IDX_NDI] + k[2197]*y[IDX_NOI] + k[2199]*y[IDX_O2I] +
        k[2201]*y[IDX_OHI] + k[2203]*y[IDX_ODI] + k[2205]*y[IDX_C2NI] +
        k[2207]*y[IDX_C3I] + k[2209]*y[IDX_H2OI] + k[2211]*y[IDX_D2OI] +
        k[2213]*y[IDX_HDOI] + k[2215]*y[IDX_HCNI] + k[2217]*y[IDX_DCNI] +
        k[2253]*y[IDX_NH2I] + k[2255]*y[IDX_ND2I] + k[2257]*y[IDX_NHDI] +
        k[2523]*y[IDX_C2HI] + k[2525]*y[IDX_C2DI] + k[2527]*y[IDX_CH2I] +
        k[2529]*y[IDX_CD2I] + k[2531]*y[IDX_CHDI] + k[2560]*y[IDX_HCOI] +
        k[2562]*y[IDX_DCOI] - k[2646]*y[IDX_HI] - k[2647]*y[IDX_HI] +
        k[2846]*y[IDX_eM] + k[2881]*y[IDX_DI];
    data[2482] = 0.0 + k[345] + k[2133]*y[IDX_CII] + k[2137]*y[IDX_HII] +
        k[2137]*y[IDX_HII] + k[2139]*y[IDX_DII] + k[2142]*y[IDX_HeII] +
        k[2145]*y[IDX_NII] + k[2148]*y[IDX_OII] + k[2150]*y[IDX_pH2II] +
        k[2151]*y[IDX_oH2II] + k[2154]*y[IDX_pD2II] + k[2155]*y[IDX_oD2II] +
        k[2158]*y[IDX_HDII] - k[2689]*y[IDX_HI] + k[2991]*y[IDX_H3OII] +
        k[2992]*y[IDX_H3OII] + k[3060]*y[IDX_H2DOII] + k[3061]*y[IDX_H2DOII] +
        k[3062]*y[IDX_H2DOII] + k[3071]*y[IDX_HD2OII] + k[3072]*y[IDX_HD2OII] +
        k[3073]*y[IDX_HD2OII] + k[3082]*y[IDX_D3OII] + k[3083]*y[IDX_D3OII];
    data[2483] = 0.0 + k[3060]*y[IDX_HM] + k[3061]*y[IDX_HM] +
        k[3062]*y[IDX_HM] + k[3065]*y[IDX_DM] + k[3066]*y[IDX_DM] +
        k[3067]*y[IDX_DM] + k[3336]*y[IDX_CM] + k[3341]*y[IDX_OM] +
        k[3346]*y[IDX_CNM] + k[3354]*y[IDX_OHM] + k[3355]*y[IDX_OHM] +
        k[3357]*y[IDX_ODM] + k[3358]*y[IDX_ODM] - k[3372]*y[IDX_HI] -
        k[3373]*y[IDX_HI] - k[3374]*y[IDX_HI] + k[3396]*y[IDX_pH2I] +
        k[3396]*y[IDX_pH2I] + k[3397]*y[IDX_oH2I] + k[3397]*y[IDX_oH2I] +
        k[3398]*y[IDX_pH2I] + k[3399]*y[IDX_oH2I] + k[3414]*y[IDX_HDI] +
        k[3414]*y[IDX_HDI] + k[3415]*y[IDX_HDI] + k[3428]*y[IDX_oD2I] +
        k[3428]*y[IDX_oD2I] + k[3429]*y[IDX_pD2I] + k[3429]*y[IDX_pD2I] +
        k[3430]*y[IDX_oD2I] + k[3431]*y[IDX_pD2I] + k[3440]*y[IDX_eM] +
        k[3440]*y[IDX_eM] + k[3441]*y[IDX_eM] + k[3445]*y[IDX_eM] +
        k[3458]*y[IDX_eM];
    data[2484] = 0.0 + k[253] + k[393] + k[435]*y[IDX_CII] +
        k[439]*y[IDX_CII] + k[907]*y[IDX_HeII] - k[1082]*y[IDX_HI] +
        k[2209]*y[IDX_HII] + k[2989]*y[IDX_oH2II] + k[2990]*y[IDX_pH2II] +
        k[3031]*y[IDX_HDII] + k[3033]*y[IDX_oD2II] + k[3034]*y[IDX_pD2II];
    data[2485] = 0.0 + k[991]*y[IDX_NI] + k[2791]*y[IDX_eM] +
        k[2795]*y[IDX_eM] + k[2795]*y[IDX_eM] + k[2997]*y[IDX_pH2I] +
        k[2998]*y[IDX_oH2I] + k[3131]*y[IDX_HDI] + k[3137]*y[IDX_oD2I] +
        k[3138]*y[IDX_pD2I];
    data[2486] = 0.0 + k[2991]*y[IDX_HM] + k[2992]*y[IDX_HM] +
        k[3017]*y[IDX_CM] + k[3018]*y[IDX_OM] + k[3019]*y[IDX_CNM] +
        k[3020]*y[IDX_OHM] - k[3021]*y[IDX_HI] - k[3022]*y[IDX_HI] +
        k[3023]*y[IDX_oH2I] + k[3023]*y[IDX_oH2I] + k[3024]*y[IDX_pH2I] +
        k[3024]*y[IDX_pH2I] + k[3025]*y[IDX_eM] + k[3025]*y[IDX_eM] +
        k[3026]*y[IDX_eM] + k[3029]*y[IDX_eM] + k[3030]*y[IDX_eM] +
        k[3055]*y[IDX_DM] + k[3056]*y[IDX_DM] + k[3057]*y[IDX_DM] +
        k[3351]*y[IDX_ODM] + k[3352]*y[IDX_ODM] + k[3412]*y[IDX_HDI] +
        k[3412]*y[IDX_HDI] + k[3413]*y[IDX_HDI] + k[3422]*y[IDX_pD2I] +
        k[3422]*y[IDX_pD2I] + k[3423]*y[IDX_oD2I] + k[3423]*y[IDX_oD2I] +
        k[3424]*y[IDX_pD2I] + k[3425]*y[IDX_oD2I];
    data[2487] = 0.0 + k[257] + k[400] + k[443]*y[IDX_CII] +
        k[445]*y[IDX_CII] + k[519]*y[IDX_OI] + k[911]*y[IDX_HeII] +
        k[917]*y[IDX_HeII] - k[1092]*y[IDX_HI] + k[2215]*y[IDX_HII];
    data[2488] = 0.0 - k[2169]*y[IDX_HI] + k[2820]*y[IDX_eM];
    data[2489] = 0.0 + k[259] + k[402] + k[515]*y[IDX_OI] +
        k[921]*y[IDX_HeII] - k[1056]*y[IDX_HI] - k[1060]*y[IDX_HI] +
        k[1214]*y[IDX_NI] + k[2057]*y[IDX_CI] + k[2560]*y[IDX_HII];
    data[2490] = 0.0 + k[200]*y[IDX_GRAINM] + k[2822]*y[IDX_eM];
    data[2491] = 0.0 + k[212] + k[217] + k[364] + k[460]*y[IDX_COII] +
        k[466]*y[IDX_COII] + k[472]*y[IDX_CNII] + k[478]*y[IDX_CNII] +
        k[501]*y[IDX_OM] + k[726]*y[IDX_pH2II] + k[727]*y[IDX_pH2II] +
        k[728]*y[IDX_oH2II] + k[729]*y[IDX_oH2II] + k[731]*y[IDX_pD2II] +
        k[732]*y[IDX_oD2II] + k[733]*y[IDX_oD2II] + k[736]*y[IDX_HDII] +
        k[737]*y[IDX_HDII] + k[872]*y[IDX_HeII] + k[1134]*y[IDX_CI] +
        k[1142]*y[IDX_CHI] + k[1150]*y[IDX_CDI] + k[1156]*y[IDX_OI] +
        k[1170]*y[IDX_OHI] + k[1172]*y[IDX_ODI] + k[1178]*y[IDX_NI] +
        k[1192]*y[IDX_NHI] + k[1194]*y[IDX_NDI] + k[1640]*y[IDX_NII] +
        k[1660]*y[IDX_OII] + k[1698]*y[IDX_C2II] + k[1743]*y[IDX_CHII] +
        k[1745]*y[IDX_CDII] + k[1810]*y[IDX_N2II] + k[1866]*y[IDX_NHII] +
        k[1868]*y[IDX_NDII] + k[1949]*y[IDX_OHII] + k[1951]*y[IDX_ODII] +
        k[3131]*y[IDX_H2OII] + k[3133]*y[IDX_HDOII] + k[3135]*y[IDX_D2OII] +
        k[3412]*y[IDX_H3OII] + k[3412]*y[IDX_H3OII] + k[3413]*y[IDX_H3OII] +
        k[3414]*y[IDX_H2DOII] + k[3414]*y[IDX_H2DOII] + k[3415]*y[IDX_H2DOII] +
        k[3417]*y[IDX_HD2OII] + k[3417]*y[IDX_HD2OII] + k[3418]*y[IDX_HD2OII] +
        k[3420]*y[IDX_D3OII];
    data[2492] = 0.0 + k[171]*y[IDX_GRAINM] + k[294] + k[634]*y[IDX_CI] +
        k[640]*y[IDX_NI] + k[646]*y[IDX_OI] + k[652]*y[IDX_C2I] +
        k[660]*y[IDX_CHI] + k[668]*y[IDX_CDI] + k[674]*y[IDX_CNI] +
        k[680]*y[IDX_COI] + k[693]*y[IDX_pH2I] + k[694]*y[IDX_pH2I] +
        k[695]*y[IDX_oH2I] + k[696]*y[IDX_oH2I] + k[718]*y[IDX_pD2I] +
        k[719]*y[IDX_pD2I] + k[720]*y[IDX_oD2I] + k[721]*y[IDX_oD2I] +
        k[736]*y[IDX_HDI] + k[737]*y[IDX_HDI] + k[744]*y[IDX_N2I] +
        k[752]*y[IDX_NHI] + k[760]*y[IDX_NDI] + k[766]*y[IDX_NOI] +
        k[772]*y[IDX_O2I] + k[780]*y[IDX_OHI] + k[788]*y[IDX_ODI] -
        k[2097]*y[IDX_HI] + k[2158]*y[IDX_HM] + k[2749]*y[IDX_eM] +
        k[2863]*y[IDX_DI] + k[2864]*y[IDX_DI] - k[2897]*y[IDX_HI] -
        k[2898]*y[IDX_HI] + k[2930]*y[IDX_oD2I] + k[2932]*y[IDX_pD2I] +
        k[3031]*y[IDX_H2OI] + k[3041]*y[IDX_HDOI] + k[3051]*y[IDX_D2OI];
    data[2493] = 0.0 + k[3071]*y[IDX_HM] + k[3072]*y[IDX_HM] +
        k[3073]*y[IDX_HM] + k[3077]*y[IDX_DM] + k[3078]*y[IDX_DM] +
        k[3338]*y[IDX_CM] + k[3343]*y[IDX_OM] + k[3348]*y[IDX_CNM] +
        k[3361]*y[IDX_OHM] + k[3362]*y[IDX_OHM] + k[3365]*y[IDX_ODM] -
        k[3375]*y[IDX_HI] - k[3376]*y[IDX_HI] - k[3377]*y[IDX_HI] -
        k[3378]*y[IDX_HI] - k[3379]*y[IDX_HI] + k[3400]*y[IDX_pH2I] +
        k[3400]*y[IDX_pH2I] + k[3401]*y[IDX_oH2I] + k[3401]*y[IDX_oH2I] +
        k[3402]*y[IDX_pH2I] + k[3403]*y[IDX_oH2I] + k[3417]*y[IDX_HDI] +
        k[3417]*y[IDX_HDI] + k[3418]*y[IDX_HDI] + k[3434]*y[IDX_oD2I] +
        k[3435]*y[IDX_pD2I] + k[3442]*y[IDX_eM] + k[3447]*y[IDX_eM] +
        k[3461]*y[IDX_eM] + k[3462]*y[IDX_eM];
    data[2494] = 0.0 + k[255] + k[395] + k[437]*y[IDX_CII] +
        k[441]*y[IDX_CII] + k[909]*y[IDX_HeII] - k[1085]*y[IDX_HI] -
        k[1086]*y[IDX_HI] + k[2213]*y[IDX_HII] + k[3037]*y[IDX_pH2II] +
        k[3038]*y[IDX_oH2II] + k[3041]*y[IDX_HDII] + k[3043]*y[IDX_pD2II] +
        k[3044]*y[IDX_oD2II];
    data[2495] = 0.0 + k[993]*y[IDX_NI] + k[2793]*y[IDX_eM] +
        k[2797]*y[IDX_eM] + k[3123]*y[IDX_pH2I] + k[3124]*y[IDX_oH2I] +
        k[3133]*y[IDX_HDI] + k[3141]*y[IDX_pD2I] + k[3142]*y[IDX_oD2I];
    data[2496] = 0.0 + k[263] + k[406] + k[449]*y[IDX_CII] +
        k[925]*y[IDX_HeII] + k[927]*y[IDX_HeII];
    data[2497] = 0.0 + k[2824]*y[IDX_eM];
    data[2498] = 0.0 + k[408] + k[933]*y[IDX_HeII] - k[1100]*y[IDX_HI] -
        k[1104]*y[IDX_HI];
    data[2499] = 0.0 + k[1572]*y[IDX_OI] + k[2826]*y[IDX_eM];
    data[2500] = 0.0 + k[2828]*y[IDX_eM];
    data[2501] = 0.0 + k[863]*y[IDX_CHI] + k[868]*y[IDX_pH2I] +
        k[869]*y[IDX_oH2I] + k[872]*y[IDX_HDI] + k[875]*y[IDX_NHI] +
        k[880]*y[IDX_OHI] + k[884]*y[IDX_C2HI] + k[895]*y[IDX_CH2I] +
        k[897]*y[IDX_CHDI] + k[907]*y[IDX_H2OI] + k[909]*y[IDX_HDOI] +
        k[911]*y[IDX_HCNI] + k[917]*y[IDX_HCNI] + k[921]*y[IDX_HCOI] +
        k[925]*y[IDX_HNCI] + k[927]*y[IDX_HNCI] + k[933]*y[IDX_HNOI] +
        k[942]*y[IDX_NH2I] + k[944]*y[IDX_NHDI] + k[2142]*y[IDX_HM] -
        k[2160]*y[IDX_HI];
    data[2502] = 0.0 - k[810]*y[IDX_HI];
    data[2503] = 0.0 - k[809]*y[IDX_HI] + k[2756]*y[IDX_eM];
    data[2504] = 0.0 + k[636]*y[IDX_pH2II] + k[637]*y[IDX_oH2II] +
        k[640]*y[IDX_HDII] + k[952]*y[IDX_C2HII] + k[968]*y[IDX_CH2II] +
        k[970]*y[IDX_CHDII] + k[991]*y[IDX_H2OII] + k[993]*y[IDX_HDOII] +
        k[1174]*y[IDX_pH2I] + k[1175]*y[IDX_oH2I] + k[1178]*y[IDX_HDI] +
        k[1197]*y[IDX_CHI] + k[1200]*y[IDX_NHI] + k[1203]*y[IDX_OHI] +
        k[1206]*y[IDX_CH2I] + k[1208]*y[IDX_CHDI] + k[1210]*y[IDX_CH2I] +
        k[1212]*y[IDX_CHDI] + k[1214]*y[IDX_HCOI] + k[1221]*y[IDX_C2HI] +
        k[1271]*y[IDX_oH3II] + k[1272]*y[IDX_pH3II] + k[1275]*y[IDX_oH2DII] +
        k[1276]*y[IDX_pH2DII] + k[1279]*y[IDX_oD2HII] + k[1280]*y[IDX_pD2HII] +
        k[1717]*y[IDX_CHII] + k[1820]*y[IDX_NHII] + k[1923]*y[IDX_OHII] +
        k[1969]*y[IDX_NH2II] + k[1971]*y[IDX_NHDII] + k[2004]*y[IDX_O2HII];
    data[2505] = 0.0 + k[1634]*y[IDX_CHI] + k[1636]*y[IDX_pH2I] +
        k[1637]*y[IDX_oH2I] + k[1640]*y[IDX_HDI] + k[1642]*y[IDX_NHI] +
        k[1647]*y[IDX_OHI] + k[2145]*y[IDX_HM];
    data[2506] = 0.0 + k[740]*y[IDX_pH2II] + k[741]*y[IDX_oH2II] +
        k[744]*y[IDX_HDII];
    data[2507] = 0.0 + k[1806]*y[IDX_pH2I] + k[1807]*y[IDX_oH2I] +
        k[1810]*y[IDX_HDI] - k[2129]*y[IDX_HI];
    data[2508] = 0.0 + k[2830]*y[IDX_eM];
    data[2509] = 0.0 - k[1126]*y[IDX_HI];
    data[2510] = 0.0 + k[562]*y[IDX_NHI] + k[754]*y[IDX_pH2II] +
        k[755]*y[IDX_oH2II] + k[760]*y[IDX_HDII] + k[1182]*y[IDX_pH2I] +
        k[1183]*y[IDX_oH2I] + k[1194]*y[IDX_HDI] + k[2195]*y[IDX_HII];
    data[2511] = 0.0 + k[1856]*y[IDX_pH2I] + k[1857]*y[IDX_oH2I] +
        k[1868]*y[IDX_HDI];
    data[2512] = 0.0 + k[2255]*y[IDX_HII];
    data[2513] = 0.0 + k[235] + k[366] + k[421]*y[IDX_CII] +
        k[561]*y[IDX_NHI] + k[561]*y[IDX_NHI] + k[561]*y[IDX_NHI] +
        k[561]*y[IDX_NHI] + k[562]*y[IDX_NDI] + k[564]*y[IDX_NOI] +
        k[746]*y[IDX_pH2II] + k[747]*y[IDX_oH2II] + k[748]*y[IDX_pD2II] +
        k[749]*y[IDX_oD2II] + k[752]*y[IDX_HDII] + k[875]*y[IDX_HeII] +
        k[1180]*y[IDX_pH2I] + k[1181]*y[IDX_oH2I] + k[1186]*y[IDX_pD2I] +
        k[1187]*y[IDX_oD2I] + k[1192]*y[IDX_HDI] + k[1200]*y[IDX_NI] +
        k[1230]*y[IDX_OI] + k[1642]*y[IDX_NII] + k[1663]*y[IDX_OII] +
        k[1702]*y[IDX_C2II] + k[1917]*y[IDX_O2II] + k[2048]*y[IDX_CI] +
        k[2193]*y[IDX_HII];
    data[2514] = 0.0 + k[1820]*y[IDX_NI] + k[1826]*y[IDX_C2I] +
        k[1838]*y[IDX_COI] + k[1854]*y[IDX_pH2I] + k[1855]*y[IDX_oH2I] +
        k[1860]*y[IDX_pD2I] + k[1861]*y[IDX_oD2I] + k[1866]*y[IDX_HDI] +
        k[2759]*y[IDX_eM];
    data[2515] = 0.0 + k[268] + k[410] + k[451]*y[IDX_CII] +
        k[527]*y[IDX_OI] + k[942]*y[IDX_HeII] + k[1045]*y[IDX_CI] +
        k[1049]*y[IDX_CI] + k[2253]*y[IDX_HII];
    data[2516] = 0.0 + k[1969]*y[IDX_NI] + k[1973]*y[IDX_OI] +
        k[2835]*y[IDX_eM] + k[2835]*y[IDX_eM] + k[2838]*y[IDX_eM];
    data[2517] = 0.0 + k[270] + k[412] + k[453]*y[IDX_CII] +
        k[529]*y[IDX_OI] + k[944]*y[IDX_HeII] + k[1047]*y[IDX_CI] +
        k[1051]*y[IDX_CI] + k[2257]*y[IDX_HII];
    data[2518] = 0.0 + k[1971]*y[IDX_NI] + k[1975]*y[IDX_OI] +
        k[2837]*y[IDX_eM] + k[2840]*y[IDX_eM];
    data[2519] = 0.0 + k[564]*y[IDX_NHI] + k[762]*y[IDX_pH2II] +
        k[763]*y[IDX_oH2II] + k[766]*y[IDX_HDII] - k[1096]*y[IDX_HI] -
        k[1098]*y[IDX_HI] + k[2197]*y[IDX_HII];
    data[2520] = 0.0 - k[1128]*y[IDX_HI];
    data[2521] = 0.0 - k[1995]*y[IDX_HI];
    data[2522] = 0.0 + k[509]*y[IDX_CH2I] + k[509]*y[IDX_CH2I] +
        k[511]*y[IDX_CHDI] + k[515]*y[IDX_HCOI] + k[519]*y[IDX_HCNI] +
        k[527]*y[IDX_NH2I] + k[529]*y[IDX_NHDI] + k[642]*y[IDX_pH2II] +
        k[643]*y[IDX_oH2II] + k[646]*y[IDX_HDII] + k[972]*y[IDX_CH2II] +
        k[974]*y[IDX_CHDII] + k[1152]*y[IDX_pH2I] + k[1153]*y[IDX_oH2I] +
        k[1156]*y[IDX_HDI] + k[1227]*y[IDX_CHI] + k[1230]*y[IDX_NHI] +
        k[1232]*y[IDX_OHI] + k[1298]*y[IDX_oH3II] + k[1299]*y[IDX_pH3II] +
        k[1302]*y[IDX_oH2DII] + k[1303]*y[IDX_pH2DII] + k[1306]*y[IDX_oD2HII] +
        k[1307]*y[IDX_pD2HII] + k[1572]*y[IDX_HNOII] + k[1719]*y[IDX_CHII] +
        k[1925]*y[IDX_OHII] + k[1973]*y[IDX_NH2II] + k[1975]*y[IDX_NHDII] +
        k[2185]*y[IDX_HII] - k[2662]*y[IDX_HI];
    data[2523] = 0.0 + k[1653]*y[IDX_CHI] + k[1656]*y[IDX_pH2I] +
        k[1657]*y[IDX_oH2I] + k[1660]*y[IDX_HDI] + k[1663]*y[IDX_NHI] +
        k[1665]*y[IDX_OHI] + k[2148]*y[IDX_HM] - k[2239]*y[IDX_HI];
    data[2524] = 0.0 + k[497]*y[IDX_pH2I] + k[498]*y[IDX_oH2I] +
        k[501]*y[IDX_HDI] - k[2716]*y[IDX_HI] + k[3018]*y[IDX_H3OII] +
        k[3341]*y[IDX_H2DOII] + k[3343]*y[IDX_HD2OII];
    data[2525] = 0.0 + k[768]*y[IDX_pH2II] + k[769]*y[IDX_oH2II] +
        k[772]*y[IDX_HDII] - k[1110]*y[IDX_HI] + k[2199]*y[IDX_HII];
    data[2526] = 0.0 + k[1917]*y[IDX_NHI];
    data[2527] = 0.0 - k[1113]*y[IDX_HI] - k[1117]*y[IDX_HI] -
        k[1121]*y[IDX_HI];
    data[2528] = 0.0 + k[278] - k[1112]*y[IDX_HI] - k[1116]*y[IDX_HI] -
        k[1120]*y[IDX_HI];
    data[2529] = 0.0 + k[2004]*y[IDX_NI] + k[2843]*y[IDX_eM];
    data[2530] = 0.0 + k[782]*y[IDX_pH2II] + k[783]*y[IDX_oH2II] +
        k[788]*y[IDX_HDII] - k[1079]*y[IDX_HI] + k[1160]*y[IDX_pH2I] +
        k[1161]*y[IDX_oH2I] + k[1172]*y[IDX_HDI] + k[2203]*y[IDX_HII] -
        k[2667]*y[IDX_HI];
    data[2531] = 0.0 + k[1939]*y[IDX_pH2I] + k[1940]*y[IDX_oH2I] +
        k[1951]*y[IDX_HDI];
    data[2532] = 0.0 - k[2733]*y[IDX_HI] + k[3351]*y[IDX_H3OII] +
        k[3352]*y[IDX_H3OII] + k[3357]*y[IDX_H2DOII] + k[3358]*y[IDX_H2DOII] +
        k[3365]*y[IDX_HD2OII];
    data[2533] = 0.0 + k[241] + k[374] + k[425]*y[IDX_CII] +
        k[551]*y[IDX_CNI] + k[557]*y[IDX_COI] + k[774]*y[IDX_pH2II] +
        k[775]*y[IDX_oH2II] + k[776]*y[IDX_pD2II] + k[777]*y[IDX_oD2II] +
        k[780]*y[IDX_HDII] + k[880]*y[IDX_HeII] - k[1078]*y[IDX_HI] +
        k[1158]*y[IDX_pH2I] + k[1159]*y[IDX_oH2I] + k[1164]*y[IDX_pD2I] +
        k[1165]*y[IDX_oD2I] + k[1170]*y[IDX_HDI] + k[1203]*y[IDX_NI] +
        k[1232]*y[IDX_OI] + k[1647]*y[IDX_NII] + k[1665]*y[IDX_OII] +
        k[2053]*y[IDX_CI] + k[2201]*y[IDX_HII] - k[2666]*y[IDX_HI];
    data[2534] = 0.0 + k[1923]*y[IDX_NI] + k[1925]*y[IDX_OI] +
        k[1937]*y[IDX_pH2I] + k[1938]*y[IDX_oH2I] + k[1943]*y[IDX_pD2I] +
        k[1944]*y[IDX_oD2I] + k[1949]*y[IDX_HDI] + k[2763]*y[IDX_eM];
    data[2535] = 0.0 - k[2732]*y[IDX_HI] + k[3020]*y[IDX_H3OII] +
        k[3354]*y[IDX_H2DOII] + k[3355]*y[IDX_H2DOII] + k[3361]*y[IDX_HD2OII] +
        k[3362]*y[IDX_HD2OII] + k[3368]*y[IDX_D3OII];
    data[2536] = 0.0 - k[2737]*y[IDX_HI] + k[2741]*y[IDX_CHII] +
        k[2745]*y[IDX_pH2II] + k[2745]*y[IDX_pH2II] + k[2746]*y[IDX_oH2II] +
        k[2746]*y[IDX_oH2II] + k[2749]*y[IDX_HDII] + k[2756]*y[IDX_HeHII] +
        k[2759]*y[IDX_NHII] + k[2763]*y[IDX_OHII] + k[2765]*y[IDX_C2HII] +
        k[2769]*y[IDX_C2HII] + k[2778]*y[IDX_CH2II] + k[2780]*y[IDX_CHDII] +
        k[2782]*y[IDX_CH2II] + k[2782]*y[IDX_CH2II] + k[2784]*y[IDX_CHDII] +
        k[2791]*y[IDX_H2OII] + k[2793]*y[IDX_HDOII] + k[2795]*y[IDX_H2OII] +
        k[2795]*y[IDX_H2OII] + k[2797]*y[IDX_HDOII] + k[2798]*y[IDX_oH3II] +
        k[2798]*y[IDX_oH3II] + k[2798]*y[IDX_oH3II] + k[2799]*y[IDX_pH3II] +
        k[2799]*y[IDX_pH3II] + k[2799]*y[IDX_pH3II] + k[2802]*y[IDX_oH2DII] +
        k[2802]*y[IDX_oH2DII] + k[2803]*y[IDX_pH2DII] + k[2803]*y[IDX_pH2DII] +
        k[2804]*y[IDX_oD2HII] + k[2805]*y[IDX_pD2HII] + k[2806]*y[IDX_oH3II] +
        k[2807]*y[IDX_pH3II] + k[2808]*y[IDX_pH3II] + k[2812]*y[IDX_oH2DII] +
        k[2813]*y[IDX_pH2DII] + k[2816]*y[IDX_oD2HII] + k[2817]*y[IDX_pD2HII] +
        k[2820]*y[IDX_HCNII] + k[2822]*y[IDX_HCOII] + k[2824]*y[IDX_HNCII] +
        k[2826]*y[IDX_HNOII] + k[2828]*y[IDX_HOCII] + k[2830]*y[IDX_N2HII] +
        k[2835]*y[IDX_NH2II] + k[2835]*y[IDX_NH2II] + k[2837]*y[IDX_NHDII] +
        k[2838]*y[IDX_NH2II] + k[2840]*y[IDX_NHDII] + k[2843]*y[IDX_O2HII] +
        k[2846]*y[IDX_HII] + k[3025]*y[IDX_H3OII] + k[3025]*y[IDX_H3OII] +
        k[3026]*y[IDX_H3OII] + k[3029]*y[IDX_H3OII] + k[3030]*y[IDX_H3OII] +
        k[3440]*y[IDX_H2DOII] + k[3440]*y[IDX_H2DOII] + k[3441]*y[IDX_H2DOII] +
        k[3442]*y[IDX_HD2OII] + k[3445]*y[IDX_H2DOII] + k[3447]*y[IDX_HD2OII] +
        k[3458]*y[IDX_H2DOII] + k[3461]*y[IDX_HD2OII] + k[3462]*y[IDX_HD2OII];
    data[2537] = 0.0 - k[2871]*y[IDX_HI];
    data[2538] = 0.0 + k[702]*y[IDX_pH2II] + k[703]*y[IDX_pH2II] +
        k[704]*y[IDX_oH2II] + k[705]*y[IDX_oH2II] + k[720]*y[IDX_HDII] +
        k[721]*y[IDX_HDII] + k[1139]*y[IDX_CHI] + k[1165]*y[IDX_OHI] +
        k[1187]*y[IDX_NHI] + k[1738]*y[IDX_CHII] + k[1861]*y[IDX_NHII] +
        k[1944]*y[IDX_OHII] + k[2930]*y[IDX_HDII] + k[3137]*y[IDX_H2OII] +
        k[3142]*y[IDX_HDOII] + k[3423]*y[IDX_H3OII] + k[3423]*y[IDX_H3OII] +
        k[3425]*y[IDX_H3OII] + k[3428]*y[IDX_H2DOII] + k[3428]*y[IDX_H2DOII] +
        k[3430]*y[IDX_H2DOII] + k[3434]*y[IDX_HD2OII];
    data[2539] = 0.0 + k[657]*y[IDX_CHI] + k[686]*y[IDX_pH2I] +
        k[688]*y[IDX_oH2I] + k[732]*y[IDX_HDI] + k[733]*y[IDX_HDI] +
        k[749]*y[IDX_NHI] + k[777]*y[IDX_OHI] - k[2096]*y[IDX_HI] +
        k[2155]*y[IDX_HM] - k[2866]*y[IDX_HI] + k[3033]*y[IDX_H2OI] +
        k[3044]*y[IDX_HDOI];
    data[2540] = 0.0 + k[189]*y[IDX_GRAINM] + k[190]*y[IDX_GRAINM] + k[338]
        + k[339] + k[1279]*y[IDX_NI] + k[1306]*y[IDX_OI] + k[2804]*y[IDX_eM] +
        k[2816]*y[IDX_eM] - k[2860]*y[IDX_HI] - k[2862]*y[IDX_HI] +
        k[2867]*y[IDX_DI] + k[2868]*y[IDX_DI];
    data[2541] = 0.0 - k[2870]*y[IDX_HI];
    data[2542] = 0.0 + k[209] + k[209] + k[214] + k[361] + k[361] +
        k[457]*y[IDX_COII] + k[463]*y[IDX_COII] + k[469]*y[IDX_CNII] +
        k[475]*y[IDX_CNII] + k[498]*y[IDX_OM] + k[683]*y[IDX_pH2II] +
        k[684]*y[IDX_oH2II] + k[687]*y[IDX_pD2II] + k[688]*y[IDX_oD2II] +
        k[695]*y[IDX_HDII] + k[696]*y[IDX_HDII] + k[869]*y[IDX_HeII] +
        k[1131]*y[IDX_CI] + k[1137]*y[IDX_CHI] + k[1145]*y[IDX_CDI] +
        k[1153]*y[IDX_OI] + k[1159]*y[IDX_OHI] + k[1161]*y[IDX_ODI] +
        k[1175]*y[IDX_NI] + k[1181]*y[IDX_NHI] + k[1183]*y[IDX_NDI] +
        k[1637]*y[IDX_NII] + k[1657]*y[IDX_OII] + k[1695]*y[IDX_C2II] +
        k[1732]*y[IDX_CHII] + k[1734]*y[IDX_CDII] + k[1807]*y[IDX_N2II] +
        k[1855]*y[IDX_NHII] + k[1857]*y[IDX_NDII] + k[1938]*y[IDX_OHII] +
        k[1940]*y[IDX_ODII] + k[2876]*y[IDX_oH2II] + k[2877]*y[IDX_pH2II] +
        k[2998]*y[IDX_H2OII] + k[3023]*y[IDX_H3OII] + k[3023]*y[IDX_H3OII] +
        k[3124]*y[IDX_HDOII] + k[3128]*y[IDX_D2OII] + k[3397]*y[IDX_H2DOII] +
        k[3397]*y[IDX_H2DOII] + k[3399]*y[IDX_H2DOII] + k[3401]*y[IDX_HD2OII] +
        k[3401]*y[IDX_HD2OII] + k[3403]*y[IDX_HD2OII] + k[3406]*y[IDX_D3OII] +
        k[3406]*y[IDX_D3OII] + k[3408]*y[IDX_D3OII];
    data[2543] = 0.0 + k[291] + k[631]*y[IDX_CI] + k[637]*y[IDX_NI] +
        k[643]*y[IDX_OI] + k[649]*y[IDX_C2I] + k[655]*y[IDX_CHI] +
        k[663]*y[IDX_CDI] + k[671]*y[IDX_CNI] + k[677]*y[IDX_COI] +
        k[682]*y[IDX_pH2I] + k[684]*y[IDX_oH2I] + k[700]*y[IDX_pD2I] +
        k[701]*y[IDX_pD2I] + k[704]*y[IDX_oD2I] + k[705]*y[IDX_oD2I] +
        k[728]*y[IDX_HDI] + k[729]*y[IDX_HDI] + k[741]*y[IDX_N2I] +
        k[747]*y[IDX_NHI] + k[755]*y[IDX_NDI] + k[763]*y[IDX_NOI] +
        k[769]*y[IDX_O2I] + k[775]*y[IDX_OHI] + k[783]*y[IDX_ODI] -
        k[2094]*y[IDX_HI] + k[2151]*y[IDX_HM] + k[2746]*y[IDX_eM] +
        k[2746]*y[IDX_eM] + k[2876]*y[IDX_oH2I] + k[2878]*y[IDX_pH2I] +
        k[2989]*y[IDX_H2OI] + k[3038]*y[IDX_HDOI] + k[3048]*y[IDX_D2OI];
    data[2544] = 0.0 + k[186]*y[IDX_GRAINM] + k[186]*y[IDX_GRAINM] +
        k[188]*y[IDX_GRAINM] + k[332] + k[1275]*y[IDX_NI] + k[1302]*y[IDX_OI] +
        k[2802]*y[IDX_eM] + k[2802]*y[IDX_eM] + k[2812]*y[IDX_eM] +
        k[2857]*y[IDX_DI] + k[2858]*y[IDX_DI] - k[2894]*y[IDX_HI] -
        k[2895]*y[IDX_HI];
    data[2545] = 0.0 + k[181]*y[IDX_GRAINM] + k[182]*y[IDX_GRAINM] +
        k[182]*y[IDX_GRAINM] + k[182]*y[IDX_GRAINM] + k[324] + k[325] +
        k[1271]*y[IDX_NI] + k[1298]*y[IDX_OI] + k[2798]*y[IDX_eM] +
        k[2798]*y[IDX_eM] + k[2798]*y[IDX_eM] + k[2806]*y[IDX_eM] +
        k[2892]*y[IDX_DI];
    data[2546] = 0.0 + k[699]*y[IDX_pH2II] + k[700]*y[IDX_oH2II] +
        k[701]*y[IDX_oH2II] + k[718]*y[IDX_HDII] + k[719]*y[IDX_HDII] +
        k[1138]*y[IDX_CHI] + k[1164]*y[IDX_OHI] + k[1186]*y[IDX_NHI] +
        k[1737]*y[IDX_CHII] + k[1860]*y[IDX_NHII] + k[1943]*y[IDX_OHII] +
        k[2932]*y[IDX_HDII] + k[3138]*y[IDX_H2OII] + k[3141]*y[IDX_HDOII] +
        k[3422]*y[IDX_H3OII] + k[3422]*y[IDX_H3OII] + k[3424]*y[IDX_H3OII] +
        k[3429]*y[IDX_H2DOII] + k[3429]*y[IDX_H2DOII] + k[3431]*y[IDX_H2DOII] +
        k[3435]*y[IDX_HD2OII];
    data[2547] = 0.0 + k[656]*y[IDX_CHI] + k[685]*y[IDX_pH2I] +
        k[687]*y[IDX_oH2I] + k[731]*y[IDX_HDI] + k[748]*y[IDX_NHI] +
        k[776]*y[IDX_OHI] - k[2095]*y[IDX_HI] + k[2154]*y[IDX_HM] -
        k[2865]*y[IDX_HI] + k[3034]*y[IDX_H2OI] + k[3043]*y[IDX_HDOI];
    data[2548] = 0.0 + k[192]*y[IDX_GRAINM] + k[194]*y[IDX_GRAINM] + k[340]
        + k[341] + k[1280]*y[IDX_NI] + k[1307]*y[IDX_OI] + k[2805]*y[IDX_eM] +
        k[2817]*y[IDX_eM] - k[2859]*y[IDX_HI] - k[2861]*y[IDX_HI] +
        k[2869]*y[IDX_DI];
    data[2549] = 0.0 + k[208] + k[208] + k[213] + k[360] + k[360] +
        k[456]*y[IDX_COII] + k[462]*y[IDX_COII] + k[468]*y[IDX_CNII] +
        k[474]*y[IDX_CNII] + k[497]*y[IDX_OM] + k[682]*y[IDX_oH2II] +
        k[685]*y[IDX_pD2II] + k[686]*y[IDX_oD2II] + k[693]*y[IDX_HDII] +
        k[694]*y[IDX_HDII] + k[868]*y[IDX_HeII] + k[1130]*y[IDX_CI] +
        k[1136]*y[IDX_CHI] + k[1144]*y[IDX_CDI] + k[1152]*y[IDX_OI] +
        k[1158]*y[IDX_OHI] + k[1160]*y[IDX_ODI] + k[1174]*y[IDX_NI] +
        k[1180]*y[IDX_NHI] + k[1182]*y[IDX_NDI] + k[1636]*y[IDX_NII] +
        k[1656]*y[IDX_OII] + k[1694]*y[IDX_C2II] + k[1731]*y[IDX_CHII] +
        k[1733]*y[IDX_CDII] + k[1806]*y[IDX_N2II] + k[1854]*y[IDX_NHII] +
        k[1856]*y[IDX_NDII] + k[1937]*y[IDX_OHII] + k[1939]*y[IDX_ODII] +
        k[2878]*y[IDX_oH2II] + k[2879]*y[IDX_pH2II] + k[2997]*y[IDX_H2OII] +
        k[3024]*y[IDX_H3OII] + k[3024]*y[IDX_H3OII] + k[3123]*y[IDX_HDOII] +
        k[3127]*y[IDX_D2OII] + k[3396]*y[IDX_H2DOII] + k[3396]*y[IDX_H2DOII] +
        k[3398]*y[IDX_H2DOII] + k[3400]*y[IDX_HD2OII] + k[3400]*y[IDX_HD2OII] +
        k[3402]*y[IDX_HD2OII] + k[3407]*y[IDX_D3OII] + k[3407]*y[IDX_D3OII] +
        k[3409]*y[IDX_D3OII];
    data[2550] = 0.0 + k[290] + k[630]*y[IDX_CI] + k[636]*y[IDX_NI] +
        k[642]*y[IDX_OI] + k[648]*y[IDX_C2I] + k[654]*y[IDX_CHI] +
        k[662]*y[IDX_CDI] + k[670]*y[IDX_CNI] + k[676]*y[IDX_COI] +
        k[683]*y[IDX_oH2I] + k[699]*y[IDX_pD2I] + k[702]*y[IDX_oD2I] +
        k[703]*y[IDX_oD2I] + k[726]*y[IDX_HDI] + k[727]*y[IDX_HDI] +
        k[740]*y[IDX_N2I] + k[746]*y[IDX_NHI] + k[754]*y[IDX_NDI] +
        k[762]*y[IDX_NOI] + k[768]*y[IDX_O2I] + k[774]*y[IDX_OHI] +
        k[782]*y[IDX_ODI] - k[2093]*y[IDX_HI] + k[2150]*y[IDX_HM] +
        k[2745]*y[IDX_eM] + k[2745]*y[IDX_eM] + k[2877]*y[IDX_oH2I] +
        k[2879]*y[IDX_pH2I] + k[2899]*y[IDX_DI] + k[2990]*y[IDX_H2OI] +
        k[3037]*y[IDX_HDOI] + k[3047]*y[IDX_D2OI];
    data[2551] = 0.0 + k[184]*y[IDX_GRAINM] + k[185]*y[IDX_GRAINM] +
        k[185]*y[IDX_GRAINM] + k[333] + k[1276]*y[IDX_NI] + k[1303]*y[IDX_OI] +
        k[2803]*y[IDX_eM] + k[2803]*y[IDX_eM] + k[2813]*y[IDX_eM] +
        k[2855]*y[IDX_DI] + k[2856]*y[IDX_DI] - k[2893]*y[IDX_HI];
    data[2552] = 0.0 + k[178]*y[IDX_GRAINM] + k[179]*y[IDX_GRAINM] +
        k[180]*y[IDX_GRAINM] + k[180]*y[IDX_GRAINM] + k[180]*y[IDX_GRAINM] +
        k[326] + k[327] + k[1272]*y[IDX_NI] + k[1299]*y[IDX_OI] +
        k[2799]*y[IDX_eM] + k[2799]*y[IDX_eM] + k[2799]*y[IDX_eM] +
        k[2807]*y[IDX_eM] + k[2808]*y[IDX_eM] + k[2890]*y[IDX_DI] +
        k[2891]*y[IDX_DI];
    data[2553] = 0.0 - k[2135]*y[IDX_HII];
    data[2554] = 0.0 - k[2187]*y[IDX_HII];
    data[2555] = 0.0 - k[1528]*y[IDX_HII] - k[2525]*y[IDX_HII];
    data[2556] = 0.0 - k[1526]*y[IDX_HII] - k[2523]*y[IDX_HII];
    data[2557] = 0.0 - k[2205]*y[IDX_HII];
    data[2558] = 0.0 - k[2207]*y[IDX_HII];
    data[2559] = 0.0 - k[1530]*y[IDX_HII];
    data[2560] = 0.0 - k[2191]*y[IDX_HII];
    data[2561] = 0.0 - k[1535]*y[IDX_HII] - k[1536]*y[IDX_HII] -
        k[2529]*y[IDX_HII];
    data[2562] = 0.0 - k[2189]*y[IDX_HII];
    data[2563] = 0.0 + k[230];
    data[2564] = 0.0 - k[1532]*y[IDX_HII] - k[2527]*y[IDX_HII];
    data[2565] = 0.0 - k[1538]*y[IDX_HII] - k[1539]*y[IDX_HII] -
        k[2531]*y[IDX_HII];
    data[2566] = 0.0 + k[2264]*y[IDX_HI];
    data[2567] = 0.0 + k[2082]*y[IDX_HI];
    data[2568] = 0.0 - k[1542]*y[IDX_HII];
    data[2569] = 0.0 + k[2374]*y[IDX_HI];
    data[2570] = 0.0 - k[2649]*y[IDX_HII] - k[2881]*y[IDX_HII];
    data[2571] = 0.0 + k[1681]*y[IDX_HNCI] + k[2853]*y[IDX_HDI] +
        k[2854]*y[IDX_HDI] + k[2880]*y[IDX_HI] + k[2882]*y[IDX_pH2I] +
        k[2883]*y[IDX_oH2I];
    data[2572] = 0.0 - k[2138]*y[IDX_HII];
    data[2573] = 0.0 - k[2211]*y[IDX_HII];
    data[2574] = 0.0 - k[2217]*y[IDX_HII];
    data[2575] = 0.0 + k[2170]*y[IDX_HI];
    data[2576] = 0.0 - k[1546]*y[IDX_HII] - k[1550]*y[IDX_HII] -
        k[2562]*y[IDX_HII];
    data[2577] = 0.0 - k[1682]*y[IDX_HII] - k[1683]*y[IDX_HII] +
        k[1683]*y[IDX_HII];
    data[2578] = 0.0 - k[1687]*y[IDX_HII];
    data[2579] = 0.0 - k[170]*y[IDX_HII];
    data[2580] = 0.0 + k[203] + k[2082]*y[IDX_COII] + k[2093]*y[IDX_pH2II] +
        k[2094]*y[IDX_oH2II] + k[2095]*y[IDX_pD2II] + k[2096]*y[IDX_oD2II] +
        k[2097]*y[IDX_HDII] + k[2129]*y[IDX_N2II] + k[2160]*y[IDX_HeII] +
        k[2169]*y[IDX_HCNII] + k[2170]*y[IDX_DCNII] + k[2239]*y[IDX_OII] +
        k[2264]*y[IDX_CNII] + k[2374]*y[IDX_CO2II] - k[2646]*y[IDX_HII] -
        k[2647]*y[IDX_HII] + k[2880]*y[IDX_DII];
    data[2581] = 0.0 - k[170]*y[IDX_GRAINM] - k[1526]*y[IDX_C2HI] -
        k[1528]*y[IDX_C2DI] - k[1530]*y[IDX_CCOI] - k[1532]*y[IDX_CH2I] -
        k[1535]*y[IDX_CD2I] - k[1536]*y[IDX_CD2I] - k[1538]*y[IDX_CHDI] -
        k[1539]*y[IDX_CHDI] - k[1542]*y[IDX_CO2I] - k[1544]*y[IDX_HCOI] -
        k[1546]*y[IDX_DCOI] - k[1548]*y[IDX_HCOI] - k[1550]*y[IDX_DCOI] -
        k[1679]*y[IDX_HNCI] + k[1679]*y[IDX_HNCI] - k[1682]*y[IDX_DNCI] -
        k[1683]*y[IDX_DNCI] + k[1683]*y[IDX_DNCI] - k[1685]*y[IDX_HNOI] -
        k[1687]*y[IDX_DNOI] - k[2135]*y[IDX_CM] - k[2137]*y[IDX_HM] -
        k[2138]*y[IDX_DM] - k[2185]*y[IDX_OI] - k[2187]*y[IDX_C2I] -
        k[2189]*y[IDX_CHI] - k[2191]*y[IDX_CDI] - k[2193]*y[IDX_NHI] -
        k[2195]*y[IDX_NDI] - k[2197]*y[IDX_NOI] - k[2199]*y[IDX_O2I] -
        k[2201]*y[IDX_OHI] - k[2203]*y[IDX_ODI] - k[2205]*y[IDX_C2NI] -
        k[2207]*y[IDX_C3I] - k[2209]*y[IDX_H2OI] - k[2211]*y[IDX_D2OI] -
        k[2213]*y[IDX_HDOI] - k[2215]*y[IDX_HCNI] - k[2217]*y[IDX_DCNI] -
        k[2253]*y[IDX_NH2I] - k[2255]*y[IDX_ND2I] - k[2257]*y[IDX_NHDI] -
        k[2523]*y[IDX_C2HI] - k[2525]*y[IDX_C2DI] - k[2527]*y[IDX_CH2I] -
        k[2529]*y[IDX_CD2I] - k[2531]*y[IDX_CHDI] - k[2560]*y[IDX_HCOI] -
        k[2562]*y[IDX_DCOI] - k[2646]*y[IDX_HI] - k[2647]*y[IDX_HI] -
        k[2649]*y[IDX_DI] - k[2846]*y[IDX_eM] - k[2851]*y[IDX_pD2I] -
        k[2852]*y[IDX_oD2I] - k[2874]*y[IDX_oH2I] + k[2874]*y[IDX_oH2I] -
        k[2875]*y[IDX_pH2I] + k[2875]*y[IDX_pH2I] - k[2881]*y[IDX_DI] -
        k[2884]*y[IDX_HDI] - k[2885]*y[IDX_HDI];
    data[2582] = 0.0 - k[2137]*y[IDX_HII];
    data[2583] = 0.0 + k[903]*y[IDX_HeII] - k[2209]*y[IDX_HII];
    data[2584] = 0.0 - k[2215]*y[IDX_HII];
    data[2585] = 0.0 + k[2169]*y[IDX_HI];
    data[2586] = 0.0 - k[1544]*y[IDX_HII] - k[1548]*y[IDX_HII] -
        k[2560]*y[IDX_HII];
    data[2587] = 0.0 + k[218] + k[223] + k[873]*y[IDX_HeII] +
        k[2853]*y[IDX_DII] + k[2854]*y[IDX_DII] - k[2884]*y[IDX_HII] -
        k[2885]*y[IDX_HII];
    data[2588] = 0.0 + k[295] + k[2097]*y[IDX_HI];
    data[2589] = 0.0 + k[906]*y[IDX_HeII] - k[2213]*y[IDX_HII];
    data[2590] = 0.0 - k[1679]*y[IDX_HII] + k[1679]*y[IDX_HII] +
        k[1681]*y[IDX_DII];
    data[2591] = 0.0 + k[931]*y[IDX_HeII] - k[1685]*y[IDX_HII];
    data[2592] = 0.0 + k[868]*y[IDX_pH2I] + k[869]*y[IDX_oH2I] +
        k[873]*y[IDX_HDI] + k[903]*y[IDX_H2OI] + k[906]*y[IDX_HDOI] +
        k[931]*y[IDX_HNOI] + k[2160]*y[IDX_HI];
    data[2593] = 0.0 + k[2129]*y[IDX_HI];
    data[2594] = 0.0 - k[2195]*y[IDX_HII];
    data[2595] = 0.0 - k[2255]*y[IDX_HII];
    data[2596] = 0.0 - k[2193]*y[IDX_HII];
    data[2597] = 0.0 - k[2253]*y[IDX_HII];
    data[2598] = 0.0 - k[2257]*y[IDX_HII];
    data[2599] = 0.0 - k[2197]*y[IDX_HII];
    data[2600] = 0.0 - k[2185]*y[IDX_HII];
    data[2601] = 0.0 + k[2239]*y[IDX_HI];
    data[2602] = 0.0 - k[2199]*y[IDX_HII];
    data[2603] = 0.0 - k[2203]*y[IDX_HII];
    data[2604] = 0.0 - k[2201]*y[IDX_HII];
    data[2605] = 0.0 + k[296];
    data[2606] = 0.0 - k[2846]*y[IDX_HII];
    data[2607] = 0.0 - k[2852]*y[IDX_HII];
    data[2608] = 0.0 + k[2096]*y[IDX_HI];
    data[2609] = 0.0 + k[318] + k[319];
    data[2610] = 0.0 + k[214] + k[220] + k[869]*y[IDX_HeII] -
        k[2874]*y[IDX_HII] + k[2874]*y[IDX_HII] + k[2883]*y[IDX_DII];
    data[2611] = 0.0 + k[291] + k[2094]*y[IDX_HI];
    data[2612] = 0.0 + k[316];
    data[2613] = 0.0 + k[304] + k[305];
    data[2614] = 0.0 - k[2851]*y[IDX_HII];
    data[2615] = 0.0 + k[2095]*y[IDX_HI];
    data[2616] = 0.0 + k[320] + k[321];
    data[2617] = 0.0 + k[213] + k[219] + k[868]*y[IDX_HeII] -
        k[2875]*y[IDX_HII] + k[2875]*y[IDX_HII] + k[2882]*y[IDX_DII];
    data[2618] = 0.0 + k[290] + k[2093]*y[IDX_HI];
    data[2619] = 0.0 + k[317];
    data[2620] = 0.0 + k[306] + k[307];
    data[2621] = 0.0 - k[2687]*y[IDX_HM];
    data[2622] = 0.0 - k[2133]*y[IDX_HM];
    data[2623] = 0.0 - k[2697]*y[IDX_HM];
    data[2624] = 0.0 - k[2701]*y[IDX_HM];
    data[2625] = 0.0 - k[2699]*y[IDX_HM];
    data[2626] = 0.0 - k[2703]*y[IDX_HM];
    data[2627] = 0.0 - k[2705]*y[IDX_HM];
    data[2628] = 0.0 - k[2691]*y[IDX_HM];
    data[2629] = 0.0 - k[2139]*y[IDX_HM];
    data[2630] = 0.0 - k[486]*y[IDX_HM] - k[487]*y[IDX_HM];
    data[2631] = 0.0 - k[3082]*y[IDX_HM] - k[3083]*y[IDX_HM] -
        k[3084]*y[IDX_HM] - k[3085]*y[IDX_HM] - k[3086]*y[IDX_HM] -
        k[3108]*y[IDX_HM] - k[3109]*y[IDX_HM] - k[3110]*y[IDX_HM];
    data[2632] = 0.0 - k[495]*y[IDX_HM];
    data[2633] = 0.0 - k[860]*y[IDX_HM];
    data[2634] = 0.0 - k[2689]*y[IDX_HM] + k[2737]*y[IDX_eM];
    data[2635] = 0.0 - k[2137]*y[IDX_HM];
    data[2636] = 0.0 - k[345] - k[483]*y[IDX_H2OI] - k[486]*y[IDX_D2OI] -
        k[487]*y[IDX_D2OI] - k[489]*y[IDX_HDOI] - k[490]*y[IDX_HDOI] -
        k[493]*y[IDX_HCNI] - k[495]*y[IDX_DCNI] - k[827]*y[IDX_oH3II] -
        k[828]*y[IDX_pH3II] - k[829]*y[IDX_pH3II] - k[836]*y[IDX_oH2DII] -
        k[837]*y[IDX_oH2DII] - k[838]*y[IDX_pH2DII] - k[839]*y[IDX_pH2DII] -
        k[847]*y[IDX_oD2HII] - k[848]*y[IDX_oD2HII] - k[849]*y[IDX_pD2HII] -
        k[850]*y[IDX_pD2HII] - k[851]*y[IDX_pD2HII] - k[852]*y[IDX_oD2HII] -
        k[853]*y[IDX_pD2HII] - k[858]*y[IDX_HCOII] - k[860]*y[IDX_DCOII] -
        k[2133]*y[IDX_CII] - k[2137]*y[IDX_HII] - k[2139]*y[IDX_DII] -
        k[2142]*y[IDX_HeII] - k[2145]*y[IDX_NII] - k[2148]*y[IDX_OII] -
        k[2150]*y[IDX_pH2II] - k[2151]*y[IDX_oH2II] - k[2154]*y[IDX_pD2II] -
        k[2155]*y[IDX_oD2II] - k[2158]*y[IDX_HDII] - k[2687]*y[IDX_CI] -
        k[2689]*y[IDX_HI] - k[2691]*y[IDX_DI] - k[2693]*y[IDX_NI] -
        k[2695]*y[IDX_OI] - k[2697]*y[IDX_C2I] - k[2699]*y[IDX_CHI] -
        k[2701]*y[IDX_CDI] - k[2703]*y[IDX_CNI] - k[2705]*y[IDX_COI] -
        k[2707]*y[IDX_NHI] - k[2709]*y[IDX_NDI] - k[2711]*y[IDX_OHI] -
        k[2713]*y[IDX_ODI] - k[2991]*y[IDX_H3OII] - k[2992]*y[IDX_H3OII] -
        k[2993]*y[IDX_H3OII] - k[2994]*y[IDX_H3OII] - k[3060]*y[IDX_H2DOII] -
        k[3061]*y[IDX_H2DOII] - k[3062]*y[IDX_H2DOII] - k[3063]*y[IDX_H2DOII] -
        k[3064]*y[IDX_H2DOII] - k[3071]*y[IDX_HD2OII] - k[3072]*y[IDX_HD2OII] -
        k[3073]*y[IDX_HD2OII] - k[3074]*y[IDX_HD2OII] - k[3075]*y[IDX_HD2OII] -
        k[3076]*y[IDX_HD2OII] - k[3082]*y[IDX_D3OII] - k[3083]*y[IDX_D3OII] -
        k[3084]*y[IDX_D3OII] - k[3085]*y[IDX_D3OII] - k[3086]*y[IDX_D3OII] -
        k[3092]*y[IDX_H2DOII] - k[3093]*y[IDX_H2DOII] - k[3094]*y[IDX_H2DOII] -
        k[3100]*y[IDX_HD2OII] - k[3101]*y[IDX_HD2OII] - k[3102]*y[IDX_HD2OII] -
        k[3103]*y[IDX_HD2OII] - k[3104]*y[IDX_HD2OII] - k[3108]*y[IDX_D3OII] -
        k[3109]*y[IDX_D3OII] - k[3110]*y[IDX_D3OII];
    data[2637] = 0.0 - k[3060]*y[IDX_HM] - k[3061]*y[IDX_HM] -
        k[3062]*y[IDX_HM] - k[3063]*y[IDX_HM] - k[3064]*y[IDX_HM] -
        k[3092]*y[IDX_HM] - k[3093]*y[IDX_HM] - k[3094]*y[IDX_HM];
    data[2638] = 0.0 - k[483]*y[IDX_HM];
    data[2639] = 0.0 - k[2991]*y[IDX_HM] - k[2992]*y[IDX_HM] -
        k[2993]*y[IDX_HM] - k[2994]*y[IDX_HM];
    data[2640] = 0.0 - k[493]*y[IDX_HM];
    data[2641] = 0.0 - k[858]*y[IDX_HM];
    data[2642] = 0.0 + k[224];
    data[2643] = 0.0 - k[2158]*y[IDX_HM];
    data[2644] = 0.0 - k[3071]*y[IDX_HM] - k[3072]*y[IDX_HM] -
        k[3073]*y[IDX_HM] - k[3074]*y[IDX_HM] - k[3075]*y[IDX_HM] -
        k[3076]*y[IDX_HM] - k[3100]*y[IDX_HM] - k[3101]*y[IDX_HM] -
        k[3102]*y[IDX_HM] - k[3103]*y[IDX_HM] - k[3104]*y[IDX_HM];
    data[2645] = 0.0 - k[489]*y[IDX_HM] - k[490]*y[IDX_HM];
    data[2646] = 0.0 - k[2142]*y[IDX_HM];
    data[2647] = 0.0 - k[2693]*y[IDX_HM];
    data[2648] = 0.0 - k[2145]*y[IDX_HM];
    data[2649] = 0.0 - k[2709]*y[IDX_HM];
    data[2650] = 0.0 - k[2707]*y[IDX_HM];
    data[2651] = 0.0 - k[2695]*y[IDX_HM];
    data[2652] = 0.0 - k[2148]*y[IDX_HM];
    data[2653] = 0.0 - k[2713]*y[IDX_HM];
    data[2654] = 0.0 - k[2711]*y[IDX_HM];
    data[2655] = 0.0 + k[2737]*y[IDX_HI];
    data[2656] = 0.0 - k[2155]*y[IDX_HM];
    data[2657] = 0.0 - k[847]*y[IDX_HM] - k[848]*y[IDX_HM] -
        k[852]*y[IDX_HM];
    data[2658] = 0.0 + k[220];
    data[2659] = 0.0 - k[2151]*y[IDX_HM];
    data[2660] = 0.0 - k[836]*y[IDX_HM] - k[837]*y[IDX_HM];
    data[2661] = 0.0 - k[827]*y[IDX_HM];
    data[2662] = 0.0 - k[2154]*y[IDX_HM];
    data[2663] = 0.0 - k[849]*y[IDX_HM] - k[850]*y[IDX_HM] -
        k[851]*y[IDX_HM] - k[853]*y[IDX_HM];
    data[2664] = 0.0 + k[219];
    data[2665] = 0.0 - k[2150]*y[IDX_HM];
    data[2666] = 0.0 - k[838]*y[IDX_HM] - k[839]*y[IDX_HM];
    data[2667] = 0.0 - k[828]*y[IDX_HM] - k[829]*y[IDX_HM];
    data[2668] = 0.0 - k[3283]*y[IDX_H2DOII] - k[3284]*y[IDX_H2DOII] -
        k[3285]*y[IDX_H2DOII];
    data[2669] = 0.0 - k[3336]*y[IDX_H2DOII] - k[3337]*y[IDX_H2DOII];
    data[2670] = 0.0 - k[3300]*y[IDX_H2DOII] - k[3301]*y[IDX_H2DOII] -
        k[3302]*y[IDX_H2DOII];
    data[2671] = 0.0 + k[3306]*y[IDX_H2OI];
    data[2672] = 0.0 - k[3291]*y[IDX_H2DOII] - k[3292]*y[IDX_H2DOII];
    data[2673] = 0.0 + k[3307]*y[IDX_HDOI];
    data[2674] = 0.0 - k[3346]*y[IDX_H2DOII] - k[3347]*y[IDX_H2DOII];
    data[2675] = 0.0 - k[3386]*y[IDX_H2DOII] - k[3387]*y[IDX_H2DOII] -
        k[3388]*y[IDX_H2DOII] - k[3389]*y[IDX_H2DOII] - k[3390]*y[IDX_H2DOII];
    data[2676] = 0.0 - k[3065]*y[IDX_H2DOII] - k[3066]*y[IDX_H2DOII] -
        k[3067]*y[IDX_H2DOII] - k[3068]*y[IDX_H2DOII] - k[3069]*y[IDX_H2DOII] -
        k[3070]*y[IDX_H2DOII] - k[3095]*y[IDX_H2DOII] - k[3096]*y[IDX_H2DOII] -
        k[3097]*y[IDX_H2DOII] - k[3098]*y[IDX_H2DOII] - k[3099]*y[IDX_H2DOII];
    data[2677] = 0.0 + k[3049]*y[IDX_pH2II] + k[3050]*y[IDX_oH2II] +
        k[3168]*y[IDX_H2OII] + k[3242]*y[IDX_pH3II] + k[3243]*y[IDX_oH3II] +
        k[3252]*y[IDX_pH2DII] + k[3253]*y[IDX_pH2DII] + k[3254]*y[IDX_oH2DII] +
        k[3255]*y[IDX_oH2DII] + k[3332]*y[IDX_NH2II];
    data[2678] = 0.0 + k[3129]*y[IDX_pH2I] + k[3130]*y[IDX_oH2I] +
        k[3160]*y[IDX_H2OI];
    data[2679] = 0.0 + k[3410]*y[IDX_oH2I] + k[3411]*y[IDX_pH2I];
    data[2680] = 0.0 + k[3113]*y[IDX_H2OI];
    data[2681] = 0.0 + k[3174]*y[IDX_H2OII];
    data[2682] = 0.0 + k[3118]*y[IDX_H2OI];
    data[2683] = 0.0 + k[3268]*y[IDX_H2OI];
    data[2684] = 0.0 + k[3273]*y[IDX_H2OI];
    data[2685] = 0.0 - k[3372]*y[IDX_H2DOII] - k[3373]*y[IDX_H2DOII] -
        k[3374]*y[IDX_H2DOII];
    data[2686] = 0.0 - k[3060]*y[IDX_H2DOII] - k[3061]*y[IDX_H2DOII] -
        k[3062]*y[IDX_H2DOII] - k[3063]*y[IDX_H2DOII] - k[3064]*y[IDX_H2DOII] -
        k[3092]*y[IDX_H2DOII] - k[3093]*y[IDX_H2DOII] - k[3094]*y[IDX_H2DOII];
    data[2687] = 0.0 - k[3060]*y[IDX_HM] - k[3061]*y[IDX_HM] -
        k[3062]*y[IDX_HM] - k[3063]*y[IDX_HM] - k[3064]*y[IDX_HM] -
        k[3065]*y[IDX_DM] - k[3066]*y[IDX_DM] - k[3067]*y[IDX_DM] -
        k[3068]*y[IDX_DM] - k[3069]*y[IDX_DM] - k[3070]*y[IDX_DM] -
        k[3092]*y[IDX_HM] - k[3093]*y[IDX_HM] - k[3094]*y[IDX_HM] -
        k[3095]*y[IDX_DM] - k[3096]*y[IDX_DM] - k[3097]*y[IDX_DM] -
        k[3098]*y[IDX_DM] - k[3099]*y[IDX_DM] - k[3283]*y[IDX_CI] -
        k[3284]*y[IDX_CI] - k[3285]*y[IDX_CI] - k[3291]*y[IDX_CHI] -
        k[3292]*y[IDX_CHI] - k[3300]*y[IDX_CDI] - k[3301]*y[IDX_CDI] -
        k[3302]*y[IDX_CDI] - k[3336]*y[IDX_CM] - k[3337]*y[IDX_CM] -
        k[3341]*y[IDX_OM] - k[3342]*y[IDX_OM] - k[3346]*y[IDX_CNM] -
        k[3347]*y[IDX_CNM] - k[3354]*y[IDX_OHM] - k[3355]*y[IDX_OHM] -
        k[3356]*y[IDX_OHM] - k[3357]*y[IDX_ODM] - k[3358]*y[IDX_ODM] -
        k[3359]*y[IDX_ODM] - k[3360]*y[IDX_ODM] - k[3372]*y[IDX_HI] -
        k[3373]*y[IDX_HI] - k[3374]*y[IDX_HI] - k[3386]*y[IDX_DI] -
        k[3387]*y[IDX_DI] - k[3388]*y[IDX_DI] - k[3389]*y[IDX_DI] -
        k[3390]*y[IDX_DI] - k[3396]*y[IDX_pH2I] + k[3396]*y[IDX_pH2I] -
        k[3397]*y[IDX_oH2I] + k[3397]*y[IDX_oH2I] - k[3398]*y[IDX_pH2I] -
        k[3399]*y[IDX_oH2I] - k[3414]*y[IDX_HDI] - k[3415]*y[IDX_HDI] +
        k[3415]*y[IDX_HDI] - k[3416]*y[IDX_HDI] - k[3428]*y[IDX_oD2I] -
        k[3429]*y[IDX_pD2I] - k[3430]*y[IDX_oD2I] - k[3431]*y[IDX_pD2I] -
        k[3432]*y[IDX_oD2I] + k[3432]*y[IDX_oD2I] - k[3433]*y[IDX_pD2I] +
        k[3433]*y[IDX_pD2I] - k[3440]*y[IDX_eM] - k[3441]*y[IDX_eM] -
        k[3445]*y[IDX_eM] - k[3446]*y[IDX_eM] - k[3450]*y[IDX_eM] -
        k[3451]*y[IDX_eM] - k[3452]*y[IDX_eM] - k[3458]*y[IDX_eM] -
        k[3459]*y[IDX_eM] - k[3460]*y[IDX_eM];
    data[2688] = 0.0 + k[3031]*y[IDX_HDII] + k[3035]*y[IDX_oD2II] +
        k[3036]*y[IDX_pD2II] + k[3113]*y[IDX_DCNII] + k[3118]*y[IDX_DCOII] +
        k[3157]*y[IDX_HDOII] + k[3160]*y[IDX_D2OII] + k[3177]*y[IDX_oH2DII] +
        k[3178]*y[IDX_oH2DII] + k[3179]*y[IDX_pH2DII] + k[3180]*y[IDX_pH2DII] +
        k[3187]*y[IDX_oD2HII] + k[3188]*y[IDX_pD2HII] + k[3200]*y[IDX_pD3II] +
        k[3201]*y[IDX_mD3II] + k[3202]*y[IDX_oD3II] + k[3203]*y[IDX_oD3II] +
        k[3268]*y[IDX_DNCII] + k[3273]*y[IDX_DNOII] + k[3278]*y[IDX_N2DII] +
        k[3306]*y[IDX_CDII] + k[3311]*y[IDX_NDII] + k[3316]*y[IDX_ODII] +
        k[3321]*y[IDX_NHDII] + k[3324]*y[IDX_ND2II];
    data[2689] = 0.0 + k[3131]*y[IDX_HDI] + k[3139]*y[IDX_oD2I] +
        k[3140]*y[IDX_pD2I] + k[3149]*y[IDX_NDI] + k[3154]*y[IDX_ODI] +
        k[3161]*y[IDX_HDOI] + k[3168]*y[IDX_D2OI] + k[3174]*y[IDX_DCOI];
    data[2690] = 0.0 + k[3412]*y[IDX_HDI] + k[3424]*y[IDX_pD2I] +
        k[3425]*y[IDX_oD2I];
    data[2691] = 0.0 + k[3114]*y[IDX_HDOI];
    data[2692] = 0.0 + k[3172]*y[IDX_HDOII];
    data[2693] = 0.0 + k[3119]*y[IDX_HDOI];
    data[2694] = 0.0 + k[3131]*y[IDX_H2OII] + k[3134]*y[IDX_HDOII] +
        k[3412]*y[IDX_H3OII] - k[3414]*y[IDX_H2DOII] - k[3415]*y[IDX_H2DOII] +
        k[3415]*y[IDX_H2DOII] - k[3416]*y[IDX_H2DOII] + k[3419]*y[IDX_HD2OII];
    data[2695] = 0.0 + k[3031]*y[IDX_H2OI] + k[3042]*y[IDX_HDOI];
    data[2696] = 0.0 + k[3402]*y[IDX_pH2I] + k[3403]*y[IDX_oH2I] +
        k[3419]*y[IDX_HDI];
    data[2697] = 0.0 + k[3037]*y[IDX_pH2II] + k[3038]*y[IDX_oH2II] +
        k[3042]*y[IDX_HDII] + k[3114]*y[IDX_HCNII] + k[3119]*y[IDX_HCOII] +
        k[3161]*y[IDX_H2OII] + k[3164]*y[IDX_HDOII] + k[3204]*y[IDX_oH3II] +
        k[3205]*y[IDX_oH3II] + k[3206]*y[IDX_pH3II] + k[3207]*y[IDX_pH3II] +
        k[3214]*y[IDX_pH2DII] + k[3215]*y[IDX_oH2DII] + k[3226]*y[IDX_pD2HII] +
        k[3227]*y[IDX_pD2HII] + k[3228]*y[IDX_oD2HII] + k[3229]*y[IDX_oD2HII] +
        k[3269]*y[IDX_HNCII] + k[3274]*y[IDX_HNOII] + k[3279]*y[IDX_N2HII] +
        k[3307]*y[IDX_CHII] + k[3312]*y[IDX_NHII] + k[3317]*y[IDX_OHII] +
        k[3325]*y[IDX_NH2II] + k[3328]*y[IDX_NHDII];
    data[2698] = 0.0 + k[3123]*y[IDX_pH2I] + k[3124]*y[IDX_oH2I] +
        k[3134]*y[IDX_HDI] + k[3147]*y[IDX_NHI] + k[3152]*y[IDX_OHI] +
        k[3157]*y[IDX_H2OI] + k[3164]*y[IDX_HDOI] + k[3172]*y[IDX_HCOI];
    data[2699] = 0.0 + k[3269]*y[IDX_HDOI];
    data[2700] = 0.0 + k[3274]*y[IDX_HDOI];
    data[2701] = 0.0 + k[3278]*y[IDX_H2OI];
    data[2702] = 0.0 + k[3279]*y[IDX_HDOI];
    data[2703] = 0.0 + k[3149]*y[IDX_H2OII];
    data[2704] = 0.0 + k[3311]*y[IDX_H2OI];
    data[2705] = 0.0 + k[3324]*y[IDX_H2OI];
    data[2706] = 0.0 + k[3147]*y[IDX_HDOII];
    data[2707] = 0.0 + k[3312]*y[IDX_HDOI];
    data[2708] = 0.0 + k[3325]*y[IDX_HDOI] + k[3332]*y[IDX_D2OI];
    data[2709] = 0.0 + k[3321]*y[IDX_H2OI] + k[3328]*y[IDX_HDOI];
    data[2710] = 0.0 - k[3341]*y[IDX_H2DOII] - k[3342]*y[IDX_H2DOII];
    data[2711] = 0.0 + k[3154]*y[IDX_H2OII];
    data[2712] = 0.0 + k[3316]*y[IDX_H2OI];
    data[2713] = 0.0 - k[3357]*y[IDX_H2DOII] - k[3358]*y[IDX_H2DOII] -
        k[3359]*y[IDX_H2DOII] - k[3360]*y[IDX_H2DOII];
    data[2714] = 0.0 + k[3152]*y[IDX_HDOII];
    data[2715] = 0.0 + k[3317]*y[IDX_HDOI];
    data[2716] = 0.0 - k[3354]*y[IDX_H2DOII] - k[3355]*y[IDX_H2DOII] -
        k[3356]*y[IDX_H2DOII];
    data[2717] = 0.0 - k[3440]*y[IDX_H2DOII] - k[3441]*y[IDX_H2DOII] -
        k[3445]*y[IDX_H2DOII] - k[3446]*y[IDX_H2DOII] - k[3450]*y[IDX_H2DOII] -
        k[3451]*y[IDX_H2DOII] - k[3452]*y[IDX_H2DOII] - k[3458]*y[IDX_H2DOII] -
        k[3459]*y[IDX_H2DOII] - k[3460]*y[IDX_H2DOII];
    data[2718] = 0.0 + k[3201]*y[IDX_H2OI];
    data[2719] = 0.0 + k[3139]*y[IDX_H2OII] + k[3425]*y[IDX_H3OII] -
        k[3428]*y[IDX_H2DOII] - k[3430]*y[IDX_H2DOII] - k[3432]*y[IDX_H2DOII] +
        k[3432]*y[IDX_H2DOII];
    data[2720] = 0.0 + k[3035]*y[IDX_H2OI];
    data[2721] = 0.0 + k[3187]*y[IDX_H2OI] + k[3228]*y[IDX_HDOI] +
        k[3229]*y[IDX_HDOI];
    data[2722] = 0.0 + k[3202]*y[IDX_H2OI] + k[3203]*y[IDX_H2OI];
    data[2723] = 0.0 + k[3124]*y[IDX_HDOII] + k[3130]*y[IDX_D2OII] -
        k[3397]*y[IDX_H2DOII] + k[3397]*y[IDX_H2DOII] - k[3399]*y[IDX_H2DOII] +
        k[3403]*y[IDX_HD2OII] + k[3410]*y[IDX_D3OII];
    data[2724] = 0.0 + k[3038]*y[IDX_HDOI] + k[3050]*y[IDX_D2OI];
    data[2725] = 0.0 + k[3177]*y[IDX_H2OI] + k[3178]*y[IDX_H2OI] +
        k[3215]*y[IDX_HDOI] + k[3254]*y[IDX_D2OI] + k[3255]*y[IDX_D2OI];
    data[2726] = 0.0 + k[3204]*y[IDX_HDOI] + k[3205]*y[IDX_HDOI] +
        k[3243]*y[IDX_D2OI];
    data[2727] = 0.0 + k[3140]*y[IDX_H2OII] + k[3424]*y[IDX_H3OII] -
        k[3429]*y[IDX_H2DOII] - k[3431]*y[IDX_H2DOII] - k[3433]*y[IDX_H2DOII] +
        k[3433]*y[IDX_H2DOII];
    data[2728] = 0.0 + k[3036]*y[IDX_H2OI];
    data[2729] = 0.0 + k[3188]*y[IDX_H2OI] + k[3226]*y[IDX_HDOI] +
        k[3227]*y[IDX_HDOI];
    data[2730] = 0.0 + k[3200]*y[IDX_H2OI];
    data[2731] = 0.0 + k[3123]*y[IDX_HDOII] + k[3129]*y[IDX_D2OII] -
        k[3396]*y[IDX_H2DOII] + k[3396]*y[IDX_H2DOII] - k[3398]*y[IDX_H2DOII] +
        k[3402]*y[IDX_HD2OII] + k[3411]*y[IDX_D3OII];
    data[2732] = 0.0 + k[3037]*y[IDX_HDOI] + k[3049]*y[IDX_D2OI];
    data[2733] = 0.0 + k[3179]*y[IDX_H2OI] + k[3180]*y[IDX_H2OI] +
        k[3214]*y[IDX_HDOI] + k[3252]*y[IDX_D2OI] + k[3253]*y[IDX_D2OI];
    data[2734] = 0.0 + k[3206]*y[IDX_HDOI] + k[3207]*y[IDX_HDOI] +
        k[3242]*y[IDX_D2OI];
    data[2735] = 0.0 - k[435]*y[IDX_H2OI] - k[439]*y[IDX_H2OI];
    data[2736] = 0.0 + k[3017]*y[IDX_H3OII] + k[3337]*y[IDX_H2DOII];
    data[2737] = 0.0 + k[2479]*y[IDX_H2OII];
    data[2738] = 0.0 - k[1705]*y[IDX_H2OI];
    data[2739] = 0.0 + k[2491]*y[IDX_H2OII];
    data[2740] = 0.0 + k[2488]*y[IDX_H2OII];
    data[2741] = 0.0 - k[960]*y[IDX_H2OI];
    data[2742] = 0.0 + k[2485]*y[IDX_H2OII] + k[3298]*y[IDX_H3OII] +
        k[3300]*y[IDX_H2DOII];
    data[2743] = 0.0 - k[1778]*y[IDX_H2OI] - k[1779]*y[IDX_H2OI] -
        k[3306]*y[IDX_H2OI];
    data[2744] = 0.0 + k[2497]*y[IDX_H2OII];
    data[2745] = 0.0 + k[2482]*y[IDX_H2OII] + k[3012]*y[IDX_H3OII] +
        k[3291]*y[IDX_H2DOII] + k[3293]*y[IDX_HD2OII];
    data[2746] = 0.0 - k[1777]*y[IDX_H2OI] - k[3013]*y[IDX_H2OI];
    data[2747] = 0.0 + k[2494]*y[IDX_H2OII];
    data[2748] = 0.0 + k[2500]*y[IDX_H2OII];
    data[2749] = 0.0 - k[599]*y[IDX_H2OI] - k[603]*y[IDX_H2OI];
    data[2750] = 0.0 + k[3019]*y[IDX_H3OII] + k[3347]*y[IDX_H2DOII];
    data[2751] = 0.0 - k[980]*y[IDX_H2OI];
    data[2752] = 0.0 - k[622]*y[IDX_H2OI] - k[2334]*y[IDX_H2OI];
    data[2753] = 0.0 + k[790]*y[IDX_pH2II] + k[791]*y[IDX_oH2II];
    data[2754] = 0.0 - k[2377]*y[IDX_H2OI];
    data[2755] = 0.0 - k[1087]*y[IDX_H2OI] - k[1088]*y[IDX_H2OI];
    data[2756] = 0.0 - k[2210]*y[IDX_H2OI];
    data[2757] = 0.0 - k[484]*y[IDX_H2OI] - k[485]*y[IDX_H2OI] +
        k[3091]*y[IDX_H3OII] + k[3098]*y[IDX_H2DOII] + k[3099]*y[IDX_H2DOII];
    data[2758] = 0.0 - k[3159]*y[IDX_H2OI] - k[3160]*y[IDX_H2OI];
    data[2759] = 0.0 - k[2381]*y[IDX_H2OI] - k[3113]*y[IDX_H2OI];
    data[2760] = 0.0 + k[2506]*y[IDX_H2OII];
    data[2761] = 0.0 - k[3118]*y[IDX_H2OI];
    data[2762] = 0.0 - k[3268]*y[IDX_H2OI];
    data[2763] = 0.0 - k[3273]*y[IDX_H2OI];
    data[2764] = 0.0 - k[1082]*y[IDX_H2OI] + k[1120]*y[IDX_O2HI] +
        k[2666]*y[IDX_OHI] + k[2732]*y[IDX_OHM];
    data[2765] = 0.0 - k[2209]*y[IDX_H2OI];
    data[2766] = 0.0 - k[483]*y[IDX_H2OI] + k[2711]*y[IDX_OHI] +
        k[2993]*y[IDX_H3OII] + k[2994]*y[IDX_H3OII] + k[3094]*y[IDX_H2DOII] +
        k[3103]*y[IDX_HD2OII] + k[3104]*y[IDX_HD2OII];
    data[2767] = 0.0 + k[3094]*y[IDX_HM] + k[3098]*y[IDX_DM] +
        k[3099]*y[IDX_DM] + k[3291]*y[IDX_CHI] + k[3300]*y[IDX_CDI] +
        k[3337]*y[IDX_CM] + k[3342]*y[IDX_OM] + k[3347]*y[IDX_CNM] +
        k[3355]*y[IDX_OHM] + k[3356]*y[IDX_OHM] + k[3360]*y[IDX_ODM] +
        k[3446]*y[IDX_eM];
    data[2768] = 0.0 - k[253] - k[393] - k[397] - k[435]*y[IDX_CII] -
        k[439]*y[IDX_CII] - k[483]*y[IDX_HM] - k[484]*y[IDX_DM] -
        k[485]*y[IDX_DM] - k[599]*y[IDX_CNII] - k[603]*y[IDX_CNII] -
        k[622]*y[IDX_COII] - k[903]*y[IDX_HeII] - k[907]*y[IDX_HeII] -
        k[960]*y[IDX_C2NII] - k[980]*y[IDX_CNCII] - k[1082]*y[IDX_HI] -
        k[1087]*y[IDX_DI] - k[1088]*y[IDX_DI] - k[1705]*y[IDX_C2II] -
        k[1777]*y[IDX_CHII] - k[1778]*y[IDX_CDII] - k[1779]*y[IDX_CDII] -
        k[1812]*y[IDX_N2II] - k[1890]*y[IDX_NHII] - k[1891]*y[IDX_NDII] -
        k[1892]*y[IDX_NDII] - k[1900]*y[IDX_NHII] - k[1901]*y[IDX_NDII] -
        k[1902]*y[IDX_NDII] - k[2209]*y[IDX_HII] - k[2210]*y[IDX_DII] -
        k[2231]*y[IDX_NII] - k[2246]*y[IDX_OII] - k[2334]*y[IDX_COII] -
        k[2371]*y[IDX_HeII] - k[2377]*y[IDX_CO2II] - k[2380]*y[IDX_HCNII] -
        k[2381]*y[IDX_DCNII] - k[2443]*y[IDX_pH2II] - k[2444]*y[IDX_oH2II] -
        k[2445]*y[IDX_pD2II] - k[2446]*y[IDX_oD2II] - k[2447]*y[IDX_HDII] -
        k[2592]*y[IDX_OHII] - k[2593]*y[IDX_ODII] - k[2622]*y[IDX_N2II] -
        k[2625]*y[IDX_NHII] - k[2626]*y[IDX_NDII] - k[2989]*y[IDX_oH2II] -
        k[2990]*y[IDX_pH2II] - k[2995]*y[IDX_HCNII] - k[2996]*y[IDX_HCOII] -
        k[3001]*y[IDX_H2OII] - k[3003]*y[IDX_oH3II] - k[3004]*y[IDX_oH3II] -
        k[3005]*y[IDX_pH3II] - k[3006]*y[IDX_pH3II] - k[3007]*y[IDX_HNCII] -
        k[3008]*y[IDX_HNOII] - k[3009]*y[IDX_N2HII] - k[3013]*y[IDX_CHII] -
        k[3014]*y[IDX_NHII] - k[3015]*y[IDX_OHII] - k[3016]*y[IDX_NH2II] -
        k[3031]*y[IDX_HDII] - k[3032]*y[IDX_HDII] - k[3033]*y[IDX_oD2II] -
        k[3034]*y[IDX_pD2II] - k[3035]*y[IDX_oD2II] - k[3036]*y[IDX_pD2II] -
        k[3113]*y[IDX_DCNII] - k[3118]*y[IDX_DCOII] - k[3157]*y[IDX_HDOII] -
        k[3158]*y[IDX_HDOII] - k[3159]*y[IDX_D2OII] - k[3160]*y[IDX_D2OII] -
        k[3177]*y[IDX_oH2DII] - k[3178]*y[IDX_oH2DII] - k[3179]*y[IDX_pH2DII] -
        k[3180]*y[IDX_pH2DII] - k[3181]*y[IDX_oH2DII] - k[3182]*y[IDX_pH2DII] -
        k[3183]*y[IDX_oD2HII] - k[3184]*y[IDX_oD2HII] - k[3185]*y[IDX_pD2HII] -
        k[3186]*y[IDX_pD2HII] - k[3187]*y[IDX_oD2HII] - k[3188]*y[IDX_pD2HII] -
        k[3189]*y[IDX_oD2HII] - k[3190]*y[IDX_pD2HII] - k[3191]*y[IDX_pD3II] -
        k[3192]*y[IDX_mD3II] - k[3193]*y[IDX_oD3II] - k[3194]*y[IDX_pD3II] -
        k[3195]*y[IDX_oD3II] - k[3196]*y[IDX_mD3II] - k[3197]*y[IDX_pD3II] -
        k[3198]*y[IDX_mD3II] - k[3199]*y[IDX_oD3II] - k[3200]*y[IDX_pD3II] -
        k[3201]*y[IDX_mD3II] - k[3202]*y[IDX_oD3II] - k[3203]*y[IDX_oD3II] -
        k[3268]*y[IDX_DNCII] - k[3273]*y[IDX_DNOII] - k[3278]*y[IDX_N2DII] -
        k[3306]*y[IDX_CDII] - k[3311]*y[IDX_NDII] - k[3316]*y[IDX_ODII] -
        k[3321]*y[IDX_NHDII] - k[3322]*y[IDX_NHDII] - k[3323]*y[IDX_ND2II] -
        k[3324]*y[IDX_ND2II];
    data[2769] = 0.0 + k[2179]*y[IDX_NOI] + k[2182]*y[IDX_O2I] +
        k[2479]*y[IDX_C2I] + k[2482]*y[IDX_CHI] + k[2485]*y[IDX_CDI] +
        k[2488]*y[IDX_C2HI] + k[2491]*y[IDX_C2DI] + k[2494]*y[IDX_CH2I] +
        k[2497]*y[IDX_CD2I] + k[2500]*y[IDX_CHDI] + k[2503]*y[IDX_HCOI] +
        k[2506]*y[IDX_DCOI] + k[2511]*y[IDX_NH2I] + k[2514]*y[IDX_ND2I] +
        k[2517]*y[IDX_NHDI] - k[3001]*y[IDX_H2OI];
    data[2770] = 0.0 + k[2993]*y[IDX_HM] + k[2994]*y[IDX_HM] +
        k[3012]*y[IDX_CHI] + k[3017]*y[IDX_CM] + k[3018]*y[IDX_OM] +
        k[3019]*y[IDX_CNM] + k[3020]*y[IDX_OHM] + k[3026]*y[IDX_eM] +
        k[3091]*y[IDX_DM] + k[3298]*y[IDX_CDI] + k[3352]*y[IDX_ODM] +
        k[3353]*y[IDX_ODM];
    data[2771] = 0.0 + k[505]*y[IDX_OHM];
    data[2772] = 0.0 - k[2380]*y[IDX_H2OI] - k[2995]*y[IDX_H2OI];
    data[2773] = 0.0 + k[572]*y[IDX_OHI] + k[2503]*y[IDX_H2OII];
    data[2774] = 0.0 - k[2996]*y[IDX_H2OI];
    data[2775] = 0.0 + k[1171]*y[IDX_OHI];
    data[2776] = 0.0 - k[2447]*y[IDX_H2OI] - k[3031]*y[IDX_H2OI] -
        k[3032]*y[IDX_H2OI];
    data[2777] = 0.0 + k[3103]*y[IDX_HM] + k[3104]*y[IDX_HM] +
        k[3293]*y[IDX_CHI] + k[3364]*y[IDX_OHM];
    data[2778] = 0.0 - k[3157]*y[IDX_H2OI] - k[3158]*y[IDX_H2OI];
    data[2779] = 0.0 - k[3007]*y[IDX_H2OI];
    data[2780] = 0.0 + k[576]*y[IDX_OHI];
    data[2781] = 0.0 - k[3008]*y[IDX_H2OI];
    data[2782] = 0.0 - k[903]*y[IDX_H2OI] - k[907]*y[IDX_H2OI] -
        k[2371]*y[IDX_H2OI];
    data[2783] = 0.0 - k[2231]*y[IDX_H2OI];
    data[2784] = 0.0 - k[1812]*y[IDX_H2OI] - k[2622]*y[IDX_H2OI];
    data[2785] = 0.0 - k[3278]*y[IDX_H2OI];
    data[2786] = 0.0 - k[3009]*y[IDX_H2OI];
    data[2787] = 0.0 - k[1891]*y[IDX_H2OI] - k[1892]*y[IDX_H2OI] -
        k[1901]*y[IDX_H2OI] - k[1902]*y[IDX_H2OI] - k[2626]*y[IDX_H2OI] -
        k[3311]*y[IDX_H2OI];
    data[2788] = 0.0 + k[2514]*y[IDX_H2OII];
    data[2789] = 0.0 - k[3323]*y[IDX_H2OI] - k[3324]*y[IDX_H2OI];
    data[2790] = 0.0 - k[1890]*y[IDX_H2OI] - k[1900]*y[IDX_H2OI] -
        k[2625]*y[IDX_H2OI] - k[3014]*y[IDX_H2OI];
    data[2791] = 0.0 + k[566]*y[IDX_NOI] + k[580]*y[IDX_OHI] +
        k[586]*y[IDX_ODI] + k[2511]*y[IDX_H2OII];
    data[2792] = 0.0 - k[3016]*y[IDX_H2OI];
    data[2793] = 0.0 + k[584]*y[IDX_OHI] + k[2517]*y[IDX_H2OII];
    data[2794] = 0.0 - k[3321]*y[IDX_H2OI] - k[3322]*y[IDX_H2OI];
    data[2795] = 0.0 + k[566]*y[IDX_NH2I] + k[2179]*y[IDX_H2OII];
    data[2796] = 0.0 + k[1997]*y[IDX_pH2I] + k[1998]*y[IDX_oH2I];
    data[2797] = 0.0 - k[2246]*y[IDX_H2OI];
    data[2798] = 0.0 + k[2723]*y[IDX_pH2I] + k[2724]*y[IDX_oH2I] +
        k[3018]*y[IDX_H3OII] + k[3342]*y[IDX_H2DOII];
    data[2799] = 0.0 + k[2182]*y[IDX_H2OII];
    data[2800] = 0.0 + k[590]*y[IDX_OHI] + k[1120]*y[IDX_HI];
    data[2801] = 0.0 + k[586]*y[IDX_NH2I] + k[1162]*y[IDX_pH2I] +
        k[1163]*y[IDX_oH2I];
    data[2802] = 0.0 - k[2593]*y[IDX_H2OI] - k[3316]*y[IDX_H2OI];
    data[2803] = 0.0 + k[3352]*y[IDX_H3OII] + k[3353]*y[IDX_H3OII] +
        k[3360]*y[IDX_H2DOII];
    data[2804] = 0.0 + k[569]*y[IDX_OHI] + k[569]*y[IDX_OHI] +
        k[572]*y[IDX_HCOI] + k[576]*y[IDX_HNOI] + k[580]*y[IDX_NH2I] +
        k[584]*y[IDX_NHDI] + k[590]*y[IDX_O2HI] + k[1158]*y[IDX_pH2I] +
        k[1159]*y[IDX_oH2I] + k[1171]*y[IDX_HDI] + k[2666]*y[IDX_HI] +
        k[2711]*y[IDX_HM];
    data[2805] = 0.0 - k[2592]*y[IDX_H2OI] - k[3015]*y[IDX_H2OI];
    data[2806] = 0.0 + k[505]*y[IDX_HCNI] + k[2732]*y[IDX_HI] +
        k[3020]*y[IDX_H3OII] + k[3355]*y[IDX_H2DOII] + k[3356]*y[IDX_H2DOII] +
        k[3364]*y[IDX_HD2OII];
    data[2807] = 0.0 + k[3026]*y[IDX_H3OII] + k[3446]*y[IDX_H2DOII];
    data[2808] = 0.0 - k[3192]*y[IDX_H2OI] - k[3196]*y[IDX_H2OI] -
        k[3198]*y[IDX_H2OI] - k[3201]*y[IDX_H2OI];
    data[2809] = 0.0 - k[2446]*y[IDX_H2OI] - k[3033]*y[IDX_H2OI] -
        k[3035]*y[IDX_H2OI];
    data[2810] = 0.0 - k[3183]*y[IDX_H2OI] - k[3184]*y[IDX_H2OI] -
        k[3187]*y[IDX_H2OI] - k[3189]*y[IDX_H2OI];
    data[2811] = 0.0 - k[3193]*y[IDX_H2OI] - k[3195]*y[IDX_H2OI] -
        k[3199]*y[IDX_H2OI] - k[3202]*y[IDX_H2OI] - k[3203]*y[IDX_H2OI];
    data[2812] = 0.0 + k[1159]*y[IDX_OHI] + k[1163]*y[IDX_ODI] +
        k[1998]*y[IDX_NO2II] + k[2724]*y[IDX_OM];
    data[2813] = 0.0 + k[791]*y[IDX_CO2I] - k[2444]*y[IDX_H2OI] -
        k[2989]*y[IDX_H2OI];
    data[2814] = 0.0 - k[3177]*y[IDX_H2OI] - k[3178]*y[IDX_H2OI] -
        k[3181]*y[IDX_H2OI];
    data[2815] = 0.0 - k[3003]*y[IDX_H2OI] - k[3004]*y[IDX_H2OI];
    data[2816] = 0.0 - k[2445]*y[IDX_H2OI] - k[3034]*y[IDX_H2OI] -
        k[3036]*y[IDX_H2OI];
    data[2817] = 0.0 - k[3185]*y[IDX_H2OI] - k[3186]*y[IDX_H2OI] -
        k[3188]*y[IDX_H2OI] - k[3190]*y[IDX_H2OI];
    data[2818] = 0.0 - k[3191]*y[IDX_H2OI] - k[3194]*y[IDX_H2OI] -
        k[3197]*y[IDX_H2OI] - k[3200]*y[IDX_H2OI];
    data[2819] = 0.0 + k[1158]*y[IDX_OHI] + k[1162]*y[IDX_ODI] +
        k[1997]*y[IDX_NO2II] + k[2723]*y[IDX_OM];
    data[2820] = 0.0 + k[790]*y[IDX_CO2I] - k[2443]*y[IDX_H2OI] -
        k[2990]*y[IDX_H2OI];
    data[2821] = 0.0 - k[3179]*y[IDX_H2OI] - k[3180]*y[IDX_H2OI] -
        k[3182]*y[IDX_H2OI];
    data[2822] = 0.0 - k[3005]*y[IDX_H2OI] - k[3006]*y[IDX_H2OI];
    data[2823] = 0.0 - k[987]*y[IDX_H2OII];
    data[2824] = 0.0 - k[1238]*y[IDX_H2OII] - k[2479]*y[IDX_H2OII];
    data[2825] = 0.0 - k[2491]*y[IDX_H2OII];
    data[2826] = 0.0 - k[2488]*y[IDX_H2OII];
    data[2827] = 0.0 - k[1247]*y[IDX_H2OII] - k[1248]*y[IDX_H2OII] -
        k[2485]*y[IDX_H2OII];
    data[2828] = 0.0 - k[2497]*y[IDX_H2OII];
    data[2829] = 0.0 - k[1242]*y[IDX_H2OII] - k[2482]*y[IDX_H2OII];
    data[2830] = 0.0 - k[2494]*y[IDX_H2OII];
    data[2831] = 0.0 - k[2500]*y[IDX_H2OII];
    data[2832] = 0.0 - k[1252]*y[IDX_H2OII];
    data[2833] = 0.0 + k[2334]*y[IDX_H2OI];
    data[2834] = 0.0 + k[2377]*y[IDX_H2OI];
    data[2835] = 0.0 + k[3385]*y[IDX_H3OII] + k[3389]*y[IDX_H2DOII] +
        k[3390]*y[IDX_H2DOII];
    data[2836] = 0.0 + k[2210]*y[IDX_H2OI];
    data[2837] = 0.0 - k[3167]*y[IDX_H2OII] - k[3168]*y[IDX_H2OII];
    data[2838] = 0.0 + k[2381]*y[IDX_H2OI];
    data[2839] = 0.0 - k[2506]*y[IDX_H2OII] - k[3174]*y[IDX_H2OII];
    data[2840] = 0.0 + k[3021]*y[IDX_H3OII] + k[3022]*y[IDX_H3OII] +
        k[3374]*y[IDX_H2DOII] + k[3378]*y[IDX_HD2OII] + k[3379]*y[IDX_HD2OII];
    data[2841] = 0.0 + k[2209]*y[IDX_H2OI];
    data[2842] = 0.0 + k[3374]*y[IDX_HI] + k[3389]*y[IDX_DI] +
        k[3390]*y[IDX_DI];
    data[2843] = 0.0 + k[397] + k[2209]*y[IDX_HII] + k[2210]*y[IDX_DII] +
        k[2231]*y[IDX_NII] + k[2246]*y[IDX_OII] + k[2334]*y[IDX_COII] +
        k[2371]*y[IDX_HeII] + k[2377]*y[IDX_CO2II] + k[2380]*y[IDX_HCNII] +
        k[2381]*y[IDX_DCNII] + k[2443]*y[IDX_pH2II] + k[2444]*y[IDX_oH2II] +
        k[2445]*y[IDX_pD2II] + k[2446]*y[IDX_oD2II] + k[2447]*y[IDX_HDII] +
        k[2592]*y[IDX_OHII] + k[2593]*y[IDX_ODII] + k[2622]*y[IDX_N2II] +
        k[2625]*y[IDX_NHII] + k[2626]*y[IDX_NDII] - k[3001]*y[IDX_H2OII];
    data[2844] = 0.0 - k[987]*y[IDX_CI] - k[991]*y[IDX_NI] -
        k[995]*y[IDX_OI] - k[1238]*y[IDX_C2I] - k[1242]*y[IDX_CHI] -
        k[1247]*y[IDX_CDI] - k[1248]*y[IDX_CDI] - k[1252]*y[IDX_COI] -
        k[2179]*y[IDX_NOI] - k[2182]*y[IDX_O2I] - k[2479]*y[IDX_C2I] -
        k[2482]*y[IDX_CHI] - k[2485]*y[IDX_CDI] - k[2488]*y[IDX_C2HI] -
        k[2491]*y[IDX_C2DI] - k[2494]*y[IDX_CH2I] - k[2497]*y[IDX_CD2I] -
        k[2500]*y[IDX_CHDI] - k[2503]*y[IDX_HCOI] - k[2506]*y[IDX_DCOI] -
        k[2511]*y[IDX_NH2I] - k[2514]*y[IDX_ND2I] - k[2517]*y[IDX_NHDI] -
        k[2788]*y[IDX_eM] - k[2791]*y[IDX_eM] - k[2795]*y[IDX_eM] -
        k[2997]*y[IDX_pH2I] - k[2998]*y[IDX_oH2I] - k[2999]*y[IDX_NHI] -
        k[3000]*y[IDX_OHI] - k[3001]*y[IDX_H2OI] - k[3002]*y[IDX_HCOI] -
        k[3131]*y[IDX_HDI] - k[3132]*y[IDX_HDI] - k[3137]*y[IDX_oD2I] -
        k[3138]*y[IDX_pD2I] - k[3139]*y[IDX_oD2I] - k[3140]*y[IDX_pD2I] -
        k[3149]*y[IDX_NDI] - k[3154]*y[IDX_ODI] - k[3161]*y[IDX_HDOI] -
        k[3162]*y[IDX_HDOI] - k[3167]*y[IDX_D2OI] - k[3168]*y[IDX_D2OI] -
        k[3174]*y[IDX_DCOI];
    data[2845] = 0.0 + k[3021]*y[IDX_HI] + k[3022]*y[IDX_HI] +
        k[3385]*y[IDX_DI];
    data[2846] = 0.0 + k[1012]*y[IDX_OHI] + k[2380]*y[IDX_H2OI];
    data[2847] = 0.0 + k[1965]*y[IDX_OHII] - k[2503]*y[IDX_H2OII] -
        k[3002]*y[IDX_H2OII];
    data[2848] = 0.0 + k[1028]*y[IDX_OHI];
    data[2849] = 0.0 + k[1950]*y[IDX_OHII] - k[3131]*y[IDX_H2OII] -
        k[3132]*y[IDX_H2OII];
    data[2850] = 0.0 + k[781]*y[IDX_OHI] + k[2447]*y[IDX_H2OI];
    data[2851] = 0.0 + k[3378]*y[IDX_HI] + k[3379]*y[IDX_HI];
    data[2852] = 0.0 - k[3161]*y[IDX_H2OII] - k[3162]*y[IDX_H2OII];
    data[2853] = 0.0 + k[1566]*y[IDX_OHI];
    data[2854] = 0.0 + k[1586]*y[IDX_OHI];
    data[2855] = 0.0 + k[2371]*y[IDX_H2OI];
    data[2856] = 0.0 - k[991]*y[IDX_H2OII];
    data[2857] = 0.0 + k[2231]*y[IDX_H2OI];
    data[2858] = 0.0 + k[2622]*y[IDX_H2OI];
    data[2859] = 0.0 + k[1630]*y[IDX_OHI];
    data[2860] = 0.0 - k[3149]*y[IDX_H2OII];
    data[2861] = 0.0 + k[2626]*y[IDX_H2OI];
    data[2862] = 0.0 - k[2514]*y[IDX_H2OII];
    data[2863] = 0.0 - k[2999]*y[IDX_H2OII];
    data[2864] = 0.0 + k[1882]*y[IDX_OHI] + k[2625]*y[IDX_H2OI];
    data[2865] = 0.0 - k[2511]*y[IDX_H2OII];
    data[2866] = 0.0 - k[2517]*y[IDX_H2OII];
    data[2867] = 0.0 - k[2179]*y[IDX_H2OII];
    data[2868] = 0.0 - k[995]*y[IDX_H2OII] + k[1298]*y[IDX_oH3II] +
        k[1299]*y[IDX_pH3II] + k[1304]*y[IDX_oH2DII] + k[1305]*y[IDX_pH2DII];
    data[2869] = 0.0 + k[2246]*y[IDX_H2OI];
    data[2870] = 0.0 - k[2182]*y[IDX_H2OII];
    data[2871] = 0.0 + k[2040]*y[IDX_OHI];
    data[2872] = 0.0 + k[784]*y[IDX_pH2II] + k[785]*y[IDX_oH2II] +
        k[1497]*y[IDX_oH3II] + k[1498]*y[IDX_pH3II] - k[3154]*y[IDX_H2OII];
    data[2873] = 0.0 + k[1941]*y[IDX_pH2I] + k[1942]*y[IDX_oH2I] +
        k[2593]*y[IDX_H2OI];
    data[2874] = 0.0 + k[774]*y[IDX_pH2II] + k[775]*y[IDX_oH2II] +
        k[781]*y[IDX_HDII] + k[1012]*y[IDX_HCNII] + k[1028]*y[IDX_HCOII] +
        k[1479]*y[IDX_oH3II] + k[1480]*y[IDX_pH3II] + k[1481]*y[IDX_pH3II] +
        k[1487]*y[IDX_oH2DII] + k[1488]*y[IDX_pH2DII] + k[1489]*y[IDX_oD2HII] +
        k[1490]*y[IDX_pD2HII] + k[1491]*y[IDX_pD2HII] + k[1566]*y[IDX_HNCII] +
        k[1586]*y[IDX_HNOII] + k[1630]*y[IDX_N2HII] + k[1882]*y[IDX_NHII] +
        k[1961]*y[IDX_OHII] + k[2040]*y[IDX_O2HII] - k[3000]*y[IDX_H2OII];
    data[2875] = 0.0 + k[1937]*y[IDX_pH2I] + k[1938]*y[IDX_oH2I] +
        k[1950]*y[IDX_HDI] + k[1961]*y[IDX_OHI] + k[1965]*y[IDX_HCOI] +
        k[2592]*y[IDX_H2OI];
    data[2876] = 0.0 - k[2788]*y[IDX_H2OII] - k[2791]*y[IDX_H2OII] -
        k[2795]*y[IDX_H2OII];
    data[2877] = 0.0 - k[3137]*y[IDX_H2OII] - k[3139]*y[IDX_H2OII];
    data[2878] = 0.0 + k[2446]*y[IDX_H2OI];
    data[2879] = 0.0 + k[1489]*y[IDX_OHI];
    data[2880] = 0.0 + k[1938]*y[IDX_OHII] + k[1942]*y[IDX_ODII] -
        k[2998]*y[IDX_H2OII];
    data[2881] = 0.0 + k[775]*y[IDX_OHI] + k[785]*y[IDX_ODI] +
        k[2444]*y[IDX_H2OI];
    data[2882] = 0.0 + k[1304]*y[IDX_OI] + k[1487]*y[IDX_OHI];
    data[2883] = 0.0 + k[1298]*y[IDX_OI] + k[1479]*y[IDX_OHI] +
        k[1497]*y[IDX_ODI];
    data[2884] = 0.0 - k[3138]*y[IDX_H2OII] - k[3140]*y[IDX_H2OII];
    data[2885] = 0.0 + k[2445]*y[IDX_H2OI];
    data[2886] = 0.0 + k[1490]*y[IDX_OHI] + k[1491]*y[IDX_OHI];
    data[2887] = 0.0 + k[1937]*y[IDX_OHII] + k[1941]*y[IDX_ODII] -
        k[2997]*y[IDX_H2OII];
    data[2888] = 0.0 + k[774]*y[IDX_OHI] + k[784]*y[IDX_ODI] +
        k[2443]*y[IDX_H2OI];
    data[2889] = 0.0 + k[1305]*y[IDX_OI] + k[1488]*y[IDX_OHI];
    data[2890] = 0.0 + k[1299]*y[IDX_OI] + k[1480]*y[IDX_OHI] +
        k[1481]*y[IDX_OHI] + k[1498]*y[IDX_ODI];
    data[2891] = 0.0 - k[3010]*y[IDX_H3OII] - k[3011]*y[IDX_H3OII];
    data[2892] = 0.0 - k[3017]*y[IDX_H3OII];
    data[2893] = 0.0 - k[3298]*y[IDX_H3OII] - k[3299]*y[IDX_H3OII];
    data[2894] = 0.0 - k[3012]*y[IDX_H3OII];
    data[2895] = 0.0 + k[3013]*y[IDX_H2OI];
    data[2896] = 0.0 - k[3019]*y[IDX_H3OII];
    data[2897] = 0.0 - k[3383]*y[IDX_H3OII] - k[3384]*y[IDX_H3OII] -
        k[3385]*y[IDX_H3OII];
    data[2898] = 0.0 - k[3055]*y[IDX_H3OII] - k[3056]*y[IDX_H3OII] -
        k[3057]*y[IDX_H3OII] - k[3058]*y[IDX_H3OII] - k[3059]*y[IDX_H3OII] -
        k[3089]*y[IDX_H3OII] - k[3090]*y[IDX_H3OII] - k[3091]*y[IDX_H3OII];
    data[2899] = 0.0 + k[3244]*y[IDX_pH3II] + k[3245]*y[IDX_oH3II] +
        k[3246]*y[IDX_pH3II] + k[3247]*y[IDX_oH3II];
    data[2900] = 0.0 - k[3021]*y[IDX_H3OII] - k[3022]*y[IDX_H3OII];
    data[2901] = 0.0 - k[2991]*y[IDX_H3OII] - k[2992]*y[IDX_H3OII] -
        k[2993]*y[IDX_H3OII] - k[2994]*y[IDX_H3OII];
    data[2902] = 0.0 + k[3398]*y[IDX_pH2I] + k[3399]*y[IDX_oH2I] +
        k[3416]*y[IDX_HDI];
    data[2903] = 0.0 + k[2989]*y[IDX_oH2II] + k[2990]*y[IDX_pH2II] +
        k[2995]*y[IDX_HCNII] + k[2996]*y[IDX_HCOII] + k[3001]*y[IDX_H2OII] +
        k[3003]*y[IDX_oH3II] + k[3004]*y[IDX_oH3II] + k[3005]*y[IDX_pH3II] +
        k[3006]*y[IDX_pH3II] + k[3007]*y[IDX_HNCII] + k[3008]*y[IDX_HNOII] +
        k[3009]*y[IDX_N2HII] + k[3013]*y[IDX_CHII] + k[3014]*y[IDX_NHII] +
        k[3015]*y[IDX_OHII] + k[3016]*y[IDX_NH2II] + k[3032]*y[IDX_HDII] +
        k[3158]*y[IDX_HDOII] + k[3181]*y[IDX_oH2DII] + k[3182]*y[IDX_pH2DII] +
        k[3189]*y[IDX_oD2HII] + k[3190]*y[IDX_pD2HII] + k[3322]*y[IDX_NHDII];
    data[2904] = 0.0 + k[2997]*y[IDX_pH2I] + k[2998]*y[IDX_oH2I] +
        k[2999]*y[IDX_NHI] + k[3000]*y[IDX_OHI] + k[3001]*y[IDX_H2OI] +
        k[3002]*y[IDX_HCOI] + k[3132]*y[IDX_HDI] + k[3162]*y[IDX_HDOI];
    data[2905] = 0.0 - k[2991]*y[IDX_HM] - k[2992]*y[IDX_HM] -
        k[2993]*y[IDX_HM] - k[2994]*y[IDX_HM] - k[3010]*y[IDX_CI] -
        k[3011]*y[IDX_CI] - k[3012]*y[IDX_CHI] - k[3017]*y[IDX_CM] -
        k[3018]*y[IDX_OM] - k[3019]*y[IDX_CNM] - k[3020]*y[IDX_OHM] -
        k[3021]*y[IDX_HI] - k[3022]*y[IDX_HI] - k[3023]*y[IDX_oH2I] +
        k[3023]*y[IDX_oH2I] - k[3024]*y[IDX_pH2I] + k[3024]*y[IDX_pH2I] -
        k[3025]*y[IDX_eM] - k[3026]*y[IDX_eM] - k[3027]*y[IDX_eM] -
        k[3028]*y[IDX_eM] - k[3029]*y[IDX_eM] - k[3030]*y[IDX_eM] -
        k[3055]*y[IDX_DM] - k[3056]*y[IDX_DM] - k[3057]*y[IDX_DM] -
        k[3058]*y[IDX_DM] - k[3059]*y[IDX_DM] - k[3089]*y[IDX_DM] -
        k[3090]*y[IDX_DM] - k[3091]*y[IDX_DM] - k[3298]*y[IDX_CDI] -
        k[3299]*y[IDX_CDI] - k[3351]*y[IDX_ODM] - k[3352]*y[IDX_ODM] -
        k[3353]*y[IDX_ODM] - k[3383]*y[IDX_DI] - k[3384]*y[IDX_DI] -
        k[3385]*y[IDX_DI] - k[3412]*y[IDX_HDI] - k[3413]*y[IDX_HDI] +
        k[3413]*y[IDX_HDI] - k[3422]*y[IDX_pD2I] - k[3423]*y[IDX_oD2I] -
        k[3424]*y[IDX_pD2I] - k[3425]*y[IDX_oD2I] - k[3426]*y[IDX_pD2I] +
        k[3426]*y[IDX_pD2I] - k[3427]*y[IDX_oD2I] + k[3427]*y[IDX_oD2I];
    data[2906] = 0.0 + k[2995]*y[IDX_H2OI];
    data[2907] = 0.0 + k[3002]*y[IDX_H2OII];
    data[2908] = 0.0 + k[2996]*y[IDX_H2OI];
    data[2909] = 0.0 + k[3132]*y[IDX_H2OII] - k[3412]*y[IDX_H3OII] -
        k[3413]*y[IDX_H3OII] + k[3413]*y[IDX_H3OII] + k[3416]*y[IDX_H2DOII];
    data[2910] = 0.0 + k[3032]*y[IDX_H2OI];
    data[2911] = 0.0 + k[3404]*y[IDX_pH2I] + k[3405]*y[IDX_oH2I];
    data[2912] = 0.0 + k[3039]*y[IDX_pH2II] + k[3040]*y[IDX_oH2II] +
        k[3162]*y[IDX_H2OII] + k[3208]*y[IDX_oH3II] + k[3209]*y[IDX_pH3II] +
        k[3216]*y[IDX_pH2DII] + k[3217]*y[IDX_pH2DII] + k[3218]*y[IDX_oH2DII] +
        k[3219]*y[IDX_oH2DII] + k[3326]*y[IDX_NH2II];
    data[2913] = 0.0 + k[3125]*y[IDX_pH2I] + k[3126]*y[IDX_oH2I] +
        k[3158]*y[IDX_H2OI];
    data[2914] = 0.0 + k[3007]*y[IDX_H2OI];
    data[2915] = 0.0 + k[3008]*y[IDX_H2OI];
    data[2916] = 0.0 + k[3009]*y[IDX_H2OI];
    data[2917] = 0.0 + k[2999]*y[IDX_H2OII];
    data[2918] = 0.0 + k[3014]*y[IDX_H2OI];
    data[2919] = 0.0 + k[3016]*y[IDX_H2OI] + k[3326]*y[IDX_HDOI];
    data[2920] = 0.0 + k[3322]*y[IDX_H2OI];
    data[2921] = 0.0 - k[3018]*y[IDX_H3OII];
    data[2922] = 0.0 - k[3351]*y[IDX_H3OII] - k[3352]*y[IDX_H3OII] -
        k[3353]*y[IDX_H3OII];
    data[2923] = 0.0 + k[3000]*y[IDX_H2OII];
    data[2924] = 0.0 + k[3015]*y[IDX_H2OI];
    data[2925] = 0.0 - k[3020]*y[IDX_H3OII];
    data[2926] = 0.0 - k[3025]*y[IDX_H3OII] - k[3026]*y[IDX_H3OII] -
        k[3027]*y[IDX_H3OII] - k[3028]*y[IDX_H3OII] - k[3029]*y[IDX_H3OII] -
        k[3030]*y[IDX_H3OII];
    data[2927] = 0.0 - k[3423]*y[IDX_H3OII] - k[3425]*y[IDX_H3OII] -
        k[3427]*y[IDX_H3OII] + k[3427]*y[IDX_H3OII];
    data[2928] = 0.0 + k[3189]*y[IDX_H2OI];
    data[2929] = 0.0 + k[2998]*y[IDX_H2OII] - k[3023]*y[IDX_H3OII] +
        k[3023]*y[IDX_H3OII] + k[3126]*y[IDX_HDOII] + k[3399]*y[IDX_H2DOII] +
        k[3405]*y[IDX_HD2OII];
    data[2930] = 0.0 + k[2989]*y[IDX_H2OI] + k[3040]*y[IDX_HDOI];
    data[2931] = 0.0 + k[3181]*y[IDX_H2OI] + k[3218]*y[IDX_HDOI] +
        k[3219]*y[IDX_HDOI];
    data[2932] = 0.0 + k[3003]*y[IDX_H2OI] + k[3004]*y[IDX_H2OI] +
        k[3208]*y[IDX_HDOI] + k[3245]*y[IDX_D2OI] + k[3247]*y[IDX_D2OI];
    data[2933] = 0.0 - k[3422]*y[IDX_H3OII] - k[3424]*y[IDX_H3OII] -
        k[3426]*y[IDX_H3OII] + k[3426]*y[IDX_H3OII];
    data[2934] = 0.0 + k[3190]*y[IDX_H2OI];
    data[2935] = 0.0 + k[2997]*y[IDX_H2OII] - k[3024]*y[IDX_H3OII] +
        k[3024]*y[IDX_H3OII] + k[3125]*y[IDX_HDOII] + k[3398]*y[IDX_H2DOII] +
        k[3404]*y[IDX_HD2OII];
    data[2936] = 0.0 + k[2990]*y[IDX_H2OI] + k[3039]*y[IDX_HDOI];
    data[2937] = 0.0 + k[3182]*y[IDX_H2OI] + k[3216]*y[IDX_HDOI] +
        k[3217]*y[IDX_HDOI];
    data[2938] = 0.0 + k[3005]*y[IDX_H2OI] + k[3006]*y[IDX_H2OI] +
        k[3209]*y[IDX_HDOI] + k[3244]*y[IDX_D2OI] + k[3246]*y[IDX_D2OI];
    data[2939] = 0.0 + k[1049]*y[IDX_NH2I] + k[1052]*y[IDX_NHDI];
    data[2940] = 0.0 - k[443]*y[IDX_HCNI] - k[445]*y[IDX_HCNI];
    data[2941] = 0.0 + k[2682]*y[IDX_NHI];
    data[2942] = 0.0 + k[960]*y[IDX_H2OI] + k[962]*y[IDX_HDOI];
    data[2943] = 0.0 - k[1788]*y[IDX_HCNI];
    data[2944] = 0.0 + k[537]*y[IDX_NOI];
    data[2945] = 0.0 - k[1787]*y[IDX_HCNI];
    data[2946] = 0.0 + k[1206]*y[IDX_NI];
    data[2947] = 0.0 + k[1209]*y[IDX_NI];
    data[2948] = 0.0 + k[549]*y[IDX_HCOI] + k[553]*y[IDX_OHI] +
        k[555]*y[IDX_HNOI] + k[2703]*y[IDX_HM];
    data[2949] = 0.0 - k[2319]*y[IDX_HCNI];
    data[2950] = 0.0 + k[2728]*y[IDX_HI];
    data[2951] = 0.0 + k[980]*y[IDX_H2OI] + k[982]*y[IDX_HDOI];
    data[2952] = 0.0 - k[2089]*y[IDX_HCNI];
    data[2953] = 0.0 - k[1094]*y[IDX_HCNI] + k[2171]*y[IDX_HCNII];
    data[2954] = 0.0 + k[1680]*y[IDX_HNCI] - k[2216]*y[IDX_HCNI];
    data[2955] = 0.0 - k[494]*y[IDX_HCNI];
    data[2956] = 0.0 + k[2382]*y[IDX_HCNII];
    data[2957] = 0.0 + k[1682]*y[IDX_HII];
    data[2958] = 0.0 - k[1092]*y[IDX_HCNI] + k[2169]*y[IDX_HCNII] +
        k[2728]*y[IDX_CNM];
    data[2959] = 0.0 + k[1679]*y[IDX_HNCI] + k[1682]*y[IDX_DNCI] -
        k[2215]*y[IDX_HCNI];
    data[2960] = 0.0 - k[493]*y[IDX_HCNI] + k[2703]*y[IDX_CNI];
    data[2961] = 0.0 + k[960]*y[IDX_C2NII] + k[980]*y[IDX_CNCII] +
        k[2380]*y[IDX_HCNII];
    data[2962] = 0.0 - k[257] - k[400] - k[443]*y[IDX_CII] -
        k[445]*y[IDX_CII] - k[493]*y[IDX_HM] - k[494]*y[IDX_DM] -
        k[503]*y[IDX_OM] - k[505]*y[IDX_OHM] - k[506]*y[IDX_ODM] -
        k[519]*y[IDX_OI] - k[911]*y[IDX_HeII] - k[913]*y[IDX_HeII] -
        k[915]*y[IDX_HeII] - k[917]*y[IDX_HeII] - k[1092]*y[IDX_HI] -
        k[1094]*y[IDX_DI] - k[1670]*y[IDX_OII] - k[1672]*y[IDX_OII] -
        k[1674]*y[IDX_OII] - k[1787]*y[IDX_CHII] - k[1788]*y[IDX_CDII] -
        k[2089]*y[IDX_COII] - k[2103]*y[IDX_pH2II] - k[2104]*y[IDX_oH2II] -
        k[2105]*y[IDX_pD2II] - k[2106]*y[IDX_oD2II] - k[2107]*y[IDX_HDII] -
        k[2215]*y[IDX_HII] - k[2216]*y[IDX_DII] - k[2234]*y[IDX_NII] -
        k[2288]*y[IDX_N2II] - k[2319]*y[IDX_CNII];
    data[2963] = 0.0 + k[2169]*y[IDX_HI] + k[2171]*y[IDX_DI] +
        k[2173]*y[IDX_OI] + k[2175]*y[IDX_NOI] + k[2177]*y[IDX_O2I] +
        k[2380]*y[IDX_H2OI] + k[2382]*y[IDX_D2OI] + k[2384]*y[IDX_HDOI];
    data[2964] = 0.0 + k[549]*y[IDX_CNI];
    data[2965] = 0.0 - k[2107]*y[IDX_HCNI];
    data[2966] = 0.0 + k[962]*y[IDX_C2NII] + k[982]*y[IDX_CNCII] +
        k[2384]*y[IDX_HCNII];
    data[2967] = 0.0 + k[1679]*y[IDX_HII] + k[1680]*y[IDX_DII];
    data[2968] = 0.0 + k[555]*y[IDX_CNI];
    data[2969] = 0.0 - k[911]*y[IDX_HCNI] - k[913]*y[IDX_HCNI] -
        k[915]*y[IDX_HCNI] - k[917]*y[IDX_HCNI];
    data[2970] = 0.0 + k[1206]*y[IDX_CH2I] + k[1209]*y[IDX_CHDI];
    data[2971] = 0.0 - k[2234]*y[IDX_HCNI];
    data[2972] = 0.0 - k[2288]*y[IDX_HCNI];
    data[2973] = 0.0 + k[2682]*y[IDX_CM];
    data[2974] = 0.0 + k[1049]*y[IDX_CI];
    data[2975] = 0.0 + k[1052]*y[IDX_CI];
    data[2976] = 0.0 + k[537]*y[IDX_CHI] + k[2175]*y[IDX_HCNII];
    data[2977] = 0.0 - k[519]*y[IDX_HCNI] + k[2173]*y[IDX_HCNII];
    data[2978] = 0.0 - k[1670]*y[IDX_HCNI] - k[1672]*y[IDX_HCNI] -
        k[1674]*y[IDX_HCNI];
    data[2979] = 0.0 - k[503]*y[IDX_HCNI];
    data[2980] = 0.0 + k[2177]*y[IDX_HCNII];
    data[2981] = 0.0 - k[506]*y[IDX_HCNI];
    data[2982] = 0.0 + k[553]*y[IDX_CNI];
    data[2983] = 0.0 - k[505]*y[IDX_HCNI];
    data[2984] = 0.0 - k[2106]*y[IDX_HCNI];
    data[2985] = 0.0 - k[2104]*y[IDX_HCNI];
    data[2986] = 0.0 - k[2105]*y[IDX_HCNI];
    data[2987] = 0.0 - k[2103]*y[IDX_HCNI];
    data[2988] = 0.0 - k[998]*y[IDX_HCNII];
    data[2989] = 0.0 + k[451]*y[IDX_NH2I] + k[454]*y[IDX_NHDI];
    data[2990] = 0.0 - k[1000]*y[IDX_HCNII] + k[1828]*y[IDX_NHII];
    data[2991] = 0.0 - k[1004]*y[IDX_HCNII];
    data[2992] = 0.0 + k[1797]*y[IDX_NH2I] + k[1803]*y[IDX_NHDI];
    data[2993] = 0.0 - k[1002]*y[IDX_HCNII];
    data[2994] = 0.0 + k[1795]*y[IDX_NH2I] + k[1798]*y[IDX_ND2I] +
        k[1802]*y[IDX_NHDI];
    data[2995] = 0.0 + k[968]*y[IDX_NI];
    data[2996] = 0.0 + k[971]*y[IDX_NI];
    data[2997] = 0.0 + k[670]*y[IDX_pH2II] + k[671]*y[IDX_oH2II] +
        k[675]*y[IDX_HDII] + k[1357]*y[IDX_oH3II] + k[1358]*y[IDX_pH3II] +
        k[1359]*y[IDX_pH3II] + k[1365]*y[IDX_oH2DII] + k[1366]*y[IDX_pH2DII] +
        k[1367]*y[IDX_oD2HII] + k[1368]*y[IDX_pD2HII] + k[1369]*y[IDX_pD2HII] +
        k[1834]*y[IDX_NHII] + k[1933]*y[IDX_OHII] + k[2014]*y[IDX_O2HII];
    data[2998] = 0.0 + k[474]*y[IDX_pH2I] + k[475]*y[IDX_oH2I] +
        k[479]*y[IDX_HDI] + k[599]*y[IDX_H2OI] + k[602]*y[IDX_HDOI] +
        k[607]*y[IDX_HCOI] + k[2319]*y[IDX_HCNI];
    data[2999] = 0.0 - k[1006]*y[IDX_HCNII];
    data[3000] = 0.0 + k[2089]*y[IDX_HCNI];
    data[3001] = 0.0 - k[2171]*y[IDX_HCNII];
    data[3002] = 0.0 + k[2216]*y[IDX_HCNI];
    data[3003] = 0.0 - k[2382]*y[IDX_HCNII] - k[3116]*y[IDX_HCNII];
    data[3004] = 0.0 - k[2169]*y[IDX_HCNII];
    data[3005] = 0.0 + k[2215]*y[IDX_HCNI];
    data[3006] = 0.0 + k[599]*y[IDX_CNII] - k[2380]*y[IDX_HCNII] -
        k[2995]*y[IDX_HCNII];
    data[3007] = 0.0 + k[2089]*y[IDX_COII] + k[2103]*y[IDX_pH2II] +
        k[2104]*y[IDX_oH2II] + k[2105]*y[IDX_pD2II] + k[2106]*y[IDX_oD2II] +
        k[2107]*y[IDX_HDII] + k[2215]*y[IDX_HII] + k[2216]*y[IDX_DII] +
        k[2234]*y[IDX_NII] + k[2288]*y[IDX_N2II] + k[2319]*y[IDX_CNII];
    data[3008] = 0.0 - k[998]*y[IDX_CI] - k[1000]*y[IDX_C2I] -
        k[1002]*y[IDX_CHI] - k[1004]*y[IDX_CDI] - k[1006]*y[IDX_COI] -
        k[1008]*y[IDX_NHI] - k[1010]*y[IDX_NDI] - k[1012]*y[IDX_OHI] -
        k[1014]*y[IDX_ODI] - k[2169]*y[IDX_HI] - k[2171]*y[IDX_DI] -
        k[2173]*y[IDX_OI] - k[2175]*y[IDX_NOI] - k[2177]*y[IDX_O2I] -
        k[2380]*y[IDX_H2OI] - k[2382]*y[IDX_D2OI] - k[2384]*y[IDX_HDOI] -
        k[2820]*y[IDX_eM] - k[2995]*y[IDX_H2OI] - k[3114]*y[IDX_HDOI] -
        k[3116]*y[IDX_D2OI];
    data[3009] = 0.0 + k[607]*y[IDX_CNII];
    data[3010] = 0.0 + k[479]*y[IDX_CNII];
    data[3011] = 0.0 + k[675]*y[IDX_CNI] + k[2107]*y[IDX_HCNI];
    data[3012] = 0.0 + k[602]*y[IDX_CNII] - k[2384]*y[IDX_HCNII] -
        k[3114]*y[IDX_HCNII];
    data[3013] = 0.0 + k[968]*y[IDX_CH2II] + k[971]*y[IDX_CHDII];
    data[3014] = 0.0 + k[2234]*y[IDX_HCNI];
    data[3015] = 0.0 + k[2288]*y[IDX_HCNI];
    data[3016] = 0.0 - k[1010]*y[IDX_HCNII];
    data[3017] = 0.0 + k[1798]*y[IDX_CHII];
    data[3018] = 0.0 - k[1008]*y[IDX_HCNII];
    data[3019] = 0.0 + k[1828]*y[IDX_C2I] + k[1834]*y[IDX_CNI];
    data[3020] = 0.0 + k[451]*y[IDX_CII] + k[1795]*y[IDX_CHII] +
        k[1797]*y[IDX_CDII];
    data[3021] = 0.0 + k[454]*y[IDX_CII] + k[1802]*y[IDX_CHII] +
        k[1803]*y[IDX_CDII];
    data[3022] = 0.0 - k[2175]*y[IDX_HCNII];
    data[3023] = 0.0 - k[2173]*y[IDX_HCNII];
    data[3024] = 0.0 - k[2177]*y[IDX_HCNII];
    data[3025] = 0.0 + k[2014]*y[IDX_CNI];
    data[3026] = 0.0 - k[1014]*y[IDX_HCNII];
    data[3027] = 0.0 - k[1012]*y[IDX_HCNII];
    data[3028] = 0.0 + k[1933]*y[IDX_CNI];
    data[3029] = 0.0 - k[2820]*y[IDX_HCNII];
    data[3030] = 0.0 + k[2106]*y[IDX_HCNI];
    data[3031] = 0.0 + k[1367]*y[IDX_CNI];
    data[3032] = 0.0 + k[475]*y[IDX_CNII];
    data[3033] = 0.0 + k[671]*y[IDX_CNI] + k[2104]*y[IDX_HCNI];
    data[3034] = 0.0 + k[1365]*y[IDX_CNI];
    data[3035] = 0.0 + k[1357]*y[IDX_CNI];
    data[3036] = 0.0 + k[2105]*y[IDX_HCNI];
    data[3037] = 0.0 + k[1368]*y[IDX_CNI] + k[1369]*y[IDX_CNI];
    data[3038] = 0.0 + k[474]*y[IDX_CNII];
    data[3039] = 0.0 + k[670]*y[IDX_CNI] + k[2103]*y[IDX_HCNI];
    data[3040] = 0.0 + k[1366]*y[IDX_CNI];
    data[3041] = 0.0 + k[1358]*y[IDX_CNI] + k[1359]*y[IDX_CNI];
    data[3042] = 0.0 - k[2055]*y[IDX_HCOI] - k[2057]*y[IDX_HCOI] +
        k[2730]*y[IDX_OHM];
    data[3043] = 0.0 - k[447]*y[IDX_HCOI] - k[2314]*y[IDX_HCOI];
    data[3044] = 0.0 + k[2685]*y[IDX_OHI];
    data[3045] = 0.0 - k[1709]*y[IDX_HCOI] - k[2619]*y[IDX_HCOI];
    data[3046] = 0.0 + k[1032]*y[IDX_O2I];
    data[3047] = 0.0 - k[1792]*y[IDX_HCOI] - k[2571]*y[IDX_HCOI];
    data[3048] = 0.0 + k[2720]*y[IDX_OM];
    data[3049] = 0.0 + k[1751]*y[IDX_O2I] - k[1791]*y[IDX_HCOI] -
        k[2570]*y[IDX_HCOI];
    data[3050] = 0.0 - k[549]*y[IDX_HCOI];
    data[3051] = 0.0 - k[607]*y[IDX_HCOI] - k[2321]*y[IDX_HCOI];
    data[3052] = 0.0 + k[2705]*y[IDX_HM];
    data[3053] = 0.0 - k[2091]*y[IDX_HCOI];
    data[3054] = 0.0 + k[1886]*y[IDX_NHII];
    data[3055] = 0.0 - k[1058]*y[IDX_HCOI] - k[1062]*y[IDX_HCOI];
    data[3056] = 0.0 - k[1545]*y[IDX_HCOI] - k[1549]*y[IDX_HCOI] -
        k[2561]*y[IDX_HCOI];
    data[3057] = 0.0 - k[2504]*y[IDX_HCOI] - k[3173]*y[IDX_HCOI];
    data[3058] = 0.0 - k[1056]*y[IDX_HCOI] - k[1060]*y[IDX_HCOI];
    data[3059] = 0.0 - k[1544]*y[IDX_HCOI] - k[1548]*y[IDX_HCOI] -
        k[2560]*y[IDX_HCOI];
    data[3060] = 0.0 + k[2705]*y[IDX_COI];
    data[3061] = 0.0 - k[2503]*y[IDX_HCOI] - k[3002]*y[IDX_HCOI];
    data[3062] = 0.0 - k[259] - k[261] - k[402] - k[404] - k[447]*y[IDX_CII]
        - k[515]*y[IDX_OI] - k[517]*y[IDX_OI] - k[549]*y[IDX_CNI] -
        k[572]*y[IDX_OHI] - k[574]*y[IDX_ODI] - k[607]*y[IDX_CNII] -
        k[795]*y[IDX_oH2II] - k[796]*y[IDX_pD2II] - k[797]*y[IDX_oD2II] -
        k[798]*y[IDX_oD2II] - k[799]*y[IDX_HDII] - k[800]*y[IDX_HDII] -
        k[919]*y[IDX_HeII] - k[921]*y[IDX_HeII] - k[923]*y[IDX_HeII] -
        k[1056]*y[IDX_HI] - k[1058]*y[IDX_DI] - k[1060]*y[IDX_HI] -
        k[1062]*y[IDX_DI] - k[1214]*y[IDX_NI] - k[1544]*y[IDX_HII] -
        k[1545]*y[IDX_DII] - k[1548]*y[IDX_HII] - k[1549]*y[IDX_DII] -
        k[1650]*y[IDX_NII] - k[1676]*y[IDX_OII] - k[1709]*y[IDX_C2II] -
        k[1791]*y[IDX_CHII] - k[1792]*y[IDX_CDII] - k[1816]*y[IDX_N2II] -
        k[1919]*y[IDX_O2II] - k[1965]*y[IDX_OHII] - k[1966]*y[IDX_ODII] -
        k[2055]*y[IDX_CI] - k[2057]*y[IDX_CI] - k[2091]*y[IDX_COII] -
        k[2303]*y[IDX_NH2II] - k[2304]*y[IDX_ND2II] - k[2305]*y[IDX_NHDII] -
        k[2314]*y[IDX_CII] - k[2321]*y[IDX_CNII] - k[2369]*y[IDX_OII] -
        k[2458]*y[IDX_pH2II] - k[2459]*y[IDX_oH2II] - k[2460]*y[IDX_pD2II] -
        k[2461]*y[IDX_oD2II] - k[2462]*y[IDX_HDII] - k[2503]*y[IDX_H2OII] -
        k[2504]*y[IDX_D2OII] - k[2505]*y[IDX_HDOII] - k[2534]*y[IDX_NII] -
        k[2560]*y[IDX_HII] - k[2561]*y[IDX_DII] - k[2570]*y[IDX_CHII] -
        k[2571]*y[IDX_CDII] - k[2574]*y[IDX_N2II] - k[2598]*y[IDX_OHII] -
        k[2599]*y[IDX_ODII] - k[2619]*y[IDX_C2II] - k[2636]*y[IDX_O2II] -
        k[2921]*y[IDX_oH2II] - k[2922]*y[IDX_pH2II] - k[3002]*y[IDX_H2OII] -
        k[3172]*y[IDX_HDOII] - k[3173]*y[IDX_D2OII];
    data[3063] = 0.0 - k[799]*y[IDX_HCOI] - k[800]*y[IDX_HCOI] -
        k[2462]*y[IDX_HCOI];
    data[3064] = 0.0 - k[2505]*y[IDX_HCOI] - k[3172]*y[IDX_HCOI];
    data[3065] = 0.0 + k[1564]*y[IDX_O2I];
    data[3066] = 0.0 - k[919]*y[IDX_HCOI] - k[921]*y[IDX_HCOI] -
        k[923]*y[IDX_HCOI];
    data[3067] = 0.0 - k[1214]*y[IDX_HCOI];
    data[3068] = 0.0 - k[1650]*y[IDX_HCOI] - k[2534]*y[IDX_HCOI];
    data[3069] = 0.0 - k[1816]*y[IDX_HCOI] - k[2574]*y[IDX_HCOI];
    data[3070] = 0.0 - k[2304]*y[IDX_HCOI];
    data[3071] = 0.0 + k[1886]*y[IDX_CO2I];
    data[3072] = 0.0 - k[2303]*y[IDX_HCOI];
    data[3073] = 0.0 - k[2305]*y[IDX_HCOI];
    data[3074] = 0.0 - k[515]*y[IDX_HCOI] - k[517]*y[IDX_HCOI];
    data[3075] = 0.0 - k[1676]*y[IDX_HCOI] - k[2369]*y[IDX_HCOI];
    data[3076] = 0.0 + k[2720]*y[IDX_CHI];
    data[3077] = 0.0 + k[1032]*y[IDX_C2HI] + k[1564]*y[IDX_HNCII] +
        k[1751]*y[IDX_CHII];
    data[3078] = 0.0 - k[1919]*y[IDX_HCOI] - k[2636]*y[IDX_HCOI];
    data[3079] = 0.0 - k[574]*y[IDX_HCOI];
    data[3080] = 0.0 - k[1966]*y[IDX_HCOI] - k[2599]*y[IDX_HCOI];
    data[3081] = 0.0 - k[572]*y[IDX_HCOI] + k[2685]*y[IDX_CM];
    data[3082] = 0.0 - k[1965]*y[IDX_HCOI] - k[2598]*y[IDX_HCOI];
    data[3083] = 0.0 + k[2730]*y[IDX_CI];
    data[3084] = 0.0 - k[797]*y[IDX_HCOI] - k[798]*y[IDX_HCOI] -
        k[2461]*y[IDX_HCOI];
    data[3085] = 0.0 - k[795]*y[IDX_HCOI] - k[2459]*y[IDX_HCOI] -
        k[2921]*y[IDX_HCOI];
    data[3086] = 0.0 - k[796]*y[IDX_HCOI] - k[2460]*y[IDX_HCOI];
    data[3087] = 0.0 - k[2458]*y[IDX_HCOI] - k[2922]*y[IDX_HCOI];
    data[3088] = 0.0 - k[1016]*y[IDX_HCOII] + k[3010]*y[IDX_H3OII] +
        k[3011]*y[IDX_H3OII] + k[3285]*y[IDX_H2DOII] + k[3287]*y[IDX_HD2OII] +
        k[3288]*y[IDX_HD2OII];
    data[3089] = 0.0 + k[435]*y[IDX_H2OI] + k[438]*y[IDX_HDOI] +
        k[2314]*y[IDX_HCOI];
    data[3090] = 0.0 - k[1018]*y[IDX_HCOII];
    data[3091] = 0.0 + k[2619]*y[IDX_HCOI];
    data[3092] = 0.0 + k[616]*y[IDX_COII];
    data[3093] = 0.0 + k[954]*y[IDX_OI];
    data[3094] = 0.0 + k[960]*y[IDX_H2OI] + k[963]*y[IDX_HDOI];
    data[3095] = 0.0 - k[1022]*y[IDX_HCOII];
    data[3096] = 0.0 + k[1779]*y[IDX_H2OI] + k[1785]*y[IDX_HDOI] +
        k[2571]*y[IDX_HCOI];
    data[3097] = 0.0 + k[610]*y[IDX_COII] - k[1020]*y[IDX_HCOII] +
        k[1913]*y[IDX_O2II] + k[2044]*y[IDX_OI];
    data[3098] = 0.0 + k[1755]*y[IDX_O2I] + k[1775]*y[IDX_CO2I] +
        k[1777]*y[IDX_H2OI] + k[1780]*y[IDX_D2OI] + k[1784]*y[IDX_HDOI] +
        k[2570]*y[IDX_HCOI];
    data[3099] = 0.0 + k[618]*y[IDX_COII];
    data[3100] = 0.0 + k[972]*y[IDX_OI] + k[976]*y[IDX_O2I];
    data[3101] = 0.0 + k[621]*y[IDX_COII];
    data[3102] = 0.0 + k[975]*y[IDX_OI] + k[979]*y[IDX_O2I];
    data[3103] = 0.0 + k[603]*y[IDX_H2OI] + k[606]*y[IDX_HDOI] +
        k[2321]*y[IDX_HCOI];
    data[3104] = 0.0 + k[980]*y[IDX_H2OI] + k[983]*y[IDX_HDOI];
    data[3105] = 0.0 + k[676]*y[IDX_pH2II] + k[677]*y[IDX_oH2II] +
        k[681]*y[IDX_HDII] + k[1006]*y[IDX_HCNII] + k[1252]*y[IDX_H2OII] +
        k[1255]*y[IDX_HDOII] + k[1372]*y[IDX_oH3II] + k[1373]*y[IDX_pH3II] +
        k[1374]*y[IDX_pH3II] + k[1380]*y[IDX_oH2DII] + k[1381]*y[IDX_pH2DII] +
        k[1382]*y[IDX_oD2HII] + k[1383]*y[IDX_pD2HII] + k[1384]*y[IDX_pD2HII] +
        k[1580]*y[IDX_HNOII] + k[1590]*y[IDX_HOCII] + k[1624]*y[IDX_N2HII] +
        k[1836]*y[IDX_NHII] + k[1935]*y[IDX_OHII] + k[2016]*y[IDX_O2HII];
    data[3106] = 0.0 + k[462]*y[IDX_pH2I] + k[463]*y[IDX_oH2I] +
        k[467]*y[IDX_HDI] + k[610]*y[IDX_CHI] + k[612]*y[IDX_NHI] +
        k[614]*y[IDX_OHI] + k[616]*y[IDX_C2HI] + k[618]*y[IDX_CH2I] +
        k[621]*y[IDX_CHDI] + k[622]*y[IDX_H2OI] + k[625]*y[IDX_HDOI] +
        k[626]*y[IDX_NH2I] + k[629]*y[IDX_NHDI] + k[2091]*y[IDX_HCOI];
    data[3107] = 0.0 + k[1542]*y[IDX_HII] + k[1775]*y[IDX_CHII];
    data[3108] = 0.0 + k[984]*y[IDX_HI];
    data[3109] = 0.0 + k[2561]*y[IDX_HCOI];
    data[3110] = 0.0 - k[859]*y[IDX_HCOII];
    data[3111] = 0.0 + k[1780]*y[IDX_CHII] - k[3121]*y[IDX_HCOII];
    data[3112] = 0.0 + k[2504]*y[IDX_HCOI];
    data[3113] = 0.0 + k[1598]*y[IDX_pH2I] + k[1599]*y[IDX_oH2I] +
        k[1611]*y[IDX_HDI] + k[1612]*y[IDX_HDI];
    data[3114] = 0.0 - k[200]*y[IDX_HCOII];
    data[3115] = 0.0 + k[984]*y[IDX_CO2II];
    data[3116] = 0.0 + k[1542]*y[IDX_CO2I] + k[2560]*y[IDX_HCOI];
    data[3117] = 0.0 - k[858]*y[IDX_HCOII];
    data[3118] = 0.0 + k[3285]*y[IDX_CI];
    data[3119] = 0.0 + k[435]*y[IDX_CII] + k[603]*y[IDX_CNII] +
        k[622]*y[IDX_COII] + k[960]*y[IDX_C2NII] + k[980]*y[IDX_CNCII] +
        k[1777]*y[IDX_CHII] + k[1779]*y[IDX_CDII] - k[2996]*y[IDX_HCOII];
    data[3120] = 0.0 + k[1252]*y[IDX_COI] + k[2503]*y[IDX_HCOI];
    data[3121] = 0.0 + k[3010]*y[IDX_CI] + k[3011]*y[IDX_CI];
    data[3122] = 0.0 + k[1674]*y[IDX_OII];
    data[3123] = 0.0 + k[1006]*y[IDX_COI];
    data[3124] = 0.0 + k[261] + k[404] + k[2091]*y[IDX_COII] +
        k[2303]*y[IDX_NH2II] + k[2304]*y[IDX_ND2II] + k[2305]*y[IDX_NHDII] +
        k[2314]*y[IDX_CII] + k[2321]*y[IDX_CNII] + k[2369]*y[IDX_OII] +
        k[2458]*y[IDX_pH2II] + k[2459]*y[IDX_oH2II] + k[2460]*y[IDX_pD2II] +
        k[2461]*y[IDX_oD2II] + k[2462]*y[IDX_HDII] + k[2503]*y[IDX_H2OII] +
        k[2504]*y[IDX_D2OII] + k[2505]*y[IDX_HDOII] + k[2534]*y[IDX_NII] +
        k[2560]*y[IDX_HII] + k[2561]*y[IDX_DII] + k[2570]*y[IDX_CHII] +
        k[2571]*y[IDX_CDII] + k[2574]*y[IDX_N2II] + k[2598]*y[IDX_OHII] +
        k[2599]*y[IDX_ODII] + k[2619]*y[IDX_C2II] + k[2636]*y[IDX_O2II];
    data[3125] = 0.0 - k[200]*y[IDX_GRAINM] - k[858]*y[IDX_HM] -
        k[859]*y[IDX_DM] - k[1016]*y[IDX_CI] - k[1018]*y[IDX_C2I] -
        k[1020]*y[IDX_CHI] - k[1022]*y[IDX_CDI] - k[1024]*y[IDX_NHI] -
        k[1026]*y[IDX_NDI] - k[1028]*y[IDX_OHI] - k[1030]*y[IDX_ODI] -
        k[2822]*y[IDX_eM] - k[2996]*y[IDX_H2OI] - k[3119]*y[IDX_HDOI] -
        k[3121]*y[IDX_D2OI];
    data[3126] = 0.0 + k[467]*y[IDX_COII] + k[1610]*y[IDX_HOCII] +
        k[1611]*y[IDX_DOCII] + k[1612]*y[IDX_DOCII];
    data[3127] = 0.0 + k[681]*y[IDX_COI] + k[2462]*y[IDX_HCOI];
    data[3128] = 0.0 + k[3287]*y[IDX_CI] + k[3288]*y[IDX_CI];
    data[3129] = 0.0 + k[438]*y[IDX_CII] + k[606]*y[IDX_CNII] +
        k[625]*y[IDX_COII] + k[963]*y[IDX_C2NII] + k[983]*y[IDX_CNCII] +
        k[1784]*y[IDX_CHII] + k[1785]*y[IDX_CDII] - k[3119]*y[IDX_HCOII];
    data[3130] = 0.0 + k[1255]*y[IDX_COI] + k[2505]*y[IDX_HCOI];
    data[3131] = 0.0 + k[1580]*y[IDX_COI];
    data[3132] = 0.0 + k[1590]*y[IDX_COI] + k[1592]*y[IDX_pH2I] +
        k[1593]*y[IDX_oH2I] + k[1594]*y[IDX_oH2I] + k[1600]*y[IDX_pD2I] +
        k[1601]*y[IDX_oD2I] + k[1602]*y[IDX_oD2I] + k[1610]*y[IDX_HDI];
    data[3133] = 0.0 + k[2534]*y[IDX_HCOI];
    data[3134] = 0.0 + k[2574]*y[IDX_HCOI];
    data[3135] = 0.0 + k[1624]*y[IDX_COI];
    data[3136] = 0.0 - k[1026]*y[IDX_HCOII];
    data[3137] = 0.0 + k[2304]*y[IDX_HCOI];
    data[3138] = 0.0 + k[612]*y[IDX_COII] - k[1024]*y[IDX_HCOII];
    data[3139] = 0.0 + k[1836]*y[IDX_COI];
    data[3140] = 0.0 + k[626]*y[IDX_COII];
    data[3141] = 0.0 + k[2303]*y[IDX_HCOI];
    data[3142] = 0.0 + k[629]*y[IDX_COII];
    data[3143] = 0.0 + k[2305]*y[IDX_HCOI];
    data[3144] = 0.0 + k[954]*y[IDX_C2HII] + k[972]*y[IDX_CH2II] +
        k[975]*y[IDX_CHDII] + k[2044]*y[IDX_CHI];
    data[3145] = 0.0 + k[1674]*y[IDX_HCNI] + k[2369]*y[IDX_HCOI];
    data[3146] = 0.0 + k[976]*y[IDX_CH2II] + k[979]*y[IDX_CHDII] +
        k[1755]*y[IDX_CHII];
    data[3147] = 0.0 + k[1913]*y[IDX_CHI] + k[2636]*y[IDX_HCOI];
    data[3148] = 0.0 + k[2016]*y[IDX_COI];
    data[3149] = 0.0 - k[1030]*y[IDX_HCOII];
    data[3150] = 0.0 + k[2599]*y[IDX_HCOI];
    data[3151] = 0.0 + k[614]*y[IDX_COII] - k[1028]*y[IDX_HCOII];
    data[3152] = 0.0 + k[1935]*y[IDX_COI] + k[2598]*y[IDX_HCOI];
    data[3153] = 0.0 - k[2822]*y[IDX_HCOII];
    data[3154] = 0.0 + k[1601]*y[IDX_HOCII] + k[1602]*y[IDX_HOCII];
    data[3155] = 0.0 + k[2461]*y[IDX_HCOI];
    data[3156] = 0.0 + k[1382]*y[IDX_COI];
    data[3157] = 0.0 + k[463]*y[IDX_COII] + k[1593]*y[IDX_HOCII] +
        k[1594]*y[IDX_HOCII] + k[1599]*y[IDX_DOCII];
    data[3158] = 0.0 + k[677]*y[IDX_COI] + k[2459]*y[IDX_HCOI];
    data[3159] = 0.0 + k[1380]*y[IDX_COI];
    data[3160] = 0.0 + k[1372]*y[IDX_COI];
    data[3161] = 0.0 + k[1600]*y[IDX_HOCII];
    data[3162] = 0.0 + k[2460]*y[IDX_HCOI];
    data[3163] = 0.0 + k[1383]*y[IDX_COI] + k[1384]*y[IDX_COI];
    data[3164] = 0.0 + k[462]*y[IDX_COII] + k[1592]*y[IDX_HOCII] +
        k[1598]*y[IDX_DOCII];
    data[3165] = 0.0 + k[676]*y[IDX_COI] + k[2458]*y[IDX_HCOI];
    data[3166] = 0.0 + k[1381]*y[IDX_COI];
    data[3167] = 0.0 + k[1373]*y[IDX_COI] + k[1374]*y[IDX_COI];
    data[3168] = 0.0 - k[1134]*y[IDX_HDI] - k[1135]*y[IDX_HDI] +
        k[1264]*y[IDX_oH2DII] + k[1265]*y[IDX_pH2DII] + k[1269]*y[IDX_oD2HII] +
        k[1270]*y[IDX_pD2HII] - k[2661]*y[IDX_HDI] + k[3285]*y[IDX_H2DOII] +
        k[3286]*y[IDX_HD2OII];
    data[3169] = 0.0 - k[2645]*y[IDX_HDI];
    data[3170] = 0.0 - k[2681]*y[IDX_HDI];
    data[3171] = 0.0 + k[1318]*y[IDX_oH2DII] + k[1319]*y[IDX_pH2DII] +
        k[1323]*y[IDX_oD2HII] + k[1324]*y[IDX_pD2HII] + k[2344]*y[IDX_HDII];
    data[3172] = 0.0 - k[1698]*y[IDX_HDI] - k[1699]*y[IDX_HDI];
    data[3173] = 0.0 + k[1528]*y[IDX_HII] + k[1763]*y[IDX_CHII] +
        k[2427]*y[IDX_HDII];
    data[3174] = 0.0 + k[1527]*y[IDX_DII] + k[1762]*y[IDX_CDII] +
        k[2422]*y[IDX_HDII];
    data[3175] = 0.0 + k[1065]*y[IDX_HI] - k[1150]*y[IDX_HDI] -
        k[1151]*y[IDX_HDI] + k[1343]*y[IDX_oH3II] + k[1344]*y[IDX_pH3II] +
        k[1350]*y[IDX_oH2DII] + k[1351]*y[IDX_pH2DII] + k[1355]*y[IDX_oD2HII] +
        k[1356]*y[IDX_pD2HII] + k[1725]*y[IDX_CHII] + k[2354]*y[IDX_HDII];
    data[3176] = 0.0 + k[1714]*y[IDX_HI] + k[1724]*y[IDX_CHI] -
        k[1745]*y[IDX_HDI] - k[1746]*y[IDX_HDI] + k[1748]*y[IDX_NHI] +
        k[1758]*y[IDX_OHI] + k[1762]*y[IDX_C2HI] + k[1767]*y[IDX_CH2I] +
        k[1774]*y[IDX_CHDI] + k[1779]*y[IDX_H2OI] + k[1786]*y[IDX_HDOI] +
        k[1788]*y[IDX_HCNI] + k[1797]*y[IDX_NH2I] + k[1804]*y[IDX_NHDI];
    data[3177] = 0.0 + k[1070]*y[IDX_HI] + k[1536]*y[IDX_HII] +
        k[1769]*y[IDX_CHII] + k[2437]*y[IDX_HDII];
    data[3178] = 0.0 + k[1066]*y[IDX_DI] - k[1142]*y[IDX_HDI] -
        k[1143]*y[IDX_HDI] + k[1328]*y[IDX_oD3II] + k[1329]*y[IDX_mD3II] +
        k[1333]*y[IDX_oH2DII] + k[1334]*y[IDX_pH2DII] + k[1338]*y[IDX_oD2HII] +
        k[1339]*y[IDX_pD2HII] + k[1724]*y[IDX_CDII] + k[2349]*y[IDX_HDII];
    data[3179] = 0.0 + k[1715]*y[IDX_DI] + k[1725]*y[IDX_CDI] -
        k[1743]*y[IDX_HDI] - k[1744]*y[IDX_HDI] + k[1749]*y[IDX_NDI] +
        k[1759]*y[IDX_ODI] + k[1763]*y[IDX_C2DI] + k[1769]*y[IDX_CD2I] +
        k[1772]*y[IDX_CHDI] + k[1781]*y[IDX_D2OI] + k[1784]*y[IDX_HDOI] +
        k[1789]*y[IDX_DCNI] + k[1799]*y[IDX_ND2I] + k[1802]*y[IDX_NHDI];
    data[3180] = 0.0 + k[1073]*y[IDX_DI] + k[1534]*y[IDX_DII] +
        k[1767]*y[IDX_CDII] + k[2432]*y[IDX_HDII];
    data[3181] = 0.0 + k[514]*y[IDX_OI] + k[894]*y[IDX_HeII] +
        k[1071]*y[IDX_HI] + k[1077]*y[IDX_DI] + k[1539]*y[IDX_HII] +
        k[1541]*y[IDX_DII] + k[1772]*y[IDX_CHII] + k[1774]*y[IDX_CDII] +
        k[2442]*y[IDX_HDII];
    data[3182] = 0.0 + k[2777]*y[IDX_eM];
    data[3183] = 0.0 + k[1365]*y[IDX_oH2DII] + k[1366]*y[IDX_pH2DII] +
        k[1370]*y[IDX_oD2HII] + k[1371]*y[IDX_pD2HII] + k[2359]*y[IDX_HDII];
    data[3184] = 0.0 - k[472]*y[IDX_HDI] - k[473]*y[IDX_HDI] -
        k[478]*y[IDX_HDI] - k[479]*y[IDX_HDI];
    data[3185] = 0.0 + k[1380]*y[IDX_oH2DII] + k[1381]*y[IDX_pH2DII] +
        k[1385]*y[IDX_oD2HII] + k[1386]*y[IDX_pD2HII] + k[1395]*y[IDX_oH2DII] +
        k[1396]*y[IDX_pH2DII] + k[1400]*y[IDX_oD2HII] + k[1401]*y[IDX_pD2HII] +
        k[2364]*y[IDX_HDII];
    data[3186] = 0.0 - k[460]*y[IDX_HDI] - k[461]*y[IDX_HDI] -
        k[466]*y[IDX_HDI] - k[467]*y[IDX_HDI];
    data[3187] = 0.0 + k[2542]*y[IDX_HDII];
    data[3188] = 0.0 + k[166]*y[IDX_HI] + k[1058]*y[IDX_HCOI] +
        k[1066]*y[IDX_CHI] + k[1073]*y[IDX_CH2I] + k[1077]*y[IDX_CHDI] +
        k[1080]*y[IDX_OHI] + k[1088]*y[IDX_H2OI] + k[1091]*y[IDX_HDOI] +
        k[1094]*y[IDX_HCNI] + k[1114]*y[IDX_O2HI] + k[1715]*y[IDX_CHII] +
        k[2102]*y[IDX_HDII] + k[2691]*y[IDX_HM] + k[3385]*y[IDX_H3OII] +
        k[3388]*y[IDX_H2DOII] + k[3391]*y[IDX_HD2OII];
    data[3189] = 0.0 + k[1527]*y[IDX_C2HI] + k[1534]*y[IDX_CH2I] +
        k[1541]*y[IDX_CHDI] + k[1545]*y[IDX_HCOI] + k[1686]*y[IDX_HNOI] -
        k[2853]*y[IDX_HDI] - k[2854]*y[IDX_HDI] + k[2882]*y[IDX_pH2I] +
        k[2883]*y[IDX_oH2I];
    data[3190] = 0.0 + k[485]*y[IDX_H2OI] + k[492]*y[IDX_HDOI] +
        k[494]*y[IDX_HCNI] + k[830]*y[IDX_oH3II] + k[831]*y[IDX_oH3II] +
        k[832]*y[IDX_pH3II] + k[833]*y[IDX_pH3II] + k[845]*y[IDX_oH2DII] +
        k[845]*y[IDX_oH2DII] + k[846]*y[IDX_pH2DII] + k[846]*y[IDX_pH2DII] +
        k[854]*y[IDX_oD2HII] + k[855]*y[IDX_oD2HII] + k[856]*y[IDX_pD2HII] +
        k[857]*y[IDX_pD2HII] + k[859]*y[IDX_HCOII] + k[2159]*y[IDX_HDII] +
        k[2690]*y[IDX_HI] + k[3057]*y[IDX_H3OII] + k[3065]*y[IDX_H2DOII] +
        k[3070]*y[IDX_H2DOII] + k[3079]*y[IDX_HD2OII] + k[3091]*y[IDX_H3OII] +
        k[3097]*y[IDX_H2DOII] + k[3105]*y[IDX_HD2OII];
    data[3191] = 0.0 + k[487]*y[IDX_HM] + k[1084]*y[IDX_HI] +
        k[1781]*y[IDX_CHII] + k[1894]*y[IDX_NHII] + k[2452]*y[IDX_HDII] +
        k[3242]*y[IDX_pH3II] + k[3243]*y[IDX_oH3II] + k[3250]*y[IDX_pH2DII] +
        k[3251]*y[IDX_oH2DII] + k[3256]*y[IDX_pD2HII] + k[3257]*y[IDX_oD2HII];
    data[3192] = 0.0 - k[3135]*y[IDX_HDI] - k[3136]*y[IDX_HDI];
    data[3193] = 0.0 + k[3084]*y[IDX_HM] + k[3108]*y[IDX_HM] +
        k[3380]*y[IDX_HI] - k[3420]*y[IDX_HDI] - k[3421]*y[IDX_HDI];
    data[3194] = 0.0 + k[495]*y[IDX_HM] + k[1093]*y[IDX_HI] +
        k[1789]*y[IDX_CHII] + k[2112]*y[IDX_HDII];
    data[3195] = 0.0 + k[1057]*y[IDX_HI] + k[1546]*y[IDX_HII] +
        k[2467]*y[IDX_HDII];
    data[3196] = 0.0 + k[860]*y[IDX_HM];
    data[3197] = 0.0 + k[1687]*y[IDX_HII];
    data[3198] = 0.0 + k[1598]*y[IDX_pH2I] + k[1599]*y[IDX_oH2I] -
        k[1611]*y[IDX_HDI] - k[1612]*y[IDX_HDI] - k[1613]*y[IDX_HDI] +
        k[1613]*y[IDX_HDI];
    data[3199] = 0.0 + k[172]*y[IDX_HDII] + k[184]*y[IDX_pH2DII] +
        k[188]*y[IDX_oH2DII] + k[191]*y[IDX_oD2HII] + k[193]*y[IDX_pD2HII];
    data[3200] = 0.0 + k[166]*y[IDX_DI] + k[1057]*y[IDX_DCOI] +
        k[1065]*y[IDX_CDI] + k[1070]*y[IDX_CD2I] + k[1071]*y[IDX_CHDI] +
        k[1079]*y[IDX_ODI] + k[1084]*y[IDX_D2OI] + k[1086]*y[IDX_HDOI] +
        k[1093]*y[IDX_DCNI] + k[1113]*y[IDX_O2DI] + k[1714]*y[IDX_CDII] +
        k[2097]*y[IDX_HDII] + k[2690]*y[IDX_DM] + k[3374]*y[IDX_H2DOII] +
        k[3377]*y[IDX_HD2OII] + k[3380]*y[IDX_D3OII];
    data[3201] = 0.0 + k[1528]*y[IDX_C2DI] + k[1536]*y[IDX_CD2I] +
        k[1539]*y[IDX_CHDI] + k[1546]*y[IDX_DCOI] + k[1687]*y[IDX_DNOI] +
        k[2851]*y[IDX_pD2I] + k[2852]*y[IDX_oD2I] - k[2884]*y[IDX_HDI] -
        k[2885]*y[IDX_HDI];
    data[3202] = 0.0 + k[487]*y[IDX_D2OI] + k[490]*y[IDX_HDOI] +
        k[495]*y[IDX_DCNI] + k[836]*y[IDX_oH2DII] + k[837]*y[IDX_oH2DII] +
        k[838]*y[IDX_pH2DII] + k[839]*y[IDX_pH2DII] + k[852]*y[IDX_oD2HII] +
        k[852]*y[IDX_oD2HII] + k[853]*y[IDX_pD2HII] + k[853]*y[IDX_pD2HII] +
        k[860]*y[IDX_DCOII] + k[2158]*y[IDX_HDII] + k[2691]*y[IDX_DI] +
        k[3062]*y[IDX_H2DOII] + k[3071]*y[IDX_HD2OII] + k[3076]*y[IDX_HD2OII] +
        k[3084]*y[IDX_D3OII] + k[3094]*y[IDX_H2DOII] + k[3102]*y[IDX_HD2OII] +
        k[3108]*y[IDX_D3OII];
    data[3203] = 0.0 + k[3062]*y[IDX_HM] + k[3065]*y[IDX_DM] +
        k[3070]*y[IDX_DM] + k[3094]*y[IDX_HM] + k[3097]*y[IDX_DM] +
        k[3285]*y[IDX_CI] + k[3374]*y[IDX_HI] + k[3388]*y[IDX_DI] -
        k[3414]*y[IDX_HDI] - k[3415]*y[IDX_HDI] - k[3416]*y[IDX_HDI] +
        k[3452]*y[IDX_eM] + k[3458]*y[IDX_eM];
    data[3204] = 0.0 + k[485]*y[IDX_DM] + k[1088]*y[IDX_DI] +
        k[1779]*y[IDX_CDII] + k[1892]*y[IDX_NDII] + k[2447]*y[IDX_HDII] +
        k[3181]*y[IDX_oH2DII] + k[3182]*y[IDX_pH2DII] + k[3187]*y[IDX_oD2HII] +
        k[3188]*y[IDX_pD2HII] + k[3197]*y[IDX_pD3II] + k[3198]*y[IDX_mD3II] +
        k[3199]*y[IDX_oD3II];
    data[3205] = 0.0 - k[3131]*y[IDX_HDI] - k[3132]*y[IDX_HDI];
    data[3206] = 0.0 + k[3057]*y[IDX_DM] + k[3091]*y[IDX_DM] +
        k[3385]*y[IDX_DI] - k[3412]*y[IDX_HDI] - k[3413]*y[IDX_HDI];
    data[3207] = 0.0 + k[494]*y[IDX_DM] + k[1094]*y[IDX_DI] +
        k[1788]*y[IDX_CDII] + k[2107]*y[IDX_HDII];
    data[3208] = 0.0 + k[1058]*y[IDX_DI] + k[1545]*y[IDX_DII] +
        k[2462]*y[IDX_HDII];
    data[3209] = 0.0 + k[859]*y[IDX_DM];
    data[3210] = 0.0 - k[10]*y[IDX_pH3II] + k[10]*y[IDX_pH3II] -
        k[11]*y[IDX_pH3II] - k[12]*y[IDX_pH3II] - k[13]*y[IDX_pH3II] -
        k[14]*y[IDX_pH3II] - k[15]*y[IDX_oH3II] + k[15]*y[IDX_oH3II] -
        k[16]*y[IDX_oH3II] - k[17]*y[IDX_oH3II] - k[18]*y[IDX_oH3II] -
        k[45]*y[IDX_pH2DII] - k[46]*y[IDX_pH2DII] - k[47]*y[IDX_pH2DII] +
        k[47]*y[IDX_pH2DII] - k[48]*y[IDX_pH2DII] - k[49]*y[IDX_pH2DII] -
        k[50]*y[IDX_pH2DII] - k[51]*y[IDX_pH2DII] - k[52]*y[IDX_oH2DII] -
        k[53]*y[IDX_oH2DII] - k[54]*y[IDX_oH2DII] - k[55]*y[IDX_oH2DII] -
        k[56]*y[IDX_oH2DII] + k[56]*y[IDX_oH2DII] - k[57]*y[IDX_oH2DII] -
        k[58]*y[IDX_oH2DII] - k[59]*y[IDX_oH2DII] - k[60]*y[IDX_oH2DII] -
        k[96]*y[IDX_pD2HII] - k[97]*y[IDX_pD2HII] - k[98]*y[IDX_pD2HII] -
        k[99]*y[IDX_pD2HII] - k[100]*y[IDX_pD2HII] + k[100]*y[IDX_pD2HII] -
        k[101]*y[IDX_pD2HII] - k[102]*y[IDX_pD2HII] - k[103]*y[IDX_oD2HII] -
        k[104]*y[IDX_oD2HII] - k[105]*y[IDX_oD2HII] - k[106]*y[IDX_oD2HII] -
        k[107]*y[IDX_oD2HII] + k[107]*y[IDX_oD2HII] - k[108]*y[IDX_oD2HII] -
        k[109]*y[IDX_oD2HII] - k[110]*y[IDX_oD2HII] - k[111]*y[IDX_oD2HII] -
        k[143]*y[IDX_mD3II] - k[144]*y[IDX_mD3II] - k[145]*y[IDX_mD3II] -
        k[146]*y[IDX_mD3II] + k[146]*y[IDX_mD3II] - k[147]*y[IDX_oD3II] -
        k[148]*y[IDX_oD3II] - k[149]*y[IDX_oD3II] - k[150]*y[IDX_oD3II] -
        k[151]*y[IDX_oD3II] + k[151]*y[IDX_oD3II] - k[212] - k[217] - k[218] -
        k[223] - k[224] - k[229] - k[364] - k[460]*y[IDX_COII] -
        k[461]*y[IDX_COII] - k[466]*y[IDX_COII] - k[467]*y[IDX_COII] -
        k[472]*y[IDX_CNII] - k[473]*y[IDX_CNII] - k[478]*y[IDX_CNII] -
        k[479]*y[IDX_CNII] - k[501]*y[IDX_OM] - k[502]*y[IDX_OM] -
        k[726]*y[IDX_pH2II] - k[727]*y[IDX_pH2II] - k[728]*y[IDX_oH2II] -
        k[729]*y[IDX_oH2II] - k[730]*y[IDX_oH2II] - k[731]*y[IDX_pD2II] -
        k[732]*y[IDX_oD2II] - k[733]*y[IDX_oD2II] - k[734]*y[IDX_pD2II] -
        k[735]*y[IDX_oD2II] - k[736]*y[IDX_HDII] - k[737]*y[IDX_HDII] -
        k[738]*y[IDX_HDII] - k[739]*y[IDX_HDII] - k[823]*y[IDX_HeHII] -
        k[824]*y[IDX_HeHII] - k[825]*y[IDX_HeDII] - k[826]*y[IDX_HeDII] -
        k[872]*y[IDX_HeII] - k[873]*y[IDX_HeII] - k[1134]*y[IDX_CI] -
        k[1135]*y[IDX_CI] - k[1142]*y[IDX_CHI] - k[1143]*y[IDX_CHI] -
        k[1150]*y[IDX_CDI] - k[1151]*y[IDX_CDI] - k[1156]*y[IDX_OI] -
        k[1157]*y[IDX_OI] - k[1170]*y[IDX_OHI] - k[1171]*y[IDX_OHI] -
        k[1172]*y[IDX_ODI] - k[1173]*y[IDX_ODI] - k[1178]*y[IDX_NI] -
        k[1179]*y[IDX_NI] - k[1192]*y[IDX_NHI] - k[1193]*y[IDX_NHI] -
        k[1194]*y[IDX_NDI] - k[1195]*y[IDX_NDI] - k[1608]*y[IDX_HOCII] -
        k[1609]*y[IDX_HOCII] - k[1610]*y[IDX_HOCII] + k[1610]*y[IDX_HOCII] -
        k[1611]*y[IDX_DOCII] - k[1612]*y[IDX_DOCII] - k[1613]*y[IDX_DOCII] +
        k[1613]*y[IDX_DOCII] - k[1640]*y[IDX_NII] - k[1641]*y[IDX_NII] -
        k[1660]*y[IDX_OII] - k[1661]*y[IDX_OII] - k[1698]*y[IDX_C2II] -
        k[1699]*y[IDX_C2II] - k[1743]*y[IDX_CHII] - k[1744]*y[IDX_CHII] -
        k[1745]*y[IDX_CDII] - k[1746]*y[IDX_CDII] - k[1810]*y[IDX_N2II] -
        k[1811]*y[IDX_N2II] - k[1850]*y[IDX_NHII] - k[1851]*y[IDX_NHII] -
        k[1852]*y[IDX_NDII] - k[1853]*y[IDX_NDII] - k[1866]*y[IDX_NHII] -
        k[1867]*y[IDX_NHII] - k[1868]*y[IDX_NDII] - k[1869]*y[IDX_NDII] -
        k[1949]*y[IDX_OHII] - k[1950]*y[IDX_OHII] - k[1951]*y[IDX_ODII] -
        k[1952]*y[IDX_ODII] - k[2001]*y[IDX_NO2II] - k[2028]*y[IDX_O2HII] -
        k[2029]*y[IDX_O2HII] - k[2030]*y[IDX_O2DII] - k[2031]*y[IDX_O2DII] -
        k[2475]*y[IDX_HeII] - k[2645]*y[IDX_CII] - k[2661]*y[IDX_CI] -
        k[2681]*y[IDX_CM] - k[2727]*y[IDX_OM] - k[2853]*y[IDX_DII] -
        k[2854]*y[IDX_DII] - k[2884]*y[IDX_HII] - k[2885]*y[IDX_HII] -
        k[2888]*y[IDX_pH2II] - k[2889]*y[IDX_oH2II] - k[2900]*y[IDX_pD2HII] -
        k[2901]*y[IDX_pD2HII] - k[2939]*y[IDX_pD3II] - k[2940]*y[IDX_pD3II] -
        k[2941]*y[IDX_pD3II] - k[2942]*y[IDX_pD3II] + k[2942]*y[IDX_pD3II] -
        k[2943]*y[IDX_oD3II] + k[2943]*y[IDX_oD3II] - k[3131]*y[IDX_H2OII] -
        k[3132]*y[IDX_H2OII] - k[3133]*y[IDX_HDOII] - k[3134]*y[IDX_HDOII] -
        k[3135]*y[IDX_D2OII] - k[3136]*y[IDX_D2OII] - k[3412]*y[IDX_H3OII] -
        k[3413]*y[IDX_H3OII] - k[3414]*y[IDX_H2DOII] - k[3415]*y[IDX_H2DOII] -
        k[3416]*y[IDX_H2DOII] - k[3417]*y[IDX_HD2OII] - k[3418]*y[IDX_HD2OII] -
        k[3419]*y[IDX_HD2OII] - k[3420]*y[IDX_D3OII] - k[3421]*y[IDX_D3OII];
    data[3211] = 0.0 + k[172]*y[IDX_GRAINM] - k[736]*y[IDX_HDI] -
        k[737]*y[IDX_HDI] - k[738]*y[IDX_HDI] - k[739]*y[IDX_HDI] +
        k[2097]*y[IDX_HI] + k[2102]*y[IDX_DI] + k[2107]*y[IDX_HCNI] +
        k[2112]*y[IDX_DCNI] + k[2117]*y[IDX_NH2I] + k[2122]*y[IDX_ND2I] +
        k[2127]*y[IDX_NHDI] + k[2158]*y[IDX_HM] + k[2159]*y[IDX_DM] +
        k[2344]*y[IDX_C2I] + k[2349]*y[IDX_CHI] + k[2354]*y[IDX_CDI] +
        k[2359]*y[IDX_CNI] + k[2364]*y[IDX_COI] + k[2392]*y[IDX_NHI] +
        k[2397]*y[IDX_NDI] + k[2402]*y[IDX_NOI] + k[2407]*y[IDX_O2I] +
        k[2412]*y[IDX_OHI] + k[2417]*y[IDX_ODI] + k[2422]*y[IDX_C2HI] +
        k[2427]*y[IDX_C2DI] + k[2432]*y[IDX_CH2I] + k[2437]*y[IDX_CD2I] +
        k[2442]*y[IDX_CHDI] + k[2447]*y[IDX_H2OI] + k[2452]*y[IDX_D2OI] +
        k[2457]*y[IDX_HDOI] + k[2462]*y[IDX_HCOI] + k[2467]*y[IDX_DCOI] +
        k[2542]*y[IDX_CO2I] + k[2755]*y[IDX_eM];
    data[3212] = 0.0 + k[3071]*y[IDX_HM] + k[3076]*y[IDX_HM] +
        k[3079]*y[IDX_DM] + k[3102]*y[IDX_HM] + k[3105]*y[IDX_DM] +
        k[3286]*y[IDX_CI] + k[3377]*y[IDX_HI] + k[3391]*y[IDX_DI] -
        k[3417]*y[IDX_HDI] - k[3418]*y[IDX_HDI] - k[3419]*y[IDX_HDI] +
        k[3453]*y[IDX_eM] + k[3463]*y[IDX_eM];
    data[3213] = 0.0 + k[490]*y[IDX_HM] + k[492]*y[IDX_DM] +
        k[1086]*y[IDX_HI] + k[1091]*y[IDX_DI] + k[1784]*y[IDX_CHII] +
        k[1786]*y[IDX_CDII] + k[1897]*y[IDX_NHII] + k[1899]*y[IDX_NDII] +
        k[2457]*y[IDX_HDII] + k[3208]*y[IDX_oH3II] + k[3209]*y[IDX_pH3II] +
        k[3214]*y[IDX_pH2DII] + k[3215]*y[IDX_oH2DII] + k[3224]*y[IDX_pD2HII] +
        k[3225]*y[IDX_oD2HII] + k[3230]*y[IDX_mD3II] + k[3231]*y[IDX_pD3II] +
        k[3232]*y[IDX_oD3II];
    data[3214] = 0.0 + k[997]*y[IDX_OI] + k[2790]*y[IDX_eM] -
        k[3133]*y[IDX_HDI] - k[3134]*y[IDX_HDI];
    data[3215] = 0.0 + k[1686]*y[IDX_DII];
    data[3216] = 0.0 + k[1603]*y[IDX_pD2I] + k[1604]*y[IDX_oD2I] -
        k[1608]*y[IDX_HDI] - k[1609]*y[IDX_HDI] - k[1610]*y[IDX_HDI] +
        k[1610]*y[IDX_HDI];
    data[3217] = 0.0 - k[872]*y[IDX_HDI] - k[873]*y[IDX_HDI] +
        k[894]*y[IDX_CHDI] + k[941]*y[IDX_NHDI] - k[2475]*y[IDX_HDI];
    data[3218] = 0.0 - k[825]*y[IDX_HDI] - k[826]*y[IDX_HDI];
    data[3219] = 0.0 - k[823]*y[IDX_HDI] - k[824]*y[IDX_HDI];
    data[3220] = 0.0 - k[1178]*y[IDX_HDI] - k[1179]*y[IDX_HDI];
    data[3221] = 0.0 - k[1640]*y[IDX_HDI] - k[1641]*y[IDX_HDI];
    data[3222] = 0.0 + k[1410]*y[IDX_oH2DII] + k[1411]*y[IDX_pH2DII] +
        k[1415]*y[IDX_oD2HII] + k[1416]*y[IDX_pD2HII];
    data[3223] = 0.0 - k[1810]*y[IDX_HDI] - k[1811]*y[IDX_HDI];
    data[3224] = 0.0 - k[1194]*y[IDX_HDI] - k[1195]*y[IDX_HDI] +
        k[1435]*y[IDX_oH3II] + k[1436]*y[IDX_pH3II] + k[1442]*y[IDX_oH2DII] +
        k[1443]*y[IDX_pH2DII] + k[1447]*y[IDX_oD2HII] + k[1448]*y[IDX_pD2HII] +
        k[1749]*y[IDX_CHII] + k[2397]*y[IDX_HDII];
    data[3225] = 0.0 - k[1852]*y[IDX_HDI] - k[1853]*y[IDX_HDI] -
        k[1868]*y[IDX_HDI] - k[1869]*y[IDX_HDI] + k[1892]*y[IDX_H2OI] +
        k[1899]*y[IDX_HDOI];
    data[3226] = 0.0 + k[1799]*y[IDX_CHII] + k[2122]*y[IDX_HDII];
    data[3227] = 0.0 - k[1192]*y[IDX_HDI] - k[1193]*y[IDX_HDI] +
        k[1420]*y[IDX_oD3II] + k[1421]*y[IDX_mD3II] + k[1425]*y[IDX_oH2DII] +
        k[1426]*y[IDX_pH2DII] + k[1430]*y[IDX_oD2HII] + k[1431]*y[IDX_pD2HII] +
        k[1748]*y[IDX_CDII] + k[2392]*y[IDX_HDII];
    data[3228] = 0.0 - k[1850]*y[IDX_HDI] - k[1851]*y[IDX_HDI] -
        k[1866]*y[IDX_HDI] - k[1867]*y[IDX_HDI] + k[1894]*y[IDX_D2OI] +
        k[1897]*y[IDX_HDOI];
    data[3229] = 0.0 + k[1797]*y[IDX_CDII] + k[2117]*y[IDX_HDII];
    data[3230] = 0.0 + k[941]*y[IDX_HeII] + k[1802]*y[IDX_CHII] +
        k[1804]*y[IDX_CDII] + k[2127]*y[IDX_HDII];
    data[3231] = 0.0 + k[1457]*y[IDX_oH2DII] + k[1458]*y[IDX_pH2DII] +
        k[1462]*y[IDX_oD2HII] + k[1463]*y[IDX_pD2HII] + k[2402]*y[IDX_HDII];
    data[3232] = 0.0 + k[1519]*y[IDX_oH2DII] + k[1520]*y[IDX_pH2DII] +
        k[1524]*y[IDX_oD2HII] + k[1525]*y[IDX_pD2HII];
    data[3233] = 0.0 - k[2001]*y[IDX_HDI];
    data[3234] = 0.0 + k[514]*y[IDX_CHDI] + k[997]*y[IDX_HDOII] -
        k[1156]*y[IDX_HDI] - k[1157]*y[IDX_HDI] + k[1291]*y[IDX_oH2DII] +
        k[1292]*y[IDX_pH2DII] + k[1296]*y[IDX_oD2HII] + k[1297]*y[IDX_pD2HII];
    data[3235] = 0.0 - k[1660]*y[IDX_HDI] - k[1661]*y[IDX_HDI];
    data[3236] = 0.0 - k[501]*y[IDX_HDI] - k[502]*y[IDX_HDI] -
        k[2727]*y[IDX_HDI];
    data[3237] = 0.0 + k[1472]*y[IDX_oH2DII] + k[1473]*y[IDX_pH2DII] +
        k[1477]*y[IDX_oD2HII] + k[1478]*y[IDX_pD2HII] + k[2407]*y[IDX_HDII];
    data[3238] = 0.0 + k[1113]*y[IDX_HI];
    data[3239] = 0.0 - k[2030]*y[IDX_HDI] - k[2031]*y[IDX_HDI];
    data[3240] = 0.0 + k[1114]*y[IDX_DI];
    data[3241] = 0.0 - k[2028]*y[IDX_HDI] - k[2029]*y[IDX_HDI];
    data[3242] = 0.0 + k[1079]*y[IDX_HI] - k[1172]*y[IDX_HDI] -
        k[1173]*y[IDX_HDI] + k[1497]*y[IDX_oH3II] + k[1498]*y[IDX_pH3II] +
        k[1504]*y[IDX_oH2DII] + k[1505]*y[IDX_pH2DII] + k[1509]*y[IDX_oD2HII] +
        k[1510]*y[IDX_pD2HII] + k[1759]*y[IDX_CHII] + k[2417]*y[IDX_HDII];
    data[3243] = 0.0 - k[1951]*y[IDX_HDI] - k[1952]*y[IDX_HDI];
    data[3244] = 0.0 + k[1080]*y[IDX_DI] - k[1170]*y[IDX_HDI] -
        k[1171]*y[IDX_HDI] + k[1482]*y[IDX_oD3II] + k[1483]*y[IDX_mD3II] +
        k[1487]*y[IDX_oH2DII] + k[1488]*y[IDX_pH2DII] + k[1492]*y[IDX_oD2HII] +
        k[1493]*y[IDX_pD2HII] + k[1758]*y[IDX_CDII] + k[2412]*y[IDX_HDII];
    data[3245] = 0.0 - k[1949]*y[IDX_HDI] - k[1950]*y[IDX_HDI];
    data[3246] = 0.0 + k[2755]*y[IDX_HDII] + k[2777]*y[IDX_CHDII] +
        k[2790]*y[IDX_HDOII] + k[2812]*y[IDX_oH2DII] + k[2813]*y[IDX_pH2DII] +
        k[2818]*y[IDX_oD2HII] + k[2819]*y[IDX_pD2HII] + k[3452]*y[IDX_H2DOII] +
        k[3453]*y[IDX_HD2OII] + k[3458]*y[IDX_H2DOII] + k[3463]*y[IDX_HD2OII];
    data[3247] = 0.0 + k[113]*y[IDX_pH2I] + k[115]*y[IDX_oH2I] -
        k[143]*y[IDX_HDI] - k[144]*y[IDX_HDI] - k[145]*y[IDX_HDI] -
        k[146]*y[IDX_HDI] + k[146]*y[IDX_HDI] + k[1329]*y[IDX_CHI] +
        k[1421]*y[IDX_NHI] + k[1483]*y[IDX_OHI] + k[3198]*y[IDX_H2OI] +
        k[3230]*y[IDX_HDOI];
    data[3248] = 0.0 + k[37]*y[IDX_pH3II] + k[38]*y[IDX_pH3II] +
        k[43]*y[IDX_oH3II] + k[83]*y[IDX_pH2DII] + k[84]*y[IDX_pH2DII] +
        k[92]*y[IDX_oH2DII] + k[93]*y[IDX_oH2DII] + k[131]*y[IDX_pD2HII] +
        k[132]*y[IDX_pD2HII] + k[141]*y[IDX_oD2HII] + k[142]*y[IDX_oD2HII] +
        k[1604]*y[IDX_HOCII] + k[2852]*y[IDX_HII] + k[2906]*y[IDX_pD2HII];
    data[3249] = 0.0 - k[732]*y[IDX_HDI] - k[733]*y[IDX_HDI] -
        k[735]*y[IDX_HDI];
    data[3250] = 0.0 + k[70]*y[IDX_pH2I] + k[71]*y[IDX_pH2I] +
        k[75]*y[IDX_oH2I] + k[76]*y[IDX_oH2I] - k[103]*y[IDX_HDI] -
        k[104]*y[IDX_HDI] - k[105]*y[IDX_HDI] - k[106]*y[IDX_HDI] -
        k[107]*y[IDX_HDI] + k[107]*y[IDX_HDI] - k[108]*y[IDX_HDI] -
        k[109]*y[IDX_HDI] - k[110]*y[IDX_HDI] - k[111]*y[IDX_HDI] +
        k[136]*y[IDX_pD2I] + k[137]*y[IDX_pD2I] + k[141]*y[IDX_oD2I] +
        k[142]*y[IDX_oD2I] + k[191]*y[IDX_GRAINM] + k[322] + k[852]*y[IDX_HM] +
        k[852]*y[IDX_HM] + k[854]*y[IDX_DM] + k[855]*y[IDX_DM] +
        k[1269]*y[IDX_CI] + k[1296]*y[IDX_OI] + k[1323]*y[IDX_C2I] +
        k[1338]*y[IDX_CHI] + k[1355]*y[IDX_CDI] + k[1370]*y[IDX_CNI] +
        k[1385]*y[IDX_COI] + k[1400]*y[IDX_COI] + k[1415]*y[IDX_N2I] +
        k[1430]*y[IDX_NHI] + k[1447]*y[IDX_NDI] + k[1462]*y[IDX_NOI] +
        k[1477]*y[IDX_O2I] + k[1492]*y[IDX_OHI] + k[1509]*y[IDX_ODI] +
        k[1524]*y[IDX_NO2I] + k[2818]*y[IDX_eM] + k[2905]*y[IDX_pD2I] +
        k[3187]*y[IDX_H2OI] + k[3225]*y[IDX_HDOI] + k[3257]*y[IDX_D2OI];
    data[3251] = 0.0 + k[118]*y[IDX_pH2I] + k[119]*y[IDX_pH2I] +
        k[122]*y[IDX_oH2I] + k[123]*y[IDX_oH2I] - k[147]*y[IDX_HDI] -
        k[148]*y[IDX_HDI] - k[149]*y[IDX_HDI] - k[150]*y[IDX_HDI] -
        k[151]*y[IDX_HDI] + k[151]*y[IDX_HDI] + k[1328]*y[IDX_CHI] +
        k[1420]*y[IDX_NHI] + k[1482]*y[IDX_OHI] - k[2943]*y[IDX_HDI] +
        k[2943]*y[IDX_HDI] + k[3199]*y[IDX_H2OI] + k[3232]*y[IDX_HDOI];
    data[3252] = 0.0 + k[21]*y[IDX_pH2DII] + k[22]*y[IDX_pH2DII] +
        k[28]*y[IDX_oH2DII] + k[29]*y[IDX_oH2DII] + k[66]*y[IDX_pD2HII] +
        k[67]*y[IDX_pD2HII] + k[75]*y[IDX_oD2HII] + k[76]*y[IDX_oD2HII] +
        k[115]*y[IDX_mD3II] + k[122]*y[IDX_oD3II] + k[123]*y[IDX_oD3II] +
        k[1599]*y[IDX_DOCII] + k[2883]*y[IDX_DII] + k[2902]*y[IDX_pD3II];
    data[3253] = 0.0 - k[728]*y[IDX_HDI] - k[729]*y[IDX_HDI] -
        k[730]*y[IDX_HDI] - k[2889]*y[IDX_HDI];
    data[3254] = 0.0 + k[25]*y[IDX_pH2I] + k[28]*y[IDX_oH2I] +
        k[29]*y[IDX_oH2I] - k[52]*y[IDX_HDI] - k[53]*y[IDX_HDI] -
        k[54]*y[IDX_HDI] - k[55]*y[IDX_HDI] - k[56]*y[IDX_HDI] +
        k[56]*y[IDX_HDI] - k[57]*y[IDX_HDI] - k[58]*y[IDX_HDI] -
        k[59]*y[IDX_HDI] - k[60]*y[IDX_HDI] + k[88]*y[IDX_pD2I] +
        k[89]*y[IDX_pD2I] + k[92]*y[IDX_oD2I] + k[93]*y[IDX_oD2I] +
        k[188]*y[IDX_GRAINM] + k[316] + k[836]*y[IDX_HM] + k[837]*y[IDX_HM] +
        k[845]*y[IDX_DM] + k[845]*y[IDX_DM] + k[1264]*y[IDX_CI] +
        k[1291]*y[IDX_OI] + k[1318]*y[IDX_C2I] + k[1333]*y[IDX_CHI] +
        k[1350]*y[IDX_CDI] + k[1365]*y[IDX_CNI] + k[1380]*y[IDX_COI] +
        k[1395]*y[IDX_COI] + k[1410]*y[IDX_N2I] + k[1425]*y[IDX_NHI] +
        k[1442]*y[IDX_NDI] + k[1457]*y[IDX_NOI] + k[1472]*y[IDX_O2I] +
        k[1487]*y[IDX_OHI] + k[1504]*y[IDX_ODI] + k[1519]*y[IDX_NO2I] +
        k[2812]*y[IDX_eM] + k[3181]*y[IDX_H2OI] + k[3215]*y[IDX_HDOI] +
        k[3251]*y[IDX_D2OI];
    data[3255] = 0.0 - k[15]*y[IDX_HDI] + k[15]*y[IDX_HDI] -
        k[16]*y[IDX_HDI] - k[17]*y[IDX_HDI] - k[18]*y[IDX_HDI] +
        k[41]*y[IDX_pD2I] + k[43]*y[IDX_oD2I] + k[830]*y[IDX_DM] +
        k[831]*y[IDX_DM] + k[1343]*y[IDX_CDI] + k[1435]*y[IDX_NDI] +
        k[1497]*y[IDX_ODI] + k[3208]*y[IDX_HDOI] + k[3243]*y[IDX_D2OI];
    data[3256] = 0.0 + k[33]*y[IDX_pH3II] + k[34]*y[IDX_pH3II] +
        k[41]*y[IDX_oH3II] + k[79]*y[IDX_pH2DII] + k[80]*y[IDX_pH2DII] +
        k[88]*y[IDX_oH2DII] + k[89]*y[IDX_oH2DII] + k[127]*y[IDX_pD2HII] +
        k[136]*y[IDX_oD2HII] + k[137]*y[IDX_oD2HII] + k[1603]*y[IDX_HOCII] +
        k[2851]*y[IDX_HII] + k[2905]*y[IDX_oD2HII] + k[2907]*y[IDX_pD2HII];
    data[3257] = 0.0 - k[731]*y[IDX_HDI] - k[734]*y[IDX_HDI];
    data[3258] = 0.0 + k[62]*y[IDX_pH2I] + k[63]*y[IDX_pH2I] +
        k[66]*y[IDX_oH2I] + k[67]*y[IDX_oH2I] - k[96]*y[IDX_HDI] -
        k[97]*y[IDX_HDI] - k[98]*y[IDX_HDI] - k[99]*y[IDX_HDI] -
        k[100]*y[IDX_HDI] + k[100]*y[IDX_HDI] - k[101]*y[IDX_HDI] -
        k[102]*y[IDX_HDI] + k[127]*y[IDX_pD2I] + k[131]*y[IDX_oD2I] +
        k[132]*y[IDX_oD2I] + k[193]*y[IDX_GRAINM] + k[323] + k[853]*y[IDX_HM] +
        k[853]*y[IDX_HM] + k[856]*y[IDX_DM] + k[857]*y[IDX_DM] +
        k[1270]*y[IDX_CI] + k[1297]*y[IDX_OI] + k[1324]*y[IDX_C2I] +
        k[1339]*y[IDX_CHI] + k[1356]*y[IDX_CDI] + k[1371]*y[IDX_CNI] +
        k[1386]*y[IDX_COI] + k[1401]*y[IDX_COI] + k[1416]*y[IDX_N2I] +
        k[1431]*y[IDX_NHI] + k[1448]*y[IDX_NDI] + k[1463]*y[IDX_NOI] +
        k[1478]*y[IDX_O2I] + k[1493]*y[IDX_OHI] + k[1510]*y[IDX_ODI] +
        k[1525]*y[IDX_NO2I] + k[2819]*y[IDX_eM] - k[2900]*y[IDX_HDI] -
        k[2901]*y[IDX_HDI] + k[2906]*y[IDX_oD2I] + k[2907]*y[IDX_pD2I] +
        k[3188]*y[IDX_H2OI] + k[3224]*y[IDX_HDOI] + k[3256]*y[IDX_D2OI];
    data[3259] = 0.0 + k[2902]*y[IDX_oH2I] + k[2937]*y[IDX_pH2I] -
        k[2939]*y[IDX_HDI] - k[2940]*y[IDX_HDI] - k[2941]*y[IDX_HDI] -
        k[2942]*y[IDX_HDI] + k[2942]*y[IDX_HDI] + k[3197]*y[IDX_H2OI] +
        k[3231]*y[IDX_HDOI];
    data[3260] = 0.0 + k[19]*y[IDX_pH2DII] + k[25]*y[IDX_oH2DII] +
        k[62]*y[IDX_pD2HII] + k[63]*y[IDX_pD2HII] + k[70]*y[IDX_oD2HII] +
        k[71]*y[IDX_oD2HII] + k[113]*y[IDX_mD3II] + k[118]*y[IDX_oD3II] +
        k[119]*y[IDX_oD3II] + k[1598]*y[IDX_DOCII] + k[2882]*y[IDX_DII] +
        k[2937]*y[IDX_pD3II];
    data[3261] = 0.0 - k[726]*y[IDX_HDI] - k[727]*y[IDX_HDI] -
        k[2888]*y[IDX_HDI];
    data[3262] = 0.0 + k[19]*y[IDX_pH2I] + k[21]*y[IDX_oH2I] +
        k[22]*y[IDX_oH2I] - k[45]*y[IDX_HDI] - k[46]*y[IDX_HDI] -
        k[47]*y[IDX_HDI] + k[47]*y[IDX_HDI] - k[48]*y[IDX_HDI] -
        k[49]*y[IDX_HDI] - k[50]*y[IDX_HDI] - k[51]*y[IDX_HDI] +
        k[79]*y[IDX_pD2I] + k[80]*y[IDX_pD2I] + k[83]*y[IDX_oD2I] +
        k[84]*y[IDX_oD2I] + k[184]*y[IDX_GRAINM] + k[317] + k[838]*y[IDX_HM] +
        k[839]*y[IDX_HM] + k[846]*y[IDX_DM] + k[846]*y[IDX_DM] +
        k[1265]*y[IDX_CI] + k[1292]*y[IDX_OI] + k[1319]*y[IDX_C2I] +
        k[1334]*y[IDX_CHI] + k[1351]*y[IDX_CDI] + k[1366]*y[IDX_CNI] +
        k[1381]*y[IDX_COI] + k[1396]*y[IDX_COI] + k[1411]*y[IDX_N2I] +
        k[1426]*y[IDX_NHI] + k[1443]*y[IDX_NDI] + k[1458]*y[IDX_NOI] +
        k[1473]*y[IDX_O2I] + k[1488]*y[IDX_OHI] + k[1505]*y[IDX_ODI] +
        k[1520]*y[IDX_NO2I] + k[2813]*y[IDX_eM] + k[3182]*y[IDX_H2OI] +
        k[3214]*y[IDX_HDOI] + k[3250]*y[IDX_D2OI];
    data[3263] = 0.0 - k[10]*y[IDX_HDI] + k[10]*y[IDX_HDI] -
        k[11]*y[IDX_HDI] - k[12]*y[IDX_HDI] - k[13]*y[IDX_HDI] -
        k[14]*y[IDX_HDI] + k[33]*y[IDX_pD2I] + k[34]*y[IDX_pD2I] +
        k[37]*y[IDX_oD2I] + k[38]*y[IDX_oD2I] + k[832]*y[IDX_DM] +
        k[833]*y[IDX_DM] + k[1344]*y[IDX_CDI] + k[1436]*y[IDX_NDI] +
        k[1498]*y[IDX_ODI] + k[3209]*y[IDX_HDOI] + k[3242]*y[IDX_D2OI];
    data[3264] = 0.0 - k[634]*y[IDX_HDII] - k[635]*y[IDX_HDII];
    data[3265] = 0.0 - k[652]*y[IDX_HDII] - k[653]*y[IDX_HDII] -
        k[2344]*y[IDX_HDII];
    data[3266] = 0.0 - k[2427]*y[IDX_HDII];
    data[3267] = 0.0 - k[2422]*y[IDX_HDII];
    data[3268] = 0.0 - k[668]*y[IDX_HDII] - k[669]*y[IDX_HDII] -
        k[2354]*y[IDX_HDII];
    data[3269] = 0.0 - k[2437]*y[IDX_HDII];
    data[3270] = 0.0 - k[660]*y[IDX_HDII] - k[661]*y[IDX_HDII] -
        k[2349]*y[IDX_HDII];
    data[3271] = 0.0 - k[2432]*y[IDX_HDII];
    data[3272] = 0.0 - k[2442]*y[IDX_HDII];
    data[3273] = 0.0 - k[674]*y[IDX_HDII] - k[675]*y[IDX_HDII] -
        k[2359]*y[IDX_HDII];
    data[3274] = 0.0 - k[680]*y[IDX_HDII] - k[681]*y[IDX_HDII] -
        k[2364]*y[IDX_HDII];
    data[3275] = 0.0 - k[794]*y[IDX_HDII] - k[2542]*y[IDX_HDII];
    data[3276] = 0.0 + k[811]*y[IDX_HeHII] - k[2102]*y[IDX_HDII] +
        k[2649]*y[IDX_HII] - k[2863]*y[IDX_HDII] - k[2864]*y[IDX_HDII] +
        k[2899]*y[IDX_pH2II];
    data[3277] = 0.0 + k[1549]*y[IDX_HCOI] + k[2648]*y[IDX_HI];
    data[3278] = 0.0 - k[2159]*y[IDX_HDII];
    data[3279] = 0.0 - k[2452]*y[IDX_HDII] - k[3051]*y[IDX_HDII] -
        k[3052]*y[IDX_HDII];
    data[3280] = 0.0 - k[2112]*y[IDX_HDII];
    data[3281] = 0.0 - k[807]*y[IDX_HDII] - k[808]*y[IDX_HDII] +
        k[1550]*y[IDX_HII] - k[2467]*y[IDX_HDII];
    data[3282] = 0.0 - k[171]*y[IDX_HDII] - k[172]*y[IDX_HDII];
    data[3283] = 0.0 + k[810]*y[IDX_HeDII] - k[2097]*y[IDX_HDII] +
        k[2648]*y[IDX_DII] + k[2865]*y[IDX_pD2II] + k[2866]*y[IDX_oD2II] -
        k[2897]*y[IDX_HDII] - k[2898]*y[IDX_HDII];
    data[3284] = 0.0 + k[1550]*y[IDX_DCOI] + k[2649]*y[IDX_DI];
    data[3285] = 0.0 - k[2158]*y[IDX_HDII];
    data[3286] = 0.0 - k[2447]*y[IDX_HDII] - k[3031]*y[IDX_HDII] -
        k[3032]*y[IDX_HDII];
    data[3287] = 0.0 - k[2107]*y[IDX_HDII];
    data[3288] = 0.0 - k[799]*y[IDX_HDII] - k[800]*y[IDX_HDII] +
        k[1549]*y[IDX_DII] - k[2462]*y[IDX_HDII];
    data[3289] = 0.0 + k[229] - k[736]*y[IDX_HDII] - k[737]*y[IDX_HDII] -
        k[738]*y[IDX_HDII] - k[739]*y[IDX_HDII] + k[2475]*y[IDX_HeII];
    data[3290] = 0.0 - k[171]*y[IDX_GRAINM] - k[172]*y[IDX_GRAINM] - k[294]
        - k[295] - k[634]*y[IDX_CI] - k[635]*y[IDX_CI] - k[640]*y[IDX_NI] -
        k[641]*y[IDX_NI] - k[646]*y[IDX_OI] - k[647]*y[IDX_OI] -
        k[652]*y[IDX_C2I] - k[653]*y[IDX_C2I] - k[660]*y[IDX_CHI] -
        k[661]*y[IDX_CHI] - k[668]*y[IDX_CDI] - k[669]*y[IDX_CDI] -
        k[674]*y[IDX_CNI] - k[675]*y[IDX_CNI] - k[680]*y[IDX_COI] -
        k[681]*y[IDX_COI] - k[693]*y[IDX_pH2I] - k[694]*y[IDX_pH2I] -
        k[695]*y[IDX_oH2I] - k[696]*y[IDX_oH2I] - k[697]*y[IDX_pH2I] -
        k[698]*y[IDX_oH2I] - k[718]*y[IDX_pD2I] - k[719]*y[IDX_pD2I] -
        k[720]*y[IDX_oD2I] - k[721]*y[IDX_oD2I] - k[722]*y[IDX_pD2I] -
        k[723]*y[IDX_pD2I] - k[724]*y[IDX_oD2I] - k[725]*y[IDX_oD2I] -
        k[736]*y[IDX_HDI] - k[737]*y[IDX_HDI] - k[738]*y[IDX_HDI] -
        k[739]*y[IDX_HDI] - k[744]*y[IDX_N2I] - k[745]*y[IDX_N2I] -
        k[752]*y[IDX_NHI] - k[753]*y[IDX_NHI] - k[760]*y[IDX_NDI] -
        k[761]*y[IDX_NDI] - k[766]*y[IDX_NOI] - k[767]*y[IDX_NOI] -
        k[772]*y[IDX_O2I] - k[773]*y[IDX_O2I] - k[780]*y[IDX_OHI] -
        k[781]*y[IDX_OHI] - k[788]*y[IDX_ODI] - k[789]*y[IDX_ODI] -
        k[794]*y[IDX_CO2I] - k[799]*y[IDX_HCOI] - k[800]*y[IDX_HCOI] -
        k[807]*y[IDX_DCOI] - k[808]*y[IDX_DCOI] - k[2097]*y[IDX_HI] -
        k[2102]*y[IDX_DI] - k[2107]*y[IDX_HCNI] - k[2112]*y[IDX_DCNI] -
        k[2117]*y[IDX_NH2I] - k[2122]*y[IDX_ND2I] - k[2127]*y[IDX_NHDI] -
        k[2158]*y[IDX_HM] - k[2159]*y[IDX_DM] - k[2344]*y[IDX_C2I] -
        k[2349]*y[IDX_CHI] - k[2354]*y[IDX_CDI] - k[2359]*y[IDX_CNI] -
        k[2364]*y[IDX_COI] - k[2392]*y[IDX_NHI] - k[2397]*y[IDX_NDI] -
        k[2402]*y[IDX_NOI] - k[2407]*y[IDX_O2I] - k[2412]*y[IDX_OHI] -
        k[2417]*y[IDX_ODI] - k[2422]*y[IDX_C2HI] - k[2427]*y[IDX_C2DI] -
        k[2432]*y[IDX_CH2I] - k[2437]*y[IDX_CD2I] - k[2442]*y[IDX_CHDI] -
        k[2447]*y[IDX_H2OI] - k[2452]*y[IDX_D2OI] - k[2457]*y[IDX_HDOI] -
        k[2462]*y[IDX_HCOI] - k[2467]*y[IDX_DCOI] - k[2542]*y[IDX_CO2I] -
        k[2749]*y[IDX_eM] - k[2755]*y[IDX_eM] - k[2863]*y[IDX_DI] -
        k[2864]*y[IDX_DI] - k[2886]*y[IDX_pH2I] - k[2887]*y[IDX_oH2I] -
        k[2897]*y[IDX_HI] - k[2898]*y[IDX_HI] - k[2930]*y[IDX_oD2I] -
        k[2932]*y[IDX_pD2I] - k[3031]*y[IDX_H2OI] - k[3032]*y[IDX_H2OI] -
        k[3041]*y[IDX_HDOI] - k[3042]*y[IDX_HDOI] - k[3051]*y[IDX_D2OI] -
        k[3052]*y[IDX_D2OI];
    data[3291] = 0.0 - k[2457]*y[IDX_HDII] - k[3041]*y[IDX_HDII] -
        k[3042]*y[IDX_HDII];
    data[3292] = 0.0 + k[2475]*y[IDX_HDI];
    data[3293] = 0.0 + k[810]*y[IDX_HI];
    data[3294] = 0.0 + k[811]*y[IDX_DI];
    data[3295] = 0.0 - k[640]*y[IDX_HDII] - k[641]*y[IDX_HDII];
    data[3296] = 0.0 - k[744]*y[IDX_HDII] - k[745]*y[IDX_HDII];
    data[3297] = 0.0 - k[760]*y[IDX_HDII] - k[761]*y[IDX_HDII] -
        k[2397]*y[IDX_HDII];
    data[3298] = 0.0 - k[2122]*y[IDX_HDII];
    data[3299] = 0.0 - k[752]*y[IDX_HDII] - k[753]*y[IDX_HDII] -
        k[2392]*y[IDX_HDII];
    data[3300] = 0.0 - k[2117]*y[IDX_HDII];
    data[3301] = 0.0 - k[2127]*y[IDX_HDII];
    data[3302] = 0.0 - k[766]*y[IDX_HDII] - k[767]*y[IDX_HDII] -
        k[2402]*y[IDX_HDII];
    data[3303] = 0.0 - k[646]*y[IDX_HDII] - k[647]*y[IDX_HDII];
    data[3304] = 0.0 - k[772]*y[IDX_HDII] - k[773]*y[IDX_HDII] -
        k[2407]*y[IDX_HDII];
    data[3305] = 0.0 - k[788]*y[IDX_HDII] - k[789]*y[IDX_HDII] -
        k[2417]*y[IDX_HDII];
    data[3306] = 0.0 - k[780]*y[IDX_HDII] - k[781]*y[IDX_HDII] -
        k[2412]*y[IDX_HDII];
    data[3307] = 0.0 - k[2749]*y[IDX_HDII] - k[2755]*y[IDX_HDII];
    data[3308] = 0.0 - k[720]*y[IDX_HDII] - k[721]*y[IDX_HDII] -
        k[724]*y[IDX_HDII] - k[725]*y[IDX_HDII] - k[2930]*y[IDX_HDII];
    data[3309] = 0.0 + k[2866]*y[IDX_HI];
    data[3310] = 0.0 + k[342];
    data[3311] = 0.0 - k[695]*y[IDX_HDII] - k[696]*y[IDX_HDII] -
        k[698]*y[IDX_HDII] - k[2887]*y[IDX_HDII];
    data[3312] = 0.0 + k[332];
    data[3313] = 0.0 - k[718]*y[IDX_HDII] - k[719]*y[IDX_HDII] -
        k[722]*y[IDX_HDII] - k[723]*y[IDX_HDII] - k[2932]*y[IDX_HDII];
    data[3314] = 0.0 + k[2865]*y[IDX_HI];
    data[3315] = 0.0 + k[343];
    data[3316] = 0.0 - k[693]*y[IDX_HDII] - k[694]*y[IDX_HDII] -
        k[697]*y[IDX_HDII] - k[2886]*y[IDX_HDII];
    data[3317] = 0.0 + k[2899]*y[IDX_DI];
    data[3318] = 0.0 + k[333];
    data[3319] = 0.0 - k[3286]*y[IDX_HD2OII] - k[3287]*y[IDX_HD2OII] -
        k[3288]*y[IDX_HD2OII];
    data[3320] = 0.0 - k[3338]*y[IDX_HD2OII] - k[3339]*y[IDX_HD2OII];
    data[3321] = 0.0 - k[3303]*y[IDX_HD2OII] - k[3304]*y[IDX_HD2OII];
    data[3322] = 0.0 + k[3308]*y[IDX_HDOI];
    data[3323] = 0.0 - k[3293]*y[IDX_HD2OII] - k[3294]*y[IDX_HD2OII] -
        k[3295]*y[IDX_HD2OII];
    data[3324] = 0.0 + k[3309]*y[IDX_D2OI];
    data[3325] = 0.0 - k[3348]*y[IDX_HD2OII] - k[3349]*y[IDX_HD2OII];
    data[3326] = 0.0 - k[3391]*y[IDX_HD2OII] - k[3392]*y[IDX_HD2OII] -
        k[3393]*y[IDX_HD2OII];
    data[3327] = 0.0 - k[3077]*y[IDX_HD2OII] - k[3078]*y[IDX_HD2OII] -
        k[3079]*y[IDX_HD2OII] - k[3080]*y[IDX_HD2OII] - k[3081]*y[IDX_HD2OII] -
        k[3105]*y[IDX_HD2OII] - k[3106]*y[IDX_HD2OII] - k[3107]*y[IDX_HD2OII];
    data[3328] = 0.0 + k[3047]*y[IDX_pH2II] + k[3048]*y[IDX_oH2II] +
        k[3052]*y[IDX_HDII] + k[3116]*y[IDX_HCNII] + k[3121]*y[IDX_HCOII] +
        k[3167]*y[IDX_H2OII] + k[3170]*y[IDX_HDOII] + k[3239]*y[IDX_pH3II] +
        k[3240]*y[IDX_pH3II] + k[3241]*y[IDX_oH3II] + k[3250]*y[IDX_pH2DII] +
        k[3251]*y[IDX_oH2DII] + k[3258]*y[IDX_pD2HII] + k[3259]*y[IDX_pD2HII] +
        k[3260]*y[IDX_oD2HII] + k[3261]*y[IDX_oD2HII] + k[3271]*y[IDX_HNCII] +
        k[3276]*y[IDX_HNOII] + k[3281]*y[IDX_N2HII] + k[3309]*y[IDX_CHII] +
        k[3314]*y[IDX_NHII] + k[3319]*y[IDX_OHII] + k[3331]*y[IDX_NH2II] +
        k[3334]*y[IDX_NHDII];
    data[3329] = 0.0 + k[3127]*y[IDX_pH2I] + k[3128]*y[IDX_oH2I] +
        k[3136]*y[IDX_HDI] + k[3148]*y[IDX_NHI] + k[3153]*y[IDX_OHI] +
        k[3159]*y[IDX_H2OI] + k[3166]*y[IDX_HDOI] + k[3173]*y[IDX_HCOI];
    data[3330] = 0.0 + k[3408]*y[IDX_oH2I] + k[3409]*y[IDX_pH2I] +
        k[3421]*y[IDX_HDI];
    data[3331] = 0.0 + k[3115]*y[IDX_HDOI];
    data[3332] = 0.0 + k[3175]*y[IDX_HDOII];
    data[3333] = 0.0 + k[3120]*y[IDX_HDOI];
    data[3334] = 0.0 + k[3270]*y[IDX_HDOI];
    data[3335] = 0.0 + k[3275]*y[IDX_HDOI];
    data[3336] = 0.0 - k[3375]*y[IDX_HD2OII] - k[3376]*y[IDX_HD2OII] -
        k[3377]*y[IDX_HD2OII] - k[3378]*y[IDX_HD2OII] - k[3379]*y[IDX_HD2OII];
    data[3337] = 0.0 - k[3071]*y[IDX_HD2OII] - k[3072]*y[IDX_HD2OII] -
        k[3073]*y[IDX_HD2OII] - k[3074]*y[IDX_HD2OII] - k[3075]*y[IDX_HD2OII] -
        k[3076]*y[IDX_HD2OII] - k[3100]*y[IDX_HD2OII] - k[3101]*y[IDX_HD2OII] -
        k[3102]*y[IDX_HD2OII] - k[3103]*y[IDX_HD2OII] - k[3104]*y[IDX_HD2OII];
    data[3338] = 0.0 + k[3414]*y[IDX_HDI] + k[3430]*y[IDX_oD2I] +
        k[3431]*y[IDX_pD2I];
    data[3339] = 0.0 + k[3033]*y[IDX_oD2II] + k[3034]*y[IDX_pD2II] +
        k[3159]*y[IDX_D2OII] + k[3183]*y[IDX_oD2HII] + k[3184]*y[IDX_oD2HII] +
        k[3185]*y[IDX_pD2HII] + k[3186]*y[IDX_pD2HII] + k[3197]*y[IDX_pD3II] +
        k[3198]*y[IDX_mD3II] + k[3199]*y[IDX_oD3II] + k[3323]*y[IDX_ND2II];
    data[3340] = 0.0 + k[3137]*y[IDX_oD2I] + k[3138]*y[IDX_pD2I] +
        k[3167]*y[IDX_D2OI];
    data[3341] = 0.0 + k[3422]*y[IDX_pD2I] + k[3423]*y[IDX_oD2I];
    data[3342] = 0.0 + k[3116]*y[IDX_D2OI];
    data[3343] = 0.0 + k[3173]*y[IDX_D2OII];
    data[3344] = 0.0 + k[3121]*y[IDX_D2OI];
    data[3345] = 0.0 + k[3133]*y[IDX_HDOII] + k[3136]*y[IDX_D2OII] +
        k[3414]*y[IDX_H2DOII] - k[3417]*y[IDX_HD2OII] - k[3418]*y[IDX_HD2OII] +
        k[3418]*y[IDX_HD2OII] - k[3419]*y[IDX_HD2OII] + k[3421]*y[IDX_D3OII];
    data[3346] = 0.0 + k[3041]*y[IDX_HDOI] + k[3052]*y[IDX_D2OI];
    data[3347] = 0.0 - k[3071]*y[IDX_HM] - k[3072]*y[IDX_HM] -
        k[3073]*y[IDX_HM] - k[3074]*y[IDX_HM] - k[3075]*y[IDX_HM] -
        k[3076]*y[IDX_HM] - k[3077]*y[IDX_DM] - k[3078]*y[IDX_DM] -
        k[3079]*y[IDX_DM] - k[3080]*y[IDX_DM] - k[3081]*y[IDX_DM] -
        k[3100]*y[IDX_HM] - k[3101]*y[IDX_HM] - k[3102]*y[IDX_HM] -
        k[3103]*y[IDX_HM] - k[3104]*y[IDX_HM] - k[3105]*y[IDX_DM] -
        k[3106]*y[IDX_DM] - k[3107]*y[IDX_DM] - k[3286]*y[IDX_CI] -
        k[3287]*y[IDX_CI] - k[3288]*y[IDX_CI] - k[3293]*y[IDX_CHI] -
        k[3294]*y[IDX_CHI] - k[3295]*y[IDX_CHI] - k[3303]*y[IDX_CDI] -
        k[3304]*y[IDX_CDI] - k[3338]*y[IDX_CM] - k[3339]*y[IDX_CM] -
        k[3343]*y[IDX_OM] - k[3344]*y[IDX_OM] - k[3348]*y[IDX_CNM] -
        k[3349]*y[IDX_CNM] - k[3361]*y[IDX_OHM] - k[3362]*y[IDX_OHM] -
        k[3363]*y[IDX_OHM] - k[3364]*y[IDX_OHM] - k[3365]*y[IDX_ODM] -
        k[3366]*y[IDX_ODM] - k[3367]*y[IDX_ODM] - k[3375]*y[IDX_HI] -
        k[3376]*y[IDX_HI] - k[3377]*y[IDX_HI] - k[3378]*y[IDX_HI] -
        k[3379]*y[IDX_HI] - k[3391]*y[IDX_DI] - k[3392]*y[IDX_DI] -
        k[3393]*y[IDX_DI] - k[3400]*y[IDX_pH2I] + k[3400]*y[IDX_pH2I] -
        k[3401]*y[IDX_oH2I] + k[3401]*y[IDX_oH2I] - k[3402]*y[IDX_pH2I] -
        k[3403]*y[IDX_oH2I] - k[3404]*y[IDX_pH2I] - k[3405]*y[IDX_oH2I] -
        k[3417]*y[IDX_HDI] - k[3418]*y[IDX_HDI] + k[3418]*y[IDX_HDI] -
        k[3419]*y[IDX_HDI] - k[3434]*y[IDX_oD2I] - k[3435]*y[IDX_pD2I] -
        k[3436]*y[IDX_oD2I] + k[3436]*y[IDX_oD2I] - k[3437]*y[IDX_pD2I] +
        k[3437]*y[IDX_pD2I] - k[3442]*y[IDX_eM] - k[3443]*y[IDX_eM] -
        k[3447]*y[IDX_eM] - k[3448]*y[IDX_eM] - k[3453]*y[IDX_eM] -
        k[3454]*y[IDX_eM] - k[3455]*y[IDX_eM] - k[3461]*y[IDX_eM] -
        k[3462]*y[IDX_eM] - k[3463]*y[IDX_eM];
    data[3348] = 0.0 + k[3041]*y[IDX_HDII] + k[3045]*y[IDX_pD2II] +
        k[3046]*y[IDX_oD2II] + k[3115]*y[IDX_DCNII] + k[3120]*y[IDX_DCOII] +
        k[3163]*y[IDX_HDOII] + k[3166]*y[IDX_D2OII] + k[3210]*y[IDX_pH2DII] +
        k[3211]*y[IDX_pH2DII] + k[3212]*y[IDX_oH2DII] + k[3213]*y[IDX_oH2DII] +
        k[3224]*y[IDX_pD2HII] + k[3225]*y[IDX_oD2HII] + k[3233]*y[IDX_mD3II] +
        k[3234]*y[IDX_mD3II] + k[3235]*y[IDX_pD3II] + k[3236]*y[IDX_pD3II] +
        k[3237]*y[IDX_oD3II] + k[3238]*y[IDX_oD3II] + k[3270]*y[IDX_DNCII] +
        k[3275]*y[IDX_DNOII] + k[3280]*y[IDX_N2DII] + k[3308]*y[IDX_CDII] +
        k[3313]*y[IDX_NDII] + k[3318]*y[IDX_ODII] + k[3327]*y[IDX_NHDII] +
        k[3330]*y[IDX_ND2II];
    data[3349] = 0.0 + k[3133]*y[IDX_HDI] + k[3143]*y[IDX_pD2I] +
        k[3144]*y[IDX_oD2I] + k[3150]*y[IDX_NDI] + k[3155]*y[IDX_ODI] +
        k[3163]*y[IDX_HDOI] + k[3170]*y[IDX_D2OI] + k[3175]*y[IDX_DCOI];
    data[3350] = 0.0 + k[3271]*y[IDX_D2OI];
    data[3351] = 0.0 + k[3276]*y[IDX_D2OI];
    data[3352] = 0.0 + k[3280]*y[IDX_HDOI];
    data[3353] = 0.0 + k[3281]*y[IDX_D2OI];
    data[3354] = 0.0 + k[3150]*y[IDX_HDOII];
    data[3355] = 0.0 + k[3313]*y[IDX_HDOI];
    data[3356] = 0.0 + k[3323]*y[IDX_H2OI] + k[3330]*y[IDX_HDOI];
    data[3357] = 0.0 + k[3148]*y[IDX_D2OII];
    data[3358] = 0.0 + k[3314]*y[IDX_D2OI];
    data[3359] = 0.0 + k[3331]*y[IDX_D2OI];
    data[3360] = 0.0 + k[3327]*y[IDX_HDOI] + k[3334]*y[IDX_D2OI];
    data[3361] = 0.0 - k[3343]*y[IDX_HD2OII] - k[3344]*y[IDX_HD2OII];
    data[3362] = 0.0 + k[3155]*y[IDX_HDOII];
    data[3363] = 0.0 + k[3318]*y[IDX_HDOI];
    data[3364] = 0.0 - k[3365]*y[IDX_HD2OII] - k[3366]*y[IDX_HD2OII] -
        k[3367]*y[IDX_HD2OII];
    data[3365] = 0.0 + k[3153]*y[IDX_D2OII];
    data[3366] = 0.0 + k[3319]*y[IDX_D2OI];
    data[3367] = 0.0 - k[3361]*y[IDX_HD2OII] - k[3362]*y[IDX_HD2OII] -
        k[3363]*y[IDX_HD2OII] - k[3364]*y[IDX_HD2OII];
    data[3368] = 0.0 - k[3442]*y[IDX_HD2OII] - k[3443]*y[IDX_HD2OII] -
        k[3447]*y[IDX_HD2OII] - k[3448]*y[IDX_HD2OII] - k[3453]*y[IDX_HD2OII] -
        k[3454]*y[IDX_HD2OII] - k[3455]*y[IDX_HD2OII] - k[3461]*y[IDX_HD2OII] -
        k[3462]*y[IDX_HD2OII] - k[3463]*y[IDX_HD2OII];
    data[3369] = 0.0 + k[3198]*y[IDX_H2OI] + k[3233]*y[IDX_HDOI] +
        k[3234]*y[IDX_HDOI];
    data[3370] = 0.0 + k[3137]*y[IDX_H2OII] + k[3144]*y[IDX_HDOII] +
        k[3423]*y[IDX_H3OII] + k[3430]*y[IDX_H2DOII] - k[3434]*y[IDX_HD2OII] -
        k[3436]*y[IDX_HD2OII] + k[3436]*y[IDX_HD2OII];
    data[3371] = 0.0 + k[3033]*y[IDX_H2OI] + k[3046]*y[IDX_HDOI];
    data[3372] = 0.0 + k[3183]*y[IDX_H2OI] + k[3184]*y[IDX_H2OI] +
        k[3225]*y[IDX_HDOI] + k[3260]*y[IDX_D2OI] + k[3261]*y[IDX_D2OI];
    data[3373] = 0.0 + k[3199]*y[IDX_H2OI] + k[3237]*y[IDX_HDOI] +
        k[3238]*y[IDX_HDOI];
    data[3374] = 0.0 + k[3128]*y[IDX_D2OII] - k[3401]*y[IDX_HD2OII] +
        k[3401]*y[IDX_HD2OII] - k[3403]*y[IDX_HD2OII] - k[3405]*y[IDX_HD2OII] +
        k[3408]*y[IDX_D3OII];
    data[3375] = 0.0 + k[3048]*y[IDX_D2OI];
    data[3376] = 0.0 + k[3212]*y[IDX_HDOI] + k[3213]*y[IDX_HDOI] +
        k[3251]*y[IDX_D2OI];
    data[3377] = 0.0 + k[3241]*y[IDX_D2OI];
    data[3378] = 0.0 + k[3138]*y[IDX_H2OII] + k[3143]*y[IDX_HDOII] +
        k[3422]*y[IDX_H3OII] + k[3431]*y[IDX_H2DOII] - k[3435]*y[IDX_HD2OII] -
        k[3437]*y[IDX_HD2OII] + k[3437]*y[IDX_HD2OII];
    data[3379] = 0.0 + k[3034]*y[IDX_H2OI] + k[3045]*y[IDX_HDOI];
    data[3380] = 0.0 + k[3185]*y[IDX_H2OI] + k[3186]*y[IDX_H2OI] +
        k[3224]*y[IDX_HDOI] + k[3258]*y[IDX_D2OI] + k[3259]*y[IDX_D2OI];
    data[3381] = 0.0 + k[3197]*y[IDX_H2OI] + k[3235]*y[IDX_HDOI] +
        k[3236]*y[IDX_HDOI];
    data[3382] = 0.0 + k[3127]*y[IDX_D2OII] - k[3400]*y[IDX_HD2OII] +
        k[3400]*y[IDX_HD2OII] - k[3402]*y[IDX_HD2OII] - k[3404]*y[IDX_HD2OII] +
        k[3409]*y[IDX_D3OII];
    data[3383] = 0.0 + k[3047]*y[IDX_D2OI];
    data[3384] = 0.0 + k[3210]*y[IDX_HDOI] + k[3211]*y[IDX_HDOI] +
        k[3250]*y[IDX_D2OI];
    data[3385] = 0.0 + k[3239]*y[IDX_D2OI] + k[3240]*y[IDX_D2OI];
    data[3386] = 0.0 - k[437]*y[IDX_HDOI] - k[438]*y[IDX_HDOI] -
        k[441]*y[IDX_HDOI] - k[442]*y[IDX_HDOI];
    data[3387] = 0.0 + k[3336]*y[IDX_H2DOII] + k[3339]*y[IDX_HD2OII];
    data[3388] = 0.0 + k[2481]*y[IDX_HDOII];
    data[3389] = 0.0 - k[1707]*y[IDX_HDOI] - k[1708]*y[IDX_HDOI];
    data[3390] = 0.0 + k[2493]*y[IDX_HDOII];
    data[3391] = 0.0 + k[2490]*y[IDX_HDOII];
    data[3392] = 0.0 - k[962]*y[IDX_HDOI] - k[963]*y[IDX_HDOI];
    data[3393] = 0.0 + k[2487]*y[IDX_HDOII] + k[3299]*y[IDX_H3OII] +
        k[3301]*y[IDX_H2DOII] + k[3303]*y[IDX_HD2OII];
    data[3394] = 0.0 - k[1785]*y[IDX_HDOI] - k[1786]*y[IDX_HDOI] -
        k[3308]*y[IDX_HDOI];
    data[3395] = 0.0 + k[2499]*y[IDX_HDOII];
    data[3396] = 0.0 + k[2484]*y[IDX_HDOII] + k[3292]*y[IDX_H2DOII] +
        k[3294]*y[IDX_HD2OII] + k[3296]*y[IDX_D3OII];
    data[3397] = 0.0 - k[1783]*y[IDX_HDOI] - k[1784]*y[IDX_HDOI] -
        k[3307]*y[IDX_HDOI];
    data[3398] = 0.0 + k[2496]*y[IDX_HDOII];
    data[3399] = 0.0 + k[2502]*y[IDX_HDOII];
    data[3400] = 0.0 - k[601]*y[IDX_HDOI] - k[602]*y[IDX_HDOI] -
        k[605]*y[IDX_HDOI] - k[606]*y[IDX_HDOI];
    data[3401] = 0.0 + k[3346]*y[IDX_H2DOII] + k[3349]*y[IDX_HD2OII];
    data[3402] = 0.0 - k[982]*y[IDX_HDOI] - k[983]*y[IDX_HDOI];
    data[3403] = 0.0 - k[624]*y[IDX_HDOI] - k[625]*y[IDX_HDOI] -
        k[2336]*y[IDX_HDOI];
    data[3404] = 0.0 + k[794]*y[IDX_HDII];
    data[3405] = 0.0 - k[2379]*y[IDX_HDOI];
    data[3406] = 0.0 - k[1090]*y[IDX_HDOI] - k[1091]*y[IDX_HDOI] +
        k[1122]*y[IDX_O2HI] + k[2668]*y[IDX_OHI] + k[2734]*y[IDX_OHM];
    data[3407] = 0.0 - k[2214]*y[IDX_HDOI];
    data[3408] = 0.0 - k[491]*y[IDX_HDOI] - k[492]*y[IDX_HDOI] +
        k[2712]*y[IDX_OHI] + k[3089]*y[IDX_H3OII] + k[3090]*y[IDX_H3OII] +
        k[3097]*y[IDX_H2DOII] + k[3106]*y[IDX_HD2OII] + k[3107]*y[IDX_HD2OII];
    data[3409] = 0.0 - k[3165]*y[IDX_HDOI] - k[3166]*y[IDX_HDOI];
    data[3410] = 0.0 + k[3109]*y[IDX_HM] + k[3110]*y[IDX_HM] +
        k[3296]*y[IDX_CHI] + k[3370]*y[IDX_OHM];
    data[3411] = 0.0 + k[507]*y[IDX_OHM];
    data[3412] = 0.0 - k[2385]*y[IDX_HDOI] - k[3115]*y[IDX_HDOI];
    data[3413] = 0.0 + k[573]*y[IDX_OHI] + k[2508]*y[IDX_HDOII];
    data[3414] = 0.0 - k[3120]*y[IDX_HDOI];
    data[3415] = 0.0 - k[3270]*y[IDX_HDOI];
    data[3416] = 0.0 + k[577]*y[IDX_OHI];
    data[3417] = 0.0 - k[3275]*y[IDX_HDOI];
    data[3418] = 0.0 - k[1085]*y[IDX_HDOI] - k[1086]*y[IDX_HDOI] +
        k[1121]*y[IDX_O2DI] + k[2667]*y[IDX_ODI] + k[2733]*y[IDX_ODM];
    data[3419] = 0.0 - k[2213]*y[IDX_HDOI];
    data[3420] = 0.0 - k[489]*y[IDX_HDOI] - k[490]*y[IDX_HDOI] +
        k[2713]*y[IDX_ODI] + k[3092]*y[IDX_H2DOII] + k[3093]*y[IDX_H2DOII] +
        k[3102]*y[IDX_HD2OII] + k[3109]*y[IDX_D3OII] + k[3110]*y[IDX_D3OII];
    data[3421] = 0.0 + k[3092]*y[IDX_HM] + k[3093]*y[IDX_HM] +
        k[3097]*y[IDX_DM] + k[3292]*y[IDX_CHI] + k[3301]*y[IDX_CDI] +
        k[3336]*y[IDX_CM] + k[3341]*y[IDX_OM] + k[3346]*y[IDX_CNM] +
        k[3354]*y[IDX_OHM] + k[3358]*y[IDX_ODM] + k[3359]*y[IDX_ODM] +
        k[3445]*y[IDX_eM];
    data[3422] = 0.0 - k[3161]*y[IDX_HDOI] - k[3162]*y[IDX_HDOI];
    data[3423] = 0.0 + k[3089]*y[IDX_DM] + k[3090]*y[IDX_DM] +
        k[3299]*y[IDX_CDI] + k[3351]*y[IDX_ODM];
    data[3424] = 0.0 + k[506]*y[IDX_ODM];
    data[3425] = 0.0 - k[2384]*y[IDX_HDOI] - k[3114]*y[IDX_HDOI];
    data[3426] = 0.0 + k[574]*y[IDX_ODI] + k[2505]*y[IDX_HDOII];
    data[3427] = 0.0 - k[3119]*y[IDX_HDOI];
    data[3428] = 0.0 + k[1170]*y[IDX_OHI] + k[1173]*y[IDX_ODI] +
        k[2001]*y[IDX_NO2II] + k[2727]*y[IDX_OM];
    data[3429] = 0.0 + k[794]*y[IDX_CO2I] - k[2457]*y[IDX_HDOI] -
        k[3041]*y[IDX_HDOI] - k[3042]*y[IDX_HDOI];
    data[3430] = 0.0 + k[3102]*y[IDX_HM] + k[3106]*y[IDX_DM] +
        k[3107]*y[IDX_DM] + k[3294]*y[IDX_CHI] + k[3303]*y[IDX_CDI] +
        k[3339]*y[IDX_CM] + k[3344]*y[IDX_OM] + k[3349]*y[IDX_CNM] +
        k[3362]*y[IDX_OHM] + k[3363]*y[IDX_OHM] + k[3367]*y[IDX_ODM] +
        k[3448]*y[IDX_eM];
    data[3431] = 0.0 - k[255] - k[256] - k[395] - k[396] - k[399] -
        k[437]*y[IDX_CII] - k[438]*y[IDX_CII] - k[441]*y[IDX_CII] -
        k[442]*y[IDX_CII] - k[489]*y[IDX_HM] - k[490]*y[IDX_HM] -
        k[491]*y[IDX_DM] - k[492]*y[IDX_DM] - k[601]*y[IDX_CNII] -
        k[602]*y[IDX_CNII] - k[605]*y[IDX_CNII] - k[606]*y[IDX_CNII] -
        k[624]*y[IDX_COII] - k[625]*y[IDX_COII] - k[905]*y[IDX_HeII] -
        k[906]*y[IDX_HeII] - k[909]*y[IDX_HeII] - k[910]*y[IDX_HeII] -
        k[962]*y[IDX_C2NII] - k[963]*y[IDX_C2NII] - k[982]*y[IDX_CNCII] -
        k[983]*y[IDX_CNCII] - k[1085]*y[IDX_HI] - k[1086]*y[IDX_HI] -
        k[1090]*y[IDX_DI] - k[1091]*y[IDX_DI] - k[1707]*y[IDX_C2II] -
        k[1708]*y[IDX_C2II] - k[1783]*y[IDX_CHII] - k[1784]*y[IDX_CHII] -
        k[1785]*y[IDX_CDII] - k[1786]*y[IDX_CDII] - k[1814]*y[IDX_N2II] -
        k[1815]*y[IDX_N2II] - k[1896]*y[IDX_NHII] - k[1897]*y[IDX_NHII] -
        k[1898]*y[IDX_NDII] - k[1899]*y[IDX_NDII] - k[1906]*y[IDX_NHII] -
        k[1907]*y[IDX_NHII] - k[1908]*y[IDX_NDII] - k[1909]*y[IDX_NDII] -
        k[2213]*y[IDX_HII] - k[2214]*y[IDX_DII] - k[2233]*y[IDX_NII] -
        k[2248]*y[IDX_OII] - k[2336]*y[IDX_COII] - k[2373]*y[IDX_HeII] -
        k[2379]*y[IDX_CO2II] - k[2384]*y[IDX_HCNII] - k[2385]*y[IDX_DCNII] -
        k[2453]*y[IDX_pH2II] - k[2454]*y[IDX_oH2II] - k[2455]*y[IDX_pD2II] -
        k[2456]*y[IDX_oD2II] - k[2457]*y[IDX_HDII] - k[2596]*y[IDX_OHII] -
        k[2597]*y[IDX_ODII] - k[2624]*y[IDX_N2II] - k[2629]*y[IDX_NHII] -
        k[2630]*y[IDX_NDII] - k[3037]*y[IDX_pH2II] - k[3038]*y[IDX_oH2II] -
        k[3039]*y[IDX_pH2II] - k[3040]*y[IDX_oH2II] - k[3041]*y[IDX_HDII] -
        k[3042]*y[IDX_HDII] - k[3043]*y[IDX_pD2II] - k[3044]*y[IDX_oD2II] -
        k[3045]*y[IDX_pD2II] - k[3046]*y[IDX_oD2II] - k[3114]*y[IDX_HCNII] -
        k[3115]*y[IDX_DCNII] - k[3119]*y[IDX_HCOII] - k[3120]*y[IDX_DCOII] -
        k[3161]*y[IDX_H2OII] - k[3162]*y[IDX_H2OII] - k[3163]*y[IDX_HDOII] -
        k[3164]*y[IDX_HDOII] - k[3165]*y[IDX_D2OII] - k[3166]*y[IDX_D2OII] -
        k[3204]*y[IDX_oH3II] - k[3205]*y[IDX_oH3II] - k[3206]*y[IDX_pH3II] -
        k[3207]*y[IDX_pH3II] - k[3208]*y[IDX_oH3II] - k[3209]*y[IDX_pH3II] -
        k[3210]*y[IDX_pH2DII] - k[3211]*y[IDX_pH2DII] - k[3212]*y[IDX_oH2DII] -
        k[3213]*y[IDX_oH2DII] - k[3214]*y[IDX_pH2DII] - k[3215]*y[IDX_oH2DII] -
        k[3216]*y[IDX_pH2DII] - k[3217]*y[IDX_pH2DII] - k[3218]*y[IDX_oH2DII] -
        k[3219]*y[IDX_oH2DII] - k[3220]*y[IDX_pD2HII] - k[3221]*y[IDX_pD2HII] -
        k[3222]*y[IDX_oD2HII] - k[3223]*y[IDX_oD2HII] - k[3224]*y[IDX_pD2HII] -
        k[3225]*y[IDX_oD2HII] - k[3226]*y[IDX_pD2HII] - k[3227]*y[IDX_pD2HII] -
        k[3228]*y[IDX_oD2HII] - k[3229]*y[IDX_oD2HII] - k[3230]*y[IDX_mD3II] -
        k[3231]*y[IDX_pD3II] - k[3232]*y[IDX_oD3II] - k[3233]*y[IDX_mD3II] -
        k[3234]*y[IDX_mD3II] - k[3235]*y[IDX_pD3II] - k[3236]*y[IDX_pD3II] -
        k[3237]*y[IDX_oD3II] - k[3238]*y[IDX_oD3II] - k[3269]*y[IDX_HNCII] -
        k[3270]*y[IDX_DNCII] - k[3274]*y[IDX_HNOII] - k[3275]*y[IDX_DNOII] -
        k[3279]*y[IDX_N2HII] - k[3280]*y[IDX_N2DII] - k[3307]*y[IDX_CHII] -
        k[3308]*y[IDX_CDII] - k[3312]*y[IDX_NHII] - k[3313]*y[IDX_NDII] -
        k[3317]*y[IDX_OHII] - k[3318]*y[IDX_ODII] - k[3325]*y[IDX_NH2II] -
        k[3326]*y[IDX_NH2II] - k[3327]*y[IDX_NHDII] - k[3328]*y[IDX_NHDII] -
        k[3329]*y[IDX_ND2II] - k[3330]*y[IDX_ND2II];
    data[3432] = 0.0 + k[2181]*y[IDX_NOI] + k[2184]*y[IDX_O2I] +
        k[2481]*y[IDX_C2I] + k[2484]*y[IDX_CHI] + k[2487]*y[IDX_CDI] +
        k[2490]*y[IDX_C2HI] + k[2493]*y[IDX_C2DI] + k[2496]*y[IDX_CH2I] +
        k[2499]*y[IDX_CD2I] + k[2502]*y[IDX_CHDI] + k[2505]*y[IDX_HCOI] +
        k[2508]*y[IDX_DCOI] + k[2513]*y[IDX_NH2I] + k[2516]*y[IDX_ND2I] +
        k[2519]*y[IDX_NHDI] - k[3163]*y[IDX_HDOI] - k[3164]*y[IDX_HDOI];
    data[3433] = 0.0 - k[3269]*y[IDX_HDOI];
    data[3434] = 0.0 + k[578]*y[IDX_ODI];
    data[3435] = 0.0 - k[3274]*y[IDX_HDOI];
    data[3436] = 0.0 - k[905]*y[IDX_HDOI] - k[906]*y[IDX_HDOI] -
        k[909]*y[IDX_HDOI] - k[910]*y[IDX_HDOI] - k[2373]*y[IDX_HDOI];
    data[3437] = 0.0 - k[2233]*y[IDX_HDOI];
    data[3438] = 0.0 - k[1814]*y[IDX_HDOI] - k[1815]*y[IDX_HDOI] -
        k[2624]*y[IDX_HDOI];
    data[3439] = 0.0 - k[3280]*y[IDX_HDOI];
    data[3440] = 0.0 - k[3279]*y[IDX_HDOI];
    data[3441] = 0.0 - k[1898]*y[IDX_HDOI] - k[1899]*y[IDX_HDOI] -
        k[1908]*y[IDX_HDOI] - k[1909]*y[IDX_HDOI] - k[2630]*y[IDX_HDOI] -
        k[3313]*y[IDX_HDOI];
    data[3442] = 0.0 + k[582]*y[IDX_OHI] + k[2516]*y[IDX_HDOII];
    data[3443] = 0.0 - k[3329]*y[IDX_HDOI] - k[3330]*y[IDX_HDOI];
    data[3444] = 0.0 - k[1896]*y[IDX_HDOI] - k[1897]*y[IDX_HDOI] -
        k[1906]*y[IDX_HDOI] - k[1907]*y[IDX_HDOI] - k[2629]*y[IDX_HDOI] -
        k[3312]*y[IDX_HDOI];
    data[3445] = 0.0 + k[585]*y[IDX_ODI] + k[2513]*y[IDX_HDOII];
    data[3446] = 0.0 - k[3325]*y[IDX_HDOI] - k[3326]*y[IDX_HDOI];
    data[3447] = 0.0 + k[568]*y[IDX_NOI] + k[583]*y[IDX_OHI] +
        k[589]*y[IDX_ODI] + k[2519]*y[IDX_HDOII];
    data[3448] = 0.0 - k[3327]*y[IDX_HDOI] - k[3328]*y[IDX_HDOI];
    data[3449] = 0.0 + k[568]*y[IDX_NHDI] + k[2181]*y[IDX_HDOII];
    data[3450] = 0.0 + k[2001]*y[IDX_HDI];
    data[3451] = 0.0 - k[2248]*y[IDX_HDOI];
    data[3452] = 0.0 + k[2727]*y[IDX_HDI] + k[3341]*y[IDX_H2DOII] +
        k[3344]*y[IDX_HD2OII];
    data[3453] = 0.0 + k[2184]*y[IDX_HDOII];
    data[3454] = 0.0 + k[591]*y[IDX_OHI] + k[1121]*y[IDX_HI];
    data[3455] = 0.0 + k[592]*y[IDX_ODI] + k[1122]*y[IDX_DI];
    data[3456] = 0.0 + k[570]*y[IDX_OHI] + k[574]*y[IDX_HCOI] +
        k[578]*y[IDX_HNOI] + k[585]*y[IDX_NH2I] + k[589]*y[IDX_NHDI] +
        k[592]*y[IDX_O2HI] + k[1160]*y[IDX_pH2I] + k[1161]*y[IDX_oH2I] +
        k[1173]*y[IDX_HDI] + k[2667]*y[IDX_HI] + k[2713]*y[IDX_HM];
    data[3457] = 0.0 - k[2597]*y[IDX_HDOI] - k[3318]*y[IDX_HDOI];
    data[3458] = 0.0 + k[506]*y[IDX_HCNI] + k[2733]*y[IDX_HI] +
        k[3351]*y[IDX_H3OII] + k[3358]*y[IDX_H2DOII] + k[3359]*y[IDX_H2DOII] +
        k[3367]*y[IDX_HD2OII];
    data[3459] = 0.0 + k[570]*y[IDX_ODI] + k[573]*y[IDX_DCOI] +
        k[577]*y[IDX_DNOI] + k[582]*y[IDX_ND2I] + k[583]*y[IDX_NHDI] +
        k[591]*y[IDX_O2DI] + k[1166]*y[IDX_pD2I] + k[1167]*y[IDX_oD2I] +
        k[1170]*y[IDX_HDI] + k[2668]*y[IDX_DI] + k[2712]*y[IDX_DM];
    data[3460] = 0.0 - k[2596]*y[IDX_HDOI] - k[3317]*y[IDX_HDOI];
    data[3461] = 0.0 + k[507]*y[IDX_DCNI] + k[2734]*y[IDX_DI] +
        k[3354]*y[IDX_H2DOII] + k[3362]*y[IDX_HD2OII] + k[3363]*y[IDX_HD2OII] +
        k[3370]*y[IDX_D3OII];
    data[3462] = 0.0 + k[3445]*y[IDX_H2DOII] + k[3448]*y[IDX_HD2OII];
    data[3463] = 0.0 - k[3230]*y[IDX_HDOI] - k[3233]*y[IDX_HDOI] -
        k[3234]*y[IDX_HDOI];
    data[3464] = 0.0 + k[1167]*y[IDX_OHI];
    data[3465] = 0.0 - k[2456]*y[IDX_HDOI] - k[3044]*y[IDX_HDOI] -
        k[3046]*y[IDX_HDOI];
    data[3466] = 0.0 - k[3222]*y[IDX_HDOI] - k[3223]*y[IDX_HDOI] -
        k[3225]*y[IDX_HDOI] - k[3228]*y[IDX_HDOI] - k[3229]*y[IDX_HDOI];
    data[3467] = 0.0 - k[3232]*y[IDX_HDOI] - k[3237]*y[IDX_HDOI] -
        k[3238]*y[IDX_HDOI];
    data[3468] = 0.0 + k[1161]*y[IDX_ODI];
    data[3469] = 0.0 - k[2454]*y[IDX_HDOI] - k[3038]*y[IDX_HDOI] -
        k[3040]*y[IDX_HDOI];
    data[3470] = 0.0 - k[3212]*y[IDX_HDOI] - k[3213]*y[IDX_HDOI] -
        k[3215]*y[IDX_HDOI] - k[3218]*y[IDX_HDOI] - k[3219]*y[IDX_HDOI];
    data[3471] = 0.0 - k[3204]*y[IDX_HDOI] - k[3205]*y[IDX_HDOI] -
        k[3208]*y[IDX_HDOI];
    data[3472] = 0.0 + k[1166]*y[IDX_OHI];
    data[3473] = 0.0 - k[2455]*y[IDX_HDOI] - k[3043]*y[IDX_HDOI] -
        k[3045]*y[IDX_HDOI];
    data[3474] = 0.0 - k[3220]*y[IDX_HDOI] - k[3221]*y[IDX_HDOI] -
        k[3224]*y[IDX_HDOI] - k[3226]*y[IDX_HDOI] - k[3227]*y[IDX_HDOI];
    data[3475] = 0.0 - k[3231]*y[IDX_HDOI] - k[3235]*y[IDX_HDOI] -
        k[3236]*y[IDX_HDOI];
    data[3476] = 0.0 + k[1160]*y[IDX_ODI];
    data[3477] = 0.0 - k[2453]*y[IDX_HDOI] - k[3037]*y[IDX_HDOI] -
        k[3039]*y[IDX_HDOI];
    data[3478] = 0.0 - k[3210]*y[IDX_HDOI] - k[3211]*y[IDX_HDOI] -
        k[3214]*y[IDX_HDOI] - k[3216]*y[IDX_HDOI] - k[3217]*y[IDX_HDOI];
    data[3479] = 0.0 - k[3206]*y[IDX_HDOI] - k[3207]*y[IDX_HDOI] -
        k[3209]*y[IDX_HDOI];
    data[3480] = 0.0 - k[989]*y[IDX_HDOII] - k[990]*y[IDX_HDOII];
    data[3481] = 0.0 - k[1240]*y[IDX_HDOII] - k[1241]*y[IDX_HDOII] -
        k[2481]*y[IDX_HDOII];
    data[3482] = 0.0 - k[2493]*y[IDX_HDOII];
    data[3483] = 0.0 - k[2490]*y[IDX_HDOII];
    data[3484] = 0.0 - k[1250]*y[IDX_HDOII] - k[1251]*y[IDX_HDOII] -
        k[2487]*y[IDX_HDOII];
    data[3485] = 0.0 - k[2499]*y[IDX_HDOII];
    data[3486] = 0.0 - k[1245]*y[IDX_HDOII] - k[1246]*y[IDX_HDOII] -
        k[2484]*y[IDX_HDOII];
    data[3487] = 0.0 - k[2496]*y[IDX_HDOII];
    data[3488] = 0.0 - k[2502]*y[IDX_HDOII];
    data[3489] = 0.0 - k[1254]*y[IDX_HDOII] - k[1255]*y[IDX_HDOII];
    data[3490] = 0.0 + k[2336]*y[IDX_HDOI];
    data[3491] = 0.0 + k[2379]*y[IDX_HDOI];
    data[3492] = 0.0 + k[3383]*y[IDX_H3OII] + k[3384]*y[IDX_H3OII] +
        k[3388]*y[IDX_H2DOII] + k[3392]*y[IDX_HD2OII] + k[3393]*y[IDX_HD2OII];
    data[3493] = 0.0 + k[2214]*y[IDX_HDOI];
    data[3494] = 0.0 - k[3169]*y[IDX_HDOII] - k[3170]*y[IDX_HDOII];
    data[3495] = 0.0 + k[3381]*y[IDX_HI] + k[3382]*y[IDX_HI];
    data[3496] = 0.0 + k[1013]*y[IDX_OHI] + k[2385]*y[IDX_HDOI];
    data[3497] = 0.0 + k[1967]*y[IDX_OHII] - k[2508]*y[IDX_HDOII] -
        k[3175]*y[IDX_HDOII];
    data[3498] = 0.0 + k[1029]*y[IDX_OHI];
    data[3499] = 0.0 + k[1567]*y[IDX_OHI];
    data[3500] = 0.0 + k[1587]*y[IDX_OHI];
    data[3501] = 0.0 + k[3372]*y[IDX_H2DOII] + k[3373]*y[IDX_H2DOII] +
        k[3377]*y[IDX_HD2OII] + k[3381]*y[IDX_D3OII] + k[3382]*y[IDX_D3OII];
    data[3502] = 0.0 + k[2213]*y[IDX_HDOI];
    data[3503] = 0.0 + k[3372]*y[IDX_HI] + k[3373]*y[IDX_HI] +
        k[3388]*y[IDX_DI];
    data[3504] = 0.0 - k[3157]*y[IDX_HDOII] - k[3158]*y[IDX_HDOII];
    data[3505] = 0.0 + k[3383]*y[IDX_DI] + k[3384]*y[IDX_DI];
    data[3506] = 0.0 + k[1014]*y[IDX_ODI] + k[2384]*y[IDX_HDOI];
    data[3507] = 0.0 + k[1966]*y[IDX_ODII] - k[2505]*y[IDX_HDOII] -
        k[3172]*y[IDX_HDOII];
    data[3508] = 0.0 + k[1030]*y[IDX_ODI];
    data[3509] = 0.0 + k[1949]*y[IDX_OHII] + k[1952]*y[IDX_ODII] -
        k[3133]*y[IDX_HDOII] - k[3134]*y[IDX_HDOII];
    data[3510] = 0.0 + k[780]*y[IDX_OHI] + k[789]*y[IDX_ODI] +
        k[2457]*y[IDX_HDOI];
    data[3511] = 0.0 + k[3377]*y[IDX_HI] + k[3392]*y[IDX_DI] +
        k[3393]*y[IDX_DI];
    data[3512] = 0.0 + k[399] + k[2213]*y[IDX_HII] + k[2214]*y[IDX_DII] +
        k[2233]*y[IDX_NII] + k[2248]*y[IDX_OII] + k[2336]*y[IDX_COII] +
        k[2373]*y[IDX_HeII] + k[2379]*y[IDX_CO2II] + k[2384]*y[IDX_HCNII] +
        k[2385]*y[IDX_DCNII] + k[2453]*y[IDX_pH2II] + k[2454]*y[IDX_oH2II] +
        k[2455]*y[IDX_pD2II] + k[2456]*y[IDX_oD2II] + k[2457]*y[IDX_HDII] +
        k[2596]*y[IDX_OHII] + k[2597]*y[IDX_ODII] + k[2624]*y[IDX_N2II] +
        k[2629]*y[IDX_NHII] + k[2630]*y[IDX_NDII] - k[3163]*y[IDX_HDOII] -
        k[3164]*y[IDX_HDOII];
    data[3513] = 0.0 - k[989]*y[IDX_CI] - k[990]*y[IDX_CI] -
        k[993]*y[IDX_NI] - k[994]*y[IDX_NI] - k[997]*y[IDX_OI] -
        k[1240]*y[IDX_C2I] - k[1241]*y[IDX_C2I] - k[1245]*y[IDX_CHI] -
        k[1246]*y[IDX_CHI] - k[1250]*y[IDX_CDI] - k[1251]*y[IDX_CDI] -
        k[1254]*y[IDX_COI] - k[1255]*y[IDX_COI] - k[2181]*y[IDX_NOI] -
        k[2184]*y[IDX_O2I] - k[2481]*y[IDX_C2I] - k[2484]*y[IDX_CHI] -
        k[2487]*y[IDX_CDI] - k[2490]*y[IDX_C2HI] - k[2493]*y[IDX_C2DI] -
        k[2496]*y[IDX_CH2I] - k[2499]*y[IDX_CD2I] - k[2502]*y[IDX_CHDI] -
        k[2505]*y[IDX_HCOI] - k[2508]*y[IDX_DCOI] - k[2513]*y[IDX_NH2I] -
        k[2516]*y[IDX_ND2I] - k[2519]*y[IDX_NHDI] - k[2790]*y[IDX_eM] -
        k[2793]*y[IDX_eM] - k[2794]*y[IDX_eM] - k[2797]*y[IDX_eM] -
        k[3123]*y[IDX_pH2I] - k[3124]*y[IDX_oH2I] - k[3125]*y[IDX_pH2I] -
        k[3126]*y[IDX_oH2I] - k[3133]*y[IDX_HDI] - k[3134]*y[IDX_HDI] -
        k[3141]*y[IDX_pD2I] - k[3142]*y[IDX_oD2I] - k[3143]*y[IDX_pD2I] -
        k[3144]*y[IDX_oD2I] - k[3147]*y[IDX_NHI] - k[3150]*y[IDX_NDI] -
        k[3152]*y[IDX_OHI] - k[3155]*y[IDX_ODI] - k[3157]*y[IDX_H2OI] -
        k[3158]*y[IDX_H2OI] - k[3163]*y[IDX_HDOI] - k[3164]*y[IDX_HDOI] -
        k[3169]*y[IDX_D2OI] - k[3170]*y[IDX_D2OI] - k[3172]*y[IDX_HCOI] -
        k[3175]*y[IDX_DCOI];
    data[3514] = 0.0 + k[1568]*y[IDX_ODI];
    data[3515] = 0.0 + k[1588]*y[IDX_ODI];
    data[3516] = 0.0 + k[2373]*y[IDX_HDOI];
    data[3517] = 0.0 - k[993]*y[IDX_HDOII] - k[994]*y[IDX_HDOII];
    data[3518] = 0.0 + k[2233]*y[IDX_HDOI];
    data[3519] = 0.0 + k[2624]*y[IDX_HDOI];
    data[3520] = 0.0 + k[1631]*y[IDX_OHI];
    data[3521] = 0.0 + k[1632]*y[IDX_ODI];
    data[3522] = 0.0 - k[3150]*y[IDX_HDOII];
    data[3523] = 0.0 + k[1883]*y[IDX_OHI] + k[2630]*y[IDX_HDOI];
    data[3524] = 0.0 - k[2516]*y[IDX_HDOII];
    data[3525] = 0.0 - k[3147]*y[IDX_HDOII];
    data[3526] = 0.0 + k[1884]*y[IDX_ODI] + k[2629]*y[IDX_HDOI];
    data[3527] = 0.0 - k[2513]*y[IDX_HDOII];
    data[3528] = 0.0 - k[2519]*y[IDX_HDOII];
    data[3529] = 0.0 - k[2181]*y[IDX_HDOII];
    data[3530] = 0.0 - k[997]*y[IDX_HDOII] + k[1302]*y[IDX_oH2DII] +
        k[1303]*y[IDX_pH2DII] + k[1308]*y[IDX_oD2HII] + k[1309]*y[IDX_pD2HII];
    data[3531] = 0.0 + k[2248]*y[IDX_HDOI];
    data[3532] = 0.0 - k[2184]*y[IDX_HDOII];
    data[3533] = 0.0 + k[2041]*y[IDX_OHI];
    data[3534] = 0.0 + k[2042]*y[IDX_ODI];
    data[3535] = 0.0 + k[782]*y[IDX_pH2II] + k[783]*y[IDX_oH2II] +
        k[789]*y[IDX_HDII] + k[1014]*y[IDX_HCNII] + k[1030]*y[IDX_HCOII] +
        k[1494]*y[IDX_oH3II] + k[1495]*y[IDX_pH3II] + k[1496]*y[IDX_pH3II] +
        k[1504]*y[IDX_oH2DII] + k[1505]*y[IDX_pH2DII] + k[1506]*y[IDX_oD2HII] +
        k[1507]*y[IDX_pD2HII] + k[1508]*y[IDX_pD2HII] + k[1568]*y[IDX_HNCII] +
        k[1588]*y[IDX_HNOII] + k[1632]*y[IDX_N2HII] + k[1884]*y[IDX_NHII] +
        k[1963]*y[IDX_OHII] + k[2042]*y[IDX_O2HII] - k[3155]*y[IDX_HDOII];
    data[3536] = 0.0 + k[1939]*y[IDX_pH2I] + k[1940]*y[IDX_oH2I] +
        k[1952]*y[IDX_HDI] + k[1962]*y[IDX_OHI] + k[1966]*y[IDX_HCOI] +
        k[2597]*y[IDX_HDOI];
    data[3537] = 0.0 + k[778]*y[IDX_pD2II] + k[779]*y[IDX_oD2II] +
        k[780]*y[IDX_HDII] + k[1013]*y[IDX_DCNII] + k[1029]*y[IDX_DCOII] +
        k[1484]*y[IDX_oH2DII] + k[1485]*y[IDX_pH2DII] + k[1486]*y[IDX_pH2DII] +
        k[1492]*y[IDX_oD2HII] + k[1493]*y[IDX_pD2HII] + k[1567]*y[IDX_DNCII] +
        k[1587]*y[IDX_DNOII] + k[1631]*y[IDX_N2DII] + k[1883]*y[IDX_NDII] +
        k[1962]*y[IDX_ODII] + k[2041]*y[IDX_O2DII] - k[3152]*y[IDX_HDOII];
    data[3538] = 0.0 + k[1945]*y[IDX_pD2I] + k[1946]*y[IDX_oD2I] +
        k[1949]*y[IDX_HDI] + k[1963]*y[IDX_ODI] + k[1967]*y[IDX_DCOI] +
        k[2596]*y[IDX_HDOI];
    data[3539] = 0.0 - k[2790]*y[IDX_HDOII] - k[2793]*y[IDX_HDOII] -
        k[2794]*y[IDX_HDOII] - k[2797]*y[IDX_HDOII];
    data[3540] = 0.0 + k[1946]*y[IDX_OHII] - k[3142]*y[IDX_HDOII] -
        k[3144]*y[IDX_HDOII];
    data[3541] = 0.0 + k[779]*y[IDX_OHI] + k[2456]*y[IDX_HDOI];
    data[3542] = 0.0 + k[1308]*y[IDX_OI] + k[1492]*y[IDX_OHI] +
        k[1506]*y[IDX_ODI];
    data[3543] = 0.0 + k[1940]*y[IDX_ODII] - k[3124]*y[IDX_HDOII] -
        k[3126]*y[IDX_HDOII];
    data[3544] = 0.0 + k[783]*y[IDX_ODI] + k[2454]*y[IDX_HDOI];
    data[3545] = 0.0 + k[1302]*y[IDX_OI] + k[1484]*y[IDX_OHI] +
        k[1504]*y[IDX_ODI];
    data[3546] = 0.0 + k[1494]*y[IDX_ODI];
    data[3547] = 0.0 + k[1945]*y[IDX_OHII] - k[3141]*y[IDX_HDOII] -
        k[3143]*y[IDX_HDOII];
    data[3548] = 0.0 + k[778]*y[IDX_OHI] + k[2455]*y[IDX_HDOI];
    data[3549] = 0.0 + k[1309]*y[IDX_OI] + k[1493]*y[IDX_OHI] +
        k[1507]*y[IDX_ODI] + k[1508]*y[IDX_ODI];
    data[3550] = 0.0 + k[1939]*y[IDX_ODII] - k[3123]*y[IDX_HDOII] -
        k[3125]*y[IDX_HDOII];
    data[3551] = 0.0 + k[782]*y[IDX_ODI] + k[2453]*y[IDX_HDOI];
    data[3552] = 0.0 + k[1303]*y[IDX_OI] + k[1485]*y[IDX_OHI] +
        k[1486]*y[IDX_OHI] + k[1505]*y[IDX_ODI];
    data[3553] = 0.0 + k[1495]*y[IDX_ODI] + k[1496]*y[IDX_ODI];
    data[3554] = 0.0 + k[1045]*y[IDX_NH2I] + k[1048]*y[IDX_NHDI];
    data[3555] = 0.0 - k[449]*y[IDX_HNCI];
    data[3556] = 0.0 + k[1210]*y[IDX_NI];
    data[3557] = 0.0 + k[1213]*y[IDX_NI];
    data[3558] = 0.0 - k[1680]*y[IDX_HNCI] - k[1681]*y[IDX_HNCI];
    data[3559] = 0.0 - k[1679]*y[IDX_HNCI];
    data[3560] = 0.0 - k[263] - k[406] - k[449]*y[IDX_CII] -
        k[925]*y[IDX_HeII] - k[927]*y[IDX_HeII] - k[929]*y[IDX_HeII] -
        k[1679]*y[IDX_HII] - k[1680]*y[IDX_DII] - k[1681]*y[IDX_DII];
    data[3561] = 0.0 + k[2219]*y[IDX_NOI];
    data[3562] = 0.0 - k[925]*y[IDX_HNCI] - k[927]*y[IDX_HNCI] -
        k[929]*y[IDX_HNCI];
    data[3563] = 0.0 + k[1210]*y[IDX_CH2I] + k[1213]*y[IDX_CHDI];
    data[3564] = 0.0 + k[1045]*y[IDX_CI];
    data[3565] = 0.0 + k[1048]*y[IDX_CI];
    data[3566] = 0.0 + k[2219]*y[IDX_HNCII];
    data[3567] = 0.0 - k[1552]*y[IDX_HNCII];
    data[3568] = 0.0 - k[1554]*y[IDX_HNCII];
    data[3569] = 0.0 - k[1558]*y[IDX_HNCII];
    data[3570] = 0.0 - k[1556]*y[IDX_HNCII];
    data[3571] = 0.0 + k[468]*y[IDX_pH2I] + k[469]*y[IDX_oH2I] +
        k[473]*y[IDX_HDI];
    data[3572] = 0.0 - k[3271]*y[IDX_HNCII];
    data[3573] = 0.0 - k[3007]*y[IDX_HNCII];
    data[3574] = 0.0 + k[473]*y[IDX_CNII];
    data[3575] = 0.0 - k[3269]*y[IDX_HNCII];
    data[3576] = 0.0 - k[1552]*y[IDX_CI] - k[1554]*y[IDX_C2I] -
        k[1556]*y[IDX_CHI] - k[1558]*y[IDX_CDI] - k[1560]*y[IDX_NHI] -
        k[1562]*y[IDX_NDI] - k[1564]*y[IDX_O2I] - k[1566]*y[IDX_OHI] -
        k[1568]*y[IDX_ODI] - k[2219]*y[IDX_NOI] - k[2824]*y[IDX_eM] -
        k[3007]*y[IDX_H2OI] - k[3269]*y[IDX_HDOI] - k[3271]*y[IDX_D2OI];
    data[3577] = 0.0 - k[1562]*y[IDX_HNCII];
    data[3578] = 0.0 - k[1560]*y[IDX_HNCII];
    data[3579] = 0.0 - k[2219]*y[IDX_HNCII];
    data[3580] = 0.0 - k[1564]*y[IDX_HNCII];
    data[3581] = 0.0 - k[1568]*y[IDX_HNCII];
    data[3582] = 0.0 - k[1566]*y[IDX_HNCII];
    data[3583] = 0.0 - k[2824]*y[IDX_HNCII];
    data[3584] = 0.0 + k[469]*y[IDX_CNII];
    data[3585] = 0.0 + k[468]*y[IDX_CNII];
    data[3586] = 0.0 - k[543]*y[IDX_HNOI];
    data[3587] = 0.0 - k[541]*y[IDX_HNOI];
    data[3588] = 0.0 - k[555]*y[IDX_HNOI];
    data[3589] = 0.0 - k[559]*y[IDX_HNOI];
    data[3590] = 0.0 - k[1102]*y[IDX_HNOI] - k[1107]*y[IDX_HNOI] -
        k[1108]*y[IDX_HNOI];
    data[3591] = 0.0 - k[1686]*y[IDX_HNOI];
    data[3592] = 0.0 - k[1100]*y[IDX_HNOI] - k[1104]*y[IDX_HNOI];
    data[3593] = 0.0 - k[1685]*y[IDX_HNOI];
    data[3594] = 0.0 - k[265] - k[408] - k[521]*y[IDX_OI] -
        k[541]*y[IDX_CHI] - k[543]*y[IDX_CDI] - k[555]*y[IDX_CNI] -
        k[559]*y[IDX_COI] - k[576]*y[IDX_OHI] - k[578]*y[IDX_ODI] -
        k[931]*y[IDX_HeII] - k[933]*y[IDX_HeII] - k[1100]*y[IDX_HI] -
        k[1102]*y[IDX_DI] - k[1104]*y[IDX_HI] - k[1107]*y[IDX_DI] -
        k[1108]*y[IDX_DI] - k[1685]*y[IDX_HII] - k[1686]*y[IDX_DII];
    data[3595] = 0.0 + k[2221]*y[IDX_NOI];
    data[3596] = 0.0 - k[931]*y[IDX_HNOI] - k[933]*y[IDX_HNOI];
    data[3597] = 0.0 + k[527]*y[IDX_OI];
    data[3598] = 0.0 + k[530]*y[IDX_OI];
    data[3599] = 0.0 + k[2221]*y[IDX_HNOII];
    data[3600] = 0.0 - k[521]*y[IDX_HNOI] + k[527]*y[IDX_NH2I] +
        k[530]*y[IDX_NHDI];
    data[3601] = 0.0 - k[578]*y[IDX_HNOI];
    data[3602] = 0.0 - k[576]*y[IDX_HNOI];
    data[3603] = 0.0 - k[1570]*y[IDX_HNOII];
    data[3604] = 0.0 - k[1574]*y[IDX_HNOII];
    data[3605] = 0.0 - k[1578]*y[IDX_HNOII];
    data[3606] = 0.0 - k[1576]*y[IDX_HNOII];
    data[3607] = 0.0 - k[1580]*y[IDX_HNOII];
    data[3608] = 0.0 + k[1888]*y[IDX_NHII];
    data[3609] = 0.0 + k[1893]*y[IDX_NHII] - k[3276]*y[IDX_HNOII];
    data[3610] = 0.0 + k[1890]*y[IDX_NHII] + k[1892]*y[IDX_NDII] -
        k[3008]*y[IDX_HNOII];
    data[3611] = 0.0 + k[991]*y[IDX_NI];
    data[3612] = 0.0 + k[767]*y[IDX_NOI];
    data[3613] = 0.0 + k[1897]*y[IDX_NHII] + k[1898]*y[IDX_NDII] -
        k[3274]*y[IDX_HNOII];
    data[3614] = 0.0 + k[994]*y[IDX_NI];
    data[3615] = 0.0 + k[265];
    data[3616] = 0.0 - k[1570]*y[IDX_CI] - k[1572]*y[IDX_OI] -
        k[1574]*y[IDX_C2I] - k[1576]*y[IDX_CHI] - k[1578]*y[IDX_CDI] -
        k[1580]*y[IDX_COI] - k[1582]*y[IDX_NHI] - k[1584]*y[IDX_NDI] -
        k[1586]*y[IDX_OHI] - k[1588]*y[IDX_ODI] - k[2221]*y[IDX_NOI] -
        k[2826]*y[IDX_eM] - k[3008]*y[IDX_H2OI] - k[3274]*y[IDX_HDOI] -
        k[3276]*y[IDX_D2OI];
    data[3617] = 0.0 + k[991]*y[IDX_H2OII] + k[994]*y[IDX_HDOII];
    data[3618] = 0.0 - k[1584]*y[IDX_HNOII];
    data[3619] = 0.0 + k[1892]*y[IDX_H2OI] + k[1898]*y[IDX_HDOI];
    data[3620] = 0.0 - k[1582]*y[IDX_HNOII] + k[1915]*y[IDX_O2II];
    data[3621] = 0.0 + k[1888]*y[IDX_CO2I] + k[1890]*y[IDX_H2OI] +
        k[1893]*y[IDX_D2OI] + k[1897]*y[IDX_HDOI];
    data[3622] = 0.0 + k[1973]*y[IDX_OI] + k[1991]*y[IDX_O2I];
    data[3623] = 0.0 + k[1976]*y[IDX_OI] + k[1994]*y[IDX_O2I];
    data[3624] = 0.0 + k[762]*y[IDX_pH2II] + k[763]*y[IDX_oH2II] +
        k[767]*y[IDX_HDII] + k[1449]*y[IDX_oH3II] + k[1450]*y[IDX_pH3II] +
        k[1451]*y[IDX_pH3II] + k[1457]*y[IDX_oH2DII] + k[1458]*y[IDX_pH2DII] +
        k[1459]*y[IDX_oD2HII] + k[1460]*y[IDX_pD2HII] + k[1461]*y[IDX_pD2HII] +
        k[1959]*y[IDX_OHII] + k[2038]*y[IDX_O2HII] - k[2221]*y[IDX_HNOII];
    data[3625] = 0.0 - k[1572]*y[IDX_HNOII] + k[1973]*y[IDX_NH2II] +
        k[1976]*y[IDX_NHDII];
    data[3626] = 0.0 + k[1991]*y[IDX_NH2II] + k[1994]*y[IDX_NHDII];
    data[3627] = 0.0 + k[1915]*y[IDX_NHI];
    data[3628] = 0.0 + k[2038]*y[IDX_NOI];
    data[3629] = 0.0 - k[1588]*y[IDX_HNOII];
    data[3630] = 0.0 - k[1586]*y[IDX_HNOII];
    data[3631] = 0.0 + k[1959]*y[IDX_NOI];
    data[3632] = 0.0 - k[2826]*y[IDX_HNOII];
    data[3633] = 0.0 + k[1459]*y[IDX_NOI];
    data[3634] = 0.0 + k[763]*y[IDX_NOI];
    data[3635] = 0.0 + k[1457]*y[IDX_NOI];
    data[3636] = 0.0 + k[1449]*y[IDX_NOI];
    data[3637] = 0.0 + k[1460]*y[IDX_NOI] + k[1461]*y[IDX_NOI];
    data[3638] = 0.0 + k[762]*y[IDX_NOI];
    data[3639] = 0.0 + k[1458]*y[IDX_NOI];
    data[3640] = 0.0 + k[1450]*y[IDX_NOI] + k[1451]*y[IDX_NOI];
    data[3641] = 0.0 + k[439]*y[IDX_H2OI] + k[442]*y[IDX_HDOI];
    data[3642] = 0.0 + k[1387]*y[IDX_oH3II] + k[1388]*y[IDX_pH3II] +
        k[1389]*y[IDX_pH3II] + k[1395]*y[IDX_oH2DII] + k[1396]*y[IDX_pH2DII] +
        k[1397]*y[IDX_oD2HII] + k[1398]*y[IDX_pD2HII] + k[1399]*y[IDX_pD2HII] -
        k[1590]*y[IDX_HOCII];
    data[3643] = 0.0 + k[456]*y[IDX_pH2I] + k[457]*y[IDX_oH2I] +
        k[461]*y[IDX_HDI];
    data[3644] = 0.0 + k[439]*y[IDX_CII];
    data[3645] = 0.0 + k[461]*y[IDX_COII] - k[1608]*y[IDX_HOCII] -
        k[1609]*y[IDX_HOCII] - k[1610]*y[IDX_HOCII];
    data[3646] = 0.0 + k[442]*y[IDX_CII];
    data[3647] = 0.0 - k[1590]*y[IDX_COI] - k[1592]*y[IDX_pH2I] -
        k[1593]*y[IDX_oH2I] - k[1594]*y[IDX_oH2I] - k[1600]*y[IDX_pD2I] -
        k[1601]*y[IDX_oD2I] - k[1602]*y[IDX_oD2I] - k[1603]*y[IDX_pD2I] -
        k[1604]*y[IDX_oD2I] - k[1608]*y[IDX_HDI] - k[1609]*y[IDX_HDI] -
        k[1610]*y[IDX_HDI] - k[1614]*y[IDX_N2I] - k[2828]*y[IDX_eM];
    data[3648] = 0.0 - k[1614]*y[IDX_HOCII];
    data[3649] = 0.0 - k[2828]*y[IDX_HOCII];
    data[3650] = 0.0 - k[1601]*y[IDX_HOCII] - k[1602]*y[IDX_HOCII] -
        k[1604]*y[IDX_HOCII];
    data[3651] = 0.0 + k[1397]*y[IDX_COI];
    data[3652] = 0.0 + k[457]*y[IDX_COII] - k[1593]*y[IDX_HOCII] -
        k[1594]*y[IDX_HOCII];
    data[3653] = 0.0 + k[1395]*y[IDX_COI];
    data[3654] = 0.0 + k[1387]*y[IDX_COI];
    data[3655] = 0.0 - k[1600]*y[IDX_HOCII] - k[1603]*y[IDX_HOCII];
    data[3656] = 0.0 + k[1398]*y[IDX_COI] + k[1399]*y[IDX_COI];
    data[3657] = 0.0 + k[456]*y[IDX_COII] - k[1592]*y[IDX_HOCII];
    data[3658] = 0.0 + k[1396]*y[IDX_COI];
    data[3659] = 0.0 + k[1388]*y[IDX_COI] + k[1389]*y[IDX_COI];
    data[3660] = 0.0 + k[2141]*y[IDX_HeII];
    data[3661] = 0.0 + k[862]*y[IDX_HeII] + k[2468]*y[IDX_HeII];
    data[3662] = 0.0 + k[883]*y[IDX_HeII] + k[885]*y[IDX_HeII] +
        k[887]*y[IDX_HeII];
    data[3663] = 0.0 + k[882]*y[IDX_HeII] + k[884]*y[IDX_HeII] +
        k[886]*y[IDX_HeII];
    data[3664] = 0.0 + k[888]*y[IDX_HeII];
    data[3665] = 0.0 + k[889]*y[IDX_HeII] + k[890]*y[IDX_HeII];
    data[3666] = 0.0 + k[891]*y[IDX_HeII];
    data[3667] = 0.0 + k[864]*y[IDX_HeII] + k[2470]*y[IDX_HeII];
    data[3668] = 0.0 + k[893]*y[IDX_HeII] + k[896]*y[IDX_HeII];
    data[3669] = 0.0 + k[863]*y[IDX_HeII] + k[2469]*y[IDX_HeII];
    data[3670] = 0.0 + k[892]*y[IDX_HeII] + k[895]*y[IDX_HeII];
    data[3671] = 0.0 + k[894]*y[IDX_HeII] + k[897]*y[IDX_HeII] +
        k[898]*y[IDX_HeII];
    data[3672] = 0.0 + k[865]*y[IDX_HeII] + k[866]*y[IDX_HeII];
    data[3673] = 0.0 + k[867]*y[IDX_HeII];
    data[3674] = 0.0 + k[899]*y[IDX_HeII] + k[900]*y[IDX_HeII] +
        k[901]*y[IDX_HeII] + k[902]*y[IDX_HeII] + k[2478]*y[IDX_HeII];
    data[3675] = 0.0 + k[811]*y[IDX_HeHII] + k[812]*y[IDX_HeDII] +
        k[2161]*y[IDX_HeII];
    data[3676] = 0.0 + k[2143]*y[IDX_HeII];
    data[3677] = 0.0 + k[904]*y[IDX_HeII] + k[908]*y[IDX_HeII] +
        k[2372]*y[IDX_HeII];
    data[3678] = 0.0 + k[912]*y[IDX_HeII] + k[914]*y[IDX_HeII] +
        k[916]*y[IDX_HeII] + k[918]*y[IDX_HeII];
    data[3679] = 0.0 + k[920]*y[IDX_HeII] + k[922]*y[IDX_HeII];
    data[3680] = 0.0 + k[926]*y[IDX_HeII] + k[928]*y[IDX_HeII] +
        k[930]*y[IDX_HeII];
    data[3681] = 0.0 + k[932]*y[IDX_HeII] + k[934]*y[IDX_HeII];
    data[3682] = 0.0 + k[177]*y[IDX_HeII];
    data[3683] = 0.0 + k[809]*y[IDX_HeHII] + k[810]*y[IDX_HeDII] +
        k[2160]*y[IDX_HeII];
    data[3684] = 0.0 + k[2142]*y[IDX_HeII];
    data[3685] = 0.0 + k[903]*y[IDX_HeII] + k[907]*y[IDX_HeII] +
        k[2371]*y[IDX_HeII];
    data[3686] = 0.0 + k[911]*y[IDX_HeII] + k[913]*y[IDX_HeII] +
        k[915]*y[IDX_HeII] + k[917]*y[IDX_HeII];
    data[3687] = 0.0 + k[919]*y[IDX_HeII] + k[921]*y[IDX_HeII];
    data[3688] = 0.0 + k[823]*y[IDX_HeHII] + k[824]*y[IDX_HeHII] +
        k[825]*y[IDX_HeDII] + k[826]*y[IDX_HeDII] + k[872]*y[IDX_HeII] +
        k[873]*y[IDX_HeII] + k[2475]*y[IDX_HeII];
    data[3689] = 0.0 + k[905]*y[IDX_HeII] + k[906]*y[IDX_HeII] +
        k[909]*y[IDX_HeII] + k[910]*y[IDX_HeII] + k[2373]*y[IDX_HeII];
    data[3690] = 0.0 + k[925]*y[IDX_HeII] + k[927]*y[IDX_HeII] +
        k[929]*y[IDX_HeII];
    data[3691] = 0.0 + k[931]*y[IDX_HeII] + k[933]*y[IDX_HeII];
    data[3692] = 0.0 - k[205];
    data[3693] = 0.0 + k[177]*y[IDX_GRAINM] + k[862]*y[IDX_C2I] +
        k[863]*y[IDX_CHI] + k[864]*y[IDX_CDI] + k[865]*y[IDX_CNI] +
        k[866]*y[IDX_CNI] + k[867]*y[IDX_COI] + k[868]*y[IDX_pH2I] +
        k[869]*y[IDX_oH2I] + k[870]*y[IDX_pD2I] + k[871]*y[IDX_oD2I] +
        k[872]*y[IDX_HDI] + k[873]*y[IDX_HDI] + k[874]*y[IDX_N2I] +
        k[875]*y[IDX_NHI] + k[876]*y[IDX_NDI] + k[877]*y[IDX_NOI] +
        k[878]*y[IDX_NOI] + k[879]*y[IDX_O2I] + k[880]*y[IDX_OHI] +
        k[881]*y[IDX_ODI] + k[882]*y[IDX_C2HI] + k[883]*y[IDX_C2DI] +
        k[884]*y[IDX_C2HI] + k[885]*y[IDX_C2DI] + k[886]*y[IDX_C2HI] +
        k[887]*y[IDX_C2DI] + k[888]*y[IDX_C2NI] + k[889]*y[IDX_C3I] +
        k[890]*y[IDX_C3I] + k[891]*y[IDX_CCOI] + k[892]*y[IDX_CH2I] +
        k[893]*y[IDX_CD2I] + k[894]*y[IDX_CHDI] + k[895]*y[IDX_CH2I] +
        k[896]*y[IDX_CD2I] + k[897]*y[IDX_CHDI] + k[898]*y[IDX_CHDI] +
        k[899]*y[IDX_CO2I] + k[900]*y[IDX_CO2I] + k[901]*y[IDX_CO2I] +
        k[902]*y[IDX_CO2I] + k[903]*y[IDX_H2OI] + k[904]*y[IDX_D2OI] +
        k[905]*y[IDX_HDOI] + k[906]*y[IDX_HDOI] + k[907]*y[IDX_H2OI] +
        k[908]*y[IDX_D2OI] + k[909]*y[IDX_HDOI] + k[910]*y[IDX_HDOI] +
        k[911]*y[IDX_HCNI] + k[912]*y[IDX_DCNI] + k[913]*y[IDX_HCNI] +
        k[914]*y[IDX_DCNI] + k[915]*y[IDX_HCNI] + k[916]*y[IDX_DCNI] +
        k[917]*y[IDX_HCNI] + k[918]*y[IDX_DCNI] + k[919]*y[IDX_HCOI] +
        k[920]*y[IDX_DCOI] + k[921]*y[IDX_HCOI] + k[922]*y[IDX_DCOI] +
        k[925]*y[IDX_HNCI] + k[926]*y[IDX_DNCI] + k[927]*y[IDX_HNCI] +
        k[928]*y[IDX_DNCI] + k[929]*y[IDX_HNCI] + k[930]*y[IDX_DNCI] +
        k[931]*y[IDX_HNOI] + k[932]*y[IDX_DNOI] + k[933]*y[IDX_HNOI] +
        k[934]*y[IDX_DNOI] + k[935]*y[IDX_N2OI] + k[936]*y[IDX_N2OI] +
        k[937]*y[IDX_N2OI] + k[938]*y[IDX_N2OI] + k[939]*y[IDX_NH2I] +
        k[940]*y[IDX_ND2I] + k[941]*y[IDX_NHDI] + k[942]*y[IDX_NH2I] +
        k[943]*y[IDX_ND2I] + k[944]*y[IDX_NHDI] + k[945]*y[IDX_NHDI] +
        k[946]*y[IDX_OCNI] + k[947]*y[IDX_OCNI] + k[2141]*y[IDX_CM] +
        k[2142]*y[IDX_HM] + k[2143]*y[IDX_DM] + k[2160]*y[IDX_HI] +
        k[2161]*y[IDX_DI] + k[2371]*y[IDX_H2OI] + k[2372]*y[IDX_D2OI] +
        k[2373]*y[IDX_HDOI] + k[2468]*y[IDX_C2I] + k[2469]*y[IDX_CHI] +
        k[2470]*y[IDX_CDI] + k[2471]*y[IDX_pH2I] + k[2472]*y[IDX_oH2I] +
        k[2473]*y[IDX_pD2I] + k[2474]*y[IDX_oD2I] + k[2475]*y[IDX_HDI] +
        k[2476]*y[IDX_N2I] + k[2477]*y[IDX_O2I] + k[2478]*y[IDX_CO2I] +
        k[2848]*y[IDX_eM];
    data[3694] = 0.0 + k[810]*y[IDX_HI] + k[812]*y[IDX_DI] +
        k[814]*y[IDX_pH2I] + k[815]*y[IDX_oH2I] + k[816]*y[IDX_oH2I] +
        k[820]*y[IDX_pD2I] + k[821]*y[IDX_oD2I] + k[822]*y[IDX_oD2I] +
        k[825]*y[IDX_HDI] + k[826]*y[IDX_HDI] + k[2757]*y[IDX_eM] +
        k[2952]*y[IDX_oD2I] + k[2953]*y[IDX_pD2I];
    data[3695] = 0.0 + k[809]*y[IDX_HI] + k[811]*y[IDX_DI] +
        k[813]*y[IDX_oH2I] + k[817]*y[IDX_pD2I] + k[818]*y[IDX_oD2I] +
        k[819]*y[IDX_oD2I] + k[823]*y[IDX_HDI] + k[824]*y[IDX_HDI] +
        k[2756]*y[IDX_eM] + k[2923]*y[IDX_oH2I] + k[2924]*y[IDX_pH2I];
    data[3696] = 0.0 + k[874]*y[IDX_HeII] + k[2476]*y[IDX_HeII];
    data[3697] = 0.0 + k[935]*y[IDX_HeII] + k[936]*y[IDX_HeII] +
        k[937]*y[IDX_HeII] + k[938]*y[IDX_HeII];
    data[3698] = 0.0 + k[876]*y[IDX_HeII];
    data[3699] = 0.0 + k[940]*y[IDX_HeII] + k[943]*y[IDX_HeII];
    data[3700] = 0.0 + k[875]*y[IDX_HeII];
    data[3701] = 0.0 + k[939]*y[IDX_HeII] + k[942]*y[IDX_HeII];
    data[3702] = 0.0 + k[941]*y[IDX_HeII] + k[944]*y[IDX_HeII] +
        k[945]*y[IDX_HeII];
    data[3703] = 0.0 + k[877]*y[IDX_HeII] + k[878]*y[IDX_HeII];
    data[3704] = 0.0 + k[879]*y[IDX_HeII] + k[2477]*y[IDX_HeII];
    data[3705] = 0.0 + k[946]*y[IDX_HeII] + k[947]*y[IDX_HeII];
    data[3706] = 0.0 + k[881]*y[IDX_HeII];
    data[3707] = 0.0 + k[880]*y[IDX_HeII];
    data[3708] = 0.0 + k[2756]*y[IDX_HeHII] + k[2757]*y[IDX_HeDII] +
        k[2848]*y[IDX_HeII];
    data[3709] = 0.0 + k[818]*y[IDX_HeHII] + k[819]*y[IDX_HeHII] +
        k[821]*y[IDX_HeDII] + k[822]*y[IDX_HeDII] + k[871]*y[IDX_HeII] +
        k[2474]*y[IDX_HeII] + k[2952]*y[IDX_HeDII];
    data[3710] = 0.0 + k[813]*y[IDX_HeHII] + k[815]*y[IDX_HeDII] +
        k[816]*y[IDX_HeDII] + k[869]*y[IDX_HeII] + k[2472]*y[IDX_HeII] +
        k[2923]*y[IDX_HeHII];
    data[3711] = 0.0 + k[817]*y[IDX_HeHII] + k[820]*y[IDX_HeDII] +
        k[870]*y[IDX_HeII] + k[2473]*y[IDX_HeII] + k[2953]*y[IDX_HeDII];
    data[3712] = 0.0 + k[814]*y[IDX_HeDII] + k[868]*y[IDX_HeII] +
        k[2471]*y[IDX_HeII] + k[2924]*y[IDX_HeHII];
    data[3713] = 0.0 - k[2141]*y[IDX_HeII];
    data[3714] = 0.0 - k[862]*y[IDX_HeII] - k[2468]*y[IDX_HeII];
    data[3715] = 0.0 - k[883]*y[IDX_HeII] - k[885]*y[IDX_HeII] -
        k[887]*y[IDX_HeII];
    data[3716] = 0.0 - k[882]*y[IDX_HeII] - k[884]*y[IDX_HeII] -
        k[886]*y[IDX_HeII];
    data[3717] = 0.0 - k[888]*y[IDX_HeII];
    data[3718] = 0.0 - k[889]*y[IDX_HeII] - k[890]*y[IDX_HeII];
    data[3719] = 0.0 - k[891]*y[IDX_HeII];
    data[3720] = 0.0 - k[864]*y[IDX_HeII] - k[2470]*y[IDX_HeII];
    data[3721] = 0.0 - k[893]*y[IDX_HeII] - k[896]*y[IDX_HeII];
    data[3722] = 0.0 - k[863]*y[IDX_HeII] - k[2469]*y[IDX_HeII];
    data[3723] = 0.0 - k[892]*y[IDX_HeII] - k[895]*y[IDX_HeII];
    data[3724] = 0.0 - k[894]*y[IDX_HeII] - k[897]*y[IDX_HeII] -
        k[898]*y[IDX_HeII];
    data[3725] = 0.0 - k[865]*y[IDX_HeII] - k[866]*y[IDX_HeII];
    data[3726] = 0.0 - k[867]*y[IDX_HeII];
    data[3727] = 0.0 - k[899]*y[IDX_HeII] - k[900]*y[IDX_HeII] -
        k[901]*y[IDX_HeII] - k[902]*y[IDX_HeII] - k[2478]*y[IDX_HeII];
    data[3728] = 0.0 - k[2161]*y[IDX_HeII];
    data[3729] = 0.0 - k[2143]*y[IDX_HeII];
    data[3730] = 0.0 - k[904]*y[IDX_HeII] - k[908]*y[IDX_HeII] -
        k[2372]*y[IDX_HeII];
    data[3731] = 0.0 - k[912]*y[IDX_HeII] - k[914]*y[IDX_HeII] -
        k[916]*y[IDX_HeII] - k[918]*y[IDX_HeII];
    data[3732] = 0.0 - k[920]*y[IDX_HeII] - k[922]*y[IDX_HeII] -
        k[924]*y[IDX_HeII];
    data[3733] = 0.0 - k[926]*y[IDX_HeII] - k[928]*y[IDX_HeII] -
        k[930]*y[IDX_HeII];
    data[3734] = 0.0 - k[932]*y[IDX_HeII] - k[934]*y[IDX_HeII];
    data[3735] = 0.0 - k[177]*y[IDX_HeII];
    data[3736] = 0.0 - k[2160]*y[IDX_HeII];
    data[3737] = 0.0 - k[2142]*y[IDX_HeII];
    data[3738] = 0.0 - k[903]*y[IDX_HeII] - k[907]*y[IDX_HeII] -
        k[2371]*y[IDX_HeII];
    data[3739] = 0.0 - k[911]*y[IDX_HeII] - k[913]*y[IDX_HeII] -
        k[915]*y[IDX_HeII] - k[917]*y[IDX_HeII];
    data[3740] = 0.0 - k[919]*y[IDX_HeII] - k[921]*y[IDX_HeII] -
        k[923]*y[IDX_HeII];
    data[3741] = 0.0 - k[872]*y[IDX_HeII] - k[873]*y[IDX_HeII] -
        k[2475]*y[IDX_HeII];
    data[3742] = 0.0 - k[905]*y[IDX_HeII] - k[906]*y[IDX_HeII] -
        k[909]*y[IDX_HeII] - k[910]*y[IDX_HeII] - k[2373]*y[IDX_HeII];
    data[3743] = 0.0 - k[925]*y[IDX_HeII] - k[927]*y[IDX_HeII] -
        k[929]*y[IDX_HeII];
    data[3744] = 0.0 - k[931]*y[IDX_HeII] - k[933]*y[IDX_HeII];
    data[3745] = 0.0 + k[205];
    data[3746] = 0.0 - k[177]*y[IDX_GRAINM] - k[862]*y[IDX_C2I] -
        k[863]*y[IDX_CHI] - k[864]*y[IDX_CDI] - k[865]*y[IDX_CNI] -
        k[866]*y[IDX_CNI] - k[867]*y[IDX_COI] - k[868]*y[IDX_pH2I] -
        k[869]*y[IDX_oH2I] - k[870]*y[IDX_pD2I] - k[871]*y[IDX_oD2I] -
        k[872]*y[IDX_HDI] - k[873]*y[IDX_HDI] - k[874]*y[IDX_N2I] -
        k[875]*y[IDX_NHI] - k[876]*y[IDX_NDI] - k[877]*y[IDX_NOI] -
        k[878]*y[IDX_NOI] - k[879]*y[IDX_O2I] - k[880]*y[IDX_OHI] -
        k[881]*y[IDX_ODI] - k[882]*y[IDX_C2HI] - k[883]*y[IDX_C2DI] -
        k[884]*y[IDX_C2HI] - k[885]*y[IDX_C2DI] - k[886]*y[IDX_C2HI] -
        k[887]*y[IDX_C2DI] - k[888]*y[IDX_C2NI] - k[889]*y[IDX_C3I] -
        k[890]*y[IDX_C3I] - k[891]*y[IDX_CCOI] - k[892]*y[IDX_CH2I] -
        k[893]*y[IDX_CD2I] - k[894]*y[IDX_CHDI] - k[895]*y[IDX_CH2I] -
        k[896]*y[IDX_CD2I] - k[897]*y[IDX_CHDI] - k[898]*y[IDX_CHDI] -
        k[899]*y[IDX_CO2I] - k[900]*y[IDX_CO2I] - k[901]*y[IDX_CO2I] -
        k[902]*y[IDX_CO2I] - k[903]*y[IDX_H2OI] - k[904]*y[IDX_D2OI] -
        k[905]*y[IDX_HDOI] - k[906]*y[IDX_HDOI] - k[907]*y[IDX_H2OI] -
        k[908]*y[IDX_D2OI] - k[909]*y[IDX_HDOI] - k[910]*y[IDX_HDOI] -
        k[911]*y[IDX_HCNI] - k[912]*y[IDX_DCNI] - k[913]*y[IDX_HCNI] -
        k[914]*y[IDX_DCNI] - k[915]*y[IDX_HCNI] - k[916]*y[IDX_DCNI] -
        k[917]*y[IDX_HCNI] - k[918]*y[IDX_DCNI] - k[919]*y[IDX_HCOI] -
        k[920]*y[IDX_DCOI] - k[921]*y[IDX_HCOI] - k[922]*y[IDX_DCOI] -
        k[923]*y[IDX_HCOI] - k[924]*y[IDX_DCOI] - k[925]*y[IDX_HNCI] -
        k[926]*y[IDX_DNCI] - k[927]*y[IDX_HNCI] - k[928]*y[IDX_DNCI] -
        k[929]*y[IDX_HNCI] - k[930]*y[IDX_DNCI] - k[931]*y[IDX_HNOI] -
        k[932]*y[IDX_DNOI] - k[933]*y[IDX_HNOI] - k[934]*y[IDX_DNOI] -
        k[935]*y[IDX_N2OI] - k[936]*y[IDX_N2OI] - k[937]*y[IDX_N2OI] -
        k[938]*y[IDX_N2OI] - k[939]*y[IDX_NH2I] - k[940]*y[IDX_ND2I] -
        k[941]*y[IDX_NHDI] - k[942]*y[IDX_NH2I] - k[943]*y[IDX_ND2I] -
        k[944]*y[IDX_NHDI] - k[945]*y[IDX_NHDI] - k[946]*y[IDX_OCNI] -
        k[947]*y[IDX_OCNI] - k[2141]*y[IDX_CM] - k[2142]*y[IDX_HM] -
        k[2143]*y[IDX_DM] - k[2160]*y[IDX_HI] - k[2161]*y[IDX_DI] -
        k[2371]*y[IDX_H2OI] - k[2372]*y[IDX_D2OI] - k[2373]*y[IDX_HDOI] -
        k[2468]*y[IDX_C2I] - k[2469]*y[IDX_CHI] - k[2470]*y[IDX_CDI] -
        k[2471]*y[IDX_pH2I] - k[2472]*y[IDX_oH2I] - k[2473]*y[IDX_pD2I] -
        k[2474]*y[IDX_oD2I] - k[2475]*y[IDX_HDI] - k[2476]*y[IDX_N2I] -
        k[2477]*y[IDX_O2I] - k[2478]*y[IDX_CO2I] - k[2848]*y[IDX_eM];
    data[3747] = 0.0 - k[874]*y[IDX_HeII] - k[2476]*y[IDX_HeII];
    data[3748] = 0.0 - k[935]*y[IDX_HeII] - k[936]*y[IDX_HeII] -
        k[937]*y[IDX_HeII] - k[938]*y[IDX_HeII];
    data[3749] = 0.0 - k[876]*y[IDX_HeII];
    data[3750] = 0.0 - k[940]*y[IDX_HeII] - k[943]*y[IDX_HeII];
    data[3751] = 0.0 - k[875]*y[IDX_HeII];
    data[3752] = 0.0 - k[939]*y[IDX_HeII] - k[942]*y[IDX_HeII];
    data[3753] = 0.0 - k[941]*y[IDX_HeII] - k[944]*y[IDX_HeII] -
        k[945]*y[IDX_HeII];
    data[3754] = 0.0 - k[877]*y[IDX_HeII] - k[878]*y[IDX_HeII];
    data[3755] = 0.0 - k[879]*y[IDX_HeII] - k[2477]*y[IDX_HeII];
    data[3756] = 0.0 - k[946]*y[IDX_HeII] - k[947]*y[IDX_HeII];
    data[3757] = 0.0 - k[881]*y[IDX_HeII];
    data[3758] = 0.0 - k[880]*y[IDX_HeII];
    data[3759] = 0.0 - k[2848]*y[IDX_HeII];
    data[3760] = 0.0 - k[871]*y[IDX_HeII] - k[2474]*y[IDX_HeII];
    data[3761] = 0.0 - k[869]*y[IDX_HeII] - k[2472]*y[IDX_HeII];
    data[3762] = 0.0 - k[870]*y[IDX_HeII] - k[2473]*y[IDX_HeII];
    data[3763] = 0.0 - k[868]*y[IDX_HeII] - k[2471]*y[IDX_HeII];
    data[3764] = 0.0 - k[812]*y[IDX_HeDII];
    data[3765] = 0.0 + k[924]*y[IDX_HeII];
    data[3766] = 0.0 - k[810]*y[IDX_HeDII];
    data[3767] = 0.0 - k[825]*y[IDX_HeDII] - k[826]*y[IDX_HeDII];
    data[3768] = 0.0 + k[924]*y[IDX_DCOI];
    data[3769] = 0.0 - k[810]*y[IDX_HI] - k[812]*y[IDX_DI] -
        k[814]*y[IDX_pH2I] - k[815]*y[IDX_oH2I] - k[816]*y[IDX_oH2I] -
        k[820]*y[IDX_pD2I] - k[821]*y[IDX_oD2I] - k[822]*y[IDX_oD2I] -
        k[825]*y[IDX_HDI] - k[826]*y[IDX_HDI] - k[2757]*y[IDX_eM] -
        k[2952]*y[IDX_oD2I] - k[2953]*y[IDX_pD2I];
    data[3770] = 0.0 - k[2757]*y[IDX_HeDII];
    data[3771] = 0.0 - k[821]*y[IDX_HeDII] - k[822]*y[IDX_HeDII] -
        k[2952]*y[IDX_HeDII];
    data[3772] = 0.0 - k[815]*y[IDX_HeDII] - k[816]*y[IDX_HeDII];
    data[3773] = 0.0 - k[820]*y[IDX_HeDII] - k[2953]*y[IDX_HeDII];
    data[3774] = 0.0 - k[814]*y[IDX_HeDII];
    data[3775] = 0.0 - k[811]*y[IDX_HeHII];
    data[3776] = 0.0 - k[809]*y[IDX_HeHII];
    data[3777] = 0.0 + k[923]*y[IDX_HeII];
    data[3778] = 0.0 - k[823]*y[IDX_HeHII] - k[824]*y[IDX_HeHII];
    data[3779] = 0.0 + k[923]*y[IDX_HCOI];
    data[3780] = 0.0 - k[809]*y[IDX_HI] - k[811]*y[IDX_DI] -
        k[813]*y[IDX_oH2I] - k[817]*y[IDX_pD2I] - k[818]*y[IDX_oD2I] -
        k[819]*y[IDX_oD2I] - k[823]*y[IDX_HDI] - k[824]*y[IDX_HDI] -
        k[2756]*y[IDX_eM] - k[2923]*y[IDX_oH2I] - k[2924]*y[IDX_pH2I];
    data[3781] = 0.0 - k[2756]*y[IDX_HeHII];
    data[3782] = 0.0 - k[818]*y[IDX_HeHII] - k[819]*y[IDX_HeHII];
    data[3783] = 0.0 - k[813]*y[IDX_HeHII] - k[2923]*y[IDX_HeHII];
    data[3784] = 0.0 - k[817]*y[IDX_HeHII];
    data[3785] = 0.0 - k[2924]*y[IDX_HeHII];
    data[3786] = 0.0 + k[1053]*y[IDX_CNI] + k[1054]*y[IDX_N2I] +
        k[1818]*y[IDX_NHII] + k[1819]*y[IDX_NDII] + k[2051]*y[IDX_NOI] -
        k[2655]*y[IDX_NI];
    data[3787] = 0.0 + k[2144]*y[IDX_NII] - k[2673]*y[IDX_NI];
    data[3788] = 0.0 - k[1196]*y[IDX_NI] + k[1824]*y[IDX_NHII] +
        k[1825]*y[IDX_NDII] + k[2223]*y[IDX_NII];
    data[3789] = 0.0 - k[1689]*y[IDX_NI] + k[1700]*y[IDX_NHI] +
        k[1701]*y[IDX_NDI];
    data[3790] = 0.0 - k[1222]*y[IDX_NI] + k[2227]*y[IDX_NII];
    data[3791] = 0.0 - k[951]*y[IDX_NI] - k[953]*y[IDX_NI];
    data[3792] = 0.0 - k[1221]*y[IDX_NI] + k[2226]*y[IDX_NII];
    data[3793] = 0.0 - k[950]*y[IDX_NI] - k[952]*y[IDX_NI];
    data[3794] = 0.0 + k[382] - k[1223]*y[IDX_NI];
    data[3795] = 0.0 + k[2772]*y[IDX_eM];
    data[3796] = 0.0 - k[1225]*y[IDX_NI];
    data[3797] = 0.0 - k[1224]*y[IDX_NI];
    data[3798] = 0.0 - k[1198]*y[IDX_NI] + k[1832]*y[IDX_NHII] +
        k[1833]*y[IDX_NDII] + k[2366]*y[IDX_NII];
    data[3799] = 0.0 - k[1718]*y[IDX_NI];
    data[3800] = 0.0 - k[1207]*y[IDX_NI] - k[1211]*y[IDX_NI] +
        k[2229]*y[IDX_NII];
    data[3801] = 0.0 - k[969]*y[IDX_NI];
    data[3802] = 0.0 - k[1197]*y[IDX_NI] + k[1830]*y[IDX_NHII] +
        k[1831]*y[IDX_NDII] + k[2365]*y[IDX_NII];
    data[3803] = 0.0 - k[1717]*y[IDX_NI];
    data[3804] = 0.0 - k[1206]*y[IDX_NI] - k[1210]*y[IDX_NI] +
        k[2228]*y[IDX_NII];
    data[3805] = 0.0 - k[968]*y[IDX_NI];
    data[3806] = 0.0 - k[1208]*y[IDX_NI] - k[1209]*y[IDX_NI] -
        k[1212]*y[IDX_NI] - k[1213]*y[IDX_NI] + k[2230]*y[IDX_NII];
    data[3807] = 0.0 - k[970]*y[IDX_NI] - k[971]*y[IDX_NI];
    data[3808] = 0.0 + k[284] + k[358] + k[865]*y[IDX_HeII] +
        k[1053]*y[IDX_CI] - k[1199]*y[IDX_NI] + k[1229]*y[IDX_OI] +
        k[1834]*y[IDX_NHII] + k[1835]*y[IDX_NDII] + k[2224]*y[IDX_NII];
    data[3809] = 0.0 + k[594]*y[IDX_NOI] + k[2743]*y[IDX_eM];
    data[3810] = 0.0 + k[2786]*y[IDX_eM];
    data[3811] = 0.0 + k[1836]*y[IDX_NHII] + k[1837]*y[IDX_NDII] +
        k[2225]*y[IDX_NII];
    data[3812] = 0.0 - k[609]*y[IDX_NI] + k[612]*y[IDX_NHI] +
        k[613]*y[IDX_NDI];
    data[3813] = 0.0 + k[2521]*y[IDX_NII];
    data[3814] = 0.0 + k[1099]*y[IDX_NOI];
    data[3815] = 0.0 + k[2146]*y[IDX_NII] - k[2694]*y[IDX_NI];
    data[3816] = 0.0 + k[2232]*y[IDX_NII] + k[3314]*y[IDX_NHII] +
        k[3315]*y[IDX_NDII];
    data[3817] = 0.0 - k[992]*y[IDX_NI] + k[3148]*y[IDX_NHI] +
        k[3151]*y[IDX_NDI];
    data[3818] = 0.0 + k[912]*y[IDX_HeII] + k[916]*y[IDX_HeII] +
        k[1675]*y[IDX_OII] + k[2235]*y[IDX_NII];
    data[3819] = 0.0 - k[1215]*y[IDX_NI] + k[2535]*y[IDX_NII];
    data[3820] = 0.0 + k[926]*y[IDX_HeII];
    data[3821] = 0.0 + k[2919]*y[IDX_NII];
    data[3822] = 0.0 + k[1098]*y[IDX_NOI];
    data[3823] = 0.0 + k[2145]*y[IDX_NII] - k[2693]*y[IDX_NI];
    data[3824] = 0.0 + k[2231]*y[IDX_NII] + k[3014]*y[IDX_NHII] +
        k[3311]*y[IDX_NDII];
    data[3825] = 0.0 - k[991]*y[IDX_NI] + k[2999]*y[IDX_NHI] +
        k[3149]*y[IDX_NDI];
    data[3826] = 0.0 + k[911]*y[IDX_HeII] + k[915]*y[IDX_HeII] +
        k[1674]*y[IDX_OII] + k[2234]*y[IDX_NII];
    data[3827] = 0.0 - k[1214]*y[IDX_NI] + k[2534]*y[IDX_NII];
    data[3828] = 0.0 - k[1178]*y[IDX_NI] - k[1179]*y[IDX_NI] +
        k[1850]*y[IDX_NHII] + k[1851]*y[IDX_NHII] + k[1852]*y[IDX_NDII] +
        k[1853]*y[IDX_NDII];
    data[3829] = 0.0 - k[640]*y[IDX_NI] - k[641]*y[IDX_NI];
    data[3830] = 0.0 + k[2233]*y[IDX_NII] + k[3312]*y[IDX_NHII] +
        k[3313]*y[IDX_NDII];
    data[3831] = 0.0 - k[993]*y[IDX_NI] - k[994]*y[IDX_NI] +
        k[3147]*y[IDX_NHI] + k[3150]*y[IDX_NDI];
    data[3832] = 0.0 + k[925]*y[IDX_HeII];
    data[3833] = 0.0 + k[865]*y[IDX_CNI] + k[874]*y[IDX_N2I] +
        k[878]*y[IDX_NOI] + k[911]*y[IDX_HCNI] + k[912]*y[IDX_DCNI] +
        k[915]*y[IDX_HCNI] + k[916]*y[IDX_DCNI] + k[925]*y[IDX_HNCI] +
        k[926]*y[IDX_DNCI] + k[938]*y[IDX_N2OI];
    data[3834] = 0.0 - k[206] - k[609]*y[IDX_COII] - k[636]*y[IDX_pH2II] -
        k[637]*y[IDX_oH2II] - k[638]*y[IDX_pD2II] - k[639]*y[IDX_oD2II] -
        k[640]*y[IDX_HDII] - k[641]*y[IDX_HDII] - k[950]*y[IDX_C2HII] -
        k[951]*y[IDX_C2DII] - k[952]*y[IDX_C2HII] - k[953]*y[IDX_C2DII] -
        k[968]*y[IDX_CH2II] - k[969]*y[IDX_CD2II] - k[970]*y[IDX_CHDII] -
        k[971]*y[IDX_CHDII] - k[991]*y[IDX_H2OII] - k[992]*y[IDX_D2OII] -
        k[993]*y[IDX_HDOII] - k[994]*y[IDX_HDOII] - k[1174]*y[IDX_pH2I] -
        k[1175]*y[IDX_oH2I] - k[1176]*y[IDX_pD2I] - k[1177]*y[IDX_oD2I] -
        k[1178]*y[IDX_HDI] - k[1179]*y[IDX_HDI] - k[1196]*y[IDX_C2I] -
        k[1197]*y[IDX_CHI] - k[1198]*y[IDX_CDI] - k[1199]*y[IDX_CNI] -
        k[1200]*y[IDX_NHI] - k[1201]*y[IDX_NDI] - k[1202]*y[IDX_NOI] -
        k[1203]*y[IDX_OHI] - k[1204]*y[IDX_ODI] - k[1205]*y[IDX_O2I] -
        k[1206]*y[IDX_CH2I] - k[1207]*y[IDX_CD2I] - k[1208]*y[IDX_CHDI] -
        k[1209]*y[IDX_CHDI] - k[1210]*y[IDX_CH2I] - k[1211]*y[IDX_CD2I] -
        k[1212]*y[IDX_CHDI] - k[1213]*y[IDX_CHDI] - k[1214]*y[IDX_HCOI] -
        k[1215]*y[IDX_DCOI] - k[1216]*y[IDX_NO2I] - k[1217]*y[IDX_NO2I] -
        k[1218]*y[IDX_NO2I] - k[1219]*y[IDX_O2HI] - k[1220]*y[IDX_O2DI] -
        k[1221]*y[IDX_C2HI] - k[1222]*y[IDX_C2DI] - k[1223]*y[IDX_C2NI] -
        k[1224]*y[IDX_CCOI] - k[1225]*y[IDX_C3I] - k[1271]*y[IDX_oH3II] -
        k[1272]*y[IDX_pH3II] - k[1273]*y[IDX_oD3II] - k[1274]*y[IDX_mD3II] -
        k[1275]*y[IDX_oH2DII] - k[1276]*y[IDX_pH2DII] - k[1277]*y[IDX_oH2DII] -
        k[1278]*y[IDX_pH2DII] - k[1279]*y[IDX_oD2HII] - k[1280]*y[IDX_pD2HII] -
        k[1281]*y[IDX_oD2HII] - k[1282]*y[IDX_pD2HII] - k[1689]*y[IDX_C2II] -
        k[1717]*y[IDX_CHII] - k[1718]*y[IDX_CDII] - k[1820]*y[IDX_NHII] -
        k[1821]*y[IDX_NDII] - k[1911]*y[IDX_O2II] - k[1923]*y[IDX_OHII] -
        k[1924]*y[IDX_ODII] - k[1969]*y[IDX_NH2II] - k[1970]*y[IDX_ND2II] -
        k[1971]*y[IDX_NHDII] - k[1972]*y[IDX_NHDII] - k[2004]*y[IDX_O2HII] -
        k[2005]*y[IDX_O2DII] - k[2131]*y[IDX_N2II] - k[2655]*y[IDX_CI] -
        k[2673]*y[IDX_CM] - k[2693]*y[IDX_HM] - k[2694]*y[IDX_DM] -
        k[2718]*y[IDX_OM] - k[2958]*y[IDX_pD3II];
    data[3835] = 0.0 + k[2144]*y[IDX_CM] + k[2145]*y[IDX_HM] +
        k[2146]*y[IDX_DM] + k[2223]*y[IDX_C2I] + k[2224]*y[IDX_CNI] +
        k[2225]*y[IDX_COI] + k[2226]*y[IDX_C2HI] + k[2227]*y[IDX_C2DI] +
        k[2228]*y[IDX_CH2I] + k[2229]*y[IDX_CD2I] + k[2230]*y[IDX_CHDI] +
        k[2231]*y[IDX_H2OI] + k[2232]*y[IDX_D2OI] + k[2233]*y[IDX_HDOI] +
        k[2234]*y[IDX_HCNI] + k[2235]*y[IDX_DCNI] + k[2236]*y[IDX_NH2I] +
        k[2237]*y[IDX_ND2I] + k[2238]*y[IDX_NHDI] + k[2365]*y[IDX_CHI] +
        k[2366]*y[IDX_CDI] + k[2367]*y[IDX_OHI] + k[2368]*y[IDX_ODI] +
        k[2509]*y[IDX_NHI] + k[2510]*y[IDX_NDI] + k[2520]*y[IDX_NOI] +
        k[2521]*y[IDX_CO2I] + k[2533]*y[IDX_O2I] + k[2534]*y[IDX_HCOI] +
        k[2535]*y[IDX_DCOI] + k[2849]*y[IDX_eM] + k[2919]*y[IDX_GRAINM];
    data[3836] = 0.0 + k[234] + k[234] + k[365] + k[365] +
        k[874]*y[IDX_HeII] + k[1054]*y[IDX_CI] + k[1662]*y[IDX_OII] +
        k[1870]*y[IDX_NHII] + k[1871]*y[IDX_NDII];
    data[3837] = 0.0 + k[1805]*y[IDX_OI] - k[2131]*y[IDX_NI] +
        k[2758]*y[IDX_eM] + k[2758]*y[IDX_eM];
    data[3838] = 0.0 + k[2833]*y[IDX_eM];
    data[3839] = 0.0 + k[2832]*y[IDX_eM];
    data[3840] = 0.0 + k[267] + k[938]*y[IDX_HeII];
    data[3841] = 0.0 + k[2834]*y[IDX_eM];
    data[3842] = 0.0 + k[236] + k[367] + k[613]*y[IDX_COII] -
        k[1201]*y[IDX_NI] + k[1701]*y[IDX_C2II] + k[1874]*y[IDX_NHII] +
        k[1875]*y[IDX_NDII] + k[2510]*y[IDX_NII] + k[3149]*y[IDX_H2OII] +
        k[3150]*y[IDX_HDOII] + k[3151]*y[IDX_D2OII];
    data[3843] = 0.0 + k[1819]*y[IDX_CI] - k[1821]*y[IDX_NI] +
        k[1823]*y[IDX_OI] + k[1825]*y[IDX_C2I] + k[1831]*y[IDX_CHI] +
        k[1833]*y[IDX_CDI] + k[1835]*y[IDX_CNI] + k[1837]*y[IDX_COI] +
        k[1841]*y[IDX_pH2I] + k[1842]*y[IDX_oH2I] + k[1843]*y[IDX_oH2I] +
        k[1847]*y[IDX_pD2I] + k[1848]*y[IDX_oD2I] + k[1849]*y[IDX_oD2I] +
        k[1852]*y[IDX_HDI] + k[1853]*y[IDX_HDI] + k[1871]*y[IDX_N2I] +
        k[1873]*y[IDX_NHI] + k[1875]*y[IDX_NDI] + k[1881]*y[IDX_O2I] +
        k[1883]*y[IDX_OHI] + k[1885]*y[IDX_ODI] + k[2760]*y[IDX_eM] +
        k[2954]*y[IDX_oD2I] + k[2955]*y[IDX_pD2I] + k[3311]*y[IDX_H2OI] +
        k[3313]*y[IDX_HDOI] + k[3315]*y[IDX_D2OI];
    data[3844] = 0.0 + k[2237]*y[IDX_NII];
    data[3845] = 0.0 - k[1970]*y[IDX_NI] + k[2836]*y[IDX_eM];
    data[3846] = 0.0 + k[235] + k[366] + k[612]*y[IDX_COII] -
        k[1200]*y[IDX_NI] + k[1700]*y[IDX_C2II] + k[1872]*y[IDX_NHII] +
        k[1873]*y[IDX_NDII] + k[2509]*y[IDX_NII] + k[2999]*y[IDX_H2OII] +
        k[3147]*y[IDX_HDOII] + k[3148]*y[IDX_D2OII];
    data[3847] = 0.0 + k[1818]*y[IDX_CI] - k[1820]*y[IDX_NI] +
        k[1822]*y[IDX_OI] + k[1824]*y[IDX_C2I] + k[1830]*y[IDX_CHI] +
        k[1832]*y[IDX_CDI] + k[1834]*y[IDX_CNI] + k[1836]*y[IDX_COI] +
        k[1840]*y[IDX_oH2I] + k[1844]*y[IDX_pD2I] + k[1845]*y[IDX_oD2I] +
        k[1846]*y[IDX_oD2I] + k[1850]*y[IDX_HDI] + k[1851]*y[IDX_HDI] +
        k[1870]*y[IDX_N2I] + k[1872]*y[IDX_NHI] + k[1874]*y[IDX_NDI] +
        k[1880]*y[IDX_O2I] + k[1882]*y[IDX_OHI] + k[1884]*y[IDX_ODI] +
        k[2759]*y[IDX_eM] + k[2925]*y[IDX_oH2I] + k[2926]*y[IDX_pH2I] +
        k[3014]*y[IDX_H2OI] + k[3312]*y[IDX_HDOI] + k[3314]*y[IDX_D2OI];
    data[3848] = 0.0 + k[2236]*y[IDX_NII];
    data[3849] = 0.0 - k[1969]*y[IDX_NI] + k[2835]*y[IDX_eM];
    data[3850] = 0.0 + k[2238]*y[IDX_NII];
    data[3851] = 0.0 - k[1971]*y[IDX_NI] - k[1972]*y[IDX_NI] +
        k[2837]*y[IDX_eM];
    data[3852] = 0.0 + k[237] + k[370] + k[594]*y[IDX_CNII] +
        k[878]*y[IDX_HeII] + k[1098]*y[IDX_HI] + k[1099]*y[IDX_DI] -
        k[1202]*y[IDX_NI] + k[2051]*y[IDX_CI] + k[2520]*y[IDX_NII];
    data[3853] = 0.0 + k[2761]*y[IDX_eM];
    data[3854] = 0.0 - k[1216]*y[IDX_NI] - k[1217]*y[IDX_NI] -
        k[1218]*y[IDX_NI];
    data[3855] = 0.0 + k[1229]*y[IDX_CNI] + k[1805]*y[IDX_N2II] +
        k[1822]*y[IDX_NHII] + k[1823]*y[IDX_NDII];
    data[3856] = 0.0 + k[1662]*y[IDX_N2I] + k[1674]*y[IDX_HCNI] +
        k[1675]*y[IDX_DCNI];
    data[3857] = 0.0 - k[2718]*y[IDX_NI];
    data[3858] = 0.0 - k[1205]*y[IDX_NI] + k[1880]*y[IDX_NHII] +
        k[1881]*y[IDX_NDII] + k[2533]*y[IDX_NII];
    data[3859] = 0.0 - k[1911]*y[IDX_NI];
    data[3860] = 0.0 - k[1220]*y[IDX_NI];
    data[3861] = 0.0 - k[2005]*y[IDX_NI];
    data[3862] = 0.0 - k[1219]*y[IDX_NI];
    data[3863] = 0.0 - k[2004]*y[IDX_NI];
    data[3864] = 0.0 - k[1204]*y[IDX_NI] + k[1884]*y[IDX_NHII] +
        k[1885]*y[IDX_NDII] + k[2368]*y[IDX_NII];
    data[3865] = 0.0 - k[1924]*y[IDX_NI];
    data[3866] = 0.0 - k[1203]*y[IDX_NI] + k[1882]*y[IDX_NHII] +
        k[1883]*y[IDX_NDII] + k[2367]*y[IDX_NII];
    data[3867] = 0.0 - k[1923]*y[IDX_NI];
    data[3868] = 0.0 + k[2743]*y[IDX_CNII] + k[2758]*y[IDX_N2II] +
        k[2758]*y[IDX_N2II] + k[2759]*y[IDX_NHII] + k[2760]*y[IDX_NDII] +
        k[2761]*y[IDX_NOII] + k[2772]*y[IDX_C2NII] + k[2786]*y[IDX_CNCII] +
        k[2832]*y[IDX_N2HII] + k[2833]*y[IDX_N2DII] + k[2834]*y[IDX_NCOII] +
        k[2835]*y[IDX_NH2II] + k[2836]*y[IDX_ND2II] + k[2837]*y[IDX_NHDII] +
        k[2849]*y[IDX_NII];
    data[3869] = 0.0 - k[1274]*y[IDX_NI];
    data[3870] = 0.0 - k[1177]*y[IDX_NI] + k[1845]*y[IDX_NHII] +
        k[1846]*y[IDX_NHII] + k[1848]*y[IDX_NDII] + k[1849]*y[IDX_NDII] +
        k[2954]*y[IDX_NDII];
    data[3871] = 0.0 - k[639]*y[IDX_NI];
    data[3872] = 0.0 - k[1279]*y[IDX_NI] - k[1281]*y[IDX_NI];
    data[3873] = 0.0 - k[1273]*y[IDX_NI];
    data[3874] = 0.0 - k[1175]*y[IDX_NI] + k[1840]*y[IDX_NHII] +
        k[1842]*y[IDX_NDII] + k[1843]*y[IDX_NDII] + k[2925]*y[IDX_NHII];
    data[3875] = 0.0 - k[637]*y[IDX_NI];
    data[3876] = 0.0 - k[1275]*y[IDX_NI] - k[1277]*y[IDX_NI];
    data[3877] = 0.0 - k[1271]*y[IDX_NI];
    data[3878] = 0.0 - k[1176]*y[IDX_NI] + k[1844]*y[IDX_NHII] +
        k[1847]*y[IDX_NDII] + k[2955]*y[IDX_NDII];
    data[3879] = 0.0 - k[638]*y[IDX_NI];
    data[3880] = 0.0 - k[1280]*y[IDX_NI] - k[1282]*y[IDX_NI];
    data[3881] = 0.0 - k[2958]*y[IDX_NI];
    data[3882] = 0.0 - k[1174]*y[IDX_NI] + k[1841]*y[IDX_NDII] +
        k[2926]*y[IDX_NHII];
    data[3883] = 0.0 - k[636]*y[IDX_NI];
    data[3884] = 0.0 - k[1276]*y[IDX_NI] - k[1278]*y[IDX_NI];
    data[3885] = 0.0 - k[1272]*y[IDX_NI];
    data[3886] = 0.0 - k[2144]*y[IDX_NII];
    data[3887] = 0.0 - k[2223]*y[IDX_NII];
    data[3888] = 0.0 - k[2227]*y[IDX_NII];
    data[3889] = 0.0 - k[2226]*y[IDX_NII];
    data[3890] = 0.0 - k[1635]*y[IDX_NII] - k[2366]*y[IDX_NII];
    data[3891] = 0.0 - k[2229]*y[IDX_NII];
    data[3892] = 0.0 - k[1634]*y[IDX_NII] - k[2365]*y[IDX_NII];
    data[3893] = 0.0 - k[2228]*y[IDX_NII];
    data[3894] = 0.0 - k[2230]*y[IDX_NII];
    data[3895] = 0.0 + k[866]*y[IDX_HeII] - k[2224]*y[IDX_NII];
    data[3896] = 0.0 - k[2064]*y[IDX_NII] - k[2225]*y[IDX_NII];
    data[3897] = 0.0 - k[1649]*y[IDX_NII] - k[2521]*y[IDX_NII];
    data[3898] = 0.0 - k[2146]*y[IDX_NII];
    data[3899] = 0.0 - k[2232]*y[IDX_NII];
    data[3900] = 0.0 + k[914]*y[IDX_HeII] - k[2235]*y[IDX_NII];
    data[3901] = 0.0 - k[1651]*y[IDX_NII] - k[2535]*y[IDX_NII];
    data[3902] = 0.0 - k[2919]*y[IDX_NII];
    data[3903] = 0.0 - k[2145]*y[IDX_NII];
    data[3904] = 0.0 - k[2231]*y[IDX_NII];
    data[3905] = 0.0 + k[913]*y[IDX_HeII] - k[2234]*y[IDX_NII];
    data[3906] = 0.0 - k[1650]*y[IDX_NII] - k[2534]*y[IDX_NII];
    data[3907] = 0.0 - k[1640]*y[IDX_NII] - k[1641]*y[IDX_NII];
    data[3908] = 0.0 - k[2233]*y[IDX_NII];
    data[3909] = 0.0 + k[866]*y[IDX_CNI] + k[874]*y[IDX_N2I] +
        k[875]*y[IDX_NHI] + k[876]*y[IDX_NDI] + k[877]*y[IDX_NOI] +
        k[913]*y[IDX_HCNI] + k[914]*y[IDX_DCNI] + k[935]*y[IDX_N2OI] +
        k[939]*y[IDX_NH2I] + k[940]*y[IDX_ND2I] + k[941]*y[IDX_NHDI];
    data[3910] = 0.0 + k[206] + k[2131]*y[IDX_N2II];
    data[3911] = 0.0 - k[1634]*y[IDX_CHI] - k[1635]*y[IDX_CDI] -
        k[1636]*y[IDX_pH2I] - k[1637]*y[IDX_oH2I] - k[1638]*y[IDX_pD2I] -
        k[1639]*y[IDX_oD2I] - k[1640]*y[IDX_HDI] - k[1641]*y[IDX_HDI] -
        k[1642]*y[IDX_NHI] - k[1643]*y[IDX_NDI] - k[1644]*y[IDX_NOI] -
        k[1645]*y[IDX_O2I] - k[1646]*y[IDX_O2I] - k[1647]*y[IDX_OHI] -
        k[1648]*y[IDX_ODI] - k[1649]*y[IDX_CO2I] - k[1650]*y[IDX_HCOI] -
        k[1651]*y[IDX_DCOI] - k[2064]*y[IDX_COI] - k[2144]*y[IDX_CM] -
        k[2145]*y[IDX_HM] - k[2146]*y[IDX_DM] - k[2223]*y[IDX_C2I] -
        k[2224]*y[IDX_CNI] - k[2225]*y[IDX_COI] - k[2226]*y[IDX_C2HI] -
        k[2227]*y[IDX_C2DI] - k[2228]*y[IDX_CH2I] - k[2229]*y[IDX_CD2I] -
        k[2230]*y[IDX_CHDI] - k[2231]*y[IDX_H2OI] - k[2232]*y[IDX_D2OI] -
        k[2233]*y[IDX_HDOI] - k[2234]*y[IDX_HCNI] - k[2235]*y[IDX_DCNI] -
        k[2236]*y[IDX_NH2I] - k[2237]*y[IDX_ND2I] - k[2238]*y[IDX_NHDI] -
        k[2365]*y[IDX_CHI] - k[2366]*y[IDX_CDI] - k[2367]*y[IDX_OHI] -
        k[2368]*y[IDX_ODI] - k[2509]*y[IDX_NHI] - k[2510]*y[IDX_NDI] -
        k[2520]*y[IDX_NOI] - k[2521]*y[IDX_CO2I] - k[2533]*y[IDX_O2I] -
        k[2534]*y[IDX_HCOI] - k[2535]*y[IDX_DCOI] - k[2849]*y[IDX_eM] -
        k[2919]*y[IDX_GRAINM];
    data[3912] = 0.0 + k[874]*y[IDX_HeII];
    data[3913] = 0.0 + k[2131]*y[IDX_NI];
    data[3914] = 0.0 + k[935]*y[IDX_HeII];
    data[3915] = 0.0 + k[876]*y[IDX_HeII] - k[1643]*y[IDX_NII] -
        k[2510]*y[IDX_NII];
    data[3916] = 0.0 + k[940]*y[IDX_HeII] - k[2237]*y[IDX_NII];
    data[3917] = 0.0 + k[875]*y[IDX_HeII] - k[1642]*y[IDX_NII] -
        k[2509]*y[IDX_NII];
    data[3918] = 0.0 + k[939]*y[IDX_HeII] - k[2236]*y[IDX_NII];
    data[3919] = 0.0 + k[941]*y[IDX_HeII] - k[2238]*y[IDX_NII];
    data[3920] = 0.0 + k[877]*y[IDX_HeII] - k[1644]*y[IDX_NII] -
        k[2520]*y[IDX_NII];
    data[3921] = 0.0 - k[1645]*y[IDX_NII] - k[1646]*y[IDX_NII] -
        k[2533]*y[IDX_NII];
    data[3922] = 0.0 - k[1648]*y[IDX_NII] - k[2368]*y[IDX_NII];
    data[3923] = 0.0 - k[1647]*y[IDX_NII] - k[2367]*y[IDX_NII];
    data[3924] = 0.0 - k[2849]*y[IDX_NII];
    data[3925] = 0.0 - k[1639]*y[IDX_NII];
    data[3926] = 0.0 - k[1637]*y[IDX_NII];
    data[3927] = 0.0 - k[1638]*y[IDX_NII];
    data[3928] = 0.0 - k[1636]*y[IDX_NII];
    data[3929] = 0.0 - k[1054]*y[IDX_N2I] + k[1616]*y[IDX_N2HII] +
        k[1617]*y[IDX_N2DII] + k[2128]*y[IDX_N2II];
    data[3930] = 0.0 + k[1618]*y[IDX_N2HII] + k[1619]*y[IDX_N2DII] +
        k[2271]*y[IDX_N2II];
    data[3931] = 0.0 + k[2283]*y[IDX_N2II];
    data[3932] = 0.0 + k[2282]*y[IDX_N2II];
    data[3933] = 0.0 + k[1622]*y[IDX_N2HII] + k[1623]*y[IDX_N2DII] +
        k[2273]*y[IDX_N2II];
    data[3934] = 0.0 + k[2285]*y[IDX_N2II];
    data[3935] = 0.0 + k[1620]*y[IDX_N2HII] + k[1621]*y[IDX_N2DII] +
        k[2272]*y[IDX_N2II];
    data[3936] = 0.0 + k[2284]*y[IDX_N2II];
    data[3937] = 0.0 + k[2286]*y[IDX_N2II];
    data[3938] = 0.0 + k[547]*y[IDX_NOI] + k[1199]*y[IDX_NI] +
        k[2274]*y[IDX_N2II];
    data[3939] = 0.0 + k[1624]*y[IDX_N2HII] + k[1625]*y[IDX_N2DII] +
        k[2275]*y[IDX_N2II];
    data[3940] = 0.0 + k[2287]*y[IDX_N2II];
    data[3941] = 0.0 + k[1127]*y[IDX_N2OI] + k[2130]*y[IDX_N2II];
    data[3942] = 0.0 + k[2623]*y[IDX_N2II] + k[3281]*y[IDX_N2HII] +
        k[3282]*y[IDX_N2DII];
    data[3943] = 0.0 + k[2289]*y[IDX_N2II];
    data[3944] = 0.0 + k[2575]*y[IDX_N2II];
    data[3945] = 0.0 - k[1615]*y[IDX_N2I];
    data[3946] = 0.0 + k[1126]*y[IDX_N2OI] + k[2129]*y[IDX_N2II];
    data[3947] = 0.0 + k[2622]*y[IDX_N2II] + k[3009]*y[IDX_N2HII] +
        k[3278]*y[IDX_N2DII];
    data[3948] = 0.0 + k[2288]*y[IDX_N2II];
    data[3949] = 0.0 + k[2574]*y[IDX_N2II];
    data[3950] = 0.0 - k[744]*y[IDX_N2I] - k[745]*y[IDX_N2I];
    data[3951] = 0.0 + k[2624]*y[IDX_N2II] + k[3279]*y[IDX_N2HII] +
        k[3280]*y[IDX_N2DII];
    data[3952] = 0.0 - k[1614]*y[IDX_N2I];
    data[3953] = 0.0 - k[874]*y[IDX_N2I] + k[936]*y[IDX_N2OI] -
        k[2476]*y[IDX_N2I];
    data[3954] = 0.0 + k[1199]*y[IDX_CNI] + k[1200]*y[IDX_NHI] +
        k[1201]*y[IDX_NDI] + k[1202]*y[IDX_NOI] + k[1216]*y[IDX_NO2I] +
        k[2131]*y[IDX_N2II];
    data[3955] = 0.0 - k[234] - k[365] - k[740]*y[IDX_pH2II] -
        k[741]*y[IDX_oH2II] - k[742]*y[IDX_pD2II] - k[743]*y[IDX_oD2II] -
        k[744]*y[IDX_HDII] - k[745]*y[IDX_HDII] - k[874]*y[IDX_HeII] -
        k[1054]*y[IDX_CI] - k[1402]*y[IDX_oH3II] - k[1403]*y[IDX_pH3II] -
        k[1404]*y[IDX_pH3II] - k[1405]*y[IDX_oD3II] - k[1406]*y[IDX_mD3II] -
        k[1407]*y[IDX_oH2DII] - k[1408]*y[IDX_pH2DII] - k[1409]*y[IDX_pH2DII] -
        k[1410]*y[IDX_oH2DII] - k[1411]*y[IDX_pH2DII] - k[1412]*y[IDX_oD2HII] -
        k[1413]*y[IDX_pD2HII] - k[1414]*y[IDX_pD2HII] - k[1415]*y[IDX_oD2HII] -
        k[1416]*y[IDX_pD2HII] - k[1614]*y[IDX_HOCII] - k[1615]*y[IDX_DOCII] -
        k[1662]*y[IDX_OII] - k[1870]*y[IDX_NHII] - k[1871]*y[IDX_NDII] -
        k[1953]*y[IDX_OHII] - k[1954]*y[IDX_ODII] - k[2032]*y[IDX_O2HII] -
        k[2033]*y[IDX_O2DII] - k[2476]*y[IDX_HeII] - k[2913]*y[IDX_pD3II] -
        k[2970]*y[IDX_pD3II];
    data[3956] = 0.0 + k[2128]*y[IDX_CI] + k[2129]*y[IDX_HI] +
        k[2130]*y[IDX_DI] + k[2131]*y[IDX_NI] + k[2271]*y[IDX_C2I] +
        k[2272]*y[IDX_CHI] + k[2273]*y[IDX_CDI] + k[2274]*y[IDX_CNI] +
        k[2275]*y[IDX_COI] + k[2276]*y[IDX_NHI] + k[2277]*y[IDX_NDI] +
        k[2278]*y[IDX_NOI] + k[2279]*y[IDX_O2I] + k[2280]*y[IDX_OHI] +
        k[2281]*y[IDX_ODI] + k[2282]*y[IDX_C2HI] + k[2283]*y[IDX_C2DI] +
        k[2284]*y[IDX_CH2I] + k[2285]*y[IDX_CD2I] + k[2286]*y[IDX_CHDI] +
        k[2287]*y[IDX_CO2I] + k[2288]*y[IDX_HCNI] + k[2289]*y[IDX_DCNI] +
        k[2290]*y[IDX_NH2I] + k[2291]*y[IDX_ND2I] + k[2292]*y[IDX_NHDI] +
        k[2574]*y[IDX_HCOI] + k[2575]*y[IDX_DCOI] + k[2621]*y[IDX_OI] +
        k[2622]*y[IDX_H2OI] + k[2623]*y[IDX_D2OI] + k[2624]*y[IDX_HDOI];
    data[3957] = 0.0 + k[1617]*y[IDX_CI] + k[1619]*y[IDX_C2I] +
        k[1621]*y[IDX_CHI] + k[1623]*y[IDX_CDI] + k[1625]*y[IDX_COI] +
        k[1627]*y[IDX_NHI] + k[1629]*y[IDX_NDI] + k[1631]*y[IDX_OHI] +
        k[1633]*y[IDX_ODI] + k[2831]*y[IDX_eM] + k[3278]*y[IDX_H2OI] +
        k[3280]*y[IDX_HDOI] + k[3282]*y[IDX_D2OI];
    data[3958] = 0.0 + k[1616]*y[IDX_CI] + k[1618]*y[IDX_C2I] +
        k[1620]*y[IDX_CHI] + k[1622]*y[IDX_CDI] + k[1624]*y[IDX_COI] +
        k[1626]*y[IDX_NHI] + k[1628]*y[IDX_NDI] + k[1630]*y[IDX_OHI] +
        k[1632]*y[IDX_ODI] + k[2830]*y[IDX_eM] + k[3009]*y[IDX_H2OI] +
        k[3279]*y[IDX_HDOI] + k[3281]*y[IDX_D2OI];
    data[3959] = 0.0 + k[936]*y[IDX_HeII] + k[1126]*y[IDX_HI] +
        k[1127]*y[IDX_DI];
    data[3960] = 0.0 + k[562]*y[IDX_NHI] + k[563]*y[IDX_NDI] +
        k[563]*y[IDX_NDI] + k[1201]*y[IDX_NI] + k[1628]*y[IDX_N2HII] +
        k[1629]*y[IDX_N2DII] + k[2277]*y[IDX_N2II];
    data[3961] = 0.0 - k[1871]*y[IDX_N2I];
    data[3962] = 0.0 + k[567]*y[IDX_NOI] + k[2291]*y[IDX_N2II];
    data[3963] = 0.0 + k[561]*y[IDX_NHI] + k[561]*y[IDX_NHI] +
        k[562]*y[IDX_NDI] + k[1200]*y[IDX_NI] + k[1626]*y[IDX_N2HII] +
        k[1627]*y[IDX_N2DII] + k[2276]*y[IDX_N2II];
    data[3964] = 0.0 - k[1870]*y[IDX_N2I];
    data[3965] = 0.0 + k[566]*y[IDX_NOI] + k[2290]*y[IDX_N2II];
    data[3966] = 0.0 + k[568]*y[IDX_NOI] + k[2292]*y[IDX_N2II];
    data[3967] = 0.0 + k[547]*y[IDX_CNI] + k[566]*y[IDX_NH2I] +
        k[567]*y[IDX_ND2I] + k[568]*y[IDX_NHDI] + k[1202]*y[IDX_NI] +
        k[2278]*y[IDX_N2II];
    data[3968] = 0.0 + k[1216]*y[IDX_NI];
    data[3969] = 0.0 + k[2621]*y[IDX_N2II];
    data[3970] = 0.0 - k[1662]*y[IDX_N2I];
    data[3971] = 0.0 + k[2279]*y[IDX_N2II];
    data[3972] = 0.0 - k[2033]*y[IDX_N2I];
    data[3973] = 0.0 - k[2032]*y[IDX_N2I];
    data[3974] = 0.0 + k[1632]*y[IDX_N2HII] + k[1633]*y[IDX_N2DII] +
        k[2281]*y[IDX_N2II];
    data[3975] = 0.0 - k[1954]*y[IDX_N2I];
    data[3976] = 0.0 + k[1630]*y[IDX_N2HII] + k[1631]*y[IDX_N2DII] +
        k[2280]*y[IDX_N2II];
    data[3977] = 0.0 - k[1953]*y[IDX_N2I];
    data[3978] = 0.0 + k[2830]*y[IDX_N2HII] + k[2831]*y[IDX_N2DII];
    data[3979] = 0.0 - k[1406]*y[IDX_N2I];
    data[3980] = 0.0 - k[743]*y[IDX_N2I];
    data[3981] = 0.0 - k[1412]*y[IDX_N2I] - k[1415]*y[IDX_N2I];
    data[3982] = 0.0 - k[1405]*y[IDX_N2I];
    data[3983] = 0.0 - k[741]*y[IDX_N2I];
    data[3984] = 0.0 - k[1407]*y[IDX_N2I] - k[1410]*y[IDX_N2I];
    data[3985] = 0.0 - k[1402]*y[IDX_N2I];
    data[3986] = 0.0 - k[742]*y[IDX_N2I];
    data[3987] = 0.0 - k[1413]*y[IDX_N2I] - k[1414]*y[IDX_N2I] -
        k[1416]*y[IDX_N2I];
    data[3988] = 0.0 - k[2913]*y[IDX_N2I] - k[2970]*y[IDX_N2I];
    data[3989] = 0.0 - k[740]*y[IDX_N2I];
    data[3990] = 0.0 - k[1408]*y[IDX_N2I] - k[1409]*y[IDX_N2I] -
        k[1411]*y[IDX_N2I];
    data[3991] = 0.0 - k[1403]*y[IDX_N2I] - k[1404]*y[IDX_N2I];
    data[3992] = 0.0 - k[2128]*y[IDX_N2II];
    data[3993] = 0.0 - k[2271]*y[IDX_N2II];
    data[3994] = 0.0 - k[2283]*y[IDX_N2II];
    data[3995] = 0.0 - k[2282]*y[IDX_N2II];
    data[3996] = 0.0 - k[2273]*y[IDX_N2II];
    data[3997] = 0.0 - k[2285]*y[IDX_N2II];
    data[3998] = 0.0 - k[2272]*y[IDX_N2II];
    data[3999] = 0.0 - k[2284]*y[IDX_N2II];
    data[4000] = 0.0 - k[2286]*y[IDX_N2II];
    data[4001] = 0.0 - k[2274]*y[IDX_N2II];
    data[4002] = 0.0 - k[2275]*y[IDX_N2II];
    data[4003] = 0.0 - k[2287]*y[IDX_N2II];
    data[4004] = 0.0 - k[2130]*y[IDX_N2II];
    data[4005] = 0.0 - k[1813]*y[IDX_N2II] - k[2623]*y[IDX_N2II];
    data[4006] = 0.0 - k[2289]*y[IDX_N2II];
    data[4007] = 0.0 - k[1817]*y[IDX_N2II] - k[2575]*y[IDX_N2II];
    data[4008] = 0.0 - k[2129]*y[IDX_N2II];
    data[4009] = 0.0 - k[1812]*y[IDX_N2II] - k[2622]*y[IDX_N2II];
    data[4010] = 0.0 - k[2288]*y[IDX_N2II];
    data[4011] = 0.0 - k[1816]*y[IDX_N2II] - k[2574]*y[IDX_N2II];
    data[4012] = 0.0 - k[1810]*y[IDX_N2II] - k[1811]*y[IDX_N2II];
    data[4013] = 0.0 - k[1814]*y[IDX_N2II] - k[1815]*y[IDX_N2II] -
        k[2624]*y[IDX_N2II];
    data[4014] = 0.0 + k[937]*y[IDX_N2OI] + k[2476]*y[IDX_N2I];
    data[4015] = 0.0 + k[1820]*y[IDX_NHII] + k[1821]*y[IDX_NDII] -
        k[2131]*y[IDX_N2II];
    data[4016] = 0.0 + k[1642]*y[IDX_NHI] + k[1643]*y[IDX_NDI] +
        k[1644]*y[IDX_NOI];
    data[4017] = 0.0 + k[2476]*y[IDX_HeII];
    data[4018] = 0.0 - k[1805]*y[IDX_OI] - k[1806]*y[IDX_pH2I] -
        k[1807]*y[IDX_oH2I] - k[1808]*y[IDX_pD2I] - k[1809]*y[IDX_oD2I] -
        k[1810]*y[IDX_HDI] - k[1811]*y[IDX_HDI] - k[1812]*y[IDX_H2OI] -
        k[1813]*y[IDX_D2OI] - k[1814]*y[IDX_HDOI] - k[1815]*y[IDX_HDOI] -
        k[1816]*y[IDX_HCOI] - k[1817]*y[IDX_DCOI] - k[2128]*y[IDX_CI] -
        k[2129]*y[IDX_HI] - k[2130]*y[IDX_DI] - k[2131]*y[IDX_NI] -
        k[2271]*y[IDX_C2I] - k[2272]*y[IDX_CHI] - k[2273]*y[IDX_CDI] -
        k[2274]*y[IDX_CNI] - k[2275]*y[IDX_COI] - k[2276]*y[IDX_NHI] -
        k[2277]*y[IDX_NDI] - k[2278]*y[IDX_NOI] - k[2279]*y[IDX_O2I] -
        k[2280]*y[IDX_OHI] - k[2281]*y[IDX_ODI] - k[2282]*y[IDX_C2HI] -
        k[2283]*y[IDX_C2DI] - k[2284]*y[IDX_CH2I] - k[2285]*y[IDX_CD2I] -
        k[2286]*y[IDX_CHDI] - k[2287]*y[IDX_CO2I] - k[2288]*y[IDX_HCNI] -
        k[2289]*y[IDX_DCNI] - k[2290]*y[IDX_NH2I] - k[2291]*y[IDX_ND2I] -
        k[2292]*y[IDX_NHDI] - k[2574]*y[IDX_HCOI] - k[2575]*y[IDX_DCOI] -
        k[2621]*y[IDX_OI] - k[2622]*y[IDX_H2OI] - k[2623]*y[IDX_D2OI] -
        k[2624]*y[IDX_HDOI] - k[2758]*y[IDX_eM];
    data[4019] = 0.0 + k[937]*y[IDX_HeII];
    data[4020] = 0.0 + k[1643]*y[IDX_NII] - k[2277]*y[IDX_N2II];
    data[4021] = 0.0 + k[1821]*y[IDX_NI];
    data[4022] = 0.0 - k[2291]*y[IDX_N2II];
    data[4023] = 0.0 + k[1642]*y[IDX_NII] - k[2276]*y[IDX_N2II];
    data[4024] = 0.0 + k[1820]*y[IDX_NI];
    data[4025] = 0.0 - k[2290]*y[IDX_N2II];
    data[4026] = 0.0 - k[2292]*y[IDX_N2II];
    data[4027] = 0.0 + k[1644]*y[IDX_NII] - k[2278]*y[IDX_N2II];
    data[4028] = 0.0 - k[1805]*y[IDX_N2II] - k[2621]*y[IDX_N2II];
    data[4029] = 0.0 - k[2279]*y[IDX_N2II];
    data[4030] = 0.0 - k[2281]*y[IDX_N2II];
    data[4031] = 0.0 - k[2280]*y[IDX_N2II];
    data[4032] = 0.0 - k[2758]*y[IDX_N2II];
    data[4033] = 0.0 - k[1809]*y[IDX_N2II];
    data[4034] = 0.0 - k[1807]*y[IDX_N2II];
    data[4035] = 0.0 - k[1808]*y[IDX_N2II];
    data[4036] = 0.0 - k[1806]*y[IDX_N2II];
    data[4037] = 0.0 - k[1617]*y[IDX_N2DII];
    data[4038] = 0.0 - k[1619]*y[IDX_N2DII];
    data[4039] = 0.0 - k[1623]*y[IDX_N2DII];
    data[4040] = 0.0 - k[1621]*y[IDX_N2DII];
    data[4041] = 0.0 - k[1625]*y[IDX_N2DII];
    data[4042] = 0.0 + k[1813]*y[IDX_N2II] - k[3282]*y[IDX_N2DII];
    data[4043] = 0.0 + k[1817]*y[IDX_N2II];
    data[4044] = 0.0 + k[1615]*y[IDX_N2I];
    data[4045] = 0.0 - k[3278]*y[IDX_N2DII];
    data[4046] = 0.0 + k[1810]*y[IDX_N2II];
    data[4047] = 0.0 + k[744]*y[IDX_N2I];
    data[4048] = 0.0 + k[1814]*y[IDX_N2II] - k[3280]*y[IDX_N2DII];
    data[4049] = 0.0 + k[1970]*y[IDX_ND2II] + k[1971]*y[IDX_NHDII];
    data[4050] = 0.0 + k[742]*y[IDX_pD2II] + k[743]*y[IDX_oD2II] +
        k[744]*y[IDX_HDII] + k[1405]*y[IDX_oD3II] + k[1406]*y[IDX_mD3II] +
        k[1407]*y[IDX_oH2DII] + k[1408]*y[IDX_pH2DII] + k[1409]*y[IDX_pH2DII] +
        k[1415]*y[IDX_oD2HII] + k[1416]*y[IDX_pD2HII] + k[1615]*y[IDX_DOCII] +
        k[1871]*y[IDX_NDII] + k[1954]*y[IDX_ODII] + k[2033]*y[IDX_O2DII] +
        k[2913]*y[IDX_pD3II] + k[2970]*y[IDX_pD3II];
    data[4051] = 0.0 + k[1808]*y[IDX_pD2I] + k[1809]*y[IDX_oD2I] +
        k[1810]*y[IDX_HDI] + k[1813]*y[IDX_D2OI] + k[1814]*y[IDX_HDOI] +
        k[1817]*y[IDX_DCOI];
    data[4052] = 0.0 - k[1617]*y[IDX_CI] - k[1619]*y[IDX_C2I] -
        k[1621]*y[IDX_CHI] - k[1623]*y[IDX_CDI] - k[1625]*y[IDX_COI] -
        k[1627]*y[IDX_NHI] - k[1629]*y[IDX_NDI] - k[1631]*y[IDX_OHI] -
        k[1633]*y[IDX_ODI] - k[2831]*y[IDX_eM] - k[2833]*y[IDX_eM] -
        k[3278]*y[IDX_H2OI] - k[3280]*y[IDX_HDOI] - k[3282]*y[IDX_D2OI];
    data[4053] = 0.0 - k[1629]*y[IDX_N2DII];
    data[4054] = 0.0 + k[1871]*y[IDX_N2I] + k[1877]*y[IDX_NOI];
    data[4055] = 0.0 + k[1970]*y[IDX_NI];
    data[4056] = 0.0 - k[1627]*y[IDX_N2DII];
    data[4057] = 0.0 + k[1971]*y[IDX_NI];
    data[4058] = 0.0 + k[1877]*y[IDX_NDII];
    data[4059] = 0.0 + k[2033]*y[IDX_N2I];
    data[4060] = 0.0 - k[1633]*y[IDX_N2DII];
    data[4061] = 0.0 + k[1954]*y[IDX_N2I];
    data[4062] = 0.0 - k[1631]*y[IDX_N2DII];
    data[4063] = 0.0 - k[2831]*y[IDX_N2DII] - k[2833]*y[IDX_N2DII];
    data[4064] = 0.0 + k[1406]*y[IDX_N2I];
    data[4065] = 0.0 + k[1809]*y[IDX_N2II];
    data[4066] = 0.0 + k[743]*y[IDX_N2I];
    data[4067] = 0.0 + k[1415]*y[IDX_N2I];
    data[4068] = 0.0 + k[1405]*y[IDX_N2I];
    data[4069] = 0.0 + k[1407]*y[IDX_N2I];
    data[4070] = 0.0 + k[1808]*y[IDX_N2II];
    data[4071] = 0.0 + k[742]*y[IDX_N2I];
    data[4072] = 0.0 + k[1416]*y[IDX_N2I];
    data[4073] = 0.0 + k[2913]*y[IDX_N2I] + k[2970]*y[IDX_N2I];
    data[4074] = 0.0 + k[1408]*y[IDX_N2I] + k[1409]*y[IDX_N2I];
    data[4075] = 0.0 - k[1616]*y[IDX_N2HII];
    data[4076] = 0.0 - k[1618]*y[IDX_N2HII];
    data[4077] = 0.0 - k[1622]*y[IDX_N2HII];
    data[4078] = 0.0 - k[1620]*y[IDX_N2HII];
    data[4079] = 0.0 - k[1624]*y[IDX_N2HII];
    data[4080] = 0.0 - k[3281]*y[IDX_N2HII];
    data[4081] = 0.0 + k[1812]*y[IDX_N2II] - k[3009]*y[IDX_N2HII];
    data[4082] = 0.0 + k[1816]*y[IDX_N2II];
    data[4083] = 0.0 + k[1811]*y[IDX_N2II];
    data[4084] = 0.0 + k[745]*y[IDX_N2I];
    data[4085] = 0.0 + k[1815]*y[IDX_N2II] - k[3279]*y[IDX_N2HII];
    data[4086] = 0.0 + k[1614]*y[IDX_N2I];
    data[4087] = 0.0 + k[1969]*y[IDX_NH2II] + k[1972]*y[IDX_NHDII];
    data[4088] = 0.0 + k[740]*y[IDX_pH2II] + k[741]*y[IDX_oH2II] +
        k[745]*y[IDX_HDII] + k[1402]*y[IDX_oH3II] + k[1403]*y[IDX_pH3II] +
        k[1404]*y[IDX_pH3II] + k[1410]*y[IDX_oH2DII] + k[1411]*y[IDX_pH2DII] +
        k[1412]*y[IDX_oD2HII] + k[1413]*y[IDX_pD2HII] + k[1414]*y[IDX_pD2HII] +
        k[1614]*y[IDX_HOCII] + k[1870]*y[IDX_NHII] + k[1953]*y[IDX_OHII] +
        k[2032]*y[IDX_O2HII];
    data[4089] = 0.0 + k[1806]*y[IDX_pH2I] + k[1807]*y[IDX_oH2I] +
        k[1811]*y[IDX_HDI] + k[1812]*y[IDX_H2OI] + k[1815]*y[IDX_HDOI] +
        k[1816]*y[IDX_HCOI];
    data[4090] = 0.0 - k[1616]*y[IDX_CI] - k[1618]*y[IDX_C2I] -
        k[1620]*y[IDX_CHI] - k[1622]*y[IDX_CDI] - k[1624]*y[IDX_COI] -
        k[1626]*y[IDX_NHI] - k[1628]*y[IDX_NDI] - k[1630]*y[IDX_OHI] -
        k[1632]*y[IDX_ODI] - k[2830]*y[IDX_eM] - k[2832]*y[IDX_eM] -
        k[3009]*y[IDX_H2OI] - k[3279]*y[IDX_HDOI] - k[3281]*y[IDX_D2OI];
    data[4091] = 0.0 - k[1628]*y[IDX_N2HII];
    data[4092] = 0.0 - k[1626]*y[IDX_N2HII];
    data[4093] = 0.0 + k[1870]*y[IDX_N2I] + k[1876]*y[IDX_NOI];
    data[4094] = 0.0 + k[1969]*y[IDX_NI];
    data[4095] = 0.0 + k[1972]*y[IDX_NI];
    data[4096] = 0.0 + k[1876]*y[IDX_NHII];
    data[4097] = 0.0 + k[2032]*y[IDX_N2I];
    data[4098] = 0.0 - k[1632]*y[IDX_N2HII];
    data[4099] = 0.0 - k[1630]*y[IDX_N2HII];
    data[4100] = 0.0 + k[1953]*y[IDX_N2I];
    data[4101] = 0.0 - k[2830]*y[IDX_N2HII] - k[2832]*y[IDX_N2HII];
    data[4102] = 0.0 + k[1412]*y[IDX_N2I];
    data[4103] = 0.0 + k[1807]*y[IDX_N2II];
    data[4104] = 0.0 + k[741]*y[IDX_N2I];
    data[4105] = 0.0 + k[1410]*y[IDX_N2I];
    data[4106] = 0.0 + k[1402]*y[IDX_N2I];
    data[4107] = 0.0 + k[1413]*y[IDX_N2I] + k[1414]*y[IDX_N2I];
    data[4108] = 0.0 + k[1806]*y[IDX_N2II];
    data[4109] = 0.0 + k[740]*y[IDX_N2I];
    data[4110] = 0.0 + k[1411]*y[IDX_N2I];
    data[4111] = 0.0 + k[1403]*y[IDX_N2I] + k[1404]*y[IDX_N2I];
    data[4112] = 0.0 - k[1127]*y[IDX_N2OI];
    data[4113] = 0.0 - k[1126]*y[IDX_N2OI];
    data[4114] = 0.0 - k[935]*y[IDX_N2OI] - k[936]*y[IDX_N2OI] -
        k[937]*y[IDX_N2OI] - k[938]*y[IDX_N2OI];
    data[4115] = 0.0 + k[1218]*y[IDX_NO2I];
    data[4116] = 0.0 - k[267] - k[935]*y[IDX_HeII] - k[936]*y[IDX_HeII] -
        k[937]*y[IDX_HeII] - k[938]*y[IDX_HeII] - k[1126]*y[IDX_HI] -
        k[1127]*y[IDX_DI] - k[1678]*y[IDX_OII];
    data[4117] = 0.0 + k[565]*y[IDX_NOI];
    data[4118] = 0.0 + k[564]*y[IDX_NOI];
    data[4119] = 0.0 + k[564]*y[IDX_NHI] + k[565]*y[IDX_NDI];
    data[4120] = 0.0 + k[1218]*y[IDX_NI];
    data[4121] = 0.0 - k[1678]*y[IDX_N2OI];
    data[4122] = 0.0 + k[594]*y[IDX_NOI] + k[596]*y[IDX_O2I] +
        k[598]*y[IDX_CO2I];
    data[4123] = 0.0 + k[1838]*y[IDX_NHII] + k[1839]*y[IDX_NDII];
    data[4124] = 0.0 + k[598]*y[IDX_CNII];
    data[4125] = 0.0 - k[2834]*y[IDX_eM];
    data[4126] = 0.0 + k[1839]*y[IDX_COI];
    data[4127] = 0.0 + k[1838]*y[IDX_COI];
    data[4128] = 0.0 + k[594]*y[IDX_CNII];
    data[4129] = 0.0 + k[596]*y[IDX_CNII];
    data[4130] = 0.0 - k[2834]*y[IDX_NCOII];
    data[4131] = 0.0 + k[1042]*y[IDX_ND2I] + k[1043]*y[IDX_NHDI] -
        k[2049]*y[IDX_NDI];
    data[4132] = 0.0 - k[422]*y[IDX_NDI];
    data[4133] = 0.0 - k[2683]*y[IDX_NDI];
    data[4134] = 0.0 + k[1978]*y[IDX_ND2II] + k[1980]*y[IDX_NHDII];
    data[4135] = 0.0 - k[1701]*y[IDX_NDI] - k[1703]*y[IDX_NDI];
    data[4136] = 0.0 + k[1987]*y[IDX_NH2II] + k[1988]*y[IDX_ND2II] +
        k[1990]*y[IDX_NHDII];
    data[4137] = 0.0 - k[1750]*y[IDX_NDI];
    data[4138] = 0.0 + k[1983]*y[IDX_ND2II] + k[1985]*y[IDX_NHDII];
    data[4139] = 0.0 - k[1749]*y[IDX_NDI];
    data[4140] = 0.0 + k[604]*y[IDX_D2OI] + k[606]*y[IDX_HDOI] -
        k[2070]*y[IDX_NDI];
    data[4141] = 0.0 + k[560]*y[IDX_DNOI];
    data[4142] = 0.0 - k[613]*y[IDX_NDI] + k[627]*y[IDX_ND2I] +
        k[629]*y[IDX_NHDI] - k[2326]*y[IDX_NDI];
    data[4143] = 0.0 + k[1097]*y[IDX_NOI] + k[1108]*y[IDX_HNOI] +
        k[1109]*y[IDX_DNOI];
    data[4144] = 0.0 - k[2196]*y[IDX_NDI];
    data[4145] = 0.0 + k[2694]*y[IDX_NI] - k[2710]*y[IDX_NDI];
    data[4146] = 0.0 + k[604]*y[IDX_CNII] + k[2628]*y[IDX_NDII] +
        k[3332]*y[IDX_NH2II] + k[3334]*y[IDX_NHDII] + k[3335]*y[IDX_ND2II];
    data[4147] = 0.0 - k[3151]*y[IDX_NDI];
    data[4148] = 0.0 + k[1671]*y[IDX_OII];
    data[4149] = 0.0 - k[1011]*y[IDX_NDI];
    data[4150] = 0.0 - k[1027]*y[IDX_NDI];
    data[4151] = 0.0 - k[1563]*y[IDX_NDI];
    data[4152] = 0.0 + k[560]*y[IDX_COI] + k[1106]*y[IDX_HI] +
        k[1109]*y[IDX_DI];
    data[4153] = 0.0 - k[1585]*y[IDX_NDI];
    data[4154] = 0.0 + k[1106]*y[IDX_DNOI];
    data[4155] = 0.0 - k[2195]*y[IDX_NDI];
    data[4156] = 0.0 - k[2709]*y[IDX_NDI];
    data[4157] = 0.0 + k[2626]*y[IDX_NDII] + k[3322]*y[IDX_NHDII] +
        k[3324]*y[IDX_ND2II];
    data[4158] = 0.0 - k[3149]*y[IDX_NDI];
    data[4159] = 0.0 - k[1010]*y[IDX_NDI];
    data[4160] = 0.0 - k[1026]*y[IDX_NDI];
    data[4161] = 0.0 + k[1178]*y[IDX_NI] - k[1194]*y[IDX_NDI] -
        k[1195]*y[IDX_NDI];
    data[4162] = 0.0 - k[760]*y[IDX_NDI] - k[761]*y[IDX_NDI] -
        k[2397]*y[IDX_NDI];
    data[4163] = 0.0 + k[606]*y[IDX_CNII] + k[2630]*y[IDX_NDII] +
        k[3326]*y[IDX_NH2II] + k[3328]*y[IDX_NHDII] + k[3330]*y[IDX_ND2II];
    data[4164] = 0.0 - k[3150]*y[IDX_NDI];
    data[4165] = 0.0 - k[1562]*y[IDX_NDI];
    data[4166] = 0.0 + k[1108]*y[IDX_DI];
    data[4167] = 0.0 - k[1584]*y[IDX_NDI];
    data[4168] = 0.0 - k[876]*y[IDX_NDI];
    data[4169] = 0.0 + k[1176]*y[IDX_pD2I] + k[1177]*y[IDX_oD2I] +
        k[1178]*y[IDX_HDI] - k[1201]*y[IDX_NDI] + k[1220]*y[IDX_O2DI] +
        k[2694]*y[IDX_DM];
    data[4170] = 0.0 - k[1643]*y[IDX_NDI] - k[2510]*y[IDX_NDI];
    data[4171] = 0.0 - k[2277]*y[IDX_NDI];
    data[4172] = 0.0 - k[1629]*y[IDX_NDI] + k[2833]*y[IDX_eM];
    data[4173] = 0.0 - k[1628]*y[IDX_NDI];
    data[4174] = 0.0 - k[236] - k[367] - k[369] - k[422]*y[IDX_CII] -
        k[562]*y[IDX_NHI] - k[563]*y[IDX_NDI] - k[563]*y[IDX_NDI] -
        k[563]*y[IDX_NDI] - k[563]*y[IDX_NDI] - k[565]*y[IDX_NOI] -
        k[613]*y[IDX_COII] - k[754]*y[IDX_pH2II] - k[755]*y[IDX_oH2II] -
        k[756]*y[IDX_pH2II] - k[757]*y[IDX_oH2II] - k[758]*y[IDX_pD2II] -
        k[759]*y[IDX_oD2II] - k[760]*y[IDX_HDII] - k[761]*y[IDX_HDII] -
        k[876]*y[IDX_HeII] - k[1010]*y[IDX_HCNII] - k[1011]*y[IDX_DCNII] -
        k[1026]*y[IDX_HCOII] - k[1027]*y[IDX_DCOII] - k[1182]*y[IDX_pH2I] -
        k[1183]*y[IDX_oH2I] - k[1184]*y[IDX_pH2I] - k[1185]*y[IDX_oH2I] -
        k[1190]*y[IDX_pD2I] - k[1191]*y[IDX_oD2I] - k[1194]*y[IDX_HDI] -
        k[1195]*y[IDX_HDI] - k[1201]*y[IDX_NI] - k[1231]*y[IDX_OI] -
        k[1432]*y[IDX_oH3II] - k[1433]*y[IDX_pH3II] - k[1434]*y[IDX_pH3II] -
        k[1435]*y[IDX_oH3II] - k[1436]*y[IDX_pH3II] - k[1437]*y[IDX_oD3II] -
        k[1438]*y[IDX_mD3II] - k[1439]*y[IDX_oH2DII] - k[1440]*y[IDX_pH2DII] -
        k[1441]*y[IDX_pH2DII] - k[1442]*y[IDX_oH2DII] - k[1443]*y[IDX_pH2DII] -
        k[1444]*y[IDX_oD2HII] - k[1445]*y[IDX_pD2HII] - k[1446]*y[IDX_pD2HII] -
        k[1447]*y[IDX_oD2HII] - k[1448]*y[IDX_pD2HII] - k[1562]*y[IDX_HNCII] -
        k[1563]*y[IDX_DNCII] - k[1584]*y[IDX_HNOII] - k[1585]*y[IDX_DNOII] -
        k[1628]*y[IDX_N2HII] - k[1629]*y[IDX_N2DII] - k[1643]*y[IDX_NII] -
        k[1664]*y[IDX_OII] - k[1701]*y[IDX_C2II] - k[1703]*y[IDX_C2II] -
        k[1749]*y[IDX_CHII] - k[1750]*y[IDX_CDII] - k[1874]*y[IDX_NHII] -
        k[1875]*y[IDX_NDII] - k[1916]*y[IDX_O2II] - k[1918]*y[IDX_O2II] -
        k[1957]*y[IDX_OHII] - k[1958]*y[IDX_ODII] - k[2036]*y[IDX_O2HII] -
        k[2037]*y[IDX_O2DII] - k[2049]*y[IDX_CI] - k[2070]*y[IDX_CNII] -
        k[2195]*y[IDX_HII] - k[2196]*y[IDX_DII] - k[2277]*y[IDX_N2II] -
        k[2326]*y[IDX_COII] - k[2393]*y[IDX_pH2II] - k[2394]*y[IDX_oH2II] -
        k[2395]*y[IDX_pD2II] - k[2396]*y[IDX_oD2II] - k[2397]*y[IDX_HDII] -
        k[2510]*y[IDX_NII] - k[2567]*y[IDX_OII] - k[2683]*y[IDX_CM] -
        k[2709]*y[IDX_HM] - k[2710]*y[IDX_DM] - k[2971]*y[IDX_pD3II] -
        k[2972]*y[IDX_pD3II] - k[3149]*y[IDX_H2OII] - k[3150]*y[IDX_HDOII] -
        k[3151]*y[IDX_D2OII];
    data[4175] = 0.0 - k[1875]*y[IDX_NDI] + k[2556]*y[IDX_NOI] +
        k[2559]*y[IDX_O2I] + k[2626]*y[IDX_H2OI] + k[2628]*y[IDX_D2OI] +
        k[2630]*y[IDX_HDOI];
    data[4176] = 0.0 + k[269] + k[411] + k[524]*y[IDX_OI] +
        k[582]*y[IDX_OHI] + k[587]*y[IDX_ODI] + k[627]*y[IDX_COII] +
        k[1042]*y[IDX_CI];
    data[4177] = 0.0 + k[1978]*y[IDX_C2I] + k[1983]*y[IDX_CHI] +
        k[1988]*y[IDX_CDI] + k[2839]*y[IDX_eM] + k[3324]*y[IDX_H2OI] +
        k[3330]*y[IDX_HDOI] + k[3335]*y[IDX_D2OI];
    data[4178] = 0.0 - k[562]*y[IDX_NDI];
    data[4179] = 0.0 - k[1874]*y[IDX_NDI];
    data[4180] = 0.0 + k[586]*y[IDX_ODI];
    data[4181] = 0.0 + k[1987]*y[IDX_CDI] + k[3326]*y[IDX_HDOI] +
        k[3332]*y[IDX_D2OI];
    data[4182] = 0.0 + k[270] + k[412] + k[526]*y[IDX_OI] +
        k[584]*y[IDX_OHI] + k[589]*y[IDX_ODI] + k[629]*y[IDX_COII] +
        k[1043]*y[IDX_CI];
    data[4183] = 0.0 + k[1980]*y[IDX_C2I] + k[1985]*y[IDX_CHI] +
        k[1990]*y[IDX_CDI] + k[2840]*y[IDX_eM] + k[3322]*y[IDX_H2OI] +
        k[3328]*y[IDX_HDOI] + k[3334]*y[IDX_D2OI];
    data[4184] = 0.0 - k[565]*y[IDX_NDI] + k[1097]*y[IDX_DI] +
        k[2556]*y[IDX_NDII];
    data[4185] = 0.0 + k[524]*y[IDX_ND2I] + k[526]*y[IDX_NHDI] -
        k[1231]*y[IDX_NDI];
    data[4186] = 0.0 - k[1664]*y[IDX_NDI] + k[1671]*y[IDX_DCNI] -
        k[2567]*y[IDX_NDI];
    data[4187] = 0.0 + k[2559]*y[IDX_NDII];
    data[4188] = 0.0 - k[1916]*y[IDX_NDI] - k[1918]*y[IDX_NDI];
    data[4189] = 0.0 + k[1220]*y[IDX_NI];
    data[4190] = 0.0 - k[2037]*y[IDX_NDI];
    data[4191] = 0.0 - k[2036]*y[IDX_NDI];
    data[4192] = 0.0 + k[586]*y[IDX_NH2I] + k[587]*y[IDX_ND2I] +
        k[589]*y[IDX_NHDI];
    data[4193] = 0.0 - k[1958]*y[IDX_NDI];
    data[4194] = 0.0 + k[582]*y[IDX_ND2I] + k[584]*y[IDX_NHDI];
    data[4195] = 0.0 - k[1957]*y[IDX_NDI];
    data[4196] = 0.0 + k[2833]*y[IDX_N2DII] + k[2839]*y[IDX_ND2II] +
        k[2840]*y[IDX_NHDII];
    data[4197] = 0.0 - k[1438]*y[IDX_NDI];
    data[4198] = 0.0 + k[1177]*y[IDX_NI] - k[1191]*y[IDX_NDI];
    data[4199] = 0.0 - k[759]*y[IDX_NDI] - k[2396]*y[IDX_NDI];
    data[4200] = 0.0 - k[1444]*y[IDX_NDI] - k[1447]*y[IDX_NDI];
    data[4201] = 0.0 - k[1437]*y[IDX_NDI];
    data[4202] = 0.0 - k[1183]*y[IDX_NDI] - k[1185]*y[IDX_NDI];
    data[4203] = 0.0 - k[755]*y[IDX_NDI] - k[757]*y[IDX_NDI] -
        k[2394]*y[IDX_NDI];
    data[4204] = 0.0 - k[1439]*y[IDX_NDI] - k[1442]*y[IDX_NDI];
    data[4205] = 0.0 - k[1432]*y[IDX_NDI] - k[1435]*y[IDX_NDI];
    data[4206] = 0.0 + k[1176]*y[IDX_NI] - k[1190]*y[IDX_NDI];
    data[4207] = 0.0 - k[758]*y[IDX_NDI] - k[2395]*y[IDX_NDI];
    data[4208] = 0.0 - k[1445]*y[IDX_NDI] - k[1446]*y[IDX_NDI] -
        k[1448]*y[IDX_NDI];
    data[4209] = 0.0 - k[2971]*y[IDX_NDI] - k[2972]*y[IDX_NDI];
    data[4210] = 0.0 - k[1182]*y[IDX_NDI] - k[1184]*y[IDX_NDI];
    data[4211] = 0.0 - k[754]*y[IDX_NDI] - k[756]*y[IDX_NDI] -
        k[2393]*y[IDX_NDI];
    data[4212] = 0.0 - k[1440]*y[IDX_NDI] - k[1441]*y[IDX_NDI] -
        k[1443]*y[IDX_NDI];
    data[4213] = 0.0 - k[1433]*y[IDX_NDI] - k[1434]*y[IDX_NDI] -
        k[1436]*y[IDX_NDI];
    data[4214] = 0.0 - k[1819]*y[IDX_NDII];
    data[4215] = 0.0 - k[1825]*y[IDX_NDII] - k[1827]*y[IDX_NDII] -
        k[1829]*y[IDX_NDII];
    data[4216] = 0.0 - k[1833]*y[IDX_NDII];
    data[4217] = 0.0 - k[1831]*y[IDX_NDII];
    data[4218] = 0.0 - k[1835]*y[IDX_NDII];
    data[4219] = 0.0 + k[2070]*y[IDX_NDI];
    data[4220] = 0.0 - k[1837]*y[IDX_NDII] - k[1839]*y[IDX_NDII];
    data[4221] = 0.0 + k[2326]*y[IDX_NDI];
    data[4222] = 0.0 - k[1887]*y[IDX_NDII] - k[1889]*y[IDX_NDII];
    data[4223] = 0.0 + k[2196]*y[IDX_NDI];
    data[4224] = 0.0 - k[1895]*y[IDX_NDII] - k[1905]*y[IDX_NDII] -
        k[2628]*y[IDX_NDII] - k[3315]*y[IDX_NDII];
    data[4225] = 0.0 + k[1651]*y[IDX_NII];
    data[4226] = 0.0 + k[930]*y[IDX_HeII];
    data[4227] = 0.0 + k[2195]*y[IDX_NDI];
    data[4228] = 0.0 - k[1891]*y[IDX_NDII] - k[1892]*y[IDX_NDII] -
        k[1901]*y[IDX_NDII] - k[1902]*y[IDX_NDII] - k[2626]*y[IDX_NDII] -
        k[3311]*y[IDX_NDII];
    data[4229] = 0.0 + k[1640]*y[IDX_NII] - k[1852]*y[IDX_NDII] -
        k[1853]*y[IDX_NDII] - k[1868]*y[IDX_NDII] - k[1869]*y[IDX_NDII];
    data[4230] = 0.0 + k[640]*y[IDX_NI] + k[2397]*y[IDX_NDI];
    data[4231] = 0.0 - k[1898]*y[IDX_NDII] - k[1899]*y[IDX_NDII] -
        k[1908]*y[IDX_NDII] - k[1909]*y[IDX_NDII] - k[2630]*y[IDX_NDII] -
        k[3313]*y[IDX_NDII];
    data[4232] = 0.0 + k[930]*y[IDX_DNCI] + k[943]*y[IDX_ND2I] +
        k[944]*y[IDX_NHDI];
    data[4233] = 0.0 + k[638]*y[IDX_pD2II] + k[639]*y[IDX_oD2II] +
        k[640]*y[IDX_HDII] - k[1821]*y[IDX_NDII];
    data[4234] = 0.0 + k[1638]*y[IDX_pD2I] + k[1639]*y[IDX_oD2I] +
        k[1640]*y[IDX_HDI] + k[1651]*y[IDX_DCOI] + k[2510]*y[IDX_NDI];
    data[4235] = 0.0 - k[1871]*y[IDX_NDII];
    data[4236] = 0.0 + k[2277]*y[IDX_NDI];
    data[4237] = 0.0 + k[369] - k[1875]*y[IDX_NDII] + k[2070]*y[IDX_CNII] +
        k[2195]*y[IDX_HII] + k[2196]*y[IDX_DII] + k[2277]*y[IDX_N2II] +
        k[2326]*y[IDX_COII] + k[2393]*y[IDX_pH2II] + k[2394]*y[IDX_oH2II] +
        k[2395]*y[IDX_pD2II] + k[2396]*y[IDX_oD2II] + k[2397]*y[IDX_HDII] +
        k[2510]*y[IDX_NII] + k[2567]*y[IDX_OII];
    data[4238] = 0.0 - k[1819]*y[IDX_CI] - k[1821]*y[IDX_NI] -
        k[1823]*y[IDX_OI] - k[1825]*y[IDX_C2I] - k[1827]*y[IDX_C2I] -
        k[1829]*y[IDX_C2I] - k[1831]*y[IDX_CHI] - k[1833]*y[IDX_CDI] -
        k[1835]*y[IDX_CNI] - k[1837]*y[IDX_COI] - k[1839]*y[IDX_COI] -
        k[1841]*y[IDX_pH2I] - k[1842]*y[IDX_oH2I] - k[1843]*y[IDX_oH2I] -
        k[1847]*y[IDX_pD2I] - k[1848]*y[IDX_oD2I] - k[1849]*y[IDX_oD2I] -
        k[1852]*y[IDX_HDI] - k[1853]*y[IDX_HDI] - k[1856]*y[IDX_pH2I] -
        k[1857]*y[IDX_oH2I] - k[1858]*y[IDX_pH2I] - k[1859]*y[IDX_oH2I] -
        k[1864]*y[IDX_pD2I] - k[1865]*y[IDX_oD2I] - k[1868]*y[IDX_HDI] -
        k[1869]*y[IDX_HDI] - k[1871]*y[IDX_N2I] - k[1873]*y[IDX_NHI] -
        k[1875]*y[IDX_NDI] - k[1877]*y[IDX_NOI] - k[1879]*y[IDX_O2I] -
        k[1881]*y[IDX_O2I] - k[1883]*y[IDX_OHI] - k[1885]*y[IDX_ODI] -
        k[1887]*y[IDX_CO2I] - k[1889]*y[IDX_CO2I] - k[1891]*y[IDX_H2OI] -
        k[1892]*y[IDX_H2OI] - k[1895]*y[IDX_D2OI] - k[1898]*y[IDX_HDOI] -
        k[1899]*y[IDX_HDOI] - k[1901]*y[IDX_H2OI] - k[1902]*y[IDX_H2OI] -
        k[1905]*y[IDX_D2OI] - k[1908]*y[IDX_HDOI] - k[1909]*y[IDX_HDOI] -
        k[2556]*y[IDX_NOI] - k[2559]*y[IDX_O2I] - k[2626]*y[IDX_H2OI] -
        k[2628]*y[IDX_D2OI] - k[2630]*y[IDX_HDOI] - k[2760]*y[IDX_eM] -
        k[2954]*y[IDX_oD2I] - k[2955]*y[IDX_pD2I] - k[3311]*y[IDX_H2OI] -
        k[3313]*y[IDX_HDOI] - k[3315]*y[IDX_D2OI];
    data[4239] = 0.0 + k[943]*y[IDX_HeII];
    data[4240] = 0.0 - k[1873]*y[IDX_NDII];
    data[4241] = 0.0 + k[944]*y[IDX_HeII];
    data[4242] = 0.0 - k[1877]*y[IDX_NDII] - k[2556]*y[IDX_NDII];
    data[4243] = 0.0 - k[1823]*y[IDX_NDII];
    data[4244] = 0.0 + k[2567]*y[IDX_NDI];
    data[4245] = 0.0 - k[1879]*y[IDX_NDII] - k[1881]*y[IDX_NDII] -
        k[2559]*y[IDX_NDII];
    data[4246] = 0.0 - k[1885]*y[IDX_NDII];
    data[4247] = 0.0 - k[1883]*y[IDX_NDII];
    data[4248] = 0.0 - k[2760]*y[IDX_NDII];
    data[4249] = 0.0 + k[1639]*y[IDX_NII] - k[1848]*y[IDX_NDII] -
        k[1849]*y[IDX_NDII] - k[1865]*y[IDX_NDII] - k[2954]*y[IDX_NDII];
    data[4250] = 0.0 + k[639]*y[IDX_NI] + k[2396]*y[IDX_NDI];
    data[4251] = 0.0 - k[1842]*y[IDX_NDII] - k[1843]*y[IDX_NDII] -
        k[1857]*y[IDX_NDII] - k[1859]*y[IDX_NDII];
    data[4252] = 0.0 + k[2394]*y[IDX_NDI];
    data[4253] = 0.0 + k[1638]*y[IDX_NII] - k[1847]*y[IDX_NDII] -
        k[1864]*y[IDX_NDII] - k[2955]*y[IDX_NDII];
    data[4254] = 0.0 + k[638]*y[IDX_NI] + k[2395]*y[IDX_NDI];
    data[4255] = 0.0 - k[1841]*y[IDX_NDII] - k[1856]*y[IDX_NDII] -
        k[1858]*y[IDX_NDII];
    data[4256] = 0.0 + k[2393]*y[IDX_NDI];
    data[4257] = 0.0 - k[1042]*y[IDX_ND2I] - k[1046]*y[IDX_ND2I] -
        k[1050]*y[IDX_ND2I];
    data[4258] = 0.0 - k[452]*y[IDX_ND2I];
    data[4259] = 0.0 - k[2544]*y[IDX_ND2I];
    data[4260] = 0.0 + k[2612]*y[IDX_ND2II];
    data[4261] = 0.0 - k[1800]*y[IDX_ND2I];
    data[4262] = 0.0 + k[2550]*y[IDX_ND2II];
    data[4263] = 0.0 + k[2609]*y[IDX_ND2II];
    data[4264] = 0.0 - k[1798]*y[IDX_ND2I] - k[1799]*y[IDX_ND2I];
    data[4265] = 0.0 + k[2547]*y[IDX_ND2II];
    data[4266] = 0.0 + k[2553]*y[IDX_ND2II];
    data[4267] = 0.0 - k[2079]*y[IDX_ND2I];
    data[4268] = 0.0 - k[627]*y[IDX_ND2I] - k[2338]*y[IDX_ND2I];
    data[4269] = 0.0 + k[1103]*y[IDX_DNOI];
    data[4270] = 0.0 - k[2256]*y[IDX_ND2I];
    data[4271] = 0.0 + k[2710]*y[IDX_NDI];
    data[4272] = 0.0 - k[2515]*y[IDX_ND2I];
    data[4273] = 0.0 + k[2307]*y[IDX_ND2II];
    data[4274] = 0.0 + k[1103]*y[IDX_DI];
    data[4275] = 0.0 - k[2255]*y[IDX_ND2I];
    data[4276] = 0.0 - k[2514]*y[IDX_ND2I];
    data[4277] = 0.0 + k[2304]*y[IDX_ND2II];
    data[4278] = 0.0 + k[1194]*y[IDX_NDI];
    data[4279] = 0.0 - k[2122]*y[IDX_ND2I];
    data[4280] = 0.0 - k[2516]*y[IDX_ND2I];
    data[4281] = 0.0 - k[940]*y[IDX_ND2I] - k[943]*y[IDX_ND2I];
    data[4282] = 0.0 - k[2237]*y[IDX_ND2I];
    data[4283] = 0.0 - k[2291]*y[IDX_ND2I];
    data[4284] = 0.0 + k[1190]*y[IDX_pD2I] + k[1191]*y[IDX_oD2I] +
        k[1194]*y[IDX_HDI] + k[2710]*y[IDX_DM];
    data[4285] = 0.0 - k[269] - k[273] - k[411] - k[415] - k[452]*y[IDX_CII]
        - k[524]*y[IDX_OI] - k[528]*y[IDX_OI] - k[567]*y[IDX_NOI] -
        k[581]*y[IDX_OHI] - k[582]*y[IDX_OHI] - k[587]*y[IDX_ODI] -
        k[627]*y[IDX_COII] - k[940]*y[IDX_HeII] - k[943]*y[IDX_HeII] -
        k[1042]*y[IDX_CI] - k[1046]*y[IDX_CI] - k[1050]*y[IDX_CI] -
        k[1798]*y[IDX_CHII] - k[1799]*y[IDX_CHII] - k[1800]*y[IDX_CDII] -
        k[2079]*y[IDX_CNII] - k[2118]*y[IDX_pH2II] - k[2119]*y[IDX_oH2II] -
        k[2120]*y[IDX_pD2II] - k[2121]*y[IDX_oD2II] - k[2122]*y[IDX_HDII] -
        k[2237]*y[IDX_NII] - k[2250]*y[IDX_OII] - k[2255]*y[IDX_HII] -
        k[2256]*y[IDX_DII] - k[2291]*y[IDX_N2II] - k[2295]*y[IDX_O2II] -
        k[2338]*y[IDX_COII] - k[2514]*y[IDX_H2OII] - k[2515]*y[IDX_D2OII] -
        k[2516]*y[IDX_HDOII] - k[2544]*y[IDX_C2II] - k[2604]*y[IDX_OHII] -
        k[2605]*y[IDX_ODII];
    data[4286] = 0.0 + k[2301]*y[IDX_NOI] + k[2304]*y[IDX_HCOI] +
        k[2307]*y[IDX_DCOI] + k[2547]*y[IDX_CH2I] + k[2550]*y[IDX_CD2I] +
        k[2553]*y[IDX_CHDI] + k[2609]*y[IDX_CHI] + k[2612]*y[IDX_CDI];
    data[4287] = 0.0 + k[1186]*y[IDX_pD2I] + k[1187]*y[IDX_oD2I];
    data[4288] = 0.0 - k[567]*y[IDX_ND2I] + k[2301]*y[IDX_ND2II];
    data[4289] = 0.0 - k[524]*y[IDX_ND2I] - k[528]*y[IDX_ND2I];
    data[4290] = 0.0 - k[2250]*y[IDX_ND2I];
    data[4291] = 0.0 - k[2295]*y[IDX_ND2I];
    data[4292] = 0.0 - k[587]*y[IDX_ND2I];
    data[4293] = 0.0 - k[2605]*y[IDX_ND2I];
    data[4294] = 0.0 - k[581]*y[IDX_ND2I] - k[582]*y[IDX_ND2I];
    data[4295] = 0.0 - k[2604]*y[IDX_ND2I];
    data[4296] = 0.0 + k[1187]*y[IDX_NHI] + k[1191]*y[IDX_NDI];
    data[4297] = 0.0 - k[2121]*y[IDX_ND2I];
    data[4298] = 0.0 - k[2119]*y[IDX_ND2I];
    data[4299] = 0.0 + k[1186]*y[IDX_NHI] + k[1190]*y[IDX_NDI];
    data[4300] = 0.0 - k[2120]*y[IDX_ND2I];
    data[4301] = 0.0 - k[2118]*y[IDX_ND2I];
    data[4302] = 0.0 - k[1978]*y[IDX_ND2II];
    data[4303] = 0.0 + k[2544]*y[IDX_ND2I];
    data[4304] = 0.0 - k[1988]*y[IDX_ND2II] - k[2612]*y[IDX_ND2II];
    data[4305] = 0.0 - k[2550]*y[IDX_ND2II];
    data[4306] = 0.0 - k[1982]*y[IDX_ND2II] - k[1983]*y[IDX_ND2II] -
        k[2609]*y[IDX_ND2II];
    data[4307] = 0.0 - k[2547]*y[IDX_ND2II];
    data[4308] = 0.0 - k[2553]*y[IDX_ND2II];
    data[4309] = 0.0 + k[2079]*y[IDX_ND2I];
    data[4310] = 0.0 + k[2338]*y[IDX_ND2I];
    data[4311] = 0.0 + k[2256]*y[IDX_ND2I];
    data[4312] = 0.0 + k[1903]*y[IDX_NHII] + k[1905]*y[IDX_NDII] -
        k[3335]*y[IDX_ND2II];
    data[4313] = 0.0 + k[2515]*y[IDX_ND2I];
    data[4314] = 0.0 + k[1011]*y[IDX_NDI];
    data[4315] = 0.0 - k[2307]*y[IDX_ND2II];
    data[4316] = 0.0 + k[1027]*y[IDX_NDI];
    data[4317] = 0.0 + k[1563]*y[IDX_NDI];
    data[4318] = 0.0 + k[1585]*y[IDX_NDI];
    data[4319] = 0.0 + k[2255]*y[IDX_ND2I];
    data[4320] = 0.0 - k[3323]*y[IDX_ND2II] - k[3324]*y[IDX_ND2II];
    data[4321] = 0.0 + k[2514]*y[IDX_ND2I];
    data[4322] = 0.0 - k[2304]*y[IDX_ND2II];
    data[4323] = 0.0 + k[1868]*y[IDX_NDII];
    data[4324] = 0.0 + k[760]*y[IDX_NDI] + k[2122]*y[IDX_ND2I];
    data[4325] = 0.0 + k[1908]*y[IDX_NDII] - k[3329]*y[IDX_ND2II] -
        k[3330]*y[IDX_ND2II];
    data[4326] = 0.0 + k[2516]*y[IDX_ND2I];
    data[4327] = 0.0 + k[1273]*y[IDX_oD3II] + k[1274]*y[IDX_mD3II] +
        k[1279]*y[IDX_oD2HII] + k[1280]*y[IDX_pD2HII] - k[1970]*y[IDX_ND2II] +
        k[2958]*y[IDX_pD3II];
    data[4328] = 0.0 + k[2237]*y[IDX_ND2I];
    data[4329] = 0.0 + k[2291]*y[IDX_ND2I];
    data[4330] = 0.0 + k[1629]*y[IDX_NDI];
    data[4331] = 0.0 + k[758]*y[IDX_pD2II] + k[759]*y[IDX_oD2II] +
        k[760]*y[IDX_HDII] + k[1011]*y[IDX_DCNII] + k[1027]*y[IDX_DCOII] +
        k[1437]*y[IDX_oD3II] + k[1438]*y[IDX_mD3II] + k[1439]*y[IDX_oH2DII] +
        k[1440]*y[IDX_pH2DII] + k[1441]*y[IDX_pH2DII] + k[1447]*y[IDX_oD2HII] +
        k[1448]*y[IDX_pD2HII] + k[1563]*y[IDX_DNCII] + k[1585]*y[IDX_DNOII] +
        k[1629]*y[IDX_N2DII] + k[1875]*y[IDX_NDII] + k[1958]*y[IDX_ODII] +
        k[2037]*y[IDX_O2DII] + k[2971]*y[IDX_pD3II] + k[2972]*y[IDX_pD3II];
    data[4332] = 0.0 + k[1864]*y[IDX_pD2I] + k[1865]*y[IDX_oD2I] +
        k[1868]*y[IDX_HDI] + k[1875]*y[IDX_NDI] + k[1905]*y[IDX_D2OI] +
        k[1908]*y[IDX_HDOI];
    data[4333] = 0.0 + k[273] + k[415] + k[2079]*y[IDX_CNII] +
        k[2118]*y[IDX_pH2II] + k[2119]*y[IDX_oH2II] + k[2120]*y[IDX_pD2II] +
        k[2121]*y[IDX_oD2II] + k[2122]*y[IDX_HDII] + k[2237]*y[IDX_NII] +
        k[2250]*y[IDX_OII] + k[2255]*y[IDX_HII] + k[2256]*y[IDX_DII] +
        k[2291]*y[IDX_N2II] + k[2295]*y[IDX_O2II] + k[2338]*y[IDX_COII] +
        k[2514]*y[IDX_H2OII] + k[2515]*y[IDX_D2OII] + k[2516]*y[IDX_HDOII] +
        k[2544]*y[IDX_C2II] + k[2604]*y[IDX_OHII] + k[2605]*y[IDX_ODII];
    data[4334] = 0.0 - k[1970]*y[IDX_NI] - k[1974]*y[IDX_OI] -
        k[1978]*y[IDX_C2I] - k[1982]*y[IDX_CHI] - k[1983]*y[IDX_CHI] -
        k[1988]*y[IDX_CDI] - k[1992]*y[IDX_O2I] - k[2301]*y[IDX_NOI] -
        k[2304]*y[IDX_HCOI] - k[2307]*y[IDX_DCOI] - k[2547]*y[IDX_CH2I] -
        k[2550]*y[IDX_CD2I] - k[2553]*y[IDX_CHDI] - k[2609]*y[IDX_CHI] -
        k[2612]*y[IDX_CDI] - k[2836]*y[IDX_eM] - k[2839]*y[IDX_eM] -
        k[3323]*y[IDX_H2OI] - k[3324]*y[IDX_H2OI] - k[3329]*y[IDX_HDOI] -
        k[3330]*y[IDX_HDOI] - k[3335]*y[IDX_D2OI];
    data[4335] = 0.0 + k[748]*y[IDX_pD2II] + k[749]*y[IDX_oD2II] +
        k[1420]*y[IDX_oD3II] + k[1421]*y[IDX_mD3II];
    data[4336] = 0.0 + k[1860]*y[IDX_pD2I] + k[1861]*y[IDX_oD2I] +
        k[1903]*y[IDX_D2OI];
    data[4337] = 0.0 - k[2301]*y[IDX_ND2II];
    data[4338] = 0.0 - k[1974]*y[IDX_ND2II];
    data[4339] = 0.0 + k[2250]*y[IDX_ND2I];
    data[4340] = 0.0 - k[1992]*y[IDX_ND2II];
    data[4341] = 0.0 + k[2295]*y[IDX_ND2I];
    data[4342] = 0.0 + k[2037]*y[IDX_NDI];
    data[4343] = 0.0 + k[1958]*y[IDX_NDI] + k[2605]*y[IDX_ND2I];
    data[4344] = 0.0 + k[2604]*y[IDX_ND2I];
    data[4345] = 0.0 - k[2836]*y[IDX_ND2II] - k[2839]*y[IDX_ND2II];
    data[4346] = 0.0 + k[1274]*y[IDX_NI] + k[1421]*y[IDX_NHI] +
        k[1438]*y[IDX_NDI];
    data[4347] = 0.0 + k[1861]*y[IDX_NHII] + k[1865]*y[IDX_NDII];
    data[4348] = 0.0 + k[749]*y[IDX_NHI] + k[759]*y[IDX_NDI] +
        k[2121]*y[IDX_ND2I];
    data[4349] = 0.0 + k[1279]*y[IDX_NI] + k[1447]*y[IDX_NDI];
    data[4350] = 0.0 + k[1273]*y[IDX_NI] + k[1420]*y[IDX_NHI] +
        k[1437]*y[IDX_NDI];
    data[4351] = 0.0 + k[2119]*y[IDX_ND2I];
    data[4352] = 0.0 + k[1439]*y[IDX_NDI];
    data[4353] = 0.0 + k[1860]*y[IDX_NHII] + k[1864]*y[IDX_NDII];
    data[4354] = 0.0 + k[748]*y[IDX_NHI] + k[758]*y[IDX_NDI] +
        k[2120]*y[IDX_ND2I];
    data[4355] = 0.0 + k[1280]*y[IDX_NI] + k[1448]*y[IDX_NDI];
    data[4356] = 0.0 + k[2958]*y[IDX_NI] + k[2971]*y[IDX_NDI] +
        k[2972]*y[IDX_NDI];
    data[4357] = 0.0 + k[2118]*y[IDX_ND2I];
    data[4358] = 0.0 + k[1440]*y[IDX_NDI] + k[1441]*y[IDX_NDI];
    data[4359] = 0.0 + k[1041]*y[IDX_NH2I] + k[1044]*y[IDX_NHDI] -
        k[2048]*y[IDX_NHI];
    data[4360] = 0.0 - k[421]*y[IDX_NHI];
    data[4361] = 0.0 - k[2682]*y[IDX_NHI];
    data[4362] = 0.0 + k[1977]*y[IDX_NH2II] + k[1979]*y[IDX_NHDII];
    data[4363] = 0.0 - k[1700]*y[IDX_NHI] - k[1702]*y[IDX_NHI];
    data[4364] = 0.0 + k[1986]*y[IDX_NH2II] + k[1989]*y[IDX_NHDII];
    data[4365] = 0.0 - k[1748]*y[IDX_NHI];
    data[4366] = 0.0 + k[1981]*y[IDX_NH2II] + k[1982]*y[IDX_ND2II] +
        k[1984]*y[IDX_NHDII];
    data[4367] = 0.0 - k[1747]*y[IDX_NHI];
    data[4368] = 0.0 + k[603]*y[IDX_H2OI] + k[605]*y[IDX_HDOI] -
        k[2069]*y[IDX_NHI];
    data[4369] = 0.0 + k[559]*y[IDX_HNOI];
    data[4370] = 0.0 - k[612]*y[IDX_NHI] + k[626]*y[IDX_NH2I] +
        k[628]*y[IDX_NHDI] - k[2325]*y[IDX_NHI];
    data[4371] = 0.0 + k[1107]*y[IDX_HNOI];
    data[4372] = 0.0 - k[2194]*y[IDX_NHI];
    data[4373] = 0.0 - k[2708]*y[IDX_NHI];
    data[4374] = 0.0 + k[2627]*y[IDX_NHII] + k[3331]*y[IDX_NH2II] +
        k[3333]*y[IDX_NHDII];
    data[4375] = 0.0 - k[3148]*y[IDX_NHI];
    data[4376] = 0.0 - k[1009]*y[IDX_NHI];
    data[4377] = 0.0 - k[1025]*y[IDX_NHI];
    data[4378] = 0.0 - k[1561]*y[IDX_NHI];
    data[4379] = 0.0 + k[1105]*y[IDX_HI];
    data[4380] = 0.0 - k[1583]*y[IDX_NHI];
    data[4381] = 0.0 + k[1096]*y[IDX_NOI] + k[1104]*y[IDX_HNOI] +
        k[1105]*y[IDX_DNOI];
    data[4382] = 0.0 - k[2193]*y[IDX_NHI];
    data[4383] = 0.0 + k[2693]*y[IDX_NI] - k[2707]*y[IDX_NHI];
    data[4384] = 0.0 + k[603]*y[IDX_CNII] + k[2625]*y[IDX_NHII] +
        k[3016]*y[IDX_NH2II] + k[3321]*y[IDX_NHDII] + k[3323]*y[IDX_ND2II];
    data[4385] = 0.0 - k[2999]*y[IDX_NHI];
    data[4386] = 0.0 + k[1670]*y[IDX_OII];
    data[4387] = 0.0 - k[1008]*y[IDX_NHI];
    data[4388] = 0.0 - k[1024]*y[IDX_NHI];
    data[4389] = 0.0 + k[1179]*y[IDX_NI] - k[1192]*y[IDX_NHI] -
        k[1193]*y[IDX_NHI];
    data[4390] = 0.0 - k[752]*y[IDX_NHI] - k[753]*y[IDX_NHI] -
        k[2392]*y[IDX_NHI];
    data[4391] = 0.0 + k[605]*y[IDX_CNII] + k[2629]*y[IDX_NHII] +
        k[3325]*y[IDX_NH2II] + k[3327]*y[IDX_NHDII] + k[3329]*y[IDX_ND2II];
    data[4392] = 0.0 - k[3147]*y[IDX_NHI];
    data[4393] = 0.0 - k[1560]*y[IDX_NHI];
    data[4394] = 0.0 + k[559]*y[IDX_COI] + k[1104]*y[IDX_HI] +
        k[1107]*y[IDX_DI];
    data[4395] = 0.0 - k[1582]*y[IDX_NHI];
    data[4396] = 0.0 - k[875]*y[IDX_NHI];
    data[4397] = 0.0 + k[1174]*y[IDX_pH2I] + k[1175]*y[IDX_oH2I] +
        k[1179]*y[IDX_HDI] - k[1200]*y[IDX_NHI] + k[1219]*y[IDX_O2HI] +
        k[2693]*y[IDX_HM];
    data[4398] = 0.0 - k[1642]*y[IDX_NHI] - k[2509]*y[IDX_NHI];
    data[4399] = 0.0 - k[2276]*y[IDX_NHI];
    data[4400] = 0.0 - k[1627]*y[IDX_NHI];
    data[4401] = 0.0 - k[1626]*y[IDX_NHI] + k[2832]*y[IDX_eM];
    data[4402] = 0.0 - k[562]*y[IDX_NHI];
    data[4403] = 0.0 - k[1873]*y[IDX_NHI];
    data[4404] = 0.0 + k[581]*y[IDX_OHI];
    data[4405] = 0.0 + k[1982]*y[IDX_CHI] + k[3323]*y[IDX_H2OI] +
        k[3329]*y[IDX_HDOI];
    data[4406] = 0.0 - k[235] - k[366] - k[368] - k[421]*y[IDX_CII] -
        k[561]*y[IDX_NHI] - k[561]*y[IDX_NHI] - k[561]*y[IDX_NHI] -
        k[561]*y[IDX_NHI] - k[562]*y[IDX_NDI] - k[564]*y[IDX_NOI] -
        k[612]*y[IDX_COII] - k[746]*y[IDX_pH2II] - k[747]*y[IDX_oH2II] -
        k[748]*y[IDX_pD2II] - k[749]*y[IDX_oD2II] - k[750]*y[IDX_pD2II] -
        k[751]*y[IDX_oD2II] - k[752]*y[IDX_HDII] - k[753]*y[IDX_HDII] -
        k[875]*y[IDX_HeII] - k[1008]*y[IDX_HCNII] - k[1009]*y[IDX_DCNII] -
        k[1024]*y[IDX_HCOII] - k[1025]*y[IDX_DCOII] - k[1180]*y[IDX_pH2I] -
        k[1181]*y[IDX_oH2I] - k[1186]*y[IDX_pD2I] - k[1187]*y[IDX_oD2I] -
        k[1188]*y[IDX_pD2I] - k[1189]*y[IDX_oD2I] - k[1192]*y[IDX_HDI] -
        k[1193]*y[IDX_HDI] - k[1200]*y[IDX_NI] - k[1230]*y[IDX_OI] -
        k[1417]*y[IDX_oH3II] - k[1418]*y[IDX_pH3II] - k[1419]*y[IDX_pH3II] -
        k[1420]*y[IDX_oD3II] - k[1421]*y[IDX_mD3II] - k[1422]*y[IDX_oH2DII] -
        k[1423]*y[IDX_pH2DII] - k[1424]*y[IDX_pH2DII] - k[1425]*y[IDX_oH2DII] -
        k[1426]*y[IDX_pH2DII] - k[1427]*y[IDX_oD2HII] - k[1428]*y[IDX_pD2HII] -
        k[1429]*y[IDX_pD2HII] - k[1430]*y[IDX_oD2HII] - k[1431]*y[IDX_pD2HII] -
        k[1560]*y[IDX_HNCII] - k[1561]*y[IDX_DNCII] - k[1582]*y[IDX_HNOII] -
        k[1583]*y[IDX_DNOII] - k[1626]*y[IDX_N2HII] - k[1627]*y[IDX_N2DII] -
        k[1642]*y[IDX_NII] - k[1663]*y[IDX_OII] - k[1700]*y[IDX_C2II] -
        k[1702]*y[IDX_C2II] - k[1747]*y[IDX_CHII] - k[1748]*y[IDX_CDII] -
        k[1872]*y[IDX_NHII] - k[1873]*y[IDX_NDII] - k[1915]*y[IDX_O2II] -
        k[1917]*y[IDX_O2II] - k[1955]*y[IDX_OHII] - k[1956]*y[IDX_ODII] -
        k[2034]*y[IDX_O2HII] - k[2035]*y[IDX_O2DII] - k[2048]*y[IDX_CI] -
        k[2069]*y[IDX_CNII] - k[2193]*y[IDX_HII] - k[2194]*y[IDX_DII] -
        k[2276]*y[IDX_N2II] - k[2325]*y[IDX_COII] - k[2388]*y[IDX_pH2II] -
        k[2389]*y[IDX_oH2II] - k[2390]*y[IDX_pD2II] - k[2391]*y[IDX_oD2II] -
        k[2392]*y[IDX_HDII] - k[2509]*y[IDX_NII] - k[2566]*y[IDX_OII] -
        k[2682]*y[IDX_CM] - k[2707]*y[IDX_HM] - k[2708]*y[IDX_DM] -
        k[2999]*y[IDX_H2OII] - k[3147]*y[IDX_HDOII] - k[3148]*y[IDX_D2OII];
    data[4407] = 0.0 - k[1872]*y[IDX_NHI] + k[2555]*y[IDX_NOI] +
        k[2558]*y[IDX_O2I] + k[2625]*y[IDX_H2OI] + k[2627]*y[IDX_D2OI] +
        k[2629]*y[IDX_HDOI];
    data[4408] = 0.0 + k[268] + k[410] + k[523]*y[IDX_OI] +
        k[580]*y[IDX_OHI] + k[585]*y[IDX_ODI] + k[626]*y[IDX_COII] +
        k[1041]*y[IDX_CI];
    data[4409] = 0.0 + k[1977]*y[IDX_C2I] + k[1981]*y[IDX_CHI] +
        k[1986]*y[IDX_CDI] + k[2838]*y[IDX_eM] + k[3016]*y[IDX_H2OI] +
        k[3325]*y[IDX_HDOI] + k[3331]*y[IDX_D2OI];
    data[4410] = 0.0 + k[271] + k[413] + k[525]*y[IDX_OI] +
        k[583]*y[IDX_OHI] + k[588]*y[IDX_ODI] + k[628]*y[IDX_COII] +
        k[1044]*y[IDX_CI];
    data[4411] = 0.0 + k[1979]*y[IDX_C2I] + k[1984]*y[IDX_CHI] +
        k[1989]*y[IDX_CDI] + k[2841]*y[IDX_eM] + k[3321]*y[IDX_H2OI] +
        k[3327]*y[IDX_HDOI] + k[3333]*y[IDX_D2OI];
    data[4412] = 0.0 - k[564]*y[IDX_NHI] + k[1096]*y[IDX_HI] +
        k[2555]*y[IDX_NHII];
    data[4413] = 0.0 + k[523]*y[IDX_NH2I] + k[525]*y[IDX_NHDI] -
        k[1230]*y[IDX_NHI];
    data[4414] = 0.0 - k[1663]*y[IDX_NHI] + k[1670]*y[IDX_HCNI] -
        k[2566]*y[IDX_NHI];
    data[4415] = 0.0 + k[2558]*y[IDX_NHII];
    data[4416] = 0.0 - k[1915]*y[IDX_NHI] - k[1917]*y[IDX_NHI];
    data[4417] = 0.0 - k[2035]*y[IDX_NHI];
    data[4418] = 0.0 + k[1219]*y[IDX_NI];
    data[4419] = 0.0 - k[2034]*y[IDX_NHI];
    data[4420] = 0.0 + k[585]*y[IDX_NH2I] + k[588]*y[IDX_NHDI];
    data[4421] = 0.0 - k[1956]*y[IDX_NHI];
    data[4422] = 0.0 + k[580]*y[IDX_NH2I] + k[581]*y[IDX_ND2I] +
        k[583]*y[IDX_NHDI];
    data[4423] = 0.0 - k[1955]*y[IDX_NHI];
    data[4424] = 0.0 + k[2832]*y[IDX_N2HII] + k[2838]*y[IDX_NH2II] +
        k[2841]*y[IDX_NHDII];
    data[4425] = 0.0 - k[1421]*y[IDX_NHI];
    data[4426] = 0.0 - k[1187]*y[IDX_NHI] - k[1189]*y[IDX_NHI];
    data[4427] = 0.0 - k[749]*y[IDX_NHI] - k[751]*y[IDX_NHI] -
        k[2391]*y[IDX_NHI];
    data[4428] = 0.0 - k[1427]*y[IDX_NHI] - k[1430]*y[IDX_NHI];
    data[4429] = 0.0 - k[1420]*y[IDX_NHI];
    data[4430] = 0.0 + k[1175]*y[IDX_NI] - k[1181]*y[IDX_NHI];
    data[4431] = 0.0 - k[747]*y[IDX_NHI] - k[2389]*y[IDX_NHI];
    data[4432] = 0.0 - k[1422]*y[IDX_NHI] - k[1425]*y[IDX_NHI];
    data[4433] = 0.0 - k[1417]*y[IDX_NHI];
    data[4434] = 0.0 - k[1186]*y[IDX_NHI] - k[1188]*y[IDX_NHI];
    data[4435] = 0.0 - k[748]*y[IDX_NHI] - k[750]*y[IDX_NHI] -
        k[2390]*y[IDX_NHI];
    data[4436] = 0.0 - k[1428]*y[IDX_NHI] - k[1429]*y[IDX_NHI] -
        k[1431]*y[IDX_NHI];
    data[4437] = 0.0 + k[1174]*y[IDX_NI] - k[1180]*y[IDX_NHI];
    data[4438] = 0.0 - k[746]*y[IDX_NHI] - k[2388]*y[IDX_NHI];
    data[4439] = 0.0 - k[1423]*y[IDX_NHI] - k[1424]*y[IDX_NHI] -
        k[1426]*y[IDX_NHI];
    data[4440] = 0.0 - k[1418]*y[IDX_NHI] - k[1419]*y[IDX_NHI];
    data[4441] = 0.0 - k[1818]*y[IDX_NHII];
    data[4442] = 0.0 - k[1824]*y[IDX_NHII] - k[1826]*y[IDX_NHII] -
        k[1828]*y[IDX_NHII];
    data[4443] = 0.0 - k[1832]*y[IDX_NHII];
    data[4444] = 0.0 - k[1830]*y[IDX_NHII];
    data[4445] = 0.0 - k[1834]*y[IDX_NHII];
    data[4446] = 0.0 + k[2069]*y[IDX_NHI];
    data[4447] = 0.0 - k[1836]*y[IDX_NHII] - k[1838]*y[IDX_NHII];
    data[4448] = 0.0 + k[2325]*y[IDX_NHI];
    data[4449] = 0.0 - k[1886]*y[IDX_NHII] - k[1888]*y[IDX_NHII];
    data[4450] = 0.0 + k[2194]*y[IDX_NHI];
    data[4451] = 0.0 - k[1893]*y[IDX_NHII] - k[1894]*y[IDX_NHII] -
        k[1903]*y[IDX_NHII] - k[1904]*y[IDX_NHII] - k[2627]*y[IDX_NHII] -
        k[3314]*y[IDX_NHII];
    data[4452] = 0.0 + k[2193]*y[IDX_NHI];
    data[4453] = 0.0 - k[1890]*y[IDX_NHII] - k[1900]*y[IDX_NHII] -
        k[2625]*y[IDX_NHII] - k[3014]*y[IDX_NHII];
    data[4454] = 0.0 + k[1650]*y[IDX_NII];
    data[4455] = 0.0 + k[1641]*y[IDX_NII] - k[1850]*y[IDX_NHII] -
        k[1851]*y[IDX_NHII] - k[1866]*y[IDX_NHII] - k[1867]*y[IDX_NHII];
    data[4456] = 0.0 + k[641]*y[IDX_NI] + k[2392]*y[IDX_NHI];
    data[4457] = 0.0 - k[1896]*y[IDX_NHII] - k[1897]*y[IDX_NHII] -
        k[1906]*y[IDX_NHII] - k[1907]*y[IDX_NHII] - k[2629]*y[IDX_NHII] -
        k[3312]*y[IDX_NHII];
    data[4458] = 0.0 + k[929]*y[IDX_HeII];
    data[4459] = 0.0 + k[929]*y[IDX_HNCI] + k[942]*y[IDX_NH2I] +
        k[945]*y[IDX_NHDI];
    data[4460] = 0.0 + k[636]*y[IDX_pH2II] + k[637]*y[IDX_oH2II] +
        k[641]*y[IDX_HDII] - k[1820]*y[IDX_NHII];
    data[4461] = 0.0 + k[1636]*y[IDX_pH2I] + k[1637]*y[IDX_oH2I] +
        k[1641]*y[IDX_HDI] + k[1650]*y[IDX_HCOI] + k[2509]*y[IDX_NHI];
    data[4462] = 0.0 - k[1870]*y[IDX_NHII];
    data[4463] = 0.0 + k[2276]*y[IDX_NHI];
    data[4464] = 0.0 - k[1874]*y[IDX_NHII];
    data[4465] = 0.0 + k[368] - k[1872]*y[IDX_NHII] + k[2069]*y[IDX_CNII] +
        k[2193]*y[IDX_HII] + k[2194]*y[IDX_DII] + k[2276]*y[IDX_N2II] +
        k[2325]*y[IDX_COII] + k[2388]*y[IDX_pH2II] + k[2389]*y[IDX_oH2II] +
        k[2390]*y[IDX_pD2II] + k[2391]*y[IDX_oD2II] + k[2392]*y[IDX_HDII] +
        k[2509]*y[IDX_NII] + k[2566]*y[IDX_OII];
    data[4466] = 0.0 - k[1818]*y[IDX_CI] - k[1820]*y[IDX_NI] -
        k[1822]*y[IDX_OI] - k[1824]*y[IDX_C2I] - k[1826]*y[IDX_C2I] -
        k[1828]*y[IDX_C2I] - k[1830]*y[IDX_CHI] - k[1832]*y[IDX_CDI] -
        k[1834]*y[IDX_CNI] - k[1836]*y[IDX_COI] - k[1838]*y[IDX_COI] -
        k[1840]*y[IDX_oH2I] - k[1844]*y[IDX_pD2I] - k[1845]*y[IDX_oD2I] -
        k[1846]*y[IDX_oD2I] - k[1850]*y[IDX_HDI] - k[1851]*y[IDX_HDI] -
        k[1854]*y[IDX_pH2I] - k[1855]*y[IDX_oH2I] - k[1860]*y[IDX_pD2I] -
        k[1861]*y[IDX_oD2I] - k[1862]*y[IDX_pD2I] - k[1863]*y[IDX_oD2I] -
        k[1866]*y[IDX_HDI] - k[1867]*y[IDX_HDI] - k[1870]*y[IDX_N2I] -
        k[1872]*y[IDX_NHI] - k[1874]*y[IDX_NDI] - k[1876]*y[IDX_NOI] -
        k[1878]*y[IDX_O2I] - k[1880]*y[IDX_O2I] - k[1882]*y[IDX_OHI] -
        k[1884]*y[IDX_ODI] - k[1886]*y[IDX_CO2I] - k[1888]*y[IDX_CO2I] -
        k[1890]*y[IDX_H2OI] - k[1893]*y[IDX_D2OI] - k[1894]*y[IDX_D2OI] -
        k[1896]*y[IDX_HDOI] - k[1897]*y[IDX_HDOI] - k[1900]*y[IDX_H2OI] -
        k[1903]*y[IDX_D2OI] - k[1904]*y[IDX_D2OI] - k[1906]*y[IDX_HDOI] -
        k[1907]*y[IDX_HDOI] - k[2555]*y[IDX_NOI] - k[2558]*y[IDX_O2I] -
        k[2625]*y[IDX_H2OI] - k[2627]*y[IDX_D2OI] - k[2629]*y[IDX_HDOI] -
        k[2759]*y[IDX_eM] - k[2925]*y[IDX_oH2I] - k[2926]*y[IDX_pH2I] -
        k[3014]*y[IDX_H2OI] - k[3312]*y[IDX_HDOI] - k[3314]*y[IDX_D2OI];
    data[4467] = 0.0 + k[942]*y[IDX_HeII];
    data[4468] = 0.0 + k[945]*y[IDX_HeII];
    data[4469] = 0.0 - k[1876]*y[IDX_NHII] - k[2555]*y[IDX_NHII];
    data[4470] = 0.0 - k[1822]*y[IDX_NHII];
    data[4471] = 0.0 + k[2566]*y[IDX_NHI];
    data[4472] = 0.0 - k[1878]*y[IDX_NHII] - k[1880]*y[IDX_NHII] -
        k[2558]*y[IDX_NHII];
    data[4473] = 0.0 - k[1884]*y[IDX_NHII];
    data[4474] = 0.0 - k[1882]*y[IDX_NHII];
    data[4475] = 0.0 - k[2759]*y[IDX_NHII];
    data[4476] = 0.0 - k[1845]*y[IDX_NHII] - k[1846]*y[IDX_NHII] -
        k[1861]*y[IDX_NHII] - k[1863]*y[IDX_NHII];
    data[4477] = 0.0 + k[2391]*y[IDX_NHI];
    data[4478] = 0.0 + k[1637]*y[IDX_NII] - k[1840]*y[IDX_NHII] -
        k[1855]*y[IDX_NHII] - k[2925]*y[IDX_NHII];
    data[4479] = 0.0 + k[637]*y[IDX_NI] + k[2389]*y[IDX_NHI];
    data[4480] = 0.0 - k[1844]*y[IDX_NHII] - k[1860]*y[IDX_NHII] -
        k[1862]*y[IDX_NHII];
    data[4481] = 0.0 + k[2390]*y[IDX_NHI];
    data[4482] = 0.0 + k[1636]*y[IDX_NII] - k[1854]*y[IDX_NHII] -
        k[2926]*y[IDX_NHII];
    data[4483] = 0.0 + k[636]*y[IDX_NI] + k[2388]*y[IDX_NHI];
    data[4484] = 0.0 - k[1041]*y[IDX_NH2I] - k[1045]*y[IDX_NH2I] -
        k[1049]*y[IDX_NH2I];
    data[4485] = 0.0 - k[451]*y[IDX_NH2I];
    data[4486] = 0.0 - k[2543]*y[IDX_NH2I];
    data[4487] = 0.0 + k[2611]*y[IDX_NH2II];
    data[4488] = 0.0 - k[1796]*y[IDX_NH2I] - k[1797]*y[IDX_NH2I];
    data[4489] = 0.0 + k[2549]*y[IDX_NH2II];
    data[4490] = 0.0 + k[2608]*y[IDX_NH2II];
    data[4491] = 0.0 - k[1795]*y[IDX_NH2I];
    data[4492] = 0.0 + k[2546]*y[IDX_NH2II];
    data[4493] = 0.0 + k[2552]*y[IDX_NH2II];
    data[4494] = 0.0 - k[2078]*y[IDX_NH2I];
    data[4495] = 0.0 - k[626]*y[IDX_NH2I] - k[2337]*y[IDX_NH2I];
    data[4496] = 0.0 - k[2254]*y[IDX_NH2I];
    data[4497] = 0.0 - k[2512]*y[IDX_NH2I];
    data[4498] = 0.0 + k[2306]*y[IDX_NH2II];
    data[4499] = 0.0 + k[1100]*y[IDX_HNOI];
    data[4500] = 0.0 - k[2253]*y[IDX_NH2I];
    data[4501] = 0.0 + k[2707]*y[IDX_NHI];
    data[4502] = 0.0 - k[2511]*y[IDX_NH2I];
    data[4503] = 0.0 + k[2303]*y[IDX_NH2II];
    data[4504] = 0.0 + k[1193]*y[IDX_NHI];
    data[4505] = 0.0 - k[2117]*y[IDX_NH2I];
    data[4506] = 0.0 - k[2513]*y[IDX_NH2I];
    data[4507] = 0.0 + k[1100]*y[IDX_HI];
    data[4508] = 0.0 - k[939]*y[IDX_NH2I] - k[942]*y[IDX_NH2I];
    data[4509] = 0.0 - k[2236]*y[IDX_NH2I];
    data[4510] = 0.0 - k[2290]*y[IDX_NH2I];
    data[4511] = 0.0 + k[1184]*y[IDX_pH2I] + k[1185]*y[IDX_oH2I];
    data[4512] = 0.0 + k[1180]*y[IDX_pH2I] + k[1181]*y[IDX_oH2I] +
        k[1193]*y[IDX_HDI] + k[2707]*y[IDX_HM];
    data[4513] = 0.0 - k[268] - k[272] - k[410] - k[414] - k[451]*y[IDX_CII]
        - k[523]*y[IDX_OI] - k[527]*y[IDX_OI] - k[566]*y[IDX_NOI] -
        k[580]*y[IDX_OHI] - k[585]*y[IDX_ODI] - k[586]*y[IDX_ODI] -
        k[626]*y[IDX_COII] - k[939]*y[IDX_HeII] - k[942]*y[IDX_HeII] -
        k[1041]*y[IDX_CI] - k[1045]*y[IDX_CI] - k[1049]*y[IDX_CI] -
        k[1795]*y[IDX_CHII] - k[1796]*y[IDX_CDII] - k[1797]*y[IDX_CDII] -
        k[2078]*y[IDX_CNII] - k[2113]*y[IDX_pH2II] - k[2114]*y[IDX_oH2II] -
        k[2115]*y[IDX_pD2II] - k[2116]*y[IDX_oD2II] - k[2117]*y[IDX_HDII] -
        k[2236]*y[IDX_NII] - k[2249]*y[IDX_OII] - k[2253]*y[IDX_HII] -
        k[2254]*y[IDX_DII] - k[2290]*y[IDX_N2II] - k[2294]*y[IDX_O2II] -
        k[2337]*y[IDX_COII] - k[2511]*y[IDX_H2OII] - k[2512]*y[IDX_D2OII] -
        k[2513]*y[IDX_HDOII] - k[2543]*y[IDX_C2II] - k[2602]*y[IDX_OHII] -
        k[2603]*y[IDX_ODII];
    data[4514] = 0.0 + k[2300]*y[IDX_NOI] + k[2303]*y[IDX_HCOI] +
        k[2306]*y[IDX_DCOI] + k[2546]*y[IDX_CH2I] + k[2549]*y[IDX_CD2I] +
        k[2552]*y[IDX_CHDI] + k[2608]*y[IDX_CHI] + k[2611]*y[IDX_CDI];
    data[4515] = 0.0 - k[566]*y[IDX_NH2I] + k[2300]*y[IDX_NH2II];
    data[4516] = 0.0 - k[523]*y[IDX_NH2I] - k[527]*y[IDX_NH2I];
    data[4517] = 0.0 - k[2249]*y[IDX_NH2I];
    data[4518] = 0.0 - k[2294]*y[IDX_NH2I];
    data[4519] = 0.0 - k[585]*y[IDX_NH2I] - k[586]*y[IDX_NH2I];
    data[4520] = 0.0 - k[2603]*y[IDX_NH2I];
    data[4521] = 0.0 - k[580]*y[IDX_NH2I];
    data[4522] = 0.0 - k[2602]*y[IDX_NH2I];
    data[4523] = 0.0 - k[2116]*y[IDX_NH2I];
    data[4524] = 0.0 + k[1181]*y[IDX_NHI] + k[1185]*y[IDX_NDI];
    data[4525] = 0.0 - k[2114]*y[IDX_NH2I];
    data[4526] = 0.0 - k[2115]*y[IDX_NH2I];
    data[4527] = 0.0 + k[1180]*y[IDX_NHI] + k[1184]*y[IDX_NDI];
    data[4528] = 0.0 - k[2113]*y[IDX_NH2I];
    data[4529] = 0.0 - k[1977]*y[IDX_NH2II];
    data[4530] = 0.0 + k[2543]*y[IDX_NH2I];
    data[4531] = 0.0 - k[1986]*y[IDX_NH2II] - k[1987]*y[IDX_NH2II] -
        k[2611]*y[IDX_NH2II];
    data[4532] = 0.0 - k[2549]*y[IDX_NH2II];
    data[4533] = 0.0 - k[1981]*y[IDX_NH2II] - k[2608]*y[IDX_NH2II];
    data[4534] = 0.0 - k[2546]*y[IDX_NH2II];
    data[4535] = 0.0 - k[2552]*y[IDX_NH2II];
    data[4536] = 0.0 + k[2078]*y[IDX_NH2I];
    data[4537] = 0.0 + k[2337]*y[IDX_NH2I];
    data[4538] = 0.0 + k[2254]*y[IDX_NH2I];
    data[4539] = 0.0 - k[3331]*y[IDX_NH2II] - k[3332]*y[IDX_NH2II];
    data[4540] = 0.0 + k[2512]*y[IDX_NH2I];
    data[4541] = 0.0 - k[2306]*y[IDX_NH2II];
    data[4542] = 0.0 + k[2253]*y[IDX_NH2I];
    data[4543] = 0.0 + k[1900]*y[IDX_NHII] + k[1902]*y[IDX_NDII] -
        k[3016]*y[IDX_NH2II];
    data[4544] = 0.0 + k[2511]*y[IDX_NH2I];
    data[4545] = 0.0 + k[1008]*y[IDX_NHI];
    data[4546] = 0.0 - k[2303]*y[IDX_NH2II];
    data[4547] = 0.0 + k[1024]*y[IDX_NHI];
    data[4548] = 0.0 + k[1867]*y[IDX_NHII];
    data[4549] = 0.0 + k[753]*y[IDX_NHI] + k[2117]*y[IDX_NH2I];
    data[4550] = 0.0 + k[1907]*y[IDX_NHII] - k[3325]*y[IDX_NH2II] -
        k[3326]*y[IDX_NH2II];
    data[4551] = 0.0 + k[2513]*y[IDX_NH2I];
    data[4552] = 0.0 + k[1560]*y[IDX_NHI];
    data[4553] = 0.0 + k[1582]*y[IDX_NHI];
    data[4554] = 0.0 + k[1271]*y[IDX_oH3II] + k[1272]*y[IDX_pH3II] +
        k[1277]*y[IDX_oH2DII] + k[1278]*y[IDX_pH2DII] - k[1969]*y[IDX_NH2II];
    data[4555] = 0.0 + k[2236]*y[IDX_NH2I];
    data[4556] = 0.0 + k[2290]*y[IDX_NH2I];
    data[4557] = 0.0 + k[1626]*y[IDX_NHI];
    data[4558] = 0.0 + k[756]*y[IDX_pH2II] + k[757]*y[IDX_oH2II] +
        k[1435]*y[IDX_oH3II] + k[1436]*y[IDX_pH3II];
    data[4559] = 0.0 + k[1858]*y[IDX_pH2I] + k[1859]*y[IDX_oH2I] +
        k[1902]*y[IDX_H2OI];
    data[4560] = 0.0 + k[746]*y[IDX_pH2II] + k[747]*y[IDX_oH2II] +
        k[753]*y[IDX_HDII] + k[1008]*y[IDX_HCNII] + k[1024]*y[IDX_HCOII] +
        k[1417]*y[IDX_oH3II] + k[1418]*y[IDX_pH3II] + k[1419]*y[IDX_pH3II] +
        k[1425]*y[IDX_oH2DII] + k[1426]*y[IDX_pH2DII] + k[1427]*y[IDX_oD2HII] +
        k[1428]*y[IDX_pD2HII] + k[1429]*y[IDX_pD2HII] + k[1560]*y[IDX_HNCII] +
        k[1582]*y[IDX_HNOII] + k[1626]*y[IDX_N2HII] + k[1872]*y[IDX_NHII] +
        k[1955]*y[IDX_OHII] + k[2034]*y[IDX_O2HII];
    data[4561] = 0.0 + k[1854]*y[IDX_pH2I] + k[1855]*y[IDX_oH2I] +
        k[1867]*y[IDX_HDI] + k[1872]*y[IDX_NHI] + k[1900]*y[IDX_H2OI] +
        k[1907]*y[IDX_HDOI];
    data[4562] = 0.0 + k[272] + k[414] + k[2078]*y[IDX_CNII] +
        k[2113]*y[IDX_pH2II] + k[2114]*y[IDX_oH2II] + k[2115]*y[IDX_pD2II] +
        k[2116]*y[IDX_oD2II] + k[2117]*y[IDX_HDII] + k[2236]*y[IDX_NII] +
        k[2249]*y[IDX_OII] + k[2253]*y[IDX_HII] + k[2254]*y[IDX_DII] +
        k[2290]*y[IDX_N2II] + k[2294]*y[IDX_O2II] + k[2337]*y[IDX_COII] +
        k[2511]*y[IDX_H2OII] + k[2512]*y[IDX_D2OII] + k[2513]*y[IDX_HDOII] +
        k[2543]*y[IDX_C2II] + k[2602]*y[IDX_OHII] + k[2603]*y[IDX_ODII];
    data[4563] = 0.0 - k[1969]*y[IDX_NI] - k[1973]*y[IDX_OI] -
        k[1977]*y[IDX_C2I] - k[1981]*y[IDX_CHI] - k[1986]*y[IDX_CDI] -
        k[1987]*y[IDX_CDI] - k[1991]*y[IDX_O2I] - k[2300]*y[IDX_NOI] -
        k[2303]*y[IDX_HCOI] - k[2306]*y[IDX_DCOI] - k[2546]*y[IDX_CH2I] -
        k[2549]*y[IDX_CD2I] - k[2552]*y[IDX_CHDI] - k[2608]*y[IDX_CHI] -
        k[2611]*y[IDX_CDI] - k[2835]*y[IDX_eM] - k[2838]*y[IDX_eM] -
        k[3016]*y[IDX_H2OI] - k[3325]*y[IDX_HDOI] - k[3326]*y[IDX_HDOI] -
        k[3331]*y[IDX_D2OI] - k[3332]*y[IDX_D2OI];
    data[4564] = 0.0 - k[2300]*y[IDX_NH2II];
    data[4565] = 0.0 - k[1973]*y[IDX_NH2II];
    data[4566] = 0.0 + k[2249]*y[IDX_NH2I];
    data[4567] = 0.0 - k[1991]*y[IDX_NH2II];
    data[4568] = 0.0 + k[2294]*y[IDX_NH2I];
    data[4569] = 0.0 + k[2034]*y[IDX_NHI];
    data[4570] = 0.0 + k[2603]*y[IDX_NH2I];
    data[4571] = 0.0 + k[1955]*y[IDX_NHI] + k[2602]*y[IDX_NH2I];
    data[4572] = 0.0 - k[2835]*y[IDX_NH2II] - k[2838]*y[IDX_NH2II];
    data[4573] = 0.0 + k[2116]*y[IDX_NH2I];
    data[4574] = 0.0 + k[1427]*y[IDX_NHI];
    data[4575] = 0.0 + k[1855]*y[IDX_NHII] + k[1859]*y[IDX_NDII];
    data[4576] = 0.0 + k[747]*y[IDX_NHI] + k[757]*y[IDX_NDI] +
        k[2114]*y[IDX_NH2I];
    data[4577] = 0.0 + k[1277]*y[IDX_NI] + k[1425]*y[IDX_NHI];
    data[4578] = 0.0 + k[1271]*y[IDX_NI] + k[1417]*y[IDX_NHI] +
        k[1435]*y[IDX_NDI];
    data[4579] = 0.0 + k[2115]*y[IDX_NH2I];
    data[4580] = 0.0 + k[1428]*y[IDX_NHI] + k[1429]*y[IDX_NHI];
    data[4581] = 0.0 + k[1854]*y[IDX_NHII] + k[1858]*y[IDX_NDII];
    data[4582] = 0.0 + k[746]*y[IDX_NHI] + k[756]*y[IDX_NDI] +
        k[2113]*y[IDX_NH2I];
    data[4583] = 0.0 + k[1278]*y[IDX_NI] + k[1426]*y[IDX_NHI];
    data[4584] = 0.0 + k[1272]*y[IDX_NI] + k[1418]*y[IDX_NHI] +
        k[1419]*y[IDX_NHI] + k[1436]*y[IDX_NDI];
    data[4585] = 0.0 - k[1043]*y[IDX_NHDI] - k[1044]*y[IDX_NHDI] -
        k[1047]*y[IDX_NHDI] - k[1048]*y[IDX_NHDI] - k[1051]*y[IDX_NHDI] -
        k[1052]*y[IDX_NHDI];
    data[4586] = 0.0 - k[453]*y[IDX_NHDI] - k[454]*y[IDX_NHDI];
    data[4587] = 0.0 - k[2545]*y[IDX_NHDI];
    data[4588] = 0.0 + k[2613]*y[IDX_NHDII];
    data[4589] = 0.0 - k[1803]*y[IDX_NHDI] - k[1804]*y[IDX_NHDI];
    data[4590] = 0.0 + k[2551]*y[IDX_NHDII];
    data[4591] = 0.0 + k[2610]*y[IDX_NHDII];
    data[4592] = 0.0 - k[1801]*y[IDX_NHDI] - k[1802]*y[IDX_NHDI];
    data[4593] = 0.0 + k[2548]*y[IDX_NHDII];
    data[4594] = 0.0 + k[2554]*y[IDX_NHDII];
    data[4595] = 0.0 - k[2080]*y[IDX_NHDI];
    data[4596] = 0.0 - k[628]*y[IDX_NHDI] - k[629]*y[IDX_NHDI] -
        k[2339]*y[IDX_NHDI];
    data[4597] = 0.0 + k[1102]*y[IDX_HNOI];
    data[4598] = 0.0 - k[2258]*y[IDX_NHDI];
    data[4599] = 0.0 + k[2708]*y[IDX_NHI];
    data[4600] = 0.0 - k[2518]*y[IDX_NHDI];
    data[4601] = 0.0 + k[2308]*y[IDX_NHDII];
    data[4602] = 0.0 + k[1101]*y[IDX_HI];
    data[4603] = 0.0 + k[1101]*y[IDX_DNOI];
    data[4604] = 0.0 - k[2257]*y[IDX_NHDI];
    data[4605] = 0.0 + k[2709]*y[IDX_NDI];
    data[4606] = 0.0 - k[2517]*y[IDX_NHDI];
    data[4607] = 0.0 + k[2305]*y[IDX_NHDII];
    data[4608] = 0.0 + k[1192]*y[IDX_NHI] + k[1195]*y[IDX_NDI];
    data[4609] = 0.0 - k[2127]*y[IDX_NHDI];
    data[4610] = 0.0 - k[2519]*y[IDX_NHDI];
    data[4611] = 0.0 + k[1102]*y[IDX_DI];
    data[4612] = 0.0 - k[941]*y[IDX_NHDI] - k[944]*y[IDX_NHDI] -
        k[945]*y[IDX_NHDI];
    data[4613] = 0.0 - k[2238]*y[IDX_NHDI];
    data[4614] = 0.0 - k[2292]*y[IDX_NHDI];
    data[4615] = 0.0 + k[1182]*y[IDX_pH2I] + k[1183]*y[IDX_oH2I] +
        k[1195]*y[IDX_HDI] + k[2709]*y[IDX_HM];
    data[4616] = 0.0 + k[1188]*y[IDX_pD2I] + k[1189]*y[IDX_oD2I] +
        k[1192]*y[IDX_HDI] + k[2708]*y[IDX_DM];
    data[4617] = 0.0 - k[270] - k[271] - k[274] - k[412] - k[413] - k[416] -
        k[453]*y[IDX_CII] - k[454]*y[IDX_CII] - k[525]*y[IDX_OI] -
        k[526]*y[IDX_OI] - k[529]*y[IDX_OI] - k[530]*y[IDX_OI] -
        k[568]*y[IDX_NOI] - k[583]*y[IDX_OHI] - k[584]*y[IDX_OHI] -
        k[588]*y[IDX_ODI] - k[589]*y[IDX_ODI] - k[628]*y[IDX_COII] -
        k[629]*y[IDX_COII] - k[941]*y[IDX_HeII] - k[944]*y[IDX_HeII] -
        k[945]*y[IDX_HeII] - k[1043]*y[IDX_CI] - k[1044]*y[IDX_CI] -
        k[1047]*y[IDX_CI] - k[1048]*y[IDX_CI] - k[1051]*y[IDX_CI] -
        k[1052]*y[IDX_CI] - k[1801]*y[IDX_CHII] - k[1802]*y[IDX_CHII] -
        k[1803]*y[IDX_CDII] - k[1804]*y[IDX_CDII] - k[2080]*y[IDX_CNII] -
        k[2123]*y[IDX_pH2II] - k[2124]*y[IDX_oH2II] - k[2125]*y[IDX_pD2II] -
        k[2126]*y[IDX_oD2II] - k[2127]*y[IDX_HDII] - k[2238]*y[IDX_NII] -
        k[2251]*y[IDX_OII] - k[2257]*y[IDX_HII] - k[2258]*y[IDX_DII] -
        k[2292]*y[IDX_N2II] - k[2296]*y[IDX_O2II] - k[2339]*y[IDX_COII] -
        k[2517]*y[IDX_H2OII] - k[2518]*y[IDX_D2OII] - k[2519]*y[IDX_HDOII] -
        k[2545]*y[IDX_C2II] - k[2606]*y[IDX_OHII] - k[2607]*y[IDX_ODII];
    data[4618] = 0.0 + k[2302]*y[IDX_NOI] + k[2305]*y[IDX_HCOI] +
        k[2308]*y[IDX_DCOI] + k[2548]*y[IDX_CH2I] + k[2551]*y[IDX_CD2I] +
        k[2554]*y[IDX_CHDI] + k[2610]*y[IDX_CHI] + k[2613]*y[IDX_CDI];
    data[4619] = 0.0 - k[568]*y[IDX_NHDI] + k[2302]*y[IDX_NHDII];
    data[4620] = 0.0 - k[525]*y[IDX_NHDI] - k[526]*y[IDX_NHDI] -
        k[529]*y[IDX_NHDI] - k[530]*y[IDX_NHDI];
    data[4621] = 0.0 - k[2251]*y[IDX_NHDI];
    data[4622] = 0.0 - k[2296]*y[IDX_NHDI];
    data[4623] = 0.0 - k[588]*y[IDX_NHDI] - k[589]*y[IDX_NHDI];
    data[4624] = 0.0 - k[2607]*y[IDX_NHDI];
    data[4625] = 0.0 - k[583]*y[IDX_NHDI] - k[584]*y[IDX_NHDI];
    data[4626] = 0.0 - k[2606]*y[IDX_NHDI];
    data[4627] = 0.0 + k[1189]*y[IDX_NHI];
    data[4628] = 0.0 - k[2126]*y[IDX_NHDI];
    data[4629] = 0.0 + k[1183]*y[IDX_NDI];
    data[4630] = 0.0 - k[2124]*y[IDX_NHDI];
    data[4631] = 0.0 + k[1188]*y[IDX_NHI];
    data[4632] = 0.0 - k[2125]*y[IDX_NHDI];
    data[4633] = 0.0 + k[1182]*y[IDX_NDI];
    data[4634] = 0.0 - k[2123]*y[IDX_NHDI];
    data[4635] = 0.0 - k[1979]*y[IDX_NHDII] - k[1980]*y[IDX_NHDII];
    data[4636] = 0.0 + k[2545]*y[IDX_NHDI];
    data[4637] = 0.0 - k[1989]*y[IDX_NHDII] - k[1990]*y[IDX_NHDII] -
        k[2613]*y[IDX_NHDII];
    data[4638] = 0.0 - k[2551]*y[IDX_NHDII];
    data[4639] = 0.0 - k[1984]*y[IDX_NHDII] - k[1985]*y[IDX_NHDII] -
        k[2610]*y[IDX_NHDII];
    data[4640] = 0.0 - k[2548]*y[IDX_NHDII];
    data[4641] = 0.0 - k[2554]*y[IDX_NHDII];
    data[4642] = 0.0 + k[2080]*y[IDX_NHDI];
    data[4643] = 0.0 + k[2339]*y[IDX_NHDI];
    data[4644] = 0.0 + k[2258]*y[IDX_NHDI];
    data[4645] = 0.0 + k[1904]*y[IDX_NHII] - k[3333]*y[IDX_NHDII] -
        k[3334]*y[IDX_NHDII];
    data[4646] = 0.0 + k[2518]*y[IDX_NHDI];
    data[4647] = 0.0 + k[1009]*y[IDX_NHI];
    data[4648] = 0.0 - k[2308]*y[IDX_NHDII];
    data[4649] = 0.0 + k[1025]*y[IDX_NHI];
    data[4650] = 0.0 + k[1561]*y[IDX_NHI];
    data[4651] = 0.0 + k[1583]*y[IDX_NHI];
    data[4652] = 0.0 + k[2257]*y[IDX_NHDI];
    data[4653] = 0.0 + k[1901]*y[IDX_NDII] - k[3321]*y[IDX_NHDII] -
        k[3322]*y[IDX_NHDII];
    data[4654] = 0.0 + k[2517]*y[IDX_NHDI];
    data[4655] = 0.0 + k[1010]*y[IDX_NDI];
    data[4656] = 0.0 - k[2305]*y[IDX_NHDII];
    data[4657] = 0.0 + k[1026]*y[IDX_NDI];
    data[4658] = 0.0 + k[1866]*y[IDX_NHII] + k[1869]*y[IDX_NDII];
    data[4659] = 0.0 + k[752]*y[IDX_NHI] + k[761]*y[IDX_NDI] +
        k[2127]*y[IDX_NHDI];
    data[4660] = 0.0 + k[1906]*y[IDX_NHII] + k[1909]*y[IDX_NDII] -
        k[3327]*y[IDX_NHDII] - k[3328]*y[IDX_NHDII];
    data[4661] = 0.0 + k[2519]*y[IDX_NHDI];
    data[4662] = 0.0 + k[1562]*y[IDX_NDI];
    data[4663] = 0.0 + k[1584]*y[IDX_NDI];
    data[4664] = 0.0 + k[1275]*y[IDX_oH2DII] + k[1276]*y[IDX_pH2DII] +
        k[1281]*y[IDX_oD2HII] + k[1282]*y[IDX_pD2HII] - k[1971]*y[IDX_NHDII] -
        k[1972]*y[IDX_NHDII];
    data[4665] = 0.0 + k[2238]*y[IDX_NHDI];
    data[4666] = 0.0 + k[2292]*y[IDX_NHDI];
    data[4667] = 0.0 + k[1627]*y[IDX_NHI];
    data[4668] = 0.0 + k[1628]*y[IDX_NDI];
    data[4669] = 0.0 + k[754]*y[IDX_pH2II] + k[755]*y[IDX_oH2II] +
        k[761]*y[IDX_HDII] + k[1010]*y[IDX_HCNII] + k[1026]*y[IDX_HCOII] +
        k[1432]*y[IDX_oH3II] + k[1433]*y[IDX_pH3II] + k[1434]*y[IDX_pH3II] +
        k[1442]*y[IDX_oH2DII] + k[1443]*y[IDX_pH2DII] + k[1444]*y[IDX_oD2HII] +
        k[1445]*y[IDX_pD2HII] + k[1446]*y[IDX_pD2HII] + k[1562]*y[IDX_HNCII] +
        k[1584]*y[IDX_HNOII] + k[1628]*y[IDX_N2HII] + k[1874]*y[IDX_NHII] +
        k[1957]*y[IDX_OHII] + k[2036]*y[IDX_O2HII];
    data[4670] = 0.0 + k[1856]*y[IDX_pH2I] + k[1857]*y[IDX_oH2I] +
        k[1869]*y[IDX_HDI] + k[1873]*y[IDX_NHI] + k[1901]*y[IDX_H2OI] +
        k[1909]*y[IDX_HDOI];
    data[4671] = 0.0 + k[750]*y[IDX_pD2II] + k[751]*y[IDX_oD2II] +
        k[752]*y[IDX_HDII] + k[1009]*y[IDX_DCNII] + k[1025]*y[IDX_DCOII] +
        k[1422]*y[IDX_oH2DII] + k[1423]*y[IDX_pH2DII] + k[1424]*y[IDX_pH2DII] +
        k[1430]*y[IDX_oD2HII] + k[1431]*y[IDX_pD2HII] + k[1561]*y[IDX_DNCII] +
        k[1583]*y[IDX_DNOII] + k[1627]*y[IDX_N2DII] + k[1873]*y[IDX_NDII] +
        k[1956]*y[IDX_ODII] + k[2035]*y[IDX_O2DII];
    data[4672] = 0.0 + k[1862]*y[IDX_pD2I] + k[1863]*y[IDX_oD2I] +
        k[1866]*y[IDX_HDI] + k[1874]*y[IDX_NDI] + k[1904]*y[IDX_D2OI] +
        k[1906]*y[IDX_HDOI];
    data[4673] = 0.0 + k[274] + k[416] + k[2080]*y[IDX_CNII] +
        k[2123]*y[IDX_pH2II] + k[2124]*y[IDX_oH2II] + k[2125]*y[IDX_pD2II] +
        k[2126]*y[IDX_oD2II] + k[2127]*y[IDX_HDII] + k[2238]*y[IDX_NII] +
        k[2251]*y[IDX_OII] + k[2257]*y[IDX_HII] + k[2258]*y[IDX_DII] +
        k[2292]*y[IDX_N2II] + k[2296]*y[IDX_O2II] + k[2339]*y[IDX_COII] +
        k[2517]*y[IDX_H2OII] + k[2518]*y[IDX_D2OII] + k[2519]*y[IDX_HDOII] +
        k[2545]*y[IDX_C2II] + k[2606]*y[IDX_OHII] + k[2607]*y[IDX_ODII];
    data[4674] = 0.0 - k[1971]*y[IDX_NI] - k[1972]*y[IDX_NI] -
        k[1975]*y[IDX_OI] - k[1976]*y[IDX_OI] - k[1979]*y[IDX_C2I] -
        k[1980]*y[IDX_C2I] - k[1984]*y[IDX_CHI] - k[1985]*y[IDX_CHI] -
        k[1989]*y[IDX_CDI] - k[1990]*y[IDX_CDI] - k[1993]*y[IDX_O2I] -
        k[1994]*y[IDX_O2I] - k[2302]*y[IDX_NOI] - k[2305]*y[IDX_HCOI] -
        k[2308]*y[IDX_DCOI] - k[2548]*y[IDX_CH2I] - k[2551]*y[IDX_CD2I] -
        k[2554]*y[IDX_CHDI] - k[2610]*y[IDX_CHI] - k[2613]*y[IDX_CDI] -
        k[2837]*y[IDX_eM] - k[2840]*y[IDX_eM] - k[2841]*y[IDX_eM] -
        k[3321]*y[IDX_H2OI] - k[3322]*y[IDX_H2OI] - k[3327]*y[IDX_HDOI] -
        k[3328]*y[IDX_HDOI] - k[3333]*y[IDX_D2OI] - k[3334]*y[IDX_D2OI];
    data[4675] = 0.0 - k[2302]*y[IDX_NHDII];
    data[4676] = 0.0 - k[1975]*y[IDX_NHDII] - k[1976]*y[IDX_NHDII];
    data[4677] = 0.0 + k[2251]*y[IDX_NHDI];
    data[4678] = 0.0 - k[1993]*y[IDX_NHDII] - k[1994]*y[IDX_NHDII];
    data[4679] = 0.0 + k[2296]*y[IDX_NHDI];
    data[4680] = 0.0 + k[2035]*y[IDX_NHI];
    data[4681] = 0.0 + k[2036]*y[IDX_NDI];
    data[4682] = 0.0 + k[1956]*y[IDX_NHI] + k[2607]*y[IDX_NHDI];
    data[4683] = 0.0 + k[1957]*y[IDX_NDI] + k[2606]*y[IDX_NHDI];
    data[4684] = 0.0 - k[2837]*y[IDX_NHDII] - k[2840]*y[IDX_NHDII] -
        k[2841]*y[IDX_NHDII];
    data[4685] = 0.0 + k[1863]*y[IDX_NHII];
    data[4686] = 0.0 + k[751]*y[IDX_NHI] + k[2126]*y[IDX_NHDI];
    data[4687] = 0.0 + k[1281]*y[IDX_NI] + k[1430]*y[IDX_NHI] +
        k[1444]*y[IDX_NDI];
    data[4688] = 0.0 + k[1857]*y[IDX_NDII];
    data[4689] = 0.0 + k[755]*y[IDX_NDI] + k[2124]*y[IDX_NHDI];
    data[4690] = 0.0 + k[1275]*y[IDX_NI] + k[1422]*y[IDX_NHI] +
        k[1442]*y[IDX_NDI];
    data[4691] = 0.0 + k[1432]*y[IDX_NDI];
    data[4692] = 0.0 + k[1862]*y[IDX_NHII];
    data[4693] = 0.0 + k[750]*y[IDX_NHI] + k[2125]*y[IDX_NHDI];
    data[4694] = 0.0 + k[1282]*y[IDX_NI] + k[1431]*y[IDX_NHI] +
        k[1445]*y[IDX_NDI] + k[1446]*y[IDX_NDI];
    data[4695] = 0.0 + k[1856]*y[IDX_NDII];
    data[4696] = 0.0 + k[754]*y[IDX_NDI] + k[2123]*y[IDX_NHDI];
    data[4697] = 0.0 + k[1276]*y[IDX_NI] + k[1423]*y[IDX_NHI] +
        k[1424]*y[IDX_NHI] + k[1443]*y[IDX_NDI];
    data[4698] = 0.0 + k[1433]*y[IDX_NDI] + k[1434]*y[IDX_NDI];
    data[4699] = 0.0 + k[1570]*y[IDX_HNOII] + k[1571]*y[IDX_DNOII] -
        k[2050]*y[IDX_NOI] - k[2051]*y[IDX_NOI];
    data[4700] = 0.0 - k[2065]*y[IDX_NOI];
    data[4701] = 0.0 - k[480]*y[IDX_NOI];
    data[4702] = 0.0 + k[1574]*y[IDX_HNOII] + k[1575]*y[IDX_DNOII];
    data[4703] = 0.0 - k[2260]*y[IDX_NOI];
    data[4704] = 0.0 - k[2163]*y[IDX_NOI];
    data[4705] = 0.0 - k[2162]*y[IDX_NOI];
    data[4706] = 0.0 - k[538]*y[IDX_NOI] + k[543]*y[IDX_HNOI] +
        k[544]*y[IDX_DNOI] + k[1578]*y[IDX_HNOII] + k[1579]*y[IDX_DNOII];
    data[4707] = 0.0 - k[2262]*y[IDX_NOI];
    data[4708] = 0.0 - k[2165]*y[IDX_NOI];
    data[4709] = 0.0 - k[537]*y[IDX_NOI] + k[541]*y[IDX_HNOI] +
        k[542]*y[IDX_DNOI] + k[1576]*y[IDX_HNOII] + k[1577]*y[IDX_DNOII];
    data[4710] = 0.0 - k[2261]*y[IDX_NOI];
    data[4711] = 0.0 - k[2164]*y[IDX_NOI];
    data[4712] = 0.0 - k[2166]*y[IDX_NOI];
    data[4713] = 0.0 - k[547]*y[IDX_NOI] + k[555]*y[IDX_HNOI] +
        k[556]*y[IDX_DNOI];
    data[4714] = 0.0 - k[594]*y[IDX_NOI] + k[597]*y[IDX_CO2I] -
        k[2316]*y[IDX_NOI];
    data[4715] = 0.0 + k[1580]*y[IDX_HNOII] + k[1581]*y[IDX_DNOII];
    data[4716] = 0.0 - k[2086]*y[IDX_NOI];
    data[4717] = 0.0 + k[597]*y[IDX_CNII] + k[1649]*y[IDX_NII];
    data[4718] = 0.0 - k[2167]*y[IDX_NOI];
    data[4719] = 0.0 - k[1097]*y[IDX_NOI] - k[1099]*y[IDX_NOI] +
        k[1129]*y[IDX_NO2I];
    data[4720] = 0.0 - k[2198]*y[IDX_NOI];
    data[4721] = 0.0 + k[3276]*y[IDX_HNOII] + k[3277]*y[IDX_DNOII];
    data[4722] = 0.0 - k[2180]*y[IDX_NOI];
    data[4723] = 0.0 - k[2176]*y[IDX_NOI];
    data[4724] = 0.0 - k[2220]*y[IDX_NOI];
    data[4725] = 0.0 + k[409] + k[522]*y[IDX_OI] + k[542]*y[IDX_CHI] +
        k[544]*y[IDX_CDI] + k[556]*y[IDX_CNI] + k[577]*y[IDX_OHI] +
        k[579]*y[IDX_ODI] + k[932]*y[IDX_HeII];
    data[4726] = 0.0 + k[1571]*y[IDX_CI] + k[1575]*y[IDX_C2I] +
        k[1577]*y[IDX_CHI] + k[1579]*y[IDX_CDI] + k[1581]*y[IDX_COI] +
        k[1583]*y[IDX_NHI] + k[1585]*y[IDX_NDI] + k[1587]*y[IDX_OHI] +
        k[1589]*y[IDX_ODI] - k[2222]*y[IDX_NOI] + k[2827]*y[IDX_eM] +
        k[3273]*y[IDX_H2OI] + k[3275]*y[IDX_HDOI] + k[3277]*y[IDX_D2OI];
    data[4727] = 0.0 - k[1096]*y[IDX_NOI] - k[1098]*y[IDX_NOI] +
        k[1128]*y[IDX_NO2I];
    data[4728] = 0.0 - k[2197]*y[IDX_NOI];
    data[4729] = 0.0 + k[3008]*y[IDX_HNOII] + k[3273]*y[IDX_DNOII];
    data[4730] = 0.0 - k[2179]*y[IDX_NOI];
    data[4731] = 0.0 - k[2175]*y[IDX_NOI];
    data[4732] = 0.0 - k[766]*y[IDX_NOI] - k[767]*y[IDX_NOI] -
        k[2402]*y[IDX_NOI];
    data[4733] = 0.0 + k[3274]*y[IDX_HNOII] + k[3275]*y[IDX_DNOII];
    data[4734] = 0.0 - k[2181]*y[IDX_NOI];
    data[4735] = 0.0 - k[2219]*y[IDX_NOI];
    data[4736] = 0.0 + k[408] + k[521]*y[IDX_OI] + k[541]*y[IDX_CHI] +
        k[543]*y[IDX_CDI] + k[555]*y[IDX_CNI] + k[576]*y[IDX_OHI] +
        k[578]*y[IDX_ODI] + k[931]*y[IDX_HeII];
    data[4737] = 0.0 + k[1570]*y[IDX_CI] + k[1574]*y[IDX_C2I] +
        k[1576]*y[IDX_CHI] + k[1578]*y[IDX_CDI] + k[1580]*y[IDX_COI] +
        k[1582]*y[IDX_NHI] + k[1584]*y[IDX_NDI] + k[1586]*y[IDX_OHI] +
        k[1588]*y[IDX_ODI] - k[2221]*y[IDX_NOI] + k[2826]*y[IDX_eM] +
        k[3008]*y[IDX_H2OI] + k[3274]*y[IDX_HDOI] + k[3276]*y[IDX_D2OI];
    data[4738] = 0.0 - k[877]*y[IDX_NOI] - k[878]*y[IDX_NOI] +
        k[931]*y[IDX_HNOI] + k[932]*y[IDX_DNOI] + k[935]*y[IDX_N2OI];
    data[4739] = 0.0 - k[1202]*y[IDX_NOI] + k[1203]*y[IDX_OHI] +
        k[1204]*y[IDX_ODI] + k[1205]*y[IDX_O2I] + k[1217]*y[IDX_NO2I] +
        k[1217]*y[IDX_NO2I] + k[2718]*y[IDX_OM];
    data[4740] = 0.0 - k[1644]*y[IDX_NOI] + k[1645]*y[IDX_O2I] +
        k[1649]*y[IDX_CO2I] - k[2520]*y[IDX_NOI];
    data[4741] = 0.0 - k[2278]*y[IDX_NOI];
    data[4742] = 0.0 + k[267] + k[935]*y[IDX_HeII] + k[1678]*y[IDX_OII];
    data[4743] = 0.0 - k[565]*y[IDX_NOI] + k[1231]*y[IDX_OI] +
        k[1584]*y[IDX_HNOII] + k[1585]*y[IDX_DNOII];
    data[4744] = 0.0 - k[1877]*y[IDX_NOI] - k[2556]*y[IDX_NOI];
    data[4745] = 0.0 - k[567]*y[IDX_NOI];
    data[4746] = 0.0 - k[2301]*y[IDX_NOI];
    data[4747] = 0.0 - k[564]*y[IDX_NOI] + k[1230]*y[IDX_OI] +
        k[1582]*y[IDX_HNOII] + k[1583]*y[IDX_DNOII];
    data[4748] = 0.0 - k[1876]*y[IDX_NOI] - k[2555]*y[IDX_NOI];
    data[4749] = 0.0 - k[566]*y[IDX_NOI];
    data[4750] = 0.0 - k[2300]*y[IDX_NOI];
    data[4751] = 0.0 - k[568]*y[IDX_NOI];
    data[4752] = 0.0 - k[2302]*y[IDX_NOI];
    data[4753] = 0.0 - k[237] - k[238] - k[370] - k[371] - k[480]*y[IDX_CM]
        - k[537]*y[IDX_CHI] - k[538]*y[IDX_CDI] - k[547]*y[IDX_CNI] -
        k[564]*y[IDX_NHI] - k[565]*y[IDX_NDI] - k[566]*y[IDX_NH2I] -
        k[567]*y[IDX_ND2I] - k[568]*y[IDX_NHDI] - k[594]*y[IDX_CNII] -
        k[762]*y[IDX_pH2II] - k[763]*y[IDX_oH2II] - k[764]*y[IDX_pD2II] -
        k[765]*y[IDX_oD2II] - k[766]*y[IDX_HDII] - k[767]*y[IDX_HDII] -
        k[877]*y[IDX_HeII] - k[878]*y[IDX_HeII] - k[1096]*y[IDX_HI] -
        k[1097]*y[IDX_DI] - k[1098]*y[IDX_HI] - k[1099]*y[IDX_DI] -
        k[1202]*y[IDX_NI] - k[1449]*y[IDX_oH3II] - k[1450]*y[IDX_pH3II] -
        k[1451]*y[IDX_pH3II] - k[1452]*y[IDX_oD3II] - k[1453]*y[IDX_mD3II] -
        k[1454]*y[IDX_oH2DII] - k[1455]*y[IDX_pH2DII] - k[1456]*y[IDX_pH2DII] -
        k[1457]*y[IDX_oH2DII] - k[1458]*y[IDX_pH2DII] - k[1459]*y[IDX_oD2HII] -
        k[1460]*y[IDX_pD2HII] - k[1461]*y[IDX_pD2HII] - k[1462]*y[IDX_oD2HII] -
        k[1463]*y[IDX_pD2HII] - k[1644]*y[IDX_NII] - k[1876]*y[IDX_NHII] -
        k[1877]*y[IDX_NDII] - k[1959]*y[IDX_OHII] - k[1960]*y[IDX_ODII] -
        k[2038]*y[IDX_O2HII] - k[2039]*y[IDX_O2DII] - k[2050]*y[IDX_CI] -
        k[2051]*y[IDX_CI] - k[2065]*y[IDX_CII] - k[2086]*y[IDX_COII] -
        k[2162]*y[IDX_C2HII] - k[2163]*y[IDX_C2DII] - k[2164]*y[IDX_CH2II] -
        k[2165]*y[IDX_CD2II] - k[2166]*y[IDX_CHDII] - k[2167]*y[IDX_CO2II] -
        k[2175]*y[IDX_HCNII] - k[2176]*y[IDX_DCNII] - k[2179]*y[IDX_H2OII] -
        k[2180]*y[IDX_D2OII] - k[2181]*y[IDX_HDOII] - k[2197]*y[IDX_HII] -
        k[2198]*y[IDX_DII] - k[2219]*y[IDX_HNCII] - k[2220]*y[IDX_DNCII] -
        k[2221]*y[IDX_HNOII] - k[2222]*y[IDX_DNOII] - k[2241]*y[IDX_OII] -
        k[2260]*y[IDX_C2II] - k[2261]*y[IDX_CHII] - k[2262]*y[IDX_CDII] -
        k[2278]*y[IDX_N2II] - k[2293]*y[IDX_O2II] - k[2300]*y[IDX_NH2II] -
        k[2301]*y[IDX_ND2II] - k[2302]*y[IDX_NHDII] - k[2316]*y[IDX_CNII] -
        k[2398]*y[IDX_pH2II] - k[2399]*y[IDX_oH2II] - k[2400]*y[IDX_pD2II] -
        k[2401]*y[IDX_oD2II] - k[2402]*y[IDX_HDII] - k[2520]*y[IDX_NII] -
        k[2536]*y[IDX_OHII] - k[2537]*y[IDX_ODII] - k[2555]*y[IDX_NHII] -
        k[2556]*y[IDX_NDII] - k[2973]*y[IDX_pD3II] - k[2974]*y[IDX_pD3II];
    data[4754] = 0.0 + k[275] + k[417] + k[531]*y[IDX_OI] +
        k[1128]*y[IDX_HI] + k[1129]*y[IDX_DI] + k[1217]*y[IDX_NI] +
        k[1217]*y[IDX_NI];
    data[4755] = 0.0 + k[2842]*y[IDX_eM];
    data[4756] = 0.0 + k[521]*y[IDX_HNOI] + k[522]*y[IDX_DNOI] +
        k[531]*y[IDX_NO2I] + k[534]*y[IDX_OCNI] + k[1230]*y[IDX_NHI] +
        k[1231]*y[IDX_NDI];
    data[4757] = 0.0 + k[1678]*y[IDX_N2OI] - k[2241]*y[IDX_NOI];
    data[4758] = 0.0 + k[2718]*y[IDX_NI];
    data[4759] = 0.0 + k[1205]*y[IDX_NI] + k[1645]*y[IDX_NII];
    data[4760] = 0.0 - k[2293]*y[IDX_NOI];
    data[4761] = 0.0 - k[2039]*y[IDX_NOI];
    data[4762] = 0.0 - k[2038]*y[IDX_NOI];
    data[4763] = 0.0 + k[534]*y[IDX_OI];
    data[4764] = 0.0 + k[578]*y[IDX_HNOI] + k[579]*y[IDX_DNOI] +
        k[1204]*y[IDX_NI] + k[1588]*y[IDX_HNOII] + k[1589]*y[IDX_DNOII];
    data[4765] = 0.0 - k[1960]*y[IDX_NOI] - k[2537]*y[IDX_NOI];
    data[4766] = 0.0 + k[576]*y[IDX_HNOI] + k[577]*y[IDX_DNOI] +
        k[1203]*y[IDX_NI] + k[1586]*y[IDX_HNOII] + k[1587]*y[IDX_DNOII];
    data[4767] = 0.0 - k[1959]*y[IDX_NOI] - k[2536]*y[IDX_NOI];
    data[4768] = 0.0 + k[2826]*y[IDX_HNOII] + k[2827]*y[IDX_DNOII] +
        k[2842]*y[IDX_NO2II];
    data[4769] = 0.0 - k[1453]*y[IDX_NOI];
    data[4770] = 0.0 - k[765]*y[IDX_NOI] - k[2401]*y[IDX_NOI];
    data[4771] = 0.0 - k[1459]*y[IDX_NOI] - k[1462]*y[IDX_NOI];
    data[4772] = 0.0 - k[1452]*y[IDX_NOI];
    data[4773] = 0.0 - k[763]*y[IDX_NOI] - k[2399]*y[IDX_NOI];
    data[4774] = 0.0 - k[1454]*y[IDX_NOI] - k[1457]*y[IDX_NOI];
    data[4775] = 0.0 - k[1449]*y[IDX_NOI];
    data[4776] = 0.0 - k[764]*y[IDX_NOI] - k[2400]*y[IDX_NOI];
    data[4777] = 0.0 - k[1460]*y[IDX_NOI] - k[1461]*y[IDX_NOI] -
        k[1463]*y[IDX_NOI];
    data[4778] = 0.0 - k[2973]*y[IDX_NOI] - k[2974]*y[IDX_NOI];
    data[4779] = 0.0 - k[762]*y[IDX_NOI] - k[2398]*y[IDX_NOI];
    data[4780] = 0.0 - k[1455]*y[IDX_NOI] - k[1456]*y[IDX_NOI] -
        k[1458]*y[IDX_NOI];
    data[4781] = 0.0 - k[1450]*y[IDX_NOI] - k[1451]*y[IDX_NOI];
    data[4782] = 0.0 + k[2065]*y[IDX_NOI];
    data[4783] = 0.0 + k[2260]*y[IDX_NOI];
    data[4784] = 0.0 + k[2163]*y[IDX_NOI];
    data[4785] = 0.0 + k[2162]*y[IDX_NOI];
    data[4786] = 0.0 + k[2262]*y[IDX_NOI];
    data[4787] = 0.0 + k[2165]*y[IDX_NOI];
    data[4788] = 0.0 + k[2261]*y[IDX_NOI];
    data[4789] = 0.0 + k[2164]*y[IDX_NOI];
    data[4790] = 0.0 + k[2166]*y[IDX_NOI];
    data[4791] = 0.0 + k[1655]*y[IDX_OII];
    data[4792] = 0.0 + k[595]*y[IDX_O2I] + k[2316]*y[IDX_NOI];
    data[4793] = 0.0 + k[2064]*y[IDX_NII];
    data[4794] = 0.0 + k[609]*y[IDX_NI] + k[2086]*y[IDX_NOI];
    data[4795] = 0.0 + k[1886]*y[IDX_NHII] + k[1887]*y[IDX_NDII];
    data[4796] = 0.0 + k[2167]*y[IDX_NOI];
    data[4797] = 0.0 + k[1996]*y[IDX_NO2II];
    data[4798] = 0.0 + k[1686]*y[IDX_HNOI] + k[1688]*y[IDX_DNOI] +
        k[2198]*y[IDX_NOI];
    data[4799] = 0.0 + k[2180]*y[IDX_NOI];
    data[4800] = 0.0 + k[1673]*y[IDX_OII];
    data[4801] = 0.0 + k[2176]*y[IDX_NOI];
    data[4802] = 0.0 + k[1565]*y[IDX_O2I] + k[2220]*y[IDX_NOI];
    data[4803] = 0.0 + k[934]*y[IDX_HeII] + k[1687]*y[IDX_HII] +
        k[1688]*y[IDX_DII];
    data[4804] = 0.0 + k[2222]*y[IDX_NOI];
    data[4805] = 0.0 + k[1995]*y[IDX_NO2II];
    data[4806] = 0.0 + k[1685]*y[IDX_HNOI] + k[1687]*y[IDX_DNOI] +
        k[2197]*y[IDX_NOI];
    data[4807] = 0.0 + k[2179]*y[IDX_NOI];
    data[4808] = 0.0 + k[1672]*y[IDX_OII];
    data[4809] = 0.0 + k[2175]*y[IDX_NOI];
    data[4810] = 0.0 + k[2001]*y[IDX_NO2II];
    data[4811] = 0.0 + k[2402]*y[IDX_NOI];
    data[4812] = 0.0 + k[2181]*y[IDX_NOI];
    data[4813] = 0.0 + k[1564]*y[IDX_O2I] + k[2219]*y[IDX_NOI];
    data[4814] = 0.0 + k[933]*y[IDX_HeII] + k[1685]*y[IDX_HII] +
        k[1686]*y[IDX_DII];
    data[4815] = 0.0 + k[2221]*y[IDX_NOI];
    data[4816] = 0.0 + k[933]*y[IDX_HNOI] + k[934]*y[IDX_DNOI] +
        k[938]*y[IDX_N2OI];
    data[4817] = 0.0 + k[609]*y[IDX_COII] + k[1911]*y[IDX_O2II] +
        k[1923]*y[IDX_OHII] + k[1924]*y[IDX_ODII];
    data[4818] = 0.0 + k[1646]*y[IDX_O2I] + k[1647]*y[IDX_OHI] +
        k[1648]*y[IDX_ODI] + k[2064]*y[IDX_COI] + k[2520]*y[IDX_NOI];
    data[4819] = 0.0 + k[1662]*y[IDX_OII];
    data[4820] = 0.0 + k[1805]*y[IDX_OI] + k[2278]*y[IDX_NOI];
    data[4821] = 0.0 + k[938]*y[IDX_HeII] + k[1678]*y[IDX_OII];
    data[4822] = 0.0 + k[1664]*y[IDX_OII];
    data[4823] = 0.0 + k[1879]*y[IDX_O2I] + k[1887]*y[IDX_CO2I] +
        k[2556]*y[IDX_NOI];
    data[4824] = 0.0 + k[2301]*y[IDX_NOI];
    data[4825] = 0.0 + k[1663]*y[IDX_OII];
    data[4826] = 0.0 + k[1878]*y[IDX_O2I] + k[1886]*y[IDX_CO2I] +
        k[2555]*y[IDX_NOI];
    data[4827] = 0.0 + k[2300]*y[IDX_NOI];
    data[4828] = 0.0 + k[2302]*y[IDX_NOI];
    data[4829] = 0.0 + k[238] + k[371] + k[2065]*y[IDX_CII] +
        k[2086]*y[IDX_COII] + k[2162]*y[IDX_C2HII] + k[2163]*y[IDX_C2DII] +
        k[2164]*y[IDX_CH2II] + k[2165]*y[IDX_CD2II] + k[2166]*y[IDX_CHDII] +
        k[2167]*y[IDX_CO2II] + k[2175]*y[IDX_HCNII] + k[2176]*y[IDX_DCNII] +
        k[2179]*y[IDX_H2OII] + k[2180]*y[IDX_D2OII] + k[2181]*y[IDX_HDOII] +
        k[2197]*y[IDX_HII] + k[2198]*y[IDX_DII] + k[2219]*y[IDX_HNCII] +
        k[2220]*y[IDX_DNCII] + k[2221]*y[IDX_HNOII] + k[2222]*y[IDX_DNOII] +
        k[2241]*y[IDX_OII] + k[2260]*y[IDX_C2II] + k[2261]*y[IDX_CHII] +
        k[2262]*y[IDX_CDII] + k[2278]*y[IDX_N2II] + k[2293]*y[IDX_O2II] +
        k[2300]*y[IDX_NH2II] + k[2301]*y[IDX_ND2II] + k[2302]*y[IDX_NHDII] +
        k[2316]*y[IDX_CNII] + k[2398]*y[IDX_pH2II] + k[2399]*y[IDX_oH2II] +
        k[2400]*y[IDX_pD2II] + k[2401]*y[IDX_oD2II] + k[2402]*y[IDX_HDII] +
        k[2520]*y[IDX_NII] + k[2536]*y[IDX_OHII] + k[2537]*y[IDX_ODII] +
        k[2555]*y[IDX_NHII] + k[2556]*y[IDX_NDII];
    data[4830] = 0.0 - k[2761]*y[IDX_eM];
    data[4831] = 0.0 + k[1511]*y[IDX_oH3II] + k[1512]*y[IDX_pH3II] +
        k[1513]*y[IDX_pH3II] + k[1514]*y[IDX_oD3II] + k[1515]*y[IDX_mD3II] +
        k[1516]*y[IDX_oH2DII] + k[1517]*y[IDX_pH2DII] + k[1518]*y[IDX_pH2DII] +
        k[1519]*y[IDX_oH2DII] + k[1520]*y[IDX_pH2DII] + k[1521]*y[IDX_oD2HII] +
        k[1522]*y[IDX_pD2HII] + k[1523]*y[IDX_pD2HII] + k[1524]*y[IDX_oD2HII] +
        k[1525]*y[IDX_pD2HII] + k[2979]*y[IDX_pD3II] + k[2980]*y[IDX_pD3II];
    data[4832] = 0.0 + k[1995]*y[IDX_HI] + k[1996]*y[IDX_DI] +
        k[1997]*y[IDX_pH2I] + k[1998]*y[IDX_oH2I] + k[1999]*y[IDX_pD2I] +
        k[2000]*y[IDX_oD2I] + k[2001]*y[IDX_HDI];
    data[4833] = 0.0 + k[1805]*y[IDX_N2II];
    data[4834] = 0.0 + k[1655]*y[IDX_CNI] + k[1662]*y[IDX_N2I] +
        k[1663]*y[IDX_NHI] + k[1664]*y[IDX_NDI] + k[1672]*y[IDX_HCNI] +
        k[1673]*y[IDX_DCNI] + k[1678]*y[IDX_N2OI] + k[2241]*y[IDX_NOI];
    data[4835] = 0.0 + k[595]*y[IDX_CNII] + k[1564]*y[IDX_HNCII] +
        k[1565]*y[IDX_DNCII] + k[1646]*y[IDX_NII] + k[1878]*y[IDX_NHII] +
        k[1879]*y[IDX_NDII];
    data[4836] = 0.0 + k[1911]*y[IDX_NI] + k[2293]*y[IDX_NOI];
    data[4837] = 0.0 + k[1648]*y[IDX_NII];
    data[4838] = 0.0 + k[1924]*y[IDX_NI] + k[2537]*y[IDX_NOI];
    data[4839] = 0.0 + k[1647]*y[IDX_NII];
    data[4840] = 0.0 + k[1923]*y[IDX_NI] + k[2536]*y[IDX_NOI];
    data[4841] = 0.0 - k[2761]*y[IDX_NOII];
    data[4842] = 0.0 + k[1515]*y[IDX_NO2I];
    data[4843] = 0.0 + k[2000]*y[IDX_NO2II];
    data[4844] = 0.0 + k[2401]*y[IDX_NOI];
    data[4845] = 0.0 + k[1521]*y[IDX_NO2I] + k[1524]*y[IDX_NO2I];
    data[4846] = 0.0 + k[1514]*y[IDX_NO2I];
    data[4847] = 0.0 + k[1998]*y[IDX_NO2II];
    data[4848] = 0.0 + k[2399]*y[IDX_NOI];
    data[4849] = 0.0 + k[1516]*y[IDX_NO2I] + k[1519]*y[IDX_NO2I];
    data[4850] = 0.0 + k[1511]*y[IDX_NO2I];
    data[4851] = 0.0 + k[1999]*y[IDX_NO2II];
    data[4852] = 0.0 + k[2400]*y[IDX_NOI];
    data[4853] = 0.0 + k[1522]*y[IDX_NO2I] + k[1523]*y[IDX_NO2I] +
        k[1525]*y[IDX_NO2I];
    data[4854] = 0.0 + k[2979]*y[IDX_NO2I] + k[2980]*y[IDX_NO2I];
    data[4855] = 0.0 + k[1997]*y[IDX_NO2II];
    data[4856] = 0.0 + k[2398]*y[IDX_NOI];
    data[4857] = 0.0 + k[1517]*y[IDX_NO2I] + k[1518]*y[IDX_NO2I] +
        k[1520]*y[IDX_NO2I];
    data[4858] = 0.0 + k[1512]*y[IDX_NO2I] + k[1513]*y[IDX_NO2I];
    data[4859] = 0.0 - k[1129]*y[IDX_NO2I];
    data[4860] = 0.0 - k[1128]*y[IDX_NO2I];
    data[4861] = 0.0 - k[1216]*y[IDX_NO2I] - k[1217]*y[IDX_NO2I] -
        k[1218]*y[IDX_NO2I];
    data[4862] = 0.0 - k[275] - k[417] - k[531]*y[IDX_OI] -
        k[1128]*y[IDX_HI] - k[1129]*y[IDX_DI] - k[1216]*y[IDX_NI] -
        k[1217]*y[IDX_NI] - k[1218]*y[IDX_NI] - k[1511]*y[IDX_oH3II] -
        k[1512]*y[IDX_pH3II] - k[1513]*y[IDX_pH3II] - k[1514]*y[IDX_oD3II] -
        k[1515]*y[IDX_mD3II] - k[1516]*y[IDX_oH2DII] - k[1517]*y[IDX_pH2DII] -
        k[1518]*y[IDX_pH2DII] - k[1519]*y[IDX_oH2DII] - k[1520]*y[IDX_pH2DII] -
        k[1521]*y[IDX_oD2HII] - k[1522]*y[IDX_pD2HII] - k[1523]*y[IDX_pD2HII] -
        k[1524]*y[IDX_oD2HII] - k[1525]*y[IDX_pD2HII] - k[2252]*y[IDX_OII] -
        k[2297]*y[IDX_O2II] - k[2979]*y[IDX_pD3II] - k[2980]*y[IDX_pD3II];
    data[4863] = 0.0 - k[531]*y[IDX_NO2I];
    data[4864] = 0.0 - k[2252]*y[IDX_NO2I];
    data[4865] = 0.0 - k[2297]*y[IDX_NO2I];
    data[4866] = 0.0 - k[1515]*y[IDX_NO2I];
    data[4867] = 0.0 - k[1521]*y[IDX_NO2I] - k[1524]*y[IDX_NO2I];
    data[4868] = 0.0 - k[1514]*y[IDX_NO2I];
    data[4869] = 0.0 - k[1516]*y[IDX_NO2I] - k[1519]*y[IDX_NO2I];
    data[4870] = 0.0 - k[1511]*y[IDX_NO2I];
    data[4871] = 0.0 - k[1522]*y[IDX_NO2I] - k[1523]*y[IDX_NO2I] -
        k[1525]*y[IDX_NO2I];
    data[4872] = 0.0 - k[2979]*y[IDX_NO2I] - k[2980]*y[IDX_NO2I];
    data[4873] = 0.0 - k[1517]*y[IDX_NO2I] - k[1518]*y[IDX_NO2I] -
        k[1520]*y[IDX_NO2I];
    data[4874] = 0.0 - k[1512]*y[IDX_NO2I] - k[1513]*y[IDX_NO2I];
    data[4875] = 0.0 - k[1996]*y[IDX_NO2II];
    data[4876] = 0.0 + k[1573]*y[IDX_OI];
    data[4877] = 0.0 - k[1995]*y[IDX_NO2II];
    data[4878] = 0.0 - k[2001]*y[IDX_NO2II];
    data[4879] = 0.0 + k[1572]*y[IDX_OI];
    data[4880] = 0.0 + k[2004]*y[IDX_O2HII] + k[2005]*y[IDX_O2DII];
    data[4881] = 0.0 + k[1918]*y[IDX_O2II];
    data[4882] = 0.0 + k[1917]*y[IDX_O2II];
    data[4883] = 0.0 + k[2252]*y[IDX_OII] + k[2297]*y[IDX_O2II];
    data[4884] = 0.0 - k[1995]*y[IDX_HI] - k[1996]*y[IDX_DI] -
        k[1997]*y[IDX_pH2I] - k[1998]*y[IDX_oH2I] - k[1999]*y[IDX_pD2I] -
        k[2000]*y[IDX_oD2I] - k[2001]*y[IDX_HDI] - k[2842]*y[IDX_eM];
    data[4885] = 0.0 + k[1572]*y[IDX_HNOII] + k[1573]*y[IDX_DNOII];
    data[4886] = 0.0 + k[2252]*y[IDX_NO2I];
    data[4887] = 0.0 + k[1917]*y[IDX_NHI] + k[1918]*y[IDX_NDI] +
        k[2297]*y[IDX_NO2I];
    data[4888] = 0.0 + k[2005]*y[IDX_NI];
    data[4889] = 0.0 + k[2004]*y[IDX_NI];
    data[4890] = 0.0 - k[2842]*y[IDX_NO2II];
    data[4891] = 0.0 - k[2000]*y[IDX_NO2II];
    data[4892] = 0.0 - k[1998]*y[IDX_NO2II];
    data[4893] = 0.0 - k[1999]*y[IDX_NO2II];
    data[4894] = 0.0 - k[1997]*y[IDX_NO2II];
    data[4895] = 0.0 + k[1055]*y[IDX_COI] + k[1910]*y[IDX_O2II] +
        k[1921]*y[IDX_OHII] + k[1922]*y[IDX_ODII] + k[2050]*y[IDX_NOI] +
        k[2052]*y[IDX_O2I] - k[2656]*y[IDX_OI];
    data[4896] = 0.0 + k[424]*y[IDX_O2I] - k[2640]*y[IDX_OI];
    data[4897] = 0.0 + k[480]*y[IDX_NOI] + k[2147]*y[IDX_OII] -
        k[2674]*y[IDX_OI];
    data[4898] = 0.0 - k[1226]*y[IDX_OI] + k[1927]*y[IDX_OHII] +
        k[1928]*y[IDX_ODII] + k[2522]*y[IDX_OII];
    data[4899] = 0.0 - k[1690]*y[IDX_OI];
    data[4900] = 0.0 - k[1235]*y[IDX_OI] + k[2569]*y[IDX_OII];
    data[4901] = 0.0 - k[955]*y[IDX_OI];
    data[4902] = 0.0 - k[1234]*y[IDX_OI] + k[2568]*y[IDX_OII];
    data[4903] = 0.0 - k[954]*y[IDX_OI];
    data[4904] = 0.0 - k[1236]*y[IDX_OI];
    data[4905] = 0.0 - k[536]*y[IDX_OI];
    data[4906] = 0.0 + k[247] - k[1237]*y[IDX_OI];
    data[4907] = 0.0 + k[538]*y[IDX_NOI] - k[1228]*y[IDX_OI] +
        k[1914]*y[IDX_O2II] + k[1931]*y[IDX_OHII] + k[1932]*y[IDX_ODII] -
        k[2045]*y[IDX_OI] + k[2565]*y[IDX_OII];
    data[4908] = 0.0 - k[1720]*y[IDX_OI] + k[1756]*y[IDX_O2I];
    data[4909] = 0.0 - k[510]*y[IDX_OI] - k[513]*y[IDX_OI] +
        k[2244]*y[IDX_OII];
    data[4910] = 0.0 - k[973]*y[IDX_OI];
    data[4911] = 0.0 + k[537]*y[IDX_NOI] - k[1227]*y[IDX_OI] +
        k[1913]*y[IDX_O2II] + k[1929]*y[IDX_OHII] + k[1930]*y[IDX_ODII] -
        k[2044]*y[IDX_OI] + k[2564]*y[IDX_OII];
    data[4912] = 0.0 - k[1719]*y[IDX_OI] + k[1755]*y[IDX_O2I];
    data[4913] = 0.0 - k[509]*y[IDX_OI] - k[512]*y[IDX_OI] +
        k[2243]*y[IDX_OII];
    data[4914] = 0.0 - k[972]*y[IDX_OI];
    data[4915] = 0.0 - k[511]*y[IDX_OI] - k[514]*y[IDX_OI] +
        k[2245]*y[IDX_OII];
    data[4916] = 0.0 - k[974]*y[IDX_OI] - k[975]*y[IDX_OI];
    data[4917] = 0.0 + k[548]*y[IDX_O2I] + k[553]*y[IDX_OHI] +
        k[554]*y[IDX_ODI] - k[1229]*y[IDX_OI] + k[1933]*y[IDX_OHII] +
        k[1934]*y[IDX_ODII] + k[2066]*y[IDX_OM];
    data[4918] = 0.0 + k[596]*y[IDX_O2I] - k[2266]*y[IDX_OI];
    data[4919] = 0.0 + k[285] + k[359] + k[867]*y[IDX_HeII] +
        k[1055]*y[IDX_CI] + k[1935]*y[IDX_OHII] + k[1936]*y[IDX_ODII];
    data[4920] = 0.0 + k[614]*y[IDX_OHI] + k[615]*y[IDX_ODI] -
        k[2084]*y[IDX_OI] + k[2744]*y[IDX_eM];
    data[4921] = 0.0 + k[252] + k[392] + k[901]*y[IDX_HeII] +
        k[1542]*y[IDX_HII] + k[1543]*y[IDX_DII];
    data[4922] = 0.0 + k[984]*y[IDX_HI] + k[985]*y[IDX_DI] -
        k[986]*y[IDX_OI] - k[2376]*y[IDX_OI] + k[2787]*y[IDX_eM];
    data[4923] = 0.0 + k[985]*y[IDX_CO2II] + k[1062]*y[IDX_HCOI] +
        k[1063]*y[IDX_DCOI] + k[1080]*y[IDX_OHI] + k[1081]*y[IDX_ODI] +
        k[1097]*y[IDX_NOI] + k[1102]*y[IDX_HNOI] + k[1103]*y[IDX_DNOI] +
        k[1111]*y[IDX_O2I] + k[1122]*y[IDX_O2HI] + k[1123]*y[IDX_O2DI] +
        k[2240]*y[IDX_OII] - k[2663]*y[IDX_OI];
    data[4924] = 0.0 + k[1543]*y[IDX_CO2I] - k[2186]*y[IDX_OI];
    data[4925] = 0.0 + k[2149]*y[IDX_OII] - k[2696]*y[IDX_OI];
    data[4926] = 0.0 + k[2247]*y[IDX_OII] + k[3319]*y[IDX_OHII] +
        k[3320]*y[IDX_ODII];
    data[4927] = 0.0 - k[996]*y[IDX_OI] + k[2789]*y[IDX_eM] +
        k[2796]*y[IDX_eM] + k[3153]*y[IDX_OHI] + k[3156]*y[IDX_ODI];
    data[4928] = 0.0 + k[3345]*y[IDX_OM] + k[3464]*y[IDX_eM] +
        k[3465]*y[IDX_eM];
    data[4929] = 0.0 - k[520]*y[IDX_OI];
    data[4930] = 0.0 - k[2174]*y[IDX_OI];
    data[4931] = 0.0 - k[516]*y[IDX_OI] - k[518]*y[IDX_OI] +
        k[920]*y[IDX_HeII] + k[1061]*y[IDX_HI] + k[1063]*y[IDX_DI] +
        k[2370]*y[IDX_OII];
    data[4932] = 0.0 - k[522]*y[IDX_OI] + k[1101]*y[IDX_HI] +
        k[1103]*y[IDX_DI];
    data[4933] = 0.0 - k[1573]*y[IDX_OI];
    data[4934] = 0.0 + k[2920]*y[IDX_OII];
    data[4935] = 0.0 + k[984]*y[IDX_CO2II] + k[1060]*y[IDX_HCOI] +
        k[1061]*y[IDX_DCOI] + k[1078]*y[IDX_OHI] + k[1079]*y[IDX_ODI] +
        k[1096]*y[IDX_NOI] + k[1100]*y[IDX_HNOI] + k[1101]*y[IDX_DNOI] +
        k[1110]*y[IDX_O2I] + k[1120]*y[IDX_O2HI] + k[1121]*y[IDX_O2DI] +
        k[2239]*y[IDX_OII] - k[2662]*y[IDX_OI];
    data[4936] = 0.0 + k[1542]*y[IDX_CO2I] - k[2185]*y[IDX_OI];
    data[4937] = 0.0 + k[2148]*y[IDX_OII] - k[2695]*y[IDX_OI];
    data[4938] = 0.0 + k[3341]*y[IDX_OM] + k[3342]*y[IDX_OM] +
        k[3458]*y[IDX_eM] + k[3459]*y[IDX_eM] + k[3460]*y[IDX_eM];
    data[4939] = 0.0 + k[2246]*y[IDX_OII] + k[3015]*y[IDX_OHII] +
        k[3316]*y[IDX_ODII];
    data[4940] = 0.0 - k[995]*y[IDX_OI] + k[2788]*y[IDX_eM] +
        k[2795]*y[IDX_eM] + k[3000]*y[IDX_OHI] + k[3154]*y[IDX_ODI];
    data[4941] = 0.0 + k[3018]*y[IDX_OM] + k[3029]*y[IDX_eM] +
        k[3030]*y[IDX_eM];
    data[4942] = 0.0 - k[519]*y[IDX_OI];
    data[4943] = 0.0 - k[2173]*y[IDX_OI];
    data[4944] = 0.0 - k[515]*y[IDX_OI] - k[517]*y[IDX_OI] +
        k[919]*y[IDX_HeII] + k[1060]*y[IDX_HI] + k[1062]*y[IDX_DI] +
        k[2369]*y[IDX_OII];
    data[4945] = 0.0 - k[1156]*y[IDX_OI] - k[1157]*y[IDX_OI];
    data[4946] = 0.0 - k[646]*y[IDX_OI] - k[647]*y[IDX_OI];
    data[4947] = 0.0 + k[3343]*y[IDX_OM] + k[3344]*y[IDX_OM] +
        k[3461]*y[IDX_eM] + k[3462]*y[IDX_eM] + k[3463]*y[IDX_eM];
    data[4948] = 0.0 + k[2248]*y[IDX_OII] + k[3317]*y[IDX_OHII] +
        k[3318]*y[IDX_ODII];
    data[4949] = 0.0 - k[997]*y[IDX_OI] + k[2790]*y[IDX_eM] +
        k[2797]*y[IDX_eM] + k[3152]*y[IDX_OHI] + k[3155]*y[IDX_ODI];
    data[4950] = 0.0 - k[521]*y[IDX_OI] + k[1100]*y[IDX_HI] +
        k[1102]*y[IDX_DI];
    data[4951] = 0.0 - k[1572]*y[IDX_OI];
    data[4952] = 0.0 + k[867]*y[IDX_COI] + k[877]*y[IDX_NOI] +
        k[879]*y[IDX_O2I] + k[901]*y[IDX_CO2I] + k[919]*y[IDX_HCOI] +
        k[920]*y[IDX_DCOI] + k[937]*y[IDX_N2OI] + k[947]*y[IDX_OCNI];
    data[4953] = 0.0 + k[1202]*y[IDX_NOI] + k[1205]*y[IDX_O2I] +
        k[1218]*y[IDX_NO2I] + k[1911]*y[IDX_O2II];
    data[4954] = 0.0 + k[1644]*y[IDX_NOI] + k[1646]*y[IDX_O2I];
    data[4955] = 0.0 + k[1953]*y[IDX_OHII] + k[1954]*y[IDX_ODII];
    data[4956] = 0.0 - k[1805]*y[IDX_OI] - k[2621]*y[IDX_OI];
    data[4957] = 0.0 + k[937]*y[IDX_HeII];
    data[4958] = 0.0 - k[1231]*y[IDX_OI] + k[1916]*y[IDX_O2II] +
        k[1957]*y[IDX_OHII] + k[1958]*y[IDX_ODII] + k[2567]*y[IDX_OII];
    data[4959] = 0.0 - k[1823]*y[IDX_OI] + k[1877]*y[IDX_NOI];
    data[4960] = 0.0 - k[524]*y[IDX_OI] - k[528]*y[IDX_OI] +
        k[2250]*y[IDX_OII];
    data[4961] = 0.0 - k[1974]*y[IDX_OI];
    data[4962] = 0.0 - k[1230]*y[IDX_OI] + k[1915]*y[IDX_O2II] +
        k[1955]*y[IDX_OHII] + k[1956]*y[IDX_ODII] + k[2566]*y[IDX_OII];
    data[4963] = 0.0 - k[1822]*y[IDX_OI] + k[1876]*y[IDX_NOI];
    data[4964] = 0.0 - k[523]*y[IDX_OI] - k[527]*y[IDX_OI] +
        k[2249]*y[IDX_OII];
    data[4965] = 0.0 - k[1973]*y[IDX_OI];
    data[4966] = 0.0 - k[525]*y[IDX_OI] - k[526]*y[IDX_OI] -
        k[529]*y[IDX_OI] - k[530]*y[IDX_OI] + k[2251]*y[IDX_OII];
    data[4967] = 0.0 - k[1975]*y[IDX_OI] - k[1976]*y[IDX_OI];
    data[4968] = 0.0 + k[237] + k[370] + k[480]*y[IDX_CM] +
        k[537]*y[IDX_CHI] + k[538]*y[IDX_CDI] + k[877]*y[IDX_HeII] +
        k[1096]*y[IDX_HI] + k[1097]*y[IDX_DI] + k[1202]*y[IDX_NI] +
        k[1644]*y[IDX_NII] + k[1876]*y[IDX_NHII] + k[1877]*y[IDX_NDII] +
        k[1959]*y[IDX_OHII] + k[1960]*y[IDX_ODII] + k[2050]*y[IDX_CI] +
        k[2241]*y[IDX_OII];
    data[4969] = 0.0 + k[2761]*y[IDX_eM];
    data[4970] = 0.0 + k[275] + k[417] - k[531]*y[IDX_OI] +
        k[1218]*y[IDX_NI] + k[2252]*y[IDX_OII];
    data[4971] = 0.0 + k[2842]*y[IDX_eM];
    data[4972] = 0.0 - k[207] - k[509]*y[IDX_CH2I] - k[510]*y[IDX_CD2I] -
        k[511]*y[IDX_CHDI] - k[512]*y[IDX_CH2I] - k[513]*y[IDX_CD2I] -
        k[514]*y[IDX_CHDI] - k[515]*y[IDX_HCOI] - k[516]*y[IDX_DCOI] -
        k[517]*y[IDX_HCOI] - k[518]*y[IDX_DCOI] - k[519]*y[IDX_HCNI] -
        k[520]*y[IDX_DCNI] - k[521]*y[IDX_HNOI] - k[522]*y[IDX_DNOI] -
        k[523]*y[IDX_NH2I] - k[524]*y[IDX_ND2I] - k[525]*y[IDX_NHDI] -
        k[526]*y[IDX_NHDI] - k[527]*y[IDX_NH2I] - k[528]*y[IDX_ND2I] -
        k[529]*y[IDX_NHDI] - k[530]*y[IDX_NHDI] - k[531]*y[IDX_NO2I] -
        k[532]*y[IDX_O2HI] - k[533]*y[IDX_O2DI] - k[534]*y[IDX_OCNI] -
        k[535]*y[IDX_OCNI] - k[536]*y[IDX_C3I] - k[642]*y[IDX_pH2II] -
        k[643]*y[IDX_oH2II] - k[644]*y[IDX_pD2II] - k[645]*y[IDX_oD2II] -
        k[646]*y[IDX_HDII] - k[647]*y[IDX_HDII] - k[954]*y[IDX_C2HII] -
        k[955]*y[IDX_C2DII] - k[972]*y[IDX_CH2II] - k[973]*y[IDX_CD2II] -
        k[974]*y[IDX_CHDII] - k[975]*y[IDX_CHDII] - k[986]*y[IDX_CO2II] -
        k[995]*y[IDX_H2OII] - k[996]*y[IDX_D2OII] - k[997]*y[IDX_HDOII] -
        k[1152]*y[IDX_pH2I] - k[1153]*y[IDX_oH2I] - k[1154]*y[IDX_pD2I] -
        k[1155]*y[IDX_oD2I] - k[1156]*y[IDX_HDI] - k[1157]*y[IDX_HDI] -
        k[1226]*y[IDX_C2I] - k[1227]*y[IDX_CHI] - k[1228]*y[IDX_CDI] -
        k[1229]*y[IDX_CNI] - k[1230]*y[IDX_NHI] - k[1231]*y[IDX_NDI] -
        k[1232]*y[IDX_OHI] - k[1233]*y[IDX_ODI] - k[1234]*y[IDX_C2HI] -
        k[1235]*y[IDX_C2DI] - k[1236]*y[IDX_C2NI] - k[1237]*y[IDX_CCOI] -
        k[1283]*y[IDX_oH3II] - k[1284]*y[IDX_pH3II] - k[1285]*y[IDX_pH3II] -
        k[1286]*y[IDX_oD3II] - k[1287]*y[IDX_mD3II] - k[1288]*y[IDX_oH2DII] -
        k[1289]*y[IDX_pH2DII] - k[1290]*y[IDX_pH2DII] - k[1291]*y[IDX_oH2DII] -
        k[1292]*y[IDX_pH2DII] - k[1293]*y[IDX_oD2HII] - k[1294]*y[IDX_pD2HII] -
        k[1295]*y[IDX_pD2HII] - k[1296]*y[IDX_oD2HII] - k[1297]*y[IDX_pD2HII] -
        k[1298]*y[IDX_oH3II] - k[1299]*y[IDX_pH3II] - k[1300]*y[IDX_oD3II] -
        k[1301]*y[IDX_mD3II] - k[1302]*y[IDX_oH2DII] - k[1303]*y[IDX_pH2DII] -
        k[1304]*y[IDX_oH2DII] - k[1305]*y[IDX_pH2DII] - k[1306]*y[IDX_oD2HII] -
        k[1307]*y[IDX_pD2HII] - k[1308]*y[IDX_oD2HII] - k[1309]*y[IDX_pD2HII] -
        k[1572]*y[IDX_HNOII] - k[1573]*y[IDX_DNOII] - k[1690]*y[IDX_C2II] -
        k[1719]*y[IDX_CHII] - k[1720]*y[IDX_CDII] - k[1805]*y[IDX_N2II] -
        k[1822]*y[IDX_NHII] - k[1823]*y[IDX_NDII] - k[1925]*y[IDX_OHII] -
        k[1926]*y[IDX_ODII] - k[1973]*y[IDX_NH2II] - k[1974]*y[IDX_ND2II] -
        k[1975]*y[IDX_NHDII] - k[1976]*y[IDX_NHDII] - k[2006]*y[IDX_O2HII] -
        k[2007]*y[IDX_O2DII] - k[2044]*y[IDX_CHI] - k[2045]*y[IDX_CDI] -
        k[2084]*y[IDX_COII] - k[2173]*y[IDX_HCNII] - k[2174]*y[IDX_DCNII] -
        k[2185]*y[IDX_HII] - k[2186]*y[IDX_DII] - k[2266]*y[IDX_CNII] -
        k[2376]*y[IDX_CO2II] - k[2621]*y[IDX_N2II] - k[2640]*y[IDX_CII] -
        k[2656]*y[IDX_CI] - k[2662]*y[IDX_HI] - k[2663]*y[IDX_DI] -
        k[2664]*y[IDX_OI] - k[2664]*y[IDX_OI] - k[2664]*y[IDX_OI] -
        k[2664]*y[IDX_OI] - k[2674]*y[IDX_CM] - k[2695]*y[IDX_HM] -
        k[2696]*y[IDX_DM] - k[2719]*y[IDX_OM] - k[2739]*y[IDX_eM] -
        k[2959]*y[IDX_pD3II] - k[2960]*y[IDX_pD3II];
    data[4973] = 0.0 + k[2147]*y[IDX_CM] + k[2148]*y[IDX_HM] +
        k[2149]*y[IDX_DM] + k[2239]*y[IDX_HI] + k[2240]*y[IDX_DI] +
        k[2241]*y[IDX_NOI] + k[2242]*y[IDX_O2I] + k[2243]*y[IDX_CH2I] +
        k[2244]*y[IDX_CD2I] + k[2245]*y[IDX_CHDI] + k[2246]*y[IDX_H2OI] +
        k[2247]*y[IDX_D2OI] + k[2248]*y[IDX_HDOI] + k[2249]*y[IDX_NH2I] +
        k[2250]*y[IDX_ND2I] + k[2251]*y[IDX_NHDI] + k[2252]*y[IDX_NO2I] +
        k[2369]*y[IDX_HCOI] + k[2370]*y[IDX_DCOI] + k[2386]*y[IDX_OHI] +
        k[2387]*y[IDX_ODI] + k[2522]*y[IDX_C2I] + k[2564]*y[IDX_CHI] +
        k[2565]*y[IDX_CDI] + k[2566]*y[IDX_NHI] + k[2567]*y[IDX_NDI] +
        k[2568]*y[IDX_C2HI] + k[2569]*y[IDX_C2DI] + k[2850]*y[IDX_eM] +
        k[2920]*y[IDX_GRAINM];
    data[4974] = 0.0 + k[347] + k[2066]*y[IDX_CNI] - k[2719]*y[IDX_OI] +
        k[3018]*y[IDX_H3OII] + k[3341]*y[IDX_H2DOII] + k[3342]*y[IDX_H2DOII] +
        k[3343]*y[IDX_HD2OII] + k[3344]*y[IDX_HD2OII] + k[3345]*y[IDX_D3OII];
    data[4975] = 0.0 + k[239] + k[239] + k[372] + k[372] + k[424]*y[IDX_CII]
        + k[548]*y[IDX_CNI] + k[596]*y[IDX_CNII] + k[879]*y[IDX_HeII] +
        k[1110]*y[IDX_HI] + k[1111]*y[IDX_DI] + k[1205]*y[IDX_NI] +
        k[1646]*y[IDX_NII] + k[1755]*y[IDX_CHII] + k[1756]*y[IDX_CDII] +
        k[2052]*y[IDX_CI] + k[2242]*y[IDX_OII];
    data[4976] = 0.0 + k[1910]*y[IDX_CI] + k[1911]*y[IDX_NI] +
        k[1913]*y[IDX_CHI] + k[1914]*y[IDX_CDI] + k[1915]*y[IDX_NHI] +
        k[1916]*y[IDX_NDI] + k[2762]*y[IDX_eM] + k[2762]*y[IDX_eM];
    data[4977] = 0.0 + k[277] - k[533]*y[IDX_OI] + k[1121]*y[IDX_HI] +
        k[1123]*y[IDX_DI];
    data[4978] = 0.0 - k[2007]*y[IDX_OI];
    data[4979] = 0.0 + k[276] - k[532]*y[IDX_OI] + k[1120]*y[IDX_HI] +
        k[1122]*y[IDX_DI];
    data[4980] = 0.0 - k[2006]*y[IDX_OI];
    data[4981] = 0.0 + k[280] + k[418] - k[534]*y[IDX_OI] - k[535]*y[IDX_OI]
        + k[947]*y[IDX_HeII];
    data[4982] = 0.0 + k[242] + k[375] + k[554]*y[IDX_CNI] +
        k[570]*y[IDX_OHI] + k[571]*y[IDX_ODI] + k[571]*y[IDX_ODI] +
        k[615]*y[IDX_COII] + k[1079]*y[IDX_HI] + k[1081]*y[IDX_DI] -
        k[1233]*y[IDX_OI] + k[1963]*y[IDX_OHII] + k[1964]*y[IDX_ODII] +
        k[2387]*y[IDX_OII] + k[3154]*y[IDX_H2OII] + k[3155]*y[IDX_HDOII] +
        k[3156]*y[IDX_D2OII];
    data[4983] = 0.0 + k[297] + k[1922]*y[IDX_CI] - k[1926]*y[IDX_OI] +
        k[1928]*y[IDX_C2I] + k[1930]*y[IDX_CHI] + k[1932]*y[IDX_CDI] +
        k[1934]*y[IDX_CNI] + k[1936]*y[IDX_COI] + k[1954]*y[IDX_N2I] +
        k[1956]*y[IDX_NHI] + k[1958]*y[IDX_NDI] + k[1960]*y[IDX_NOI] +
        k[1962]*y[IDX_OHI] + k[1964]*y[IDX_ODI] + k[2764]*y[IDX_eM] +
        k[3316]*y[IDX_H2OI] + k[3318]*y[IDX_HDOI] + k[3320]*y[IDX_D2OI];
    data[4984] = 0.0 + k[241] + k[374] + k[553]*y[IDX_CNI] +
        k[569]*y[IDX_OHI] + k[569]*y[IDX_OHI] + k[570]*y[IDX_ODI] +
        k[614]*y[IDX_COII] + k[1078]*y[IDX_HI] + k[1080]*y[IDX_DI] -
        k[1232]*y[IDX_OI] + k[1961]*y[IDX_OHII] + k[1962]*y[IDX_ODII] +
        k[2386]*y[IDX_OII] + k[3000]*y[IDX_H2OII] + k[3152]*y[IDX_HDOII] +
        k[3153]*y[IDX_D2OII];
    data[4985] = 0.0 + k[296] + k[1921]*y[IDX_CI] - k[1925]*y[IDX_OI] +
        k[1927]*y[IDX_C2I] + k[1929]*y[IDX_CHI] + k[1931]*y[IDX_CDI] +
        k[1933]*y[IDX_CNI] + k[1935]*y[IDX_COI] + k[1953]*y[IDX_N2I] +
        k[1955]*y[IDX_NHI] + k[1957]*y[IDX_NDI] + k[1959]*y[IDX_NOI] +
        k[1961]*y[IDX_OHI] + k[1963]*y[IDX_ODI] + k[2763]*y[IDX_eM] +
        k[3015]*y[IDX_H2OI] + k[3317]*y[IDX_HDOI] + k[3319]*y[IDX_D2OI];
    data[4986] = 0.0 - k[2739]*y[IDX_OI] + k[2744]*y[IDX_COII] +
        k[2761]*y[IDX_NOII] + k[2762]*y[IDX_O2II] + k[2762]*y[IDX_O2II] +
        k[2763]*y[IDX_OHII] + k[2764]*y[IDX_ODII] + k[2787]*y[IDX_CO2II] +
        k[2788]*y[IDX_H2OII] + k[2789]*y[IDX_D2OII] + k[2790]*y[IDX_HDOII] +
        k[2795]*y[IDX_H2OII] + k[2796]*y[IDX_D2OII] + k[2797]*y[IDX_HDOII] +
        k[2842]*y[IDX_NO2II] + k[2850]*y[IDX_OII] + k[3029]*y[IDX_H3OII] +
        k[3030]*y[IDX_H3OII] + k[3458]*y[IDX_H2DOII] + k[3459]*y[IDX_H2DOII] +
        k[3460]*y[IDX_H2DOII] + k[3461]*y[IDX_HD2OII] + k[3462]*y[IDX_HD2OII] +
        k[3463]*y[IDX_HD2OII] + k[3464]*y[IDX_D3OII] + k[3465]*y[IDX_D3OII];
    data[4987] = 0.0 - k[1287]*y[IDX_OI] - k[1301]*y[IDX_OI];
    data[4988] = 0.0 - k[1155]*y[IDX_OI];
    data[4989] = 0.0 - k[645]*y[IDX_OI];
    data[4990] = 0.0 - k[1293]*y[IDX_OI] - k[1296]*y[IDX_OI] -
        k[1306]*y[IDX_OI] - k[1308]*y[IDX_OI];
    data[4991] = 0.0 - k[1286]*y[IDX_OI] - k[1300]*y[IDX_OI];
    data[4992] = 0.0 - k[1153]*y[IDX_OI];
    data[4993] = 0.0 - k[643]*y[IDX_OI];
    data[4994] = 0.0 - k[1288]*y[IDX_OI] - k[1291]*y[IDX_OI] -
        k[1302]*y[IDX_OI] - k[1304]*y[IDX_OI];
    data[4995] = 0.0 - k[1283]*y[IDX_OI] - k[1298]*y[IDX_OI];
    data[4996] = 0.0 - k[1154]*y[IDX_OI];
    data[4997] = 0.0 - k[644]*y[IDX_OI];
    data[4998] = 0.0 - k[1294]*y[IDX_OI] - k[1295]*y[IDX_OI] -
        k[1297]*y[IDX_OI] - k[1307]*y[IDX_OI] - k[1309]*y[IDX_OI];
    data[4999] = 0.0 - k[2959]*y[IDX_OI] - k[2960]*y[IDX_OI];
    data[5000] = 0.0 - k[1152]*y[IDX_OI];
    data[5001] = 0.0 - k[642]*y[IDX_OI];
    data[5002] = 0.0 - k[1289]*y[IDX_OI] - k[1290]*y[IDX_OI] -
        k[1292]*y[IDX_OI] - k[1303]*y[IDX_OI] - k[1305]*y[IDX_OI];
    data[5003] = 0.0 - k[1284]*y[IDX_OI] - k[1285]*y[IDX_OI] -
        k[1299]*y[IDX_OI];
    data[5004] = 0.0 + k[423]*y[IDX_O2I];
    data[5005] = 0.0 - k[2147]*y[IDX_OII];
    data[5006] = 0.0 - k[1652]*y[IDX_OII] - k[2522]*y[IDX_OII];
    data[5007] = 0.0 - k[1668]*y[IDX_OII] - k[2569]*y[IDX_OII];
    data[5008] = 0.0 - k[1667]*y[IDX_OII] - k[2568]*y[IDX_OII];
    data[5009] = 0.0 - k[1654]*y[IDX_OII] - k[2565]*y[IDX_OII];
    data[5010] = 0.0 + k[1752]*y[IDX_O2I];
    data[5011] = 0.0 - k[2244]*y[IDX_OII];
    data[5012] = 0.0 - k[1653]*y[IDX_OII] - k[2564]*y[IDX_OII];
    data[5013] = 0.0 + k[1751]*y[IDX_O2I];
    data[5014] = 0.0 - k[2243]*y[IDX_OII];
    data[5015] = 0.0 - k[2245]*y[IDX_OII];
    data[5016] = 0.0 - k[1655]*y[IDX_OII];
    data[5017] = 0.0 + k[2266]*y[IDX_OI];
    data[5018] = 0.0 + k[2084]*y[IDX_OI];
    data[5019] = 0.0 + k[900]*y[IDX_HeII] - k[1669]*y[IDX_OII];
    data[5020] = 0.0 + k[2376]*y[IDX_OI];
    data[5021] = 0.0 - k[2240]*y[IDX_OII];
    data[5022] = 0.0 + k[2186]*y[IDX_OI];
    data[5023] = 0.0 - k[2149]*y[IDX_OII];
    data[5024] = 0.0 - k[2247]*y[IDX_OII];
    data[5025] = 0.0 - k[1671]*y[IDX_OII] - k[1673]*y[IDX_OII] -
        k[1675]*y[IDX_OII];
    data[5026] = 0.0 + k[2174]*y[IDX_OI];
    data[5027] = 0.0 - k[1677]*y[IDX_OII] - k[2370]*y[IDX_OII];
    data[5028] = 0.0 - k[2920]*y[IDX_OII];
    data[5029] = 0.0 - k[2239]*y[IDX_OII];
    data[5030] = 0.0 + k[2185]*y[IDX_OI];
    data[5031] = 0.0 - k[2148]*y[IDX_OII];
    data[5032] = 0.0 - k[2246]*y[IDX_OII];
    data[5033] = 0.0 - k[1670]*y[IDX_OII] - k[1672]*y[IDX_OII] -
        k[1674]*y[IDX_OII];
    data[5034] = 0.0 + k[2173]*y[IDX_OI];
    data[5035] = 0.0 - k[1676]*y[IDX_OII] - k[2369]*y[IDX_OII];
    data[5036] = 0.0 - k[1660]*y[IDX_OII] - k[1661]*y[IDX_OII];
    data[5037] = 0.0 - k[2248]*y[IDX_OII];
    data[5038] = 0.0 + k[878]*y[IDX_NOI] + k[879]*y[IDX_O2I] +
        k[880]*y[IDX_OHI] + k[881]*y[IDX_ODI] + k[900]*y[IDX_CO2I] +
        k[936]*y[IDX_N2OI] + k[946]*y[IDX_OCNI];
    data[5039] = 0.0 + k[1645]*y[IDX_O2I];
    data[5040] = 0.0 - k[1662]*y[IDX_OII];
    data[5041] = 0.0 + k[2621]*y[IDX_OI];
    data[5042] = 0.0 + k[936]*y[IDX_HeII] - k[1678]*y[IDX_OII];
    data[5043] = 0.0 - k[1664]*y[IDX_OII] - k[2567]*y[IDX_OII];
    data[5044] = 0.0 - k[2250]*y[IDX_OII];
    data[5045] = 0.0 - k[1663]*y[IDX_OII] - k[2566]*y[IDX_OII];
    data[5046] = 0.0 - k[2249]*y[IDX_OII];
    data[5047] = 0.0 - k[2251]*y[IDX_OII];
    data[5048] = 0.0 + k[878]*y[IDX_HeII] - k[2241]*y[IDX_OII];
    data[5049] = 0.0 - k[2252]*y[IDX_OII];
    data[5050] = 0.0 + k[207] + k[2084]*y[IDX_COII] + k[2173]*y[IDX_HCNII] +
        k[2174]*y[IDX_DCNII] + k[2185]*y[IDX_HII] + k[2186]*y[IDX_DII] +
        k[2266]*y[IDX_CNII] + k[2376]*y[IDX_CO2II] + k[2621]*y[IDX_N2II];
    data[5051] = 0.0 - k[1652]*y[IDX_C2I] - k[1653]*y[IDX_CHI] -
        k[1654]*y[IDX_CDI] - k[1655]*y[IDX_CNI] - k[1656]*y[IDX_pH2I] -
        k[1657]*y[IDX_oH2I] - k[1658]*y[IDX_pD2I] - k[1659]*y[IDX_oD2I] -
        k[1660]*y[IDX_HDI] - k[1661]*y[IDX_HDI] - k[1662]*y[IDX_N2I] -
        k[1663]*y[IDX_NHI] - k[1664]*y[IDX_NDI] - k[1665]*y[IDX_OHI] -
        k[1666]*y[IDX_ODI] - k[1667]*y[IDX_C2HI] - k[1668]*y[IDX_C2DI] -
        k[1669]*y[IDX_CO2I] - k[1670]*y[IDX_HCNI] - k[1671]*y[IDX_DCNI] -
        k[1672]*y[IDX_HCNI] - k[1673]*y[IDX_DCNI] - k[1674]*y[IDX_HCNI] -
        k[1675]*y[IDX_DCNI] - k[1676]*y[IDX_HCOI] - k[1677]*y[IDX_DCOI] -
        k[1678]*y[IDX_N2OI] - k[2147]*y[IDX_CM] - k[2148]*y[IDX_HM] -
        k[2149]*y[IDX_DM] - k[2239]*y[IDX_HI] - k[2240]*y[IDX_DI] -
        k[2241]*y[IDX_NOI] - k[2242]*y[IDX_O2I] - k[2243]*y[IDX_CH2I] -
        k[2244]*y[IDX_CD2I] - k[2245]*y[IDX_CHDI] - k[2246]*y[IDX_H2OI] -
        k[2247]*y[IDX_D2OI] - k[2248]*y[IDX_HDOI] - k[2249]*y[IDX_NH2I] -
        k[2250]*y[IDX_ND2I] - k[2251]*y[IDX_NHDI] - k[2252]*y[IDX_NO2I] -
        k[2369]*y[IDX_HCOI] - k[2370]*y[IDX_DCOI] - k[2386]*y[IDX_OHI] -
        k[2387]*y[IDX_ODI] - k[2522]*y[IDX_C2I] - k[2564]*y[IDX_CHI] -
        k[2565]*y[IDX_CDI] - k[2566]*y[IDX_NHI] - k[2567]*y[IDX_NDI] -
        k[2568]*y[IDX_C2HI] - k[2569]*y[IDX_C2DI] - k[2850]*y[IDX_eM] -
        k[2920]*y[IDX_GRAINM];
    data[5052] = 0.0 + k[423]*y[IDX_CII] + k[879]*y[IDX_HeII] +
        k[1645]*y[IDX_NII] + k[1751]*y[IDX_CHII] + k[1752]*y[IDX_CDII] -
        k[2242]*y[IDX_OII];
    data[5053] = 0.0 + k[946]*y[IDX_HeII];
    data[5054] = 0.0 + k[881]*y[IDX_HeII] - k[1666]*y[IDX_OII] -
        k[2387]*y[IDX_OII];
    data[5055] = 0.0 + k[880]*y[IDX_HeII] - k[1665]*y[IDX_OII] -
        k[2386]*y[IDX_OII];
    data[5056] = 0.0 - k[2850]*y[IDX_OII];
    data[5057] = 0.0 - k[1659]*y[IDX_OII];
    data[5058] = 0.0 - k[1657]*y[IDX_OII];
    data[5059] = 0.0 - k[1658]*y[IDX_OII];
    data[5060] = 0.0 - k[1656]*y[IDX_OII];
    data[5061] = 0.0 - k[2715]*y[IDX_OM];
    data[5062] = 0.0 + k[481]*y[IDX_O2I];
    data[5063] = 0.0 - k[2721]*y[IDX_OM];
    data[5064] = 0.0 - k[2720]*y[IDX_OM];
    data[5065] = 0.0 - k[2066]*y[IDX_OM];
    data[5066] = 0.0 - k[2722]*y[IDX_OM];
    data[5067] = 0.0 - k[2717]*y[IDX_OM];
    data[5068] = 0.0 - k[3345]*y[IDX_OM];
    data[5069] = 0.0 - k[504]*y[IDX_OM];
    data[5070] = 0.0 - k[2716]*y[IDX_OM];
    data[5071] = 0.0 - k[3341]*y[IDX_OM] - k[3342]*y[IDX_OM];
    data[5072] = 0.0 - k[3018]*y[IDX_OM];
    data[5073] = 0.0 - k[503]*y[IDX_OM];
    data[5074] = 0.0 - k[501]*y[IDX_OM] - k[502]*y[IDX_OM] -
        k[2727]*y[IDX_OM];
    data[5075] = 0.0 - k[3343]*y[IDX_OM] - k[3344]*y[IDX_OM];
    data[5076] = 0.0 - k[2718]*y[IDX_OM];
    data[5077] = 0.0 - k[2719]*y[IDX_OM] + k[2739]*y[IDX_eM];
    data[5078] = 0.0 - k[347] - k[497]*y[IDX_pH2I] - k[498]*y[IDX_oH2I] -
        k[499]*y[IDX_pD2I] - k[500]*y[IDX_oD2I] - k[501]*y[IDX_HDI] -
        k[502]*y[IDX_HDI] - k[503]*y[IDX_HCNI] - k[504]*y[IDX_DCNI] -
        k[2066]*y[IDX_CNI] - k[2715]*y[IDX_CI] - k[2716]*y[IDX_HI] -
        k[2717]*y[IDX_DI] - k[2718]*y[IDX_NI] - k[2719]*y[IDX_OI] -
        k[2720]*y[IDX_CHI] - k[2721]*y[IDX_CDI] - k[2722]*y[IDX_COI] -
        k[2723]*y[IDX_pH2I] - k[2724]*y[IDX_oH2I] - k[2725]*y[IDX_pD2I] -
        k[2726]*y[IDX_oD2I] - k[2727]*y[IDX_HDI] - k[3018]*y[IDX_H3OII] -
        k[3341]*y[IDX_H2DOII] - k[3342]*y[IDX_H2DOII] - k[3343]*y[IDX_HD2OII] -
        k[3344]*y[IDX_HD2OII] - k[3345]*y[IDX_D3OII];
    data[5079] = 0.0 + k[481]*y[IDX_CM];
    data[5080] = 0.0 + k[2739]*y[IDX_OI];
    data[5081] = 0.0 - k[500]*y[IDX_OM] - k[2726]*y[IDX_OM];
    data[5082] = 0.0 - k[498]*y[IDX_OM] - k[2724]*y[IDX_OM];
    data[5083] = 0.0 - k[499]*y[IDX_OM] - k[2725]*y[IDX_OM];
    data[5084] = 0.0 - k[497]*y[IDX_OM] - k[2723]*y[IDX_OM];
    data[5085] = 0.0 + k[2002]*y[IDX_O2HII] + k[2003]*y[IDX_O2DII] -
        k[2052]*y[IDX_O2I] + k[2557]*y[IDX_O2II];
    data[5086] = 0.0 - k[423]*y[IDX_O2I] - k[424]*y[IDX_O2I];
    data[5087] = 0.0 - k[481]*y[IDX_O2I] - k[2684]*y[IDX_O2I];
    data[5088] = 0.0 + k[2008]*y[IDX_O2HII] + k[2009]*y[IDX_O2DII];
    data[5089] = 0.0 - k[1704]*y[IDX_O2I];
    data[5090] = 0.0 - k[1033]*y[IDX_O2I];
    data[5091] = 0.0 - k[1032]*y[IDX_O2I];
    data[5092] = 0.0 - k[540]*y[IDX_O2I] + k[2012]*y[IDX_O2HII] +
        k[2013]*y[IDX_O2DII] + k[2632]*y[IDX_O2II];
    data[5093] = 0.0 - k[1752]*y[IDX_O2I] - k[1754]*y[IDX_O2I] -
        k[1756]*y[IDX_O2I];
    data[5094] = 0.0 + k[2634]*y[IDX_O2II];
    data[5095] = 0.0 - k[977]*y[IDX_O2I];
    data[5096] = 0.0 - k[539]*y[IDX_O2I] + k[2010]*y[IDX_O2HII] +
        k[2011]*y[IDX_O2DII] + k[2631]*y[IDX_O2II];
    data[5097] = 0.0 - k[1751]*y[IDX_O2I] - k[1753]*y[IDX_O2I] -
        k[1755]*y[IDX_O2I];
    data[5098] = 0.0 + k[2633]*y[IDX_O2II];
    data[5099] = 0.0 - k[976]*y[IDX_O2I];
    data[5100] = 0.0 + k[2635]*y[IDX_O2II];
    data[5101] = 0.0 - k[978]*y[IDX_O2I] - k[979]*y[IDX_O2I];
    data[5102] = 0.0 - k[548]*y[IDX_O2I] + k[2014]*y[IDX_O2HII] +
        k[2015]*y[IDX_O2DII];
    data[5103] = 0.0 - k[595]*y[IDX_O2I] - k[596]*y[IDX_O2I] -
        k[2317]*y[IDX_O2I];
    data[5104] = 0.0 + k[2016]*y[IDX_O2HII] + k[2017]*y[IDX_O2DII];
    data[5105] = 0.0 - k[2087]*y[IDX_O2I];
    data[5106] = 0.0 + k[899]*y[IDX_HeII];
    data[5107] = 0.0 - k[2168]*y[IDX_O2I];
    data[5108] = 0.0 - k[1111]*y[IDX_O2I] + k[1114]*y[IDX_O2HI] +
        k[1115]*y[IDX_O2DI];
    data[5109] = 0.0 - k[2200]*y[IDX_O2I];
    data[5110] = 0.0 - k[2183]*y[IDX_O2I];
    data[5111] = 0.0 - k[2178]*y[IDX_O2I];
    data[5112] = 0.0 + k[2637]*y[IDX_O2II];
    data[5113] = 0.0 - k[1565]*y[IDX_O2I];
    data[5114] = 0.0 - k[1110]*y[IDX_O2I] + k[1112]*y[IDX_O2HI] +
        k[1113]*y[IDX_O2DI];
    data[5115] = 0.0 - k[2199]*y[IDX_O2I];
    data[5116] = 0.0 - k[2182]*y[IDX_O2I];
    data[5117] = 0.0 - k[2177]*y[IDX_O2I];
    data[5118] = 0.0 + k[2636]*y[IDX_O2II];
    data[5119] = 0.0 + k[2028]*y[IDX_O2HII] + k[2029]*y[IDX_O2HII] +
        k[2030]*y[IDX_O2DII] + k[2031]*y[IDX_O2DII];
    data[5120] = 0.0 - k[772]*y[IDX_O2I] - k[773]*y[IDX_O2I] -
        k[2407]*y[IDX_O2I];
    data[5121] = 0.0 - k[2184]*y[IDX_O2I];
    data[5122] = 0.0 - k[1564]*y[IDX_O2I];
    data[5123] = 0.0 - k[879]*y[IDX_O2I] + k[899]*y[IDX_CO2I] -
        k[2477]*y[IDX_O2I];
    data[5124] = 0.0 - k[1205]*y[IDX_O2I] + k[1216]*y[IDX_NO2I] +
        k[1219]*y[IDX_O2HI] + k[1220]*y[IDX_O2DI];
    data[5125] = 0.0 - k[1645]*y[IDX_O2I] - k[1646]*y[IDX_O2I] -
        k[2533]*y[IDX_O2I];
    data[5126] = 0.0 + k[2032]*y[IDX_O2HII] + k[2033]*y[IDX_O2DII];
    data[5127] = 0.0 - k[2279]*y[IDX_O2I];
    data[5128] = 0.0 + k[2036]*y[IDX_O2HII] + k[2037]*y[IDX_O2DII];
    data[5129] = 0.0 - k[1879]*y[IDX_O2I] - k[1881]*y[IDX_O2I] -
        k[2559]*y[IDX_O2I];
    data[5130] = 0.0 + k[2295]*y[IDX_O2II];
    data[5131] = 0.0 - k[1992]*y[IDX_O2I];
    data[5132] = 0.0 + k[2034]*y[IDX_O2HII] + k[2035]*y[IDX_O2DII];
    data[5133] = 0.0 - k[1878]*y[IDX_O2I] - k[1880]*y[IDX_O2I] -
        k[2558]*y[IDX_O2I];
    data[5134] = 0.0 + k[2294]*y[IDX_O2II];
    data[5135] = 0.0 - k[1991]*y[IDX_O2I];
    data[5136] = 0.0 + k[2296]*y[IDX_O2II];
    data[5137] = 0.0 - k[1993]*y[IDX_O2I] - k[1994]*y[IDX_O2I];
    data[5138] = 0.0 + k[2038]*y[IDX_O2HII] + k[2039]*y[IDX_O2DII] +
        k[2293]*y[IDX_O2II];
    data[5139] = 0.0 + k[531]*y[IDX_OI] + k[1216]*y[IDX_NI] +
        k[2297]*y[IDX_O2II];
    data[5140] = 0.0 + k[531]*y[IDX_NO2I] + k[532]*y[IDX_O2HI] +
        k[533]*y[IDX_O2DI] + k[535]*y[IDX_OCNI] + k[1232]*y[IDX_OHI] +
        k[1233]*y[IDX_ODI] + k[2006]*y[IDX_O2HII] + k[2007]*y[IDX_O2DII] +
        k[2664]*y[IDX_OI] + k[2664]*y[IDX_OI] + k[2719]*y[IDX_OM];
    data[5141] = 0.0 - k[2242]*y[IDX_O2I];
    data[5142] = 0.0 + k[2719]*y[IDX_OI];
    data[5143] = 0.0 - k[239] - k[240] - k[372] - k[373] - k[423]*y[IDX_CII]
        - k[424]*y[IDX_CII] - k[481]*y[IDX_CM] - k[539]*y[IDX_CHI] -
        k[540]*y[IDX_CDI] - k[548]*y[IDX_CNI] - k[595]*y[IDX_CNII] -
        k[596]*y[IDX_CNII] - k[768]*y[IDX_pH2II] - k[769]*y[IDX_oH2II] -
        k[770]*y[IDX_pD2II] - k[771]*y[IDX_oD2II] - k[772]*y[IDX_HDII] -
        k[773]*y[IDX_HDII] - k[879]*y[IDX_HeII] - k[976]*y[IDX_CH2II] -
        k[977]*y[IDX_CD2II] - k[978]*y[IDX_CHDII] - k[979]*y[IDX_CHDII] -
        k[1032]*y[IDX_C2HI] - k[1033]*y[IDX_C2DI] - k[1110]*y[IDX_HI] -
        k[1111]*y[IDX_DI] - k[1205]*y[IDX_NI] - k[1464]*y[IDX_oH3II] -
        k[1465]*y[IDX_pH3II] - k[1466]*y[IDX_pH3II] - k[1467]*y[IDX_oD3II] -
        k[1468]*y[IDX_mD3II] - k[1469]*y[IDX_oH2DII] - k[1470]*y[IDX_pH2DII] -
        k[1471]*y[IDX_pH2DII] - k[1472]*y[IDX_oH2DII] - k[1473]*y[IDX_pH2DII] -
        k[1474]*y[IDX_oD2HII] - k[1475]*y[IDX_pD2HII] - k[1476]*y[IDX_pD2HII] -
        k[1477]*y[IDX_oD2HII] - k[1478]*y[IDX_pD2HII] - k[1564]*y[IDX_HNCII] -
        k[1565]*y[IDX_DNCII] - k[1645]*y[IDX_NII] - k[1646]*y[IDX_NII] -
        k[1704]*y[IDX_C2II] - k[1751]*y[IDX_CHII] - k[1752]*y[IDX_CDII] -
        k[1753]*y[IDX_CHII] - k[1754]*y[IDX_CDII] - k[1755]*y[IDX_CHII] -
        k[1756]*y[IDX_CDII] - k[1878]*y[IDX_NHII] - k[1879]*y[IDX_NDII] -
        k[1880]*y[IDX_NHII] - k[1881]*y[IDX_NDII] - k[1991]*y[IDX_NH2II] -
        k[1992]*y[IDX_ND2II] - k[1993]*y[IDX_NHDII] - k[1994]*y[IDX_NHDII] -
        k[2052]*y[IDX_CI] - k[2087]*y[IDX_COII] - k[2168]*y[IDX_CO2II] -
        k[2177]*y[IDX_HCNII] - k[2178]*y[IDX_DCNII] - k[2182]*y[IDX_H2OII] -
        k[2183]*y[IDX_D2OII] - k[2184]*y[IDX_HDOII] - k[2199]*y[IDX_HII] -
        k[2200]*y[IDX_DII] - k[2242]*y[IDX_OII] - k[2279]*y[IDX_N2II] -
        k[2298]*y[IDX_OHII] - k[2299]*y[IDX_ODII] - k[2317]*y[IDX_CNII] -
        k[2403]*y[IDX_pH2II] - k[2404]*y[IDX_oH2II] - k[2405]*y[IDX_pD2II] -
        k[2406]*y[IDX_oD2II] - k[2407]*y[IDX_HDII] - k[2477]*y[IDX_HeII] -
        k[2533]*y[IDX_NII] - k[2558]*y[IDX_NHII] - k[2559]*y[IDX_NDII] -
        k[2684]*y[IDX_CM] - k[2975]*y[IDX_pD3II] - k[2976]*y[IDX_pD3II];
    data[5144] = 0.0 + k[2293]*y[IDX_NOI] + k[2294]*y[IDX_NH2I] +
        k[2295]*y[IDX_ND2I] + k[2296]*y[IDX_NHDI] + k[2297]*y[IDX_NO2I] +
        k[2557]*y[IDX_CI] + k[2631]*y[IDX_CHI] + k[2632]*y[IDX_CDI] +
        k[2633]*y[IDX_CH2I] + k[2634]*y[IDX_CD2I] + k[2635]*y[IDX_CHDI] +
        k[2636]*y[IDX_HCOI] + k[2637]*y[IDX_DCOI];
    data[5145] = 0.0 + k[279] + k[533]*y[IDX_OI] + k[591]*y[IDX_OHI] +
        k[593]*y[IDX_ODI] + k[1113]*y[IDX_HI] + k[1115]*y[IDX_DI] +
        k[1220]*y[IDX_NI];
    data[5146] = 0.0 + k[2003]*y[IDX_CI] + k[2007]*y[IDX_OI] +
        k[2009]*y[IDX_C2I] + k[2011]*y[IDX_CHI] + k[2013]*y[IDX_CDI] +
        k[2015]*y[IDX_CNI] + k[2017]*y[IDX_COI] + k[2019]*y[IDX_pH2I] +
        k[2020]*y[IDX_oH2I] + k[2021]*y[IDX_oH2I] + k[2025]*y[IDX_pD2I] +
        k[2026]*y[IDX_oD2I] + k[2027]*y[IDX_oD2I] + k[2030]*y[IDX_HDI] +
        k[2031]*y[IDX_HDI] + k[2033]*y[IDX_N2I] + k[2035]*y[IDX_NHI] +
        k[2037]*y[IDX_NDI] + k[2039]*y[IDX_NOI] + k[2041]*y[IDX_OHI] +
        k[2043]*y[IDX_ODI] + k[2844]*y[IDX_eM] + k[2981]*y[IDX_oD2I] +
        k[2982]*y[IDX_pD2I];
    data[5147] = 0.0 + k[278] + k[532]*y[IDX_OI] + k[590]*y[IDX_OHI] +
        k[592]*y[IDX_ODI] + k[1112]*y[IDX_HI] + k[1114]*y[IDX_DI] +
        k[1219]*y[IDX_NI];
    data[5148] = 0.0 + k[2002]*y[IDX_CI] + k[2006]*y[IDX_OI] +
        k[2008]*y[IDX_C2I] + k[2010]*y[IDX_CHI] + k[2012]*y[IDX_CDI] +
        k[2014]*y[IDX_CNI] + k[2016]*y[IDX_COI] + k[2018]*y[IDX_oH2I] +
        k[2022]*y[IDX_pD2I] + k[2023]*y[IDX_oD2I] + k[2024]*y[IDX_oD2I] +
        k[2028]*y[IDX_HDI] + k[2029]*y[IDX_HDI] + k[2032]*y[IDX_N2I] +
        k[2034]*y[IDX_NHI] + k[2036]*y[IDX_NDI] + k[2038]*y[IDX_NOI] +
        k[2040]*y[IDX_OHI] + k[2042]*y[IDX_ODI] + k[2843]*y[IDX_eM] +
        k[2927]*y[IDX_oH2I] + k[2928]*y[IDX_pH2I];
    data[5149] = 0.0 + k[535]*y[IDX_OI];
    data[5150] = 0.0 + k[592]*y[IDX_O2HI] + k[593]*y[IDX_O2DI] +
        k[1233]*y[IDX_OI] + k[2042]*y[IDX_O2HII] + k[2043]*y[IDX_O2DII];
    data[5151] = 0.0 - k[2299]*y[IDX_O2I];
    data[5152] = 0.0 + k[590]*y[IDX_O2HI] + k[591]*y[IDX_O2DI] +
        k[1232]*y[IDX_OI] + k[2040]*y[IDX_O2HII] + k[2041]*y[IDX_O2DII];
    data[5153] = 0.0 - k[2298]*y[IDX_O2I];
    data[5154] = 0.0 + k[2843]*y[IDX_O2HII] + k[2844]*y[IDX_O2DII];
    data[5155] = 0.0 - k[1468]*y[IDX_O2I];
    data[5156] = 0.0 + k[2023]*y[IDX_O2HII] + k[2024]*y[IDX_O2HII] +
        k[2026]*y[IDX_O2DII] + k[2027]*y[IDX_O2DII] + k[2981]*y[IDX_O2DII];
    data[5157] = 0.0 - k[771]*y[IDX_O2I] - k[2406]*y[IDX_O2I];
    data[5158] = 0.0 - k[1474]*y[IDX_O2I] - k[1477]*y[IDX_O2I];
    data[5159] = 0.0 - k[1467]*y[IDX_O2I];
    data[5160] = 0.0 + k[2018]*y[IDX_O2HII] + k[2020]*y[IDX_O2DII] +
        k[2021]*y[IDX_O2DII] + k[2927]*y[IDX_O2HII];
    data[5161] = 0.0 - k[769]*y[IDX_O2I] - k[2404]*y[IDX_O2I];
    data[5162] = 0.0 - k[1469]*y[IDX_O2I] - k[1472]*y[IDX_O2I];
    data[5163] = 0.0 - k[1464]*y[IDX_O2I];
    data[5164] = 0.0 + k[2022]*y[IDX_O2HII] + k[2025]*y[IDX_O2DII] +
        k[2982]*y[IDX_O2DII];
    data[5165] = 0.0 - k[770]*y[IDX_O2I] - k[2405]*y[IDX_O2I];
    data[5166] = 0.0 - k[1475]*y[IDX_O2I] - k[1476]*y[IDX_O2I] -
        k[1478]*y[IDX_O2I];
    data[5167] = 0.0 - k[2975]*y[IDX_O2I] - k[2976]*y[IDX_O2I];
    data[5168] = 0.0 + k[2019]*y[IDX_O2DII] + k[2928]*y[IDX_O2HII];
    data[5169] = 0.0 - k[768]*y[IDX_O2I] - k[2403]*y[IDX_O2I];
    data[5170] = 0.0 - k[1470]*y[IDX_O2I] - k[1471]*y[IDX_O2I] -
        k[1473]*y[IDX_O2I];
    data[5171] = 0.0 - k[1465]*y[IDX_O2I] - k[1466]*y[IDX_O2I];
    data[5172] = 0.0 - k[1910]*y[IDX_O2II] - k[2557]*y[IDX_O2II];
    data[5173] = 0.0 - k[1912]*y[IDX_O2II];
    data[5174] = 0.0 - k[1914]*y[IDX_O2II] - k[2632]*y[IDX_O2II];
    data[5175] = 0.0 - k[2634]*y[IDX_O2II];
    data[5176] = 0.0 - k[1913]*y[IDX_O2II] - k[2631]*y[IDX_O2II];
    data[5177] = 0.0 - k[2633]*y[IDX_O2II];
    data[5178] = 0.0 - k[2635]*y[IDX_O2II];
    data[5179] = 0.0 + k[2317]*y[IDX_O2I];
    data[5180] = 0.0 + k[2087]*y[IDX_O2I];
    data[5181] = 0.0 + k[902]*y[IDX_HeII] + k[1669]*y[IDX_OII];
    data[5182] = 0.0 + k[986]*y[IDX_OI] + k[2168]*y[IDX_O2I];
    data[5183] = 0.0 + k[2200]*y[IDX_O2I];
    data[5184] = 0.0 + k[996]*y[IDX_OI] + k[2183]*y[IDX_O2I];
    data[5185] = 0.0 + k[2178]*y[IDX_O2I];
    data[5186] = 0.0 - k[1920]*y[IDX_O2II] - k[2637]*y[IDX_O2II];
    data[5187] = 0.0 + k[2199]*y[IDX_O2I];
    data[5188] = 0.0 + k[995]*y[IDX_OI] + k[2182]*y[IDX_O2I];
    data[5189] = 0.0 + k[2177]*y[IDX_O2I];
    data[5190] = 0.0 - k[1919]*y[IDX_O2II] - k[2636]*y[IDX_O2II];
    data[5191] = 0.0 + k[2407]*y[IDX_O2I];
    data[5192] = 0.0 + k[997]*y[IDX_OI] + k[2184]*y[IDX_O2I];
    data[5193] = 0.0 + k[902]*y[IDX_CO2I] + k[2477]*y[IDX_O2I];
    data[5194] = 0.0 - k[1911]*y[IDX_O2II];
    data[5195] = 0.0 + k[2533]*y[IDX_O2I];
    data[5196] = 0.0 + k[2279]*y[IDX_O2I];
    data[5197] = 0.0 - k[1916]*y[IDX_O2II] - k[1918]*y[IDX_O2II];
    data[5198] = 0.0 + k[2559]*y[IDX_O2I];
    data[5199] = 0.0 - k[2295]*y[IDX_O2II];
    data[5200] = 0.0 - k[1915]*y[IDX_O2II] - k[1917]*y[IDX_O2II];
    data[5201] = 0.0 + k[2558]*y[IDX_O2I];
    data[5202] = 0.0 - k[2294]*y[IDX_O2II];
    data[5203] = 0.0 - k[2296]*y[IDX_O2II];
    data[5204] = 0.0 - k[2293]*y[IDX_O2II];
    data[5205] = 0.0 - k[2297]*y[IDX_O2II];
    data[5206] = 0.0 + k[986]*y[IDX_CO2II] + k[995]*y[IDX_H2OII] +
        k[996]*y[IDX_D2OII] + k[997]*y[IDX_HDOII] + k[1925]*y[IDX_OHII] +
        k[1926]*y[IDX_ODII];
    data[5207] = 0.0 + k[1665]*y[IDX_OHI] + k[1666]*y[IDX_ODI] +
        k[1669]*y[IDX_CO2I] + k[2242]*y[IDX_O2I];
    data[5208] = 0.0 + k[240] + k[373] + k[2087]*y[IDX_COII] +
        k[2168]*y[IDX_CO2II] + k[2177]*y[IDX_HCNII] + k[2178]*y[IDX_DCNII] +
        k[2182]*y[IDX_H2OII] + k[2183]*y[IDX_D2OII] + k[2184]*y[IDX_HDOII] +
        k[2199]*y[IDX_HII] + k[2200]*y[IDX_DII] + k[2242]*y[IDX_OII] +
        k[2279]*y[IDX_N2II] + k[2298]*y[IDX_OHII] + k[2299]*y[IDX_ODII] +
        k[2317]*y[IDX_CNII] + k[2403]*y[IDX_pH2II] + k[2404]*y[IDX_oH2II] +
        k[2405]*y[IDX_pD2II] + k[2406]*y[IDX_oD2II] + k[2407]*y[IDX_HDII] +
        k[2477]*y[IDX_HeII] + k[2533]*y[IDX_NII] + k[2558]*y[IDX_NHII] +
        k[2559]*y[IDX_NDII];
    data[5209] = 0.0 - k[1910]*y[IDX_CI] - k[1911]*y[IDX_NI] -
        k[1912]*y[IDX_C2I] - k[1913]*y[IDX_CHI] - k[1914]*y[IDX_CDI] -
        k[1915]*y[IDX_NHI] - k[1916]*y[IDX_NDI] - k[1917]*y[IDX_NHI] -
        k[1918]*y[IDX_NDI] - k[1919]*y[IDX_HCOI] - k[1920]*y[IDX_DCOI] -
        k[2293]*y[IDX_NOI] - k[2294]*y[IDX_NH2I] - k[2295]*y[IDX_ND2I] -
        k[2296]*y[IDX_NHDI] - k[2297]*y[IDX_NO2I] - k[2557]*y[IDX_CI] -
        k[2631]*y[IDX_CHI] - k[2632]*y[IDX_CDI] - k[2633]*y[IDX_CH2I] -
        k[2634]*y[IDX_CD2I] - k[2635]*y[IDX_CHDI] - k[2636]*y[IDX_HCOI] -
        k[2637]*y[IDX_DCOI] - k[2762]*y[IDX_eM];
    data[5210] = 0.0 + k[1666]*y[IDX_OII];
    data[5211] = 0.0 + k[1926]*y[IDX_OI] + k[2299]*y[IDX_O2I];
    data[5212] = 0.0 + k[1665]*y[IDX_OII];
    data[5213] = 0.0 + k[1925]*y[IDX_OI] + k[2298]*y[IDX_O2I];
    data[5214] = 0.0 - k[2762]*y[IDX_O2II];
    data[5215] = 0.0 + k[2406]*y[IDX_O2I];
    data[5216] = 0.0 + k[2404]*y[IDX_O2I];
    data[5217] = 0.0 + k[2405]*y[IDX_O2I];
    data[5218] = 0.0 + k[2403]*y[IDX_O2I];
    data[5219] = 0.0 - k[1115]*y[IDX_O2DI] - k[1119]*y[IDX_O2DI] -
        k[1123]*y[IDX_O2DI];
    data[5220] = 0.0 - k[1113]*y[IDX_O2DI] - k[1117]*y[IDX_O2DI] -
        k[1121]*y[IDX_O2DI];
    data[5221] = 0.0 - k[1220]*y[IDX_O2DI];
    data[5222] = 0.0 - k[533]*y[IDX_O2DI];
    data[5223] = 0.0 - k[277] - k[279] - k[533]*y[IDX_OI] -
        k[591]*y[IDX_OHI] - k[593]*y[IDX_ODI] - k[1113]*y[IDX_HI] -
        k[1115]*y[IDX_DI] - k[1117]*y[IDX_HI] - k[1119]*y[IDX_DI] -
        k[1121]*y[IDX_HI] - k[1123]*y[IDX_DI] - k[1220]*y[IDX_NI];
    data[5224] = 0.0 - k[593]*y[IDX_O2DI];
    data[5225] = 0.0 - k[591]*y[IDX_O2DI];
    data[5226] = 0.0 - k[2003]*y[IDX_O2DII];
    data[5227] = 0.0 - k[2009]*y[IDX_O2DII];
    data[5228] = 0.0 - k[2013]*y[IDX_O2DII];
    data[5229] = 0.0 - k[2011]*y[IDX_O2DII];
    data[5230] = 0.0 - k[2015]*y[IDX_O2DII];
    data[5231] = 0.0 - k[2017]*y[IDX_O2DII];
    data[5232] = 0.0 + k[1920]*y[IDX_O2II];
    data[5233] = 0.0 - k[2030]*y[IDX_O2DII] - k[2031]*y[IDX_O2DII];
    data[5234] = 0.0 + k[772]*y[IDX_O2I];
    data[5235] = 0.0 - k[2005]*y[IDX_O2DII];
    data[5236] = 0.0 - k[2033]*y[IDX_O2DII];
    data[5237] = 0.0 - k[2037]*y[IDX_O2DII];
    data[5238] = 0.0 + k[1881]*y[IDX_O2I];
    data[5239] = 0.0 - k[2035]*y[IDX_O2DII];
    data[5240] = 0.0 - k[2039]*y[IDX_O2DII];
    data[5241] = 0.0 - k[2007]*y[IDX_O2DII];
    data[5242] = 0.0 + k[770]*y[IDX_pD2II] + k[771]*y[IDX_oD2II] +
        k[772]*y[IDX_HDII] + k[1467]*y[IDX_oD3II] + k[1468]*y[IDX_mD3II] +
        k[1469]*y[IDX_oH2DII] + k[1470]*y[IDX_pH2DII] + k[1471]*y[IDX_pH2DII] +
        k[1477]*y[IDX_oD2HII] + k[1478]*y[IDX_pD2HII] + k[1881]*y[IDX_NDII] +
        k[2975]*y[IDX_pD3II] + k[2976]*y[IDX_pD3II];
    data[5243] = 0.0 + k[1920]*y[IDX_DCOI];
    data[5244] = 0.0 - k[2003]*y[IDX_CI] - k[2005]*y[IDX_NI] -
        k[2007]*y[IDX_OI] - k[2009]*y[IDX_C2I] - k[2011]*y[IDX_CHI] -
        k[2013]*y[IDX_CDI] - k[2015]*y[IDX_CNI] - k[2017]*y[IDX_COI] -
        k[2019]*y[IDX_pH2I] - k[2020]*y[IDX_oH2I] - k[2021]*y[IDX_oH2I] -
        k[2025]*y[IDX_pD2I] - k[2026]*y[IDX_oD2I] - k[2027]*y[IDX_oD2I] -
        k[2030]*y[IDX_HDI] - k[2031]*y[IDX_HDI] - k[2033]*y[IDX_N2I] -
        k[2035]*y[IDX_NHI] - k[2037]*y[IDX_NDI] - k[2039]*y[IDX_NOI] -
        k[2041]*y[IDX_OHI] - k[2043]*y[IDX_ODI] - k[2844]*y[IDX_eM] -
        k[2981]*y[IDX_oD2I] - k[2982]*y[IDX_pD2I];
    data[5245] = 0.0 - k[2043]*y[IDX_O2DII];
    data[5246] = 0.0 - k[2041]*y[IDX_O2DII];
    data[5247] = 0.0 - k[2844]*y[IDX_O2DII];
    data[5248] = 0.0 + k[1468]*y[IDX_O2I];
    data[5249] = 0.0 - k[2026]*y[IDX_O2DII] - k[2027]*y[IDX_O2DII] -
        k[2981]*y[IDX_O2DII];
    data[5250] = 0.0 + k[771]*y[IDX_O2I];
    data[5251] = 0.0 + k[1477]*y[IDX_O2I];
    data[5252] = 0.0 + k[1467]*y[IDX_O2I];
    data[5253] = 0.0 - k[2020]*y[IDX_O2DII] - k[2021]*y[IDX_O2DII];
    data[5254] = 0.0 + k[1469]*y[IDX_O2I];
    data[5255] = 0.0 - k[2025]*y[IDX_O2DII] - k[2982]*y[IDX_O2DII];
    data[5256] = 0.0 + k[770]*y[IDX_O2I];
    data[5257] = 0.0 + k[1478]*y[IDX_O2I];
    data[5258] = 0.0 + k[2975]*y[IDX_O2I] + k[2976]*y[IDX_O2I];
    data[5259] = 0.0 - k[2019]*y[IDX_O2DII];
    data[5260] = 0.0 + k[1470]*y[IDX_O2I] + k[1471]*y[IDX_O2I];
    data[5261] = 0.0 - k[1114]*y[IDX_O2HI] - k[1118]*y[IDX_O2HI] -
        k[1122]*y[IDX_O2HI];
    data[5262] = 0.0 - k[1112]*y[IDX_O2HI] - k[1116]*y[IDX_O2HI] -
        k[1120]*y[IDX_O2HI];
    data[5263] = 0.0 - k[1219]*y[IDX_O2HI];
    data[5264] = 0.0 - k[532]*y[IDX_O2HI];
    data[5265] = 0.0 - k[276] - k[278] - k[532]*y[IDX_OI] -
        k[590]*y[IDX_OHI] - k[592]*y[IDX_ODI] - k[1112]*y[IDX_HI] -
        k[1114]*y[IDX_DI] - k[1116]*y[IDX_HI] - k[1118]*y[IDX_DI] -
        k[1120]*y[IDX_HI] - k[1122]*y[IDX_DI] - k[1219]*y[IDX_NI];
    data[5266] = 0.0 - k[592]*y[IDX_O2HI];
    data[5267] = 0.0 - k[590]*y[IDX_O2HI];
    data[5268] = 0.0 - k[2002]*y[IDX_O2HII];
    data[5269] = 0.0 - k[2008]*y[IDX_O2HII];
    data[5270] = 0.0 - k[2012]*y[IDX_O2HII];
    data[5271] = 0.0 - k[2010]*y[IDX_O2HII];
    data[5272] = 0.0 - k[2014]*y[IDX_O2HII];
    data[5273] = 0.0 - k[2016]*y[IDX_O2HII];
    data[5274] = 0.0 + k[1919]*y[IDX_O2II];
    data[5275] = 0.0 - k[2028]*y[IDX_O2HII] - k[2029]*y[IDX_O2HII];
    data[5276] = 0.0 + k[773]*y[IDX_O2I];
    data[5277] = 0.0 - k[2004]*y[IDX_O2HII];
    data[5278] = 0.0 - k[2032]*y[IDX_O2HII];
    data[5279] = 0.0 - k[2036]*y[IDX_O2HII];
    data[5280] = 0.0 - k[2034]*y[IDX_O2HII];
    data[5281] = 0.0 + k[1880]*y[IDX_O2I];
    data[5282] = 0.0 - k[2038]*y[IDX_O2HII];
    data[5283] = 0.0 - k[2006]*y[IDX_O2HII];
    data[5284] = 0.0 + k[768]*y[IDX_pH2II] + k[769]*y[IDX_oH2II] +
        k[773]*y[IDX_HDII] + k[1464]*y[IDX_oH3II] + k[1465]*y[IDX_pH3II] +
        k[1466]*y[IDX_pH3II] + k[1472]*y[IDX_oH2DII] + k[1473]*y[IDX_pH2DII] +
        k[1474]*y[IDX_oD2HII] + k[1475]*y[IDX_pD2HII] + k[1476]*y[IDX_pD2HII] +
        k[1880]*y[IDX_NHII];
    data[5285] = 0.0 + k[1919]*y[IDX_HCOI];
    data[5286] = 0.0 - k[2002]*y[IDX_CI] - k[2004]*y[IDX_NI] -
        k[2006]*y[IDX_OI] - k[2008]*y[IDX_C2I] - k[2010]*y[IDX_CHI] -
        k[2012]*y[IDX_CDI] - k[2014]*y[IDX_CNI] - k[2016]*y[IDX_COI] -
        k[2018]*y[IDX_oH2I] - k[2022]*y[IDX_pD2I] - k[2023]*y[IDX_oD2I] -
        k[2024]*y[IDX_oD2I] - k[2028]*y[IDX_HDI] - k[2029]*y[IDX_HDI] -
        k[2032]*y[IDX_N2I] - k[2034]*y[IDX_NHI] - k[2036]*y[IDX_NDI] -
        k[2038]*y[IDX_NOI] - k[2040]*y[IDX_OHI] - k[2042]*y[IDX_ODI] -
        k[2843]*y[IDX_eM] - k[2927]*y[IDX_oH2I] - k[2928]*y[IDX_pH2I];
    data[5287] = 0.0 - k[2042]*y[IDX_O2HII];
    data[5288] = 0.0 - k[2040]*y[IDX_O2HII];
    data[5289] = 0.0 - k[2843]*y[IDX_O2HII];
    data[5290] = 0.0 - k[2023]*y[IDX_O2HII] - k[2024]*y[IDX_O2HII];
    data[5291] = 0.0 + k[1474]*y[IDX_O2I];
    data[5292] = 0.0 - k[2018]*y[IDX_O2HII] - k[2927]*y[IDX_O2HII];
    data[5293] = 0.0 + k[769]*y[IDX_O2I];
    data[5294] = 0.0 + k[1472]*y[IDX_O2I];
    data[5295] = 0.0 + k[1464]*y[IDX_O2I];
    data[5296] = 0.0 - k[2022]*y[IDX_O2HII];
    data[5297] = 0.0 + k[1475]*y[IDX_O2I] + k[1476]*y[IDX_O2I];
    data[5298] = 0.0 - k[2928]*y[IDX_O2HII];
    data[5299] = 0.0 + k[768]*y[IDX_O2I];
    data[5300] = 0.0 + k[1473]*y[IDX_O2I];
    data[5301] = 0.0 + k[1465]*y[IDX_O2I] + k[1466]*y[IDX_O2I];
    data[5302] = 0.0 - k[2063]*y[IDX_OCNI];
    data[5303] = 0.0 - k[455]*y[IDX_OCNI];
    data[5304] = 0.0 + k[548]*y[IDX_O2I] + k[551]*y[IDX_OHI] +
        k[552]*y[IDX_ODI];
    data[5305] = 0.0 + k[520]*y[IDX_OI];
    data[5306] = 0.0 + k[1215]*y[IDX_NI];
    data[5307] = 0.0 + k[519]*y[IDX_OI];
    data[5308] = 0.0 + k[1214]*y[IDX_NI];
    data[5309] = 0.0 - k[946]*y[IDX_OCNI] - k[947]*y[IDX_OCNI];
    data[5310] = 0.0 + k[1214]*y[IDX_HCOI] + k[1215]*y[IDX_DCOI];
    data[5311] = 0.0 + k[519]*y[IDX_HCNI] + k[520]*y[IDX_DCNI] -
        k[534]*y[IDX_OCNI] - k[535]*y[IDX_OCNI];
    data[5312] = 0.0 + k[548]*y[IDX_CNI];
    data[5313] = 0.0 - k[280] - k[418] - k[455]*y[IDX_CII] -
        k[534]*y[IDX_OI] - k[535]*y[IDX_OI] - k[946]*y[IDX_HeII] -
        k[947]*y[IDX_HeII] - k[2063]*y[IDX_CI];
    data[5314] = 0.0 + k[552]*y[IDX_CNI];
    data[5315] = 0.0 + k[551]*y[IDX_CNI];
    data[5316] = 0.0 + k[988]*y[IDX_D2OII] + k[990]*y[IDX_HDOII] -
        k[2054]*y[IDX_ODI];
    data[5317] = 0.0 - k[426]*y[IDX_ODI];
    data[5318] = 0.0 - k[2686]*y[IDX_ODI];
    data[5319] = 0.0 + k[1239]*y[IDX_D2OII] + k[1241]*y[IDX_HDOII] +
        k[2577]*y[IDX_ODII];
    data[5320] = 0.0 + k[1706]*y[IDX_D2OI] + k[1708]*y[IDX_HDOI];
    data[5321] = 0.0 + k[2585]*y[IDX_ODII];
    data[5322] = 0.0 + k[2583]*y[IDX_ODII];
    data[5323] = 0.0 + k[540]*y[IDX_O2I] + k[1248]*y[IDX_H2OII] +
        k[1249]*y[IDX_D2OII] + k[1251]*y[IDX_HDOII] + k[2581]*y[IDX_ODII];
    data[5324] = 0.0 + k[1754]*y[IDX_O2I] - k[1760]*y[IDX_ODI];
    data[5325] = 0.0 + k[2589]*y[IDX_ODII];
    data[5326] = 0.0 + k[977]*y[IDX_O2I];
    data[5327] = 0.0 + k[1244]*y[IDX_D2OII] + k[1246]*y[IDX_HDOII] +
        k[2579]*y[IDX_ODII];
    data[5328] = 0.0 - k[1759]*y[IDX_ODI];
    data[5329] = 0.0 + k[2587]*y[IDX_ODII];
    data[5330] = 0.0 + k[2591]*y[IDX_ODII];
    data[5331] = 0.0 + k[979]*y[IDX_O2I];
    data[5332] = 0.0 - k[552]*y[IDX_ODI] - k[554]*y[IDX_ODI] +
        k[2068]*y[IDX_ODM];
    data[5333] = 0.0 + k[600]*y[IDX_D2OI] + k[602]*y[IDX_HDOI] -
        k[2072]*y[IDX_ODI];
    data[5334] = 0.0 - k[558]*y[IDX_ODI] + k[1253]*y[IDX_D2OII] +
        k[1255]*y[IDX_HDOII];
    data[5335] = 0.0 - k[615]*y[IDX_ODI] + k[623]*y[IDX_D2OI] +
        k[625]*y[IDX_HDOI] - k[2328]*y[IDX_ODI];
    data[5336] = 0.0 + k[1125]*y[IDX_DI];
    data[5337] = 0.0 - k[1081]*y[IDX_ODI] + k[1087]*y[IDX_H2OI] +
        k[1089]*y[IDX_D2OI] + k[1091]*y[IDX_HDOI] + k[1099]*y[IDX_NOI] +
        k[1107]*y[IDX_HNOI] + k[1109]*y[IDX_DNOI] + k[1111]*y[IDX_O2I] +
        k[1118]*y[IDX_O2HI] + k[1119]*y[IDX_O2DI] + k[1119]*y[IDX_O2DI] +
        k[1125]*y[IDX_CO2I] + k[1127]*y[IDX_N2OI] + k[1129]*y[IDX_NO2I] +
        k[1996]*y[IDX_NO2II] + k[2663]*y[IDX_OI] - k[2669]*y[IDX_ODI] +
        k[2717]*y[IDX_OM];
    data[5338] = 0.0 - k[2204]*y[IDX_ODI];
    data[5339] = 0.0 + k[2696]*y[IDX_OI] - k[2714]*y[IDX_ODI] +
        k[3055]*y[IDX_H3OII] + k[3056]*y[IDX_H3OII] + k[3065]*y[IDX_H2DOII] +
        k[3068]*y[IDX_H2DOII] + k[3069]*y[IDX_H2DOII] + k[3077]*y[IDX_HD2OII] +
        k[3078]*y[IDX_HD2OII] + k[3079]*y[IDX_HD2OII] + k[3087]*y[IDX_D3OII] +
        k[3088]*y[IDX_D3OII];
    data[5340] = 0.0 + k[254] + k[394] + k[600]*y[IDX_CNII] +
        k[623]*y[IDX_COII] + k[904]*y[IDX_HeII] + k[1084]*y[IDX_HI] +
        k[1089]*y[IDX_DI] + k[1706]*y[IDX_C2II] + k[1813]*y[IDX_N2II] +
        k[1904]*y[IDX_NHII] + k[1905]*y[IDX_NDII] + k[2595]*y[IDX_ODII] +
        k[3168]*y[IDX_H2OII] + k[3170]*y[IDX_HDOII] + k[3171]*y[IDX_D2OII];
    data[5341] = 0.0 + k[988]*y[IDX_CI] + k[1239]*y[IDX_C2I] +
        k[1244]*y[IDX_CHI] + k[1249]*y[IDX_CDI] + k[1253]*y[IDX_COI] +
        k[2792]*y[IDX_eM] - k[3156]*y[IDX_ODI] + k[3160]*y[IDX_H2OI] +
        k[3166]*y[IDX_HDOI] + k[3171]*y[IDX_D2OI];
    data[5342] = 0.0 + k[3082]*y[IDX_HM] + k[3083]*y[IDX_HM] +
        k[3084]*y[IDX_HM] + k[3087]*y[IDX_DM] + k[3088]*y[IDX_DM] +
        k[3368]*y[IDX_OHM] + k[3370]*y[IDX_OHM] + k[3371]*y[IDX_ODM] +
        k[3444]*y[IDX_eM] + k[3456]*y[IDX_eM] + k[3457]*y[IDX_eM];
    data[5343] = 0.0 + k[504]*y[IDX_OM];
    data[5344] = 0.0 - k[1015]*y[IDX_ODI];
    data[5345] = 0.0 + k[518]*y[IDX_OI] - k[575]*y[IDX_ODI] +
        k[2601]*y[IDX_ODII];
    data[5346] = 0.0 - k[1031]*y[IDX_ODI];
    data[5347] = 0.0 - k[1569]*y[IDX_ODI];
    data[5348] = 0.0 + k[522]*y[IDX_OI] - k[579]*y[IDX_ODI] +
        k[1105]*y[IDX_HI] + k[1109]*y[IDX_DI];
    data[5349] = 0.0 - k[1589]*y[IDX_ODI];
    data[5350] = 0.0 - k[1079]*y[IDX_ODI] + k[1084]*y[IDX_D2OI] +
        k[1085]*y[IDX_HDOI] + k[1105]*y[IDX_DNOI] + k[1117]*y[IDX_O2DI] -
        k[2667]*y[IDX_ODI];
    data[5351] = 0.0 - k[2203]*y[IDX_ODI];
    data[5352] = 0.0 - k[2713]*y[IDX_ODI] + k[3060]*y[IDX_H2DOII] +
        k[3061]*y[IDX_H2DOII] + k[3071]*y[IDX_HD2OII] + k[3074]*y[IDX_HD2OII] +
        k[3075]*y[IDX_HD2OII] + k[3082]*y[IDX_D3OII] + k[3083]*y[IDX_D3OII] +
        k[3084]*y[IDX_D3OII];
    data[5353] = 0.0 + k[3060]*y[IDX_HM] + k[3061]*y[IDX_HM] +
        k[3065]*y[IDX_DM] + k[3068]*y[IDX_DM] + k[3069]*y[IDX_DM] +
        k[3355]*y[IDX_OHM] + k[3358]*y[IDX_ODM] + k[3360]*y[IDX_ODM] +
        k[3440]*y[IDX_eM] + k[3450]*y[IDX_eM] + k[3451]*y[IDX_eM];
    data[5354] = 0.0 + k[1087]*y[IDX_DI] + k[1902]*y[IDX_NDII] +
        k[2593]*y[IDX_ODII] + k[3158]*y[IDX_HDOII] + k[3160]*y[IDX_D2OII];
    data[5355] = 0.0 + k[1248]*y[IDX_CDI] - k[3154]*y[IDX_ODI] +
        k[3162]*y[IDX_HDOI] + k[3168]*y[IDX_D2OI];
    data[5356] = 0.0 + k[3055]*y[IDX_DM] + k[3056]*y[IDX_DM] +
        k[3352]*y[IDX_ODM];
    data[5357] = 0.0 - k[1014]*y[IDX_ODI];
    data[5358] = 0.0 - k[574]*y[IDX_ODI] + k[2599]*y[IDX_ODII];
    data[5359] = 0.0 - k[1030]*y[IDX_ODI];
    data[5360] = 0.0 + k[1156]*y[IDX_OI] - k[1172]*y[IDX_ODI] -
        k[1173]*y[IDX_ODI];
    data[5361] = 0.0 - k[788]*y[IDX_ODI] - k[789]*y[IDX_ODI] -
        k[2417]*y[IDX_ODI];
    data[5362] = 0.0 + k[3071]*y[IDX_HM] + k[3074]*y[IDX_HM] +
        k[3075]*y[IDX_HM] + k[3077]*y[IDX_DM] + k[3078]*y[IDX_DM] +
        k[3079]*y[IDX_DM] + k[3362]*y[IDX_OHM] + k[3364]*y[IDX_OHM] +
        k[3365]*y[IDX_ODM] + k[3367]*y[IDX_ODM] + k[3442]*y[IDX_eM] +
        k[3453]*y[IDX_eM];
    data[5363] = 0.0 + k[255] + k[395] + k[602]*y[IDX_CNII] +
        k[625]*y[IDX_COII] + k[906]*y[IDX_HeII] + k[1085]*y[IDX_HI] +
        k[1091]*y[IDX_DI] + k[1708]*y[IDX_C2II] + k[1815]*y[IDX_N2II] +
        k[1907]*y[IDX_NHII] + k[1909]*y[IDX_NDII] + k[2597]*y[IDX_ODII] +
        k[3162]*y[IDX_H2OII] + k[3164]*y[IDX_HDOII] + k[3166]*y[IDX_D2OII];
    data[5364] = 0.0 + k[990]*y[IDX_CI] + k[1241]*y[IDX_C2I] +
        k[1246]*y[IDX_CHI] + k[1251]*y[IDX_CDI] + k[1255]*y[IDX_COI] +
        k[2793]*y[IDX_eM] - k[3155]*y[IDX_ODI] + k[3158]*y[IDX_H2OI] +
        k[3164]*y[IDX_HDOI] + k[3170]*y[IDX_D2OI];
    data[5365] = 0.0 - k[1568]*y[IDX_ODI];
    data[5366] = 0.0 - k[578]*y[IDX_ODI] + k[1107]*y[IDX_DI];
    data[5367] = 0.0 - k[1588]*y[IDX_ODI];
    data[5368] = 0.0 - k[881]*y[IDX_ODI] + k[904]*y[IDX_D2OI] +
        k[906]*y[IDX_HDOI];
    data[5369] = 0.0 - k[1204]*y[IDX_ODI];
    data[5370] = 0.0 - k[1648]*y[IDX_ODI] - k[2368]*y[IDX_ODI];
    data[5371] = 0.0 + k[1813]*y[IDX_D2OI] + k[1815]*y[IDX_HDOI] -
        k[2281]*y[IDX_ODI];
    data[5372] = 0.0 - k[1633]*y[IDX_ODI];
    data[5373] = 0.0 - k[1632]*y[IDX_ODI];
    data[5374] = 0.0 + k[1127]*y[IDX_DI];
    data[5375] = 0.0 + k[1879]*y[IDX_O2I] - k[1885]*y[IDX_ODI] +
        k[1902]*y[IDX_H2OI] + k[1905]*y[IDX_D2OI] + k[1909]*y[IDX_HDOI];
    data[5376] = 0.0 + k[524]*y[IDX_OI] - k[587]*y[IDX_ODI] +
        k[2605]*y[IDX_ODII];
    data[5377] = 0.0 + k[1992]*y[IDX_O2I];
    data[5378] = 0.0 - k[1884]*y[IDX_ODI] + k[1904]*y[IDX_D2OI] +
        k[1907]*y[IDX_HDOI];
    data[5379] = 0.0 - k[585]*y[IDX_ODI] - k[586]*y[IDX_ODI] +
        k[2603]*y[IDX_ODII];
    data[5380] = 0.0 + k[525]*y[IDX_OI] - k[588]*y[IDX_ODI] -
        k[589]*y[IDX_ODI] + k[2607]*y[IDX_ODII];
    data[5381] = 0.0 + k[1994]*y[IDX_O2I];
    data[5382] = 0.0 + k[1099]*y[IDX_DI] + k[2537]*y[IDX_ODII];
    data[5383] = 0.0 + k[1129]*y[IDX_DI] + k[1514]*y[IDX_oD3II] +
        k[1515]*y[IDX_mD3II] + k[1516]*y[IDX_oH2DII] + k[1517]*y[IDX_pH2DII] +
        k[1518]*y[IDX_pH2DII] + k[1524]*y[IDX_oD2HII] + k[1525]*y[IDX_pD2HII] +
        k[2979]*y[IDX_pD3II] + k[2980]*y[IDX_pD3II];
    data[5384] = 0.0 + k[1996]*y[IDX_DI];
    data[5385] = 0.0 + k[518]*y[IDX_DCOI] + k[522]*y[IDX_DNOI] +
        k[524]*y[IDX_ND2I] + k[525]*y[IDX_NHDI] + k[533]*y[IDX_O2DI] +
        k[1154]*y[IDX_pD2I] + k[1155]*y[IDX_oD2I] + k[1156]*y[IDX_HDI] -
        k[1233]*y[IDX_ODI] + k[2663]*y[IDX_DI] + k[2696]*y[IDX_DM];
    data[5386] = 0.0 - k[1666]*y[IDX_ODI] - k[2387]*y[IDX_ODI];
    data[5387] = 0.0 + k[504]*y[IDX_DCNI] + k[2717]*y[IDX_DI];
    data[5388] = 0.0 + k[540]*y[IDX_CDI] + k[977]*y[IDX_CD2II] +
        k[979]*y[IDX_CHDII] + k[1111]*y[IDX_DI] + k[1754]*y[IDX_CDII] +
        k[1879]*y[IDX_NDII] + k[1992]*y[IDX_ND2II] + k[1994]*y[IDX_NHDII] +
        k[2299]*y[IDX_ODII];
    data[5389] = 0.0 + k[277] + k[533]*y[IDX_OI] - k[593]*y[IDX_ODI] +
        k[1117]*y[IDX_HI] + k[1119]*y[IDX_DI] + k[1119]*y[IDX_DI];
    data[5390] = 0.0 - k[2043]*y[IDX_ODI];
    data[5391] = 0.0 - k[592]*y[IDX_ODI] + k[1118]*y[IDX_DI];
    data[5392] = 0.0 - k[2042]*y[IDX_ODI];
    data[5393] = 0.0 - k[242] - k[375] - k[377] - k[426]*y[IDX_CII] -
        k[552]*y[IDX_CNI] - k[554]*y[IDX_CNI] - k[558]*y[IDX_COI] -
        k[570]*y[IDX_OHI] - k[571]*y[IDX_ODI] - k[571]*y[IDX_ODI] -
        k[571]*y[IDX_ODI] - k[571]*y[IDX_ODI] - k[574]*y[IDX_HCOI] -
        k[575]*y[IDX_DCOI] - k[578]*y[IDX_HNOI] - k[579]*y[IDX_DNOI] -
        k[585]*y[IDX_NH2I] - k[586]*y[IDX_NH2I] - k[587]*y[IDX_ND2I] -
        k[588]*y[IDX_NHDI] - k[589]*y[IDX_NHDI] - k[592]*y[IDX_O2HI] -
        k[593]*y[IDX_O2DI] - k[615]*y[IDX_COII] - k[782]*y[IDX_pH2II] -
        k[783]*y[IDX_oH2II] - k[784]*y[IDX_pH2II] - k[785]*y[IDX_oH2II] -
        k[786]*y[IDX_pD2II] - k[787]*y[IDX_oD2II] - k[788]*y[IDX_HDII] -
        k[789]*y[IDX_HDII] - k[881]*y[IDX_HeII] - k[1014]*y[IDX_HCNII] -
        k[1015]*y[IDX_DCNII] - k[1030]*y[IDX_HCOII] - k[1031]*y[IDX_DCOII] -
        k[1079]*y[IDX_HI] - k[1081]*y[IDX_DI] - k[1160]*y[IDX_pH2I] -
        k[1161]*y[IDX_oH2I] - k[1162]*y[IDX_pH2I] - k[1163]*y[IDX_oH2I] -
        k[1168]*y[IDX_pD2I] - k[1169]*y[IDX_oD2I] - k[1172]*y[IDX_HDI] -
        k[1173]*y[IDX_HDI] - k[1204]*y[IDX_NI] - k[1233]*y[IDX_OI] -
        k[1494]*y[IDX_oH3II] - k[1495]*y[IDX_pH3II] - k[1496]*y[IDX_pH3II] -
        k[1497]*y[IDX_oH3II] - k[1498]*y[IDX_pH3II] - k[1499]*y[IDX_oD3II] -
        k[1500]*y[IDX_mD3II] - k[1501]*y[IDX_oH2DII] - k[1502]*y[IDX_pH2DII] -
        k[1503]*y[IDX_pH2DII] - k[1504]*y[IDX_oH2DII] - k[1505]*y[IDX_pH2DII] -
        k[1506]*y[IDX_oD2HII] - k[1507]*y[IDX_pD2HII] - k[1508]*y[IDX_pD2HII] -
        k[1509]*y[IDX_oD2HII] - k[1510]*y[IDX_pD2HII] - k[1568]*y[IDX_HNCII] -
        k[1569]*y[IDX_DNCII] - k[1588]*y[IDX_HNOII] - k[1589]*y[IDX_DNOII] -
        k[1632]*y[IDX_N2HII] - k[1633]*y[IDX_N2DII] - k[1648]*y[IDX_NII] -
        k[1666]*y[IDX_OII] - k[1759]*y[IDX_CHII] - k[1760]*y[IDX_CDII] -
        k[1884]*y[IDX_NHII] - k[1885]*y[IDX_NDII] - k[1963]*y[IDX_OHII] -
        k[1964]*y[IDX_ODII] - k[2042]*y[IDX_O2HII] - k[2043]*y[IDX_O2DII] -
        k[2054]*y[IDX_CI] - k[2072]*y[IDX_CNII] - k[2203]*y[IDX_HII] -
        k[2204]*y[IDX_DII] - k[2281]*y[IDX_N2II] - k[2328]*y[IDX_COII] -
        k[2368]*y[IDX_NII] - k[2387]*y[IDX_OII] - k[2413]*y[IDX_pH2II] -
        k[2414]*y[IDX_oH2II] - k[2415]*y[IDX_pD2II] - k[2416]*y[IDX_oD2II] -
        k[2417]*y[IDX_HDII] - k[2667]*y[IDX_HI] - k[2669]*y[IDX_DI] -
        k[2686]*y[IDX_CM] - k[2713]*y[IDX_HM] - k[2714]*y[IDX_DM] -
        k[2977]*y[IDX_pD3II] - k[2978]*y[IDX_pD3II] - k[3154]*y[IDX_H2OII] -
        k[3155]*y[IDX_HDOII] - k[3156]*y[IDX_D2OII];
    data[5394] = 0.0 - k[1964]*y[IDX_ODI] + k[2299]*y[IDX_O2I] +
        k[2537]*y[IDX_NOI] + k[2577]*y[IDX_C2I] + k[2579]*y[IDX_CHI] +
        k[2581]*y[IDX_CDI] + k[2583]*y[IDX_C2HI] + k[2585]*y[IDX_C2DI] +
        k[2587]*y[IDX_CH2I] + k[2589]*y[IDX_CD2I] + k[2591]*y[IDX_CHDI] +
        k[2593]*y[IDX_H2OI] + k[2595]*y[IDX_D2OI] + k[2597]*y[IDX_HDOI] +
        k[2599]*y[IDX_HCOI] + k[2601]*y[IDX_DCOI] + k[2603]*y[IDX_NH2I] +
        k[2605]*y[IDX_ND2I] + k[2607]*y[IDX_NHDI];
    data[5395] = 0.0 + k[350] + k[2068]*y[IDX_CNI] + k[3352]*y[IDX_H3OII] +
        k[3358]*y[IDX_H2DOII] + k[3360]*y[IDX_H2DOII] + k[3365]*y[IDX_HD2OII] +
        k[3367]*y[IDX_HD2OII] + k[3371]*y[IDX_D3OII];
    data[5396] = 0.0 - k[570]*y[IDX_ODI];
    data[5397] = 0.0 - k[1963]*y[IDX_ODI];
    data[5398] = 0.0 + k[3355]*y[IDX_H2DOII] + k[3362]*y[IDX_HD2OII] +
        k[3364]*y[IDX_HD2OII] + k[3368]*y[IDX_D3OII] + k[3370]*y[IDX_D3OII];
    data[5399] = 0.0 + k[2792]*y[IDX_D2OII] + k[2793]*y[IDX_HDOII] +
        k[3440]*y[IDX_H2DOII] + k[3442]*y[IDX_HD2OII] + k[3444]*y[IDX_D3OII] +
        k[3450]*y[IDX_H2DOII] + k[3451]*y[IDX_H2DOII] + k[3453]*y[IDX_HD2OII] +
        k[3456]*y[IDX_D3OII] + k[3457]*y[IDX_D3OII];
    data[5400] = 0.0 - k[1500]*y[IDX_ODI] + k[1515]*y[IDX_NO2I];
    data[5401] = 0.0 + k[1155]*y[IDX_OI] - k[1169]*y[IDX_ODI];
    data[5402] = 0.0 - k[787]*y[IDX_ODI] - k[2416]*y[IDX_ODI];
    data[5403] = 0.0 - k[1506]*y[IDX_ODI] - k[1509]*y[IDX_ODI] +
        k[1524]*y[IDX_NO2I];
    data[5404] = 0.0 - k[1499]*y[IDX_ODI] + k[1514]*y[IDX_NO2I];
    data[5405] = 0.0 - k[1161]*y[IDX_ODI] - k[1163]*y[IDX_ODI];
    data[5406] = 0.0 - k[783]*y[IDX_ODI] - k[785]*y[IDX_ODI] -
        k[2414]*y[IDX_ODI];
    data[5407] = 0.0 - k[1501]*y[IDX_ODI] - k[1504]*y[IDX_ODI] +
        k[1516]*y[IDX_NO2I];
    data[5408] = 0.0 - k[1494]*y[IDX_ODI] - k[1497]*y[IDX_ODI];
    data[5409] = 0.0 + k[1154]*y[IDX_OI] - k[1168]*y[IDX_ODI];
    data[5410] = 0.0 - k[786]*y[IDX_ODI] - k[2415]*y[IDX_ODI];
    data[5411] = 0.0 - k[1507]*y[IDX_ODI] - k[1508]*y[IDX_ODI] -
        k[1510]*y[IDX_ODI] + k[1525]*y[IDX_NO2I];
    data[5412] = 0.0 - k[2977]*y[IDX_ODI] - k[2978]*y[IDX_ODI] +
        k[2979]*y[IDX_NO2I] + k[2980]*y[IDX_NO2I];
    data[5413] = 0.0 - k[1160]*y[IDX_ODI] - k[1162]*y[IDX_ODI];
    data[5414] = 0.0 - k[782]*y[IDX_ODI] - k[784]*y[IDX_ODI] -
        k[2413]*y[IDX_ODI];
    data[5415] = 0.0 - k[1502]*y[IDX_ODI] - k[1503]*y[IDX_ODI] -
        k[1505]*y[IDX_ODI] + k[1517]*y[IDX_NO2I] + k[1518]*y[IDX_NO2I];
    data[5416] = 0.0 - k[1495]*y[IDX_ODI] - k[1496]*y[IDX_ODI] -
        k[1498]*y[IDX_ODI];
    data[5417] = 0.0 - k[1922]*y[IDX_ODII];
    data[5418] = 0.0 - k[1928]*y[IDX_ODII] - k[2577]*y[IDX_ODII];
    data[5419] = 0.0 - k[2585]*y[IDX_ODII];
    data[5420] = 0.0 - k[2583]*y[IDX_ODII];
    data[5421] = 0.0 - k[1932]*y[IDX_ODII] - k[2581]*y[IDX_ODII];
    data[5422] = 0.0 - k[2589]*y[IDX_ODII];
    data[5423] = 0.0 - k[1930]*y[IDX_ODII] - k[2579]*y[IDX_ODII];
    data[5424] = 0.0 - k[2587]*y[IDX_ODII];
    data[5425] = 0.0 - k[2591]*y[IDX_ODII];
    data[5426] = 0.0 - k[1934]*y[IDX_ODII];
    data[5427] = 0.0 + k[2072]*y[IDX_ODI];
    data[5428] = 0.0 - k[1936]*y[IDX_ODII];
    data[5429] = 0.0 + k[2328]*y[IDX_ODI];
    data[5430] = 0.0 + k[2204]*y[IDX_ODI];
    data[5431] = 0.0 + k[908]*y[IDX_HeII] - k[2595]*y[IDX_ODII] -
        k[3320]*y[IDX_ODII];
    data[5432] = 0.0 + k[1677]*y[IDX_OII] - k[1968]*y[IDX_ODII] -
        k[2601]*y[IDX_ODII];
    data[5433] = 0.0 + k[2203]*y[IDX_ODI];
    data[5434] = 0.0 - k[2593]*y[IDX_ODII] - k[3316]*y[IDX_ODII];
    data[5435] = 0.0 - k[1966]*y[IDX_ODII] - k[2599]*y[IDX_ODII];
    data[5436] = 0.0 + k[1660]*y[IDX_OII] - k[1951]*y[IDX_ODII] -
        k[1952]*y[IDX_ODII];
    data[5437] = 0.0 + k[646]*y[IDX_OI] + k[2417]*y[IDX_ODI];
    data[5438] = 0.0 + k[909]*y[IDX_HeII] - k[2597]*y[IDX_ODII] -
        k[3318]*y[IDX_ODII];
    data[5439] = 0.0 + k[908]*y[IDX_D2OI] + k[909]*y[IDX_HDOI];
    data[5440] = 0.0 - k[1924]*y[IDX_ODII];
    data[5441] = 0.0 + k[2368]*y[IDX_ODI];
    data[5442] = 0.0 - k[1954]*y[IDX_ODII];
    data[5443] = 0.0 + k[2281]*y[IDX_ODI];
    data[5444] = 0.0 - k[1958]*y[IDX_ODII];
    data[5445] = 0.0 + k[1823]*y[IDX_OI];
    data[5446] = 0.0 - k[2605]*y[IDX_ODII];
    data[5447] = 0.0 - k[1956]*y[IDX_ODII];
    data[5448] = 0.0 - k[2603]*y[IDX_ODII];
    data[5449] = 0.0 - k[2607]*y[IDX_ODII];
    data[5450] = 0.0 - k[1960]*y[IDX_ODII] - k[2537]*y[IDX_ODII];
    data[5451] = 0.0 + k[644]*y[IDX_pD2II] + k[645]*y[IDX_oD2II] +
        k[646]*y[IDX_HDII] + k[1286]*y[IDX_oD3II] + k[1287]*y[IDX_mD3II] +
        k[1288]*y[IDX_oH2DII] + k[1289]*y[IDX_pH2DII] + k[1290]*y[IDX_pH2DII] +
        k[1296]*y[IDX_oD2HII] + k[1297]*y[IDX_pD2HII] + k[1823]*y[IDX_NDII] -
        k[1926]*y[IDX_ODII] + k[2007]*y[IDX_O2DII] + k[2959]*y[IDX_pD3II] +
        k[2960]*y[IDX_pD3II];
    data[5452] = 0.0 + k[1658]*y[IDX_pD2I] + k[1659]*y[IDX_oD2I] +
        k[1660]*y[IDX_HDI] + k[1677]*y[IDX_DCOI] + k[2387]*y[IDX_ODI];
    data[5453] = 0.0 - k[2299]*y[IDX_ODII];
    data[5454] = 0.0 + k[2007]*y[IDX_OI];
    data[5455] = 0.0 + k[377] - k[1964]*y[IDX_ODII] + k[2072]*y[IDX_CNII] +
        k[2203]*y[IDX_HII] + k[2204]*y[IDX_DII] + k[2281]*y[IDX_N2II] +
        k[2328]*y[IDX_COII] + k[2368]*y[IDX_NII] + k[2387]*y[IDX_OII] +
        k[2413]*y[IDX_pH2II] + k[2414]*y[IDX_oH2II] + k[2415]*y[IDX_pD2II] +
        k[2416]*y[IDX_oD2II] + k[2417]*y[IDX_HDII];
    data[5456] = 0.0 - k[297] - k[1922]*y[IDX_CI] - k[1924]*y[IDX_NI] -
        k[1926]*y[IDX_OI] - k[1928]*y[IDX_C2I] - k[1930]*y[IDX_CHI] -
        k[1932]*y[IDX_CDI] - k[1934]*y[IDX_CNI] - k[1936]*y[IDX_COI] -
        k[1939]*y[IDX_pH2I] - k[1940]*y[IDX_oH2I] - k[1941]*y[IDX_pH2I] -
        k[1942]*y[IDX_oH2I] - k[1947]*y[IDX_pD2I] - k[1948]*y[IDX_oD2I] -
        k[1951]*y[IDX_HDI] - k[1952]*y[IDX_HDI] - k[1954]*y[IDX_N2I] -
        k[1956]*y[IDX_NHI] - k[1958]*y[IDX_NDI] - k[1960]*y[IDX_NOI] -
        k[1962]*y[IDX_OHI] - k[1964]*y[IDX_ODI] - k[1966]*y[IDX_HCOI] -
        k[1968]*y[IDX_DCOI] - k[2299]*y[IDX_O2I] - k[2537]*y[IDX_NOI] -
        k[2577]*y[IDX_C2I] - k[2579]*y[IDX_CHI] - k[2581]*y[IDX_CDI] -
        k[2583]*y[IDX_C2HI] - k[2585]*y[IDX_C2DI] - k[2587]*y[IDX_CH2I] -
        k[2589]*y[IDX_CD2I] - k[2591]*y[IDX_CHDI] - k[2593]*y[IDX_H2OI] -
        k[2595]*y[IDX_D2OI] - k[2597]*y[IDX_HDOI] - k[2599]*y[IDX_HCOI] -
        k[2601]*y[IDX_DCOI] - k[2603]*y[IDX_NH2I] - k[2605]*y[IDX_ND2I] -
        k[2607]*y[IDX_NHDI] - k[2764]*y[IDX_eM] - k[3316]*y[IDX_H2OI] -
        k[3318]*y[IDX_HDOI] - k[3320]*y[IDX_D2OI];
    data[5457] = 0.0 - k[1962]*y[IDX_ODII];
    data[5458] = 0.0 - k[2764]*y[IDX_ODII];
    data[5459] = 0.0 + k[1287]*y[IDX_OI];
    data[5460] = 0.0 + k[1659]*y[IDX_OII] - k[1948]*y[IDX_ODII];
    data[5461] = 0.0 + k[645]*y[IDX_OI] + k[2416]*y[IDX_ODI];
    data[5462] = 0.0 + k[1296]*y[IDX_OI];
    data[5463] = 0.0 + k[1286]*y[IDX_OI];
    data[5464] = 0.0 - k[1940]*y[IDX_ODII] - k[1942]*y[IDX_ODII];
    data[5465] = 0.0 + k[2414]*y[IDX_ODI];
    data[5466] = 0.0 + k[1288]*y[IDX_OI];
    data[5467] = 0.0 + k[1658]*y[IDX_OII] - k[1947]*y[IDX_ODII];
    data[5468] = 0.0 + k[644]*y[IDX_OI] + k[2415]*y[IDX_ODI];
    data[5469] = 0.0 + k[1297]*y[IDX_OI];
    data[5470] = 0.0 + k[2959]*y[IDX_OI] + k[2960]*y[IDX_OI];
    data[5471] = 0.0 - k[1939]*y[IDX_ODII] - k[1941]*y[IDX_ODII];
    data[5472] = 0.0 + k[2413]*y[IDX_ODI];
    data[5473] = 0.0 + k[1289]*y[IDX_OI] + k[1290]*y[IDX_OI];
    data[5474] = 0.0 - k[2731]*y[IDX_ODM];
    data[5475] = 0.0 - k[2068]*y[IDX_ODM];
    data[5476] = 0.0 - k[2735]*y[IDX_ODM];
    data[5477] = 0.0 + k[484]*y[IDX_H2OI] + k[488]*y[IDX_D2OI] +
        k[492]*y[IDX_HDOI];
    data[5478] = 0.0 + k[487]*y[IDX_HM] + k[488]*y[IDX_DM];
    data[5479] = 0.0 - k[3371]*y[IDX_ODM];
    data[5480] = 0.0 - k[508]*y[IDX_ODM];
    data[5481] = 0.0 - k[2733]*y[IDX_ODM];
    data[5482] = 0.0 + k[487]*y[IDX_D2OI] + k[489]*y[IDX_HDOI];
    data[5483] = 0.0 - k[3357]*y[IDX_ODM] - k[3358]*y[IDX_ODM] -
        k[3359]*y[IDX_ODM] - k[3360]*y[IDX_ODM];
    data[5484] = 0.0 + k[484]*y[IDX_DM];
    data[5485] = 0.0 - k[3351]*y[IDX_ODM] - k[3352]*y[IDX_ODM] -
        k[3353]*y[IDX_ODM];
    data[5486] = 0.0 - k[506]*y[IDX_ODM];
    data[5487] = 0.0 + k[501]*y[IDX_OM];
    data[5488] = 0.0 - k[3365]*y[IDX_ODM] - k[3366]*y[IDX_ODM] -
        k[3367]*y[IDX_ODM];
    data[5489] = 0.0 + k[489]*y[IDX_HM] + k[492]*y[IDX_DM];
    data[5490] = 0.0 + k[499]*y[IDX_pD2I] + k[500]*y[IDX_oD2I] +
        k[501]*y[IDX_HDI];
    data[5491] = 0.0 - k[350] - k[506]*y[IDX_HCNI] - k[508]*y[IDX_DCNI] -
        k[2068]*y[IDX_CNI] - k[2731]*y[IDX_CI] - k[2733]*y[IDX_HI] -
        k[2735]*y[IDX_DI] - k[3351]*y[IDX_H3OII] - k[3352]*y[IDX_H3OII] -
        k[3353]*y[IDX_H3OII] - k[3357]*y[IDX_H2DOII] - k[3358]*y[IDX_H2DOII] -
        k[3359]*y[IDX_H2DOII] - k[3360]*y[IDX_H2DOII] - k[3365]*y[IDX_HD2OII] -
        k[3366]*y[IDX_HD2OII] - k[3367]*y[IDX_HD2OII] - k[3371]*y[IDX_D3OII];
    data[5492] = 0.0 + k[500]*y[IDX_OM];
    data[5493] = 0.0 + k[499]*y[IDX_OM];
    data[5494] = 0.0 + k[987]*y[IDX_H2OII] + k[989]*y[IDX_HDOII] -
        k[2053]*y[IDX_OHI];
    data[5495] = 0.0 - k[425]*y[IDX_OHI];
    data[5496] = 0.0 - k[2685]*y[IDX_OHI];
    data[5497] = 0.0 + k[1238]*y[IDX_H2OII] + k[1240]*y[IDX_HDOII] +
        k[2576]*y[IDX_OHII];
    data[5498] = 0.0 + k[1705]*y[IDX_H2OI] + k[1707]*y[IDX_HDOI];
    data[5499] = 0.0 + k[2584]*y[IDX_OHII];
    data[5500] = 0.0 + k[2582]*y[IDX_OHII];
    data[5501] = 0.0 + k[1247]*y[IDX_H2OII] + k[1250]*y[IDX_HDOII] +
        k[2580]*y[IDX_OHII];
    data[5502] = 0.0 - k[1758]*y[IDX_OHI];
    data[5503] = 0.0 + k[2588]*y[IDX_OHII];
    data[5504] = 0.0 + k[539]*y[IDX_O2I] + k[1242]*y[IDX_H2OII] +
        k[1243]*y[IDX_D2OII] + k[1245]*y[IDX_HDOII] + k[2578]*y[IDX_OHII];
    data[5505] = 0.0 + k[1753]*y[IDX_O2I] - k[1757]*y[IDX_OHI];
    data[5506] = 0.0 + k[2586]*y[IDX_OHII];
    data[5507] = 0.0 + k[976]*y[IDX_O2I];
    data[5508] = 0.0 + k[2590]*y[IDX_OHII];
    data[5509] = 0.0 + k[978]*y[IDX_O2I];
    data[5510] = 0.0 - k[551]*y[IDX_OHI] - k[553]*y[IDX_OHI] +
        k[2067]*y[IDX_OHM];
    data[5511] = 0.0 + k[599]*y[IDX_H2OI] + k[601]*y[IDX_HDOI] -
        k[2071]*y[IDX_OHI];
    data[5512] = 0.0 - k[557]*y[IDX_OHI] + k[1252]*y[IDX_H2OII] +
        k[1254]*y[IDX_HDOII];
    data[5513] = 0.0 - k[614]*y[IDX_OHI] + k[622]*y[IDX_H2OI] +
        k[624]*y[IDX_HDOI] - k[2327]*y[IDX_OHI];
    data[5514] = 0.0 + k[1124]*y[IDX_HI];
    data[5515] = 0.0 - k[1080]*y[IDX_OHI] + k[1088]*y[IDX_H2OI] +
        k[1090]*y[IDX_HDOI] + k[1108]*y[IDX_HNOI] + k[1118]*y[IDX_O2HI] -
        k[2668]*y[IDX_OHI];
    data[5516] = 0.0 - k[2202]*y[IDX_OHI];
    data[5517] = 0.0 - k[2712]*y[IDX_OHI] + k[3057]*y[IDX_H3OII] +
        k[3058]*y[IDX_H3OII] + k[3059]*y[IDX_H3OII] + k[3066]*y[IDX_H2DOII] +
        k[3067]*y[IDX_H2DOII] + k[3070]*y[IDX_H2DOII] + k[3080]*y[IDX_HD2OII] +
        k[3081]*y[IDX_HD2OII];
    data[5518] = 0.0 + k[1083]*y[IDX_HI] + k[1903]*y[IDX_NHII] +
        k[2594]*y[IDX_OHII] + k[3167]*y[IDX_H2OII] + k[3169]*y[IDX_HDOII];
    data[5519] = 0.0 + k[1243]*y[IDX_CHI] - k[3153]*y[IDX_OHI] +
        k[3159]*y[IDX_H2OI] + k[3165]*y[IDX_HDOI];
    data[5520] = 0.0 + k[3085]*y[IDX_HM] + k[3086]*y[IDX_HM] +
        k[3369]*y[IDX_OHM];
    data[5521] = 0.0 - k[1013]*y[IDX_OHI];
    data[5522] = 0.0 - k[573]*y[IDX_OHI] + k[2600]*y[IDX_OHII];
    data[5523] = 0.0 - k[1029]*y[IDX_OHI];
    data[5524] = 0.0 - k[1567]*y[IDX_OHI];
    data[5525] = 0.0 - k[577]*y[IDX_OHI] + k[1106]*y[IDX_HI];
    data[5526] = 0.0 - k[1587]*y[IDX_OHI];
    data[5527] = 0.0 - k[1078]*y[IDX_OHI] + k[1082]*y[IDX_H2OI] +
        k[1083]*y[IDX_D2OI] + k[1086]*y[IDX_HDOI] + k[1098]*y[IDX_NOI] +
        k[1104]*y[IDX_HNOI] + k[1106]*y[IDX_DNOI] + k[1110]*y[IDX_O2I] +
        k[1116]*y[IDX_O2HI] + k[1116]*y[IDX_O2HI] + k[1117]*y[IDX_O2DI] +
        k[1124]*y[IDX_CO2I] + k[1126]*y[IDX_N2OI] + k[1128]*y[IDX_NO2I] +
        k[1995]*y[IDX_NO2II] + k[2662]*y[IDX_OI] - k[2666]*y[IDX_OHI] +
        k[2716]*y[IDX_OM];
    data[5528] = 0.0 - k[2201]*y[IDX_OHI];
    data[5529] = 0.0 + k[2695]*y[IDX_OI] - k[2711]*y[IDX_OHI] +
        k[2991]*y[IDX_H3OII] + k[2992]*y[IDX_H3OII] + k[3062]*y[IDX_H2DOII] +
        k[3063]*y[IDX_H2DOII] + k[3064]*y[IDX_H2DOII] + k[3072]*y[IDX_HD2OII] +
        k[3073]*y[IDX_HD2OII] + k[3076]*y[IDX_HD2OII] + k[3085]*y[IDX_D3OII] +
        k[3086]*y[IDX_D3OII];
    data[5530] = 0.0 + k[3062]*y[IDX_HM] + k[3063]*y[IDX_HM] +
        k[3064]*y[IDX_HM] + k[3066]*y[IDX_DM] + k[3067]*y[IDX_DM] +
        k[3070]*y[IDX_DM] + k[3354]*y[IDX_OHM] + k[3356]*y[IDX_OHM] +
        k[3357]*y[IDX_ODM] + k[3359]*y[IDX_ODM] + k[3441]*y[IDX_eM] +
        k[3452]*y[IDX_eM];
    data[5531] = 0.0 + k[253] + k[393] + k[599]*y[IDX_CNII] +
        k[622]*y[IDX_COII] + k[903]*y[IDX_HeII] + k[1082]*y[IDX_HI] +
        k[1088]*y[IDX_DI] + k[1705]*y[IDX_C2II] + k[1812]*y[IDX_N2II] +
        k[1900]*y[IDX_NHII] + k[1901]*y[IDX_NDII] + k[2592]*y[IDX_OHII] +
        k[3001]*y[IDX_H2OII] + k[3157]*y[IDX_HDOII] + k[3159]*y[IDX_D2OII];
    data[5532] = 0.0 + k[987]*y[IDX_CI] + k[1238]*y[IDX_C2I] +
        k[1242]*y[IDX_CHI] + k[1247]*y[IDX_CDI] + k[1252]*y[IDX_COI] +
        k[2791]*y[IDX_eM] - k[3000]*y[IDX_OHI] + k[3001]*y[IDX_H2OI] +
        k[3161]*y[IDX_HDOI] + k[3167]*y[IDX_D2OI];
    data[5533] = 0.0 + k[2991]*y[IDX_HM] + k[2992]*y[IDX_HM] +
        k[3020]*y[IDX_OHM] + k[3025]*y[IDX_eM] + k[3027]*y[IDX_eM] +
        k[3028]*y[IDX_eM] + k[3057]*y[IDX_DM] + k[3058]*y[IDX_DM] +
        k[3059]*y[IDX_DM] + k[3351]*y[IDX_ODM] + k[3353]*y[IDX_ODM];
    data[5534] = 0.0 + k[503]*y[IDX_OM];
    data[5535] = 0.0 - k[1012]*y[IDX_OHI];
    data[5536] = 0.0 + k[517]*y[IDX_OI] - k[572]*y[IDX_OHI] +
        k[2598]*y[IDX_OHII];
    data[5537] = 0.0 - k[1028]*y[IDX_OHI];
    data[5538] = 0.0 + k[1157]*y[IDX_OI] - k[1170]*y[IDX_OHI] -
        k[1171]*y[IDX_OHI];
    data[5539] = 0.0 - k[780]*y[IDX_OHI] - k[781]*y[IDX_OHI] -
        k[2412]*y[IDX_OHI];
    data[5540] = 0.0 + k[3072]*y[IDX_HM] + k[3073]*y[IDX_HM] +
        k[3076]*y[IDX_HM] + k[3080]*y[IDX_DM] + k[3081]*y[IDX_DM] +
        k[3361]*y[IDX_OHM] + k[3363]*y[IDX_OHM] + k[3366]*y[IDX_ODM] +
        k[3443]*y[IDX_eM] + k[3454]*y[IDX_eM] + k[3455]*y[IDX_eM];
    data[5541] = 0.0 + k[256] + k[396] + k[601]*y[IDX_CNII] +
        k[624]*y[IDX_COII] + k[905]*y[IDX_HeII] + k[1086]*y[IDX_HI] +
        k[1090]*y[IDX_DI] + k[1707]*y[IDX_C2II] + k[1814]*y[IDX_N2II] +
        k[1906]*y[IDX_NHII] + k[1908]*y[IDX_NDII] + k[2596]*y[IDX_OHII] +
        k[3161]*y[IDX_H2OII] + k[3163]*y[IDX_HDOII] + k[3165]*y[IDX_D2OII];
    data[5542] = 0.0 + k[989]*y[IDX_CI] + k[1240]*y[IDX_C2I] +
        k[1245]*y[IDX_CHI] + k[1250]*y[IDX_CDI] + k[1254]*y[IDX_COI] +
        k[2794]*y[IDX_eM] - k[3152]*y[IDX_OHI] + k[3157]*y[IDX_H2OI] +
        k[3163]*y[IDX_HDOI] + k[3169]*y[IDX_D2OI];
    data[5543] = 0.0 - k[1566]*y[IDX_OHI];
    data[5544] = 0.0 + k[521]*y[IDX_OI] - k[576]*y[IDX_OHI] +
        k[1104]*y[IDX_HI] + k[1108]*y[IDX_DI];
    data[5545] = 0.0 - k[1586]*y[IDX_OHI];
    data[5546] = 0.0 - k[880]*y[IDX_OHI] + k[903]*y[IDX_H2OI] +
        k[905]*y[IDX_HDOI];
    data[5547] = 0.0 - k[1203]*y[IDX_OHI];
    data[5548] = 0.0 - k[1647]*y[IDX_OHI] - k[2367]*y[IDX_OHI];
    data[5549] = 0.0 + k[1812]*y[IDX_H2OI] + k[1814]*y[IDX_HDOI] -
        k[2280]*y[IDX_OHI];
    data[5550] = 0.0 - k[1631]*y[IDX_OHI];
    data[5551] = 0.0 - k[1630]*y[IDX_OHI];
    data[5552] = 0.0 + k[1126]*y[IDX_HI];
    data[5553] = 0.0 - k[1883]*y[IDX_OHI] + k[1901]*y[IDX_H2OI] +
        k[1908]*y[IDX_HDOI];
    data[5554] = 0.0 - k[581]*y[IDX_OHI] - k[582]*y[IDX_OHI] +
        k[2604]*y[IDX_OHII];
    data[5555] = 0.0 + k[1878]*y[IDX_O2I] - k[1882]*y[IDX_OHI] +
        k[1900]*y[IDX_H2OI] + k[1903]*y[IDX_D2OI] + k[1906]*y[IDX_HDOI];
    data[5556] = 0.0 + k[523]*y[IDX_OI] - k[580]*y[IDX_OHI] +
        k[2602]*y[IDX_OHII];
    data[5557] = 0.0 + k[1991]*y[IDX_O2I];
    data[5558] = 0.0 + k[526]*y[IDX_OI] - k[583]*y[IDX_OHI] -
        k[584]*y[IDX_OHI] + k[2606]*y[IDX_OHII];
    data[5559] = 0.0 + k[1993]*y[IDX_O2I];
    data[5560] = 0.0 + k[1098]*y[IDX_HI] + k[2536]*y[IDX_OHII];
    data[5561] = 0.0 + k[1128]*y[IDX_HI] + k[1511]*y[IDX_oH3II] +
        k[1512]*y[IDX_pH3II] + k[1513]*y[IDX_pH3II] + k[1519]*y[IDX_oH2DII] +
        k[1520]*y[IDX_pH2DII] + k[1521]*y[IDX_oD2HII] + k[1522]*y[IDX_pD2HII] +
        k[1523]*y[IDX_pD2HII];
    data[5562] = 0.0 + k[1995]*y[IDX_HI];
    data[5563] = 0.0 + k[517]*y[IDX_HCOI] + k[521]*y[IDX_HNOI] +
        k[523]*y[IDX_NH2I] + k[526]*y[IDX_NHDI] + k[532]*y[IDX_O2HI] +
        k[1152]*y[IDX_pH2I] + k[1153]*y[IDX_oH2I] + k[1157]*y[IDX_HDI] -
        k[1232]*y[IDX_OHI] + k[2662]*y[IDX_HI] + k[2695]*y[IDX_HM];
    data[5564] = 0.0 - k[1665]*y[IDX_OHI] - k[2386]*y[IDX_OHI];
    data[5565] = 0.0 + k[503]*y[IDX_HCNI] + k[2716]*y[IDX_HI];
    data[5566] = 0.0 + k[539]*y[IDX_CHI] + k[976]*y[IDX_CH2II] +
        k[978]*y[IDX_CHDII] + k[1110]*y[IDX_HI] + k[1753]*y[IDX_CHII] +
        k[1878]*y[IDX_NHII] + k[1991]*y[IDX_NH2II] + k[1993]*y[IDX_NHDII] +
        k[2298]*y[IDX_OHII];
    data[5567] = 0.0 - k[591]*y[IDX_OHI] + k[1117]*y[IDX_HI];
    data[5568] = 0.0 - k[2041]*y[IDX_OHI];
    data[5569] = 0.0 + k[276] + k[532]*y[IDX_OI] - k[590]*y[IDX_OHI] +
        k[1116]*y[IDX_HI] + k[1116]*y[IDX_HI] + k[1118]*y[IDX_DI];
    data[5570] = 0.0 - k[2040]*y[IDX_OHI];
    data[5571] = 0.0 - k[570]*y[IDX_OHI];
    data[5572] = 0.0 - k[1962]*y[IDX_OHI];
    data[5573] = 0.0 + k[3351]*y[IDX_H3OII] + k[3353]*y[IDX_H3OII] +
        k[3357]*y[IDX_H2DOII] + k[3359]*y[IDX_H2DOII] + k[3366]*y[IDX_HD2OII];
    data[5574] = 0.0 - k[241] - k[374] - k[376] - k[425]*y[IDX_CII] -
        k[551]*y[IDX_CNI] - k[553]*y[IDX_CNI] - k[557]*y[IDX_COI] -
        k[569]*y[IDX_OHI] - k[569]*y[IDX_OHI] - k[569]*y[IDX_OHI] -
        k[569]*y[IDX_OHI] - k[570]*y[IDX_ODI] - k[572]*y[IDX_HCOI] -
        k[573]*y[IDX_DCOI] - k[576]*y[IDX_HNOI] - k[577]*y[IDX_DNOI] -
        k[580]*y[IDX_NH2I] - k[581]*y[IDX_ND2I] - k[582]*y[IDX_ND2I] -
        k[583]*y[IDX_NHDI] - k[584]*y[IDX_NHDI] - k[590]*y[IDX_O2HI] -
        k[591]*y[IDX_O2DI] - k[614]*y[IDX_COII] - k[774]*y[IDX_pH2II] -
        k[775]*y[IDX_oH2II] - k[776]*y[IDX_pD2II] - k[777]*y[IDX_oD2II] -
        k[778]*y[IDX_pD2II] - k[779]*y[IDX_oD2II] - k[780]*y[IDX_HDII] -
        k[781]*y[IDX_HDII] - k[880]*y[IDX_HeII] - k[1012]*y[IDX_HCNII] -
        k[1013]*y[IDX_DCNII] - k[1028]*y[IDX_HCOII] - k[1029]*y[IDX_DCOII] -
        k[1078]*y[IDX_HI] - k[1080]*y[IDX_DI] - k[1158]*y[IDX_pH2I] -
        k[1159]*y[IDX_oH2I] - k[1164]*y[IDX_pD2I] - k[1165]*y[IDX_oD2I] -
        k[1166]*y[IDX_pD2I] - k[1167]*y[IDX_oD2I] - k[1170]*y[IDX_HDI] -
        k[1171]*y[IDX_HDI] - k[1203]*y[IDX_NI] - k[1232]*y[IDX_OI] -
        k[1479]*y[IDX_oH3II] - k[1480]*y[IDX_pH3II] - k[1481]*y[IDX_pH3II] -
        k[1482]*y[IDX_oD3II] - k[1483]*y[IDX_mD3II] - k[1484]*y[IDX_oH2DII] -
        k[1485]*y[IDX_pH2DII] - k[1486]*y[IDX_pH2DII] - k[1487]*y[IDX_oH2DII] -
        k[1488]*y[IDX_pH2DII] - k[1489]*y[IDX_oD2HII] - k[1490]*y[IDX_pD2HII] -
        k[1491]*y[IDX_pD2HII] - k[1492]*y[IDX_oD2HII] - k[1493]*y[IDX_pD2HII] -
        k[1566]*y[IDX_HNCII] - k[1567]*y[IDX_DNCII] - k[1586]*y[IDX_HNOII] -
        k[1587]*y[IDX_DNOII] - k[1630]*y[IDX_N2HII] - k[1631]*y[IDX_N2DII] -
        k[1647]*y[IDX_NII] - k[1665]*y[IDX_OII] - k[1757]*y[IDX_CHII] -
        k[1758]*y[IDX_CDII] - k[1882]*y[IDX_NHII] - k[1883]*y[IDX_NDII] -
        k[1961]*y[IDX_OHII] - k[1962]*y[IDX_ODII] - k[2040]*y[IDX_O2HII] -
        k[2041]*y[IDX_O2DII] - k[2053]*y[IDX_CI] - k[2071]*y[IDX_CNII] -
        k[2201]*y[IDX_HII] - k[2202]*y[IDX_DII] - k[2280]*y[IDX_N2II] -
        k[2327]*y[IDX_COII] - k[2367]*y[IDX_NII] - k[2386]*y[IDX_OII] -
        k[2408]*y[IDX_pH2II] - k[2409]*y[IDX_oH2II] - k[2410]*y[IDX_pD2II] -
        k[2411]*y[IDX_oD2II] - k[2412]*y[IDX_HDII] - k[2666]*y[IDX_HI] -
        k[2668]*y[IDX_DI] - k[2685]*y[IDX_CM] - k[2711]*y[IDX_HM] -
        k[2712]*y[IDX_DM] - k[3000]*y[IDX_H2OII] - k[3152]*y[IDX_HDOII] -
        k[3153]*y[IDX_D2OII];
    data[5575] = 0.0 - k[1961]*y[IDX_OHI] + k[2298]*y[IDX_O2I] +
        k[2536]*y[IDX_NOI] + k[2576]*y[IDX_C2I] + k[2578]*y[IDX_CHI] +
        k[2580]*y[IDX_CDI] + k[2582]*y[IDX_C2HI] + k[2584]*y[IDX_C2DI] +
        k[2586]*y[IDX_CH2I] + k[2588]*y[IDX_CD2I] + k[2590]*y[IDX_CHDI] +
        k[2592]*y[IDX_H2OI] + k[2594]*y[IDX_D2OI] + k[2596]*y[IDX_HDOI] +
        k[2598]*y[IDX_HCOI] + k[2600]*y[IDX_DCOI] + k[2602]*y[IDX_NH2I] +
        k[2604]*y[IDX_ND2I] + k[2606]*y[IDX_NHDI];
    data[5576] = 0.0 + k[349] + k[2067]*y[IDX_CNI] + k[3020]*y[IDX_H3OII] +
        k[3354]*y[IDX_H2DOII] + k[3356]*y[IDX_H2DOII] + k[3361]*y[IDX_HD2OII] +
        k[3363]*y[IDX_HD2OII] + k[3369]*y[IDX_D3OII];
    data[5577] = 0.0 + k[2791]*y[IDX_H2OII] + k[2794]*y[IDX_HDOII] +
        k[3025]*y[IDX_H3OII] + k[3027]*y[IDX_H3OII] + k[3028]*y[IDX_H3OII] +
        k[3441]*y[IDX_H2DOII] + k[3443]*y[IDX_HD2OII] + k[3452]*y[IDX_H2DOII] +
        k[3454]*y[IDX_HD2OII] + k[3455]*y[IDX_HD2OII];
    data[5578] = 0.0 - k[1483]*y[IDX_OHI];
    data[5579] = 0.0 - k[1165]*y[IDX_OHI] - k[1167]*y[IDX_OHI];
    data[5580] = 0.0 - k[777]*y[IDX_OHI] - k[779]*y[IDX_OHI] -
        k[2411]*y[IDX_OHI];
    data[5581] = 0.0 - k[1489]*y[IDX_OHI] - k[1492]*y[IDX_OHI] +
        k[1521]*y[IDX_NO2I];
    data[5582] = 0.0 - k[1482]*y[IDX_OHI];
    data[5583] = 0.0 + k[1153]*y[IDX_OI] - k[1159]*y[IDX_OHI];
    data[5584] = 0.0 - k[775]*y[IDX_OHI] - k[2409]*y[IDX_OHI];
    data[5585] = 0.0 - k[1484]*y[IDX_OHI] - k[1487]*y[IDX_OHI] +
        k[1519]*y[IDX_NO2I];
    data[5586] = 0.0 - k[1479]*y[IDX_OHI] + k[1511]*y[IDX_NO2I];
    data[5587] = 0.0 - k[1164]*y[IDX_OHI] - k[1166]*y[IDX_OHI];
    data[5588] = 0.0 - k[776]*y[IDX_OHI] - k[778]*y[IDX_OHI] -
        k[2410]*y[IDX_OHI];
    data[5589] = 0.0 - k[1490]*y[IDX_OHI] - k[1491]*y[IDX_OHI] -
        k[1493]*y[IDX_OHI] + k[1522]*y[IDX_NO2I] + k[1523]*y[IDX_NO2I];
    data[5590] = 0.0 + k[1152]*y[IDX_OI] - k[1158]*y[IDX_OHI];
    data[5591] = 0.0 - k[774]*y[IDX_OHI] - k[2408]*y[IDX_OHI];
    data[5592] = 0.0 - k[1485]*y[IDX_OHI] - k[1486]*y[IDX_OHI] -
        k[1488]*y[IDX_OHI] + k[1520]*y[IDX_NO2I];
    data[5593] = 0.0 - k[1480]*y[IDX_OHI] - k[1481]*y[IDX_OHI] +
        k[1512]*y[IDX_NO2I] + k[1513]*y[IDX_NO2I];
    data[5594] = 0.0 - k[1921]*y[IDX_OHII];
    data[5595] = 0.0 - k[1927]*y[IDX_OHII] - k[2576]*y[IDX_OHII];
    data[5596] = 0.0 - k[2584]*y[IDX_OHII];
    data[5597] = 0.0 - k[2582]*y[IDX_OHII];
    data[5598] = 0.0 - k[1931]*y[IDX_OHII] - k[2580]*y[IDX_OHII];
    data[5599] = 0.0 - k[2588]*y[IDX_OHII];
    data[5600] = 0.0 - k[1929]*y[IDX_OHII] - k[2578]*y[IDX_OHII];
    data[5601] = 0.0 - k[2586]*y[IDX_OHII];
    data[5602] = 0.0 - k[2590]*y[IDX_OHII];
    data[5603] = 0.0 - k[1933]*y[IDX_OHII];
    data[5604] = 0.0 + k[2071]*y[IDX_OHI];
    data[5605] = 0.0 - k[1935]*y[IDX_OHII];
    data[5606] = 0.0 + k[2327]*y[IDX_OHI];
    data[5607] = 0.0 + k[2202]*y[IDX_OHI];
    data[5608] = 0.0 - k[2594]*y[IDX_OHII] - k[3319]*y[IDX_OHII];
    data[5609] = 0.0 - k[1967]*y[IDX_OHII] - k[2600]*y[IDX_OHII];
    data[5610] = 0.0 + k[2201]*y[IDX_OHI];
    data[5611] = 0.0 + k[907]*y[IDX_HeII] - k[2592]*y[IDX_OHII] -
        k[3015]*y[IDX_OHII];
    data[5612] = 0.0 + k[1676]*y[IDX_OII] - k[1965]*y[IDX_OHII] -
        k[2598]*y[IDX_OHII];
    data[5613] = 0.0 + k[1661]*y[IDX_OII] - k[1949]*y[IDX_OHII] -
        k[1950]*y[IDX_OHII];
    data[5614] = 0.0 + k[647]*y[IDX_OI] + k[2412]*y[IDX_OHI];
    data[5615] = 0.0 + k[910]*y[IDX_HeII] - k[2596]*y[IDX_OHII] -
        k[3317]*y[IDX_OHII];
    data[5616] = 0.0 + k[907]*y[IDX_H2OI] + k[910]*y[IDX_HDOI];
    data[5617] = 0.0 - k[1923]*y[IDX_OHII];
    data[5618] = 0.0 + k[2367]*y[IDX_OHI];
    data[5619] = 0.0 - k[1953]*y[IDX_OHII];
    data[5620] = 0.0 + k[2280]*y[IDX_OHI];
    data[5621] = 0.0 - k[1957]*y[IDX_OHII];
    data[5622] = 0.0 - k[2604]*y[IDX_OHII];
    data[5623] = 0.0 - k[1955]*y[IDX_OHII];
    data[5624] = 0.0 + k[1822]*y[IDX_OI];
    data[5625] = 0.0 - k[2602]*y[IDX_OHII];
    data[5626] = 0.0 - k[2606]*y[IDX_OHII];
    data[5627] = 0.0 - k[1959]*y[IDX_OHII] - k[2536]*y[IDX_OHII];
    data[5628] = 0.0 + k[642]*y[IDX_pH2II] + k[643]*y[IDX_oH2II] +
        k[647]*y[IDX_HDII] + k[1283]*y[IDX_oH3II] + k[1284]*y[IDX_pH3II] +
        k[1285]*y[IDX_pH3II] + k[1291]*y[IDX_oH2DII] + k[1292]*y[IDX_pH2DII] +
        k[1293]*y[IDX_oD2HII] + k[1294]*y[IDX_pD2HII] + k[1295]*y[IDX_pD2HII] +
        k[1822]*y[IDX_NHII] - k[1925]*y[IDX_OHII] + k[2006]*y[IDX_O2HII];
    data[5629] = 0.0 + k[1656]*y[IDX_pH2I] + k[1657]*y[IDX_oH2I] +
        k[1661]*y[IDX_HDI] + k[1676]*y[IDX_HCOI] + k[2386]*y[IDX_OHI];
    data[5630] = 0.0 - k[2298]*y[IDX_OHII];
    data[5631] = 0.0 + k[2006]*y[IDX_OI];
    data[5632] = 0.0 - k[1963]*y[IDX_OHII];
    data[5633] = 0.0 + k[376] - k[1961]*y[IDX_OHII] + k[2071]*y[IDX_CNII] +
        k[2201]*y[IDX_HII] + k[2202]*y[IDX_DII] + k[2280]*y[IDX_N2II] +
        k[2327]*y[IDX_COII] + k[2367]*y[IDX_NII] + k[2386]*y[IDX_OII] +
        k[2408]*y[IDX_pH2II] + k[2409]*y[IDX_oH2II] + k[2410]*y[IDX_pD2II] +
        k[2411]*y[IDX_oD2II] + k[2412]*y[IDX_HDII];
    data[5634] = 0.0 - k[296] - k[1921]*y[IDX_CI] - k[1923]*y[IDX_NI] -
        k[1925]*y[IDX_OI] - k[1927]*y[IDX_C2I] - k[1929]*y[IDX_CHI] -
        k[1931]*y[IDX_CDI] - k[1933]*y[IDX_CNI] - k[1935]*y[IDX_COI] -
        k[1937]*y[IDX_pH2I] - k[1938]*y[IDX_oH2I] - k[1943]*y[IDX_pD2I] -
        k[1944]*y[IDX_oD2I] - k[1945]*y[IDX_pD2I] - k[1946]*y[IDX_oD2I] -
        k[1949]*y[IDX_HDI] - k[1950]*y[IDX_HDI] - k[1953]*y[IDX_N2I] -
        k[1955]*y[IDX_NHI] - k[1957]*y[IDX_NDI] - k[1959]*y[IDX_NOI] -
        k[1961]*y[IDX_OHI] - k[1963]*y[IDX_ODI] - k[1965]*y[IDX_HCOI] -
        k[1967]*y[IDX_DCOI] - k[2298]*y[IDX_O2I] - k[2536]*y[IDX_NOI] -
        k[2576]*y[IDX_C2I] - k[2578]*y[IDX_CHI] - k[2580]*y[IDX_CDI] -
        k[2582]*y[IDX_C2HI] - k[2584]*y[IDX_C2DI] - k[2586]*y[IDX_CH2I] -
        k[2588]*y[IDX_CD2I] - k[2590]*y[IDX_CHDI] - k[2592]*y[IDX_H2OI] -
        k[2594]*y[IDX_D2OI] - k[2596]*y[IDX_HDOI] - k[2598]*y[IDX_HCOI] -
        k[2600]*y[IDX_DCOI] - k[2602]*y[IDX_NH2I] - k[2604]*y[IDX_ND2I] -
        k[2606]*y[IDX_NHDI] - k[2763]*y[IDX_eM] - k[3015]*y[IDX_H2OI] -
        k[3317]*y[IDX_HDOI] - k[3319]*y[IDX_D2OI];
    data[5635] = 0.0 - k[2763]*y[IDX_OHII];
    data[5636] = 0.0 - k[1944]*y[IDX_OHII] - k[1946]*y[IDX_OHII];
    data[5637] = 0.0 + k[2411]*y[IDX_OHI];
    data[5638] = 0.0 + k[1293]*y[IDX_OI];
    data[5639] = 0.0 + k[1657]*y[IDX_OII] - k[1938]*y[IDX_OHII];
    data[5640] = 0.0 + k[643]*y[IDX_OI] + k[2409]*y[IDX_OHI];
    data[5641] = 0.0 + k[1291]*y[IDX_OI];
    data[5642] = 0.0 + k[1283]*y[IDX_OI];
    data[5643] = 0.0 - k[1943]*y[IDX_OHII] - k[1945]*y[IDX_OHII];
    data[5644] = 0.0 + k[2410]*y[IDX_OHI];
    data[5645] = 0.0 + k[1294]*y[IDX_OI] + k[1295]*y[IDX_OI];
    data[5646] = 0.0 + k[1656]*y[IDX_OII] - k[1937]*y[IDX_OHII];
    data[5647] = 0.0 + k[642]*y[IDX_OI] + k[2408]*y[IDX_OHI];
    data[5648] = 0.0 + k[1292]*y[IDX_OI];
    data[5649] = 0.0 + k[1284]*y[IDX_OI] + k[1285]*y[IDX_OI];
    data[5650] = 0.0 - k[2730]*y[IDX_OHM];
    data[5651] = 0.0 - k[2067]*y[IDX_OHM];
    data[5652] = 0.0 - k[2734]*y[IDX_OHM];
    data[5653] = 0.0 + k[485]*y[IDX_H2OI] + k[491]*y[IDX_HDOI];
    data[5654] = 0.0 + k[486]*y[IDX_HM];
    data[5655] = 0.0 - k[3368]*y[IDX_OHM] - k[3369]*y[IDX_OHM] -
        k[3370]*y[IDX_OHM];
    data[5656] = 0.0 - k[507]*y[IDX_OHM];
    data[5657] = 0.0 - k[2732]*y[IDX_OHM];
    data[5658] = 0.0 + k[483]*y[IDX_H2OI] + k[486]*y[IDX_D2OI] +
        k[490]*y[IDX_HDOI];
    data[5659] = 0.0 - k[3354]*y[IDX_OHM] - k[3355]*y[IDX_OHM] -
        k[3356]*y[IDX_OHM];
    data[5660] = 0.0 + k[483]*y[IDX_HM] + k[485]*y[IDX_DM];
    data[5661] = 0.0 - k[3020]*y[IDX_OHM];
    data[5662] = 0.0 - k[505]*y[IDX_OHM];
    data[5663] = 0.0 + k[502]*y[IDX_OM];
    data[5664] = 0.0 - k[3361]*y[IDX_OHM] - k[3362]*y[IDX_OHM] -
        k[3363]*y[IDX_OHM] - k[3364]*y[IDX_OHM];
    data[5665] = 0.0 + k[490]*y[IDX_HM] + k[491]*y[IDX_DM];
    data[5666] = 0.0 + k[497]*y[IDX_pH2I] + k[498]*y[IDX_oH2I] +
        k[502]*y[IDX_HDI];
    data[5667] = 0.0 - k[349] - k[505]*y[IDX_HCNI] - k[507]*y[IDX_DCNI] -
        k[2067]*y[IDX_CNI] - k[2730]*y[IDX_CI] - k[2732]*y[IDX_HI] -
        k[2734]*y[IDX_DI] - k[3020]*y[IDX_H3OII] - k[3354]*y[IDX_H2DOII] -
        k[3355]*y[IDX_H2DOII] - k[3356]*y[IDX_H2DOII] - k[3361]*y[IDX_HD2OII] -
        k[3362]*y[IDX_HD2OII] - k[3363]*y[IDX_HD2OII] - k[3364]*y[IDX_HD2OII] -
        k[3368]*y[IDX_D3OII] - k[3369]*y[IDX_D3OII] - k[3370]*y[IDX_D3OII];
    data[5668] = 0.0 + k[498]*y[IDX_OM];
    data[5669] = 0.0 + k[497]*y[IDX_OM];
    data[5670] = 0.0 + k[202] + k[351] + k[2670]*y[IDX_CM] +
        k[2687]*y[IDX_HM] + k[2688]*y[IDX_DM] + k[2715]*y[IDX_OM] +
        k[2730]*y[IDX_OHM] + k[2731]*y[IDX_ODM] - k[2736]*y[IDX_eM];
    data[5671] = 0.0 - k[2845]*y[IDX_eM];
    data[5672] = 0.0 + k[344] + k[482]*y[IDX_CO2I] + k[2670]*y[IDX_CI] +
        k[2671]*y[IDX_HI] + k[2672]*y[IDX_DI] + k[2673]*y[IDX_NI] +
        k[2674]*y[IDX_OI] + k[2675]*y[IDX_CHI] + k[2676]*y[IDX_CDI] +
        k[2677]*y[IDX_pH2I] + k[2678]*y[IDX_oH2I] + k[2679]*y[IDX_pD2I] +
        k[2680]*y[IDX_oD2I] + k[2681]*y[IDX_HDI] + k[2682]*y[IDX_NHI] +
        k[2683]*y[IDX_NDI] + k[2684]*y[IDX_O2I] + k[2685]*y[IDX_OHI] +
        k[2686]*y[IDX_ODI];
    data[5673] = 0.0 + k[353] + k[2697]*y[IDX_HM] + k[2698]*y[IDX_DM];
    data[5674] = 0.0 - k[2740]*y[IDX_eM];
    data[5675] = 0.0 + k[381];
    data[5676] = 0.0 - k[2766]*y[IDX_eM] - k[2768]*y[IDX_eM] -
        k[2770]*y[IDX_eM];
    data[5677] = 0.0 + k[380];
    data[5678] = 0.0 - k[2765]*y[IDX_eM] - k[2767]*y[IDX_eM] -
        k[2769]*y[IDX_eM];
    data[5679] = 0.0 - k[2771]*y[IDX_eM] - k[2772]*y[IDX_eM];
    data[5680] = 0.0 - k[2773]*y[IDX_eM];
    data[5681] = 0.0 - k[2774]*y[IDX_eM];
    data[5682] = 0.0 + k[357] + k[2045]*y[IDX_OI] + k[2676]*y[IDX_CM] +
        k[2701]*y[IDX_HM] + k[2702]*y[IDX_DM] + k[2721]*y[IDX_OM];
    data[5683] = 0.0 - k[2742]*y[IDX_eM];
    data[5684] = 0.0 + k[250] + k[390];
    data[5685] = 0.0 - k[2776]*y[IDX_eM] - k[2779]*y[IDX_eM] -
        k[2783]*y[IDX_eM];
    data[5686] = 0.0 + k[356] + k[2044]*y[IDX_OI] + k[2675]*y[IDX_CM] +
        k[2699]*y[IDX_HM] + k[2700]*y[IDX_DM] + k[2720]*y[IDX_OM];
    data[5687] = 0.0 - k[2741]*y[IDX_eM];
    data[5688] = 0.0 + k[249] + k[389];
    data[5689] = 0.0 - k[2775]*y[IDX_eM] - k[2778]*y[IDX_eM] -
        k[2782]*y[IDX_eM];
    data[5690] = 0.0 + k[251] + k[391];
    data[5691] = 0.0 - k[2777]*y[IDX_eM] - k[2780]*y[IDX_eM] -
        k[2781]*y[IDX_eM] - k[2784]*y[IDX_eM];
    data[5692] = 0.0 + k[2703]*y[IDX_HM] + k[2704]*y[IDX_DM];
    data[5693] = 0.0 - k[2743]*y[IDX_eM];
    data[5694] = 0.0 + k[348] + k[2728]*y[IDX_HI] + k[2729]*y[IDX_DI];
    data[5695] = 0.0 - k[2785]*y[IDX_eM] - k[2786]*y[IDX_eM];
    data[5696] = 0.0 + k[286] + k[2705]*y[IDX_HM] + k[2706]*y[IDX_DM] +
        k[2722]*y[IDX_OM];
    data[5697] = 0.0 - k[2744]*y[IDX_eM];
    data[5698] = 0.0 + k[482]*y[IDX_CM];
    data[5699] = 0.0 - k[2787]*y[IDX_eM];
    data[5700] = 0.0 + k[204] + k[2672]*y[IDX_CM] + k[2691]*y[IDX_HM] +
        k[2692]*y[IDX_DM] + k[2717]*y[IDX_OM] + k[2729]*y[IDX_CNM] +
        k[2734]*y[IDX_OHM] + k[2735]*y[IDX_ODM] - k[2738]*y[IDX_eM];
    data[5701] = 0.0 - k[2847]*y[IDX_eM];
    data[5702] = 0.0 + k[346] + k[2688]*y[IDX_CI] + k[2690]*y[IDX_HI] +
        k[2692]*y[IDX_DI] + k[2694]*y[IDX_NI] + k[2696]*y[IDX_OI] +
        k[2698]*y[IDX_C2I] + k[2700]*y[IDX_CHI] + k[2702]*y[IDX_CDI] +
        k[2704]*y[IDX_CNI] + k[2706]*y[IDX_COI] + k[2708]*y[IDX_NHI] +
        k[2710]*y[IDX_NDI] + k[2712]*y[IDX_OHI] + k[2714]*y[IDX_ODI];
    data[5703] = 0.0 + k[398];
    data[5704] = 0.0 - k[2789]*y[IDX_eM] - k[2792]*y[IDX_eM] -
        k[2796]*y[IDX_eM];
    data[5705] = 0.0 - k[3444]*y[IDX_eM] - k[3449]*y[IDX_eM] -
        k[3456]*y[IDX_eM] - k[3457]*y[IDX_eM] - k[3464]*y[IDX_eM] -
        k[3465]*y[IDX_eM];
    data[5706] = 0.0 - k[2821]*y[IDX_eM];
    data[5707] = 0.0 + k[262] + k[405];
    data[5708] = 0.0 - k[2823]*y[IDX_eM];
    data[5709] = 0.0 - k[2825]*y[IDX_eM];
    data[5710] = 0.0 + k[266];
    data[5711] = 0.0 - k[2827]*y[IDX_eM];
    data[5712] = 0.0 - k[2829]*y[IDX_eM];
    data[5713] = 0.0 - k[169]*y[IDX_eM];
    data[5714] = 0.0 + k[203] + k[2671]*y[IDX_CM] + k[2689]*y[IDX_HM] +
        k[2690]*y[IDX_DM] + k[2716]*y[IDX_OM] + k[2728]*y[IDX_CNM] +
        k[2732]*y[IDX_OHM] + k[2733]*y[IDX_ODM] - k[2737]*y[IDX_eM];
    data[5715] = 0.0 - k[2846]*y[IDX_eM];
    data[5716] = 0.0 + k[345] + k[2687]*y[IDX_CI] + k[2689]*y[IDX_HI] +
        k[2691]*y[IDX_DI] + k[2693]*y[IDX_NI] + k[2695]*y[IDX_OI] +
        k[2697]*y[IDX_C2I] + k[2699]*y[IDX_CHI] + k[2701]*y[IDX_CDI] +
        k[2703]*y[IDX_CNI] + k[2705]*y[IDX_COI] + k[2707]*y[IDX_NHI] +
        k[2709]*y[IDX_NDI] + k[2711]*y[IDX_OHI] + k[2713]*y[IDX_ODI];
    data[5717] = 0.0 - k[3440]*y[IDX_eM] - k[3441]*y[IDX_eM] -
        k[3445]*y[IDX_eM] - k[3446]*y[IDX_eM] - k[3450]*y[IDX_eM] -
        k[3451]*y[IDX_eM] - k[3452]*y[IDX_eM] - k[3458]*y[IDX_eM] -
        k[3459]*y[IDX_eM] - k[3460]*y[IDX_eM];
    data[5718] = 0.0 + k[397];
    data[5719] = 0.0 - k[2788]*y[IDX_eM] - k[2791]*y[IDX_eM] -
        k[2795]*y[IDX_eM];
    data[5720] = 0.0 - k[3025]*y[IDX_eM] - k[3026]*y[IDX_eM] -
        k[3027]*y[IDX_eM] - k[3028]*y[IDX_eM] - k[3029]*y[IDX_eM] -
        k[3030]*y[IDX_eM];
    data[5721] = 0.0 - k[2820]*y[IDX_eM];
    data[5722] = 0.0 + k[261] + k[404];
    data[5723] = 0.0 - k[2822]*y[IDX_eM];
    data[5724] = 0.0 + k[217] + k[218] + k[229] + k[2681]*y[IDX_CM] +
        k[2727]*y[IDX_OM];
    data[5725] = 0.0 - k[2749]*y[IDX_eM] - k[2755]*y[IDX_eM];
    data[5726] = 0.0 - k[3442]*y[IDX_eM] - k[3443]*y[IDX_eM] -
        k[3447]*y[IDX_eM] - k[3448]*y[IDX_eM] - k[3453]*y[IDX_eM] -
        k[3454]*y[IDX_eM] - k[3455]*y[IDX_eM] - k[3461]*y[IDX_eM] -
        k[3462]*y[IDX_eM] - k[3463]*y[IDX_eM];
    data[5727] = 0.0 + k[399];
    data[5728] = 0.0 - k[2790]*y[IDX_eM] - k[2793]*y[IDX_eM] -
        k[2794]*y[IDX_eM] - k[2797]*y[IDX_eM];
    data[5729] = 0.0 - k[2824]*y[IDX_eM];
    data[5730] = 0.0 + k[265];
    data[5731] = 0.0 - k[2826]*y[IDX_eM];
    data[5732] = 0.0 - k[2828]*y[IDX_eM];
    data[5733] = 0.0 + k[205];
    data[5734] = 0.0 - k[2848]*y[IDX_eM];
    data[5735] = 0.0 - k[2757]*y[IDX_eM];
    data[5736] = 0.0 - k[2756]*y[IDX_eM];
    data[5737] = 0.0 + k[206] + k[2673]*y[IDX_CM] + k[2693]*y[IDX_HM] +
        k[2694]*y[IDX_DM] + k[2718]*y[IDX_OM];
    data[5738] = 0.0 - k[2849]*y[IDX_eM];
    data[5739] = 0.0 - k[2758]*y[IDX_eM];
    data[5740] = 0.0 - k[2831]*y[IDX_eM] - k[2833]*y[IDX_eM];
    data[5741] = 0.0 - k[2830]*y[IDX_eM] - k[2832]*y[IDX_eM];
    data[5742] = 0.0 - k[2834]*y[IDX_eM];
    data[5743] = 0.0 + k[369] + k[2683]*y[IDX_CM] + k[2709]*y[IDX_HM] +
        k[2710]*y[IDX_DM];
    data[5744] = 0.0 - k[2760]*y[IDX_eM];
    data[5745] = 0.0 + k[273] + k[415];
    data[5746] = 0.0 - k[2836]*y[IDX_eM] - k[2839]*y[IDX_eM];
    data[5747] = 0.0 + k[368] + k[2682]*y[IDX_CM] + k[2707]*y[IDX_HM] +
        k[2708]*y[IDX_DM];
    data[5748] = 0.0 - k[2759]*y[IDX_eM];
    data[5749] = 0.0 + k[272] + k[414];
    data[5750] = 0.0 - k[2835]*y[IDX_eM] - k[2838]*y[IDX_eM];
    data[5751] = 0.0 + k[274] + k[416];
    data[5752] = 0.0 - k[2837]*y[IDX_eM] - k[2840]*y[IDX_eM] -
        k[2841]*y[IDX_eM];
    data[5753] = 0.0 + k[238] + k[371];
    data[5754] = 0.0 - k[2761]*y[IDX_eM];
    data[5755] = 0.0 - k[2842]*y[IDX_eM];
    data[5756] = 0.0 + k[207] + k[2044]*y[IDX_CHI] + k[2045]*y[IDX_CDI] +
        k[2674]*y[IDX_CM] + k[2695]*y[IDX_HM] + k[2696]*y[IDX_DM] +
        k[2719]*y[IDX_OM] - k[2739]*y[IDX_eM];
    data[5757] = 0.0 - k[2850]*y[IDX_eM];
    data[5758] = 0.0 + k[347] + k[2715]*y[IDX_CI] + k[2716]*y[IDX_HI] +
        k[2717]*y[IDX_DI] + k[2718]*y[IDX_NI] + k[2719]*y[IDX_OI] +
        k[2720]*y[IDX_CHI] + k[2721]*y[IDX_CDI] + k[2722]*y[IDX_COI] +
        k[2723]*y[IDX_pH2I] + k[2724]*y[IDX_oH2I] + k[2725]*y[IDX_pD2I] +
        k[2726]*y[IDX_oD2I] + k[2727]*y[IDX_HDI];
    data[5759] = 0.0 + k[240] + k[373] + k[2684]*y[IDX_CM];
    data[5760] = 0.0 - k[2762]*y[IDX_eM];
    data[5761] = 0.0 - k[2844]*y[IDX_eM];
    data[5762] = 0.0 - k[2843]*y[IDX_eM];
    data[5763] = 0.0 + k[377] + k[2686]*y[IDX_CM] + k[2713]*y[IDX_HM] +
        k[2714]*y[IDX_DM];
    data[5764] = 0.0 - k[2764]*y[IDX_eM];
    data[5765] = 0.0 + k[350] + k[2731]*y[IDX_CI] + k[2733]*y[IDX_HI] +
        k[2735]*y[IDX_DI];
    data[5766] = 0.0 + k[376] + k[2685]*y[IDX_CM] + k[2711]*y[IDX_HM] +
        k[2712]*y[IDX_DM];
    data[5767] = 0.0 - k[2763]*y[IDX_eM];
    data[5768] = 0.0 + k[349] + k[2730]*y[IDX_CI] + k[2732]*y[IDX_HI] +
        k[2734]*y[IDX_DI];
    data[5769] = 0.0 - k[169]*y[IDX_GRAIN0I] - k[2736]*y[IDX_CI] -
        k[2737]*y[IDX_HI] - k[2738]*y[IDX_DI] - k[2739]*y[IDX_OI] -
        k[2740]*y[IDX_C2II] - k[2741]*y[IDX_CHII] - k[2742]*y[IDX_CDII] -
        k[2743]*y[IDX_CNII] - k[2744]*y[IDX_COII] - k[2745]*y[IDX_pH2II] -
        k[2746]*y[IDX_oH2II] - k[2747]*y[IDX_pD2II] - k[2748]*y[IDX_oD2II] -
        k[2749]*y[IDX_HDII] - k[2750]*y[IDX_pH2II] - k[2751]*y[IDX_oH2II] -
        k[2752]*y[IDX_pD2II] - k[2753]*y[IDX_pD2II] - k[2754]*y[IDX_oD2II] -
        k[2755]*y[IDX_HDII] - k[2756]*y[IDX_HeHII] - k[2757]*y[IDX_HeDII] -
        k[2758]*y[IDX_N2II] - k[2759]*y[IDX_NHII] - k[2760]*y[IDX_NDII] -
        k[2761]*y[IDX_NOII] - k[2762]*y[IDX_O2II] - k[2763]*y[IDX_OHII] -
        k[2764]*y[IDX_ODII] - k[2765]*y[IDX_C2HII] - k[2766]*y[IDX_C2DII] -
        k[2767]*y[IDX_C2HII] - k[2768]*y[IDX_C2DII] - k[2769]*y[IDX_C2HII] -
        k[2770]*y[IDX_C2DII] - k[2771]*y[IDX_C2NII] - k[2772]*y[IDX_C2NII] -
        k[2773]*y[IDX_C2OII] - k[2774]*y[IDX_C3II] - k[2775]*y[IDX_CH2II] -
        k[2776]*y[IDX_CD2II] - k[2777]*y[IDX_CHDII] - k[2778]*y[IDX_CH2II] -
        k[2779]*y[IDX_CD2II] - k[2780]*y[IDX_CHDII] - k[2781]*y[IDX_CHDII] -
        k[2782]*y[IDX_CH2II] - k[2783]*y[IDX_CD2II] - k[2784]*y[IDX_CHDII] -
        k[2785]*y[IDX_CNCII] - k[2786]*y[IDX_CNCII] - k[2787]*y[IDX_CO2II] -
        k[2788]*y[IDX_H2OII] - k[2789]*y[IDX_D2OII] - k[2790]*y[IDX_HDOII] -
        k[2791]*y[IDX_H2OII] - k[2792]*y[IDX_D2OII] - k[2793]*y[IDX_HDOII] -
        k[2794]*y[IDX_HDOII] - k[2795]*y[IDX_H2OII] - k[2796]*y[IDX_D2OII] -
        k[2797]*y[IDX_HDOII] - k[2798]*y[IDX_oH3II] - k[2799]*y[IDX_pH3II] -
        k[2800]*y[IDX_mD3II] - k[2801]*y[IDX_oD3II] - k[2802]*y[IDX_oH2DII] -
        k[2803]*y[IDX_pH2DII] - k[2804]*y[IDX_oD2HII] - k[2805]*y[IDX_pD2HII] -
        k[2806]*y[IDX_oH3II] - k[2807]*y[IDX_pH3II] - k[2808]*y[IDX_pH3II] -
        k[2809]*y[IDX_mD3II] - k[2810]*y[IDX_oD3II] - k[2811]*y[IDX_oD3II] -
        k[2812]*y[IDX_oH2DII] - k[2813]*y[IDX_pH2DII] - k[2814]*y[IDX_oH2DII] -
        k[2815]*y[IDX_pH2DII] - k[2816]*y[IDX_oD2HII] - k[2817]*y[IDX_pD2HII] -
        k[2818]*y[IDX_oD2HII] - k[2819]*y[IDX_pD2HII] - k[2820]*y[IDX_HCNII] -
        k[2821]*y[IDX_DCNII] - k[2822]*y[IDX_HCOII] - k[2823]*y[IDX_DCOII] -
        k[2824]*y[IDX_HNCII] - k[2825]*y[IDX_DNCII] - k[2826]*y[IDX_HNOII] -
        k[2827]*y[IDX_DNOII] - k[2828]*y[IDX_HOCII] - k[2829]*y[IDX_DOCII] -
        k[2830]*y[IDX_N2HII] - k[2831]*y[IDX_N2DII] - k[2832]*y[IDX_N2HII] -
        k[2833]*y[IDX_N2DII] - k[2834]*y[IDX_NCOII] - k[2835]*y[IDX_NH2II] -
        k[2836]*y[IDX_ND2II] - k[2837]*y[IDX_NHDII] - k[2838]*y[IDX_NH2II] -
        k[2839]*y[IDX_ND2II] - k[2840]*y[IDX_NHDII] - k[2841]*y[IDX_NHDII] -
        k[2842]*y[IDX_NO2II] - k[2843]*y[IDX_O2HII] - k[2844]*y[IDX_O2DII] -
        k[2845]*y[IDX_CII] - k[2846]*y[IDX_HII] - k[2847]*y[IDX_DII] -
        k[2848]*y[IDX_HeII] - k[2849]*y[IDX_NII] - k[2850]*y[IDX_OII] -
        k[2914]*y[IDX_pD3II] - k[2915]*y[IDX_pD3II] - k[3025]*y[IDX_H3OII] -
        k[3026]*y[IDX_H3OII] - k[3027]*y[IDX_H3OII] - k[3028]*y[IDX_H3OII] -
        k[3029]*y[IDX_H3OII] - k[3030]*y[IDX_H3OII] - k[3440]*y[IDX_H2DOII] -
        k[3441]*y[IDX_H2DOII] - k[3442]*y[IDX_HD2OII] - k[3443]*y[IDX_HD2OII] -
        k[3444]*y[IDX_D3OII] - k[3445]*y[IDX_H2DOII] - k[3446]*y[IDX_H2DOII] -
        k[3447]*y[IDX_HD2OII] - k[3448]*y[IDX_HD2OII] - k[3449]*y[IDX_D3OII] -
        k[3450]*y[IDX_H2DOII] - k[3451]*y[IDX_H2DOII] - k[3452]*y[IDX_H2DOII] -
        k[3453]*y[IDX_HD2OII] - k[3454]*y[IDX_HD2OII] - k[3455]*y[IDX_HD2OII] -
        k[3456]*y[IDX_D3OII] - k[3457]*y[IDX_D3OII] - k[3458]*y[IDX_H2DOII] -
        k[3459]*y[IDX_H2DOII] - k[3460]*y[IDX_H2DOII] - k[3461]*y[IDX_HD2OII] -
        k[3462]*y[IDX_HD2OII] - k[3463]*y[IDX_HD2OII] - k[3464]*y[IDX_D3OII] -
        k[3465]*y[IDX_D3OII];
    data[5770] = 0.0 - k[2800]*y[IDX_eM] - k[2809]*y[IDX_eM];
    data[5771] = 0.0 + k[216] + k[228] + k[2680]*y[IDX_CM] +
        k[2726]*y[IDX_OM];
    data[5772] = 0.0 - k[2748]*y[IDX_eM] - k[2754]*y[IDX_eM];
    data[5773] = 0.0 - k[2804]*y[IDX_eM] - k[2816]*y[IDX_eM] -
        k[2818]*y[IDX_eM];
    data[5774] = 0.0 - k[2801]*y[IDX_eM] - k[2810]*y[IDX_eM] -
        k[2811]*y[IDX_eM];
    data[5775] = 0.0 + k[214] + k[226] + k[2678]*y[IDX_CM] +
        k[2724]*y[IDX_OM];
    data[5776] = 0.0 - k[2746]*y[IDX_eM] - k[2751]*y[IDX_eM];
    data[5777] = 0.0 - k[2802]*y[IDX_eM] - k[2812]*y[IDX_eM] -
        k[2814]*y[IDX_eM];
    data[5778] = 0.0 - k[2798]*y[IDX_eM] - k[2806]*y[IDX_eM];
    data[5779] = 0.0 + k[215] + k[227] + k[2679]*y[IDX_CM] +
        k[2725]*y[IDX_OM];
    data[5780] = 0.0 - k[2747]*y[IDX_eM] - k[2752]*y[IDX_eM] -
        k[2753]*y[IDX_eM];
    data[5781] = 0.0 - k[2805]*y[IDX_eM] - k[2817]*y[IDX_eM] -
        k[2819]*y[IDX_eM];
    data[5782] = 0.0 - k[2914]*y[IDX_eM] - k[2915]*y[IDX_eM];
    data[5783] = 0.0 + k[213] + k[225] + k[2677]*y[IDX_CM] +
        k[2723]*y[IDX_OM];
    data[5784] = 0.0 - k[2745]*y[IDX_eM] - k[2750]*y[IDX_eM];
    data[5785] = 0.0 - k[2803]*y[IDX_eM] - k[2813]*y[IDX_eM] -
        k[2815]*y[IDX_eM];
    data[5786] = 0.0 - k[2799]*y[IDX_eM] - k[2807]*y[IDX_eM] -
        k[2808]*y[IDX_eM];
    data[5787] = 0.0 - k[1260]*y[IDX_mD3II];
    data[5788] = 0.0 - k[1314]*y[IDX_mD3II];
    data[5789] = 0.0 - k[1346]*y[IDX_mD3II];
    data[5790] = 0.0 - k[1329]*y[IDX_mD3II];
    data[5791] = 0.0 - k[1361]*y[IDX_mD3II];
    data[5792] = 0.0 - k[1376]*y[IDX_mD3II] - k[1391]*y[IDX_mD3II];
    data[5793] = 0.0 + k[2868]*y[IDX_oD2HII];
    data[5794] = 0.0 - k[835]*y[IDX_mD3II];
    data[5795] = 0.0 - k[3264]*y[IDX_mD3II] - k[3265]*y[IDX_mD3II];
    data[5796] = 0.0 + k[804]*y[IDX_pD2II] + k[806]*y[IDX_oD2II];
    data[5797] = 0.0 - k[195]*y[IDX_mD3II] - k[196]*y[IDX_mD3II];
    data[5798] = 0.0 - k[2871]*y[IDX_mD3II];
    data[5799] = 0.0 - k[3192]*y[IDX_mD3II] - k[3196]*y[IDX_mD3II] -
        k[3198]*y[IDX_mD3II] - k[3201]*y[IDX_mD3II];
    data[5800] = 0.0 + k[108]*y[IDX_oD2HII] + k[109]*y[IDX_oD2HII] -
        k[143]*y[IDX_mD3II] - k[144]*y[IDX_mD3II] - k[145]*y[IDX_mD3II] -
        k[146]*y[IDX_mD3II] + k[151]*y[IDX_oD3II] + k[733]*y[IDX_oD2II];
    data[5801] = 0.0 + k[719]*y[IDX_pD2I] + k[721]*y[IDX_oD2I];
    data[5802] = 0.0 - k[3230]*y[IDX_mD3II] - k[3233]*y[IDX_mD3II] -
        k[3234]*y[IDX_mD3II];
    data[5803] = 0.0 + k[820]*y[IDX_pD2I] + k[822]*y[IDX_oD2I];
    data[5804] = 0.0 - k[1274]*y[IDX_mD3II];
    data[5805] = 0.0 - k[1406]*y[IDX_mD3II];
    data[5806] = 0.0 - k[1438]*y[IDX_mD3II];
    data[5807] = 0.0 + k[1847]*y[IDX_pD2I] + k[1849]*y[IDX_oD2I];
    data[5808] = 0.0 - k[1421]*y[IDX_mD3II];
    data[5809] = 0.0 - k[1453]*y[IDX_mD3II];
    data[5810] = 0.0 - k[1515]*y[IDX_mD3II];
    data[5811] = 0.0 - k[1287]*y[IDX_mD3II] - k[1301]*y[IDX_mD3II];
    data[5812] = 0.0 - k[1468]*y[IDX_mD3II];
    data[5813] = 0.0 + k[2025]*y[IDX_pD2I] + k[2027]*y[IDX_oD2I];
    data[5814] = 0.0 - k[1500]*y[IDX_mD3II];
    data[5815] = 0.0 - k[1483]*y[IDX_mD3II];
    data[5816] = 0.0 - k[2800]*y[IDX_mD3II] - k[2809]*y[IDX_mD3II];
    data[5817] = 0.0 - k[112]*y[IDX_pH2I] - k[113]*y[IDX_pH2I] -
        k[114]*y[IDX_oH2I] - k[115]*y[IDX_oH2I] - k[143]*y[IDX_HDI] -
        k[144]*y[IDX_HDI] - k[145]*y[IDX_HDI] - k[146]*y[IDX_HDI] -
        k[152]*y[IDX_pD2I] + k[152]*y[IDX_pD2I] - k[153]*y[IDX_pD2I] -
        k[154]*y[IDX_pD2I] - k[155]*y[IDX_oD2I] + k[155]*y[IDX_oD2I] -
        k[156]*y[IDX_oD2I] - k[157]*y[IDX_oD2I] - k[195]*y[IDX_GRAINM] -
        k[196]*y[IDX_GRAINM] - k[310] - k[311] - k[330] - k[331] -
        k[835]*y[IDX_DM] - k[1260]*y[IDX_CI] - k[1274]*y[IDX_NI] -
        k[1287]*y[IDX_OI] - k[1301]*y[IDX_OI] - k[1314]*y[IDX_C2I] -
        k[1329]*y[IDX_CHI] - k[1346]*y[IDX_CDI] - k[1361]*y[IDX_CNI] -
        k[1376]*y[IDX_COI] - k[1391]*y[IDX_COI] - k[1406]*y[IDX_N2I] -
        k[1421]*y[IDX_NHI] - k[1438]*y[IDX_NDI] - k[1453]*y[IDX_NOI] -
        k[1468]*y[IDX_O2I] - k[1483]*y[IDX_OHI] - k[1500]*y[IDX_ODI] -
        k[1515]*y[IDX_NO2I] - k[2800]*y[IDX_eM] - k[2809]*y[IDX_eM] -
        k[2871]*y[IDX_HI] - k[2908]*y[IDX_pD2I] - k[3192]*y[IDX_H2OI] -
        k[3196]*y[IDX_H2OI] - k[3198]*y[IDX_H2OI] - k[3201]*y[IDX_H2OI] -
        k[3230]*y[IDX_HDOI] - k[3233]*y[IDX_HDOI] - k[3234]*y[IDX_HDOI] -
        k[3264]*y[IDX_D2OI] - k[3265]*y[IDX_D2OI];
    data[5818] = 0.0 + k[85]*y[IDX_pH2DII] + k[94]*y[IDX_oH2DII] +
        k[131]*y[IDX_pD2HII] + k[141]*y[IDX_oD2HII] - k[155]*y[IDX_mD3II] +
        k[155]*y[IDX_mD3II] - k[156]*y[IDX_mD3II] - k[157]*y[IDX_mD3II] +
        k[161]*y[IDX_oD3II] + k[162]*y[IDX_oD3II] + k[715]*y[IDX_pD2II] +
        k[717]*y[IDX_oD2II] + k[721]*y[IDX_HDII] + k[822]*y[IDX_HeDII] +
        k[1849]*y[IDX_NDII] + k[2027]*y[IDX_O2DII] + k[2910]*y[IDX_pD3II];
    data[5819] = 0.0 + k[713]*y[IDX_pD2I] + k[717]*y[IDX_oD2I] +
        k[733]*y[IDX_HDI] + k[806]*y[IDX_DCOI];
    data[5820] = 0.0 + k[108]*y[IDX_HDI] + k[109]*y[IDX_HDI] +
        k[136]*y[IDX_pD2I] + k[141]*y[IDX_oD2I] + k[2868]*y[IDX_DI];
    data[5821] = 0.0 + k[151]*y[IDX_HDI] + k[158]*y[IDX_pD2I] +
        k[159]*y[IDX_pD2I] + k[161]*y[IDX_oD2I] + k[162]*y[IDX_oD2I];
    data[5822] = 0.0 - k[114]*y[IDX_mD3II] - k[115]*y[IDX_mD3II];
    data[5823] = 0.0 + k[94]*y[IDX_oD2I];
    data[5824] = 0.0 + k[136]*y[IDX_oD2HII] - k[152]*y[IDX_mD3II] +
        k[152]*y[IDX_mD3II] - k[153]*y[IDX_mD3II] - k[154]*y[IDX_mD3II] +
        k[158]*y[IDX_oD3II] + k[159]*y[IDX_oD3II] + k[711]*y[IDX_pD2II] +
        k[713]*y[IDX_oD2II] + k[719]*y[IDX_HDII] + k[820]*y[IDX_HeDII] +
        k[1847]*y[IDX_NDII] + k[2025]*y[IDX_O2DII] - k[2908]*y[IDX_mD3II];
    data[5825] = 0.0 + k[711]*y[IDX_pD2I] + k[715]*y[IDX_oD2I] +
        k[804]*y[IDX_DCOI];
    data[5826] = 0.0 + k[131]*y[IDX_oD2I];
    data[5827] = 0.0 + k[2910]*y[IDX_oD2I];
    data[5828] = 0.0 - k[112]*y[IDX_mD3II] - k[113]*y[IDX_mD3II];
    data[5829] = 0.0 + k[85]*y[IDX_oD2I];
    data[5830] = 0.0 - k[1133]*y[IDX_oD2I] + k[1259]*y[IDX_oD3II] +
        k[1266]*y[IDX_oD2HII] + k[1268]*y[IDX_pD2HII] - k[2660]*y[IDX_oD2I] +
        k[2956]*y[IDX_pD3II] + k[3288]*y[IDX_HD2OII] + k[3290]*y[IDX_D3OII];
    data[5831] = 0.0 - k[2644]*y[IDX_oD2I];
    data[5832] = 0.0 - k[2680]*y[IDX_oD2I];
    data[5833] = 0.0 + k[1313]*y[IDX_oD3II] + k[1320]*y[IDX_oD2HII] +
        k[1322]*y[IDX_pD2HII] + k[2343]*y[IDX_oD2II] + k[2961]*y[IDX_pD3II];
    data[5834] = 0.0 - k[1697]*y[IDX_oD2I];
    data[5835] = 0.0 + k[2426]*y[IDX_oD2II];
    data[5836] = 0.0 + k[2421]*y[IDX_oD2II];
    data[5837] = 0.0 - k[1149]*y[IDX_oD2I] + k[1345]*y[IDX_oD3II] +
        k[1352]*y[IDX_oD2HII] + k[1354]*y[IDX_pD2HII] + k[2353]*y[IDX_oD2II] +
        k[2963]*y[IDX_pD3II];
    data[5838] = 0.0 - k[1742]*y[IDX_oD2I];
    data[5839] = 0.0 + k[2436]*y[IDX_oD2II];
    data[5840] = 0.0 - k[1139]*y[IDX_oD2I] - k[1141]*y[IDX_oD2I] +
        k[1335]*y[IDX_oD2HII] + k[1337]*y[IDX_pD2HII] + k[2348]*y[IDX_oD2II];
    data[5841] = 0.0 - k[1738]*y[IDX_oD2I] - k[1740]*y[IDX_oD2I];
    data[5842] = 0.0 + k[2431]*y[IDX_oD2II];
    data[5843] = 0.0 + k[2441]*y[IDX_oD2II];
    data[5844] = 0.0 + k[1360]*y[IDX_oD3II] + k[1367]*y[IDX_oD2HII] +
        k[1369]*y[IDX_pD2HII] + k[2358]*y[IDX_oD2II] + k[2965]*y[IDX_pD3II];
    data[5845] = 0.0 - k[471]*y[IDX_oD2I] - k[477]*y[IDX_oD2I];
    data[5846] = 0.0 + k[1375]*y[IDX_oD3II] + k[1382]*y[IDX_oD2HII] +
        k[1384]*y[IDX_pD2HII] + k[1390]*y[IDX_oD3II] + k[1397]*y[IDX_oD2HII] +
        k[1399]*y[IDX_pD2HII] + k[2363]*y[IDX_oD2II] + k[2967]*y[IDX_pD3II] +
        k[2968]*y[IDX_pD3II];
    data[5847] = 0.0 - k[459]*y[IDX_oD2I] - k[465]*y[IDX_oD2I];
    data[5848] = 0.0 + k[2541]*y[IDX_oD2II];
    data[5849] = 0.0 + k[168]*y[IDX_DI] + k[168]*y[IDX_DI] +
        k[2101]*y[IDX_oD2II] + k[3389]*y[IDX_H2DOII] + k[3393]*y[IDX_HD2OII] +
        k[3394]*y[IDX_D3OII];
    data[5850] = 0.0 + k[2854]*y[IDX_HDI] - k[2872]*y[IDX_oD2I] +
        k[2873]*y[IDX_pD2I];
    data[5851] = 0.0 + k[834]*y[IDX_oD3II] + k[840]*y[IDX_oH2DII] +
        k[843]*y[IDX_pH2DII] + k[855]*y[IDX_oD2HII] + k[857]*y[IDX_pD2HII] +
        k[2157]*y[IDX_oD2II] + k[2983]*y[IDX_pD3II] + k[3067]*y[IDX_H2DOII] +
        k[3078]*y[IDX_HD2OII] + k[3081]*y[IDX_HD2OII] + k[3088]*y[IDX_D3OII] +
        k[3098]*y[IDX_H2DOII] + k[3107]*y[IDX_HD2OII] + k[3111]*y[IDX_D3OII];
    data[5852] = 0.0 + k[2451]*y[IDX_oD2II] + k[3245]*y[IDX_oH3II] +
        k[3246]*y[IDX_pH3II] + k[3252]*y[IDX_pH2DII] + k[3255]*y[IDX_oH2DII] +
        k[3258]*y[IDX_pD2HII] + k[3260]*y[IDX_oD2HII] + k[3263]*y[IDX_pD3II] +
        k[3265]*y[IDX_mD3II] + k[3267]*y[IDX_oD3II];
    data[5853] = 0.0 - k[3145]*y[IDX_oD2I];
    data[5854] = 0.0 + k[3083]*y[IDX_HM] + k[3086]*y[IDX_HM] +
        k[3088]*y[IDX_DM] + k[3110]*y[IDX_HM] + k[3111]*y[IDX_DM] +
        k[3290]*y[IDX_CI] + k[3382]*y[IDX_HI] + k[3394]*y[IDX_DI] -
        k[3438]*y[IDX_oD2I] + k[3457]*y[IDX_eM] + k[3465]*y[IDX_eM];
    data[5855] = 0.0 + k[2111]*y[IDX_oD2II];
    data[5856] = 0.0 + k[2466]*y[IDX_oD2II];
    data[5857] = 0.0 - k[1606]*y[IDX_oD2I] - k[1607]*y[IDX_oD2I] +
        k[1607]*y[IDX_oD2I] + k[1612]*y[IDX_HDI];
    data[5858] = 0.0 + k[174]*y[IDX_oD2II] + k[190]*y[IDX_oD2HII] +
        k[196]*y[IDX_mD3II] + k[198]*y[IDX_oD3II];
    data[5859] = 0.0 + k[2096]*y[IDX_oD2II] + k[3379]*y[IDX_HD2OII] +
        k[3382]*y[IDX_D3OII];
    data[5860] = 0.0 - k[2852]*y[IDX_oD2I];
    data[5861] = 0.0 + k[847]*y[IDX_oD2HII] + k[850]*y[IDX_pD2HII] +
        k[2155]*y[IDX_oD2II] + k[3073]*y[IDX_HD2OII] + k[3083]*y[IDX_D3OII] +
        k[3086]*y[IDX_D3OII] + k[3104]*y[IDX_HD2OII] + k[3110]*y[IDX_D3OII];
    data[5862] = 0.0 + k[3067]*y[IDX_DM] + k[3098]*y[IDX_DM] +
        k[3389]*y[IDX_DI] - k[3428]*y[IDX_oD2I] - k[3430]*y[IDX_oD2I] -
        k[3432]*y[IDX_oD2I];
    data[5863] = 0.0 + k[2446]*y[IDX_oD2II] + k[3189]*y[IDX_oD2HII] +
        k[3201]*y[IDX_mD3II] + k[3202]*y[IDX_oD3II];
    data[5864] = 0.0 - k[3137]*y[IDX_oD2I] - k[3139]*y[IDX_oD2I];
    data[5865] = 0.0 - k[3423]*y[IDX_oD2I] - k[3425]*y[IDX_oD2I] -
        k[3427]*y[IDX_oD2I];
    data[5866] = 0.0 + k[2106]*y[IDX_oD2II];
    data[5867] = 0.0 + k[2461]*y[IDX_oD2II];
    data[5868] = 0.0 + k[46]*y[IDX_pH2DII] + k[53]*y[IDX_oH2DII] +
        k[55]*y[IDX_oH2DII] + k[97]*y[IDX_pD2HII] + k[99]*y[IDX_pD2HII] +
        k[104]*y[IDX_oD2HII] + k[106]*y[IDX_oD2HII] + k[143]*y[IDX_mD3II] +
        k[145]*y[IDX_mD3II] + k[148]*y[IDX_oD3II] + k[150]*y[IDX_oD3II] +
        k[1612]*y[IDX_DOCII] + k[2854]*y[IDX_DII] + k[2940]*y[IDX_pD3II];
    data[5869] = 0.0 - k[720]*y[IDX_oD2I] - k[721]*y[IDX_oD2I] -
        k[724]*y[IDX_oD2I] - k[725]*y[IDX_oD2I] - k[2930]*y[IDX_oD2I];
    data[5870] = 0.0 + k[3073]*y[IDX_HM] + k[3078]*y[IDX_DM] +
        k[3081]*y[IDX_DM] + k[3104]*y[IDX_HM] + k[3107]*y[IDX_DM] +
        k[3288]*y[IDX_CI] + k[3379]*y[IDX_HI] + k[3393]*y[IDX_DI] -
        k[3434]*y[IDX_oD2I] - k[3436]*y[IDX_oD2I] + k[3455]*y[IDX_eM] +
        k[3462]*y[IDX_eM];
    data[5871] = 0.0 + k[2456]*y[IDX_oD2II] + k[3217]*y[IDX_pH2DII] +
        k[3219]*y[IDX_oH2DII] + k[3226]*y[IDX_pD2HII] + k[3228]*y[IDX_oD2HII] +
        k[3233]*y[IDX_mD3II] + k[3235]*y[IDX_pD3II] + k[3237]*y[IDX_oD3II];
    data[5872] = 0.0 - k[3142]*y[IDX_oD2I] - k[3144]*y[IDX_oD2I];
    data[5873] = 0.0 - k[1601]*y[IDX_oD2I] - k[1602]*y[IDX_oD2I] +
        k[1602]*y[IDX_oD2I] - k[1604]*y[IDX_oD2I];
    data[5874] = 0.0 - k[871]*y[IDX_oD2I] - k[2474]*y[IDX_oD2I];
    data[5875] = 0.0 - k[821]*y[IDX_oD2I] - k[822]*y[IDX_oD2I] -
        k[2952]*y[IDX_oD2I];
    data[5876] = 0.0 - k[818]*y[IDX_oD2I] - k[819]*y[IDX_oD2I];
    data[5877] = 0.0 - k[1177]*y[IDX_oD2I];
    data[5878] = 0.0 - k[1639]*y[IDX_oD2I];
    data[5879] = 0.0 + k[1405]*y[IDX_oD3II] + k[1412]*y[IDX_oD2HII] +
        k[1414]*y[IDX_pD2HII] + k[2970]*y[IDX_pD3II];
    data[5880] = 0.0 - k[1809]*y[IDX_oD2I];
    data[5881] = 0.0 - k[1191]*y[IDX_oD2I] + k[1437]*y[IDX_oD3II] +
        k[1444]*y[IDX_oD2HII] + k[1446]*y[IDX_pD2HII] + k[2396]*y[IDX_oD2II] +
        k[2971]*y[IDX_pD3II];
    data[5882] = 0.0 - k[1848]*y[IDX_oD2I] - k[1849]*y[IDX_oD2I] -
        k[1865]*y[IDX_oD2I] - k[2954]*y[IDX_oD2I];
    data[5883] = 0.0 + k[2121]*y[IDX_oD2II];
    data[5884] = 0.0 - k[1187]*y[IDX_oD2I] - k[1189]*y[IDX_oD2I] +
        k[1427]*y[IDX_oD2HII] + k[1429]*y[IDX_pD2HII] + k[2391]*y[IDX_oD2II];
    data[5885] = 0.0 - k[1845]*y[IDX_oD2I] - k[1846]*y[IDX_oD2I] -
        k[1861]*y[IDX_oD2I] - k[1863]*y[IDX_oD2I];
    data[5886] = 0.0 + k[2116]*y[IDX_oD2II];
    data[5887] = 0.0 + k[2126]*y[IDX_oD2II];
    data[5888] = 0.0 + k[1452]*y[IDX_oD3II] + k[1459]*y[IDX_oD2HII] +
        k[1461]*y[IDX_pD2HII] + k[2401]*y[IDX_oD2II] + k[2973]*y[IDX_pD3II];
    data[5889] = 0.0 + k[1514]*y[IDX_oD3II] + k[1521]*y[IDX_oD2HII] +
        k[1523]*y[IDX_pD2HII] + k[2979]*y[IDX_pD3II];
    data[5890] = 0.0 - k[2000]*y[IDX_oD2I];
    data[5891] = 0.0 - k[1155]*y[IDX_oD2I] + k[1286]*y[IDX_oD3II] +
        k[1293]*y[IDX_oD2HII] + k[1295]*y[IDX_pD2HII] + k[2959]*y[IDX_pD3II];
    data[5892] = 0.0 - k[1659]*y[IDX_oD2I];
    data[5893] = 0.0 - k[500]*y[IDX_oD2I] - k[2726]*y[IDX_oD2I];
    data[5894] = 0.0 + k[1467]*y[IDX_oD3II] + k[1474]*y[IDX_oD2HII] +
        k[1476]*y[IDX_pD2HII] + k[2406]*y[IDX_oD2II] + k[2975]*y[IDX_pD3II];
    data[5895] = 0.0 - k[2026]*y[IDX_oD2I] - k[2027]*y[IDX_oD2I] -
        k[2981]*y[IDX_oD2I];
    data[5896] = 0.0 - k[2023]*y[IDX_oD2I] - k[2024]*y[IDX_oD2I];
    data[5897] = 0.0 - k[1169]*y[IDX_oD2I] + k[1499]*y[IDX_oD3II] +
        k[1506]*y[IDX_oD2HII] + k[1508]*y[IDX_pD2HII] + k[2416]*y[IDX_oD2II] +
        k[2977]*y[IDX_pD3II];
    data[5898] = 0.0 - k[1948]*y[IDX_oD2I];
    data[5899] = 0.0 - k[1165]*y[IDX_oD2I] - k[1167]*y[IDX_oD2I] +
        k[1489]*y[IDX_oD2HII] + k[1491]*y[IDX_pD2HII] + k[2411]*y[IDX_oD2II];
    data[5900] = 0.0 - k[1944]*y[IDX_oD2I] - k[1946]*y[IDX_oD2I];
    data[5901] = 0.0 + k[2753]*y[IDX_pD2II] + k[2754]*y[IDX_oD2II] +
        k[2809]*y[IDX_mD3II] + k[2811]*y[IDX_oD3II] + k[2816]*y[IDX_oD2HII] +
        k[3455]*y[IDX_HD2OII] + k[3457]*y[IDX_D3OII] + k[3462]*y[IDX_HD2OII] +
        k[3465]*y[IDX_D3OII];
    data[5902] = 0.0 + k[112]*y[IDX_pH2I] + k[114]*y[IDX_oH2I] +
        k[143]*y[IDX_HDI] + k[145]*y[IDX_HDI] + k[152]*y[IDX_pD2I] +
        k[154]*y[IDX_pD2I] - k[155]*y[IDX_oD2I] - k[156]*y[IDX_oD2I] -
        k[157]*y[IDX_oD2I] + k[157]*y[IDX_oD2I] + k[196]*y[IDX_GRAINM] + k[311]
        + k[2809]*y[IDX_eM] + k[2908]*y[IDX_pD2I] + k[3201]*y[IDX_H2OI] +
        k[3233]*y[IDX_HDOI] + k[3265]*y[IDX_D2OI];
    data[5903] = 0.0 - k[37]*y[IDX_pH3II] - k[38]*y[IDX_pH3II] -
        k[39]*y[IDX_pH3II] - k[40]*y[IDX_pH3II] - k[43]*y[IDX_oH3II] -
        k[44]*y[IDX_oH3II] - k[82]*y[IDX_pH2DII] - k[83]*y[IDX_pH2DII] -
        k[84]*y[IDX_pH2DII] - k[85]*y[IDX_pH2DII] - k[86]*y[IDX_pH2DII] -
        k[91]*y[IDX_oH2DII] - k[92]*y[IDX_oH2DII] - k[93]*y[IDX_oH2DII] -
        k[94]*y[IDX_oH2DII] - k[95]*y[IDX_oH2DII] - k[128]*y[IDX_pD2HII] -
        k[129]*y[IDX_pD2HII] - k[130]*y[IDX_pD2HII] + k[130]*y[IDX_pD2HII] -
        k[131]*y[IDX_pD2HII] - k[132]*y[IDX_pD2HII] - k[138]*y[IDX_oD2HII] -
        k[139]*y[IDX_oD2HII] + k[139]*y[IDX_oD2HII] - k[140]*y[IDX_oD2HII] -
        k[141]*y[IDX_oD2HII] - k[142]*y[IDX_oD2HII] - k[155]*y[IDX_mD3II] -
        k[156]*y[IDX_mD3II] - k[157]*y[IDX_mD3II] + k[157]*y[IDX_mD3II] -
        k[161]*y[IDX_oD3II] - k[162]*y[IDX_oD3II] + k[162]*y[IDX_oD3II] -
        k[163]*y[IDX_oD3II] - k[211] - k[216] - k[222] - k[228] - k[363] -
        k[459]*y[IDX_COII] - k[465]*y[IDX_COII] - k[471]*y[IDX_CNII] -
        k[477]*y[IDX_CNII] - k[500]*y[IDX_OM] - k[702]*y[IDX_pH2II] -
        k[703]*y[IDX_pH2II] - k[704]*y[IDX_oH2II] - k[705]*y[IDX_oH2II] -
        k[708]*y[IDX_pH2II] - k[709]*y[IDX_oH2II] - k[714]*y[IDX_pD2II] -
        k[715]*y[IDX_pD2II] - k[716]*y[IDX_oD2II] - k[717]*y[IDX_oD2II] -
        k[720]*y[IDX_HDII] - k[721]*y[IDX_HDII] - k[724]*y[IDX_HDII] -
        k[725]*y[IDX_HDII] - k[818]*y[IDX_HeHII] - k[819]*y[IDX_HeHII] -
        k[821]*y[IDX_HeDII] - k[822]*y[IDX_HeDII] - k[871]*y[IDX_HeII] -
        k[1133]*y[IDX_CI] - k[1139]*y[IDX_CHI] - k[1141]*y[IDX_CHI] -
        k[1149]*y[IDX_CDI] - k[1155]*y[IDX_OI] - k[1165]*y[IDX_OHI] -
        k[1167]*y[IDX_OHI] - k[1169]*y[IDX_ODI] - k[1177]*y[IDX_NI] -
        k[1187]*y[IDX_NHI] - k[1189]*y[IDX_NHI] - k[1191]*y[IDX_NDI] -
        k[1601]*y[IDX_HOCII] - k[1602]*y[IDX_HOCII] + k[1602]*y[IDX_HOCII] -
        k[1604]*y[IDX_HOCII] - k[1606]*y[IDX_DOCII] - k[1607]*y[IDX_DOCII] +
        k[1607]*y[IDX_DOCII] - k[1639]*y[IDX_NII] - k[1659]*y[IDX_OII] -
        k[1697]*y[IDX_C2II] - k[1738]*y[IDX_CHII] - k[1740]*y[IDX_CHII] -
        k[1742]*y[IDX_CDII] - k[1809]*y[IDX_N2II] - k[1845]*y[IDX_NHII] -
        k[1846]*y[IDX_NHII] - k[1848]*y[IDX_NDII] - k[1849]*y[IDX_NDII] -
        k[1861]*y[IDX_NHII] - k[1863]*y[IDX_NHII] - k[1865]*y[IDX_NDII] -
        k[1944]*y[IDX_OHII] - k[1946]*y[IDX_OHII] - k[1948]*y[IDX_ODII] -
        k[2000]*y[IDX_NO2II] - k[2023]*y[IDX_O2HII] - k[2024]*y[IDX_O2HII] -
        k[2026]*y[IDX_O2DII] - k[2027]*y[IDX_O2DII] - k[2474]*y[IDX_HeII] -
        k[2644]*y[IDX_CII] - k[2660]*y[IDX_CI] - k[2680]*y[IDX_CM] -
        k[2726]*y[IDX_OM] - k[2852]*y[IDX_HII] - k[2872]*y[IDX_DII] -
        k[2906]*y[IDX_pD2HII] - k[2910]*y[IDX_pD3II] - k[2911]*y[IDX_pD3II] -
        k[2930]*y[IDX_HDII] - k[2931]*y[IDX_oD2II] - k[2934]*y[IDX_pD2II] -
        k[2946]*y[IDX_pD3II] + k[2946]*y[IDX_pD3II] - k[2948]*y[IDX_oD3II] -
        k[2949]*y[IDX_oD3II] + k[2949]*y[IDX_oD3II] - k[2952]*y[IDX_HeDII] -
        k[2954]*y[IDX_NDII] - k[2981]*y[IDX_O2DII] - k[3137]*y[IDX_H2OII] -
        k[3139]*y[IDX_H2OII] - k[3142]*y[IDX_HDOII] - k[3144]*y[IDX_HDOII] -
        k[3145]*y[IDX_D2OII] - k[3423]*y[IDX_H3OII] - k[3425]*y[IDX_H3OII] -
        k[3427]*y[IDX_H3OII] - k[3428]*y[IDX_H2DOII] - k[3430]*y[IDX_H2DOII] -
        k[3432]*y[IDX_H2DOII] - k[3434]*y[IDX_HD2OII] - k[3436]*y[IDX_HD2OII] -
        k[3438]*y[IDX_D3OII];
    data[5904] = 0.0 + k[174]*y[IDX_GRAINM] - k[716]*y[IDX_oD2I] -
        k[717]*y[IDX_oD2I] + k[2096]*y[IDX_HI] + k[2101]*y[IDX_DI] +
        k[2106]*y[IDX_HCNI] + k[2111]*y[IDX_DCNI] + k[2116]*y[IDX_NH2I] +
        k[2121]*y[IDX_ND2I] + k[2126]*y[IDX_NHDI] + k[2155]*y[IDX_HM] +
        k[2157]*y[IDX_DM] + k[2343]*y[IDX_C2I] + k[2348]*y[IDX_CHI] +
        k[2353]*y[IDX_CDI] + k[2358]*y[IDX_CNI] + k[2363]*y[IDX_COI] +
        k[2391]*y[IDX_NHI] + k[2396]*y[IDX_NDI] + k[2401]*y[IDX_NOI] +
        k[2406]*y[IDX_O2I] + k[2411]*y[IDX_OHI] + k[2416]*y[IDX_ODI] +
        k[2421]*y[IDX_C2HI] + k[2426]*y[IDX_C2DI] + k[2431]*y[IDX_CH2I] +
        k[2436]*y[IDX_CD2I] + k[2441]*y[IDX_CHDI] + k[2446]*y[IDX_H2OI] +
        k[2451]*y[IDX_D2OI] + k[2456]*y[IDX_HDOI] + k[2461]*y[IDX_HCOI] +
        k[2466]*y[IDX_DCOI] + k[2541]*y[IDX_CO2I] + k[2754]*y[IDX_eM] -
        k[2931]*y[IDX_oD2I];
    data[5905] = 0.0 + k[69]*y[IDX_pH2I] + k[73]*y[IDX_oH2I] +
        k[74]*y[IDX_oH2I] + k[104]*y[IDX_HDI] + k[106]*y[IDX_HDI] +
        k[134]*y[IDX_pD2I] + k[135]*y[IDX_pD2I] - k[138]*y[IDX_oD2I] -
        k[139]*y[IDX_oD2I] + k[139]*y[IDX_oD2I] - k[140]*y[IDX_oD2I] -
        k[141]*y[IDX_oD2I] - k[142]*y[IDX_oD2I] + k[190]*y[IDX_GRAINM] + k[319]
        + k[847]*y[IDX_HM] + k[855]*y[IDX_DM] + k[1266]*y[IDX_CI] +
        k[1293]*y[IDX_OI] + k[1320]*y[IDX_C2I] + k[1335]*y[IDX_CHI] +
        k[1352]*y[IDX_CDI] + k[1367]*y[IDX_CNI] + k[1382]*y[IDX_COI] +
        k[1397]*y[IDX_COI] + k[1412]*y[IDX_N2I] + k[1427]*y[IDX_NHI] +
        k[1444]*y[IDX_NDI] + k[1459]*y[IDX_NOI] + k[1474]*y[IDX_O2I] +
        k[1489]*y[IDX_OHI] + k[1506]*y[IDX_ODI] + k[1521]*y[IDX_NO2I] +
        k[2816]*y[IDX_eM] + k[3189]*y[IDX_H2OI] + k[3228]*y[IDX_HDOI] +
        k[3260]*y[IDX_D2OI];
    data[5906] = 0.0 + k[117]*y[IDX_pH2I] + k[121]*y[IDX_oH2I] +
        k[148]*y[IDX_HDI] + k[150]*y[IDX_HDI] + k[159]*y[IDX_pD2I] +
        k[160]*y[IDX_pD2I] - k[161]*y[IDX_oD2I] - k[162]*y[IDX_oD2I] +
        k[162]*y[IDX_oD2I] - k[163]*y[IDX_oD2I] + k[198]*y[IDX_GRAINM] + k[309]
        + k[834]*y[IDX_DM] + k[1259]*y[IDX_CI] + k[1286]*y[IDX_OI] +
        k[1313]*y[IDX_C2I] + k[1345]*y[IDX_CDI] + k[1360]*y[IDX_CNI] +
        k[1375]*y[IDX_COI] + k[1390]*y[IDX_COI] + k[1405]*y[IDX_N2I] +
        k[1437]*y[IDX_NDI] + k[1452]*y[IDX_NOI] + k[1467]*y[IDX_O2I] +
        k[1499]*y[IDX_ODI] + k[1514]*y[IDX_NO2I] + k[2811]*y[IDX_eM] +
        k[2909]*y[IDX_pD2I] - k[2948]*y[IDX_oD2I] - k[2949]*y[IDX_oD2I] +
        k[2949]*y[IDX_oD2I] + k[3202]*y[IDX_H2OI] + k[3237]*y[IDX_HDOI] +
        k[3267]*y[IDX_D2OI];
    data[5907] = 0.0 + k[73]*y[IDX_oD2HII] + k[74]*y[IDX_oD2HII] +
        k[114]*y[IDX_mD3II] + k[121]*y[IDX_oD3II];
    data[5908] = 0.0 - k[704]*y[IDX_oD2I] - k[705]*y[IDX_oD2I] -
        k[709]*y[IDX_oD2I];
    data[5909] = 0.0 + k[53]*y[IDX_HDI] + k[55]*y[IDX_HDI] +
        k[87]*y[IDX_pD2I] - k[91]*y[IDX_oD2I] - k[92]*y[IDX_oD2I] -
        k[93]*y[IDX_oD2I] - k[94]*y[IDX_oD2I] - k[95]*y[IDX_oD2I] +
        k[840]*y[IDX_DM] + k[3219]*y[IDX_HDOI] + k[3255]*y[IDX_D2OI];
    data[5910] = 0.0 - k[43]*y[IDX_oD2I] - k[44]*y[IDX_oD2I] +
        k[3245]*y[IDX_D2OI];
    data[5911] = 0.0 + k[78]*y[IDX_pH2DII] + k[87]*y[IDX_oH2DII] +
        k[124]*y[IDX_pD2HII] + k[126]*y[IDX_pD2HII] + k[134]*y[IDX_oD2HII] +
        k[135]*y[IDX_oD2HII] + k[152]*y[IDX_mD3II] + k[154]*y[IDX_mD3II] +
        k[159]*y[IDX_oD3II] + k[160]*y[IDX_oD3II] + k[2873]*y[IDX_DII] +
        k[2908]*y[IDX_mD3II] + k[2909]*y[IDX_oD3II] + k[2945]*y[IDX_pD3II];
    data[5912] = 0.0 - k[714]*y[IDX_oD2I] - k[715]*y[IDX_oD2I] +
        k[2753]*y[IDX_eM] - k[2934]*y[IDX_oD2I];
    data[5913] = 0.0 + k[97]*y[IDX_HDI] + k[99]*y[IDX_HDI] +
        k[124]*y[IDX_pD2I] + k[126]*y[IDX_pD2I] - k[128]*y[IDX_oD2I] -
        k[129]*y[IDX_oD2I] - k[130]*y[IDX_oD2I] + k[130]*y[IDX_oD2I] -
        k[131]*y[IDX_oD2I] - k[132]*y[IDX_oD2I] + k[321] + k[850]*y[IDX_HM] +
        k[857]*y[IDX_DM] + k[1268]*y[IDX_CI] + k[1295]*y[IDX_OI] +
        k[1322]*y[IDX_C2I] + k[1337]*y[IDX_CHI] + k[1354]*y[IDX_CDI] +
        k[1369]*y[IDX_CNI] + k[1384]*y[IDX_COI] + k[1399]*y[IDX_COI] +
        k[1414]*y[IDX_N2I] + k[1429]*y[IDX_NHI] + k[1446]*y[IDX_NDI] +
        k[1461]*y[IDX_NOI] + k[1476]*y[IDX_O2I] + k[1491]*y[IDX_OHI] +
        k[1508]*y[IDX_ODI] + k[1523]*y[IDX_NO2I] - k[2906]*y[IDX_oD2I] +
        k[3226]*y[IDX_HDOI] + k[3258]*y[IDX_D2OI];
    data[5914] = 0.0 - k[2910]*y[IDX_oD2I] - k[2911]*y[IDX_oD2I] +
        k[2940]*y[IDX_HDI] + k[2945]*y[IDX_pD2I] - k[2946]*y[IDX_oD2I] +
        k[2946]*y[IDX_oD2I] + k[2956]*y[IDX_CI] + k[2959]*y[IDX_OI] +
        k[2961]*y[IDX_C2I] + k[2963]*y[IDX_CDI] + k[2965]*y[IDX_CNI] +
        k[2967]*y[IDX_COI] + k[2968]*y[IDX_COI] + k[2970]*y[IDX_N2I] +
        k[2971]*y[IDX_NDI] + k[2973]*y[IDX_NOI] + k[2975]*y[IDX_O2I] +
        k[2977]*y[IDX_ODI] + k[2979]*y[IDX_NO2I] + k[2983]*y[IDX_DM] + k[2985] +
        k[3235]*y[IDX_HDOI] + k[3263]*y[IDX_D2OI];
    data[5915] = 0.0 + k[69]*y[IDX_oD2HII] + k[112]*y[IDX_mD3II] +
        k[117]*y[IDX_oD3II];
    data[5916] = 0.0 - k[702]*y[IDX_oD2I] - k[703]*y[IDX_oD2I] -
        k[708]*y[IDX_oD2I];
    data[5917] = 0.0 + k[46]*y[IDX_HDI] + k[78]*y[IDX_pD2I] -
        k[82]*y[IDX_oD2I] - k[83]*y[IDX_oD2I] - k[84]*y[IDX_oD2I] -
        k[85]*y[IDX_oD2I] - k[86]*y[IDX_oD2I] + k[843]*y[IDX_DM] +
        k[3217]*y[IDX_HDOI] + k[3252]*y[IDX_D2OI];
    data[5918] = 0.0 - k[37]*y[IDX_oD2I] - k[38]*y[IDX_oD2I] -
        k[39]*y[IDX_oD2I] - k[40]*y[IDX_oD2I] + k[3246]*y[IDX_D2OI];
    data[5919] = 0.0 - k[633]*y[IDX_oD2II];
    data[5920] = 0.0 - k[651]*y[IDX_oD2II] - k[2343]*y[IDX_oD2II];
    data[5921] = 0.0 - k[2426]*y[IDX_oD2II];
    data[5922] = 0.0 - k[2421]*y[IDX_oD2II];
    data[5923] = 0.0 - k[667]*y[IDX_oD2II] - k[2353]*y[IDX_oD2II];
    data[5924] = 0.0 - k[2436]*y[IDX_oD2II];
    data[5925] = 0.0 - k[657]*y[IDX_oD2II] - k[659]*y[IDX_oD2II] -
        k[2348]*y[IDX_oD2II];
    data[5926] = 0.0 - k[2431]*y[IDX_oD2II];
    data[5927] = 0.0 - k[2441]*y[IDX_oD2II];
    data[5928] = 0.0 - k[673]*y[IDX_oD2II] - k[2358]*y[IDX_oD2II];
    data[5929] = 0.0 - k[679]*y[IDX_oD2II] - k[2363]*y[IDX_oD2II];
    data[5930] = 0.0 - k[793]*y[IDX_oD2II] - k[2541]*y[IDX_oD2II];
    data[5931] = 0.0 - k[2101]*y[IDX_oD2II] + k[2651]*y[IDX_DII] +
        k[2864]*y[IDX_HDII];
    data[5932] = 0.0 + k[2651]*y[IDX_DI];
    data[5933] = 0.0 - k[2157]*y[IDX_oD2II];
    data[5934] = 0.0 - k[2451]*y[IDX_oD2II] - k[3053]*y[IDX_oD2II];
    data[5935] = 0.0 - k[2111]*y[IDX_oD2II];
    data[5936] = 0.0 - k[805]*y[IDX_oD2II] - k[806]*y[IDX_oD2II] -
        k[2466]*y[IDX_oD2II] - k[2950]*y[IDX_oD2II];
    data[5937] = 0.0 - k[173]*y[IDX_oD2II] - k[174]*y[IDX_oD2II];
    data[5938] = 0.0 - k[2096]*y[IDX_oD2II] - k[2866]*y[IDX_oD2II];
    data[5939] = 0.0 - k[2155]*y[IDX_oD2II];
    data[5940] = 0.0 - k[2446]*y[IDX_oD2II] - k[3033]*y[IDX_oD2II] -
        k[3035]*y[IDX_oD2II];
    data[5941] = 0.0 - k[2106]*y[IDX_oD2II];
    data[5942] = 0.0 - k[797]*y[IDX_oD2II] - k[798]*y[IDX_oD2II] -
        k[2461]*y[IDX_oD2II];
    data[5943] = 0.0 - k[732]*y[IDX_oD2II] - k[733]*y[IDX_oD2II] -
        k[735]*y[IDX_oD2II];
    data[5944] = 0.0 + k[2864]*y[IDX_DI];
    data[5945] = 0.0 - k[2456]*y[IDX_oD2II] - k[3044]*y[IDX_oD2II] -
        k[3046]*y[IDX_oD2II];
    data[5946] = 0.0 + k[2474]*y[IDX_oD2I];
    data[5947] = 0.0 - k[639]*y[IDX_oD2II];
    data[5948] = 0.0 - k[743]*y[IDX_oD2II];
    data[5949] = 0.0 - k[759]*y[IDX_oD2II] - k[2396]*y[IDX_oD2II];
    data[5950] = 0.0 - k[2121]*y[IDX_oD2II];
    data[5951] = 0.0 - k[749]*y[IDX_oD2II] - k[751]*y[IDX_oD2II] -
        k[2391]*y[IDX_oD2II];
    data[5952] = 0.0 - k[2116]*y[IDX_oD2II];
    data[5953] = 0.0 - k[2126]*y[IDX_oD2II];
    data[5954] = 0.0 - k[765]*y[IDX_oD2II] - k[2401]*y[IDX_oD2II];
    data[5955] = 0.0 - k[645]*y[IDX_oD2II];
    data[5956] = 0.0 - k[771]*y[IDX_oD2II] - k[2406]*y[IDX_oD2II];
    data[5957] = 0.0 - k[787]*y[IDX_oD2II] - k[2416]*y[IDX_oD2II];
    data[5958] = 0.0 - k[777]*y[IDX_oD2II] - k[779]*y[IDX_oD2II] -
        k[2411]*y[IDX_oD2II];
    data[5959] = 0.0 - k[2748]*y[IDX_oD2II] - k[2754]*y[IDX_oD2II];
    data[5960] = 0.0 + k[331];
    data[5961] = 0.0 + k[228] - k[716]*y[IDX_oD2II] - k[717]*y[IDX_oD2II] +
        k[2474]*y[IDX_HeII] - k[2931]*y[IDX_oD2II];
    data[5962] = 0.0 - k[173]*y[IDX_GRAINM] - k[174]*y[IDX_GRAINM] - k[293]
        - k[633]*y[IDX_CI] - k[639]*y[IDX_NI] - k[645]*y[IDX_OI] -
        k[651]*y[IDX_C2I] - k[657]*y[IDX_CHI] - k[659]*y[IDX_CHI] -
        k[667]*y[IDX_CDI] - k[673]*y[IDX_CNI] - k[679]*y[IDX_COI] -
        k[686]*y[IDX_pH2I] - k[688]*y[IDX_oH2I] - k[690]*y[IDX_pH2I] -
        k[692]*y[IDX_oH2I] - k[712]*y[IDX_pD2I] - k[713]*y[IDX_pD2I] -
        k[716]*y[IDX_oD2I] - k[717]*y[IDX_oD2I] - k[732]*y[IDX_HDI] -
        k[733]*y[IDX_HDI] - k[735]*y[IDX_HDI] - k[743]*y[IDX_N2I] -
        k[749]*y[IDX_NHI] - k[751]*y[IDX_NHI] - k[759]*y[IDX_NDI] -
        k[765]*y[IDX_NOI] - k[771]*y[IDX_O2I] - k[777]*y[IDX_OHI] -
        k[779]*y[IDX_OHI] - k[787]*y[IDX_ODI] - k[793]*y[IDX_CO2I] -
        k[797]*y[IDX_HCOI] - k[798]*y[IDX_HCOI] - k[805]*y[IDX_DCOI] -
        k[806]*y[IDX_DCOI] - k[2096]*y[IDX_HI] - k[2101]*y[IDX_DI] -
        k[2106]*y[IDX_HCNI] - k[2111]*y[IDX_DCNI] - k[2116]*y[IDX_NH2I] -
        k[2121]*y[IDX_ND2I] - k[2126]*y[IDX_NHDI] - k[2155]*y[IDX_HM] -
        k[2157]*y[IDX_DM] - k[2343]*y[IDX_C2I] - k[2348]*y[IDX_CHI] -
        k[2353]*y[IDX_CDI] - k[2358]*y[IDX_CNI] - k[2363]*y[IDX_COI] -
        k[2391]*y[IDX_NHI] - k[2396]*y[IDX_NDI] - k[2401]*y[IDX_NOI] -
        k[2406]*y[IDX_O2I] - k[2411]*y[IDX_OHI] - k[2416]*y[IDX_ODI] -
        k[2421]*y[IDX_C2HI] - k[2426]*y[IDX_C2DI] - k[2431]*y[IDX_CH2I] -
        k[2436]*y[IDX_CD2I] - k[2441]*y[IDX_CHDI] - k[2446]*y[IDX_H2OI] -
        k[2451]*y[IDX_D2OI] - k[2456]*y[IDX_HDOI] - k[2461]*y[IDX_HCOI] -
        k[2466]*y[IDX_DCOI] - k[2541]*y[IDX_CO2I] - k[2748]*y[IDX_eM] -
        k[2754]*y[IDX_eM] - k[2866]*y[IDX_HI] - k[2931]*y[IDX_oD2I] -
        k[2933]*y[IDX_pD2I] - k[2950]*y[IDX_DCOI] - k[3033]*y[IDX_H2OI] -
        k[3035]*y[IDX_H2OI] - k[3044]*y[IDX_HDOI] - k[3046]*y[IDX_HDOI] -
        k[3053]*y[IDX_D2OI];
    data[5963] = 0.0 + k[339];
    data[5964] = 0.0 + k[329];
    data[5965] = 0.0 - k[688]*y[IDX_oD2II] - k[692]*y[IDX_oD2II];
    data[5966] = 0.0 - k[712]*y[IDX_oD2II] - k[713]*y[IDX_oD2II] -
        k[2933]*y[IDX_oD2II];
    data[5967] = 0.0 + k[341];
    data[5968] = 0.0 + k[2987];
    data[5969] = 0.0 - k[686]*y[IDX_oD2II] - k[690]*y[IDX_oD2II];
    data[5970] = 0.0 - k[1266]*y[IDX_oD2HII] - k[1269]*y[IDX_oD2HII];
    data[5971] = 0.0 - k[1320]*y[IDX_oD2HII] - k[1323]*y[IDX_oD2HII];
    data[5972] = 0.0 - k[1352]*y[IDX_oD2HII] - k[1355]*y[IDX_oD2HII];
    data[5973] = 0.0 - k[1335]*y[IDX_oD2HII] - k[1338]*y[IDX_oD2HII];
    data[5974] = 0.0 - k[1367]*y[IDX_oD2HII] - k[1370]*y[IDX_oD2HII];
    data[5975] = 0.0 - k[1382]*y[IDX_oD2HII] - k[1385]*y[IDX_oD2HII] -
        k[1397]*y[IDX_oD2HII] - k[1400]*y[IDX_oD2HII];
    data[5976] = 0.0 + k[2856]*y[IDX_pH2DII] + k[2858]*y[IDX_oH2DII] -
        k[2867]*y[IDX_oD2HII] - k[2868]*y[IDX_oD2HII];
    data[5977] = 0.0 - k[854]*y[IDX_oD2HII] - k[855]*y[IDX_oD2HII];
    data[5978] = 0.0 - k[3257]*y[IDX_oD2HII] - k[3260]*y[IDX_oD2HII] -
        k[3261]*y[IDX_oD2HII];
    data[5979] = 0.0 + k[807]*y[IDX_HDII];
    data[5980] = 0.0 - k[189]*y[IDX_oD2HII] - k[190]*y[IDX_oD2HII] -
        k[191]*y[IDX_oD2HII];
    data[5981] = 0.0 - k[2860]*y[IDX_oD2HII] - k[2862]*y[IDX_oD2HII] +
        k[2871]*y[IDX_mD3II];
    data[5982] = 0.0 - k[847]*y[IDX_oD2HII] - k[848]*y[IDX_oD2HII] -
        k[852]*y[IDX_oD2HII];
    data[5983] = 0.0 - k[3183]*y[IDX_oD2HII] - k[3184]*y[IDX_oD2HII] -
        k[3187]*y[IDX_oD2HII] - k[3189]*y[IDX_oD2HII];
    data[5984] = 0.0 + k[797]*y[IDX_oD2II];
    data[5985] = 0.0 + k[50]*y[IDX_pH2DII] + k[51]*y[IDX_pH2DII] +
        k[59]*y[IDX_oH2DII] + k[60]*y[IDX_oH2DII] + k[100]*y[IDX_pD2HII] -
        k[103]*y[IDX_oD2HII] - k[104]*y[IDX_oD2HII] - k[105]*y[IDX_oD2HII] -
        k[106]*y[IDX_oD2HII] - k[107]*y[IDX_oD2HII] - k[108]*y[IDX_oD2HII] -
        k[109]*y[IDX_oD2HII] - k[110]*y[IDX_oD2HII] - k[111]*y[IDX_oD2HII] +
        k[144]*y[IDX_mD3II] + k[145]*y[IDX_mD3II] + k[149]*y[IDX_oD3II] +
        k[150]*y[IDX_oD3II] + k[735]*y[IDX_oD2II] + k[736]*y[IDX_HDII] +
        k[825]*y[IDX_HeDII] + k[1852]*y[IDX_NDII] + k[2030]*y[IDX_O2DII] +
        k[2941]*y[IDX_pD3II];
    data[5986] = 0.0 + k[722]*y[IDX_pD2I] + k[724]*y[IDX_oD2I] +
        k[736]*y[IDX_HDI] + k[807]*y[IDX_DCOI];
    data[5987] = 0.0 - k[3222]*y[IDX_oD2HII] - k[3223]*y[IDX_oD2HII] -
        k[3225]*y[IDX_oD2HII] - k[3228]*y[IDX_oD2HII] - k[3229]*y[IDX_oD2HII];
    data[5988] = 0.0 + k[825]*y[IDX_HDI];
    data[5989] = 0.0 + k[818]*y[IDX_oD2I];
    data[5990] = 0.0 - k[1279]*y[IDX_oD2HII] - k[1281]*y[IDX_oD2HII];
    data[5991] = 0.0 - k[1412]*y[IDX_oD2HII] - k[1415]*y[IDX_oD2HII];
    data[5992] = 0.0 - k[1444]*y[IDX_oD2HII] - k[1447]*y[IDX_oD2HII];
    data[5993] = 0.0 + k[1852]*y[IDX_HDI];
    data[5994] = 0.0 - k[1427]*y[IDX_oD2HII] - k[1430]*y[IDX_oD2HII];
    data[5995] = 0.0 + k[1845]*y[IDX_oD2I];
    data[5996] = 0.0 - k[1459]*y[IDX_oD2HII] - k[1462]*y[IDX_oD2HII];
    data[5997] = 0.0 - k[1521]*y[IDX_oD2HII] - k[1524]*y[IDX_oD2HII];
    data[5998] = 0.0 - k[1293]*y[IDX_oD2HII] - k[1296]*y[IDX_oD2HII] -
        k[1306]*y[IDX_oD2HII] - k[1308]*y[IDX_oD2HII];
    data[5999] = 0.0 - k[1474]*y[IDX_oD2HII] - k[1477]*y[IDX_oD2HII];
    data[6000] = 0.0 + k[2030]*y[IDX_HDI];
    data[6001] = 0.0 + k[2023]*y[IDX_oD2I];
    data[6002] = 0.0 - k[1506]*y[IDX_oD2HII] - k[1509]*y[IDX_oD2HII];
    data[6003] = 0.0 - k[1489]*y[IDX_oD2HII] - k[1492]*y[IDX_oD2HII];
    data[6004] = 0.0 - k[2804]*y[IDX_oD2HII] - k[2816]*y[IDX_oD2HII] -
        k[2818]*y[IDX_oD2HII];
    data[6005] = 0.0 + k[113]*y[IDX_pH2I] + k[115]*y[IDX_oH2I] +
        k[144]*y[IDX_HDI] + k[145]*y[IDX_HDI] + k[2871]*y[IDX_HI];
    data[6006] = 0.0 + k[39]*y[IDX_pH3II] + k[40]*y[IDX_pH3II] +
        k[44]*y[IDX_oH3II] + k[84]*y[IDX_pH2DII] + k[93]*y[IDX_oH2DII] +
        k[129]*y[IDX_pD2HII] + k[130]*y[IDX_pD2HII] - k[138]*y[IDX_oD2HII] -
        k[139]*y[IDX_oD2HII] - k[140]*y[IDX_oD2HII] + k[140]*y[IDX_oD2HII] -
        k[141]*y[IDX_oD2HII] - k[142]*y[IDX_oD2HII] + k[702]*y[IDX_pH2II] +
        k[704]*y[IDX_oH2II] + k[724]*y[IDX_HDII] + k[818]*y[IDX_HeHII] +
        k[1845]*y[IDX_NHII] + k[2023]*y[IDX_O2HII];
    data[6007] = 0.0 + k[686]*y[IDX_pH2I] + k[688]*y[IDX_oH2I] +
        k[735]*y[IDX_HDI] + k[797]*y[IDX_HCOI];
    data[6008] = 0.0 - k[69]*y[IDX_pH2I] - k[70]*y[IDX_pH2I] -
        k[71]*y[IDX_pH2I] - k[72]*y[IDX_pH2I] + k[72]*y[IDX_pH2I] -
        k[73]*y[IDX_oH2I] - k[74]*y[IDX_oH2I] - k[75]*y[IDX_oH2I] -
        k[76]*y[IDX_oH2I] - k[77]*y[IDX_oH2I] + k[77]*y[IDX_oH2I] -
        k[103]*y[IDX_HDI] - k[104]*y[IDX_HDI] - k[105]*y[IDX_HDI] -
        k[106]*y[IDX_HDI] - k[107]*y[IDX_HDI] - k[108]*y[IDX_HDI] -
        k[109]*y[IDX_HDI] - k[110]*y[IDX_HDI] - k[111]*y[IDX_HDI] -
        k[133]*y[IDX_pD2I] - k[134]*y[IDX_pD2I] - k[135]*y[IDX_pD2I] +
        k[135]*y[IDX_pD2I] - k[136]*y[IDX_pD2I] - k[137]*y[IDX_pD2I] -
        k[138]*y[IDX_oD2I] - k[139]*y[IDX_oD2I] - k[140]*y[IDX_oD2I] +
        k[140]*y[IDX_oD2I] - k[141]*y[IDX_oD2I] - k[142]*y[IDX_oD2I] -
        k[189]*y[IDX_GRAINM] - k[190]*y[IDX_GRAINM] - k[191]*y[IDX_GRAINM] -
        k[318] - k[319] - k[322] - k[338] - k[339] - k[342] - k[847]*y[IDX_HM] -
        k[848]*y[IDX_HM] - k[852]*y[IDX_HM] - k[854]*y[IDX_DM] -
        k[855]*y[IDX_DM] - k[1266]*y[IDX_CI] - k[1269]*y[IDX_CI] -
        k[1279]*y[IDX_NI] - k[1281]*y[IDX_NI] - k[1293]*y[IDX_OI] -
        k[1296]*y[IDX_OI] - k[1306]*y[IDX_OI] - k[1308]*y[IDX_OI] -
        k[1320]*y[IDX_C2I] - k[1323]*y[IDX_C2I] - k[1335]*y[IDX_CHI] -
        k[1338]*y[IDX_CHI] - k[1352]*y[IDX_CDI] - k[1355]*y[IDX_CDI] -
        k[1367]*y[IDX_CNI] - k[1370]*y[IDX_CNI] - k[1382]*y[IDX_COI] -
        k[1385]*y[IDX_COI] - k[1397]*y[IDX_COI] - k[1400]*y[IDX_COI] -
        k[1412]*y[IDX_N2I] - k[1415]*y[IDX_N2I] - k[1427]*y[IDX_NHI] -
        k[1430]*y[IDX_NHI] - k[1444]*y[IDX_NDI] - k[1447]*y[IDX_NDI] -
        k[1459]*y[IDX_NOI] - k[1462]*y[IDX_NOI] - k[1474]*y[IDX_O2I] -
        k[1477]*y[IDX_O2I] - k[1489]*y[IDX_OHI] - k[1492]*y[IDX_OHI] -
        k[1506]*y[IDX_ODI] - k[1509]*y[IDX_ODI] - k[1521]*y[IDX_NO2I] -
        k[1524]*y[IDX_NO2I] - k[2804]*y[IDX_eM] - k[2816]*y[IDX_eM] -
        k[2818]*y[IDX_eM] - k[2860]*y[IDX_HI] - k[2862]*y[IDX_HI] -
        k[2867]*y[IDX_DI] - k[2868]*y[IDX_DI] - k[2905]*y[IDX_pD2I] -
        k[3183]*y[IDX_H2OI] - k[3184]*y[IDX_H2OI] - k[3187]*y[IDX_H2OI] -
        k[3189]*y[IDX_H2OI] - k[3222]*y[IDX_HDOI] - k[3223]*y[IDX_HDOI] -
        k[3225]*y[IDX_HDOI] - k[3228]*y[IDX_HDOI] - k[3229]*y[IDX_HDOI] -
        k[3257]*y[IDX_D2OI] - k[3260]*y[IDX_D2OI] - k[3261]*y[IDX_D2OI];
    data[6009] = 0.0 + k[119]*y[IDX_pH2I] + k[123]*y[IDX_oH2I] +
        k[149]*y[IDX_HDI] + k[150]*y[IDX_HDI];
    data[6010] = 0.0 - k[73]*y[IDX_oD2HII] - k[74]*y[IDX_oD2HII] -
        k[75]*y[IDX_oD2HII] - k[76]*y[IDX_oD2HII] - k[77]*y[IDX_oD2HII] +
        k[77]*y[IDX_oD2HII] + k[115]*y[IDX_mD3II] + k[123]*y[IDX_oD3II] +
        k[688]*y[IDX_oD2II];
    data[6011] = 0.0 + k[700]*y[IDX_pD2I] + k[704]*y[IDX_oD2I];
    data[6012] = 0.0 + k[59]*y[IDX_HDI] + k[60]*y[IDX_HDI] +
        k[89]*y[IDX_pD2I] + k[93]*y[IDX_oD2I] + k[2858]*y[IDX_DI];
    data[6013] = 0.0 + k[44]*y[IDX_oD2I];
    data[6014] = 0.0 + k[80]*y[IDX_pH2DII] + k[89]*y[IDX_oH2DII] +
        k[125]*y[IDX_pD2HII] + k[126]*y[IDX_pD2HII] - k[133]*y[IDX_oD2HII] -
        k[134]*y[IDX_oD2HII] - k[135]*y[IDX_oD2HII] + k[135]*y[IDX_oD2HII] -
        k[136]*y[IDX_oD2HII] - k[137]*y[IDX_oD2HII] + k[700]*y[IDX_oH2II] +
        k[722]*y[IDX_HDII] - k[2905]*y[IDX_oD2HII];
    data[6015] = 0.0 + k[100]*y[IDX_HDI] + k[125]*y[IDX_pD2I] +
        k[126]*y[IDX_pD2I] + k[129]*y[IDX_oD2I] + k[130]*y[IDX_oD2I];
    data[6016] = 0.0 + k[2941]*y[IDX_HDI];
    data[6017] = 0.0 - k[69]*y[IDX_oD2HII] - k[70]*y[IDX_oD2HII] -
        k[71]*y[IDX_oD2HII] - k[72]*y[IDX_oD2HII] + k[72]*y[IDX_oD2HII] +
        k[113]*y[IDX_mD3II] + k[119]*y[IDX_oD3II] + k[686]*y[IDX_oD2II];
    data[6018] = 0.0 + k[702]*y[IDX_oD2I];
    data[6019] = 0.0 + k[50]*y[IDX_HDI] + k[51]*y[IDX_HDI] +
        k[80]*y[IDX_pD2I] + k[84]*y[IDX_oD2I] + k[2856]*y[IDX_DI];
    data[6020] = 0.0 + k[39]*y[IDX_oD2I] + k[40]*y[IDX_oD2I];
    data[6021] = 0.0 - k[1259]*y[IDX_oD3II];
    data[6022] = 0.0 - k[1313]*y[IDX_oD3II];
    data[6023] = 0.0 - k[1345]*y[IDX_oD3II];
    data[6024] = 0.0 - k[1328]*y[IDX_oD3II];
    data[6025] = 0.0 - k[1360]*y[IDX_oD3II];
    data[6026] = 0.0 - k[1375]*y[IDX_oD3II] - k[1390]*y[IDX_oD3II];
    data[6027] = 0.0 + k[2867]*y[IDX_oD2HII] + k[2869]*y[IDX_pD2HII];
    data[6028] = 0.0 - k[834]*y[IDX_oD3II];
    data[6029] = 0.0 - k[3266]*y[IDX_oD3II] - k[3267]*y[IDX_oD3II];
    data[6030] = 0.0 + k[805]*y[IDX_oD2II];
    data[6031] = 0.0 - k[197]*y[IDX_oD3II] - k[198]*y[IDX_oD3II] -
        k[199]*y[IDX_oD3II];
    data[6032] = 0.0 - k[2870]*y[IDX_oD3II];
    data[6033] = 0.0 - k[3193]*y[IDX_oD3II] - k[3195]*y[IDX_oD3II] -
        k[3199]*y[IDX_oD3II] - k[3202]*y[IDX_oD3II] - k[3203]*y[IDX_oD3II];
    data[6034] = 0.0 + k[101]*y[IDX_pD2HII] + k[102]*y[IDX_pD2HII] +
        k[110]*y[IDX_oD2HII] + k[111]*y[IDX_oD2HII] + k[146]*y[IDX_mD3II] -
        k[147]*y[IDX_oD3II] - k[148]*y[IDX_oD3II] - k[149]*y[IDX_oD3II] -
        k[150]*y[IDX_oD3II] - k[151]*y[IDX_oD3II] + k[731]*y[IDX_pD2II] +
        k[732]*y[IDX_oD2II] + k[2942]*y[IDX_pD3II] - k[2943]*y[IDX_oD3II];
    data[6035] = 0.0 + k[718]*y[IDX_pD2I] + k[720]*y[IDX_oD2I];
    data[6036] = 0.0 - k[3232]*y[IDX_oD3II] - k[3237]*y[IDX_oD3II] -
        k[3238]*y[IDX_oD3II];
    data[6037] = 0.0 + k[821]*y[IDX_oD2I];
    data[6038] = 0.0 - k[1273]*y[IDX_oD3II];
    data[6039] = 0.0 - k[1405]*y[IDX_oD3II];
    data[6040] = 0.0 - k[1437]*y[IDX_oD3II];
    data[6041] = 0.0 + k[1848]*y[IDX_oD2I];
    data[6042] = 0.0 - k[1420]*y[IDX_oD3II];
    data[6043] = 0.0 - k[1452]*y[IDX_oD3II];
    data[6044] = 0.0 - k[1514]*y[IDX_oD3II];
    data[6045] = 0.0 - k[1286]*y[IDX_oD3II] - k[1300]*y[IDX_oD3II];
    data[6046] = 0.0 - k[1467]*y[IDX_oD3II];
    data[6047] = 0.0 + k[2026]*y[IDX_oD2I];
    data[6048] = 0.0 - k[1499]*y[IDX_oD3II];
    data[6049] = 0.0 - k[1482]*y[IDX_oD3II];
    data[6050] = 0.0 - k[2801]*y[IDX_oD3II] - k[2810]*y[IDX_oD3II] -
        k[2811]*y[IDX_oD3II];
    data[6051] = 0.0 + k[146]*y[IDX_HDI] + k[153]*y[IDX_pD2I] +
        k[154]*y[IDX_pD2I] + k[156]*y[IDX_oD2I] + k[157]*y[IDX_oD2I];
    data[6052] = 0.0 + k[86]*y[IDX_pH2DII] + k[95]*y[IDX_oH2DII] +
        k[132]*y[IDX_pD2HII] + k[142]*y[IDX_oD2HII] + k[156]*y[IDX_mD3II] +
        k[157]*y[IDX_mD3II] - k[161]*y[IDX_oD3II] - k[162]*y[IDX_oD3II] -
        k[163]*y[IDX_oD3II] + k[163]*y[IDX_oD3II] + k[714]*y[IDX_pD2II] +
        k[716]*y[IDX_oD2II] + k[720]*y[IDX_HDII] + k[821]*y[IDX_HeDII] +
        k[1848]*y[IDX_NDII] + k[2026]*y[IDX_O2DII] + k[2911]*y[IDX_pD3II] +
        k[2946]*y[IDX_pD3II] - k[2948]*y[IDX_oD3II] - k[2949]*y[IDX_oD3II];
    data[6053] = 0.0 + k[712]*y[IDX_pD2I] + k[716]*y[IDX_oD2I] +
        k[732]*y[IDX_HDI] + k[805]*y[IDX_DCOI];
    data[6054] = 0.0 + k[110]*y[IDX_HDI] + k[111]*y[IDX_HDI] +
        k[137]*y[IDX_pD2I] + k[142]*y[IDX_oD2I] + k[2867]*y[IDX_DI];
    data[6055] = 0.0 - k[116]*y[IDX_pH2I] - k[117]*y[IDX_pH2I] -
        k[118]*y[IDX_pH2I] - k[119]*y[IDX_pH2I] - k[120]*y[IDX_oH2I] -
        k[121]*y[IDX_oH2I] - k[122]*y[IDX_oH2I] - k[123]*y[IDX_oH2I] -
        k[147]*y[IDX_HDI] - k[148]*y[IDX_HDI] - k[149]*y[IDX_HDI] -
        k[150]*y[IDX_HDI] - k[151]*y[IDX_HDI] - k[158]*y[IDX_pD2I] -
        k[159]*y[IDX_pD2I] - k[160]*y[IDX_pD2I] + k[160]*y[IDX_pD2I] -
        k[161]*y[IDX_oD2I] - k[162]*y[IDX_oD2I] - k[163]*y[IDX_oD2I] +
        k[163]*y[IDX_oD2I] - k[197]*y[IDX_GRAINM] - k[198]*y[IDX_GRAINM] -
        k[199]*y[IDX_GRAINM] - k[308] - k[309] - k[328] - k[329] -
        k[834]*y[IDX_DM] - k[1259]*y[IDX_CI] - k[1273]*y[IDX_NI] -
        k[1286]*y[IDX_OI] - k[1300]*y[IDX_OI] - k[1313]*y[IDX_C2I] -
        k[1328]*y[IDX_CHI] - k[1345]*y[IDX_CDI] - k[1360]*y[IDX_CNI] -
        k[1375]*y[IDX_COI] - k[1390]*y[IDX_COI] - k[1405]*y[IDX_N2I] -
        k[1420]*y[IDX_NHI] - k[1437]*y[IDX_NDI] - k[1452]*y[IDX_NOI] -
        k[1467]*y[IDX_O2I] - k[1482]*y[IDX_OHI] - k[1499]*y[IDX_ODI] -
        k[1514]*y[IDX_NO2I] - k[2801]*y[IDX_eM] - k[2810]*y[IDX_eM] -
        k[2811]*y[IDX_eM] - k[2870]*y[IDX_HI] - k[2909]*y[IDX_pD2I] -
        k[2943]*y[IDX_HDI] - k[2947]*y[IDX_pD2I] - k[2948]*y[IDX_oD2I] -
        k[2949]*y[IDX_oD2I] - k[3193]*y[IDX_H2OI] - k[3195]*y[IDX_H2OI] -
        k[3199]*y[IDX_H2OI] - k[3202]*y[IDX_H2OI] - k[3203]*y[IDX_H2OI] -
        k[3232]*y[IDX_HDOI] - k[3237]*y[IDX_HDOI] - k[3238]*y[IDX_HDOI] -
        k[3266]*y[IDX_D2OI] - k[3267]*y[IDX_D2OI];
    data[6056] = 0.0 - k[120]*y[IDX_oD3II] - k[121]*y[IDX_oD3II] -
        k[122]*y[IDX_oD3II] - k[123]*y[IDX_oD3II];
    data[6057] = 0.0 + k[90]*y[IDX_pD2I] + k[95]*y[IDX_oD2I];
    data[6058] = 0.0 + k[81]*y[IDX_pH2DII] + k[90]*y[IDX_oH2DII] +
        k[127]*y[IDX_pD2HII] + k[137]*y[IDX_oD2HII] + k[153]*y[IDX_mD3II] +
        k[154]*y[IDX_mD3II] - k[158]*y[IDX_oD3II] - k[159]*y[IDX_oD3II] -
        k[160]*y[IDX_oD3II] + k[160]*y[IDX_oD3II] + k[710]*y[IDX_pD2II] +
        k[712]*y[IDX_oD2II] + k[718]*y[IDX_HDII] - k[2909]*y[IDX_oD3II] +
        k[2944]*y[IDX_pD3II] + k[2945]*y[IDX_pD3II] - k[2947]*y[IDX_oD3II];
    data[6059] = 0.0 + k[710]*y[IDX_pD2I] + k[714]*y[IDX_oD2I] +
        k[731]*y[IDX_HDI];
    data[6060] = 0.0 + k[101]*y[IDX_HDI] + k[102]*y[IDX_HDI] +
        k[127]*y[IDX_pD2I] + k[132]*y[IDX_oD2I] + k[2869]*y[IDX_DI];
    data[6061] = 0.0 + k[2911]*y[IDX_oD2I] + k[2942]*y[IDX_HDI] +
        k[2944]*y[IDX_pD2I] + k[2945]*y[IDX_pD2I] + k[2946]*y[IDX_oD2I];
    data[6062] = 0.0 - k[116]*y[IDX_oD3II] - k[117]*y[IDX_oD3II] -
        k[118]*y[IDX_oD3II] - k[119]*y[IDX_oD3II];
    data[6063] = 0.0 + k[81]*y[IDX_pD2I] + k[86]*y[IDX_oD2I];
    data[6064] = 0.0 - k[1131]*y[IDX_oH2I] + k[1256]*y[IDX_oH3II] +
        k[1258]*y[IDX_pH3II] + k[1261]*y[IDX_oH2DII] + k[1263]*y[IDX_pH2DII] -
        k[2658]*y[IDX_oH2I] + k[3011]*y[IDX_H3OII] + k[3284]*y[IDX_H2DOII];
    data[6065] = 0.0 - k[2642]*y[IDX_oH2I];
    data[6066] = 0.0 - k[2678]*y[IDX_oH2I];
    data[6067] = 0.0 + k[1310]*y[IDX_oH3II] + k[1312]*y[IDX_pH3II] +
        k[1315]*y[IDX_oH2DII] + k[1317]*y[IDX_pH2DII] + k[2341]*y[IDX_oH2II];
    data[6068] = 0.0 - k[1695]*y[IDX_oH2I];
    data[6069] = 0.0 + k[2424]*y[IDX_oH2II];
    data[6070] = 0.0 + k[2419]*y[IDX_oH2II];
    data[6071] = 0.0 - k[1145]*y[IDX_oH2I] - k[1147]*y[IDX_oH2I] +
        k[1340]*y[IDX_oH3II] + k[1342]*y[IDX_pH3II] + k[1347]*y[IDX_oH2DII] +
        k[1349]*y[IDX_pH2DII] + k[2351]*y[IDX_oH2II];
    data[6072] = 0.0 - k[1734]*y[IDX_oH2I] - k[1736]*y[IDX_oH2I];
    data[6073] = 0.0 + k[2434]*y[IDX_oH2II];
    data[6074] = 0.0 - k[1137]*y[IDX_oH2I] + k[1325]*y[IDX_oH3II] +
        k[1327]*y[IDX_pH3II] + k[1330]*y[IDX_oH2DII] + k[1332]*y[IDX_pH2DII] +
        k[2346]*y[IDX_oH2II];
    data[6075] = 0.0 - k[1732]*y[IDX_oH2I];
    data[6076] = 0.0 + k[2429]*y[IDX_oH2II];
    data[6077] = 0.0 + k[2439]*y[IDX_oH2II];
    data[6078] = 0.0 + k[1357]*y[IDX_oH3II] + k[1359]*y[IDX_pH3II] +
        k[1362]*y[IDX_oH2DII] + k[1364]*y[IDX_pH2DII] + k[2356]*y[IDX_oH2II];
    data[6079] = 0.0 - k[469]*y[IDX_oH2I] - k[475]*y[IDX_oH2I];
    data[6080] = 0.0 + k[1372]*y[IDX_oH3II] + k[1374]*y[IDX_pH3II] +
        k[1377]*y[IDX_oH2DII] + k[1379]*y[IDX_pH2DII] + k[1387]*y[IDX_oH3II] +
        k[1389]*y[IDX_pH3II] + k[1392]*y[IDX_oH2DII] + k[1394]*y[IDX_pH2DII] +
        k[2361]*y[IDX_oH2II];
    data[6081] = 0.0 - k[457]*y[IDX_oH2I] - k[463]*y[IDX_oH2I];
    data[6082] = 0.0 + k[2539]*y[IDX_oH2II];
    data[6083] = 0.0 + k[2099]*y[IDX_oH2II] + k[3384]*y[IDX_H3OII] +
        k[3387]*y[IDX_H2DOII];
    data[6084] = 0.0 - k[2883]*y[IDX_oH2I];
    data[6085] = 0.0 + k[831]*y[IDX_oH3II] + k[833]*y[IDX_pH3II] +
        k[841]*y[IDX_oH2DII] + k[844]*y[IDX_pH2DII] + k[2153]*y[IDX_oH2II] +
        k[3055]*y[IDX_H3OII] + k[3058]*y[IDX_H3OII] + k[3069]*y[IDX_H2DOII] +
        k[3090]*y[IDX_H3OII] + k[3096]*y[IDX_H2DOII];
    data[6086] = 0.0 + k[2449]*y[IDX_oH2II] + k[3240]*y[IDX_pH3II] +
        k[3241]*y[IDX_oH3II] + k[3249]*y[IDX_oH2DII];
    data[6087] = 0.0 - k[3128]*y[IDX_oH2I] - k[3130]*y[IDX_oH2I];
    data[6088] = 0.0 - k[3406]*y[IDX_oH2I] - k[3408]*y[IDX_oH2I] -
        k[3410]*y[IDX_oH2I];
    data[6089] = 0.0 + k[2109]*y[IDX_oH2II];
    data[6090] = 0.0 + k[2464]*y[IDX_oH2II];
    data[6091] = 0.0 - k[1596]*y[IDX_oH2I] - k[1597]*y[IDX_oH2I] +
        k[1597]*y[IDX_oH2I] - k[1599]*y[IDX_oH2I];
    data[6092] = 0.0 + k[179]*y[IDX_pH3II] + k[181]*y[IDX_oH3II] +
        k[187]*y[IDX_oH2DII];
    data[6093] = 0.0 + k[165]*y[IDX_HI] + k[165]*y[IDX_HI] +
        k[2094]*y[IDX_oH2II] + k[3022]*y[IDX_H3OII] + k[3373]*y[IDX_H2DOII] +
        k[3376]*y[IDX_HD2OII];
    data[6094] = 0.0 - k[2874]*y[IDX_oH2I] + k[2875]*y[IDX_pH2I] +
        k[2885]*y[IDX_HDI];
    data[6095] = 0.0 + k[827]*y[IDX_oH3II] + k[829]*y[IDX_pH3II] +
        k[837]*y[IDX_oH2DII] + k[839]*y[IDX_pH2DII] + k[848]*y[IDX_oD2HII] +
        k[851]*y[IDX_pD2HII] + k[2151]*y[IDX_oH2II] + k[2991]*y[IDX_H3OII] +
        k[2994]*y[IDX_H3OII] + k[3060]*y[IDX_H2DOII] + k[3063]*y[IDX_H2DOII] +
        k[3074]*y[IDX_HD2OII] + k[3093]*y[IDX_H2DOII] + k[3101]*y[IDX_HD2OII];
    data[6096] = 0.0 + k[3060]*y[IDX_HM] + k[3063]*y[IDX_HM] +
        k[3069]*y[IDX_DM] + k[3093]*y[IDX_HM] + k[3096]*y[IDX_DM] +
        k[3284]*y[IDX_CI] + k[3373]*y[IDX_HI] + k[3387]*y[IDX_DI] -
        k[3397]*y[IDX_oH2I] - k[3399]*y[IDX_oH2I] + k[3451]*y[IDX_eM] +
        k[3460]*y[IDX_eM];
    data[6097] = 0.0 + k[2444]*y[IDX_oH2II] + k[3003]*y[IDX_oH3II] +
        k[3005]*y[IDX_pH3II] + k[3178]*y[IDX_oH2DII] + k[3180]*y[IDX_pH2DII] +
        k[3184]*y[IDX_oD2HII] + k[3186]*y[IDX_pD2HII] + k[3191]*y[IDX_pD3II] +
        k[3192]*y[IDX_mD3II] + k[3195]*y[IDX_oD3II];
    data[6098] = 0.0 - k[2998]*y[IDX_oH2I];
    data[6099] = 0.0 + k[2991]*y[IDX_HM] + k[2994]*y[IDX_HM] +
        k[3011]*y[IDX_CI] + k[3022]*y[IDX_HI] - k[3023]*y[IDX_oH2I] +
        k[3028]*y[IDX_eM] + k[3029]*y[IDX_eM] + k[3055]*y[IDX_DM] +
        k[3058]*y[IDX_DM] + k[3090]*y[IDX_DM] + k[3384]*y[IDX_DI];
    data[6100] = 0.0 + k[2104]*y[IDX_oH2II];
    data[6101] = 0.0 + k[2459]*y[IDX_oH2II];
    data[6102] = 0.0 + k[12]*y[IDX_pH3II] + k[14]*y[IDX_pH3II] +
        k[16]*y[IDX_oH3II] + k[18]*y[IDX_oH3II] + k[49]*y[IDX_pH2DII] +
        k[51]*y[IDX_pH2DII] + k[58]*y[IDX_oH2DII] + k[60]*y[IDX_oH2DII] +
        k[102]*y[IDX_pD2HII] + k[109]*y[IDX_oD2HII] + k[111]*y[IDX_oD2HII] +
        k[1609]*y[IDX_HOCII] + k[2885]*y[IDX_HII] + k[2901]*y[IDX_pD2HII];
    data[6103] = 0.0 - k[695]*y[IDX_oH2I] - k[696]*y[IDX_oH2I] -
        k[698]*y[IDX_oH2I] - k[2887]*y[IDX_oH2I];
    data[6104] = 0.0 + k[3074]*y[IDX_HM] + k[3101]*y[IDX_HM] +
        k[3376]*y[IDX_HI] - k[3401]*y[IDX_oH2I] - k[3403]*y[IDX_oH2I] -
        k[3405]*y[IDX_oH2I];
    data[6105] = 0.0 + k[2454]*y[IDX_oH2II] + k[3205]*y[IDX_oH3II] +
        k[3207]*y[IDX_pH3II] + k[3211]*y[IDX_pH2DII] + k[3213]*y[IDX_oH2DII] +
        k[3221]*y[IDX_pD2HII] + k[3223]*y[IDX_oD2HII];
    data[6106] = 0.0 - k[3124]*y[IDX_oH2I] - k[3126]*y[IDX_oH2I];
    data[6107] = 0.0 - k[1593]*y[IDX_oH2I] - k[1594]*y[IDX_oH2I] +
        k[1594]*y[IDX_oH2I] + k[1609]*y[IDX_HDI];
    data[6108] = 0.0 - k[869]*y[IDX_oH2I] - k[2472]*y[IDX_oH2I];
    data[6109] = 0.0 - k[815]*y[IDX_oH2I] - k[816]*y[IDX_oH2I];
    data[6110] = 0.0 - k[813]*y[IDX_oH2I] - k[2923]*y[IDX_oH2I];
    data[6111] = 0.0 - k[1175]*y[IDX_oH2I];
    data[6112] = 0.0 - k[1637]*y[IDX_oH2I];
    data[6113] = 0.0 + k[1402]*y[IDX_oH3II] + k[1404]*y[IDX_pH3II] +
        k[1407]*y[IDX_oH2DII] + k[1409]*y[IDX_pH2DII];
    data[6114] = 0.0 - k[1807]*y[IDX_oH2I];
    data[6115] = 0.0 - k[1183]*y[IDX_oH2I] - k[1185]*y[IDX_oH2I] +
        k[1432]*y[IDX_oH3II] + k[1434]*y[IDX_pH3II] + k[1439]*y[IDX_oH2DII] +
        k[1441]*y[IDX_pH2DII] + k[2394]*y[IDX_oH2II];
    data[6116] = 0.0 - k[1842]*y[IDX_oH2I] - k[1843]*y[IDX_oH2I] -
        k[1857]*y[IDX_oH2I] - k[1859]*y[IDX_oH2I];
    data[6117] = 0.0 + k[2119]*y[IDX_oH2II];
    data[6118] = 0.0 - k[1181]*y[IDX_oH2I] + k[1417]*y[IDX_oH3II] +
        k[1419]*y[IDX_pH3II] + k[1422]*y[IDX_oH2DII] + k[1424]*y[IDX_pH2DII] +
        k[2389]*y[IDX_oH2II];
    data[6119] = 0.0 - k[1840]*y[IDX_oH2I] - k[1855]*y[IDX_oH2I] -
        k[2925]*y[IDX_oH2I];
    data[6120] = 0.0 + k[2114]*y[IDX_oH2II];
    data[6121] = 0.0 + k[2124]*y[IDX_oH2II];
    data[6122] = 0.0 + k[1449]*y[IDX_oH3II] + k[1451]*y[IDX_pH3II] +
        k[1454]*y[IDX_oH2DII] + k[1456]*y[IDX_pH2DII] + k[2399]*y[IDX_oH2II];
    data[6123] = 0.0 + k[1511]*y[IDX_oH3II] + k[1513]*y[IDX_pH3II] +
        k[1516]*y[IDX_oH2DII] + k[1518]*y[IDX_pH2DII];
    data[6124] = 0.0 - k[1998]*y[IDX_oH2I];
    data[6125] = 0.0 - k[1153]*y[IDX_oH2I] + k[1283]*y[IDX_oH3II] +
        k[1285]*y[IDX_pH3II] + k[1288]*y[IDX_oH2DII] + k[1290]*y[IDX_pH2DII];
    data[6126] = 0.0 - k[1657]*y[IDX_oH2I];
    data[6127] = 0.0 - k[498]*y[IDX_oH2I] - k[2724]*y[IDX_oH2I];
    data[6128] = 0.0 + k[1464]*y[IDX_oH3II] + k[1466]*y[IDX_pH3II] +
        k[1469]*y[IDX_oH2DII] + k[1471]*y[IDX_pH2DII] + k[2404]*y[IDX_oH2II];
    data[6129] = 0.0 - k[2020]*y[IDX_oH2I] - k[2021]*y[IDX_oH2I];
    data[6130] = 0.0 - k[2018]*y[IDX_oH2I] - k[2927]*y[IDX_oH2I];
    data[6131] = 0.0 - k[1161]*y[IDX_oH2I] - k[1163]*y[IDX_oH2I] +
        k[1494]*y[IDX_oH3II] + k[1496]*y[IDX_pH3II] + k[1501]*y[IDX_oH2DII] +
        k[1503]*y[IDX_pH2DII] + k[2414]*y[IDX_oH2II];
    data[6132] = 0.0 - k[1940]*y[IDX_oH2I] - k[1942]*y[IDX_oH2I];
    data[6133] = 0.0 - k[1159]*y[IDX_oH2I] + k[1479]*y[IDX_oH3II] +
        k[1481]*y[IDX_pH3II] + k[1484]*y[IDX_oH2DII] + k[1486]*y[IDX_pH2DII] +
        k[2409]*y[IDX_oH2II];
    data[6134] = 0.0 - k[1938]*y[IDX_oH2I];
    data[6135] = 0.0 + k[2751]*y[IDX_oH2II] + k[2806]*y[IDX_oH3II] +
        k[2807]*y[IDX_pH3II] + k[2814]*y[IDX_oH2DII] + k[3028]*y[IDX_H3OII] +
        k[3029]*y[IDX_H3OII] + k[3451]*y[IDX_H2DOII] + k[3460]*y[IDX_H2DOII];
    data[6136] = 0.0 - k[114]*y[IDX_oH2I] - k[115]*y[IDX_oH2I] +
        k[3192]*y[IDX_H2OI];
    data[6137] = 0.0 + k[40]*y[IDX_pH3II] + k[44]*y[IDX_oH3II] +
        k[94]*y[IDX_oH2DII] + k[95]*y[IDX_oH2DII];
    data[6138] = 0.0 - k[688]*y[IDX_oH2I] - k[692]*y[IDX_oH2I];
    data[6139] = 0.0 + k[72]*y[IDX_pH2I] - k[73]*y[IDX_oH2I] -
        k[74]*y[IDX_oH2I] - k[75]*y[IDX_oH2I] - k[76]*y[IDX_oH2I] -
        k[77]*y[IDX_oH2I] + k[109]*y[IDX_HDI] + k[111]*y[IDX_HDI] +
        k[848]*y[IDX_HM] + k[3184]*y[IDX_H2OI] + k[3223]*y[IDX_HDOI];
    data[6140] = 0.0 - k[120]*y[IDX_oH2I] - k[121]*y[IDX_oH2I] -
        k[122]*y[IDX_oH2I] - k[123]*y[IDX_oH2I] + k[3195]*y[IDX_H2OI];
    data[6141] = 0.0 - k[2]*y[IDX_pH3II] - k[3]*y[IDX_pH3II] -
        k[4]*y[IDX_pH3II] + k[4]*y[IDX_pH3II] - k[7]*y[IDX_oH3II] -
        k[8]*y[IDX_oH3II] + k[8]*y[IDX_oH3II] - k[9]*y[IDX_oH3II] -
        k[21]*y[IDX_pH2DII] - k[22]*y[IDX_pH2DII] - k[23]*y[IDX_pH2DII] -
        k[24]*y[IDX_pH2DII] + k[24]*y[IDX_pH2DII] - k[28]*y[IDX_oH2DII] -
        k[29]*y[IDX_oH2DII] - k[30]*y[IDX_oH2DII] - k[31]*y[IDX_oH2DII] +
        k[31]*y[IDX_oH2DII] - k[32]*y[IDX_oH2DII] - k[65]*y[IDX_pD2HII] -
        k[66]*y[IDX_pD2HII] - k[67]*y[IDX_pD2HII] - k[68]*y[IDX_pD2HII] -
        k[73]*y[IDX_oD2HII] - k[74]*y[IDX_oD2HII] - k[75]*y[IDX_oD2HII] -
        k[76]*y[IDX_oD2HII] - k[77]*y[IDX_oD2HII] - k[114]*y[IDX_mD3II] -
        k[115]*y[IDX_mD3II] - k[120]*y[IDX_oD3II] - k[121]*y[IDX_oD3II] -
        k[122]*y[IDX_oD3II] - k[123]*y[IDX_oD3II] - k[209] - k[214] - k[220] -
        k[226] - k[361] - k[457]*y[IDX_COII] - k[463]*y[IDX_COII] -
        k[469]*y[IDX_CNII] - k[475]*y[IDX_CNII] - k[498]*y[IDX_OM] -
        k[683]*y[IDX_pH2II] - k[684]*y[IDX_oH2II] - k[687]*y[IDX_pD2II] -
        k[688]*y[IDX_oD2II] - k[691]*y[IDX_pD2II] - k[692]*y[IDX_oD2II] -
        k[695]*y[IDX_HDII] - k[696]*y[IDX_HDII] - k[698]*y[IDX_HDII] -
        k[813]*y[IDX_HeHII] - k[815]*y[IDX_HeDII] - k[816]*y[IDX_HeDII] -
        k[869]*y[IDX_HeII] - k[1131]*y[IDX_CI] - k[1137]*y[IDX_CHI] -
        k[1145]*y[IDX_CDI] - k[1147]*y[IDX_CDI] - k[1153]*y[IDX_OI] -
        k[1159]*y[IDX_OHI] - k[1161]*y[IDX_ODI] - k[1163]*y[IDX_ODI] -
        k[1175]*y[IDX_NI] - k[1181]*y[IDX_NHI] - k[1183]*y[IDX_NDI] -
        k[1185]*y[IDX_NDI] - k[1593]*y[IDX_HOCII] - k[1594]*y[IDX_HOCII] +
        k[1594]*y[IDX_HOCII] - k[1596]*y[IDX_DOCII] - k[1597]*y[IDX_DOCII] +
        k[1597]*y[IDX_DOCII] - k[1599]*y[IDX_DOCII] - k[1637]*y[IDX_NII] -
        k[1657]*y[IDX_OII] - k[1695]*y[IDX_C2II] - k[1732]*y[IDX_CHII] -
        k[1734]*y[IDX_CDII] - k[1736]*y[IDX_CDII] - k[1807]*y[IDX_N2II] -
        k[1840]*y[IDX_NHII] - k[1842]*y[IDX_NDII] - k[1843]*y[IDX_NDII] -
        k[1855]*y[IDX_NHII] - k[1857]*y[IDX_NDII] - k[1859]*y[IDX_NDII] -
        k[1938]*y[IDX_OHII] - k[1940]*y[IDX_ODII] - k[1942]*y[IDX_ODII] -
        k[1998]*y[IDX_NO2II] - k[2018]*y[IDX_O2HII] - k[2020]*y[IDX_O2DII] -
        k[2021]*y[IDX_O2DII] - k[2472]*y[IDX_HeII] - k[2642]*y[IDX_CII] -
        k[2658]*y[IDX_CI] - k[2678]*y[IDX_CM] - k[2724]*y[IDX_OM] -
        k[2874]*y[IDX_HII] - k[2876]*y[IDX_oH2II] - k[2877]*y[IDX_pH2II] -
        k[2883]*y[IDX_DII] - k[2887]*y[IDX_HDII] - k[2902]*y[IDX_pD3II] -
        k[2923]*y[IDX_HeHII] - k[2925]*y[IDX_NHII] - k[2927]*y[IDX_O2HII] -
        k[2938]*y[IDX_pD3II] - k[2998]*y[IDX_H2OII] - k[3023]*y[IDX_H3OII] -
        k[3124]*y[IDX_HDOII] - k[3126]*y[IDX_HDOII] - k[3128]*y[IDX_D2OII] -
        k[3130]*y[IDX_D2OII] - k[3397]*y[IDX_H2DOII] - k[3399]*y[IDX_H2DOII] -
        k[3401]*y[IDX_HD2OII] - k[3403]*y[IDX_HD2OII] - k[3405]*y[IDX_HD2OII] -
        k[3406]*y[IDX_D3OII] - k[3408]*y[IDX_D3OII] - k[3410]*y[IDX_D3OII];
    data[6142] = 0.0 - k[684]*y[IDX_oH2I] + k[2094]*y[IDX_HI] +
        k[2099]*y[IDX_DI] + k[2104]*y[IDX_HCNI] + k[2109]*y[IDX_DCNI] +
        k[2114]*y[IDX_NH2I] + k[2119]*y[IDX_ND2I] + k[2124]*y[IDX_NHDI] +
        k[2151]*y[IDX_HM] + k[2153]*y[IDX_DM] + k[2341]*y[IDX_C2I] +
        k[2346]*y[IDX_CHI] + k[2351]*y[IDX_CDI] + k[2356]*y[IDX_CNI] +
        k[2361]*y[IDX_COI] + k[2389]*y[IDX_NHI] + k[2394]*y[IDX_NDI] +
        k[2399]*y[IDX_NOI] + k[2404]*y[IDX_O2I] + k[2409]*y[IDX_OHI] +
        k[2414]*y[IDX_ODI] + k[2419]*y[IDX_C2HI] + k[2424]*y[IDX_C2DI] +
        k[2429]*y[IDX_CH2I] + k[2434]*y[IDX_CD2I] + k[2439]*y[IDX_CHDI] +
        k[2444]*y[IDX_H2OI] + k[2449]*y[IDX_D2OI] + k[2454]*y[IDX_HDOI] +
        k[2459]*y[IDX_HCOI] + k[2464]*y[IDX_DCOI] + k[2539]*y[IDX_CO2I] +
        k[2751]*y[IDX_eM] - k[2876]*y[IDX_oH2I];
    data[6143] = 0.0 + k[26]*y[IDX_pH2I] + k[27]*y[IDX_pH2I] -
        k[28]*y[IDX_oH2I] - k[29]*y[IDX_oH2I] - k[30]*y[IDX_oH2I] -
        k[31]*y[IDX_oH2I] + k[31]*y[IDX_oH2I] - k[32]*y[IDX_oH2I] +
        k[58]*y[IDX_HDI] + k[60]*y[IDX_HDI] + k[90]*y[IDX_pD2I] +
        k[94]*y[IDX_oD2I] + k[95]*y[IDX_oD2I] + k[187]*y[IDX_GRAINM] + k[313] +
        k[837]*y[IDX_HM] + k[841]*y[IDX_DM] + k[1261]*y[IDX_CI] +
        k[1288]*y[IDX_OI] + k[1315]*y[IDX_C2I] + k[1330]*y[IDX_CHI] +
        k[1347]*y[IDX_CDI] + k[1362]*y[IDX_CNI] + k[1377]*y[IDX_COI] +
        k[1392]*y[IDX_COI] + k[1407]*y[IDX_N2I] + k[1422]*y[IDX_NHI] +
        k[1439]*y[IDX_NDI] + k[1454]*y[IDX_NOI] + k[1469]*y[IDX_O2I] +
        k[1484]*y[IDX_OHI] + k[1501]*y[IDX_ODI] + k[1516]*y[IDX_NO2I] +
        k[2814]*y[IDX_eM] + k[2903]*y[IDX_pD2I] + k[3178]*y[IDX_H2OI] +
        k[3213]*y[IDX_HDOI] + k[3249]*y[IDX_D2OI];
    data[6144] = 0.0 + k[5]*y[IDX_pH2I] + k[6]*y[IDX_pH2I] -
        k[7]*y[IDX_oH2I] - k[8]*y[IDX_oH2I] + k[8]*y[IDX_oH2I] -
        k[9]*y[IDX_oH2I] + k[16]*y[IDX_HDI] + k[18]*y[IDX_HDI] +
        k[42]*y[IDX_pD2I] + k[44]*y[IDX_oD2I] + k[181]*y[IDX_GRAINM] + k[305] +
        k[827]*y[IDX_HM] + k[831]*y[IDX_DM] + k[1256]*y[IDX_CI] +
        k[1283]*y[IDX_OI] + k[1310]*y[IDX_C2I] + k[1325]*y[IDX_CHI] +
        k[1340]*y[IDX_CDI] + k[1357]*y[IDX_CNI] + k[1372]*y[IDX_COI] +
        k[1387]*y[IDX_COI] + k[1402]*y[IDX_N2I] + k[1417]*y[IDX_NHI] +
        k[1432]*y[IDX_NDI] + k[1449]*y[IDX_NOI] + k[1464]*y[IDX_O2I] +
        k[1479]*y[IDX_OHI] + k[1494]*y[IDX_ODI] + k[1511]*y[IDX_NO2I] +
        k[2806]*y[IDX_eM] + k[3003]*y[IDX_H2OI] + k[3205]*y[IDX_HDOI] +
        k[3241]*y[IDX_D2OI];
    data[6145] = 0.0 + k[36]*y[IDX_pH3II] + k[42]*y[IDX_oH3II] +
        k[90]*y[IDX_oH2DII] + k[2903]*y[IDX_oH2DII];
    data[6146] = 0.0 - k[687]*y[IDX_oH2I] - k[691]*y[IDX_oH2I];
    data[6147] = 0.0 + k[64]*y[IDX_pH2I] - k[65]*y[IDX_oH2I] -
        k[66]*y[IDX_oH2I] - k[67]*y[IDX_oH2I] - k[68]*y[IDX_oH2I] +
        k[102]*y[IDX_HDI] + k[851]*y[IDX_HM] + k[2901]*y[IDX_HDI] +
        k[3186]*y[IDX_H2OI] + k[3221]*y[IDX_HDOI];
    data[6148] = 0.0 - k[2902]*y[IDX_oH2I] - k[2938]*y[IDX_oH2I] +
        k[3191]*y[IDX_H2OI];
    data[6149] = 0.0 + k[0]*y[IDX_pH3II] + k[1]*y[IDX_pH3II] +
        k[5]*y[IDX_oH3II] + k[6]*y[IDX_oH3II] + k[20]*y[IDX_pH2DII] +
        k[26]*y[IDX_oH2DII] + k[27]*y[IDX_oH2DII] + k[64]*y[IDX_pD2HII] +
        k[72]*y[IDX_oD2HII] + k[2875]*y[IDX_HII];
    data[6150] = 0.0 - k[683]*y[IDX_oH2I] - k[2877]*y[IDX_oH2I];
    data[6151] = 0.0 + k[20]*y[IDX_pH2I] - k[21]*y[IDX_oH2I] -
        k[22]*y[IDX_oH2I] - k[23]*y[IDX_oH2I] - k[24]*y[IDX_oH2I] +
        k[24]*y[IDX_oH2I] + k[49]*y[IDX_HDI] + k[51]*y[IDX_HDI] + k[315] +
        k[839]*y[IDX_HM] + k[844]*y[IDX_DM] + k[1263]*y[IDX_CI] +
        k[1290]*y[IDX_OI] + k[1317]*y[IDX_C2I] + k[1332]*y[IDX_CHI] +
        k[1349]*y[IDX_CDI] + k[1364]*y[IDX_CNI] + k[1379]*y[IDX_COI] +
        k[1394]*y[IDX_COI] + k[1409]*y[IDX_N2I] + k[1424]*y[IDX_NHI] +
        k[1441]*y[IDX_NDI] + k[1456]*y[IDX_NOI] + k[1471]*y[IDX_O2I] +
        k[1486]*y[IDX_OHI] + k[1503]*y[IDX_ODI] + k[1518]*y[IDX_NO2I] +
        k[3180]*y[IDX_H2OI] + k[3211]*y[IDX_HDOI];
    data[6152] = 0.0 + k[0]*y[IDX_pH2I] + k[1]*y[IDX_pH2I] -
        k[2]*y[IDX_oH2I] - k[3]*y[IDX_oH2I] - k[4]*y[IDX_oH2I] +
        k[4]*y[IDX_oH2I] + k[12]*y[IDX_HDI] + k[14]*y[IDX_HDI] +
        k[36]*y[IDX_pD2I] + k[40]*y[IDX_oD2I] + k[179]*y[IDX_GRAINM] + k[307] +
        k[829]*y[IDX_HM] + k[833]*y[IDX_DM] + k[1258]*y[IDX_CI] +
        k[1285]*y[IDX_OI] + k[1312]*y[IDX_C2I] + k[1327]*y[IDX_CHI] +
        k[1342]*y[IDX_CDI] + k[1359]*y[IDX_CNI] + k[1374]*y[IDX_COI] +
        k[1389]*y[IDX_COI] + k[1404]*y[IDX_N2I] + k[1419]*y[IDX_NHI] +
        k[1434]*y[IDX_NDI] + k[1451]*y[IDX_NOI] + k[1466]*y[IDX_O2I] +
        k[1481]*y[IDX_OHI] + k[1496]*y[IDX_ODI] + k[1513]*y[IDX_NO2I] +
        k[2807]*y[IDX_eM] + k[3005]*y[IDX_H2OI] + k[3207]*y[IDX_HDOI] +
        k[3240]*y[IDX_D2OI];
    data[6153] = 0.0 - k[631]*y[IDX_oH2II];
    data[6154] = 0.0 - k[649]*y[IDX_oH2II] - k[2341]*y[IDX_oH2II];
    data[6155] = 0.0 - k[2424]*y[IDX_oH2II];
    data[6156] = 0.0 - k[2419]*y[IDX_oH2II];
    data[6157] = 0.0 - k[663]*y[IDX_oH2II] - k[665]*y[IDX_oH2II] -
        k[2351]*y[IDX_oH2II];
    data[6158] = 0.0 - k[2434]*y[IDX_oH2II];
    data[6159] = 0.0 - k[655]*y[IDX_oH2II] - k[2346]*y[IDX_oH2II];
    data[6160] = 0.0 - k[2429]*y[IDX_oH2II];
    data[6161] = 0.0 - k[2439]*y[IDX_oH2II];
    data[6162] = 0.0 - k[671]*y[IDX_oH2II] - k[2356]*y[IDX_oH2II];
    data[6163] = 0.0 - k[677]*y[IDX_oH2II] - k[2361]*y[IDX_oH2II];
    data[6164] = 0.0 - k[791]*y[IDX_oH2II] - k[2539]*y[IDX_oH2II];
    data[6165] = 0.0 - k[2099]*y[IDX_oH2II];
    data[6166] = 0.0 - k[2153]*y[IDX_oH2II];
    data[6167] = 0.0 - k[2449]*y[IDX_oH2II] - k[3048]*y[IDX_oH2II] -
        k[3050]*y[IDX_oH2II];
    data[6168] = 0.0 - k[2109]*y[IDX_oH2II];
    data[6169] = 0.0 - k[802]*y[IDX_oH2II] - k[803]*y[IDX_oH2II] -
        k[2464]*y[IDX_oH2II];
    data[6170] = 0.0 - k[2094]*y[IDX_oH2II] + k[2647]*y[IDX_HII];
    data[6171] = 0.0 + k[2647]*y[IDX_HI];
    data[6172] = 0.0 - k[2151]*y[IDX_oH2II];
    data[6173] = 0.0 - k[2444]*y[IDX_oH2II] - k[2989]*y[IDX_oH2II];
    data[6174] = 0.0 - k[2104]*y[IDX_oH2II];
    data[6175] = 0.0 - k[795]*y[IDX_oH2II] - k[2459]*y[IDX_oH2II] -
        k[2921]*y[IDX_oH2II];
    data[6176] = 0.0 - k[728]*y[IDX_oH2II] - k[729]*y[IDX_oH2II] -
        k[730]*y[IDX_oH2II] - k[2889]*y[IDX_oH2II];
    data[6177] = 0.0 - k[2454]*y[IDX_oH2II] - k[3038]*y[IDX_oH2II] -
        k[3040]*y[IDX_oH2II];
    data[6178] = 0.0 + k[2472]*y[IDX_oH2I];
    data[6179] = 0.0 - k[637]*y[IDX_oH2II];
    data[6180] = 0.0 - k[741]*y[IDX_oH2II];
    data[6181] = 0.0 - k[755]*y[IDX_oH2II] - k[757]*y[IDX_oH2II] -
        k[2394]*y[IDX_oH2II];
    data[6182] = 0.0 - k[2119]*y[IDX_oH2II];
    data[6183] = 0.0 - k[747]*y[IDX_oH2II] - k[2389]*y[IDX_oH2II];
    data[6184] = 0.0 - k[2114]*y[IDX_oH2II];
    data[6185] = 0.0 - k[2124]*y[IDX_oH2II];
    data[6186] = 0.0 - k[763]*y[IDX_oH2II] - k[2399]*y[IDX_oH2II];
    data[6187] = 0.0 - k[643]*y[IDX_oH2II];
    data[6188] = 0.0 - k[769]*y[IDX_oH2II] - k[2404]*y[IDX_oH2II];
    data[6189] = 0.0 - k[783]*y[IDX_oH2II] - k[785]*y[IDX_oH2II] -
        k[2414]*y[IDX_oH2II];
    data[6190] = 0.0 - k[775]*y[IDX_oH2II] - k[2409]*y[IDX_oH2II];
    data[6191] = 0.0 - k[2746]*y[IDX_oH2II] - k[2751]*y[IDX_oH2II];
    data[6192] = 0.0 - k[704]*y[IDX_oH2II] - k[705]*y[IDX_oH2II] -
        k[709]*y[IDX_oH2II];
    data[6193] = 0.0 + k[226] - k[684]*y[IDX_oH2II] + k[2472]*y[IDX_HeII] -
        k[2876]*y[IDX_oH2II];
    data[6194] = 0.0 - k[291] - k[631]*y[IDX_CI] - k[637]*y[IDX_NI] -
        k[643]*y[IDX_OI] - k[649]*y[IDX_C2I] - k[655]*y[IDX_CHI] -
        k[663]*y[IDX_CDI] - k[665]*y[IDX_CDI] - k[671]*y[IDX_CNI] -
        k[677]*y[IDX_COI] - k[682]*y[IDX_pH2I] - k[684]*y[IDX_oH2I] -
        k[700]*y[IDX_pD2I] - k[701]*y[IDX_pD2I] - k[704]*y[IDX_oD2I] -
        k[705]*y[IDX_oD2I] - k[707]*y[IDX_pD2I] - k[709]*y[IDX_oD2I] -
        k[728]*y[IDX_HDI] - k[729]*y[IDX_HDI] - k[730]*y[IDX_HDI] -
        k[741]*y[IDX_N2I] - k[747]*y[IDX_NHI] - k[755]*y[IDX_NDI] -
        k[757]*y[IDX_NDI] - k[763]*y[IDX_NOI] - k[769]*y[IDX_O2I] -
        k[775]*y[IDX_OHI] - k[783]*y[IDX_ODI] - k[785]*y[IDX_ODI] -
        k[791]*y[IDX_CO2I] - k[795]*y[IDX_HCOI] - k[802]*y[IDX_DCOI] -
        k[803]*y[IDX_DCOI] - k[2094]*y[IDX_HI] - k[2099]*y[IDX_DI] -
        k[2104]*y[IDX_HCNI] - k[2109]*y[IDX_DCNI] - k[2114]*y[IDX_NH2I] -
        k[2119]*y[IDX_ND2I] - k[2124]*y[IDX_NHDI] - k[2151]*y[IDX_HM] -
        k[2153]*y[IDX_DM] - k[2341]*y[IDX_C2I] - k[2346]*y[IDX_CHI] -
        k[2351]*y[IDX_CDI] - k[2356]*y[IDX_CNI] - k[2361]*y[IDX_COI] -
        k[2389]*y[IDX_NHI] - k[2394]*y[IDX_NDI] - k[2399]*y[IDX_NOI] -
        k[2404]*y[IDX_O2I] - k[2409]*y[IDX_OHI] - k[2414]*y[IDX_ODI] -
        k[2419]*y[IDX_C2HI] - k[2424]*y[IDX_C2DI] - k[2429]*y[IDX_CH2I] -
        k[2434]*y[IDX_CD2I] - k[2439]*y[IDX_CHDI] - k[2444]*y[IDX_H2OI] -
        k[2449]*y[IDX_D2OI] - k[2454]*y[IDX_HDOI] - k[2459]*y[IDX_HCOI] -
        k[2464]*y[IDX_DCOI] - k[2539]*y[IDX_CO2I] - k[2746]*y[IDX_eM] -
        k[2751]*y[IDX_eM] - k[2876]*y[IDX_oH2I] - k[2878]*y[IDX_pH2I] -
        k[2889]*y[IDX_HDI] - k[2921]*y[IDX_HCOI] - k[2989]*y[IDX_H2OI] -
        k[3038]*y[IDX_HDOI] - k[3040]*y[IDX_HDOI] - k[3048]*y[IDX_D2OI] -
        k[3050]*y[IDX_D2OI];
    data[6195] = 0.0 + k[335];
    data[6196] = 0.0 + k[325];
    data[6197] = 0.0 - k[700]*y[IDX_oH2II] - k[701]*y[IDX_oH2II] -
        k[707]*y[IDX_oH2II];
    data[6198] = 0.0 - k[682]*y[IDX_oH2II] - k[2878]*y[IDX_oH2II];
    data[6199] = 0.0 + k[337];
    data[6200] = 0.0 + k[327];
    data[6201] = 0.0 - k[1261]*y[IDX_oH2DII] - k[1264]*y[IDX_oH2DII];
    data[6202] = 0.0 - k[1315]*y[IDX_oH2DII] - k[1318]*y[IDX_oH2DII];
    data[6203] = 0.0 - k[1347]*y[IDX_oH2DII] - k[1350]*y[IDX_oH2DII];
    data[6204] = 0.0 - k[1330]*y[IDX_oH2DII] - k[1333]*y[IDX_oH2DII];
    data[6205] = 0.0 - k[1362]*y[IDX_oH2DII] - k[1365]*y[IDX_oH2DII];
    data[6206] = 0.0 - k[1377]*y[IDX_oH2DII] - k[1380]*y[IDX_oH2DII] -
        k[1392]*y[IDX_oH2DII] - k[1395]*y[IDX_oH2DII];
    data[6207] = 0.0 - k[2857]*y[IDX_oH2DII] - k[2858]*y[IDX_oH2DII] +
        k[2891]*y[IDX_pH3II] + k[2892]*y[IDX_oH3II];
    data[6208] = 0.0 - k[840]*y[IDX_oH2DII] - k[841]*y[IDX_oH2DII] -
        k[845]*y[IDX_oH2DII];
    data[6209] = 0.0 - k[3249]*y[IDX_oH2DII] - k[3251]*y[IDX_oH2DII] -
        k[3254]*y[IDX_oH2DII] - k[3255]*y[IDX_oH2DII];
    data[6210] = 0.0 + k[802]*y[IDX_oH2II];
    data[6211] = 0.0 - k[186]*y[IDX_oH2DII] - k[187]*y[IDX_oH2DII] -
        k[188]*y[IDX_oH2DII];
    data[6212] = 0.0 + k[2861]*y[IDX_pD2HII] + k[2862]*y[IDX_oD2HII] -
        k[2894]*y[IDX_oH2DII] - k[2895]*y[IDX_oH2DII];
    data[6213] = 0.0 - k[836]*y[IDX_oH2DII] - k[837]*y[IDX_oH2DII];
    data[6214] = 0.0 - k[3177]*y[IDX_oH2DII] - k[3178]*y[IDX_oH2DII] -
        k[3181]*y[IDX_oH2DII];
    data[6215] = 0.0 + k[799]*y[IDX_HDII];
    data[6216] = 0.0 + k[13]*y[IDX_pH3II] + k[14]*y[IDX_pH3II] +
        k[17]*y[IDX_oH3II] + k[18]*y[IDX_oH3II] + k[47]*y[IDX_pH2DII] -
        k[52]*y[IDX_oH2DII] - k[53]*y[IDX_oH2DII] - k[54]*y[IDX_oH2DII] -
        k[55]*y[IDX_oH2DII] - k[56]*y[IDX_oH2DII] - k[57]*y[IDX_oH2DII] -
        k[58]*y[IDX_oH2DII] - k[59]*y[IDX_oH2DII] - k[60]*y[IDX_oH2DII] +
        k[98]*y[IDX_pD2HII] + k[99]*y[IDX_pD2HII] + k[105]*y[IDX_oD2HII] +
        k[106]*y[IDX_oD2HII] + k[726]*y[IDX_pH2II] + k[728]*y[IDX_oH2II] +
        k[738]*y[IDX_HDII] + k[823]*y[IDX_HeHII] + k[1850]*y[IDX_NHII] +
        k[2028]*y[IDX_O2HII];
    data[6217] = 0.0 + k[693]*y[IDX_pH2I] + k[695]*y[IDX_oH2I] +
        k[738]*y[IDX_HDI] + k[799]*y[IDX_HCOI];
    data[6218] = 0.0 - k[3212]*y[IDX_oH2DII] - k[3213]*y[IDX_oH2DII] -
        k[3215]*y[IDX_oH2DII] - k[3218]*y[IDX_oH2DII] - k[3219]*y[IDX_oH2DII];
    data[6219] = 0.0 + k[815]*y[IDX_oH2I];
    data[6220] = 0.0 + k[823]*y[IDX_HDI];
    data[6221] = 0.0 - k[1275]*y[IDX_oH2DII] - k[1277]*y[IDX_oH2DII];
    data[6222] = 0.0 - k[1407]*y[IDX_oH2DII] - k[1410]*y[IDX_oH2DII];
    data[6223] = 0.0 - k[1439]*y[IDX_oH2DII] - k[1442]*y[IDX_oH2DII];
    data[6224] = 0.0 + k[1842]*y[IDX_oH2I];
    data[6225] = 0.0 - k[1422]*y[IDX_oH2DII] - k[1425]*y[IDX_oH2DII];
    data[6226] = 0.0 + k[1850]*y[IDX_HDI];
    data[6227] = 0.0 - k[1454]*y[IDX_oH2DII] - k[1457]*y[IDX_oH2DII];
    data[6228] = 0.0 - k[1516]*y[IDX_oH2DII] - k[1519]*y[IDX_oH2DII];
    data[6229] = 0.0 - k[1288]*y[IDX_oH2DII] - k[1291]*y[IDX_oH2DII] -
        k[1302]*y[IDX_oH2DII] - k[1304]*y[IDX_oH2DII];
    data[6230] = 0.0 - k[1469]*y[IDX_oH2DII] - k[1472]*y[IDX_oH2DII];
    data[6231] = 0.0 + k[2020]*y[IDX_oH2I];
    data[6232] = 0.0 + k[2028]*y[IDX_HDI];
    data[6233] = 0.0 - k[1501]*y[IDX_oH2DII] - k[1504]*y[IDX_oH2DII];
    data[6234] = 0.0 - k[1484]*y[IDX_oH2DII] - k[1487]*y[IDX_oH2DII];
    data[6235] = 0.0 - k[2802]*y[IDX_oH2DII] - k[2812]*y[IDX_oH2DII] -
        k[2814]*y[IDX_oH2DII];
    data[6236] = 0.0 + k[114]*y[IDX_oH2I];
    data[6237] = 0.0 + k[38]*y[IDX_pH3II] + k[43]*y[IDX_oH3II] -
        k[91]*y[IDX_oH2DII] + k[91]*y[IDX_oH2DII] - k[92]*y[IDX_oH2DII] -
        k[93]*y[IDX_oH2DII] - k[94]*y[IDX_oH2DII] - k[95]*y[IDX_oH2DII] +
        k[709]*y[IDX_oH2II];
    data[6238] = 0.0 + k[692]*y[IDX_oH2I];
    data[6239] = 0.0 + k[71]*y[IDX_pH2I] + k[76]*y[IDX_oH2I] +
        k[105]*y[IDX_HDI] + k[106]*y[IDX_HDI] + k[2862]*y[IDX_HI];
    data[6240] = 0.0 + k[120]*y[IDX_oH2I] + k[121]*y[IDX_oH2I];
    data[6241] = 0.0 + k[23]*y[IDX_pH2DII] + k[24]*y[IDX_pH2DII] -
        k[28]*y[IDX_oH2DII] - k[29]*y[IDX_oH2DII] - k[30]*y[IDX_oH2DII] -
        k[31]*y[IDX_oH2DII] - k[32]*y[IDX_oH2DII] + k[32]*y[IDX_oH2DII] +
        k[67]*y[IDX_pD2HII] + k[76]*y[IDX_oD2HII] + k[114]*y[IDX_mD3II] +
        k[120]*y[IDX_oD3II] + k[121]*y[IDX_oD3II] + k[691]*y[IDX_pD2II] +
        k[692]*y[IDX_oD2II] + k[695]*y[IDX_HDII] + k[815]*y[IDX_HeDII] +
        k[1842]*y[IDX_NDII] + k[2020]*y[IDX_O2DII] + k[2938]*y[IDX_pD3II];
    data[6242] = 0.0 + k[707]*y[IDX_pD2I] + k[709]*y[IDX_oD2I] +
        k[728]*y[IDX_HDI] + k[802]*y[IDX_DCOI];
    data[6243] = 0.0 - k[25]*y[IDX_pH2I] - k[26]*y[IDX_pH2I] -
        k[27]*y[IDX_pH2I] + k[27]*y[IDX_pH2I] - k[28]*y[IDX_oH2I] -
        k[29]*y[IDX_oH2I] - k[30]*y[IDX_oH2I] - k[31]*y[IDX_oH2I] -
        k[32]*y[IDX_oH2I] + k[32]*y[IDX_oH2I] - k[52]*y[IDX_HDI] -
        k[53]*y[IDX_HDI] - k[54]*y[IDX_HDI] - k[55]*y[IDX_HDI] -
        k[56]*y[IDX_HDI] - k[57]*y[IDX_HDI] - k[58]*y[IDX_HDI] -
        k[59]*y[IDX_HDI] - k[60]*y[IDX_HDI] - k[87]*y[IDX_pD2I] +
        k[87]*y[IDX_pD2I] - k[88]*y[IDX_pD2I] - k[89]*y[IDX_pD2I] -
        k[90]*y[IDX_pD2I] - k[91]*y[IDX_oD2I] + k[91]*y[IDX_oD2I] -
        k[92]*y[IDX_oD2I] - k[93]*y[IDX_oD2I] - k[94]*y[IDX_oD2I] -
        k[95]*y[IDX_oD2I] - k[186]*y[IDX_GRAINM] - k[187]*y[IDX_GRAINM] -
        k[188]*y[IDX_GRAINM] - k[312] - k[313] - k[316] - k[332] - k[334] -
        k[335] - k[836]*y[IDX_HM] - k[837]*y[IDX_HM] - k[840]*y[IDX_DM] -
        k[841]*y[IDX_DM] - k[845]*y[IDX_DM] - k[1261]*y[IDX_CI] -
        k[1264]*y[IDX_CI] - k[1275]*y[IDX_NI] - k[1277]*y[IDX_NI] -
        k[1288]*y[IDX_OI] - k[1291]*y[IDX_OI] - k[1302]*y[IDX_OI] -
        k[1304]*y[IDX_OI] - k[1315]*y[IDX_C2I] - k[1318]*y[IDX_C2I] -
        k[1330]*y[IDX_CHI] - k[1333]*y[IDX_CHI] - k[1347]*y[IDX_CDI] -
        k[1350]*y[IDX_CDI] - k[1362]*y[IDX_CNI] - k[1365]*y[IDX_CNI] -
        k[1377]*y[IDX_COI] - k[1380]*y[IDX_COI] - k[1392]*y[IDX_COI] -
        k[1395]*y[IDX_COI] - k[1407]*y[IDX_N2I] - k[1410]*y[IDX_N2I] -
        k[1422]*y[IDX_NHI] - k[1425]*y[IDX_NHI] - k[1439]*y[IDX_NDI] -
        k[1442]*y[IDX_NDI] - k[1454]*y[IDX_NOI] - k[1457]*y[IDX_NOI] -
        k[1469]*y[IDX_O2I] - k[1472]*y[IDX_O2I] - k[1484]*y[IDX_OHI] -
        k[1487]*y[IDX_OHI] - k[1501]*y[IDX_ODI] - k[1504]*y[IDX_ODI] -
        k[1516]*y[IDX_NO2I] - k[1519]*y[IDX_NO2I] - k[2802]*y[IDX_eM] -
        k[2812]*y[IDX_eM] - k[2814]*y[IDX_eM] - k[2857]*y[IDX_DI] -
        k[2858]*y[IDX_DI] - k[2894]*y[IDX_HI] - k[2895]*y[IDX_HI] -
        k[2903]*y[IDX_pD2I] - k[3177]*y[IDX_H2OI] - k[3178]*y[IDX_H2OI] -
        k[3181]*y[IDX_H2OI] - k[3212]*y[IDX_HDOI] - k[3213]*y[IDX_HDOI] -
        k[3215]*y[IDX_HDOI] - k[3218]*y[IDX_HDOI] - k[3219]*y[IDX_HDOI] -
        k[3249]*y[IDX_D2OI] - k[3251]*y[IDX_D2OI] - k[3254]*y[IDX_D2OI] -
        k[3255]*y[IDX_D2OI];
    data[6244] = 0.0 + k[17]*y[IDX_HDI] + k[18]*y[IDX_HDI] +
        k[41]*y[IDX_pD2I] + k[43]*y[IDX_oD2I] + k[2892]*y[IDX_DI];
    data[6245] = 0.0 + k[34]*y[IDX_pH3II] + k[41]*y[IDX_oH3II] -
        k[87]*y[IDX_oH2DII] + k[87]*y[IDX_oH2DII] - k[88]*y[IDX_oH2DII] -
        k[89]*y[IDX_oH2DII] - k[90]*y[IDX_oH2DII] + k[707]*y[IDX_oH2II] -
        k[2903]*y[IDX_oH2DII];
    data[6246] = 0.0 + k[691]*y[IDX_oH2I];
    data[6247] = 0.0 + k[63]*y[IDX_pH2I] + k[67]*y[IDX_oH2I] +
        k[98]*y[IDX_HDI] + k[99]*y[IDX_HDI] + k[2861]*y[IDX_HI];
    data[6248] = 0.0 + k[2938]*y[IDX_oH2I];
    data[6249] = 0.0 + k[20]*y[IDX_pH2DII] - k[25]*y[IDX_oH2DII] -
        k[26]*y[IDX_oH2DII] - k[27]*y[IDX_oH2DII] + k[27]*y[IDX_oH2DII] +
        k[63]*y[IDX_pD2HII] + k[71]*y[IDX_oD2HII] + k[693]*y[IDX_HDII];
    data[6250] = 0.0 + k[726]*y[IDX_HDI];
    data[6251] = 0.0 + k[20]*y[IDX_pH2I] + k[23]*y[IDX_oH2I] +
        k[24]*y[IDX_oH2I] + k[47]*y[IDX_HDI];
    data[6252] = 0.0 + k[13]*y[IDX_HDI] + k[14]*y[IDX_HDI] +
        k[34]*y[IDX_pD2I] + k[38]*y[IDX_oD2I] + k[2891]*y[IDX_DI];
    data[6253] = 0.0 - k[1256]*y[IDX_oH3II];
    data[6254] = 0.0 - k[1310]*y[IDX_oH3II];
    data[6255] = 0.0 - k[1340]*y[IDX_oH3II] - k[1343]*y[IDX_oH3II];
    data[6256] = 0.0 - k[1325]*y[IDX_oH3II];
    data[6257] = 0.0 - k[1357]*y[IDX_oH3II];
    data[6258] = 0.0 - k[1372]*y[IDX_oH3II] - k[1387]*y[IDX_oH3II];
    data[6259] = 0.0 - k[2892]*y[IDX_oH3II];
    data[6260] = 0.0 - k[830]*y[IDX_oH3II] - k[831]*y[IDX_oH3II];
    data[6261] = 0.0 - k[3241]*y[IDX_oH3II] - k[3243]*y[IDX_oH3II] -
        k[3245]*y[IDX_oH3II] - k[3247]*y[IDX_oH3II];
    data[6262] = 0.0 - k[181]*y[IDX_oH3II] - k[182]*y[IDX_oH3II];
    data[6263] = 0.0 + k[2894]*y[IDX_oH2DII];
    data[6264] = 0.0 - k[827]*y[IDX_oH3II];
    data[6265] = 0.0 - k[3003]*y[IDX_oH3II] - k[3004]*y[IDX_oH3II];
    data[6266] = 0.0 + k[795]*y[IDX_oH2II];
    data[6267] = 0.0 + k[10]*y[IDX_pH3II] - k[15]*y[IDX_oH3II] -
        k[16]*y[IDX_oH3II] - k[17]*y[IDX_oH3II] - k[18]*y[IDX_oH3II] +
        k[54]*y[IDX_oH2DII] + k[55]*y[IDX_oH2DII] + k[730]*y[IDX_oH2II];
    data[6268] = 0.0 + k[697]*y[IDX_pH2I] + k[698]*y[IDX_oH2I];
    data[6269] = 0.0 - k[3204]*y[IDX_oH3II] - k[3205]*y[IDX_oH3II] -
        k[3208]*y[IDX_oH3II];
    data[6270] = 0.0 + k[813]*y[IDX_oH2I];
    data[6271] = 0.0 - k[1271]*y[IDX_oH3II];
    data[6272] = 0.0 - k[1402]*y[IDX_oH3II];
    data[6273] = 0.0 - k[1432]*y[IDX_oH3II] - k[1435]*y[IDX_oH3II];
    data[6274] = 0.0 - k[1417]*y[IDX_oH3II];
    data[6275] = 0.0 + k[1840]*y[IDX_oH2I];
    data[6276] = 0.0 - k[1449]*y[IDX_oH3II];
    data[6277] = 0.0 - k[1511]*y[IDX_oH3II];
    data[6278] = 0.0 - k[1283]*y[IDX_oH3II] - k[1298]*y[IDX_oH3II];
    data[6279] = 0.0 - k[1464]*y[IDX_oH3II];
    data[6280] = 0.0 + k[2018]*y[IDX_oH2I];
    data[6281] = 0.0 - k[1494]*y[IDX_oH3II] - k[1497]*y[IDX_oH3II];
    data[6282] = 0.0 - k[1479]*y[IDX_oH3II];
    data[6283] = 0.0 - k[2798]*y[IDX_oH3II] - k[2806]*y[IDX_oH3II];
    data[6284] = 0.0 - k[43]*y[IDX_oH3II] - k[44]*y[IDX_oH3II];
    data[6285] = 0.0 + k[74]*y[IDX_oH2I];
    data[6286] = 0.0 + k[3]*y[IDX_pH3II] + k[4]*y[IDX_pH3II] -
        k[7]*y[IDX_oH3II] - k[8]*y[IDX_oH3II] - k[9]*y[IDX_oH3II] +
        k[9]*y[IDX_oH3II] + k[22]*y[IDX_pH2DII] + k[29]*y[IDX_oH2DII] +
        k[65]*y[IDX_pD2HII] + k[74]*y[IDX_oD2HII] + k[683]*y[IDX_pH2II] +
        k[684]*y[IDX_oH2II] + k[698]*y[IDX_HDII] + k[813]*y[IDX_HeHII] +
        k[1840]*y[IDX_NHII] + k[2018]*y[IDX_O2HII];
    data[6287] = 0.0 + k[682]*y[IDX_pH2I] + k[684]*y[IDX_oH2I] +
        k[730]*y[IDX_HDI] + k[795]*y[IDX_HCOI];
    data[6288] = 0.0 + k[29]*y[IDX_oH2I] + k[54]*y[IDX_HDI] +
        k[55]*y[IDX_HDI] + k[2894]*y[IDX_HI];
    data[6289] = 0.0 - k[5]*y[IDX_pH2I] - k[6]*y[IDX_pH2I] +
        k[6]*y[IDX_pH2I] - k[7]*y[IDX_oH2I] - k[8]*y[IDX_oH2I] -
        k[9]*y[IDX_oH2I] + k[9]*y[IDX_oH2I] - k[15]*y[IDX_HDI] -
        k[16]*y[IDX_HDI] - k[17]*y[IDX_HDI] - k[18]*y[IDX_HDI] -
        k[41]*y[IDX_pD2I] - k[42]*y[IDX_pD2I] - k[43]*y[IDX_oD2I] -
        k[44]*y[IDX_oD2I] - k[181]*y[IDX_GRAINM] - k[182]*y[IDX_GRAINM] - k[304]
        - k[305] - k[324] - k[325] - k[827]*y[IDX_HM] - k[830]*y[IDX_DM] -
        k[831]*y[IDX_DM] - k[1256]*y[IDX_CI] - k[1271]*y[IDX_NI] -
        k[1283]*y[IDX_OI] - k[1298]*y[IDX_OI] - k[1310]*y[IDX_C2I] -
        k[1325]*y[IDX_CHI] - k[1340]*y[IDX_CDI] - k[1343]*y[IDX_CDI] -
        k[1357]*y[IDX_CNI] - k[1372]*y[IDX_COI] - k[1387]*y[IDX_COI] -
        k[1402]*y[IDX_N2I] - k[1417]*y[IDX_NHI] - k[1432]*y[IDX_NDI] -
        k[1435]*y[IDX_NDI] - k[1449]*y[IDX_NOI] - k[1464]*y[IDX_O2I] -
        k[1479]*y[IDX_OHI] - k[1494]*y[IDX_ODI] - k[1497]*y[IDX_ODI] -
        k[1511]*y[IDX_NO2I] - k[2798]*y[IDX_eM] - k[2806]*y[IDX_eM] -
        k[2892]*y[IDX_DI] - k[3003]*y[IDX_H2OI] - k[3004]*y[IDX_H2OI] -
        k[3204]*y[IDX_HDOI] - k[3205]*y[IDX_HDOI] - k[3208]*y[IDX_HDOI] -
        k[3241]*y[IDX_D2OI] - k[3243]*y[IDX_D2OI] - k[3245]*y[IDX_D2OI] -
        k[3247]*y[IDX_D2OI];
    data[6290] = 0.0 - k[41]*y[IDX_oH3II] - k[42]*y[IDX_oH3II];
    data[6291] = 0.0 + k[65]*y[IDX_oH2I];
    data[6292] = 0.0 + k[1]*y[IDX_pH3II] - k[5]*y[IDX_oH3II] -
        k[6]*y[IDX_oH3II] + k[6]*y[IDX_oH3II] + k[682]*y[IDX_oH2II] +
        k[697]*y[IDX_HDII];
    data[6293] = 0.0 + k[683]*y[IDX_oH2I];
    data[6294] = 0.0 + k[22]*y[IDX_oH2I];
    data[6295] = 0.0 + k[1]*y[IDX_pH2I] + k[3]*y[IDX_oH2I] +
        k[4]*y[IDX_oH2I] + k[10]*y[IDX_HDI];
    data[6296] = 0.0 - k[1132]*y[IDX_pD2I] + k[1260]*y[IDX_mD3II] +
        k[1267]*y[IDX_pD2HII] - k[2659]*y[IDX_pD2I] + k[2957]*y[IDX_pD3II] +
        k[3287]*y[IDX_HD2OII] + k[3289]*y[IDX_D3OII];
    data[6297] = 0.0 - k[2643]*y[IDX_pD2I];
    data[6298] = 0.0 - k[2679]*y[IDX_pD2I];
    data[6299] = 0.0 + k[1314]*y[IDX_mD3II] + k[1321]*y[IDX_pD2HII] +
        k[2342]*y[IDX_pD2II] + k[2962]*y[IDX_pD3II];
    data[6300] = 0.0 - k[1696]*y[IDX_pD2I];
    data[6301] = 0.0 + k[1529]*y[IDX_DII] + k[1764]*y[IDX_CDII] +
        k[2425]*y[IDX_pD2II];
    data[6302] = 0.0 + k[2420]*y[IDX_pD2II];
    data[6303] = 0.0 + k[1067]*y[IDX_DI] - k[1148]*y[IDX_pD2I] +
        k[1346]*y[IDX_mD3II] + k[1353]*y[IDX_pD2HII] + k[1726]*y[IDX_CDII] +
        k[2352]*y[IDX_pD2II] + k[2964]*y[IDX_pD3II];
    data[6304] = 0.0 + k[1716]*y[IDX_DI] + k[1726]*y[IDX_CDI] -
        k[1741]*y[IDX_pD2I] + k[1750]*y[IDX_NDI] + k[1760]*y[IDX_ODI] +
        k[1764]*y[IDX_C2DI] + k[1770]*y[IDX_CD2I] + k[1773]*y[IDX_CHDI] +
        k[1782]*y[IDX_D2OI] + k[1785]*y[IDX_HDOI] + k[1790]*y[IDX_DCNI] +
        k[1800]*y[IDX_ND2I] + k[1803]*y[IDX_NHDI];
    data[6305] = 0.0 + k[513]*y[IDX_OI] + k[893]*y[IDX_HeII] +
        k[1069]*y[IDX_HI] + k[1075]*y[IDX_DI] + k[1535]*y[IDX_HII] +
        k[1537]*y[IDX_DII] + k[1768]*y[IDX_CHII] + k[1770]*y[IDX_CDII] +
        k[2435]*y[IDX_pD2II];
    data[6306] = 0.0 + k[2776]*y[IDX_eM];
    data[6307] = 0.0 - k[1138]*y[IDX_pD2I] - k[1140]*y[IDX_pD2I] +
        k[1336]*y[IDX_pD2HII] + k[2347]*y[IDX_pD2II];
    data[6308] = 0.0 - k[1737]*y[IDX_pD2I] - k[1739]*y[IDX_pD2I] +
        k[1768]*y[IDX_CD2I] + k[1780]*y[IDX_D2OI] + k[1798]*y[IDX_ND2I];
    data[6309] = 0.0 + k[2430]*y[IDX_pD2II];
    data[6310] = 0.0 + k[1076]*y[IDX_DI] + k[1540]*y[IDX_DII] +
        k[1773]*y[IDX_CDII] + k[2440]*y[IDX_pD2II];
    data[6311] = 0.0 + k[1361]*y[IDX_mD3II] + k[1368]*y[IDX_pD2HII] +
        k[2357]*y[IDX_pD2II] + k[2966]*y[IDX_pD3II];
    data[6312] = 0.0 - k[470]*y[IDX_pD2I] - k[476]*y[IDX_pD2I];
    data[6313] = 0.0 + k[1376]*y[IDX_mD3II] + k[1383]*y[IDX_pD2HII] +
        k[1391]*y[IDX_mD3II] + k[1398]*y[IDX_pD2HII] + k[2362]*y[IDX_pD2II] +
        k[2912]*y[IDX_pD3II] + k[2969]*y[IDX_pD3II];
    data[6314] = 0.0 - k[458]*y[IDX_pD2I] - k[464]*y[IDX_pD2I];
    data[6315] = 0.0 + k[2540]*y[IDX_pD2II];
    data[6316] = 0.0 + k[167]*y[IDX_DI] + k[167]*y[IDX_DI] +
        k[1059]*y[IDX_DCOI] + k[1067]*y[IDX_CDI] + k[1075]*y[IDX_CD2I] +
        k[1076]*y[IDX_CHDI] + k[1081]*y[IDX_ODI] + k[1089]*y[IDX_D2OI] +
        k[1090]*y[IDX_HDOI] + k[1095]*y[IDX_DCNI] + k[1115]*y[IDX_O2DI] +
        k[1716]*y[IDX_CDII] + k[2100]*y[IDX_pD2II] + k[2692]*y[IDX_DM] +
        k[3390]*y[IDX_H2DOII] + k[3392]*y[IDX_HD2OII] + k[3395]*y[IDX_D3OII];
    data[6317] = 0.0 + k[1529]*y[IDX_C2DI] + k[1537]*y[IDX_CD2I] +
        k[1540]*y[IDX_CHDI] + k[1547]*y[IDX_DCOI] + k[1688]*y[IDX_DNOI] +
        k[2853]*y[IDX_HDI] + k[2872]*y[IDX_oD2I] - k[2873]*y[IDX_pD2I];
    data[6318] = 0.0 + k[488]*y[IDX_D2OI] + k[491]*y[IDX_HDOI] +
        k[496]*y[IDX_DCNI] + k[834]*y[IDX_oD3II] + k[835]*y[IDX_mD3II] +
        k[835]*y[IDX_mD3II] + k[841]*y[IDX_oH2DII] + k[842]*y[IDX_pH2DII] +
        k[844]*y[IDX_pH2DII] + k[854]*y[IDX_oD2HII] + k[856]*y[IDX_pD2HII] +
        k[861]*y[IDX_DCOII] + k[2156]*y[IDX_pD2II] + k[2692]*y[IDX_DI] +
        k[2983]*y[IDX_pD3II] + k[2984]*y[IDX_pD3II] + k[2984]*y[IDX_pD3II] +
        k[3066]*y[IDX_H2DOII] + k[3077]*y[IDX_HD2OII] + k[3080]*y[IDX_HD2OII] +
        k[3087]*y[IDX_D3OII] + k[3099]*y[IDX_H2DOII] + k[3106]*y[IDX_HD2OII] +
        k[3112]*y[IDX_D3OII];
    data[6319] = 0.0 + k[486]*y[IDX_HM] + k[488]*y[IDX_DM] +
        k[1083]*y[IDX_HI] + k[1089]*y[IDX_DI] + k[1780]*y[IDX_CHII] +
        k[1782]*y[IDX_CDII] + k[1893]*y[IDX_NHII] + k[1895]*y[IDX_NDII] +
        k[2450]*y[IDX_pD2II] + k[3244]*y[IDX_pH3II] + k[3247]*y[IDX_oH3II] +
        k[3253]*y[IDX_pH2DII] + k[3254]*y[IDX_oH2DII] + k[3259]*y[IDX_pD2HII] +
        k[3261]*y[IDX_oD2HII] + k[3262]*y[IDX_pD3II] + k[3264]*y[IDX_mD3II] +
        k[3266]*y[IDX_oD3II];
    data[6320] = 0.0 + k[996]*y[IDX_OI] + k[2789]*y[IDX_eM] -
        k[3146]*y[IDX_pD2I];
    data[6321] = 0.0 + k[3082]*y[IDX_HM] + k[3085]*y[IDX_HM] +
        k[3087]*y[IDX_DM] + k[3109]*y[IDX_HM] + k[3112]*y[IDX_DM] +
        k[3289]*y[IDX_CI] + k[3381]*y[IDX_HI] + k[3395]*y[IDX_DI] -
        k[3439]*y[IDX_pD2I] + k[3456]*y[IDX_eM] + k[3464]*y[IDX_eM];
    data[6322] = 0.0 + k[496]*y[IDX_DM] + k[1095]*y[IDX_DI] +
        k[1790]*y[IDX_CDII] + k[2110]*y[IDX_pD2II];
    data[6323] = 0.0 + k[1059]*y[IDX_DI] + k[1547]*y[IDX_DII] +
        k[2465]*y[IDX_pD2II];
    data[6324] = 0.0 + k[861]*y[IDX_DM];
    data[6325] = 0.0 + k[1688]*y[IDX_DII];
    data[6326] = 0.0 - k[1605]*y[IDX_pD2I] + k[1605]*y[IDX_pD2I] +
        k[1606]*y[IDX_oD2I] + k[1611]*y[IDX_HDI];
    data[6327] = 0.0 + k[176]*y[IDX_pD2II] + k[192]*y[IDX_pD2HII] +
        k[199]*y[IDX_oD3II] + k[2917]*y[IDX_pD3II];
    data[6328] = 0.0 + k[1069]*y[IDX_CD2I] + k[1083]*y[IDX_D2OI] +
        k[2095]*y[IDX_pD2II] + k[3378]*y[IDX_HD2OII] + k[3381]*y[IDX_D3OII];
    data[6329] = 0.0 + k[1535]*y[IDX_CD2I] - k[2851]*y[IDX_pD2I];
    data[6330] = 0.0 + k[486]*y[IDX_D2OI] + k[848]*y[IDX_oD2HII] +
        k[849]*y[IDX_pD2HII] + k[851]*y[IDX_pD2HII] + k[2154]*y[IDX_pD2II] +
        k[3072]*y[IDX_HD2OII] + k[3082]*y[IDX_D3OII] + k[3085]*y[IDX_D3OII] +
        k[3103]*y[IDX_HD2OII] + k[3109]*y[IDX_D3OII];
    data[6331] = 0.0 + k[3066]*y[IDX_DM] + k[3099]*y[IDX_DM] +
        k[3390]*y[IDX_DI] - k[3429]*y[IDX_pD2I] - k[3431]*y[IDX_pD2I] -
        k[3433]*y[IDX_pD2I];
    data[6332] = 0.0 + k[2445]*y[IDX_pD2II] + k[3190]*y[IDX_pD2HII] +
        k[3200]*y[IDX_pD3II] + k[3203]*y[IDX_oD3II];
    data[6333] = 0.0 - k[3138]*y[IDX_pD2I] - k[3140]*y[IDX_pD2I];
    data[6334] = 0.0 - k[3422]*y[IDX_pD2I] - k[3424]*y[IDX_pD2I] -
        k[3426]*y[IDX_pD2I];
    data[6335] = 0.0 + k[2105]*y[IDX_pD2II];
    data[6336] = 0.0 + k[2460]*y[IDX_pD2II];
    data[6337] = 0.0 + k[45]*y[IDX_pH2DII] + k[52]*y[IDX_oH2DII] +
        k[54]*y[IDX_oH2DII] + k[96]*y[IDX_pD2HII] + k[98]*y[IDX_pD2HII] +
        k[103]*y[IDX_oD2HII] + k[105]*y[IDX_oD2HII] + k[144]*y[IDX_mD3II] +
        k[147]*y[IDX_oD3II] + k[149]*y[IDX_oD3II] + k[1611]*y[IDX_DOCII] +
        k[2853]*y[IDX_DII] + k[2939]*y[IDX_pD3II] + k[2941]*y[IDX_pD3II];
    data[6338] = 0.0 - k[718]*y[IDX_pD2I] - k[719]*y[IDX_pD2I] -
        k[722]*y[IDX_pD2I] - k[723]*y[IDX_pD2I] - k[2932]*y[IDX_pD2I];
    data[6339] = 0.0 + k[3072]*y[IDX_HM] + k[3077]*y[IDX_DM] +
        k[3080]*y[IDX_DM] + k[3103]*y[IDX_HM] + k[3106]*y[IDX_DM] +
        k[3287]*y[IDX_CI] + k[3378]*y[IDX_HI] + k[3392]*y[IDX_DI] -
        k[3435]*y[IDX_pD2I] - k[3437]*y[IDX_pD2I] + k[3454]*y[IDX_eM] +
        k[3461]*y[IDX_eM];
    data[6340] = 0.0 + k[491]*y[IDX_DM] + k[1090]*y[IDX_DI] +
        k[1785]*y[IDX_CDII] + k[1898]*y[IDX_NDII] + k[2455]*y[IDX_pD2II] +
        k[3216]*y[IDX_pH2DII] + k[3218]*y[IDX_oH2DII] + k[3227]*y[IDX_pD2HII] +
        k[3229]*y[IDX_oD2HII] + k[3234]*y[IDX_mD3II] + k[3236]*y[IDX_pD3II] +
        k[3238]*y[IDX_oD3II];
    data[6341] = 0.0 - k[3141]*y[IDX_pD2I] - k[3143]*y[IDX_pD2I];
    data[6342] = 0.0 - k[1600]*y[IDX_pD2I] + k[1600]*y[IDX_pD2I] +
        k[1601]*y[IDX_oD2I] - k[1603]*y[IDX_pD2I];
    data[6343] = 0.0 - k[870]*y[IDX_pD2I] + k[893]*y[IDX_CD2I] +
        k[940]*y[IDX_ND2I] - k[2473]*y[IDX_pD2I];
    data[6344] = 0.0 - k[820]*y[IDX_pD2I] - k[2953]*y[IDX_pD2I];
    data[6345] = 0.0 - k[817]*y[IDX_pD2I];
    data[6346] = 0.0 - k[1176]*y[IDX_pD2I];
    data[6347] = 0.0 - k[1638]*y[IDX_pD2I];
    data[6348] = 0.0 + k[1406]*y[IDX_mD3II] + k[1413]*y[IDX_pD2HII] +
        k[2913]*y[IDX_pD3II];
    data[6349] = 0.0 - k[1808]*y[IDX_pD2I];
    data[6350] = 0.0 - k[1190]*y[IDX_pD2I] + k[1438]*y[IDX_mD3II] +
        k[1445]*y[IDX_pD2HII] + k[1750]*y[IDX_CDII] + k[2395]*y[IDX_pD2II] +
        k[2972]*y[IDX_pD3II];
    data[6351] = 0.0 - k[1847]*y[IDX_pD2I] - k[1864]*y[IDX_pD2I] +
        k[1895]*y[IDX_D2OI] + k[1898]*y[IDX_HDOI] - k[2955]*y[IDX_pD2I];
    data[6352] = 0.0 + k[940]*y[IDX_HeII] + k[1798]*y[IDX_CHII] +
        k[1800]*y[IDX_CDII] + k[2120]*y[IDX_pD2II];
    data[6353] = 0.0 - k[1186]*y[IDX_pD2I] - k[1188]*y[IDX_pD2I] +
        k[1428]*y[IDX_pD2HII] + k[2390]*y[IDX_pD2II];
    data[6354] = 0.0 - k[1844]*y[IDX_pD2I] - k[1860]*y[IDX_pD2I] -
        k[1862]*y[IDX_pD2I] + k[1893]*y[IDX_D2OI];
    data[6355] = 0.0 + k[2115]*y[IDX_pD2II];
    data[6356] = 0.0 + k[1803]*y[IDX_CDII] + k[2125]*y[IDX_pD2II];
    data[6357] = 0.0 + k[1453]*y[IDX_mD3II] + k[1460]*y[IDX_pD2HII] +
        k[2400]*y[IDX_pD2II] + k[2974]*y[IDX_pD3II];
    data[6358] = 0.0 + k[1515]*y[IDX_mD3II] + k[1522]*y[IDX_pD2HII] +
        k[2980]*y[IDX_pD3II];
    data[6359] = 0.0 - k[1999]*y[IDX_pD2I];
    data[6360] = 0.0 + k[513]*y[IDX_CD2I] + k[996]*y[IDX_D2OII] -
        k[1154]*y[IDX_pD2I] + k[1287]*y[IDX_mD3II] + k[1294]*y[IDX_pD2HII] +
        k[2960]*y[IDX_pD3II];
    data[6361] = 0.0 - k[1658]*y[IDX_pD2I];
    data[6362] = 0.0 - k[499]*y[IDX_pD2I] - k[2725]*y[IDX_pD2I];
    data[6363] = 0.0 + k[1468]*y[IDX_mD3II] + k[1475]*y[IDX_pD2HII] +
        k[2405]*y[IDX_pD2II] + k[2976]*y[IDX_pD3II];
    data[6364] = 0.0 + k[1115]*y[IDX_DI];
    data[6365] = 0.0 - k[2025]*y[IDX_pD2I] - k[2982]*y[IDX_pD2I];
    data[6366] = 0.0 - k[2022]*y[IDX_pD2I];
    data[6367] = 0.0 + k[1081]*y[IDX_DI] - k[1168]*y[IDX_pD2I] +
        k[1500]*y[IDX_mD3II] + k[1507]*y[IDX_pD2HII] + k[1760]*y[IDX_CDII] +
        k[2415]*y[IDX_pD2II] + k[2978]*y[IDX_pD3II];
    data[6368] = 0.0 - k[1947]*y[IDX_pD2I];
    data[6369] = 0.0 - k[1164]*y[IDX_pD2I] - k[1166]*y[IDX_pD2I] +
        k[1490]*y[IDX_pD2HII] + k[2410]*y[IDX_pD2II];
    data[6370] = 0.0 - k[1943]*y[IDX_pD2I] - k[1945]*y[IDX_pD2I];
    data[6371] = 0.0 + k[2752]*y[IDX_pD2II] + k[2776]*y[IDX_CD2II] +
        k[2789]*y[IDX_D2OII] + k[2810]*y[IDX_oD3II] + k[2817]*y[IDX_pD2HII] +
        k[2915]*y[IDX_pD3II] + k[3454]*y[IDX_HD2OII] + k[3456]*y[IDX_D3OII] +
        k[3461]*y[IDX_HD2OII] + k[3464]*y[IDX_D3OII];
    data[6372] = 0.0 + k[144]*y[IDX_HDI] - k[152]*y[IDX_pD2I] -
        k[153]*y[IDX_pD2I] + k[153]*y[IDX_pD2I] - k[154]*y[IDX_pD2I] +
        k[155]*y[IDX_oD2I] + k[156]*y[IDX_oD2I] + k[310] + k[835]*y[IDX_DM] +
        k[835]*y[IDX_DM] + k[1260]*y[IDX_CI] + k[1287]*y[IDX_OI] +
        k[1314]*y[IDX_C2I] + k[1346]*y[IDX_CDI] + k[1361]*y[IDX_CNI] +
        k[1376]*y[IDX_COI] + k[1391]*y[IDX_COI] + k[1406]*y[IDX_N2I] +
        k[1438]*y[IDX_NDI] + k[1453]*y[IDX_NOI] + k[1468]*y[IDX_O2I] +
        k[1500]*y[IDX_ODI] + k[1515]*y[IDX_NO2I] - k[2908]*y[IDX_pD2I] +
        k[3234]*y[IDX_HDOI] + k[3264]*y[IDX_D2OI];
    data[6373] = 0.0 + k[82]*y[IDX_pH2DII] + k[91]*y[IDX_oH2DII] +
        k[128]*y[IDX_pD2HII] + k[129]*y[IDX_pD2HII] + k[138]*y[IDX_oD2HII] +
        k[140]*y[IDX_oD2HII] + k[155]*y[IDX_mD3II] + k[156]*y[IDX_mD3II] +
        k[161]*y[IDX_oD3II] + k[163]*y[IDX_oD3II] + k[1601]*y[IDX_HOCII] +
        k[1606]*y[IDX_DOCII] + k[2872]*y[IDX_DII] + k[2910]*y[IDX_pD3II] +
        k[2911]*y[IDX_pD3II] + k[2948]*y[IDX_oD3II];
    data[6374] = 0.0 - k[712]*y[IDX_pD2I] - k[713]*y[IDX_pD2I] -
        k[2933]*y[IDX_pD2I];
    data[6375] = 0.0 + k[103]*y[IDX_HDI] + k[105]*y[IDX_HDI] -
        k[133]*y[IDX_pD2I] + k[133]*y[IDX_pD2I] - k[134]*y[IDX_pD2I] -
        k[135]*y[IDX_pD2I] - k[136]*y[IDX_pD2I] - k[137]*y[IDX_pD2I] +
        k[138]*y[IDX_oD2I] + k[140]*y[IDX_oD2I] + k[318] + k[848]*y[IDX_HM] +
        k[854]*y[IDX_DM] - k[2905]*y[IDX_pD2I] + k[3229]*y[IDX_HDOI] +
        k[3261]*y[IDX_D2OI];
    data[6376] = 0.0 + k[116]*y[IDX_pH2I] + k[120]*y[IDX_oH2I] +
        k[147]*y[IDX_HDI] + k[149]*y[IDX_HDI] - k[158]*y[IDX_pD2I] +
        k[158]*y[IDX_pD2I] - k[159]*y[IDX_pD2I] - k[160]*y[IDX_pD2I] +
        k[161]*y[IDX_oD2I] + k[163]*y[IDX_oD2I] + k[199]*y[IDX_GRAINM] + k[308]
        + k[834]*y[IDX_DM] + k[2810]*y[IDX_eM] - k[2909]*y[IDX_pD2I] -
        k[2947]*y[IDX_pD2I] + k[2947]*y[IDX_pD2I] + k[2948]*y[IDX_oD2I] +
        k[3203]*y[IDX_H2OI] + k[3238]*y[IDX_HDOI] + k[3266]*y[IDX_D2OI];
    data[6377] = 0.0 + k[65]*y[IDX_pD2HII] + k[120]*y[IDX_oD3II] +
        k[2938]*y[IDX_pD3II];
    data[6378] = 0.0 - k[700]*y[IDX_pD2I] - k[701]*y[IDX_pD2I] -
        k[707]*y[IDX_pD2I];
    data[6379] = 0.0 + k[52]*y[IDX_HDI] + k[54]*y[IDX_HDI] -
        k[87]*y[IDX_pD2I] - k[88]*y[IDX_pD2I] - k[89]*y[IDX_pD2I] -
        k[90]*y[IDX_pD2I] + k[91]*y[IDX_oD2I] + k[841]*y[IDX_DM] -
        k[2903]*y[IDX_pD2I] + k[3218]*y[IDX_HDOI] + k[3254]*y[IDX_D2OI];
    data[6380] = 0.0 - k[41]*y[IDX_pD2I] - k[42]*y[IDX_pD2I] +
        k[3247]*y[IDX_D2OI];
    data[6381] = 0.0 - k[33]*y[IDX_pH3II] - k[34]*y[IDX_pH3II] -
        k[35]*y[IDX_pH3II] - k[36]*y[IDX_pH3II] - k[41]*y[IDX_oH3II] -
        k[42]*y[IDX_oH3II] - k[78]*y[IDX_pH2DII] - k[79]*y[IDX_pH2DII] -
        k[80]*y[IDX_pH2DII] - k[81]*y[IDX_pH2DII] - k[87]*y[IDX_oH2DII] -
        k[88]*y[IDX_oH2DII] - k[89]*y[IDX_oH2DII] - k[90]*y[IDX_oH2DII] -
        k[124]*y[IDX_pD2HII] - k[125]*y[IDX_pD2HII] + k[125]*y[IDX_pD2HII] -
        k[126]*y[IDX_pD2HII] - k[127]*y[IDX_pD2HII] - k[133]*y[IDX_oD2HII] +
        k[133]*y[IDX_oD2HII] - k[134]*y[IDX_oD2HII] - k[135]*y[IDX_oD2HII] -
        k[136]*y[IDX_oD2HII] - k[137]*y[IDX_oD2HII] - k[152]*y[IDX_mD3II] -
        k[153]*y[IDX_mD3II] + k[153]*y[IDX_mD3II] - k[154]*y[IDX_mD3II] -
        k[158]*y[IDX_oD3II] + k[158]*y[IDX_oD3II] - k[159]*y[IDX_oD3II] -
        k[160]*y[IDX_oD3II] - k[210] - k[215] - k[221] - k[227] - k[362] -
        k[458]*y[IDX_COII] - k[464]*y[IDX_COII] - k[470]*y[IDX_CNII] -
        k[476]*y[IDX_CNII] - k[499]*y[IDX_OM] - k[699]*y[IDX_pH2II] -
        k[700]*y[IDX_oH2II] - k[701]*y[IDX_oH2II] - k[706]*y[IDX_pH2II] -
        k[707]*y[IDX_oH2II] - k[710]*y[IDX_pD2II] - k[711]*y[IDX_pD2II] -
        k[712]*y[IDX_oD2II] - k[713]*y[IDX_oD2II] - k[718]*y[IDX_HDII] -
        k[719]*y[IDX_HDII] - k[722]*y[IDX_HDII] - k[723]*y[IDX_HDII] -
        k[817]*y[IDX_HeHII] - k[820]*y[IDX_HeDII] - k[870]*y[IDX_HeII] -
        k[1132]*y[IDX_CI] - k[1138]*y[IDX_CHI] - k[1140]*y[IDX_CHI] -
        k[1148]*y[IDX_CDI] - k[1154]*y[IDX_OI] - k[1164]*y[IDX_OHI] -
        k[1166]*y[IDX_OHI] - k[1168]*y[IDX_ODI] - k[1176]*y[IDX_NI] -
        k[1186]*y[IDX_NHI] - k[1188]*y[IDX_NHI] - k[1190]*y[IDX_NDI] -
        k[1600]*y[IDX_HOCII] + k[1600]*y[IDX_HOCII] - k[1603]*y[IDX_HOCII] -
        k[1605]*y[IDX_DOCII] + k[1605]*y[IDX_DOCII] - k[1638]*y[IDX_NII] -
        k[1658]*y[IDX_OII] - k[1696]*y[IDX_C2II] - k[1737]*y[IDX_CHII] -
        k[1739]*y[IDX_CHII] - k[1741]*y[IDX_CDII] - k[1808]*y[IDX_N2II] -
        k[1844]*y[IDX_NHII] - k[1847]*y[IDX_NDII] - k[1860]*y[IDX_NHII] -
        k[1862]*y[IDX_NHII] - k[1864]*y[IDX_NDII] - k[1943]*y[IDX_OHII] -
        k[1945]*y[IDX_OHII] - k[1947]*y[IDX_ODII] - k[1999]*y[IDX_NO2II] -
        k[2022]*y[IDX_O2HII] - k[2025]*y[IDX_O2DII] - k[2473]*y[IDX_HeII] -
        k[2643]*y[IDX_CII] - k[2659]*y[IDX_CI] - k[2679]*y[IDX_CM] -
        k[2725]*y[IDX_OM] - k[2851]*y[IDX_HII] - k[2873]*y[IDX_DII] -
        k[2903]*y[IDX_oH2DII] - k[2904]*y[IDX_pH2DII] - k[2905]*y[IDX_oD2HII] -
        k[2907]*y[IDX_pD2HII] - k[2908]*y[IDX_mD3II] - k[2909]*y[IDX_oD3II] -
        k[2932]*y[IDX_HDII] - k[2933]*y[IDX_oD2II] - k[2935]*y[IDX_pD2II] -
        k[2944]*y[IDX_pD3II] + k[2944]*y[IDX_pD3II] - k[2945]*y[IDX_pD3II] -
        k[2947]*y[IDX_oD3II] + k[2947]*y[IDX_oD3II] - k[2953]*y[IDX_HeDII] -
        k[2955]*y[IDX_NDII] - k[2982]*y[IDX_O2DII] - k[3138]*y[IDX_H2OII] -
        k[3140]*y[IDX_H2OII] - k[3141]*y[IDX_HDOII] - k[3143]*y[IDX_HDOII] -
        k[3146]*y[IDX_D2OII] - k[3422]*y[IDX_H3OII] - k[3424]*y[IDX_H3OII] -
        k[3426]*y[IDX_H3OII] - k[3429]*y[IDX_H2DOII] - k[3431]*y[IDX_H2DOII] -
        k[3433]*y[IDX_H2DOII] - k[3435]*y[IDX_HD2OII] - k[3437]*y[IDX_HD2OII] -
        k[3439]*y[IDX_D3OII];
    data[6382] = 0.0 + k[176]*y[IDX_GRAINM] - k[710]*y[IDX_pD2I] -
        k[711]*y[IDX_pD2I] + k[2095]*y[IDX_HI] + k[2100]*y[IDX_DI] +
        k[2105]*y[IDX_HCNI] + k[2110]*y[IDX_DCNI] + k[2115]*y[IDX_NH2I] +
        k[2120]*y[IDX_ND2I] + k[2125]*y[IDX_NHDI] + k[2154]*y[IDX_HM] +
        k[2156]*y[IDX_DM] + k[2342]*y[IDX_C2I] + k[2347]*y[IDX_CHI] +
        k[2352]*y[IDX_CDI] + k[2357]*y[IDX_CNI] + k[2362]*y[IDX_COI] +
        k[2390]*y[IDX_NHI] + k[2395]*y[IDX_NDI] + k[2400]*y[IDX_NOI] +
        k[2405]*y[IDX_O2I] + k[2410]*y[IDX_OHI] + k[2415]*y[IDX_ODI] +
        k[2420]*y[IDX_C2HI] + k[2425]*y[IDX_C2DI] + k[2430]*y[IDX_CH2I] +
        k[2435]*y[IDX_CD2I] + k[2440]*y[IDX_CHDI] + k[2445]*y[IDX_H2OI] +
        k[2450]*y[IDX_D2OI] + k[2455]*y[IDX_HDOI] + k[2460]*y[IDX_HCOI] +
        k[2465]*y[IDX_DCOI] + k[2540]*y[IDX_CO2I] + k[2752]*y[IDX_eM] -
        k[2935]*y[IDX_pD2I];
    data[6383] = 0.0 + k[61]*y[IDX_pH2I] + k[65]*y[IDX_oH2I] +
        k[96]*y[IDX_HDI] + k[98]*y[IDX_HDI] - k[124]*y[IDX_pD2I] -
        k[125]*y[IDX_pD2I] + k[125]*y[IDX_pD2I] - k[126]*y[IDX_pD2I] -
        k[127]*y[IDX_pD2I] + k[128]*y[IDX_oD2I] + k[129]*y[IDX_oD2I] +
        k[192]*y[IDX_GRAINM] + k[320] + k[849]*y[IDX_HM] + k[851]*y[IDX_HM] +
        k[856]*y[IDX_DM] + k[1267]*y[IDX_CI] + k[1294]*y[IDX_OI] +
        k[1321]*y[IDX_C2I] + k[1336]*y[IDX_CHI] + k[1353]*y[IDX_CDI] +
        k[1368]*y[IDX_CNI] + k[1383]*y[IDX_COI] + k[1398]*y[IDX_COI] +
        k[1413]*y[IDX_N2I] + k[1428]*y[IDX_NHI] + k[1445]*y[IDX_NDI] +
        k[1460]*y[IDX_NOI] + k[1475]*y[IDX_O2I] + k[1490]*y[IDX_OHI] +
        k[1507]*y[IDX_ODI] + k[1522]*y[IDX_NO2I] + k[2817]*y[IDX_eM] -
        k[2907]*y[IDX_pD2I] + k[3190]*y[IDX_H2OI] + k[3227]*y[IDX_HDOI] +
        k[3259]*y[IDX_D2OI];
    data[6384] = 0.0 + k[2910]*y[IDX_oD2I] + k[2911]*y[IDX_oD2I] +
        k[2912]*y[IDX_COI] + k[2913]*y[IDX_N2I] + k[2915]*y[IDX_eM] +
        k[2917]*y[IDX_GRAINM] + k[2936]*y[IDX_pH2I] + k[2938]*y[IDX_oH2I] +
        k[2939]*y[IDX_HDI] + k[2941]*y[IDX_HDI] - k[2944]*y[IDX_pD2I] +
        k[2944]*y[IDX_pD2I] - k[2945]*y[IDX_pD2I] + k[2957]*y[IDX_CI] +
        k[2960]*y[IDX_OI] + k[2962]*y[IDX_C2I] + k[2964]*y[IDX_CDI] +
        k[2966]*y[IDX_CNI] + k[2969]*y[IDX_COI] + k[2972]*y[IDX_NDI] +
        k[2974]*y[IDX_NOI] + k[2976]*y[IDX_O2I] + k[2978]*y[IDX_ODI] +
        k[2980]*y[IDX_NO2I] + k[2983]*y[IDX_DM] + k[2984]*y[IDX_DM] +
        k[2984]*y[IDX_DM] + k[2986] + k[3200]*y[IDX_H2OI] + k[3236]*y[IDX_HDOI]
        + k[3262]*y[IDX_D2OI];
    data[6385] = 0.0 + k[61]*y[IDX_pD2HII] + k[116]*y[IDX_oD3II] +
        k[2936]*y[IDX_pD3II];
    data[6386] = 0.0 - k[699]*y[IDX_pD2I] - k[706]*y[IDX_pD2I];
    data[6387] = 0.0 + k[45]*y[IDX_HDI] - k[78]*y[IDX_pD2I] -
        k[79]*y[IDX_pD2I] - k[80]*y[IDX_pD2I] - k[81]*y[IDX_pD2I] +
        k[82]*y[IDX_oD2I] + k[842]*y[IDX_DM] + k[844]*y[IDX_DM] -
        k[2904]*y[IDX_pD2I] + k[3216]*y[IDX_HDOI] + k[3253]*y[IDX_D2OI];
    data[6388] = 0.0 - k[33]*y[IDX_pD2I] - k[34]*y[IDX_pD2I] -
        k[35]*y[IDX_pD2I] - k[36]*y[IDX_pD2I] + k[3244]*y[IDX_D2OI];
    data[6389] = 0.0 - k[632]*y[IDX_pD2II];
    data[6390] = 0.0 - k[650]*y[IDX_pD2II] - k[2342]*y[IDX_pD2II];
    data[6391] = 0.0 - k[2425]*y[IDX_pD2II];
    data[6392] = 0.0 - k[2420]*y[IDX_pD2II];
    data[6393] = 0.0 - k[666]*y[IDX_pD2II] - k[2352]*y[IDX_pD2II];
    data[6394] = 0.0 - k[2435]*y[IDX_pD2II];
    data[6395] = 0.0 - k[656]*y[IDX_pD2II] - k[658]*y[IDX_pD2II] -
        k[2347]*y[IDX_pD2II];
    data[6396] = 0.0 - k[2430]*y[IDX_pD2II];
    data[6397] = 0.0 - k[2440]*y[IDX_pD2II];
    data[6398] = 0.0 - k[672]*y[IDX_pD2II] - k[2357]*y[IDX_pD2II];
    data[6399] = 0.0 - k[678]*y[IDX_pD2II] - k[2362]*y[IDX_pD2II];
    data[6400] = 0.0 - k[792]*y[IDX_pD2II] - k[2540]*y[IDX_pD2II];
    data[6401] = 0.0 + k[812]*y[IDX_HeDII] - k[2100]*y[IDX_pD2II] +
        k[2650]*y[IDX_DII] + k[2863]*y[IDX_HDII];
    data[6402] = 0.0 + k[1551]*y[IDX_DCOI] + k[2650]*y[IDX_DI];
    data[6403] = 0.0 - k[2156]*y[IDX_pD2II];
    data[6404] = 0.0 - k[2450]*y[IDX_pD2II] - k[3054]*y[IDX_pD2II];
    data[6405] = 0.0 - k[2110]*y[IDX_pD2II];
    data[6406] = 0.0 - k[804]*y[IDX_pD2II] + k[1551]*y[IDX_DII] -
        k[2465]*y[IDX_pD2II] - k[2951]*y[IDX_pD2II];
    data[6407] = 0.0 - k[175]*y[IDX_pD2II] - k[176]*y[IDX_pD2II];
    data[6408] = 0.0 - k[2095]*y[IDX_pD2II] - k[2865]*y[IDX_pD2II];
    data[6409] = 0.0 - k[2154]*y[IDX_pD2II];
    data[6410] = 0.0 - k[2445]*y[IDX_pD2II] - k[3034]*y[IDX_pD2II] -
        k[3036]*y[IDX_pD2II];
    data[6411] = 0.0 - k[2105]*y[IDX_pD2II];
    data[6412] = 0.0 - k[796]*y[IDX_pD2II] - k[2460]*y[IDX_pD2II];
    data[6413] = 0.0 - k[731]*y[IDX_pD2II] - k[734]*y[IDX_pD2II];
    data[6414] = 0.0 + k[2863]*y[IDX_DI];
    data[6415] = 0.0 - k[2455]*y[IDX_pD2II] - k[3043]*y[IDX_pD2II] -
        k[3045]*y[IDX_pD2II];
    data[6416] = 0.0 + k[2473]*y[IDX_pD2I];
    data[6417] = 0.0 + k[812]*y[IDX_DI];
    data[6418] = 0.0 - k[638]*y[IDX_pD2II];
    data[6419] = 0.0 - k[742]*y[IDX_pD2II];
    data[6420] = 0.0 - k[758]*y[IDX_pD2II] - k[2395]*y[IDX_pD2II];
    data[6421] = 0.0 - k[2120]*y[IDX_pD2II];
    data[6422] = 0.0 - k[748]*y[IDX_pD2II] - k[750]*y[IDX_pD2II] -
        k[2390]*y[IDX_pD2II];
    data[6423] = 0.0 - k[2115]*y[IDX_pD2II];
    data[6424] = 0.0 - k[2125]*y[IDX_pD2II];
    data[6425] = 0.0 - k[764]*y[IDX_pD2II] - k[2400]*y[IDX_pD2II];
    data[6426] = 0.0 - k[644]*y[IDX_pD2II];
    data[6427] = 0.0 - k[770]*y[IDX_pD2II] - k[2405]*y[IDX_pD2II];
    data[6428] = 0.0 - k[786]*y[IDX_pD2II] - k[2415]*y[IDX_pD2II];
    data[6429] = 0.0 - k[776]*y[IDX_pD2II] - k[778]*y[IDX_pD2II] -
        k[2410]*y[IDX_pD2II];
    data[6430] = 0.0 - k[2747]*y[IDX_pD2II] - k[2752]*y[IDX_pD2II] -
        k[2753]*y[IDX_pD2II];
    data[6431] = 0.0 + k[330];
    data[6432] = 0.0 - k[714]*y[IDX_pD2II] - k[715]*y[IDX_pD2II] -
        k[2934]*y[IDX_pD2II];
    data[6433] = 0.0 + k[338];
    data[6434] = 0.0 + k[328];
    data[6435] = 0.0 - k[687]*y[IDX_pD2II] - k[691]*y[IDX_pD2II];
    data[6436] = 0.0 + k[227] - k[710]*y[IDX_pD2II] - k[711]*y[IDX_pD2II] +
        k[2473]*y[IDX_HeII] - k[2935]*y[IDX_pD2II];
    data[6437] = 0.0 - k[175]*y[IDX_GRAINM] - k[176]*y[IDX_GRAINM] - k[292]
        - k[632]*y[IDX_CI] - k[638]*y[IDX_NI] - k[644]*y[IDX_OI] -
        k[650]*y[IDX_C2I] - k[656]*y[IDX_CHI] - k[658]*y[IDX_CHI] -
        k[666]*y[IDX_CDI] - k[672]*y[IDX_CNI] - k[678]*y[IDX_COI] -
        k[685]*y[IDX_pH2I] - k[687]*y[IDX_oH2I] - k[689]*y[IDX_pH2I] -
        k[691]*y[IDX_oH2I] - k[710]*y[IDX_pD2I] - k[711]*y[IDX_pD2I] -
        k[714]*y[IDX_oD2I] - k[715]*y[IDX_oD2I] - k[731]*y[IDX_HDI] -
        k[734]*y[IDX_HDI] - k[742]*y[IDX_N2I] - k[748]*y[IDX_NHI] -
        k[750]*y[IDX_NHI] - k[758]*y[IDX_NDI] - k[764]*y[IDX_NOI] -
        k[770]*y[IDX_O2I] - k[776]*y[IDX_OHI] - k[778]*y[IDX_OHI] -
        k[786]*y[IDX_ODI] - k[792]*y[IDX_CO2I] - k[796]*y[IDX_HCOI] -
        k[804]*y[IDX_DCOI] - k[2095]*y[IDX_HI] - k[2100]*y[IDX_DI] -
        k[2105]*y[IDX_HCNI] - k[2110]*y[IDX_DCNI] - k[2115]*y[IDX_NH2I] -
        k[2120]*y[IDX_ND2I] - k[2125]*y[IDX_NHDI] - k[2154]*y[IDX_HM] -
        k[2156]*y[IDX_DM] - k[2342]*y[IDX_C2I] - k[2347]*y[IDX_CHI] -
        k[2352]*y[IDX_CDI] - k[2357]*y[IDX_CNI] - k[2362]*y[IDX_COI] -
        k[2390]*y[IDX_NHI] - k[2395]*y[IDX_NDI] - k[2400]*y[IDX_NOI] -
        k[2405]*y[IDX_O2I] - k[2410]*y[IDX_OHI] - k[2415]*y[IDX_ODI] -
        k[2420]*y[IDX_C2HI] - k[2425]*y[IDX_C2DI] - k[2430]*y[IDX_CH2I] -
        k[2435]*y[IDX_CD2I] - k[2440]*y[IDX_CHDI] - k[2445]*y[IDX_H2OI] -
        k[2450]*y[IDX_D2OI] - k[2455]*y[IDX_HDOI] - k[2460]*y[IDX_HCOI] -
        k[2465]*y[IDX_DCOI] - k[2540]*y[IDX_CO2I] - k[2747]*y[IDX_eM] -
        k[2752]*y[IDX_eM] - k[2753]*y[IDX_eM] - k[2865]*y[IDX_HI] -
        k[2934]*y[IDX_oD2I] - k[2935]*y[IDX_pD2I] - k[2951]*y[IDX_DCOI] -
        k[3034]*y[IDX_H2OI] - k[3036]*y[IDX_H2OI] - k[3043]*y[IDX_HDOI] -
        k[3045]*y[IDX_HDOI] - k[3054]*y[IDX_D2OI];
    data[6438] = 0.0 + k[340];
    data[6439] = 0.0 + k[2988];
    data[6440] = 0.0 - k[685]*y[IDX_pD2II] - k[689]*y[IDX_pD2II];
    data[6441] = 0.0 - k[1267]*y[IDX_pD2HII] - k[1268]*y[IDX_pD2HII] -
        k[1270]*y[IDX_pD2HII];
    data[6442] = 0.0 - k[1321]*y[IDX_pD2HII] - k[1322]*y[IDX_pD2HII] -
        k[1324]*y[IDX_pD2HII];
    data[6443] = 0.0 - k[1353]*y[IDX_pD2HII] - k[1354]*y[IDX_pD2HII] -
        k[1356]*y[IDX_pD2HII];
    data[6444] = 0.0 - k[1336]*y[IDX_pD2HII] - k[1337]*y[IDX_pD2HII] -
        k[1339]*y[IDX_pD2HII];
    data[6445] = 0.0 - k[1368]*y[IDX_pD2HII] - k[1369]*y[IDX_pD2HII] -
        k[1371]*y[IDX_pD2HII];
    data[6446] = 0.0 - k[1383]*y[IDX_pD2HII] - k[1384]*y[IDX_pD2HII] -
        k[1386]*y[IDX_pD2HII] - k[1398]*y[IDX_pD2HII] - k[1399]*y[IDX_pD2HII] -
        k[1401]*y[IDX_pD2HII];
    data[6447] = 0.0 + k[2855]*y[IDX_pH2DII] + k[2857]*y[IDX_oH2DII] -
        k[2869]*y[IDX_pD2HII];
    data[6448] = 0.0 - k[856]*y[IDX_pD2HII] - k[857]*y[IDX_pD2HII];
    data[6449] = 0.0 - k[3256]*y[IDX_pD2HII] - k[3258]*y[IDX_pD2HII] -
        k[3259]*y[IDX_pD2HII];
    data[6450] = 0.0 + k[808]*y[IDX_HDII];
    data[6451] = 0.0 - k[192]*y[IDX_pD2HII] - k[193]*y[IDX_pD2HII] -
        k[194]*y[IDX_pD2HII];
    data[6452] = 0.0 - k[2859]*y[IDX_pD2HII] - k[2861]*y[IDX_pD2HII] +
        k[2870]*y[IDX_oD3II];
    data[6453] = 0.0 - k[849]*y[IDX_pD2HII] - k[850]*y[IDX_pD2HII] -
        k[851]*y[IDX_pD2HII] - k[853]*y[IDX_pD2HII];
    data[6454] = 0.0 - k[3185]*y[IDX_pD2HII] - k[3186]*y[IDX_pD2HII] -
        k[3188]*y[IDX_pD2HII] - k[3190]*y[IDX_pD2HII];
    data[6455] = 0.0 + k[796]*y[IDX_pD2II] + k[798]*y[IDX_oD2II];
    data[6456] = 0.0 + k[48]*y[IDX_pH2DII] + k[49]*y[IDX_pH2DII] +
        k[57]*y[IDX_oH2DII] + k[58]*y[IDX_oH2DII] - k[96]*y[IDX_pD2HII] -
        k[97]*y[IDX_pD2HII] - k[98]*y[IDX_pD2HII] - k[99]*y[IDX_pD2HII] -
        k[100]*y[IDX_pD2HII] - k[101]*y[IDX_pD2HII] - k[102]*y[IDX_pD2HII] +
        k[107]*y[IDX_oD2HII] + k[143]*y[IDX_mD3II] + k[147]*y[IDX_oD3II] +
        k[148]*y[IDX_oD3II] + k[734]*y[IDX_pD2II] + k[737]*y[IDX_HDII] +
        k[826]*y[IDX_HeDII] + k[1853]*y[IDX_NDII] + k[2031]*y[IDX_O2DII] -
        k[2900]*y[IDX_pD2HII] - k[2901]*y[IDX_pD2HII] + k[2939]*y[IDX_pD3II] +
        k[2940]*y[IDX_pD3II];
    data[6457] = 0.0 + k[723]*y[IDX_pD2I] + k[725]*y[IDX_oD2I] +
        k[737]*y[IDX_HDI] + k[808]*y[IDX_DCOI];
    data[6458] = 0.0 - k[3220]*y[IDX_pD2HII] - k[3221]*y[IDX_pD2HII] -
        k[3224]*y[IDX_pD2HII] - k[3226]*y[IDX_pD2HII] - k[3227]*y[IDX_pD2HII];
    data[6459] = 0.0 + k[826]*y[IDX_HDI];
    data[6460] = 0.0 + k[817]*y[IDX_pD2I] + k[819]*y[IDX_oD2I];
    data[6461] = 0.0 - k[1280]*y[IDX_pD2HII] - k[1282]*y[IDX_pD2HII];
    data[6462] = 0.0 - k[1413]*y[IDX_pD2HII] - k[1414]*y[IDX_pD2HII] -
        k[1416]*y[IDX_pD2HII];
    data[6463] = 0.0 - k[1445]*y[IDX_pD2HII] - k[1446]*y[IDX_pD2HII] -
        k[1448]*y[IDX_pD2HII];
    data[6464] = 0.0 + k[1853]*y[IDX_HDI];
    data[6465] = 0.0 - k[1428]*y[IDX_pD2HII] - k[1429]*y[IDX_pD2HII] -
        k[1431]*y[IDX_pD2HII];
    data[6466] = 0.0 + k[1844]*y[IDX_pD2I] + k[1846]*y[IDX_oD2I];
    data[6467] = 0.0 - k[1460]*y[IDX_pD2HII] - k[1461]*y[IDX_pD2HII] -
        k[1463]*y[IDX_pD2HII];
    data[6468] = 0.0 - k[1522]*y[IDX_pD2HII] - k[1523]*y[IDX_pD2HII] -
        k[1525]*y[IDX_pD2HII];
    data[6469] = 0.0 - k[1294]*y[IDX_pD2HII] - k[1295]*y[IDX_pD2HII] -
        k[1297]*y[IDX_pD2HII] - k[1307]*y[IDX_pD2HII] - k[1309]*y[IDX_pD2HII];
    data[6470] = 0.0 - k[1475]*y[IDX_pD2HII] - k[1476]*y[IDX_pD2HII] -
        k[1478]*y[IDX_pD2HII];
    data[6471] = 0.0 + k[2031]*y[IDX_HDI];
    data[6472] = 0.0 + k[2022]*y[IDX_pD2I] + k[2024]*y[IDX_oD2I];
    data[6473] = 0.0 - k[1507]*y[IDX_pD2HII] - k[1508]*y[IDX_pD2HII] -
        k[1510]*y[IDX_pD2HII];
    data[6474] = 0.0 - k[1490]*y[IDX_pD2HII] - k[1491]*y[IDX_pD2HII] -
        k[1493]*y[IDX_pD2HII];
    data[6475] = 0.0 - k[2805]*y[IDX_pD2HII] - k[2817]*y[IDX_pD2HII] -
        k[2819]*y[IDX_pD2HII];
    data[6476] = 0.0 + k[143]*y[IDX_HDI];
    data[6477] = 0.0 + k[83]*y[IDX_pH2DII] + k[92]*y[IDX_oH2DII] -
        k[128]*y[IDX_pD2HII] + k[128]*y[IDX_pD2HII] - k[129]*y[IDX_pD2HII] -
        k[130]*y[IDX_pD2HII] - k[131]*y[IDX_pD2HII] - k[132]*y[IDX_pD2HII] +
        k[138]*y[IDX_oD2HII] + k[139]*y[IDX_oD2HII] + k[703]*y[IDX_pH2II] +
        k[705]*y[IDX_oH2II] + k[725]*y[IDX_HDII] + k[819]*y[IDX_HeHII] +
        k[1846]*y[IDX_NHII] + k[2024]*y[IDX_O2HII] - k[2906]*y[IDX_pD2HII];
    data[6478] = 0.0 + k[798]*y[IDX_HCOI];
    data[6479] = 0.0 + k[107]*y[IDX_HDI] + k[133]*y[IDX_pD2I] +
        k[134]*y[IDX_pD2I] + k[138]*y[IDX_oD2I] + k[139]*y[IDX_oD2I];
    data[6480] = 0.0 + k[118]*y[IDX_pH2I] + k[122]*y[IDX_oH2I] +
        k[147]*y[IDX_HDI] + k[148]*y[IDX_HDI] + k[2870]*y[IDX_HI];
    data[6481] = 0.0 - k[65]*y[IDX_pD2HII] - k[66]*y[IDX_pD2HII] -
        k[67]*y[IDX_pD2HII] - k[68]*y[IDX_pD2HII] + k[68]*y[IDX_pD2HII] +
        k[122]*y[IDX_oD3II] + k[687]*y[IDX_pD2II] + k[2902]*y[IDX_pD3II];
    data[6482] = 0.0 + k[701]*y[IDX_pD2I] + k[705]*y[IDX_oD2I];
    data[6483] = 0.0 + k[57]*y[IDX_HDI] + k[58]*y[IDX_HDI] +
        k[88]*y[IDX_pD2I] + k[92]*y[IDX_oD2I] + k[2857]*y[IDX_DI];
    data[6484] = 0.0 + k[42]*y[IDX_pD2I];
    data[6485] = 0.0 + k[35]*y[IDX_pH3II] + k[36]*y[IDX_pH3II] +
        k[42]*y[IDX_oH3II] + k[79]*y[IDX_pH2DII] + k[88]*y[IDX_oH2DII] -
        k[124]*y[IDX_pD2HII] + k[124]*y[IDX_pD2HII] - k[125]*y[IDX_pD2HII] -
        k[126]*y[IDX_pD2HII] - k[127]*y[IDX_pD2HII] + k[133]*y[IDX_oD2HII] +
        k[134]*y[IDX_oD2HII] + k[699]*y[IDX_pH2II] + k[701]*y[IDX_oH2II] +
        k[723]*y[IDX_HDII] + k[817]*y[IDX_HeHII] + k[1844]*y[IDX_NHII] +
        k[2022]*y[IDX_O2HII] - k[2907]*y[IDX_pD2HII];
    data[6486] = 0.0 + k[685]*y[IDX_pH2I] + k[687]*y[IDX_oH2I] +
        k[734]*y[IDX_HDI] + k[796]*y[IDX_HCOI];
    data[6487] = 0.0 - k[61]*y[IDX_pH2I] - k[62]*y[IDX_pH2I] -
        k[63]*y[IDX_pH2I] - k[64]*y[IDX_pH2I] + k[64]*y[IDX_pH2I] -
        k[65]*y[IDX_oH2I] - k[66]*y[IDX_oH2I] - k[67]*y[IDX_oH2I] -
        k[68]*y[IDX_oH2I] + k[68]*y[IDX_oH2I] - k[96]*y[IDX_HDI] -
        k[97]*y[IDX_HDI] - k[98]*y[IDX_HDI] - k[99]*y[IDX_HDI] -
        k[100]*y[IDX_HDI] - k[101]*y[IDX_HDI] - k[102]*y[IDX_HDI] -
        k[124]*y[IDX_pD2I] + k[124]*y[IDX_pD2I] - k[125]*y[IDX_pD2I] -
        k[126]*y[IDX_pD2I] - k[127]*y[IDX_pD2I] - k[128]*y[IDX_oD2I] +
        k[128]*y[IDX_oD2I] - k[129]*y[IDX_oD2I] - k[130]*y[IDX_oD2I] -
        k[131]*y[IDX_oD2I] - k[132]*y[IDX_oD2I] - k[192]*y[IDX_GRAINM] -
        k[193]*y[IDX_GRAINM] - k[194]*y[IDX_GRAINM] - k[320] - k[321] - k[323] -
        k[340] - k[341] - k[343] - k[849]*y[IDX_HM] - k[850]*y[IDX_HM] -
        k[851]*y[IDX_HM] - k[853]*y[IDX_HM] - k[856]*y[IDX_DM] -
        k[857]*y[IDX_DM] - k[1267]*y[IDX_CI] - k[1268]*y[IDX_CI] -
        k[1270]*y[IDX_CI] - k[1280]*y[IDX_NI] - k[1282]*y[IDX_NI] -
        k[1294]*y[IDX_OI] - k[1295]*y[IDX_OI] - k[1297]*y[IDX_OI] -
        k[1307]*y[IDX_OI] - k[1309]*y[IDX_OI] - k[1321]*y[IDX_C2I] -
        k[1322]*y[IDX_C2I] - k[1324]*y[IDX_C2I] - k[1336]*y[IDX_CHI] -
        k[1337]*y[IDX_CHI] - k[1339]*y[IDX_CHI] - k[1353]*y[IDX_CDI] -
        k[1354]*y[IDX_CDI] - k[1356]*y[IDX_CDI] - k[1368]*y[IDX_CNI] -
        k[1369]*y[IDX_CNI] - k[1371]*y[IDX_CNI] - k[1383]*y[IDX_COI] -
        k[1384]*y[IDX_COI] - k[1386]*y[IDX_COI] - k[1398]*y[IDX_COI] -
        k[1399]*y[IDX_COI] - k[1401]*y[IDX_COI] - k[1413]*y[IDX_N2I] -
        k[1414]*y[IDX_N2I] - k[1416]*y[IDX_N2I] - k[1428]*y[IDX_NHI] -
        k[1429]*y[IDX_NHI] - k[1431]*y[IDX_NHI] - k[1445]*y[IDX_NDI] -
        k[1446]*y[IDX_NDI] - k[1448]*y[IDX_NDI] - k[1460]*y[IDX_NOI] -
        k[1461]*y[IDX_NOI] - k[1463]*y[IDX_NOI] - k[1475]*y[IDX_O2I] -
        k[1476]*y[IDX_O2I] - k[1478]*y[IDX_O2I] - k[1490]*y[IDX_OHI] -
        k[1491]*y[IDX_OHI] - k[1493]*y[IDX_OHI] - k[1507]*y[IDX_ODI] -
        k[1508]*y[IDX_ODI] - k[1510]*y[IDX_ODI] - k[1522]*y[IDX_NO2I] -
        k[1523]*y[IDX_NO2I] - k[1525]*y[IDX_NO2I] - k[2805]*y[IDX_eM] -
        k[2817]*y[IDX_eM] - k[2819]*y[IDX_eM] - k[2859]*y[IDX_HI] -
        k[2861]*y[IDX_HI] - k[2869]*y[IDX_DI] - k[2900]*y[IDX_HDI] -
        k[2901]*y[IDX_HDI] - k[2906]*y[IDX_oD2I] - k[2907]*y[IDX_pD2I] -
        k[3185]*y[IDX_H2OI] - k[3186]*y[IDX_H2OI] - k[3188]*y[IDX_H2OI] -
        k[3190]*y[IDX_H2OI] - k[3220]*y[IDX_HDOI] - k[3221]*y[IDX_HDOI] -
        k[3224]*y[IDX_HDOI] - k[3226]*y[IDX_HDOI] - k[3227]*y[IDX_HDOI] -
        k[3256]*y[IDX_D2OI] - k[3258]*y[IDX_D2OI] - k[3259]*y[IDX_D2OI];
    data[6488] = 0.0 + k[2902]*y[IDX_oH2I] + k[2937]*y[IDX_pH2I] +
        k[2939]*y[IDX_HDI] + k[2940]*y[IDX_HDI];
    data[6489] = 0.0 - k[61]*y[IDX_pD2HII] - k[62]*y[IDX_pD2HII] -
        k[63]*y[IDX_pD2HII] - k[64]*y[IDX_pD2HII] + k[64]*y[IDX_pD2HII] +
        k[118]*y[IDX_oD3II] + k[685]*y[IDX_pD2II] + k[2937]*y[IDX_pD3II];
    data[6490] = 0.0 + k[699]*y[IDX_pD2I] + k[703]*y[IDX_oD2I];
    data[6491] = 0.0 + k[48]*y[IDX_HDI] + k[49]*y[IDX_HDI] +
        k[79]*y[IDX_pD2I] + k[83]*y[IDX_oD2I] + k[2855]*y[IDX_DI];
    data[6492] = 0.0 + k[35]*y[IDX_pD2I] + k[36]*y[IDX_pD2I];
    data[6493] = 0.0 - k[2956]*y[IDX_pD3II] - k[2957]*y[IDX_pD3II];
    data[6494] = 0.0 - k[2961]*y[IDX_pD3II] - k[2962]*y[IDX_pD3II];
    data[6495] = 0.0 - k[2963]*y[IDX_pD3II] - k[2964]*y[IDX_pD3II];
    data[6496] = 0.0 - k[2965]*y[IDX_pD3II] - k[2966]*y[IDX_pD3II];
    data[6497] = 0.0 - k[2912]*y[IDX_pD3II] - k[2967]*y[IDX_pD3II] -
        k[2968]*y[IDX_pD3II] - k[2969]*y[IDX_pD3II];
    data[6498] = 0.0 - k[2983]*y[IDX_pD3II] - k[2984]*y[IDX_pD3II];
    data[6499] = 0.0 - k[3262]*y[IDX_pD3II] - k[3263]*y[IDX_pD3II];
    data[6500] = 0.0 + k[2950]*y[IDX_oD2II] + k[2951]*y[IDX_pD2II];
    data[6501] = 0.0 - k[2916]*y[IDX_pD3II] - k[2917]*y[IDX_pD3II];
    data[6502] = 0.0 - k[3191]*y[IDX_pD3II] - k[3194]*y[IDX_pD3II] -
        k[3197]*y[IDX_pD3II] - k[3200]*y[IDX_pD3II];
    data[6503] = 0.0 + k[2900]*y[IDX_pD2HII] + k[2901]*y[IDX_pD2HII] -
        k[2939]*y[IDX_pD3II] - k[2940]*y[IDX_pD3II] - k[2941]*y[IDX_pD3II] -
        k[2942]*y[IDX_pD3II] + k[2943]*y[IDX_oD3II];
    data[6504] = 0.0 + k[2930]*y[IDX_oD2I] + k[2932]*y[IDX_pD2I];
    data[6505] = 0.0 - k[3231]*y[IDX_pD3II] - k[3235]*y[IDX_pD3II] -
        k[3236]*y[IDX_pD3II];
    data[6506] = 0.0 + k[2952]*y[IDX_oD2I] + k[2953]*y[IDX_pD2I];
    data[6507] = 0.0 - k[2958]*y[IDX_pD3II];
    data[6508] = 0.0 - k[2913]*y[IDX_pD3II] - k[2970]*y[IDX_pD3II];
    data[6509] = 0.0 - k[2971]*y[IDX_pD3II] - k[2972]*y[IDX_pD3II];
    data[6510] = 0.0 + k[2954]*y[IDX_oD2I] + k[2955]*y[IDX_pD2I];
    data[6511] = 0.0 - k[2973]*y[IDX_pD3II] - k[2974]*y[IDX_pD3II];
    data[6512] = 0.0 - k[2979]*y[IDX_pD3II] - k[2980]*y[IDX_pD3II];
    data[6513] = 0.0 - k[2959]*y[IDX_pD3II] - k[2960]*y[IDX_pD3II];
    data[6514] = 0.0 - k[2975]*y[IDX_pD3II] - k[2976]*y[IDX_pD3II];
    data[6515] = 0.0 + k[2981]*y[IDX_oD2I] + k[2982]*y[IDX_pD2I];
    data[6516] = 0.0 - k[2977]*y[IDX_pD3II] - k[2978]*y[IDX_pD3II];
    data[6517] = 0.0 - k[2914]*y[IDX_pD3II] - k[2915]*y[IDX_pD3II];
    data[6518] = 0.0 + k[2908]*y[IDX_pD2I];
    data[6519] = 0.0 + k[2906]*y[IDX_pD2HII] - k[2910]*y[IDX_pD3II] -
        k[2911]*y[IDX_pD3II] + k[2930]*y[IDX_HDII] + k[2931]*y[IDX_oD2II] +
        k[2934]*y[IDX_pD2II] - k[2946]*y[IDX_pD3II] + k[2948]*y[IDX_oD3II] +
        k[2949]*y[IDX_oD3II] + k[2952]*y[IDX_HeDII] + k[2954]*y[IDX_NDII] +
        k[2981]*y[IDX_O2DII];
    data[6520] = 0.0 + k[2931]*y[IDX_oD2I] + k[2933]*y[IDX_pD2I] +
        k[2950]*y[IDX_DCOI];
    data[6521] = 0.0 + k[2905]*y[IDX_pD2I];
    data[6522] = 0.0 + k[2909]*y[IDX_pD2I] + k[2943]*y[IDX_HDI] +
        k[2947]*y[IDX_pD2I] + k[2948]*y[IDX_oD2I] + k[2949]*y[IDX_oD2I];
    data[6523] = 0.0 - k[2902]*y[IDX_pD3II] - k[2938]*y[IDX_pD3II];
    data[6524] = 0.0 + k[2903]*y[IDX_pD2I];
    data[6525] = 0.0 + k[2903]*y[IDX_oH2DII] + k[2904]*y[IDX_pH2DII] +
        k[2905]*y[IDX_oD2HII] + k[2907]*y[IDX_pD2HII] + k[2908]*y[IDX_mD3II] +
        k[2909]*y[IDX_oD3II] + k[2932]*y[IDX_HDII] + k[2933]*y[IDX_oD2II] +
        k[2935]*y[IDX_pD2II] - k[2944]*y[IDX_pD3II] - k[2945]*y[IDX_pD3II] +
        k[2947]*y[IDX_oD3II] + k[2953]*y[IDX_HeDII] + k[2955]*y[IDX_NDII] +
        k[2982]*y[IDX_O2DII];
    data[6526] = 0.0 + k[2934]*y[IDX_oD2I] + k[2935]*y[IDX_pD2I] +
        k[2951]*y[IDX_DCOI];
    data[6527] = 0.0 + k[2900]*y[IDX_HDI] + k[2901]*y[IDX_HDI] +
        k[2906]*y[IDX_oD2I] + k[2907]*y[IDX_pD2I];
    data[6528] = 0.0 - k[2902]*y[IDX_oH2I] - k[2910]*y[IDX_oD2I] -
        k[2911]*y[IDX_oD2I] - k[2912]*y[IDX_COI] - k[2913]*y[IDX_N2I] -
        k[2914]*y[IDX_eM] - k[2915]*y[IDX_eM] - k[2916]*y[IDX_GRAINM] -
        k[2917]*y[IDX_GRAINM] - k[2936]*y[IDX_pH2I] - k[2937]*y[IDX_pH2I] -
        k[2938]*y[IDX_oH2I] - k[2939]*y[IDX_HDI] - k[2940]*y[IDX_HDI] -
        k[2941]*y[IDX_HDI] - k[2942]*y[IDX_HDI] - k[2944]*y[IDX_pD2I] -
        k[2945]*y[IDX_pD2I] - k[2946]*y[IDX_oD2I] - k[2956]*y[IDX_CI] -
        k[2957]*y[IDX_CI] - k[2958]*y[IDX_NI] - k[2959]*y[IDX_OI] -
        k[2960]*y[IDX_OI] - k[2961]*y[IDX_C2I] - k[2962]*y[IDX_C2I] -
        k[2963]*y[IDX_CDI] - k[2964]*y[IDX_CDI] - k[2965]*y[IDX_CNI] -
        k[2966]*y[IDX_CNI] - k[2967]*y[IDX_COI] - k[2968]*y[IDX_COI] -
        k[2969]*y[IDX_COI] - k[2970]*y[IDX_N2I] - k[2971]*y[IDX_NDI] -
        k[2972]*y[IDX_NDI] - k[2973]*y[IDX_NOI] - k[2974]*y[IDX_NOI] -
        k[2975]*y[IDX_O2I] - k[2976]*y[IDX_O2I] - k[2977]*y[IDX_ODI] -
        k[2978]*y[IDX_ODI] - k[2979]*y[IDX_NO2I] - k[2980]*y[IDX_NO2I] -
        k[2983]*y[IDX_DM] - k[2984]*y[IDX_DM] - k[2985] - k[2986] - k[2987] -
        k[2988] - k[3191]*y[IDX_H2OI] - k[3194]*y[IDX_H2OI] -
        k[3197]*y[IDX_H2OI] - k[3200]*y[IDX_H2OI] - k[3231]*y[IDX_HDOI] -
        k[3235]*y[IDX_HDOI] - k[3236]*y[IDX_HDOI] - k[3262]*y[IDX_D2OI] -
        k[3263]*y[IDX_D2OI];
    data[6529] = 0.0 - k[2936]*y[IDX_pD3II] - k[2937]*y[IDX_pD3II];
    data[6530] = 0.0 + k[2904]*y[IDX_pD2I];
    data[6531] = 0.0 - k[1130]*y[IDX_pH2I] + k[1257]*y[IDX_pH3II] +
        k[1262]*y[IDX_pH2DII] - k[2657]*y[IDX_pH2I] + k[3010]*y[IDX_H3OII] +
        k[3283]*y[IDX_H2DOII];
    data[6532] = 0.0 - k[2641]*y[IDX_pH2I];
    data[6533] = 0.0 - k[2677]*y[IDX_pH2I];
    data[6534] = 0.0 + k[1311]*y[IDX_pH3II] + k[1316]*y[IDX_pH2DII] +
        k[2340]*y[IDX_pH2II];
    data[6535] = 0.0 - k[1694]*y[IDX_pH2I];
    data[6536] = 0.0 + k[2423]*y[IDX_pH2II];
    data[6537] = 0.0 + k[1526]*y[IDX_HII] + k[1761]*y[IDX_CHII] +
        k[2418]*y[IDX_pH2II];
    data[6538] = 0.0 - k[1144]*y[IDX_pH2I] - k[1146]*y[IDX_pH2I] +
        k[1341]*y[IDX_pH3II] + k[1348]*y[IDX_pH2DII] + k[2350]*y[IDX_pH2II];
    data[6539] = 0.0 - k[1733]*y[IDX_pH2I] - k[1735]*y[IDX_pH2I] +
        k[1766]*y[IDX_CH2I] + k[1778]*y[IDX_H2OI] + k[1796]*y[IDX_NH2I];
    data[6540] = 0.0 + k[2433]*y[IDX_pH2II];
    data[6541] = 0.0 + k[1064]*y[IDX_HI] - k[1136]*y[IDX_pH2I] +
        k[1326]*y[IDX_pH3II] + k[1331]*y[IDX_pH2DII] + k[1723]*y[IDX_CHII] +
        k[2345]*y[IDX_pH2II];
    data[6542] = 0.0 + k[1713]*y[IDX_HI] + k[1723]*y[IDX_CHI] -
        k[1731]*y[IDX_pH2I] + k[1747]*y[IDX_NHI] + k[1757]*y[IDX_OHI] +
        k[1761]*y[IDX_C2HI] + k[1765]*y[IDX_CH2I] + k[1771]*y[IDX_CHDI] +
        k[1777]*y[IDX_H2OI] + k[1783]*y[IDX_HDOI] + k[1787]*y[IDX_HCNI] +
        k[1795]*y[IDX_NH2I] + k[1801]*y[IDX_NHDI];
    data[6543] = 0.0 + k[512]*y[IDX_OI] + k[892]*y[IDX_HeII] +
        k[1068]*y[IDX_HI] + k[1074]*y[IDX_DI] + k[1532]*y[IDX_HII] +
        k[1533]*y[IDX_DII] + k[1765]*y[IDX_CHII] + k[1766]*y[IDX_CDII] +
        k[2428]*y[IDX_pH2II];
    data[6544] = 0.0 + k[2775]*y[IDX_eM];
    data[6545] = 0.0 + k[1072]*y[IDX_HI] + k[1538]*y[IDX_HII] +
        k[1771]*y[IDX_CHII] + k[2438]*y[IDX_pH2II];
    data[6546] = 0.0 + k[1358]*y[IDX_pH3II] + k[1363]*y[IDX_pH2DII] +
        k[2355]*y[IDX_pH2II];
    data[6547] = 0.0 - k[468]*y[IDX_pH2I] - k[474]*y[IDX_pH2I];
    data[6548] = 0.0 + k[1373]*y[IDX_pH3II] + k[1378]*y[IDX_pH2DII] +
        k[1388]*y[IDX_pH3II] + k[1393]*y[IDX_pH2DII] + k[2360]*y[IDX_pH2II];
    data[6549] = 0.0 - k[456]*y[IDX_pH2I] - k[462]*y[IDX_pH2I];
    data[6550] = 0.0 + k[2538]*y[IDX_pH2II];
    data[6551] = 0.0 + k[1074]*y[IDX_CH2I] + k[1087]*y[IDX_H2OI] +
        k[2098]*y[IDX_pH2II] + k[3383]*y[IDX_H3OII] + k[3386]*y[IDX_H2DOII];
    data[6552] = 0.0 + k[1533]*y[IDX_CH2I] - k[2882]*y[IDX_pH2I];
    data[6553] = 0.0 + k[484]*y[IDX_H2OI] + k[830]*y[IDX_oH3II] +
        k[832]*y[IDX_pH3II] + k[840]*y[IDX_oH2DII] + k[842]*y[IDX_pH2DII] +
        k[843]*y[IDX_pH2DII] + k[2152]*y[IDX_pH2II] + k[3056]*y[IDX_H3OII] +
        k[3059]*y[IDX_H3OII] + k[3068]*y[IDX_H2DOII] + k[3089]*y[IDX_H3OII] +
        k[3095]*y[IDX_H2DOII];
    data[6554] = 0.0 + k[2448]*y[IDX_pH2II] + k[3239]*y[IDX_pH3II] +
        k[3248]*y[IDX_pH2DII];
    data[6555] = 0.0 - k[3127]*y[IDX_pH2I] - k[3129]*y[IDX_pH2I];
    data[6556] = 0.0 - k[3407]*y[IDX_pH2I] - k[3409]*y[IDX_pH2I] -
        k[3411]*y[IDX_pH2I];
    data[6557] = 0.0 + k[2108]*y[IDX_pH2II];
    data[6558] = 0.0 + k[2463]*y[IDX_pH2II];
    data[6559] = 0.0 - k[1595]*y[IDX_pH2I] + k[1595]*y[IDX_pH2I] +
        k[1596]*y[IDX_oH2I] - k[1598]*y[IDX_pH2I];
    data[6560] = 0.0 + k[178]*y[IDX_pH3II] + k[183]*y[IDX_pH2DII];
    data[6561] = 0.0 + k[164]*y[IDX_HI] + k[164]*y[IDX_HI] +
        k[1056]*y[IDX_HCOI] + k[1064]*y[IDX_CHI] + k[1068]*y[IDX_CH2I] +
        k[1072]*y[IDX_CHDI] + k[1078]*y[IDX_OHI] + k[1082]*y[IDX_H2OI] +
        k[1085]*y[IDX_HDOI] + k[1092]*y[IDX_HCNI] + k[1112]*y[IDX_O2HI] +
        k[1713]*y[IDX_CHII] + k[2093]*y[IDX_pH2II] + k[2689]*y[IDX_HM] +
        k[3021]*y[IDX_H3OII] + k[3372]*y[IDX_H2DOII] + k[3375]*y[IDX_HD2OII];
    data[6562] = 0.0 + k[1526]*y[IDX_C2HI] + k[1532]*y[IDX_CH2I] +
        k[1538]*y[IDX_CHDI] + k[1544]*y[IDX_HCOI] + k[1685]*y[IDX_HNOI] +
        k[2874]*y[IDX_oH2I] - k[2875]*y[IDX_pH2I] + k[2884]*y[IDX_HDI];
    data[6563] = 0.0 + k[483]*y[IDX_H2OI] + k[489]*y[IDX_HDOI] +
        k[493]*y[IDX_HCNI] + k[827]*y[IDX_oH3II] + k[828]*y[IDX_pH3II] +
        k[828]*y[IDX_pH3II] + k[829]*y[IDX_pH3II] + k[836]*y[IDX_oH2DII] +
        k[838]*y[IDX_pH2DII] + k[847]*y[IDX_oD2HII] + k[849]*y[IDX_pD2HII] +
        k[850]*y[IDX_pD2HII] + k[858]*y[IDX_HCOII] + k[2150]*y[IDX_pH2II] +
        k[2689]*y[IDX_HI] + k[2992]*y[IDX_H3OII] + k[2993]*y[IDX_H3OII] +
        k[3061]*y[IDX_H2DOII] + k[3064]*y[IDX_H2DOII] + k[3075]*y[IDX_HD2OII] +
        k[3092]*y[IDX_H2DOII] + k[3100]*y[IDX_HD2OII];
    data[6564] = 0.0 + k[3061]*y[IDX_HM] + k[3064]*y[IDX_HM] +
        k[3068]*y[IDX_DM] + k[3092]*y[IDX_HM] + k[3095]*y[IDX_DM] +
        k[3283]*y[IDX_CI] + k[3372]*y[IDX_HI] + k[3386]*y[IDX_DI] -
        k[3396]*y[IDX_pH2I] - k[3398]*y[IDX_pH2I] + k[3450]*y[IDX_eM] +
        k[3459]*y[IDX_eM];
    data[6565] = 0.0 + k[483]*y[IDX_HM] + k[484]*y[IDX_DM] +
        k[1082]*y[IDX_HI] + k[1087]*y[IDX_DI] + k[1777]*y[IDX_CHII] +
        k[1778]*y[IDX_CDII] + k[1890]*y[IDX_NHII] + k[1891]*y[IDX_NDII] +
        k[2443]*y[IDX_pH2II] + k[3004]*y[IDX_oH3II] + k[3006]*y[IDX_pH3II] +
        k[3177]*y[IDX_oH2DII] + k[3179]*y[IDX_pH2DII] + k[3183]*y[IDX_oD2HII] +
        k[3185]*y[IDX_pD2HII] + k[3193]*y[IDX_oD3II] + k[3194]*y[IDX_pD3II] +
        k[3196]*y[IDX_mD3II];
    data[6566] = 0.0 + k[995]*y[IDX_OI] + k[2788]*y[IDX_eM] -
        k[2997]*y[IDX_pH2I];
    data[6567] = 0.0 + k[2992]*y[IDX_HM] + k[2993]*y[IDX_HM] +
        k[3010]*y[IDX_CI] + k[3021]*y[IDX_HI] - k[3024]*y[IDX_pH2I] +
        k[3027]*y[IDX_eM] + k[3030]*y[IDX_eM] + k[3056]*y[IDX_DM] +
        k[3059]*y[IDX_DM] + k[3089]*y[IDX_DM] + k[3383]*y[IDX_DI];
    data[6568] = 0.0 + k[493]*y[IDX_HM] + k[1092]*y[IDX_HI] +
        k[1787]*y[IDX_CHII] + k[2103]*y[IDX_pH2II];
    data[6569] = 0.0 + k[1056]*y[IDX_HI] + k[1544]*y[IDX_HII] +
        k[2458]*y[IDX_pH2II];
    data[6570] = 0.0 + k[858]*y[IDX_HM];
    data[6571] = 0.0 + k[11]*y[IDX_pH3II] + k[13]*y[IDX_pH3II] +
        k[17]*y[IDX_oH3II] + k[48]*y[IDX_pH2DII] + k[50]*y[IDX_pH2DII] +
        k[57]*y[IDX_oH2DII] + k[59]*y[IDX_oH2DII] + k[101]*y[IDX_pD2HII] +
        k[108]*y[IDX_oD2HII] + k[110]*y[IDX_oD2HII] + k[1608]*y[IDX_HOCII] +
        k[2884]*y[IDX_HII] + k[2900]*y[IDX_pD2HII];
    data[6572] = 0.0 - k[693]*y[IDX_pH2I] - k[694]*y[IDX_pH2I] -
        k[697]*y[IDX_pH2I] - k[2886]*y[IDX_pH2I];
    data[6573] = 0.0 + k[3075]*y[IDX_HM] + k[3100]*y[IDX_HM] +
        k[3375]*y[IDX_HI] - k[3400]*y[IDX_pH2I] - k[3402]*y[IDX_pH2I] -
        k[3404]*y[IDX_pH2I];
    data[6574] = 0.0 + k[489]*y[IDX_HM] + k[1085]*y[IDX_HI] +
        k[1783]*y[IDX_CHII] + k[1896]*y[IDX_NHII] + k[2453]*y[IDX_pH2II] +
        k[3204]*y[IDX_oH3II] + k[3206]*y[IDX_pH3II] + k[3210]*y[IDX_pH2DII] +
        k[3212]*y[IDX_oH2DII] + k[3220]*y[IDX_pD2HII] + k[3222]*y[IDX_oD2HII];
    data[6575] = 0.0 - k[3123]*y[IDX_pH2I] - k[3125]*y[IDX_pH2I];
    data[6576] = 0.0 + k[1685]*y[IDX_HII];
    data[6577] = 0.0 - k[1592]*y[IDX_pH2I] + k[1592]*y[IDX_pH2I] +
        k[1593]*y[IDX_oH2I] + k[1608]*y[IDX_HDI];
    data[6578] = 0.0 - k[868]*y[IDX_pH2I] + k[892]*y[IDX_CH2I] +
        k[939]*y[IDX_NH2I] - k[2471]*y[IDX_pH2I];
    data[6579] = 0.0 - k[814]*y[IDX_pH2I];
    data[6580] = 0.0 - k[2924]*y[IDX_pH2I];
    data[6581] = 0.0 - k[1174]*y[IDX_pH2I];
    data[6582] = 0.0 - k[1636]*y[IDX_pH2I];
    data[6583] = 0.0 + k[1403]*y[IDX_pH3II] + k[1408]*y[IDX_pH2DII];
    data[6584] = 0.0 - k[1806]*y[IDX_pH2I];
    data[6585] = 0.0 - k[1182]*y[IDX_pH2I] - k[1184]*y[IDX_pH2I] +
        k[1433]*y[IDX_pH3II] + k[1440]*y[IDX_pH2DII] + k[2393]*y[IDX_pH2II];
    data[6586] = 0.0 - k[1841]*y[IDX_pH2I] - k[1856]*y[IDX_pH2I] -
        k[1858]*y[IDX_pH2I] + k[1891]*y[IDX_H2OI];
    data[6587] = 0.0 + k[2118]*y[IDX_pH2II];
    data[6588] = 0.0 - k[1180]*y[IDX_pH2I] + k[1418]*y[IDX_pH3II] +
        k[1423]*y[IDX_pH2DII] + k[1747]*y[IDX_CHII] + k[2388]*y[IDX_pH2II];
    data[6589] = 0.0 - k[1854]*y[IDX_pH2I] + k[1890]*y[IDX_H2OI] +
        k[1896]*y[IDX_HDOI] - k[2926]*y[IDX_pH2I];
    data[6590] = 0.0 + k[939]*y[IDX_HeII] + k[1795]*y[IDX_CHII] +
        k[1796]*y[IDX_CDII] + k[2113]*y[IDX_pH2II];
    data[6591] = 0.0 + k[1801]*y[IDX_CHII] + k[2123]*y[IDX_pH2II];
    data[6592] = 0.0 + k[1450]*y[IDX_pH3II] + k[1455]*y[IDX_pH2DII] +
        k[2398]*y[IDX_pH2II];
    data[6593] = 0.0 + k[1512]*y[IDX_pH3II] + k[1517]*y[IDX_pH2DII];
    data[6594] = 0.0 - k[1997]*y[IDX_pH2I];
    data[6595] = 0.0 + k[512]*y[IDX_CH2I] + k[995]*y[IDX_H2OII] -
        k[1152]*y[IDX_pH2I] + k[1284]*y[IDX_pH3II] + k[1289]*y[IDX_pH2DII];
    data[6596] = 0.0 - k[1656]*y[IDX_pH2I];
    data[6597] = 0.0 - k[497]*y[IDX_pH2I] - k[2723]*y[IDX_pH2I];
    data[6598] = 0.0 + k[1465]*y[IDX_pH3II] + k[1470]*y[IDX_pH2DII] +
        k[2403]*y[IDX_pH2II];
    data[6599] = 0.0 - k[2019]*y[IDX_pH2I];
    data[6600] = 0.0 + k[1112]*y[IDX_HI];
    data[6601] = 0.0 - k[2928]*y[IDX_pH2I];
    data[6602] = 0.0 - k[1160]*y[IDX_pH2I] - k[1162]*y[IDX_pH2I] +
        k[1495]*y[IDX_pH3II] + k[1502]*y[IDX_pH2DII] + k[2413]*y[IDX_pH2II];
    data[6603] = 0.0 - k[1939]*y[IDX_pH2I] - k[1941]*y[IDX_pH2I];
    data[6604] = 0.0 + k[1078]*y[IDX_HI] - k[1158]*y[IDX_pH2I] +
        k[1480]*y[IDX_pH3II] + k[1485]*y[IDX_pH2DII] + k[1757]*y[IDX_CHII] +
        k[2408]*y[IDX_pH2II];
    data[6605] = 0.0 - k[1937]*y[IDX_pH2I];
    data[6606] = 0.0 + k[2750]*y[IDX_pH2II] + k[2775]*y[IDX_CH2II] +
        k[2788]*y[IDX_H2OII] + k[2808]*y[IDX_pH3II] + k[2815]*y[IDX_pH2DII] +
        k[3027]*y[IDX_H3OII] + k[3030]*y[IDX_H3OII] + k[3450]*y[IDX_H2DOII] +
        k[3459]*y[IDX_H2DOII];
    data[6607] = 0.0 - k[112]*y[IDX_pH2I] - k[113]*y[IDX_pH2I] +
        k[3196]*y[IDX_H2OI];
    data[6608] = 0.0 + k[39]*y[IDX_pH3II] + k[85]*y[IDX_pH2DII] +
        k[86]*y[IDX_pH2DII];
    data[6609] = 0.0 - k[686]*y[IDX_pH2I] - k[690]*y[IDX_pH2I];
    data[6610] = 0.0 - k[69]*y[IDX_pH2I] - k[70]*y[IDX_pH2I] -
        k[71]*y[IDX_pH2I] - k[72]*y[IDX_pH2I] + k[77]*y[IDX_oH2I] +
        k[108]*y[IDX_HDI] + k[110]*y[IDX_HDI] + k[847]*y[IDX_HM] +
        k[3183]*y[IDX_H2OI] + k[3222]*y[IDX_HDOI];
    data[6611] = 0.0 - k[116]*y[IDX_pH2I] - k[117]*y[IDX_pH2I] -
        k[118]*y[IDX_pH2I] - k[119]*y[IDX_pH2I] + k[3193]*y[IDX_H2OI];
    data[6612] = 0.0 + k[2]*y[IDX_pH3II] + k[3]*y[IDX_pH3II] +
        k[7]*y[IDX_oH3II] + k[9]*y[IDX_oH3II] + k[23]*y[IDX_pH2DII] +
        k[30]*y[IDX_oH2DII] + k[32]*y[IDX_oH2DII] + k[68]*y[IDX_pD2HII] +
        k[77]*y[IDX_oD2HII] + k[1593]*y[IDX_HOCII] + k[1596]*y[IDX_DOCII] +
        k[2874]*y[IDX_HII];
    data[6613] = 0.0 - k[682]*y[IDX_pH2I] - k[2878]*y[IDX_pH2I];
    data[6614] = 0.0 - k[25]*y[IDX_pH2I] - k[26]*y[IDX_pH2I] -
        k[27]*y[IDX_pH2I] + k[30]*y[IDX_oH2I] + k[32]*y[IDX_oH2I] +
        k[57]*y[IDX_HDI] + k[59]*y[IDX_HDI] + k[312] + k[836]*y[IDX_HM] +
        k[840]*y[IDX_DM] + k[3177]*y[IDX_H2OI] + k[3212]*y[IDX_HDOI];
    data[6615] = 0.0 - k[5]*y[IDX_pH2I] - k[6]*y[IDX_pH2I] +
        k[7]*y[IDX_oH2I] + k[9]*y[IDX_oH2I] + k[17]*y[IDX_HDI] + k[304] +
        k[827]*y[IDX_HM] + k[830]*y[IDX_DM] + k[3004]*y[IDX_H2OI] +
        k[3204]*y[IDX_HDOI];
    data[6616] = 0.0 + k[35]*y[IDX_pH3II] + k[81]*y[IDX_pH2DII] +
        k[2904]*y[IDX_pH2DII];
    data[6617] = 0.0 - k[685]*y[IDX_pH2I] - k[689]*y[IDX_pH2I];
    data[6618] = 0.0 - k[61]*y[IDX_pH2I] - k[62]*y[IDX_pH2I] -
        k[63]*y[IDX_pH2I] - k[64]*y[IDX_pH2I] + k[68]*y[IDX_oH2I] +
        k[101]*y[IDX_HDI] + k[849]*y[IDX_HM] + k[850]*y[IDX_HM] +
        k[2900]*y[IDX_HDI] + k[3185]*y[IDX_H2OI] + k[3220]*y[IDX_HDOI];
    data[6619] = 0.0 - k[2936]*y[IDX_pH2I] - k[2937]*y[IDX_pH2I] +
        k[3194]*y[IDX_H2OI];
    data[6620] = 0.0 - k[0]*y[IDX_pH3II] - k[1]*y[IDX_pH3II] -
        k[5]*y[IDX_oH3II] - k[6]*y[IDX_oH3II] - k[19]*y[IDX_pH2DII] -
        k[20]*y[IDX_pH2DII] - k[25]*y[IDX_oH2DII] - k[26]*y[IDX_oH2DII] -
        k[27]*y[IDX_oH2DII] - k[61]*y[IDX_pD2HII] - k[62]*y[IDX_pD2HII] -
        k[63]*y[IDX_pD2HII] - k[64]*y[IDX_pD2HII] - k[69]*y[IDX_oD2HII] -
        k[70]*y[IDX_oD2HII] - k[71]*y[IDX_oD2HII] - k[72]*y[IDX_oD2HII] -
        k[112]*y[IDX_mD3II] - k[113]*y[IDX_mD3II] - k[116]*y[IDX_oD3II] -
        k[117]*y[IDX_oD3II] - k[118]*y[IDX_oD3II] - k[119]*y[IDX_oD3II] - k[208]
        - k[213] - k[219] - k[225] - k[360] - k[456]*y[IDX_COII] -
        k[462]*y[IDX_COII] - k[468]*y[IDX_CNII] - k[474]*y[IDX_CNII] -
        k[497]*y[IDX_OM] - k[682]*y[IDX_oH2II] - k[685]*y[IDX_pD2II] -
        k[686]*y[IDX_oD2II] - k[689]*y[IDX_pD2II] - k[690]*y[IDX_oD2II] -
        k[693]*y[IDX_HDII] - k[694]*y[IDX_HDII] - k[697]*y[IDX_HDII] -
        k[814]*y[IDX_HeDII] - k[868]*y[IDX_HeII] - k[1130]*y[IDX_CI] -
        k[1136]*y[IDX_CHI] - k[1144]*y[IDX_CDI] - k[1146]*y[IDX_CDI] -
        k[1152]*y[IDX_OI] - k[1158]*y[IDX_OHI] - k[1160]*y[IDX_ODI] -
        k[1162]*y[IDX_ODI] - k[1174]*y[IDX_NI] - k[1180]*y[IDX_NHI] -
        k[1182]*y[IDX_NDI] - k[1184]*y[IDX_NDI] - k[1592]*y[IDX_HOCII] +
        k[1592]*y[IDX_HOCII] - k[1595]*y[IDX_DOCII] + k[1595]*y[IDX_DOCII] -
        k[1598]*y[IDX_DOCII] - k[1636]*y[IDX_NII] - k[1656]*y[IDX_OII] -
        k[1694]*y[IDX_C2II] - k[1731]*y[IDX_CHII] - k[1733]*y[IDX_CDII] -
        k[1735]*y[IDX_CDII] - k[1806]*y[IDX_N2II] - k[1841]*y[IDX_NDII] -
        k[1854]*y[IDX_NHII] - k[1856]*y[IDX_NDII] - k[1858]*y[IDX_NDII] -
        k[1937]*y[IDX_OHII] - k[1939]*y[IDX_ODII] - k[1941]*y[IDX_ODII] -
        k[1997]*y[IDX_NO2II] - k[2019]*y[IDX_O2DII] - k[2471]*y[IDX_HeII] -
        k[2641]*y[IDX_CII] - k[2657]*y[IDX_CI] - k[2677]*y[IDX_CM] -
        k[2723]*y[IDX_OM] - k[2875]*y[IDX_HII] - k[2878]*y[IDX_oH2II] -
        k[2879]*y[IDX_pH2II] - k[2882]*y[IDX_DII] - k[2886]*y[IDX_HDII] -
        k[2924]*y[IDX_HeHII] - k[2926]*y[IDX_NHII] - k[2928]*y[IDX_O2HII] -
        k[2936]*y[IDX_pD3II] - k[2937]*y[IDX_pD3II] - k[2997]*y[IDX_H2OII] -
        k[3024]*y[IDX_H3OII] - k[3123]*y[IDX_HDOII] - k[3125]*y[IDX_HDOII] -
        k[3127]*y[IDX_D2OII] - k[3129]*y[IDX_D2OII] - k[3396]*y[IDX_H2DOII] -
        k[3398]*y[IDX_H2DOII] - k[3400]*y[IDX_HD2OII] - k[3402]*y[IDX_HD2OII] -
        k[3404]*y[IDX_HD2OII] - k[3407]*y[IDX_D3OII] - k[3409]*y[IDX_D3OII] -
        k[3411]*y[IDX_D3OII];
    data[6621] = 0.0 + k[2093]*y[IDX_HI] + k[2098]*y[IDX_DI] +
        k[2103]*y[IDX_HCNI] + k[2108]*y[IDX_DCNI] + k[2113]*y[IDX_NH2I] +
        k[2118]*y[IDX_ND2I] + k[2123]*y[IDX_NHDI] + k[2150]*y[IDX_HM] +
        k[2152]*y[IDX_DM] + k[2340]*y[IDX_C2I] + k[2345]*y[IDX_CHI] +
        k[2350]*y[IDX_CDI] + k[2355]*y[IDX_CNI] + k[2360]*y[IDX_COI] +
        k[2388]*y[IDX_NHI] + k[2393]*y[IDX_NDI] + k[2398]*y[IDX_NOI] +
        k[2403]*y[IDX_O2I] + k[2408]*y[IDX_OHI] + k[2413]*y[IDX_ODI] +
        k[2418]*y[IDX_C2HI] + k[2423]*y[IDX_C2DI] + k[2428]*y[IDX_CH2I] +
        k[2433]*y[IDX_CD2I] + k[2438]*y[IDX_CHDI] + k[2443]*y[IDX_H2OI] +
        k[2448]*y[IDX_D2OI] + k[2453]*y[IDX_HDOI] + k[2458]*y[IDX_HCOI] +
        k[2463]*y[IDX_DCOI] + k[2538]*y[IDX_CO2I] + k[2750]*y[IDX_eM] -
        k[2879]*y[IDX_pH2I];
    data[6622] = 0.0 - k[19]*y[IDX_pH2I] - k[20]*y[IDX_pH2I] +
        k[23]*y[IDX_oH2I] + k[48]*y[IDX_HDI] + k[50]*y[IDX_HDI] +
        k[81]*y[IDX_pD2I] + k[85]*y[IDX_oD2I] + k[86]*y[IDX_oD2I] +
        k[183]*y[IDX_GRAINM] + k[314] + k[838]*y[IDX_HM] + k[842]*y[IDX_DM] +
        k[843]*y[IDX_DM] + k[1262]*y[IDX_CI] + k[1289]*y[IDX_OI] +
        k[1316]*y[IDX_C2I] + k[1331]*y[IDX_CHI] + k[1348]*y[IDX_CDI] +
        k[1363]*y[IDX_CNI] + k[1378]*y[IDX_COI] + k[1393]*y[IDX_COI] +
        k[1408]*y[IDX_N2I] + k[1423]*y[IDX_NHI] + k[1440]*y[IDX_NDI] +
        k[1455]*y[IDX_NOI] + k[1470]*y[IDX_O2I] + k[1485]*y[IDX_OHI] +
        k[1502]*y[IDX_ODI] + k[1517]*y[IDX_NO2I] + k[2815]*y[IDX_eM] +
        k[2904]*y[IDX_pD2I] + k[3179]*y[IDX_H2OI] + k[3210]*y[IDX_HDOI] +
        k[3248]*y[IDX_D2OI];
    data[6623] = 0.0 - k[0]*y[IDX_pH2I] - k[1]*y[IDX_pH2I] +
        k[2]*y[IDX_oH2I] + k[3]*y[IDX_oH2I] + k[11]*y[IDX_HDI] +
        k[13]*y[IDX_HDI] + k[35]*y[IDX_pD2I] + k[39]*y[IDX_oD2I] +
        k[178]*y[IDX_GRAINM] + k[306] + k[828]*y[IDX_HM] + k[828]*y[IDX_HM] +
        k[829]*y[IDX_HM] + k[832]*y[IDX_DM] + k[1257]*y[IDX_CI] +
        k[1284]*y[IDX_OI] + k[1311]*y[IDX_C2I] + k[1326]*y[IDX_CHI] +
        k[1341]*y[IDX_CDI] + k[1358]*y[IDX_CNI] + k[1373]*y[IDX_COI] +
        k[1388]*y[IDX_COI] + k[1403]*y[IDX_N2I] + k[1418]*y[IDX_NHI] +
        k[1433]*y[IDX_NDI] + k[1450]*y[IDX_NOI] + k[1465]*y[IDX_O2I] +
        k[1480]*y[IDX_OHI] + k[1495]*y[IDX_ODI] + k[1512]*y[IDX_NO2I] +
        k[2808]*y[IDX_eM] + k[3006]*y[IDX_H2OI] + k[3206]*y[IDX_HDOI] +
        k[3239]*y[IDX_D2OI];
    data[6624] = 0.0 - k[630]*y[IDX_pH2II];
    data[6625] = 0.0 - k[648]*y[IDX_pH2II] - k[2340]*y[IDX_pH2II];
    data[6626] = 0.0 - k[2423]*y[IDX_pH2II];
    data[6627] = 0.0 - k[2418]*y[IDX_pH2II];
    data[6628] = 0.0 - k[662]*y[IDX_pH2II] - k[664]*y[IDX_pH2II] -
        k[2350]*y[IDX_pH2II];
    data[6629] = 0.0 - k[2433]*y[IDX_pH2II];
    data[6630] = 0.0 - k[654]*y[IDX_pH2II] - k[2345]*y[IDX_pH2II];
    data[6631] = 0.0 - k[2428]*y[IDX_pH2II];
    data[6632] = 0.0 - k[2438]*y[IDX_pH2II];
    data[6633] = 0.0 - k[670]*y[IDX_pH2II] - k[2355]*y[IDX_pH2II];
    data[6634] = 0.0 - k[676]*y[IDX_pH2II] - k[2360]*y[IDX_pH2II];
    data[6635] = 0.0 - k[790]*y[IDX_pH2II] - k[2538]*y[IDX_pH2II];
    data[6636] = 0.0 - k[2098]*y[IDX_pH2II] - k[2896]*y[IDX_pH2II] -
        k[2899]*y[IDX_pH2II];
    data[6637] = 0.0 - k[2152]*y[IDX_pH2II];
    data[6638] = 0.0 - k[2448]*y[IDX_pH2II] - k[3047]*y[IDX_pH2II] -
        k[3049]*y[IDX_pH2II];
    data[6639] = 0.0 - k[2108]*y[IDX_pH2II];
    data[6640] = 0.0 - k[801]*y[IDX_pH2II] - k[2463]*y[IDX_pH2II];
    data[6641] = 0.0 + k[809]*y[IDX_HeHII] - k[2093]*y[IDX_pH2II] +
        k[2646]*y[IDX_HII] + k[2898]*y[IDX_HDII];
    data[6642] = 0.0 + k[1548]*y[IDX_HCOI] + k[2646]*y[IDX_HI];
    data[6643] = 0.0 - k[2150]*y[IDX_pH2II];
    data[6644] = 0.0 - k[2443]*y[IDX_pH2II] - k[2990]*y[IDX_pH2II];
    data[6645] = 0.0 - k[2103]*y[IDX_pH2II];
    data[6646] = 0.0 + k[1548]*y[IDX_HII] - k[2458]*y[IDX_pH2II] -
        k[2922]*y[IDX_pH2II];
    data[6647] = 0.0 - k[726]*y[IDX_pH2II] - k[727]*y[IDX_pH2II] -
        k[2888]*y[IDX_pH2II];
    data[6648] = 0.0 + k[2898]*y[IDX_HI];
    data[6649] = 0.0 - k[2453]*y[IDX_pH2II] - k[3037]*y[IDX_pH2II] -
        k[3039]*y[IDX_pH2II];
    data[6650] = 0.0 + k[2471]*y[IDX_pH2I];
    data[6651] = 0.0 + k[809]*y[IDX_HI];
    data[6652] = 0.0 - k[636]*y[IDX_pH2II];
    data[6653] = 0.0 - k[740]*y[IDX_pH2II];
    data[6654] = 0.0 - k[754]*y[IDX_pH2II] - k[756]*y[IDX_pH2II] -
        k[2393]*y[IDX_pH2II];
    data[6655] = 0.0 - k[2118]*y[IDX_pH2II];
    data[6656] = 0.0 - k[746]*y[IDX_pH2II] - k[2388]*y[IDX_pH2II];
    data[6657] = 0.0 - k[2113]*y[IDX_pH2II];
    data[6658] = 0.0 - k[2123]*y[IDX_pH2II];
    data[6659] = 0.0 - k[762]*y[IDX_pH2II] - k[2398]*y[IDX_pH2II];
    data[6660] = 0.0 - k[642]*y[IDX_pH2II];
    data[6661] = 0.0 - k[768]*y[IDX_pH2II] - k[2403]*y[IDX_pH2II];
    data[6662] = 0.0 - k[782]*y[IDX_pH2II] - k[784]*y[IDX_pH2II] -
        k[2413]*y[IDX_pH2II];
    data[6663] = 0.0 - k[774]*y[IDX_pH2II] - k[2408]*y[IDX_pH2II];
    data[6664] = 0.0 - k[2745]*y[IDX_pH2II] - k[2750]*y[IDX_pH2II];
    data[6665] = 0.0 - k[702]*y[IDX_pH2II] - k[703]*y[IDX_pH2II] -
        k[708]*y[IDX_pH2II];
    data[6666] = 0.0 - k[683]*y[IDX_pH2II] - k[2877]*y[IDX_pH2II];
    data[6667] = 0.0 + k[334];
    data[6668] = 0.0 + k[324];
    data[6669] = 0.0 - k[699]*y[IDX_pH2II] - k[706]*y[IDX_pH2II];
    data[6670] = 0.0 + k[225] + k[2471]*y[IDX_HeII] - k[2879]*y[IDX_pH2II];
    data[6671] = 0.0 - k[290] - k[630]*y[IDX_CI] - k[636]*y[IDX_NI] -
        k[642]*y[IDX_OI] - k[648]*y[IDX_C2I] - k[654]*y[IDX_CHI] -
        k[662]*y[IDX_CDI] - k[664]*y[IDX_CDI] - k[670]*y[IDX_CNI] -
        k[676]*y[IDX_COI] - k[683]*y[IDX_oH2I] - k[699]*y[IDX_pD2I] -
        k[702]*y[IDX_oD2I] - k[703]*y[IDX_oD2I] - k[706]*y[IDX_pD2I] -
        k[708]*y[IDX_oD2I] - k[726]*y[IDX_HDI] - k[727]*y[IDX_HDI] -
        k[740]*y[IDX_N2I] - k[746]*y[IDX_NHI] - k[754]*y[IDX_NDI] -
        k[756]*y[IDX_NDI] - k[762]*y[IDX_NOI] - k[768]*y[IDX_O2I] -
        k[774]*y[IDX_OHI] - k[782]*y[IDX_ODI] - k[784]*y[IDX_ODI] -
        k[790]*y[IDX_CO2I] - k[801]*y[IDX_DCOI] - k[2093]*y[IDX_HI] -
        k[2098]*y[IDX_DI] - k[2103]*y[IDX_HCNI] - k[2108]*y[IDX_DCNI] -
        k[2113]*y[IDX_NH2I] - k[2118]*y[IDX_ND2I] - k[2123]*y[IDX_NHDI] -
        k[2150]*y[IDX_HM] - k[2152]*y[IDX_DM] - k[2340]*y[IDX_C2I] -
        k[2345]*y[IDX_CHI] - k[2350]*y[IDX_CDI] - k[2355]*y[IDX_CNI] -
        k[2360]*y[IDX_COI] - k[2388]*y[IDX_NHI] - k[2393]*y[IDX_NDI] -
        k[2398]*y[IDX_NOI] - k[2403]*y[IDX_O2I] - k[2408]*y[IDX_OHI] -
        k[2413]*y[IDX_ODI] - k[2418]*y[IDX_C2HI] - k[2423]*y[IDX_C2DI] -
        k[2428]*y[IDX_CH2I] - k[2433]*y[IDX_CD2I] - k[2438]*y[IDX_CHDI] -
        k[2443]*y[IDX_H2OI] - k[2448]*y[IDX_D2OI] - k[2453]*y[IDX_HDOI] -
        k[2458]*y[IDX_HCOI] - k[2463]*y[IDX_DCOI] - k[2538]*y[IDX_CO2I] -
        k[2745]*y[IDX_eM] - k[2750]*y[IDX_eM] - k[2877]*y[IDX_oH2I] -
        k[2879]*y[IDX_pH2I] - k[2888]*y[IDX_HDI] - k[2896]*y[IDX_DI] -
        k[2899]*y[IDX_DI] - k[2922]*y[IDX_HCOI] - k[2990]*y[IDX_H2OI] -
        k[3037]*y[IDX_HDOI] - k[3039]*y[IDX_HDOI] - k[3047]*y[IDX_D2OI] -
        k[3049]*y[IDX_D2OI];
    data[6672] = 0.0 + k[336];
    data[6673] = 0.0 + k[326];
    data[6674] = 0.0 - k[1262]*y[IDX_pH2DII] - k[1263]*y[IDX_pH2DII] -
        k[1265]*y[IDX_pH2DII];
    data[6675] = 0.0 - k[1316]*y[IDX_pH2DII] - k[1317]*y[IDX_pH2DII] -
        k[1319]*y[IDX_pH2DII];
    data[6676] = 0.0 - k[1348]*y[IDX_pH2DII] - k[1349]*y[IDX_pH2DII] -
        k[1351]*y[IDX_pH2DII];
    data[6677] = 0.0 - k[1331]*y[IDX_pH2DII] - k[1332]*y[IDX_pH2DII] -
        k[1334]*y[IDX_pH2DII];
    data[6678] = 0.0 - k[1363]*y[IDX_pH2DII] - k[1364]*y[IDX_pH2DII] -
        k[1366]*y[IDX_pH2DII];
    data[6679] = 0.0 - k[1378]*y[IDX_pH2DII] - k[1379]*y[IDX_pH2DII] -
        k[1381]*y[IDX_pH2DII] - k[1393]*y[IDX_pH2DII] - k[1394]*y[IDX_pH2DII] -
        k[1396]*y[IDX_pH2DII];
    data[6680] = 0.0 - k[2855]*y[IDX_pH2DII] - k[2856]*y[IDX_pH2DII] +
        k[2890]*y[IDX_pH3II] + k[2896]*y[IDX_pH2II];
    data[6681] = 0.0 - k[842]*y[IDX_pH2DII] - k[843]*y[IDX_pH2DII] -
        k[844]*y[IDX_pH2DII] - k[846]*y[IDX_pH2DII];
    data[6682] = 0.0 - k[3248]*y[IDX_pH2DII] - k[3250]*y[IDX_pH2DII] -
        k[3252]*y[IDX_pH2DII] - k[3253]*y[IDX_pH2DII];
    data[6683] = 0.0 + k[801]*y[IDX_pH2II] + k[803]*y[IDX_oH2II];
    data[6684] = 0.0 - k[183]*y[IDX_pH2DII] - k[184]*y[IDX_pH2DII] -
        k[185]*y[IDX_pH2DII];
    data[6685] = 0.0 + k[2859]*y[IDX_pD2HII] + k[2860]*y[IDX_oD2HII] -
        k[2893]*y[IDX_pH2DII] + k[2897]*y[IDX_HDII];
    data[6686] = 0.0 - k[838]*y[IDX_pH2DII] - k[839]*y[IDX_pH2DII];
    data[6687] = 0.0 - k[3179]*y[IDX_pH2DII] - k[3180]*y[IDX_pH2DII] -
        k[3182]*y[IDX_pH2DII];
    data[6688] = 0.0 + k[800]*y[IDX_HDII];
    data[6689] = 0.0 + k[11]*y[IDX_pH3II] + k[12]*y[IDX_pH3II] +
        k[16]*y[IDX_oH3II] - k[45]*y[IDX_pH2DII] - k[46]*y[IDX_pH2DII] -
        k[47]*y[IDX_pH2DII] - k[48]*y[IDX_pH2DII] - k[49]*y[IDX_pH2DII] -
        k[50]*y[IDX_pH2DII] - k[51]*y[IDX_pH2DII] + k[56]*y[IDX_oH2DII] +
        k[96]*y[IDX_pD2HII] + k[97]*y[IDX_pD2HII] + k[103]*y[IDX_oD2HII] +
        k[104]*y[IDX_oD2HII] + k[727]*y[IDX_pH2II] + k[729]*y[IDX_oH2II] +
        k[739]*y[IDX_HDII] + k[824]*y[IDX_HeHII] + k[1851]*y[IDX_NHII] +
        k[2029]*y[IDX_O2HII];
    data[6690] = 0.0 + k[694]*y[IDX_pH2I] + k[696]*y[IDX_oH2I] +
        k[739]*y[IDX_HDI] + k[800]*y[IDX_HCOI] + k[2897]*y[IDX_HI];
    data[6691] = 0.0 - k[3210]*y[IDX_pH2DII] - k[3211]*y[IDX_pH2DII] -
        k[3214]*y[IDX_pH2DII] - k[3216]*y[IDX_pH2DII] - k[3217]*y[IDX_pH2DII];
    data[6692] = 0.0 + k[814]*y[IDX_pH2I] + k[816]*y[IDX_oH2I];
    data[6693] = 0.0 + k[824]*y[IDX_HDI];
    data[6694] = 0.0 - k[1276]*y[IDX_pH2DII] - k[1278]*y[IDX_pH2DII];
    data[6695] = 0.0 - k[1408]*y[IDX_pH2DII] - k[1409]*y[IDX_pH2DII] -
        k[1411]*y[IDX_pH2DII];
    data[6696] = 0.0 - k[1440]*y[IDX_pH2DII] - k[1441]*y[IDX_pH2DII] -
        k[1443]*y[IDX_pH2DII];
    data[6697] = 0.0 + k[1841]*y[IDX_pH2I] + k[1843]*y[IDX_oH2I];
    data[6698] = 0.0 - k[1423]*y[IDX_pH2DII] - k[1424]*y[IDX_pH2DII] -
        k[1426]*y[IDX_pH2DII];
    data[6699] = 0.0 + k[1851]*y[IDX_HDI];
    data[6700] = 0.0 - k[1455]*y[IDX_pH2DII] - k[1456]*y[IDX_pH2DII] -
        k[1458]*y[IDX_pH2DII];
    data[6701] = 0.0 - k[1517]*y[IDX_pH2DII] - k[1518]*y[IDX_pH2DII] -
        k[1520]*y[IDX_pH2DII];
    data[6702] = 0.0 - k[1289]*y[IDX_pH2DII] - k[1290]*y[IDX_pH2DII] -
        k[1292]*y[IDX_pH2DII] - k[1303]*y[IDX_pH2DII] - k[1305]*y[IDX_pH2DII];
    data[6703] = 0.0 - k[1470]*y[IDX_pH2DII] - k[1471]*y[IDX_pH2DII] -
        k[1473]*y[IDX_pH2DII];
    data[6704] = 0.0 + k[2019]*y[IDX_pH2I] + k[2021]*y[IDX_oH2I];
    data[6705] = 0.0 + k[2029]*y[IDX_HDI];
    data[6706] = 0.0 - k[1502]*y[IDX_pH2DII] - k[1503]*y[IDX_pH2DII] -
        k[1505]*y[IDX_pH2DII];
    data[6707] = 0.0 - k[1485]*y[IDX_pH2DII] - k[1486]*y[IDX_pH2DII] -
        k[1488]*y[IDX_pH2DII];
    data[6708] = 0.0 - k[2803]*y[IDX_pH2DII] - k[2813]*y[IDX_pH2DII] -
        k[2815]*y[IDX_pH2DII];
    data[6709] = 0.0 + k[112]*y[IDX_pH2I];
    data[6710] = 0.0 + k[37]*y[IDX_pH3II] - k[82]*y[IDX_pH2DII] +
        k[82]*y[IDX_pH2DII] - k[83]*y[IDX_pH2DII] - k[84]*y[IDX_pH2DII] -
        k[85]*y[IDX_pH2DII] - k[86]*y[IDX_pH2DII] + k[708]*y[IDX_pH2II];
    data[6711] = 0.0 + k[690]*y[IDX_pH2I];
    data[6712] = 0.0 + k[70]*y[IDX_pH2I] + k[75]*y[IDX_oH2I] +
        k[103]*y[IDX_HDI] + k[104]*y[IDX_HDI] + k[2860]*y[IDX_HI];
    data[6713] = 0.0 + k[116]*y[IDX_pH2I] + k[117]*y[IDX_pH2I];
    data[6714] = 0.0 - k[21]*y[IDX_pH2DII] - k[22]*y[IDX_pH2DII] -
        k[23]*y[IDX_pH2DII] - k[24]*y[IDX_pH2DII] + k[30]*y[IDX_oH2DII] +
        k[31]*y[IDX_oH2DII] + k[66]*y[IDX_pD2HII] + k[75]*y[IDX_oD2HII] +
        k[696]*y[IDX_HDII] + k[816]*y[IDX_HeDII] + k[1843]*y[IDX_NDII] +
        k[2021]*y[IDX_O2DII];
    data[6715] = 0.0 + k[729]*y[IDX_HDI] + k[803]*y[IDX_DCOI];
    data[6716] = 0.0 + k[26]*y[IDX_pH2I] + k[30]*y[IDX_oH2I] +
        k[31]*y[IDX_oH2I] + k[56]*y[IDX_HDI];
    data[6717] = 0.0 + k[16]*y[IDX_HDI];
    data[6718] = 0.0 + k[33]*y[IDX_pH3II] - k[78]*y[IDX_pH2DII] +
        k[78]*y[IDX_pH2DII] - k[79]*y[IDX_pH2DII] - k[80]*y[IDX_pH2DII] -
        k[81]*y[IDX_pH2DII] + k[706]*y[IDX_pH2II] - k[2904]*y[IDX_pH2DII];
    data[6719] = 0.0 + k[689]*y[IDX_pH2I];
    data[6720] = 0.0 + k[62]*y[IDX_pH2I] + k[66]*y[IDX_oH2I] +
        k[96]*y[IDX_HDI] + k[97]*y[IDX_HDI] + k[2859]*y[IDX_HI];
    data[6721] = 0.0 + k[2936]*y[IDX_pH2I];
    data[6722] = 0.0 - k[19]*y[IDX_pH2DII] - k[20]*y[IDX_pH2DII] +
        k[26]*y[IDX_oH2DII] + k[62]*y[IDX_pD2HII] + k[70]*y[IDX_oD2HII] +
        k[112]*y[IDX_mD3II] + k[116]*y[IDX_oD3II] + k[117]*y[IDX_oD3II] +
        k[689]*y[IDX_pD2II] + k[690]*y[IDX_oD2II] + k[694]*y[IDX_HDII] +
        k[814]*y[IDX_HeDII] + k[1841]*y[IDX_NDII] + k[2019]*y[IDX_O2DII] +
        k[2936]*y[IDX_pD3II];
    data[6723] = 0.0 + k[706]*y[IDX_pD2I] + k[708]*y[IDX_oD2I] +
        k[727]*y[IDX_HDI] + k[801]*y[IDX_DCOI] + k[2896]*y[IDX_DI];
    data[6724] = 0.0 - k[19]*y[IDX_pH2I] - k[20]*y[IDX_pH2I] -
        k[21]*y[IDX_oH2I] - k[22]*y[IDX_oH2I] - k[23]*y[IDX_oH2I] -
        k[24]*y[IDX_oH2I] - k[45]*y[IDX_HDI] - k[46]*y[IDX_HDI] -
        k[47]*y[IDX_HDI] - k[48]*y[IDX_HDI] - k[49]*y[IDX_HDI] -
        k[50]*y[IDX_HDI] - k[51]*y[IDX_HDI] - k[78]*y[IDX_pD2I] +
        k[78]*y[IDX_pD2I] - k[79]*y[IDX_pD2I] - k[80]*y[IDX_pD2I] -
        k[81]*y[IDX_pD2I] - k[82]*y[IDX_oD2I] + k[82]*y[IDX_oD2I] -
        k[83]*y[IDX_oD2I] - k[84]*y[IDX_oD2I] - k[85]*y[IDX_oD2I] -
        k[86]*y[IDX_oD2I] - k[183]*y[IDX_GRAINM] - k[184]*y[IDX_GRAINM] -
        k[185]*y[IDX_GRAINM] - k[314] - k[315] - k[317] - k[333] - k[336] -
        k[337] - k[838]*y[IDX_HM] - k[839]*y[IDX_HM] - k[842]*y[IDX_DM] -
        k[843]*y[IDX_DM] - k[844]*y[IDX_DM] - k[846]*y[IDX_DM] -
        k[1262]*y[IDX_CI] - k[1263]*y[IDX_CI] - k[1265]*y[IDX_CI] -
        k[1276]*y[IDX_NI] - k[1278]*y[IDX_NI] - k[1289]*y[IDX_OI] -
        k[1290]*y[IDX_OI] - k[1292]*y[IDX_OI] - k[1303]*y[IDX_OI] -
        k[1305]*y[IDX_OI] - k[1316]*y[IDX_C2I] - k[1317]*y[IDX_C2I] -
        k[1319]*y[IDX_C2I] - k[1331]*y[IDX_CHI] - k[1332]*y[IDX_CHI] -
        k[1334]*y[IDX_CHI] - k[1348]*y[IDX_CDI] - k[1349]*y[IDX_CDI] -
        k[1351]*y[IDX_CDI] - k[1363]*y[IDX_CNI] - k[1364]*y[IDX_CNI] -
        k[1366]*y[IDX_CNI] - k[1378]*y[IDX_COI] - k[1379]*y[IDX_COI] -
        k[1381]*y[IDX_COI] - k[1393]*y[IDX_COI] - k[1394]*y[IDX_COI] -
        k[1396]*y[IDX_COI] - k[1408]*y[IDX_N2I] - k[1409]*y[IDX_N2I] -
        k[1411]*y[IDX_N2I] - k[1423]*y[IDX_NHI] - k[1424]*y[IDX_NHI] -
        k[1426]*y[IDX_NHI] - k[1440]*y[IDX_NDI] - k[1441]*y[IDX_NDI] -
        k[1443]*y[IDX_NDI] - k[1455]*y[IDX_NOI] - k[1456]*y[IDX_NOI] -
        k[1458]*y[IDX_NOI] - k[1470]*y[IDX_O2I] - k[1471]*y[IDX_O2I] -
        k[1473]*y[IDX_O2I] - k[1485]*y[IDX_OHI] - k[1486]*y[IDX_OHI] -
        k[1488]*y[IDX_OHI] - k[1502]*y[IDX_ODI] - k[1503]*y[IDX_ODI] -
        k[1505]*y[IDX_ODI] - k[1517]*y[IDX_NO2I] - k[1518]*y[IDX_NO2I] -
        k[1520]*y[IDX_NO2I] - k[2803]*y[IDX_eM] - k[2813]*y[IDX_eM] -
        k[2815]*y[IDX_eM] - k[2855]*y[IDX_DI] - k[2856]*y[IDX_DI] -
        k[2893]*y[IDX_HI] - k[2904]*y[IDX_pD2I] - k[3179]*y[IDX_H2OI] -
        k[3180]*y[IDX_H2OI] - k[3182]*y[IDX_H2OI] - k[3210]*y[IDX_HDOI] -
        k[3211]*y[IDX_HDOI] - k[3214]*y[IDX_HDOI] - k[3216]*y[IDX_HDOI] -
        k[3217]*y[IDX_HDOI] - k[3248]*y[IDX_D2OI] - k[3250]*y[IDX_D2OI] -
        k[3252]*y[IDX_D2OI] - k[3253]*y[IDX_D2OI];
    data[6725] = 0.0 + k[11]*y[IDX_HDI] + k[12]*y[IDX_HDI] +
        k[33]*y[IDX_pD2I] + k[37]*y[IDX_oD2I] + k[2890]*y[IDX_DI];
    data[6726] = 0.0 - k[1257]*y[IDX_pH3II] - k[1258]*y[IDX_pH3II];
    data[6727] = 0.0 - k[1311]*y[IDX_pH3II] - k[1312]*y[IDX_pH3II];
    data[6728] = 0.0 - k[1341]*y[IDX_pH3II] - k[1342]*y[IDX_pH3II] -
        k[1344]*y[IDX_pH3II];
    data[6729] = 0.0 - k[1326]*y[IDX_pH3II] - k[1327]*y[IDX_pH3II];
    data[6730] = 0.0 - k[1358]*y[IDX_pH3II] - k[1359]*y[IDX_pH3II];
    data[6731] = 0.0 - k[1373]*y[IDX_pH3II] - k[1374]*y[IDX_pH3II] -
        k[1388]*y[IDX_pH3II] - k[1389]*y[IDX_pH3II];
    data[6732] = 0.0 - k[2890]*y[IDX_pH3II] - k[2891]*y[IDX_pH3II];
    data[6733] = 0.0 - k[832]*y[IDX_pH3II] - k[833]*y[IDX_pH3II];
    data[6734] = 0.0 - k[3239]*y[IDX_pH3II] - k[3240]*y[IDX_pH3II] -
        k[3242]*y[IDX_pH3II] - k[3244]*y[IDX_pH3II] - k[3246]*y[IDX_pH3II];
    data[6735] = 0.0 - k[178]*y[IDX_pH3II] - k[179]*y[IDX_pH3II] -
        k[180]*y[IDX_pH3II];
    data[6736] = 0.0 + k[2893]*y[IDX_pH2DII] + k[2895]*y[IDX_oH2DII];
    data[6737] = 0.0 - k[828]*y[IDX_pH3II] - k[829]*y[IDX_pH3II];
    data[6738] = 0.0 - k[3005]*y[IDX_pH3II] - k[3006]*y[IDX_pH3II];
    data[6739] = 0.0 + k[2921]*y[IDX_oH2II] + k[2922]*y[IDX_pH2II];
    data[6740] = 0.0 - k[10]*y[IDX_pH3II] - k[11]*y[IDX_pH3II] -
        k[12]*y[IDX_pH3II] - k[13]*y[IDX_pH3II] - k[14]*y[IDX_pH3II] +
        k[15]*y[IDX_oH3II] + k[45]*y[IDX_pH2DII] + k[46]*y[IDX_pH2DII] +
        k[52]*y[IDX_oH2DII] + k[53]*y[IDX_oH2DII] + k[2888]*y[IDX_pH2II] +
        k[2889]*y[IDX_oH2II];
    data[6741] = 0.0 + k[2886]*y[IDX_pH2I] + k[2887]*y[IDX_oH2I];
    data[6742] = 0.0 - k[3206]*y[IDX_pH3II] - k[3207]*y[IDX_pH3II] -
        k[3209]*y[IDX_pH3II];
    data[6743] = 0.0 + k[2923]*y[IDX_oH2I] + k[2924]*y[IDX_pH2I];
    data[6744] = 0.0 - k[1272]*y[IDX_pH3II];
    data[6745] = 0.0 - k[1403]*y[IDX_pH3II] - k[1404]*y[IDX_pH3II];
    data[6746] = 0.0 - k[1433]*y[IDX_pH3II] - k[1434]*y[IDX_pH3II] -
        k[1436]*y[IDX_pH3II];
    data[6747] = 0.0 - k[1418]*y[IDX_pH3II] - k[1419]*y[IDX_pH3II];
    data[6748] = 0.0 + k[2925]*y[IDX_oH2I] + k[2926]*y[IDX_pH2I];
    data[6749] = 0.0 - k[1450]*y[IDX_pH3II] - k[1451]*y[IDX_pH3II];
    data[6750] = 0.0 - k[1512]*y[IDX_pH3II] - k[1513]*y[IDX_pH3II];
    data[6751] = 0.0 - k[1284]*y[IDX_pH3II] - k[1285]*y[IDX_pH3II] -
        k[1299]*y[IDX_pH3II];
    data[6752] = 0.0 - k[1465]*y[IDX_pH3II] - k[1466]*y[IDX_pH3II];
    data[6753] = 0.0 + k[2927]*y[IDX_oH2I] + k[2928]*y[IDX_pH2I];
    data[6754] = 0.0 - k[1495]*y[IDX_pH3II] - k[1496]*y[IDX_pH3II] -
        k[1498]*y[IDX_pH3II];
    data[6755] = 0.0 - k[1480]*y[IDX_pH3II] - k[1481]*y[IDX_pH3II];
    data[6756] = 0.0 - k[2799]*y[IDX_pH3II] - k[2807]*y[IDX_pH3II] -
        k[2808]*y[IDX_pH3II];
    data[6757] = 0.0 - k[37]*y[IDX_pH3II] - k[38]*y[IDX_pH3II] -
        k[39]*y[IDX_pH3II] - k[40]*y[IDX_pH3II];
    data[6758] = 0.0 + k[69]*y[IDX_pH2I] + k[73]*y[IDX_oH2I];
    data[6759] = 0.0 - k[2]*y[IDX_pH3II] + k[2]*y[IDX_pH3II] -
        k[3]*y[IDX_pH3II] - k[4]*y[IDX_pH3II] + k[7]*y[IDX_oH3II] +
        k[8]*y[IDX_oH3II] + k[21]*y[IDX_pH2DII] + k[28]*y[IDX_oH2DII] +
        k[73]*y[IDX_oD2HII] + k[2876]*y[IDX_oH2II] + k[2877]*y[IDX_pH2II] +
        k[2887]*y[IDX_HDII] + k[2923]*y[IDX_HeHII] + k[2925]*y[IDX_NHII] +
        k[2927]*y[IDX_O2HII];
    data[6760] = 0.0 + k[2876]*y[IDX_oH2I] + k[2878]*y[IDX_pH2I] +
        k[2889]*y[IDX_HDI] + k[2921]*y[IDX_HCOI];
    data[6761] = 0.0 + k[25]*y[IDX_pH2I] + k[28]*y[IDX_oH2I] +
        k[52]*y[IDX_HDI] + k[53]*y[IDX_HDI] + k[2895]*y[IDX_HI];
    data[6762] = 0.0 + k[5]*y[IDX_pH2I] + k[7]*y[IDX_oH2I] +
        k[8]*y[IDX_oH2I] + k[15]*y[IDX_HDI];
    data[6763] = 0.0 - k[33]*y[IDX_pH3II] - k[34]*y[IDX_pH3II] -
        k[35]*y[IDX_pH3II] - k[36]*y[IDX_pH3II];
    data[6764] = 0.0 + k[61]*y[IDX_pH2I];
    data[6765] = 0.0 - k[0]*y[IDX_pH3II] + k[0]*y[IDX_pH3II] -
        k[1]*y[IDX_pH3II] + k[5]*y[IDX_oH3II] + k[19]*y[IDX_pH2DII] +
        k[25]*y[IDX_oH2DII] + k[61]*y[IDX_pD2HII] + k[69]*y[IDX_oD2HII] +
        k[2878]*y[IDX_oH2II] + k[2879]*y[IDX_pH2II] + k[2886]*y[IDX_HDII] +
        k[2924]*y[IDX_HeHII] + k[2926]*y[IDX_NHII] + k[2928]*y[IDX_O2HII];
    data[6766] = 0.0 + k[2877]*y[IDX_oH2I] + k[2879]*y[IDX_pH2I] +
        k[2888]*y[IDX_HDI] + k[2922]*y[IDX_HCOI];
    data[6767] = 0.0 + k[19]*y[IDX_pH2I] + k[21]*y[IDX_oH2I] +
        k[45]*y[IDX_HDI] + k[46]*y[IDX_HDI] + k[2893]*y[IDX_HI];
    data[6768] = 0.0 - k[0]*y[IDX_pH2I] + k[0]*y[IDX_pH2I] -
        k[1]*y[IDX_pH2I] - k[2]*y[IDX_oH2I] + k[2]*y[IDX_oH2I] -
        k[3]*y[IDX_oH2I] - k[4]*y[IDX_oH2I] - k[10]*y[IDX_HDI] -
        k[11]*y[IDX_HDI] - k[12]*y[IDX_HDI] - k[13]*y[IDX_HDI] -
        k[14]*y[IDX_HDI] - k[33]*y[IDX_pD2I] - k[34]*y[IDX_pD2I] -
        k[35]*y[IDX_pD2I] - k[36]*y[IDX_pD2I] - k[37]*y[IDX_oD2I] -
        k[38]*y[IDX_oD2I] - k[39]*y[IDX_oD2I] - k[40]*y[IDX_oD2I] -
        k[178]*y[IDX_GRAINM] - k[179]*y[IDX_GRAINM] - k[180]*y[IDX_GRAINM] -
        k[306] - k[307] - k[326] - k[327] - k[828]*y[IDX_HM] - k[829]*y[IDX_HM]
        - k[832]*y[IDX_DM] - k[833]*y[IDX_DM] - k[1257]*y[IDX_CI] -
        k[1258]*y[IDX_CI] - k[1272]*y[IDX_NI] - k[1284]*y[IDX_OI] -
        k[1285]*y[IDX_OI] - k[1299]*y[IDX_OI] - k[1311]*y[IDX_C2I] -
        k[1312]*y[IDX_C2I] - k[1326]*y[IDX_CHI] - k[1327]*y[IDX_CHI] -
        k[1341]*y[IDX_CDI] - k[1342]*y[IDX_CDI] - k[1344]*y[IDX_CDI] -
        k[1358]*y[IDX_CNI] - k[1359]*y[IDX_CNI] - k[1373]*y[IDX_COI] -
        k[1374]*y[IDX_COI] - k[1388]*y[IDX_COI] - k[1389]*y[IDX_COI] -
        k[1403]*y[IDX_N2I] - k[1404]*y[IDX_N2I] - k[1418]*y[IDX_NHI] -
        k[1419]*y[IDX_NHI] - k[1433]*y[IDX_NDI] - k[1434]*y[IDX_NDI] -
        k[1436]*y[IDX_NDI] - k[1450]*y[IDX_NOI] - k[1451]*y[IDX_NOI] -
        k[1465]*y[IDX_O2I] - k[1466]*y[IDX_O2I] - k[1480]*y[IDX_OHI] -
        k[1481]*y[IDX_OHI] - k[1495]*y[IDX_ODI] - k[1496]*y[IDX_ODI] -
        k[1498]*y[IDX_ODI] - k[1512]*y[IDX_NO2I] - k[1513]*y[IDX_NO2I] -
        k[2799]*y[IDX_eM] - k[2807]*y[IDX_eM] - k[2808]*y[IDX_eM] -
        k[2890]*y[IDX_DI] - k[2891]*y[IDX_DI] - k[3005]*y[IDX_H2OI] -
        k[3006]*y[IDX_H2OI] - k[3206]*y[IDX_HDOI] - k[3207]*y[IDX_HDOI] -
        k[3209]*y[IDX_HDOI] - k[3239]*y[IDX_D2OI] - k[3240]*y[IDX_D2OI] -
        k[3242]*y[IDX_D2OI] - k[3244]*y[IDX_D2OI] - k[3246]*y[IDX_D2OI];
    
    // clang-format on

    /* */

    return NAUNET_SUCCESS;
}