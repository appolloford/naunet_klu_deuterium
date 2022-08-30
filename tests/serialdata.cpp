// 
#include <stdio.h>

#include <stdexcept>
#include <vector>

#include "naunet.h"
#include "naunet_data.h"
#include "naunet_macros.h"
#include "naunet_ode.h"
#include "naunet_timer.h"

int main() {
    double spy       = 86400.0 * 365.0;

    //
    int nsystem               = 4096;
    double rawdata[4096][133] = {0.0};

    FILE *fab                 = fopen("testgrids.dat", "r");
    for (int i = 0; i < 4096; i++) {
        if (feof(fab)) break;

        for (int j = 0; j < 133; j++) {
            fscanf(fab, "%lf", &(rawdata[i][j]));
        }
    }
    fclose(fab);

#ifdef NAUNET_DEBUG
    printf("First row of input data\n");
    for (int i = 0; i < 133; i++) {
        printf("%13.7e ", rawdata[0][i]);
    }
    printf("\n");
#endif

    NaunetData *data = new NaunetData[nsystem];
    //
    double pi                 = 3.14159265;
    double rD                 = 1.0e-5;
    double rhoD               = 3.0;
    double DtoGM              = 7.09e-3;
    double amH                = 1.66043e-24;
    double OPRH2              = 0.1;

    for (int isys = 0; isys < nsystem; isys++) {
        data[isys].nH          = rawdata[isys][0];
        data[isys].Tgas        = rawdata[isys][1];
        // data[isys].Tgas = 15.0;
        data[isys].user_Av     = 30.0;
        data[isys].user_crflux = 2.5e-17;
        data[isys].user_GtoDN =
            (4.e0 * pi * rhoD * rD * rD * rD) / (3.e0 * DtoGM * amH);
    }

    Naunet naunet;
    if (naunet.Init() == NAUNET_FAIL) {
        printf("Initialize Fail\n");
        return 1;
    }

#ifdef USE_CUDA
    if (naunet.Reset(1) == NAUNET_FAIL) {
        throw std::runtime_error("Fail to reset the number of systems");
    }
#endif

    //
    double **y = new double *[nsystem];

    for (int isys = 0; isys < nsystem; isys++) {
        y[isys] = new double[NEQUATIONS];
    }

    for (int isys = 0; isys < nsystem; isys++) {
        for (int i = 0; i < NEQUATIONS; i++) {
            y[isys][i] = rawdata[isys][i + 2];
        }
    }

#ifdef NAUNET_DEBUG
    printf("Abundances in the first system\n");
    for (int i = 0; i < NEQUATIONS; i++) {
        printf("%13.7e ", y[0][i]);
    }
    printf("\n");
#endif

    FILE *fbin = fopen("evolution_serialdata.bin", "w");
    FILE *ftxt = fopen("evolution_serialdata.txt", "w");
    FILE *ttxt = fopen("time_serialdata.txt", "w");

#ifdef NAUNET_DEBUG
    printf("Initialization is done. Start to evolve.\n");
    // FILE *rtxt = fopen("reactionrates.txt", "w");
    // double rates[NREACTIONS];
#endif

    //
    std::vector<double> timesteps;
    double logtstart = 2.0, logtend = 4.0, logtstep = 0.1;
    double time = 0.0;
    for (double logtime = logtstart; logtime < logtend + 0.1 * logtstep;
         logtime += logtstep) {
        double dtyr = pow(10.0, logtime) - time;
        timesteps.push_back(dtyr);
        time += dtyr;
    }
    //

    double dtyr = 0.0, curtime = 0.0;

    // write the initial abundances
    for (int isys = 0; isys < nsystem; isys++) {
        fwrite((double *)&isys, sizeof(double), 1, fbin);
        fwrite(&time, sizeof(double), 1, fbin);
        fwrite(y[isys], sizeof(double), NEQUATIONS, fbin);

        fprintf(ftxt, "%13.7e ", (double)isys);
        fprintf(ftxt, "%13.7e ", time);
        for (int j = 0; j < NEQUATIONS; j++) {
            fprintf(ftxt, "%13.7e ", y[isys][j]);
        }
        fprintf(ftxt, "\n");
    }

    for (auto step = timesteps.begin(); step != timesteps.end(); step++) {
#ifdef NAUNET_DEBUG
        // EvalRates only receive one system as input, disabled in parallel test
        // EvalRates(rates, y, data);
        // for (int j = 0; j < NREACTIONS; j++) {
        //     fprintf(rtxt, "%13.7e ", rates[j]);
        // }
        // fprintf(rtxt, "\n");
#endif
        //
        //

        dtyr = *step;

        Timer timer;
        timer.start();
        for (int isys = 0; isys < nsystem; isys++) {
            naunet.Solve(y[isys], dtyr * spy, &data[isys]);
        }
        timer.stop();

        curtime += dtyr;

        // write the abundances after each step
        for (int isys = 0; isys < nsystem; isys++) {
            fwrite((double *)&isys, sizeof(double), 1, fbin);
            fwrite(&time, sizeof(double), 1, fbin);
            fwrite(y[isys], sizeof(double), NEQUATIONS, fbin);

            fprintf(ftxt, "%13.7e ", (double)isys);
            fprintf(ftxt, "%13.7e ", time);
            for (int j = 0; j < NEQUATIONS; j++) {
                fprintf(ftxt, "%13.7e ", y[isys][j]);
            }
            fprintf(ftxt, "\n");
        }

        // float duration = (float)timer.elapsed() / 1e6;
        double duration = timer.elapsed();
        fprintf(ttxt, "%8.5e \n", duration);
        printf("Time = %13.7e yr, elapsed: %8.5e sec\n", curtime, duration);
    }

    fclose(fbin);
    fclose(ftxt);
    fclose(ttxt);

#ifdef NAUNET_DEBUG
    // fclose(rtxt);
#endif

    if (naunet.Finalize() == NAUNET_FAIL) {
        printf("Finalize Fail\n");
        return 1;
    }

    delete[] data;
    delete[] y;

    return 0;
}