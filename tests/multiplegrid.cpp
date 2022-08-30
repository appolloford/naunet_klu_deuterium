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
    int nsystem      = 2048;
    double spy       = 86400.0 * 365.0;

    NaunetData *data = new NaunetData[nsystem];
    //
    double pi        = 3.14159265;
    double rD        = 1.0e-5;
    double rhoD      = 3.0;
    double DtoGM     = 7.09e-3;
    double amH       = 1.66043e-24;
    double nH        = 1e5;
    double OPRH2     = 0.1;

    for (int isys = 0; isys < nsystem; isys++) {
        data[isys].nH          = nH;
        data[isys].Tgas        = 15.0;
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

    if (naunet.Reset(nsystem) == NAUNET_FAIL) {
        throw std::runtime_error("Fail to reset the number of systems");
    }

    //
    double *y = new double[nsystem * NEQUATIONS];
    for (int isys = 0; isys < nsystem; isys++) {
        for (int i = 0; i < NEQUATIONS; i++) {
            y[isys * NEQUATIONS + i] = 1.e-40;
        }
        y[isys * NEQUATIONS + IDX_pH2I]    = 1.0 / (1.0 + OPRH2) * 0.5 * nH;
        y[isys * NEQUATIONS + IDX_oH2I]    = OPRH2 / (1.0 + OPRH2) * 0.5 * nH;
        y[isys * NEQUATIONS + IDX_HDI]     = 1.5e-5 * nH;
        y[isys * NEQUATIONS + IDX_HeI]     = 1.0e-1 * nH;
        y[isys * NEQUATIONS + IDX_NI]      = 2.1e-6 * nH;
        y[isys * NEQUATIONS + IDX_OI]      = 1.8e-5 * nH;
        y[isys * NEQUATIONS + IDX_CI]      = 7.3e-6 * nH;
        y[isys * NEQUATIONS + IDX_GRAIN0I] = 1.3215e-12 * nH;
    }


    FILE *fbin = fopen("evolution_multiplegrid.bin", "w");
    FILE *ftxt = fopen("evolution_multiplegrid.txt", "w");
    FILE *ttxt = fopen("time_parallel.txt", "w");

#ifdef NAUNET_DEBUG
    printf("Initialization is done. Start to evolve.\n");
    // FILE *rtxt = fopen("reactionrates.txt", "w");
    // double rates[NREACTIONS];
#endif

    //
    std::vector<double> timesteps;
    double time, next;
    int nsteps = 50;  // number of steps, maximum 10046 in `timeres.dat`

    FILE *tfile = fopen("timeres.dat", "r");

    fscanf(tfile, "%lf\n", &time);  // read the first one
    for (int i = 0; i < nsteps; i++) {
        fscanf(tfile, "%lf\n", &next);
        timesteps.push_back(next - time);
        time = next;
    }
    fclose(tfile);


    double dtyr = 0.0, curtime = 0.0;

    // write the initial abundances
    for (int isys = 0; isys < nsystem; isys++) {
        fwrite((double *)&isys, sizeof(double), 1, fbin);
        fwrite(&curtime, sizeof(double), 1, fbin);
        fwrite(&y[isys * NEQUATIONS], sizeof(double), NEQUATIONS, fbin);

        fprintf(ftxt, "%13.7e ", (double)isys);
        fprintf(ftxt, "%13.7e ", curtime);
        for (int j = 0; j < NEQUATIONS; j++) {
            fprintf(ftxt, "%13.7e ", y[isys * NEQUATIONS + j]);
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
        naunet.Solve(y, dtyr * spy, data);
        timer.stop();

        curtime += dtyr;

        // write the abundances after each step
        for (int isys = 0; isys < nsystem; isys++) {
            fwrite((double *)&isys, sizeof(double), 1, fbin);
            fwrite(&curtime, sizeof(double), 1, fbin);
            fwrite(&y[isys * NEQUATIONS], sizeof(double), NEQUATIONS, fbin);

            fprintf(ftxt, "%13.7e ", (double)isys);
            fprintf(ftxt, "%13.7e ", curtime);
            for (int j = 0; j < NEQUATIONS; j++) {
                fprintf(ftxt, "%13.7e ", y[isys * NEQUATIONS + j]);
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