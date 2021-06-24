#include <cstdio>
#include <cstdlib>
#include <algorithm>
#include <iostream>
#include <vector>
#include <complex>
#include <cmath>
#include <thread>
#include <chrono>
#include <ctime>
#include <random>
#include "pn_seq_chan_sounder.h"
#include <mpi.h>

using namespace std::chrono;

int main(int argc, char *argv[]) {
    //Initialzing MPI
    int provided;
    MPI_Init_thread(NULL, NULL, MPI_THREAD_MULTIPLE, &provided);
	//getting size and rank
	int gsize, grank;
	MPI_Comm_size(MPI_COMM_WORLD, &gsize);
	MPI_Comm_rank(MPI_COMM_WORLD, &grank);

    char *proc_name;
    int name_len;
    proc_name = (char *)malloc(MPI_MAX_PROCESSOR_NAME*sizeof(char));
    MPI_Get_processor_name(proc_name, &name_len);
    std::cout << "Proc name: " << proc_name << "\n";

    int fft_size = 64, prefix_size = 16, num_ants = 4, num_rx_ants = 1, num_threads = 1, offset_ants = 0, num_reps = 10, num_times = 1;
    float samp_rate = 20e6, sounding_time = 1e-3;
    high_resolution_clock::time_point start, finish;

    if (argc > 1) {
        fft_size = atoi(argv[1]);
    }
    if (argc > 2) {
        prefix_size = atoi(argv[2]);
    }
    if (argc > 3) {
        num_ants = atoi(argv[3]);
    }
    if (argc > 4) {
        num_threads = atoi(argv[4]);
    }
    if (argc > 5) {
        offset_ants = atoi(argv[5]);
    }
    if (argc > 6) {
        num_reps = atoi(argv[6]);
    }
    if (argc > 7) {
        num_rx_ants = atoi(argv[7]);
    }
    if (argc > 8) {
        num_times = atoi(argv[8]);
    }
    if (argc > 9) {
        samp_rate = strtof(argv[9], NULL);
    }
    if (argc > 10) {
        sounding_time = strtof(argv[10], NULL);
    }

    //chan_sounder chsdr(fft_size, prefix_size, 1, num_ants);
    int pn_len = 1023;
    std::vector<std::vector<uint16_t>> polynomial(4, std::vector<uint16_t>(11));
    polynomial[0] = {1,0,0,0,0,0,0,1,0,0,1};
    polynomial[1] = {1,0,0,0,0,0,1,1,0,1,1};
    polynomial[2] = {1,0,0,0,0,1,0,0,1,1,1};
    polynomial[3] = {1,0,0,0,0,1,0,1,1,0,1};
    std::vector<int> pn_seq = pn_seq_gen(polynomial[0], pn_len);
    std::vector<std::complex<float>> pn_comp(pn_len);
    for (int i = 0; i < pn_len; i++) {
        pn_comp[i] = std::complex<float>(pn_seq[i],0);
    }
    std::vector<std::vector<int>> pn_seq_chan_sound;
    std::vector<std::complex<float>> pilots(fft_size - 1);
    std::vector<std::vector<std::complex<float>>> out_vec;
    std::vector<std::vector<std::complex<float>>> demod_vec;
    std::vector<std::complex<float>> global_demod_vec;

    double create_time = 0, sound_time = 0, corr_time = 0;

    int global_rx_ants = num_rx_ants;
    global_demod_vec.resize(num_rx_ants*num_ants*(pn_len*2 - 1));

    wh_matrix wh_mat;
    wh_mat.create_walsh_mat(3);


    //Creating random BPSK pilots
    
    std::cout << "Starting Sounding experiment...\n";
    pn_seq_chan_sound = create_pn_seqs(polynomial, 0, num_ants, num_ants);
    //std::cout << "PN sequences created...\n";
    for (int times = 0; times < num_times; times++) {
        srand(times);
        //std::cout << "Creating pilot vector...\n";
        //for (int i = 0; i < fft_size - 1; i++) {
        //    pilots[i] = (float)0.707 * std::complex<float>((float)(2*(rand() % 2) - 1), (float)(2*(rand() % 2) - 1));
        //}

        //std::cout << "Creating channel sounding frame...\n";
        start = high_resolution_clock::now();
        create_pn_seq_frame(pn_seq_chan_sound, out_vec, num_ants, num_ants, num_threads);
        finish = high_resolution_clock::now();
        create_time += duration_cast<duration<double>>(finish - start).count();

        //std::cout << "Copying channel vector...\n";
        std::vector<int> size_of_proc_data(gsize), displ(gsize);
        load_balancing_mpi(&size_of_proc_data[0], &displ[0], gsize, global_rx_ants);
        std::vector<std::vector<std::complex<float>>> chan_vec;
        chan_vec.resize(size_of_proc_data[grank]);
        //Number of rx antennas get distributed here to multiple servers
        num_rx_ants = size_of_proc_data[grank];
        //std::cout << "Reallocated number of rx antennas...\n";
        for (int rx = 0; rx < num_rx_ants; rx++) {
            chan_vec[rx].resize((int)out_vec[0].size());
            for (int tx = 0; tx < num_ants; tx++) {
                for (int i = 0; i < out_vec[tx].size(); i++) {
                    chan_vec[rx][i] = chan_vec[rx][i] + out_vec[tx][i];
                }
            }
        }

        //std::cout << "Performing channel sounding...\n";
        
        start = high_resolution_clock::now();
        sound_pn_frame(pn_seq_chan_sound, chan_vec, demod_vec, num_ants, num_rx_ants, num_threads);
        //std::cout << "Averaging...\n";
        average_multiple_pn_syms(demod_vec, pn_seq_chan_sound[0].size(), num_ants, num_rx_ants, num_threads);
        for (int i = displ[grank]; i < size_of_proc_data[grank]; i++) {
            for (int j = 0; j < demod_vec[0].size(); j++) {
                global_demod_vec[i*(demod_vec[0].size()) + j] = demod_vec[i][j];
            }
        }
        if (grank == 0) {
            MPI_Gather(MPI_IN_PLACE, demod_vec[0].size()*size_of_proc_data[grank], MPI_COMPLEX, (void *)&global_demod_vec[displ[grank]*demod_vec[0].size()], demod_vec[0].size()*size_of_proc_data[grank], MPI_COMPLEX, 0, MPI_COMM_WORLD);
        } else {
            MPI_Gather((void *)&global_demod_vec[displ[grank]*demod_vec[0].size()], demod_vec[0].size()*size_of_proc_data[grank], MPI_COMPLEX, (void *)&global_demod_vec[displ[grank]*demod_vec[0].size()], demod_vec[0].size()*size_of_proc_data[grank], MPI_COMPLEX, 0, MPI_COMM_WORLD);
        }

        finish = high_resolution_clock::now();
        sound_time += duration_cast<duration<double>>(finish - start).count();

        MPI_Barrier(MPI_COMM_WORLD);
        
    }
    std::cout << "Sounding experiment done...\n";
    MPI_Barrier(MPI_COMM_WORLD);
    
   /*
    for (int times = 0; times < num_times; times++) {
        

        //std::cout << "Creating channel sounding frame...\n";
        start = high_resolution_clock::now();
        create_pn_seq_frame(polynomial, out_vec, num_ants, 0, num_ants, samp_rate, sounding_time, num_threads);
        finish = high_resolution_clock::now();
        create_time += duration_cast<duration<double>>(finish - start).count();

        //std::cout << "Copying channel vector...\n";
        std::vector<int> size_of_proc_data(gsize), displ(gsize);
        load_balancing_mpi(&size_of_proc_data[0], &displ[0], gsize, global_rx_ants);
        std::vector<std::vector<std::complex<float>>> chan_vec;
        chan_vec.resize(size_of_proc_data[grank]);
        //Number of rx antennas get distributed here to multiple servers
        num_rx_ants = size_of_proc_data[grank];
        //std::cout << "Reallocated number of rx antennas...\n";
        for (int rx = 0; rx < num_rx_ants; rx++) {
            chan_vec[rx].resize((int)out_vec[0].size());
            for (int tx = 0; tx < num_ants; tx++) {
                for (int i = 0; i < out_vec[tx].size(); i++) {
                    chan_vec[rx][i] = chan_vec[rx][i] + out_vec[tx][i];
                }
            }
        }

        //std::cout << "Performing channel sounding...\n";
        start = high_resolution_clock::now();
        sound_pn_frame(polynomial, chan_vec, demod_vec, num_ants, num_rx_ants, samp_rate, sounding_time, num_threads);
        for (int i = displ[grank]; i < size_of_proc_data[grank]; i++) {
            for (int j = 0; j < demod_vec[0].size(); j++) {
                global_demod_vec[i*(demod_vec[0].size()) + j] = demod_vec[i][j];
            }
        }
        if (grank == 0) {
            MPI_Gather(MPI_IN_PLACE, demod_vec[0].size()*size_of_proc_data[grank], MPI_COMPLEX, (void *)&global_demod_vec[displ[grank]*demod_vec[0].size()], demod_vec[0].size()*size_of_proc_data[grank], MPI_COMPLEX, 0, MPI_COMM_WORLD);
        } else {
            MPI_Gather((void *)&global_demod_vec[displ[grank]*demod_vec[0].size()], demod_vec[0].size()*size_of_proc_data[grank], MPI_COMPLEX, (void *)&global_demod_vec[displ[grank]*demod_vec[0].size()], demod_vec[0].size()*size_of_proc_data[grank], MPI_COMPLEX, 0, MPI_COMM_WORLD);
        }

        finish = high_resolution_clock::now();
        sound_time += duration_cast<duration<double>>(finish - start).count();

        MPI_Barrier(MPI_COMM_WORLD);
    }
    std::cout << "OFDM sounding experiment done...\n";
    MPI_Barrier(MPI_COMM_WORLD);

    //Testing linear correlation using FFT
    //std::vector<std::complex<float>> corr_out(2*pn_len - 1);
    //start = high_resolution_clock::now();
    //lin_corr_fft(&pn_comp[0], &pn_comp[0], &corr_out[0], pn_len, pn_len);
    //finish = high_resolution_clock::now();
    //corr_time += duration_cast<duration<double>>(finish - start).count();
    
    for (int i = 0; i < global_demod_vec; i++) {
        if (std::abs(corr_out[i]) > 0.1) {
            std::cout << corr_out[i] << ",";
        } else {
            std::cout << "0,";
        }
    }
    std::cout << "\n";
    */

    /*
    if (grank == 0) {
        std::cout << "Showing output...\n";
        for (int i = 0; i < num_ants; i++) {
            for (int j = 0; j < out_vec[i].size(); j++) {
                std::cout << out_vec[i][j] << ",";
            }
            std::cout << std::endl << std::endl;
        }

        for (int i = 0; i < 1; i++) {
            for (int j = 0; j < demod_vec[i].size(); j++) {
                std::cout << demod_vec[i][j] << ",";
            }
            std::cout << std::endl << std::endl;
        }

        for (int i = 0; i < pilots.size(); i++) {
            std::cout << pilots[i] << ",";
        }
        std::cout << std::endl << std::endl;
    }
    
    

    float err = 0, global_err = 0, phase_err = 0;
    for (int i = 0; i < num_rx_ants; i++) {
        for (int j = 0; j < demod_vec[i].size(); j++) {
            err += std::abs(demod_vec[i][j] - std::complex<float>(1,0));
        }
    }
    err = err/(float)(num_rx_ants*num_ants*(fft_size - 1));
    if (grank == 0) {
        for (int i = 0; i < num_rx_ants; i++) {
            for (int j = 0; j < num_ants*(fft_size - 1); j++) {
                global_err += std::abs(global_demod_vec[i*num_ants*(fft_size - 1) + j] - std::complex<float>(1,0));
            }
        }
        global_err = global_err/(float)(global_rx_ants*num_ants*(fft_size - 1));

        for (int i = 0; i < num_rx_ants; i++) {
            for (int j = 0; j < num_ants*(fft_size - 1); j++) {
                phase_err += std::abs(std::arg(global_demod_vec[i*num_ants*(fft_size - 1) + j])  - std::arg(std::complex<float>(1,0)));
            }
        }
        phase_err = (phase_err * 180/3.141592654)/(float)(global_rx_ants*num_ants*(fft_size - 1));
    }
    */

    MPI_Barrier(MPI_COMM_WORLD);
    double global_create_time, global_sound_time;
    MPI_Reduce(&create_time, &global_create_time, 1, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);
    MPI_Reduce(&sound_time, &global_sound_time, 1, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);
    //MPI_Reduce(&err, &global_err, 1, MPI_FLOAT, MPI_SUM, 0, MPI_COMM_WORLD);

    //std::cout << "Rank " << grank << " Mean processing error: " << err << "\n";
    if (grank == 0) {
        std::cout << "Rank " << grank << " Frame creation time: " << (global_create_time/(double)(num_times*gsize))*1e3 << " millisecond(s)\n";
        std::cout << "Rank " << grank << " Frame sounding time: " << (global_sound_time/(double)(num_times*gsize))*1e3 << " millisecond(s)\n";
        //std::cout << "Rank " << grank << " PN Sequence correlation time: " <<  corr_time*1e3 << " millisecond(s)\n";
        //std::cout << "Rank " << grank << " Mean processing error: " << (double)global_err << "\n";
        //std::cout << "Rank " << grank << " Mean processing phase error: " << (double)phase_err << " degrees\n"; 
    }
    free(proc_name);
    MPI_Barrier(MPI_COMM_WORLD);
    MPI_Finalize();
    //return 0;
}