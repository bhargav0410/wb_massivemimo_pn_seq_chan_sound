#ifndef PN_SEQ_CHAN_SOUNDER_H
#define PN_SEQ_CHAN_SOUNDER_H

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <algorithm>
#include <iostream>
//#include <fftw3.h>
#include <vector>
#include <complex>
#include <cmath>
#include <thread>
#include <omp.h>
#include "../muFFT/fft.h"
#include "../muFFT/fft_internal.h"
//#include "../muFFT/fft.h"
//#include "../muFFT/fft_internal.h"
#include <immintrin.h>


#define PI 3.141592654


//Walsh matrix struct
struct wh_matrix {
    std::vector<std::vector<int>> wh_mat;

    void create_walsh_mat(int power) {
        wh_mat.resize((int)pow(2,power));
        for (int i = 0; i < wh_mat.size(); i++) {
            wh_mat[i].resize((int)pow(2,power));
        }
        wh_mat[0][0] = 1;
        for (int i = 0; i < power; i++) {
            for (int ii = (int)pow(2,i); ii < (int)pow(2,i+1); ii++) {
                for (int jj = (int)pow(2,i); jj < (int)pow(2,i+1); jj++) {
                    wh_mat[ii - (int)pow(2,i)][jj] = wh_mat[ii - (int)pow(2,i)][jj - (int)pow(2,i)];
                    wh_mat[jj][ii - (int)pow(2,i)] = wh_mat[ii - (int)pow(2,i)][jj];
                    wh_mat[ii][jj] = -1*wh_mat[ii - (int)pow(2,i)][jj - (int)pow(2,i)];
                }
            }
        }
    }
};

//Distributed the tasks amongst workers with as much fairness as possible
void load_balancing_mpi(int *size_of_proc_data, int *displ, int num_procs, int len) {
    int displ_of_proc = 0;
    for (int i = 0; i < num_procs; i++) {
        displ[i] = displ_of_proc;
        size_of_proc_data[i] = (int)floor((float)len/(float)num_procs);
        if (i < (int)len % num_procs) {
            size_of_proc_data[i] += 1;
        }
        displ_of_proc += size_of_proc_data[i];
    }
}

//Generates PN sequence of 1 and -1 using LFSR based on polynomial given
std::vector<int> pn_seq_gen(std::vector<uint16_t> polynomial, uint16_t length) {
    std::vector<int> start_state(polynomial.size()-1, 0), output(length);
    start_state[0] = 1;
    int temp_bit;
    //std::cout << "Starting PN seq gen...\n";
    for (int i = 0; i < length; i++) {
        temp_bit = 0;
        for (int p = 0; p < polynomial.size() - 1; p++) {
            if (polynomial[p] > 0) {
                temp_bit += start_state[p];
                temp_bit = temp_bit % 2;
            } 
        }
        //std::cout << temp_bit << ",";
        for (int r = 0; r < start_state.size() - 1; r++) {
            start_state[r] = start_state[r+1];
        }
        start_state[start_state.size() - 1] = temp_bit;
        output[i] = 2*temp_bit - 1;
        //std::cout << output[i] << ",";
    }
    //std::cout << "\n";
    //std::cout << "PN sequence generated...\n";
    return output;
}

//Performs linear cross-correlation with a PN sequence to find PN sequence peaks in the received sequence and returns the index at which PN sequence starts
int find_pn_seq(std::complex<float> *in_seq, int *pn_seq, int in_size, int pn_size, float thres, int threads) {
    if (in_size < pn_size) {
        std::cout << "Correlation not possible...\n";
    }
    //out_seq = (std::complex<float> *)malloc((size_t)((in_size - pn_size + 1) * sizeof(std::complex<float>)));
    std::complex<float> temp;
    for (int i = 0; i < in_size - pn_size + 1; i++) {
        temp = 0;
        #pragma omp parallel num_threads(threads)
        {
            std::complex<float> local_temp = 0;
            #pragma omp for
            for (int j = 0; j < pn_size; j++) {
                local_temp += (in_seq[i + j] * std::complex<float>((float)pn_seq[j], 0));
            }
            //std::cout << "Local sum: " << local_temp << "\n";
            #pragma omp critical
            temp += local_temp;
        }
        //std::cout << temp << ",";
        if ((float)std::abs(temp)/(float)pn_size > thres) {
            return i;
        }
    }
    //std::cout << "\n";
    return -1;
}

//FFT of one row
void single_thread_fft(std::complex<float> *fft_in, std::complex<float> *fft_out, int fft_size) {
    //Setting up plan to execute
    //fftwf_plan plan;
    //plan = fftwf_plan_dft_1d(fft_size, (fftwf_complex *)fft_in, (fftwf_complex *)fft_out, FFTW_FORWARD, /*FFTW_MEASURE*/ FFTW_ESTIMATE);

    //Executing fft
    //fftwf_execute(plan);
    //Destroying plan
    //fftwf_destroy_plan(plan);

    mufft_plan_1d *muplan = mufft_create_plan_1d_c2c(fft_size, MUFFT_FORWARD, 1);
    mufft_execute_plan_1d(muplan, (cfloat *)fft_out, (cfloat *)fft_in);
    mufft_free_plan_1d(muplan);
}

//IFFT of one row
void single_thread_ifft(std::complex<float> *fft_in, std::complex<float> *fft_out, int fft_size) {
    //Setting up plan to execute
    //fftwf_plan plan;
    //plan = fftwf_plan_dft_1d(fft_size, (fftwf_complex *)fft_in, (fftwf_complex *)fft_out, FFTW_BACKWARD, /*FFTW_MEASURE*/ FFTW_ESTIMATE);

    //Executin ifft
    //fftwf_execute(plan);
    //Destroying plan
    //fftwf_destroy_plan(plan);

    mufft_plan_1d *muplan = mufft_create_plan_1d_c2c(fft_size, MUFFT_INVERSE, 1);
    mufft_execute_plan_1d(muplan, (cfloat *)fft_out, (cfloat *)fft_in);
    mufft_free_plan_1d(muplan);
}

/*Averages multiple vectors into one vector*/
void vector_averaging(std::complex<float> *input, std::complex<float> *output, int num_vectors, int vector_len) {

    //std::complex<float> *temp;
    //temp = (std::complex<float> *)malloc((size_t)(vector_len * sizeof(std::complex<float>)));
    //Output vector has first input vector
    for (int j = 0; j < vector_len; j++) {
        output[j] = input[j];
    }
    //Summing values of all vectors in one vector
    for (int i =  1; i < num_vectors; i++) {
        for (int j = 0; j < vector_len; j += 16) {
            if ((j + 16) < vector_len) {
                for (int jj = 0; jj < 16; jj++) {
                    output[jj + j] = output[jj + j] + input[jj + j + i*vector_len];
                }
            } else {
                for (int jj = i; jj < vector_len; jj++) {
                    output[jj] = output[jj] + input[jj + i*vector_len];
                }
            }
        }
    }
    //Dividing by number of vectors
    for (int j = 0; j < vector_len; j++) {
        output[j] = output[j]/(float)num_vectors;
    }
}

//Swaps the [0:FFTsize/2-1] and [-FFTsize/2:FFTsize-1] halves of OFDM symbols and stores in same vector
void swap_halves(std::complex<float> *vec, int fft_size) {
    std::vector<std::complex<float>> temp((int)ceil((float)fft_size/(float)2));
    for (int i = 0; i < fft_size/2; i++) {
        temp[i] = vec[i];
        vec[i] = vec[i + fft_size/2];
        vec[i + fft_size/2] = temp[i];
    }
}

//Dividing 8-16 elements at same time
inline void divide_8_elems(std::complex<float> *in1, std::complex<float> *in2, std::complex<float> *out) {
    for (int i = 0; i < 8; i++) {
        out[i] = in1[i]/in2[i];
    }
}

inline void divide_16_elems(std::complex<float> *in1, std::complex<float> *in2, std::complex<float> *out) {
    for (int i = 0; i < 16; i++) {
        out[i] = in1[i]/in2[i];
    }
}

//Performs element by element division of complex vectors and stores answer in numerator
void divide_by_vec(std::complex<float> *numer, std::complex<float> *denom, std::complex<float> *out, int len) {

	for (int i = 0; i < len; i += 16) {
        if (i + 16 < len) {
            divide_16_elems(&numer[i], &denom[i], &out[i]);
        } else {
            for (int j = i; j < len; j++) {
                out[i] = numer[i]/denom[i];
            }
        }
    }
}

//Performs element by element multiplication of one complex vector and conjugate of another complex vector and stores answer in third vector
void mult_by_conj(std::complex<float> *in_vec, std::complex<float> *conj_vec, std::complex<float> *out, int len) {
	for (int i = 0; i < len; i += 16) {
        if (i + 16 < len) {
            for (int j = 0; j < 16; j++) {
                out[i + j] = in_vec[i + j] * std::conj(conj_vec[i + j]);
            }
        } else {
            for (int j = i; j < len; j++) {
                out[j] = in_vec[j] * std::conj(conj_vec[j]);
            }
        }
    }
}

//Finding maximum absolute value within vector
float find_max_val(std::complex<float> *in_vec, int len, int threads) {
    std::vector<float> abs_vec(len);
    float temp_ret;

    //Getting absolute value of complex number
    #pragma omp parallel for num_threads(threads)
    for (int i = 0; i < len; i++) {
        abs_vec[i] = std::abs(in_vec[i]);
    }

    //Finding max value of absolute value vector
    for (int step = 1; step < len; step *= 2) {

        #pragma omp parallel for num_threads(threads)
        for (int i = 0; i < len; i += step*2) {
            if (i + step < len) {
                abs_vec[i] = std::max(abs_vec[i], abs_vec[i+step]);
            }
        }
    }
    return abs_vec[0];
}

//Correlates one vector with cyclic shifts of another vector and gives output in a third vector.
//Total number of cyclic shifts are given by num_cyclic_shifts value. This value should include the zero cyclic shift which is basically correlation with a non-roatated vector.
void circ_correlate(std::complex<float> *in_vec, std::complex<float> *cyclic_shift_vec, std::complex<float> *out, int num_cyclic_shifts, int len) {
    //Performing correlation
    for (int i = 0; i < num_cyclic_shifts; i++) {
        for (int j = 0; j < len; j++) {
            out[i] = out[i] + (in_vec[j] * cyclic_shift_vec[(int)((j + i) % num_cyclic_shifts)]);
        }
    }
}

void circ_corr_fft(std::complex<float> *in_vec, std::complex<float> *cyclic_shift_vec, std::complex<float> *out, int len) {
    int new_len = len;
    std::vector<std::complex<float>> temp_in1, temp_in2, temp_out;
    if ((float)ceil(log2((float)len)) - (float)log2((float)len) > 0.001) {
        new_len = (int)ceil(log2((float)len));
    }
    temp_in1.resize(new_len);
    temp_in2.resize(new_len);
    temp_out.resize(len);
    for (int i = 0; i < len; i++) {
        temp_in1[i] = in_vec[i];
        temp_in2[i] = cyclic_shift_vec[i];
    }
    single_thread_fft(&temp_in1[0], &temp_in1[0], new_len);
    single_thread_fft(&temp_in2[0], &temp_in2[0], new_len);
    mult_by_conj(&temp_in1[0], &temp_in2[0], &temp_out[0], new_len);
    single_thread_ifft(&temp_out[0], &temp_out[0], new_len);
    for (int i = 0; i < len; i++) {
        out[i] = temp_out[i];
    }
}

void lin_corr_fft(std::complex<float> *in_vec1, std::complex<float> *in_vec2, std::complex<float> *out, int len1, int len2) {
    int corr_len = len1 + len2 - 1;
    int new_len = corr_len;
    std::vector<std::complex<float>> temp_in1, temp_in2, temp_out;
    if ((float)ceil(log2((float)corr_len)) - (float)log2((float)corr_len) > 0.001) {
        new_len = (int)pow(2,(int)ceil(log2((float)corr_len)));
    }
    //std::cout << "New length: " << new_len << "\n";
    temp_in1.resize(new_len, 0);
    temp_in2.resize(new_len, 0);
    temp_out.resize(new_len, 0);
    memcpy((void *)&temp_in1[0], (const void *)&in_vec1[0], len1*sizeof(std::complex<float>));
    memcpy((void *)&temp_in2[0], (const void *)&in_vec2[0], len1*sizeof(std::complex<float>));
    //std::cout << "Performing fft...\n";
    single_thread_fft(&temp_in1[0], &temp_in1[0], new_len);
    single_thread_fft(&temp_in2[0], &temp_in2[0], new_len);
    //std::cout << "Multiplying with conjugate...\n";
    mult_by_conj(&temp_in1[0], &temp_in2[0], &temp_out[0], new_len);
    //std::cout << "Performing ifft...\n";
    single_thread_ifft(&temp_out[0], &temp_out[0], new_len);
    //std::cout << "Copying output...\n";
    memcpy((void *)&out[0], (const void *)&temp_out[0], corr_len*sizeof(std::complex<float>));

}


std::vector<std::vector<int>> create_pn_seqs(std::vector<std::vector<uint16_t>> polys, uint16_t pre_ants, uint16_t num_tx_ants, uint16_t total_tx_ants) {
    std::vector<std::vector<int>> pn_seq(num_tx_ants);
    uint16_t pn_len = (int)pow(2, polys[0].size()-1) - 1;
    std::vector<int> temp_pn;
    if (polys.size() >= total_tx_ants) {
        for (uint16_t i = pre_ants; i < pre_ants + num_tx_ants; i++) {
            pn_seq[i - pre_ants] = pn_seq_gen(polys[i], pn_len);
        }
    } else {
        for (uint16_t i = pre_ants; i < pre_ants + num_tx_ants; i++) {
            temp_pn = pn_seq_gen(polys[i % polys.size()], pn_len);
            pn_seq[i - pre_ants].resize(pn_len);
            int circshift = 0;//i * (int)std::floor((float)pn_len/(float)std::floor((float)total_tx_ants/(float)polys.size()));
            for (int j = 0; j < pn_len; j++) {
                pn_seq[i - pre_ants][j] = temp_pn[(j + circshift) % pn_len];
            }
        }
    }
    return pn_seq;
}

void create_pn_seq_frame(std::vector<std::vector<int>> &pn_seq, std::vector<std::vector<std::complex<float>>> &out_vec, uint16_t num_tx_ants, uint16_t total_tx_ants, uint16_t num_threads) {

//    std::vector<std::vector<int>> pn_seq(num_tx_ants);
    uint16_t pn_len = pn_seq[0].size();
    //std::cout << "PN sequence created...\n";
    //int total_frame_samps = (int)std::floor(max_frame_time * samp_rate);
    //int num_reps = (int)pow(2, (int)std::floor(log2((float)total_frame_samps/(float)pn_len)));
    int num_reps = (int)pow(2,std::ceil(log2(total_tx_ants)));
    wh_matrix wh_mat;
    wh_mat.create_walsh_mat((int)log2(num_reps));

    if (out_vec.size() < num_tx_ants) {
        out_vec.resize(num_tx_ants);
    }
    for (int i = 0; i < num_tx_ants; i++) {
        out_vec[i].resize(pn_len*num_reps);
    }

    //std::cout << "Resized out vec...\n";
    //Repeating PN sequence syms for each tx antenna with pre and post offset between repetitions of syms
    if (num_reps < num_threads) {
        for (int reps = 0; reps < num_reps; reps++) {
            #pragma omp parallel for num_threads(num_threads)
            for (int tx = 0; tx < num_tx_ants; tx++) {
                for (int i = 0; i < pn_len; i++) {
                    out_vec[tx][reps*pn_len + i] = std::complex<float>((float)(wh_mat.wh_mat[tx][reps] * pn_seq[tx][i]),0);
                }
            }
            #pragma omp barrier
        }
    } else {
        //std::cout << "Copying PN sequences in out vec...\n";
        for (int tx = 0; tx < num_tx_ants; tx++) {
            #pragma omp parallel for num_threads(num_threads)
            for (int reps = 0; reps < num_reps; reps++) {
                //memcpy((void *)&out_vec[tx][reps*(pre_offset + num_tx_ants*(fft_size + prefix_size) + post_offset) + tx*(fft_size + prefix_size)], (const void *)&temp_out[0], fft_size + prefix_size);
                for (int i = 0; i < pn_len; i++) {
                    out_vec[tx][reps*pn_len + i] = std::complex<float>((float)(wh_mat.wh_mat[tx][reps] * pn_seq[tx][i]),0);
                }
            }
            #pragma omp barrier
        }
        //std::cout << "Copying done ...\n";
    }

}


void sound_pn_frame(std::vector<std::vector<int>> pn_seq, std::vector<std::vector<std::complex<float>>> &in_vec, std::vector<std::vector<std::complex<float>>> &out_vec, uint16_t total_tx_ants, uint16_t num_rx_ants, uint16_t num_threads) {

    uint16_t pn_len = pn_seq[0].size();
    //int total_frame_samps = (int)std::floor(max_frame_time * samp_rate);
    //uint16_t num_reps = (int)pow(2, (int)std::floor(log2((float)total_frame_samps/(float)pn_len)));
    int num_reps = (int)pow(2,std::ceil(log2(total_tx_ants)));
    wh_matrix wh_mat;
    wh_mat.create_walsh_mat((int)log2(num_reps));
    
    out_vec.resize(num_rx_ants);
    for (int i = 0; i < in_vec.size(); i++) {
        out_vec[i].resize(num_reps*total_tx_ants*(2*pn_len - 1));
    }

    //std::vector<std::thread> thread_vec(num_threads);
    //std::vector<std::complex<float>> *in_ptr(num_threads), *out_ptr(num_threads);
    //omp_set_dynamic(0);
    //omp_set_num_threads(num_threads);

    std::vector<std::complex<float>> pn_with_whmat(pn_len);
    if (num_reps >= num_threads) {
        //std::cout << "Num syms greater...\n";
        for (int rx = 0; rx < num_rx_ants; rx++) {
            #pragma omp parallel for num_threads(num_threads)
            for (int sym = 0; sym < num_reps; sym += 1) {
                for (int tx = 0; tx < total_tx_ants; tx++) {
                    for (int j = 0; j < pn_len; j++) {
                        pn_with_whmat[j] = wh_mat.wh_mat[tx][sym] * pn_seq[tx][j];
                    }
                    lin_corr_fft(&in_vec[rx][sym*pn_len], &pn_with_whmat[0], &out_vec[rx][(tx*num_reps + sym)*(2*pn_len - 1)], pn_len, pn_len);
                }
            }
        }
    } else {
        int num_rx_threads = std::max(1,(int)floor(((float)num_rx_ants/(float)(num_reps + num_rx_ants))*num_threads));
        int num_sym_threads = std::max(1,(int)floor((float)num_threads/(float)num_rx_threads));
        int extra_procs = num_threads - (num_rx_threads*num_sym_threads);
        //std::cout << "Num ants greater...\n";
        #pragma omp parallel for num_threads(num_rx_threads)
        for (int rx = 0; rx < num_rx_ants; rx += 1) {
            int offset = 0;
            if (rx < extra_procs) {
                offset = 1;
            }
            #pragma omp parallel for num_threads(num_sym_threads + offset)
            for (int sym = 0; sym < num_reps; sym += 1) {
                for (int tx = 0; tx < total_tx_ants; tx++) {
                    for (int j = 0; j < pn_len; j++) {
                        pn_with_whmat[j] = wh_mat.wh_mat[tx][sym] * pn_seq[tx][j];
                    }
                    lin_corr_fft(&in_vec[rx][sym*pn_len], &pn_with_whmat[0], &out_vec[rx][(tx*num_reps + sym)*(2*pn_len - 1)], pn_len, pn_len);
                }
            }
        }
        #pragma omp barrier
    }
    
}

//Ditributed averaging of multiple repitions of each transmitted PN sequence
void average_multiple_pn_syms(std::vector<std::vector<std::complex<float>>> &out_vec, uint16_t pn_len, uint16_t total_tx_ants, uint16_t num_rx_ants, uint16_t num_threads) {
    
    int corr_len = 2*pn_len-1;
    int num_reps = (uint16_t)(out_vec[0].size()/(corr_len*total_tx_ants));
    if (num_reps == 1) {
        return;
    }
    int num_rx_threads = std::max(1,(int)floor(((float)num_rx_ants/(float)(total_tx_ants + num_rx_ants))*num_threads));
    int num_tx_threads = std::max(1,(int)floor((float)num_threads/(float)num_rx_threads));
    int extra_procs = num_threads - (num_rx_threads*num_tx_threads);

    #pragma omp parallel for num_threads(num_rx_threads)
    for (int rx = 0; rx < num_rx_ants; rx += 1) {
        int offset = 0;
        if (rx < extra_procs) {
            offset = 1;
        }
        #pragma omp parallel for num_threads(num_tx_threads + offset)
        for (int tx = 0; tx < total_tx_ants; tx += 1) {
            std::vector<std::complex<float>> temp(num_reps * corr_len);
            //for (int i = 0; i < num_reps; i += 1) {
                memcpy((void *)&temp[0], (const void *)&out_vec[rx][tx*num_reps*corr_len], num_reps*corr_len*sizeof(std::complex<float>));
            //}
            vector_averaging(&temp[0], &temp[0], num_reps, 2*pn_len - 1);
            memcpy((void *)&out_vec[rx][tx*corr_len], (const void *)&temp[0], corr_len*sizeof(std::complex<float>));
        }
        out_vec[rx].resize(total_tx_ants*(2*pn_len-1));
    }
}


#endif