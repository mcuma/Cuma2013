This directory contains a sample code for gravity forward modeling.

The main routine sets up the domain and receivers and then calls several routines for calculating of the gravity and gravity gradient response. It times the calculation and outputs the response to files outData*.dat

The response routines are as follows:

gr_fun_seq - naive sequential kernel with limited data reuse.
gr_fun_seq_opt - optimized sequential kernel with maximum data reuse and avoidance of expensive division operations.
gr_fun_seq_opt_omp - same as gr_fun_seq_opt with OpenMP parallelization, this is the CPU implementation discussed in the paper.
gr_fun_gpu_cpu_switch - kernel used for the GPU implementation, but ran on CPU, using switch statement to differentiate between different gravity response components. This kernel is very inefficient on the CPU since it recalculates all the data for each receiver and component and does not vectorize well on the CPU.
gr_fun_gpu_cpu - same as gr_fun_gpu_cpu_switch but with if statement instead of the switch. The if statement is used on the true GPU kernel since PGI compilers at the moment of writing do not support switch statement larger than 5.
gr_fun_gpu - OpenACC kernel for the GPU, this is the GPU implementation discussed in the paper.
gr_fun_vec - reference CPU vectorized kernel, not discussed in the current paper. The drawback of this vectorized implementation is no flexibility with what components to calculate at runtime. Introduction of such flexibility would require conditions in the inner loop which would considerably reduce the vectorization performance, as mentioned in the paper.
gr_fun_vec_omp - same as gr_dun_vec with OpenMP parallelization.

The executable included, grav_fwd, has been built with pgcc 12.8 and should run on Red Hat EL 5 Linux machine. 

File sample_output.txt shows an output ran on the same hardware as described in the paper, that is Intel Xeon X5660 and NVidia Tesla M2090.

To compile the code without OpenACC, using PGI compilers:
pgcc -mp=numa -fastsse grav_fwd.c -Minfo=all -o grav_fwd 

To compile with OpenACC, using PGI:
pgcc -mp=numa -acc -ta=nvidia,time -fastsse grav_fwd.c -Minfo=all -o grav_fwd 

Free 2 weeks trial of PGI compilers can be obtained at http://www.pgroup.com/support/trial.htm

For questions, contact Martin Cuma, m.cuma at utah.edu.

We acknowledge Yue Zhu for initial implementation of the forward modeling code and for the CPU vectorized routines.
