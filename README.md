# Genetic Tempering with a Graphics Processing Unit (GT-GPU)
A flexible genetic algorithm used to temper Monte Carlo results

# Overview

The goal of this code is to generate some basic principles for making Monte Carlo on a GPU that is no more complicated than the Metropolis-Hastings algorithm.  The parallelization scheme looks like parallel tempering but represents the idealized limit of perfectly uncorrelated Monte Carlo iterations and all possible crossings.  GT-GPU does not reproduce the exact crossing structure, as commented in the main text, and so is a separate method.  The main source of systematic error leads to a hysteresis loop from unthermalized samples, but this is reduced with more sampling.  The results converge very quickly.

Further, the many difficult lattice models possible are captured by this method, making it extremely flexible.  This GT-GPU algorithm is tested on RBIMs (with a separate bond configuration on each thread) and Kagome lattices.  All are found to similar precision as the main text's Ising model in seconds.

A full code and guidelines on what parameters to pick are discussed.  The code is 300 lines long or so.  A separate file is dedicated to the GPU and is about 100 lines.  Some efficiency in writing the code for fewer lines was not performed in order to keep the code as transparent as possible.  Simple function calls and data structures were used and the code remained close to the standard functions provided in julia.  Obviously some features may be removed for different models or preferences and modification is encouraged.  Other models may be added in the future.  This program only computes the 2D Ising model with periodic boundary conditions, though other extensions are within reach.

Reference for paper are coming soon!

# Licensing

This implementation of GT-GPU is distributed under the 3-clause BSD license (see license file).

# Programming Language

This program was coded with julia v0.6.1.  The "OpenCL" package must be added to julia's library before running.  Additionally, an OpenCL driver must be installed.  This was chosen because an alternative such as CUDA works specifically with NVIDIA graphics cards, so the widest possible use was chosen.  The method can be adapted to a different programming language if necessary.

It is not guaranteed that the test code will work on other machines.  Often, changes required are type conversions that are automatically implemented by one compiler may be missing on another.  So, adding 'Float32' or similar to some variables may be necessary.  Some compilers are not able to understand what is commented in the GPU portion, so this has remained uncommented.

# Test Machines

The algorithm is tested here on a low-level machine with a dual-core Unix-based operating system (MacOS) with a processor speed of 1.3 GHz.   The GPU is the Intel HD Graphics 5000 with 1536 MB of memory. The sampling took about 3 minutes on this machine and half that with double the processing speed.  Yet high quality results were obtained.

With a sufficiently small enough MC samples ($\mathcal{N}$) but arbitrary multiplication factor $\eta$, even this low level machine were easily obtained for reasonable lattice sizes.  No loss of precision occurred for the quantities themselves since they were returned to the CPU and converted to 64 bit floats before any division or similar operation occurred. Also note that setting $\Nloop{}$ higher by itself in the code with constant $\eta$ might give an overflow (especially for $\langle M^4\rangle$) which will show up in strange results.  It is important to keep $\Nloop{}$ low for this reason and increase $\eta$.  There is little overhead for partitioning instead of running all at once, except on the random number generator's side in julia.

# Graphics Processing Unit Limits

Note that the number of threads on a given GPU is not physically limited by the memory on the GPU, differing from CPUs which are a fixed hardware feature.  On low-level the test machine (1536 MB), the upper maximum can be several million.  If such a large number of threads will be used, the code spends an inordinate amount of time pre-generating random numbers.  In the main text, it was showed that this extreme level of precision was not necessary to rival simple implementations of MC.  Some parallel implementation of the random number generation is recommended if the method will be pushed to this limit.  The code sample below does not implement this to show it is not explicitly necessary for what was done here.  For large sizes and ultra-precision, this may become necessary for some studies.

# Random Number Generation

Random number generation requires some careful consideration to ensure that a proper generator is random enough.  The goal of this program was to use as minimal a quality machine as possible to produce the best possible quality results.  For this reason, not much sophistication was applied to the random number generation either in the generator itself or in parallelizing.  By using several CPUs, one could speed up the code by a considerable amount since this represents the bottleneck for the calculations.  This was not implemented here since the concern of this work is for one processor and one GPU only.

Here, random numbers are simply generated on the CPU and buffered into the GPU.  This operation constitutes the largest amount of time for the method but it is not burdensome.  This operation can be parallelized at the cost of more CPUs.

Since not all applications can be anticipated, the user is only cautioned that this is an issue that may appear in some instances and is encouraged to understand fully the intricacies of random number generation before making modifications.

# Implementation Details

Where possible, a 1-byte integer was used to store the information if this is applicable, {\it i.e.} for storing the wavefunction and couplings.  This allows large systems to be run even when the memory on a GPU is small, though it can be upgraded to 4-byte variables for extremely high coordinate number or other issues when calculating $\delta E$ or another quantity.

The initial state is know to be uniformly up at low temperatures, so the convention is adopted that scans started from low temperatures use an initial state of uniformly up spins.  From this configuration, the first 1000 MC iterations are not recorded in the calculation of the ensemble average.  This allows the system to thermalize to an appropriate initial state.  If the system is approached from high temperatures, the wavefunction is randomly assigned $\pm1$ for each spin.  

The energy difference is calculated before flipping a spin.  The energy difference of a single cluster,
$$
\delta E=E_\mathrm{new}-E_\mathrm{old},
$$
can be expressed in terms of only the current spin state for spin-$\frac12$ problems as
$$
\delta E=-2E_\mathrm{old}
$$
since $E_\mathrm{new}=-E_\mathrm{old}$.  Note that since the new energy can be calculated without flipping a spin, it is more advantageous to use. The energy difference then becomes
$$
\delta E=J\sum_{c}\left(S^z_iS^z_j\right)^\mathrm{(old)}
$$
so only a single cluster ($c$) of five spins needs to be calculated so the summation may be restricted to the randomly chosen spin and its four neighbors on the Ising lattice, for example.  Recalculating $\delta E$ in each iteration fast enough in comparison with using a lookup table.  For high coordination number, this might be insufficient and a lookup table may be better.

Note that the left hand side of Eq.~\eqref{flipcondition} is rewritten with a logarithm in the code and can be pre-generated at the beginning, leaving only the energy difference $\delta E$ needs to be generated on the GPU.  The return variables are individual $M$ and $E$ values for each stored ensemble.

\begin{figure}[b]
Ising_Orders.png
\caption{\label{Ising_Orders} The specific heat ($c_V$) and magnetic susceptibility ($\chi_M$) per site and Binder cumulant ($B$) for the Ising model generated with the same parameters as Fig.~\ref{Ising_EM}. Finite size effects prevent $c_V$ and $\chi_M$ from having a singularity.  The peaks in the curves are misaligned due to the level of precision run and the finite size effects.}
\end{figure}

It will look a little different from this:

![The specific heat ($c_V$) and magnetic susceptibility ($\chi_M$) per site and Binder cumulant ($B$) for the Ising model generated with the same parameters as Fig.~\ref{Ising_EM}. Finite size effects prevent $c_V$ and $\chi_M$ from having a singularity.  The peaks in the curves are misaligned due to the level of precision run and the finite size effects.](GeneticTemperingMC/Ising_Orders.png )

In order to implement a new model, at least the calculation of the energy change in the GPU part of the program is necessary, although other parts of the program may require modification (such as a new buffer).

To avoid any lingering unthermalized states, the program selects the thread whose $\overline{x}^{(w)}$ is closest to $\langle x\rangle$ after the first set of delay loops.

# Parameters and Hysteresis

The number of threads $\mathcal{J}$ should be made as large as the memory on the GPU allows since this has very little overhead.

A good guideline for the number of samples in a single loop algorithm is ($\Omega$=lattice sites) $\Omega\ll\eta\mathcal{N}$ ({\it i.e.} the number of MC samples visits each site enough times), but for GT-GPU, it is only necessary that enough samples are thermalized and $\Jloops{}$ is high. A systematic review of parameters is avoided here since each system has a different relaxation time.  The advice to follow is to examine carefully the hysteresis loop.  This conveys all the information about how relaxed the ensemble of samples are.  If necessary, that information can be printed out directly.

In the language of J. Neirotti, D. L. Freeman, and J. Doll, Phys. Rev. E
62, 7445 (2000)., the sampling has not left the correlated-noise region into the desired stochastic-noise region.  This points to the one caution when using GT-GPU: enough samples must be used to ensure that correlated noise is minimized in the final result.  

# Phase Transitions

One way to find a phase transition is to plot one of the quantities of interest and note where $T_c$ is approximately.  This first calculation is done with a low level of precision and accuracy.

After this, another run with increased threads and MC loops is run over a reduced temperature range.  Typically, having an error less than the change in temperature is advised.  It is not important for the hysteresis loop to be accurate, especially when the final run is far more accurate (either by an increase of $\mathcal{J}$ or $\mathcal{N}_\mathrm{GPU}$).
