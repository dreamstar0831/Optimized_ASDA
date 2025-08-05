## Changelog

**August 2025: v1.0 released.**

This README file is a modification of README of ASDA (Jiajia Liu, https://github.com/PyDL/ASDA).

Based on the findings in our recent work, which is published at A&A (https://doi.org/10.1051/0004-6361/202554524), we improve the original ASDA code and release it as a new repository, named Optimized ASDA.

Compared to ASDA, main improvements of Optimized ASDA are as follows:

- Revise the threshold of Γ1 from 0.89 to 0.63 (2/π)
- Modify kernel size used to calculate Γ1 and Γ2: apply the combination of kernel sizes of 5, 7, 9, and 11.

## Cite:
Liu, J., Nelson, C, Erdelyi, R, Automated Swirl Detection Algorithm (ASDA) and Its Application to Simulation and Observational Data, ApJ, 872, 22, 2019 (https://iopscience.iop.org/article/10.3847/1538-4357/aabd34/meta)

Xie, Q, Liu, J., Erdelyi, R, Wang, Y., Improving Γ-functions method for vortex identification, A&A, 700, A6, 2025                （https://doi.org/10.1051/0004-6361/202554524）

## Please contact the authors and ask for permissions before using any part of these code.
Email: xq30@mail.ustc.edu.cn or ljj128@ustc.edu.cn

# Optimzied ASDA
Optimized Automated Swirl Detection Algorithms

## System Requirements
### OS Requirements
Optimized ASDA can be run on Windows, Mac OSX or Linux systems with Python 3 and the following dependencies installed.

### Dependencies:
**Python 3** with libraries including numpy, scipy, getopt, sys, matplotlib, random, os, subprocess, multiprocessing, scikit-image, mpi4py</br>
**pyflct**: https://github.com/PyDL/pyflct </br>
**Python management softwares including Anaconda or Virtualenv are recommended**

## Hardware Requirements:
Optimized ASDA requires a standard computer with enough CPU and computation power depending on the dataset used. To be able to use the parallel functions, a multi-core CPU supporting the MPI libraries will be needed.

## Installation Guide:
Install Anaconda (https://www.anaconda.com/download)

Create conda environment and install dependencies. We provide an example environment.yml.

```
conda env create -f environment.yml -n op_asda
```

The above commend will create a conda environment named as *op_asda*. Activate it by typing in

```
source activate op_asda
```

run the following codes in your terminal to install ASDA:
```bash
git clone https://github.com/dreamstar0831/Optimized_ASDA
cd Optimized_ASDA
pip install .
```
Optimized ASDA will be then installed in your default Python environment.

## Description of Files (More information can be found in each file):
**vortex.py**: Main programm of the implication of the optimized swirl detection algorithms</br>
**gamma_values_mpi.py**: MPI version of the vortex.gamma_values() function</br>
**points_in_poly.py**: return all points within a polygon</br>
**swirl_lifetime.py**: using different methods to give labels to all swirls in order to estimate their lifetimes</br>
**lamb_oseen.py**: Object of a lamb_oseen vortex</br>
**test_synthetic.py**: Main program generating and testing a series of synthetic data (see reference)</br>
**correct.npz**: correct swirl detection result using Optimized ASDA</br>
**setup.py**: setup file used for pip</br>

## Instructions for Use:
You can also find the following steps from line 249 in `test_synthetic.py`.
Suppose you have two succesive 2d images in **(x, y)** order: data0 and data1</br>
1. import neccessary libraries, including:
```python
from optimized_asda.pyflct import flct, vcimagein
from optimized_asda.vortex import gamma_values, center_edge, vortex_property
```
2. you need to use the pyflct package to estimate the velocity field connecting the above two images: 
`vx, vy, vm = flct(data0, data1, 1.0, 1.0, 10, outfile='vel.dat')`. Please notice that, vx, vy and vm are also in **(x, y)** order. Here, vx and vy are the velocity field. Usually, vm are not necessary.</br>
1. calculate gamma1 and gamma2 values (see the reference) with `gamma1, gamma2 = gamma_values(vx, vy, 'adaptive', factor=1)`, using the variable gamma calculating method. You can calculate gamma1 and gamma2 using a single kernel size (e.g., r=3) with `gamma1, gamma2 = gamma_values(vx, vy, 3, factor=1)`. </br>
2. perform the detection of vortices using `center, edge, point, peak, radius = center_edge(gamma1, gamma2, factor=1)`. Results are from the threshold of Γ1 = 0.63. center is a list containing the pixel location of all vortices in the image. edge is a list of the edges of all vortices. point is a list of all points within vortices. peak is a list of the peak gamma1 values of all vortices. radius is a list of the effective radii of all vortices.</br>
3. use `ve, vr, vc, ia = vortex_property(center, edge, points, vx, vy, data0)` to calculate the expanding, rotating, center speeds of above vortices. ia is the average intensity from data0 of all points within each vortex.</br>
4. **Notice**: radius, ve, vr and vc calculated above are in units of 1. Suppose for data0 and data1, the pixel size is *ds* (in units of actual physical units such as Mm, km, m...) and the time difference of *dt* (in units of second, minute, hour...), then you should use `radius * ds` and `ve * ds / dt`, `vr * ds / dt`, `vc * ds / dt` as your final results.

## Demo
A demo **demo.py** is available with the demo data **demo_data.sav**:
1. To run the demo, `cd Optimized_ASDA` and run `python demo.py`
2. The demo data consists of the following 4 variables: data0 (SOT Ca II observation at 2007-03-05T05:48:06.737), data1 (SOT Ca II observation at 2007-03-05T05:48:13.138), ds (pixel size of the observations), and dt (time difference in seconds between data0 and data1)
### Expected Output
After running the code, you will see 3 files as the output: **vel_demo.dat** (binary file storing the calculated velocity field, 6MB), **gamma_demo.dat** (binary file storing gamma1 and gamma2, 4MB), and **vortex_demo.npz** (numpy file storing the information of all deteced vortices, 1 MB). These are results of Optimized ASDA, different to those of ASDA. All the differences printed out should be 0.</br>
</br>
Use `vortex = dict(np.load('vortex_demo.npz'))`, you should see the variable `vortex` stores the center, edge, points, peak (gamma1 value), radius, ve (average expanding/shrinking velocity), vr (average rotating speed), vc (speed of center), and ia (average observation intensity) for **303** detected swirls, more than the number **52** detected by ASDA. You can compare these results with the correct detection result stored in **correct.npz**.

### Expected Running Time
Once finished, the command line will give the time consumption of the demo code, which should be ~50 seconds on an Apple M4.

