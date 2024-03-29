# PyWBGT_analytic
The model developed by Liljegren et al (2008) [1] is the recommended approach for esmating wet-bulb globe temperature (WBGT) [2], but it requires iterative calculation from solving the nonlinear enegy balance equations of the wet wick and black globe sensors. By carefully exmaining self-nonlinearities in Liljegren's model, we develop an analytic WBGT approximation that doesn't require iteration while maintaining high accuracy and most of the physics of Liljegren's model.

****
### What is in this repository?
- `WBGT_analytic.py`: Python script for calculating the analytic approximation of WBGT.
- `coszenith.pyx`: Cython code for calculating cosine zenith angle as develped by Kong and Huber (2022) [3]
- `setupcoszenith.py`: Python setup tool script for compiling `coszenith.pyx`
- `Calculate_analytic_WBGT_with_CMIP6_data.ipynb`: A jupyter nobtebook introducing the usage of our code.
- `environment.yml`: a YAML file that can be used to build conda environment containing all needed python packages.

****
### How to compile Cython code for calculating cosine zenith angle
`coszenith.pyx` need to be compiled first to obtain a shared object file `.so` that can be import in Python. The setup tool script `setupcoszenith.py` is used for such purpose:
- for Intel compiler: 
  - `LDSHARED="icc -shared" CC=icc python setupcoszenith.py develop`
- for gcc compiler:
  - replace '-qopenmp' with '-fopenmp' in the `setupcoszenith.py` file.
  - `python setupcoszenith.py build_ext --inplace`
  

****
### Citation
If you want to use our code, please consider cite `upcoming`

****
### References

[1] Liljegren JC, Carhart RA, Lawday P, Tschopp S, Sharp R. Modeling the wet bulb globe temperature using standard meteorological measurements. J Occup Environ Hyg. 2008;5(10):645-55. 

[2] Lemke B, Kjellstrom T. Calculating workplace WBGT from meteorological data: a tool for climate change assessment. Ind Health. 2012;50(4):267-78. 

[3] Kong, Q. & Huber, M. Explicit calculations of Wet Bulb Globe Temperature compared with approximations and why it matters for labor productivity. Earth’s Future (2022) doi:10.1029/2021EF002334.

