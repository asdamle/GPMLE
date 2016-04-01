README
=====================

This code implements the methodology presented in

Victor Minden, Anil Damle, Kenneth L. Ho, and Lexing Ying "Fast spatial Gaussian process maximum likelihood estimation via skeletonization factorizations," [arXiv:1603.08057](http://arxiv.org/abs/1603.08057)

for maximum likelihood estimation for parameter-fitting given observations from a kernelized Gaussian process in two spatial dimensions. This code is not being actively developed, nor supported as a package.

The file example.m contains a commented demonstrative example using the included routines to perform MLE on some synthetically generated data.

**Dependencies**
This code depends on the [FLAM library](https://github.com/klho/FLAM)

**Acknowledgments**
The ocean data used in some examples is ICOADS, citation below:

National Climatic Data Center/NESDIS/NOAA/U.S. Department of Commerce, Data Support Section/Computational and Information Systems Laboratory/National Center for Atmospheric Research/University Corporation for Atmospheric Research, Earth System Research Laboratory/NOAA/U.S. Department of Commerce, and Cooperative Institute for Research in Environmental Sciences/University of Colorado (1984): International Comprehensive Ocean-Atmosphere Data Set (ICOADS) Release 2.5, Individual Observations. Research Data Archive at the National Center for Atmospheric Research, Computational and Information Systems Laboratory. Dataset. http://dx.doi.org/10.5065/D6H70CSV. Accessed 11 Nov 2015.