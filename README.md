# CP4 (T) correction to CCSD

A Method to approximate the (T) correction in CCSD(T) by using a CP4 form of the ERI, attained by approximating the ISDF-ERI.

## Features
- Approximation of the ISDF-ERI by a CP4
- Computation of the (T) correction in O(N^5) time
- Performance depends on the internal size of the CP4 approximation

## Workflow
- Choose System and basis set
- Perform ISDF using Coqui
- Approximate ISDF THC Integral by CP4 using e.g. ALS
- Calculate (T) correction






