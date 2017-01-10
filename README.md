PhotALPsConv
============

1. Introduction
---------------

The python scripts included in this package can be used
to compute the oscillation probability of photons into 
axion-like particles in different settings of the ambient
magnetic field.

If you use any of the packages, please give a reference to the 
following papers:

- Horns et al., 2012, Physical Review D, vol. 86, Issue 7
- Meyer et al., 2013, Physical Review D, vol. 87, Issue 3, id. 035027

2. Prerequisites
----------------

The scripts require further packages written available at https://github.com/me-manu/
namely the packages eblstud, gmf, and a running version of the modified
NE2001 code (Cordes & Lazio, 2001), also available at the above repository.
The latter two packages are only required for conversion calculations in the Galactic
magnetic field of the Milky Way.
Download the packages and the add the paths to your PYTHONPATH variable.

You will also need the following python packages:
- numpy
- scipy
- matplotlib
- astropy
- yaml
- iminuit (if you are using the iminuit_fit script, see below)


3. Package contents:
--------------------
- README.md: the file you are currently reading
- __init__.py: init the python packages
- conversion.py:photon-ALP conversions in the intergalactic medium
- conversion_ICM.py: photon-ALP conversions in the intracluster medium
- conversion_GMF.py: photon-ALP conversions in Galactic magnetic field
- conversion_Jet.py: photon-ALP conversions in AGN Jet
- iminuit_fit.py: Power law and log parabola fit of spectrum corrected for ALP effect (Jet/ICM + GMF only so far)
- deltas.py: auxilliary functions to calculate the delta (momentum difference) parameters
- example.py: example script
- yaml/PG1553.yaml: example config file to be run with example.py script

4. Usage
--------
Test the installation with 
> python example.py -c yaml/PG1553.yaml

This calculates and plots the photon survival probability for gamma rays 
originating from the blazar PG1553+113. 
Both the script and the yaml file are extensively commented to explain the usage.

5. License
----------
PhotALPsConv is distributed under the modified BSD License.

Redistribution and use in source and binary forms, with or without
modification, are permitted provided that the following conditions are met:
- Redistributions of source code must retain the above copyright
notice, this list of conditions and the following disclaimer.
- Redistributions in binary form must reproduce the above copyright
notice, this list of conditions and the following disclaimer in the
documentation and/or other materials provided with the distribution.
- Neither the name of the PhotALPsConv developers  nor the
names of its contributors may be used to endorse or promote products
derived from this software without specific prior written permission.

THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND
ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED
WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
DISCLAIMED. IN NO EVENT SHALL THE PHOTALPSCONV DEVELOPERS BE LIABLE FOR ANY
DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES
(INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES;
LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND
ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
(INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS
SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
