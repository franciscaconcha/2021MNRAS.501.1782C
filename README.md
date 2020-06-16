# Effects of stellar density on the photoevaporation of circumstellar discs [arXiv:2006.07378]
#### Francisca Concha-Ramírez, Martijn J. C. Wilhelm, Simon Portegies Zwart, Sierk E. van Terwisga, Alvaro Hacar
<!-- [![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.3537675.svg)](https://doi.org/10.5281/zenodo.3537675) [![License: GPL v3](https://img.shields.io/badge/License-GPL%20v3-blue.svg)](https://www.gnu.org/licenses/gpl-3.0) ![python](https://img.shields.io/badge/python-2.7-yellow.svg)-->

(Code to run the simulations and generate figures of the paper: [Effects of stellar density on the photoevaporation of circumstellar discs](https://arxiv.org/abs/2006.07378).)

For questions please contact Francisca Concha-Ramírez, fconcha at strw.leidenuniv.nl

## Getting Started

### Prerequisites
* Python 2.7. Should work fine with Python 3 but it has not been tested.
* AMUSE: https://amusecode.github.io
* vader: https://bitbucket.org/krumholz/vader/src
* scipy

### Running the simulations

The main file to run the simulations is ```vader_cluster_parallel.py``` located in the ```src``` folder. You can run an individual simulation by using the AMUSE script directly from the home directory:

```
amuse.sh src/vader_cluster_parallel.py
```

The script has extensive options which can be passed through the command line, for number of stars, radius of the cluster, etc. For a list of these options run:

```
amuse.sh src/vader_cluster_parallel.py --help
```

### Creating the plots

There are several different scripts to create the Figures of the paper. All of them are located in the ```plots``` folder:

* Figure 1: ```tracks.py```
* Figures 2 and 3: ```disk_fractions.py```
* Figure 4: ```radiation.py```
* Figures 5 and 6: ```binned.py```
* Figures 7 and 8: ```cumulative_distributions.py```
* Figures 9 and 10: ```scatter_plots.py```

Each script can be run with

```
amuse.sh plots/<script>.py
```

A list of options is available for each script, including the path to the files that you want to use for the plots. To see the list of options add ```--help``` or ```-h``` to the line above.


## License

This project is licensed under the GPL v3 License - see the [LICENSE.md](LICENSE.md) file for details
