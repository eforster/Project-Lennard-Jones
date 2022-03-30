# Lennard Jones Project
### Author : E Forster, s1639706
***

This is a project written as a final assessment for Computer Modelling.

The main goals are to explore N particles undergoing Lennard Jones interactions to analyse data for experimental simulation.
More specifically, exploring the difference phases of Argon using periodic boundary conditions.

## Usage
***
Download project_s1639706.zip file, extract all and open in desired IDE (I used PyCharm Community Edition 2021.2.2)

To run the program, the main file is verlet.py and you need input_file as an argument/parameter in any Run/Debug feature of your IDE. 

You can run the program as follows in IDE terminal :

``` python
# Go to file in your computer
C:\Users\path\to\the\file> cd project_s1639706
# Then input the following :
python verlet.py input_file
```

## File Contents
***
These are the contents of the project file:
- particle3D.py is the Particle3D class
- verlet.py is the main file which is to be run
- input_file is the input file that may be edited for user input
- output_file.xyz is the trajectory file for visualising particle movement
- energy_file is the output file for the system energies
- msd_file is the output file for the mean squared displacement function
- rdf_file is the output file for the radial distribution function
- mdutilities.py contains the functions to simulate an FCC cell, 
the authors being instructors Miguel Martinez-Canales and Joe Zuntz of the Computer Modelling course
- pydoc documentation for particle3D and verlet files and this README

## User Input & Output
***
There is a file called input_file, wherein you can find where you can input initial conditions for running a simulation.
First two lines specify dt, numstep and the number of particles for simulation.
Next two lines specify initial conditions in rho and temperature.
The non # lines are to be edited for user convenience.

Lastly, the names of desired output files for plotting the following for later analysis :

- xyz trajectory, giving particle name and positions in x, y and z
- energies, giving the kinetic energy, potential energy and total energy with time
- mean square displacement with time
- radial distribution function, with time

### Units
***
There is only one single type of atom and units are defined such that it makes the Lennard Jones potential simpler:
- Distance is in units of sigma
- Energy is in units of epsilon
- Mass of the atom being used is 1
- Time being sigma * math.sqrt(1 / epsilon)

## Testing with Initial Conditions
***
The following are tried and tested initial conditions and give a good indication of what to expect from this simulation.

- Solid Argon: N = 32 particles, rho = 1.0, T = 0.1, dt 0.01, numstep = 1000

- Gaseous Argon: N = 30 particles, rho = 0.05, T = 1.0, dt = 0.005, numstep = 2000

