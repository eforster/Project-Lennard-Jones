"""
Velocity Verlet Time Integrator for simulations of N particles undergoing Lennard Jones interactions using periodic boundary conditions.

Produces plots of the particle's total energy, kinetic energy, potential energy, mean squared displacement and radial
distribution function, all as functions of time.  This is saved to output files for user convenience and further analysis.

Included methods for calculating :
- periodic boundary conditions and minimum image conventions
- pair separations between particle pairs
- Lennard Jones Force
- Lennard Jones Potential
- Mean Squared Displacement
- Radial Distribution Function

Note : capital R indicates a vector R whereas lower-case r would indicate the modulus of vector R

Author : E Forster, s1639706
"""

import copy
import sys
import math
import numpy as np
import matplotlib.pyplot as pyplot
from particle3D import Particle3D
import mdutilities as md

def minimum_image_convention(particle, different_particle, box_size) :

    """
    Minimum Image Convention chooses the nearest neighbour of all the surrounding identical images of each particle :

    MIC(R_12) = ((R_12 + L/2) mod L) - L / 2

    where :
    - R_12 = R_1 - R_2, the difference between two different particle positions

    :param particle: a Particle3D instance
    :param different_particle: a different Particle3D instance
    :param box_size: simulation box of size L, vector (L, L, L)

    :return mic: nearest neighbour location
    """

    mic = np.mod((particle.position - different_particle.position + box_size / 2), box_size) - (box_size / 2)
    return mic

def calculate_pair_separation(particle_list, box_size) :

    """
    Calculates all separations between each pair of particles, implementing MIC

    :param particle_list: list of Particle3D instances
    :param box_size: simulation box of size L, vector (L, L, L)

    :return separations_matrix: a matrix of separations between each pair of particles
    """

    N = len(particle_list)
    separations_matrix = np.zeros((N, N, 3))

    for i in range(N) :
        for j in range(i + 1, N) :
            
            separation = minimum_image_convention(particle_list[i], particle_list[j], box_size)

            separations_matrix[i, j] = separation
            separations_matrix[j, i] = - separation     # Applies symmetry to reduce computing time

    return separations_matrix

def periodic_boundary_conditions(particle, box_size) :

    """
    Periodic Boundary Conditions makes each particle in the simulation box has a mirror image in every other box :

    PBC(X) = X mod L

    where :
    - X = the vector representation of a particle position

    :param particle: a Particle3D instance
    :param box_size: simulation box of size L, vector (L, L, L)

    :return pbc: makes sure every moving particle stays in the simulation box
    """

    pbc = np.mod(particle.position, box_size)
    return pbc

def lennard_jones_force(particle_list, box_size, cut_off_radius) :

    """
    Computes the Lennard Jones Force using this equation :

    F_i = - 48 * summation_j [r ** (- 14) - 0.5 * R ** (- 8)] (R_i - R_j)

    where :
    - summation_j = large sigma, meaning to sum over all j particles
    - r = the modulus of R_i - R_j

    :param particle_list: list of Particle3D instances
    :param box_size: simulation box of size L, vector (L, L, L)
    :param cut_off_radius: allows for setting forces to zero beyond this radius

    :return lj_force_matrix: a matrix array of the force on a particle, as the sum of the force on it from all other particles
    """

    N = len(particle_list)
    sep_matrix = calculate_pair_separation(particle_list, box_size)
    
    lj_force_matrix = np.zeros((N, N, 3))

    for i in range(N) :
        for j in range(i + 1, N) :

            modulus_sep_matrix = np.linalg.norm(sep_matrix[i, j])

            if modulus_sep_matrix > cut_off_radius or modulus_sep_matrix == 0 :         # Applies cut off radius to reduce computing time

                lj_force_matrix[i, j] = 0

            else :
                lj_force_matrix[i, j] = 48 * (modulus_sep_matrix ** (-14) - 0.5 * modulus_sep_matrix ** (-8)) * sep_matrix[i, j]
                lj_force_matrix[j, i] = - lj_force_matrix[i, j]                 # Applies Newton's 3rd Law to reduce computing time

    return lj_force_matrix

def lennard_jones_potential(particle_list, box_size, cut_off_radius, sep_matrix) :

    """
    Computes the Lennard Jones Potential from this equation :

    U = summation_i * summation_j_greater_than_i * 4 * [r ** (- 12) - r ** (-6)]

    where :
    - summation_i = large sigma, meaning to sum over all i particles
    - summation_j_greater_than_i = large sigma, meaning to sum over all j > i particles
    - r = modulus of R_i - R_j

    :param sep_matrix: a matrix of separations between each pair of particles
    :param cut_off_radius: allows for setting the potential for separation > cut_off_radius to be the potential calculated from this radius
    :param particle_list: list of Particle3D instances
    :param box_size: simulation box of size L, vector (L, L, L)

    :return lj_potential: the Lennard Jones potential of particles
    """

    N = len(particle_list)
    sep_matrix = calculate_pair_separation(particle_list, box_size)
    lj_potential = 0

    for i in range (N) :
        for j in range (i + 1, N) :
            
            modulus_sep_matrix = np.linalg.norm(sep_matrix[i, j])

            if modulus_sep_matrix > cut_off_radius :                # Applies cut off radius to reduce computing time

                lj_potential += 4 * (cut_off_radius ** (- 12) - cut_off_radius ** (- 6))

            else :

                lj_potential += 4 * (modulus_sep_matrix ** (- 12) - modulus_sep_matrix ** (- 6))

    return lj_potential

def mean_squared_displacement(particle_list, initial_particle_list, time, box_size) :

    """
    Calculates the Mean Squared Displacement, MSD, a measure of how far particles have moved on average from their initial position at some time, t,
    whilst still obeying minimum image convention :

    MSD(t) = 1 / N * summation_i * (magnitude(R_i(t) - R_i(0)) ** 2)

    where :
    - N = number of particles
    - summation_i = large sigma, meaning to sum over all i particles
    - R_i(t) = position vector for particles i at time, t
    - R_i(0) = initial position of i particles

    :param particle_list: list of Particle3D instances
    :param initial_particle_list: list of Particle3D instances before any calculation manipulation
    :param time:
    :param box_size: simulation box of size L, vector (L, L, L)

    :return msd: the Mean Squared Displacement of particles
    """

    N = len(particle_list)
    msd = 0

    for i in range(N) :

        mic_msd = minimum_image_convention(particle_list[i], initial_particle_list[i], box_size)

        msd += (1 / N) * np.linalg.norm(mic_msd) ** 2

    return msd

def radial_distribution_function(particle_list, box_size, rho, separations_matrix) :

    """
    Calculates the Radial Distribution Function, RDF, a measure of the probability to find a particle at a given distance from a reference particle :

    g(r) = (1 / N * rho_nought(r)) * summation_ij (dirac_delta(r_ij - r))

    where :
    - g(r) = the RDF
    - N = number of particles
    - rho_nought(r) = representation of expected value for RDF for a perfectly homogeneous system = 4 * pi * rho_star * r ** 2 * dr
    - summation_ij = big sigma, meaning to sum over all particles i and j
    - dirac_delta = describes positions in this function

    Given the magic of Python and numpy, using the magnitudes of the pair separations matrix, numpy histogram was used to obtain the rdf and positions for later plotting

    :param particle_list: list of Particle3D instances
    :param box_size: simulation box of size L, vector (L, L, L)
    :param rho: particle density
    :param separations_matrix: a matrix of separations between each pair of particles

    :return normalised rdf: the y-axis of the RDF
    :return binned_r: the x-axis of the RDF, known as binned r (position) values
    """

    separation_matrix = calculate_pair_separation(particle_list, box_size)
    N = len(particle_list)
    mag_separations = []

    for i in range(N) :
        for j in range(i + 1, N) :

            modulus_separation_matrix = np.linalg.norm(separation_matrix, axis = 2)     # Obtains required magnitudes for the histogram

            if modulus_separation_matrix[i, j] > 0 :                        # Narrowing down to only non-zero magnitudes

                mag_separations.append(modulus_separation_matrix[i, j])

    bin_size = 0.05
    bins = 100

    rdf_histogram = np.histogram(mag_separations, bins = bins, range = (0, box_size))
    rdf = rdf_histogram[0]
    start_binned_r = rdf_histogram[1]

    binned_r = start_binned_r[1:]       # First element needs to be sliced out to avoid division by zero later

    dr = box_size / bins
    rho_nought = 4 * math.pi * rho * dr * binned_r ** 2

    normalised_rdf = rdf / (N * rho_nought)       # RDF needs to be normalised to account for the more space for particles further out from the bulk

    return normalised_rdf, binned_r

# Begin main code
def main() :

    """
    The main method carries out the simulation in a few parts :

    1.) Reads in data file from the command line and input file
    2.) Specifies initial conditions and initialises data lists for plotting later
    3.) Starts a time integration loop
    4.) Plots the system total energy to screen
    5.) Plots the system kinetic energy to screen
    6.) Plots the system potential energy to screen
    7.) Plots the mean squared displacement to screen
    8.) Plots the Radial Distribution Function to screen
    9.) Measures the energy inaccuracy of the simulation and prints it to the terminal
    """

    # Part 1.) Reads in data file from the command line and input file

    with open(sys.argv[1], "r") as infile :

        # Read name of files from command line which needs 2 parts
        if len(sys.argv) != 2 :

            # Helpful error message if the format is incorrect
            print("Wrong number of arguments.")
            print("Usage: " + sys.argv[0] + "<input file>")
            quit()

        else :

            line1 = infile.readline()                   # Processes line 1 of input file
            line2 = infile.readline()                   # Processes line 2 of input file
            line2 = line2.split()                       # Separates the parameters in line 2

            # Helpful error message if there is not 2 parameters in line 2
            if len(line2) != 3 :
                print("Wrong number of arguments in line 2 of input data file, i.e. dt, numstep and number of particles. ")

            else:
                dt = float(line2[0])                # Reads in time-step for simulation
                numstep = int(line2[1])             # Reads in number of steps for simulation
                number_particles = int(line2[2])

            line3 = infile.readline()           # Processes line 3 of input file
            line4 = infile.readline()           # Processes line 4 of input file
            line4 = line4.split()               # Separates the parameters in line 4

            # Helpful error message if there is not 2 parameters in line 2
            if len(line4) != 2 :
                print("Wrong number of arguments in line 2 of input data file, i.e. temperature and rho.")

            else :
                temperature = float(line4[0])   # Reads in temperature for simulation
                rho = float(line4[1])           # Reads in density for simulation

            line5 = infile.readline()           # Processes line 5 of input file
            line6 = infile.readline()           # Processes line 6 of input file
            line6 = line6.split()               # Separates the parameters in line 6

            # Helpful error message if there is not 4 parameters in line 6
            if len(line6) != 4 :
                print("Wrong number of arguments in line 6 of input data file, i.e. trajectory file, energy file, msd file and rdf file. ")

            else :
                xyz_trajectory = line6[0]       # Reads in xyz_trajectory output file
                energies_file = line6[1]        # Reads in energies output file
                msd_file = line6[2]             # Reads in msd output file
                rdf_file = line6[3]             # Reads in rdf output file

                # Open output file
                outfile1 = open(xyz_trajectory, "w")
                outfile2 = open(energies_file, "w")
                outfile3 = open(msd_file, "w")
                outfile4 = open(rdf_file, "w")

                # Writes out the format for output files
                outfile2.write("Time, Kinetic Energy, Potential Energy, Total Energy\n")
                outfile3.write("Time, MSD\n")
                outfile4.write("Position, RDF(Position)\n")

    infile.close()

    # Part 2.) Specifies initial conditions and initialises data lists for plotting later

    cut_off_radius = 3.5
    time = 0.0
    msd = 0
    bins = 100

    rdf_array = np.zeros(bins)
    particle_list = []
    msd_list = []

    N = len(particle_list)

    # Initialising particle list to have the desired number of particles
    for particle in range(number_particles) :
        particle_list.append(Particle3D(label = f"particle_{particle}", mass = 1, position = np.zeros(3), velocity = np.zeros(3)))

    # Uses mdutilities file to give initial positions and velocities to all the particles in particle list, and gives an appropriate box size to be used
    box_size, full_lattice = md.set_initial_positions(rho, particle_list)
    box_size = box_size[0]
    md.set_initial_velocities(temperature, particle_list)
    separation_matrix = calculate_pair_separation(particle_list, box_size)

    # Makes a copy of initial particle list data before any calculation or manipulation
    initial_particle_list = copy.deepcopy(particle_list)

    # Calculates the initial MSD
    msd = mean_squared_displacement(particle_list, initial_particle_list, time, box_size)
    msd_list.append(msd)
    outfile3.write(f"{time}, {msd} \n")

    # Calculates the initial system energies and forces
    for n in range(number_particles) :
        kinetic_energy = particle_list[n].calculate_system_kinetic_energy(particle_list)
        potential_energy = lennard_jones_potential(particle_list, box_size, cut_off_radius, separation_matrix[n])
        force_matrix = lennard_jones_force(particle_list, box_size, cut_off_radius)
        total_energy = kinetic_energy + potential_energy

    # Writes initial system energies to file
    outfile2.write(f"{time}, {kinetic_energy}, {potential_energy}, {total_energy}\n")

    # Initialises lists for plotting later
    time_list = [time]
    potential_energy_list = [potential_energy]
    kinetic_energy_list = [kinetic_energy]
    total_energy_list = [total_energy]

    # Part 3.) Starts a time integration loop

    for i in range(numstep) :

        # Starts the formatting for the xyz_trajectory file
        outfile1.write(f"{number_particles} \n")
        outfile1.write(f"Point = {i} \n")

        # Updates all the particle positions and writes to file
        for n in range(len(particle_list)) :

            # Slices the force matrix so the corresponding forces and particles are matched up
            particle_list[n].update_2nd_position(dt, np.sum(force_matrix[:, n], axis = 0) * (-1))
            particle_list[n].position = periodic_boundary_conditions(particle_list[n], box_size)

            outfile1.write(f"{str(particle_list[n])}")

        # Updates the forces
        new_force_matrix = lennard_jones_force(particle_list, box_size, cut_off_radius)

        # Updates all the particle velocities by averaging current and new forces
        for k in range(len(particle_list)) :

            particle_list[k].update_velocity(dt, (-0.5) * ((np.sum(force_matrix[:, k], axis = 0)) + (np.sum(new_force_matrix[:, k], axis = 0))))

        # Re-define force value
        force_matrix = new_force_matrix

        # Increases time
        time += dt

        # Calculates the new kinetic energy for the system
        for m in range(len(particle_list)) :

            kinetic_energy = particle_list[m].calculate_system_kinetic_energy(particle_list)

        # Calculates the next potential and total energies and writes to file
        potential_energy = lennard_jones_potential(particle_list, box_size, cut_off_radius, separation_matrix)
        total_energy = kinetic_energy + potential_energy
        outfile2.write(f"{time}, {kinetic_energy}, {potential_energy}, {total_energy}\n")

        # Calculates the updated msd and write to file
        msd = (mean_squared_displacement(particle_list, initial_particle_list, time, box_size))
        outfile3.write(f"{time}, {msd} \n")
        msd_list.append(msd)

        # Calculates data required for the RDF function
        normalised_rdf, binned_r = radial_distribution_function(particle_list, box_size, rho, separation_matrix)
        rdf_array += normalised_rdf

        # Appends lists for plotting later
        kinetic_energy_list.append(kinetic_energy)
        potential_energy_list.append(potential_energy)
        total_energy_list.append(total_energy)
        time_list.append(time)

    # Calculates the average rdf
    average_rdf = rdf_array / numstep

    # Writes the data for the rdf to file
    for p in range(len(average_rdf)) :

        outfile4.write(f" {binned_r[p]} {average_rdf[p]} \n")
     
    # Post-simulation:
    # Close output file
    outfile1.close()
    outfile2.close()
    outfile3.close()
    outfile4.close()

    # Part 4.) Plots the system total energy to screen

    pyplot.title('Total Energy vs Time')
    pyplot.xlabel('Time : ')
    pyplot.ylabel('Energy : ')
    pyplot.plot(time_list, total_energy_list)
    pyplot.show()

    # Part 5.) Plots the system kinetic energy to screen

    pyplot.title('Kinetic Energy vs Time')
    pyplot.xlabel('Time : ')
    pyplot.ylabel('Kinetic Energy : ')
    pyplot.plot(time_list, kinetic_energy_list)
    pyplot.show()

    # Part 6.) Plots the system potential energy to screen

    pyplot.title('Potential Energy vs Time')
    pyplot.xlabel('Time : ')
    pyplot.ylabel('Potential Energy : ')
    pyplot.plot(time_list, potential_energy_list)
    pyplot.show()

    # Part 7.) Plots the mean squared displacement to screen

    pyplot.title('MSD vs Time')
    pyplot.xlabel('Time : ')
    pyplot.ylabel('MSD : ')
    pyplot.plot(time_list, msd_list)
    pyplot.show()

    # Part 8.) Plots the Radial Distribution Function to screen

    pyplot.title('RDF vs r')
    pyplot.xlabel('r : ')
    pyplot.ylabel('RDF : ')
    pyplot.plot(binned_r, average_rdf)
    pyplot.show()

    # Part 9.) Measures the energy inaccuracy of the simulation and prints it to the terminal

    initial_energy_k = kinetic_energy_list[0]
    initial_energy_p = potential_energy_list[0]

    initial_energy = initial_energy_p + initial_energy_k

    max_energy = max(total_energy_list)
    min_energy = min(total_energy_list)

    delta_energy = max_energy - min_energy
    energy_inaccuracy = delta_energy / initial_energy

    print("Energy inaccuracy : +/-", energy_inaccuracy)

# Execute main method, but only when directly invoked
if __name__ == "__main__" :
    main()
