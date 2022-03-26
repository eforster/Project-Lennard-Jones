"""
Velocity Verlet Time Integrator for simulations of N particles.

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

    :param particle:
    :param different_particle:
    :param box_size:
    :return:
    """

    mic = np.mod((particle.position - different_particle.position + box_size / 2), box_size) - (box_size / 2)
    return mic

def calculate_pair_separation(particle_list, box_size):
    """

    :param particle_list:
    :param box_size:
    :return:
    """

    N = len(particle_list)
    separations_matrix = np.zeros((N, N, 3))

    for i in range(N) :
        for j in range(i + 1, N) :
            
            separation = minimum_image_convention(particle_list[i], particle_list[j], box_size)

            separations_matrix[i, j] = separation
            separations_matrix[j, i] = - separation

    return separations_matrix

def periodic_boundary_conditions(particle, box_size) :
    """

    :param particle:
    :param position:
    :param box_size:
    :param number_particles:
    :return:
    """

    pbc = np.mod(particle.position, box_size)
    return pbc

def lennard_jones_force(particle_list, box_size, cut_off_radius) :
    """

    :param sep_matrix:
    :param particle_list:
    :param box_size:
    :param cut_off_radius:
    :return:
    """

    N = len(particle_list)
    sep_matrix = calculate_pair_separation(particle_list, box_size)
    
    lj_force_matrix = np.zeros((N, N, 3))

    for i in range(N) :
        for j in range(i + 1, N) :

            modulus_sep_matrix = np.linalg.norm(sep_matrix[i, j])

            if modulus_sep_matrix > cut_off_radius or modulus_sep_matrix == 0 :

                lj_force_matrix[i, j] = 0

            else :
                lj_force_matrix[i, j] = 48 * (modulus_sep_matrix ** (-14) - 0.5 * modulus_sep_matrix ** (-8)) * sep_matrix[i, j]
                lj_force_matrix[j, i] = - lj_force_matrix[i, j]

    return lj_force_matrix
    

def lennard_jones_potential(particle_list, box_size, cut_off_radius, sep_matrix) :

    """


    :param sep_matrix:
    :param pair_sep:
    :param cut_off_radius:
    :param particle_list:
    :param box_size:

    :return:
    """

    N = len(particle_list)
    sep_matrix = calculate_pair_separation(particle_list, box_size)
    lj_potential = 0

    for i in range (N) :
        for j in range (i + 1, N) :
            
            modulus_sep_matrix = np.linalg.norm(sep_matrix[i, j])

            if modulus_sep_matrix > cut_off_radius :

                lj_potential += 4 * (cut_off_radius ** (- 12) - cut_off_radius ** (- 6))

            else :

                lj_potential += 4 * (modulus_sep_matrix ** (- 12) - modulus_sep_matrix ** (- 6))


    return lj_potential

def mean_squared_displacement(particle_list, initial_particle_list, time, box_size) :

    N = len(particle_list)
    msd = 0

    for i in range(N) :

        mic_msd = minimum_image_convention(particle_list[i], initial_particle_list[i], box_size)

        msd += np.linalg.norm((1 / N) * np.linalg.norm(mic_msd ** 2))

    return msd

# Begin main code
def main() :
    """
    The main method carries out the simulation in a few parts:

    1.) Reads in data file from the command line
    2.) Specifies initial conditions
    3.) Initialises data lists for plotting later
    4.) Starts a time integration loop
    5.) Plots particle trajectory to screen
    6.) Plots particle energy to screen
    7.) Measures the energy inaccuracy of the simulation and prints it to the screen

    """

    with open(sys.argv[1], "r") as infile :

        # Part 1.) Reads in data file from the command line

        # Read name of files from command line which needs 3 parts
        if len(sys.argv) != 2 :

            # Helpful error message if the format is incorrect
            print("Wrong number of arguments.")
            print("Usage: " + sys.argv[0] + "<input file>")
            quit()

        else :

            line1 = infile.readline()  # Processes line 1 of input file
            line2 = infile.readline()  # Processes line 2 of input file
            line2 = line2.split()  # Separates the parameters in line 2

            # Helpful error message if there is not 2 parameters in line 2
            if len(line2) != 3:
                print("Wrong number of arguments in line 2 of input data file, i.e. dt, numstep and number of particles. ")

            else:
                dt = float(line2[0])  # Reads in time-step for simulation
                numstep = int(line2[1])  # Reads in number of steps for simulation
                number_particles = int(line2[2])

            line3 = infile.readline()
            line4 = infile.readline()
            line4 = line4.split()

            if len(line4) != 2 :
                print("Wrong number of arguments in line 2 of input data file, i.e. temperature and rho.")

            else :
                temperature = float(line4[0])
                rho = float(line4[1])

            line5 = infile.readline()
            line6 = infile.readline()
            line6 = line6.split()

            if len(line6) != 4 :

                print("Wrong number of arguments in line 6 of input data file, i.e. trajectory file, energy file, msd file and rdf file. ")

            else :

                xyz_trajectory = line6[0]
                energies_file = line6[1]
                msd_file = line6[2]
                rdf_file = line6[3]

                # Open output file
                outfile1 = open(xyz_trajectory, "w")
                outfile2 = open(energies_file, "w")
                outfile3 = open(msd_file, "w")
                outfile4 = open(rdf_file, "w")

                outfile2.write("Time, Kinetic Energy, Potential Energy, Total Energy\n")
                outfile3.write("Time, MSD\n")
                outfile4.write("Position, RDF(Position)\n")

    infile.close()

    # Part 2.) Specifies initial conditions

    cut_off_radius = 3.5
    time = 0.0
    msd = 0

    particle_list = []
    msd_list = []
    rdf_list = []

    N = len(particle_list)
    outfile1.write(f"{str(number_particles)}\n")

    for particle in range(number_particles) :

        particle_list.append(Particle3D(label = f"n_{particle}", mass = 1, position = np.zeros(3), velocity = np.zeros(3)))

    box_size, full_lattice = md.set_initial_positions(rho, particle_list)
    box_size = box_size[0]
    md.set_initial_velocities(temperature, particle_list)
    separation_matrix = calculate_pair_separation(particle_list, box_size)

    initial_particle_list = copy.deepcopy(particle_list)
    msd += mean_squared_displacement(particle_list, initial_particle_list, time, box_size)
    msd_list.append(msd)
    outfile3.write(f"{time}, {msd} \n")
                                                                                                      
    for n in range(number_particles) :

        outfile1.write(f"{str(particle_list[n])}")
        kinetic_energy = particle_list[n].calculate_system_kinetic_energy(particle_list)
        potential_energy = lennard_jones_potential(particle_list, box_size, cut_off_radius, separation_matrix[n])
        force_matrix = lennard_jones_force(particle_list, box_size, cut_off_radius)
        total_energy = kinetic_energy + potential_energy
        outfile2.write(f"{time}, {kinetic_energy}, {potential_energy}, {total_energy}\n")

    time_list = [time]
    potential_energy_list = [potential_energy]
    kinetic_energy_list = [kinetic_energy]
    total_energy_list = [total_energy]

    for i in range(numstep) :

        for n in range(len(particle_list)) :

            particle_list[n].update_2nd_position(dt, np.sum(force_matrix[:, n], axis = 0) * (-1))
            particle_list[n].position = periodic_boundary_conditions(particle_list[n], box_size)
            outfile1.write(f"{str(particle_list[n])}")

        for j in range(len(particle_list)) :

            new_force_matrix = lennard_jones_force(particle_list, box_size, cut_off_radius)

        for k in range(len(particle_list)) :

            particle_list[k].update_velocity(dt, (-0.5) * ((np.sum(force_matrix[:, k], axis = 0)) + (np.sum(new_force_matrix[:, k], axis = 0))))

        force_matrix = new_force_matrix

        time += dt

        for m in range(len(particle_list)) :

            kinetic_energy = particle_list[m].calculate_system_kinetic_energy(particle_list)
            potential_energy = lennard_jones_potential(particle_list, box_size, cut_off_radius, separation_matrix[m])
            total_energy = kinetic_energy + potential_energy
            outfile2.write(f"{time}, {kinetic_energy}, {potential_energy}, {total_energy}\n")

        msd += (mean_squared_displacement(particle_list, initial_particle_list, time, box_size))
        outfile3.write(f"{time}, {msd} \n")
        msd_list.append(msd)

        kinetic_energy_list.append(kinetic_energy)
        potential_energy_list.append(potential_energy)
        total_energy_list.append(total_energy)
        time_list.append(time)

     
    # Post-simulation:
    # Close output file
    outfile1.close()
    outfile2.close()
    outfile3.close()
    outfile4.close()

    # Part 6.) Plots particle energy to screen

    # Plot particle energy to screen
    pyplot.title('Total Energy vs Time')
    pyplot.xlabel('Time : ')
    pyplot.ylabel('Energy : ')
    pyplot.plot(time_list, total_energy_list)
    pyplot.show()

    pyplot.title('Kinetic Energy vs Time')
    pyplot.xlabel('Time : ')
    pyplot.ylabel('Kinetic Energy : ')
    pyplot.plot(time_list, kinetic_energy_list)
    pyplot.show()

    pyplot.title('Potential Energy vs Time')
    pyplot.xlabel('Time : ')
    pyplot.ylabel('Potential Energy : ')
    pyplot.plot(time_list, potential_energy_list)
    pyplot.show()

    pyplot.title('MSD vs Time')
    pyplot.xlabel('Time : ')
    pyplot.ylabel('MSD : ')
    pyplot.plot(time_list, msd_list)
    pyplot.show()


    # Part 7.) Measures the energy inaccuracy of the simulation and prints it to the screen


    initial_energy_k = kinetic_energy_list[0]
    initial_energy_p = potential_energy_list[0]

    initial_energy = initial_energy_p + initial_energy_k

    max_energy = max(total_energy_list)
    min_energy = min(total_energy_list)

    delta_energy = max_energy - min_energy
    energy_inaccuracy = delta_energy / initial_energy

    print("Energy inaccuracy : +/-", energy_inaccuracy)

# Execute main method, but only when directly invoked
if __name__ == "__main__":
    main()
