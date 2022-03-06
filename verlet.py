"""
Velocity Verlet Time Integrator for simulations of N particles.

"""

import sys
import math
import numpy as np
import matplotlib.pyplot as pyplot
from particle3D import Particle3D
import mdutilities as md

def mirror_image_convention(particle, different_particle, box_size) :
    """

    :param particle:
    :param different_particle:
    :param box_size:
    :return:
    """

    mic = ((particle.position - different_particle.position + box_size / 2) * np.linalg.norm(box_size)) - (box_size / 2)
    modulus_mic = np.linalg.norm(mic)
    return modulus_mic

def calculate_pair_separation(particle_list, box_size):
    """

    :param particle_list:
    :param box_size:
    :return:
    """

    N = len(particle_list)
    separations_matrix = np.zeros((N, N, 3))
    pair_sep = float

    for i in range(N) :
        for j in range(i + 1, N) :

            separation = mirror_image_convention(particle_list[i], particle_list[j], box_size)

            separations_matrix[i, j] = separation
            separations_matrix[j, i] = - separation

            pair_sep = np.linalg.norm(separations_matrix[i, j])

    return separations_matrix, pair_sep

def periodic_boundary_conditions(particle, box_size, number_particles) :
    """

    :param particle:
    :param box_size:
    :param number_particles:
    :return:
    """

    for i in range(number_particles) :

        pbc = particle.position[i] * np.linalg.norm(box_size)
        return pbc

def lennard_jones_force(particle_list, box_size, cut_off_radius) :
    """

    :param particle_list:
    :param box_size:
    :param cut_off_radius:
    :return:
    """

    N = len(particle_list)
    lj_force = float
    pair_sep = calculate_pair_separation(particle_list, box_size)
    pair_sep = pair_sep[1]

    if pair_sep < cut_off_radius :
        for j in range(N):
            lj_force = - 48 * ((pair_sep ** - 14) - (1 / 2) * (pair_sep ** - 8)) * pair_sep

    elif pair_sep == cut_off_radius :

        lj_force = - 48 * ((pair_sep ** - 14) - (1 / 2) * (pair_sep ** - 8)) * cut_off_radius

    else :
        for j in range(N) :
            lj_force = 0

    return lj_force


def lennard_jones_potential(particle_list, box_size) :

    """


    :param particle_list:
    :param box_size:

    :return:
    """

    N = len(particle_list)
    lj_potential = float
    pair_sep = calculate_pair_separation(particle_list, box_size)
    pair_sep = pair_sep[1]

    for i in range(N) :
        for j in range(i + 1, N) :

            lj_potential = 4 * ((pair_sep ** - 12) - (pair_sep ** - 6))

    return lj_potential


# Begin main code
def main():
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
    with open(sys.argv[1], "r") as infile:

        # Part 1.) Reads in data file from the command line

        # Read name of files from command line which needs 3 parts
        if len(sys.argv) != 3:

            # Helpful error message if the format is incorrect
            print("Wrong number of arguments.")
            print("Usage: " + sys.argv[0] + "<input file>" + "<output file>")
            quit()
        else:
            outfile_name = sys.argv[2]

            # Open output file
            outfile = open(outfile_name, "w")

            line1 = infile.readline()  # Processes line 1 of input file
            line2 = infile.readline()  # Processes line 2 of input file
            line2 = line2.split()  # Separates the parameters in line 2

            # Helpful error message if there is not 2 parameters in line 2
            if len(line2) != 3:
                print("Wrong number of arguments in line 2 of input data file, i.e. simulation parameters. ")

            else:
                dt = float(line2[0])  # Reads in time-step for simulation
                numstep = int(line2[1])  # Reads in number of steps for
                number_particles = int(line2[2])


    infile.close()

    # Part 2.) Specifies initial conditions

    cut_off_radius = 3.5
    particle_list = []

    for particle in range(number_particles) :
        particle_list.append(
            Particle3D(
                label = f"n_{particle}",
                mass = 1,
                position = np.zeros(3),
                velocity = np.zeros(3)
                )
            )

    rho = 1
    time = 0.0
    temperature = 1

    box_size, full_lattice = md.set_initial_positions(rho, particle_list)
    box_size = box_size[0]
    md.set_initial_velocities(temperature, particle_list)

    for particle in particle_list:
        separations_matrix = calculate_pair_separation(particle_list, number_particles)
        print(separations_matrix)
        lj_force = lennard_jones_force(particle_list, box_size, cut_off_radius)
        print(lj_force)
        lj_potential = lennard_jones_potential(particle_list, box_size)
        print(lj_potential)

    for particle in range(number_particles) :
        outfile.write(particle.__str__())  # Formats output file being written

    quit()

    # Get initial force
    force1 = lennard_jones_force(particle_list, box_size, cut_off_radius)
    force2 = - force1

    # Part 3.) Initialises data lists for plotting later

    time_list = [time]
    position_list = [particle[i].position]
    energy_list = [energy]

    # Part 4.) Starts a time integration loop

    for i in range(numstep):
        # Update particle position
        particle[i].update_2nd_position(dt, force[i])

        # Update force
        force1_new = lennard_jones_force(pair_sep, cut_off_radius, number_particles)
        force2_new = - force1_new

        # Update particle velocity by averaging current and new forces
        particle1.update_velocity(dt, 0.5 * (force1 + force1_new))
        particle2.update_velocity(dt, 0.5 * (force2 + force2_new))

        # Re-define force value
        force1 = force1_new
        force2 = force2_new

        # Increase time
        time += dt

        # Output particle information
        energy = particle1.calculate_kinetic_energy() + particle2.calculate_kinetic_energy() + lennard_jones_potential(number_particles, pair_sep)
        outfile.write("{0:f} {1:f} {2:12.8f}\n".format(time, p1_to_p2, energy))

        # Append information to data lists
        time_list.append(time)
        pos1_list.append(particle1.position)
        pos2_list.append(particle2.position)
        energy_list.append(energy)

    # Post-simulation:
    # Close output file
    outfile.close()

    # Part 5.) Plots particle trajectory to screen

    pyplot.title('Position vs Time')
    pyplot.xlabel('Time : ')
    pyplot.ylabel('Position : ')
    pyplot.plot(time_list, pos1_list)
    pyplot.plot(time_list, pos2_list)
    pyplot.show()

    # Part 6.) Plots particle energy to screen

    # Plot particle energy to screen
    pyplot.title('Total Energy vs Time')
    pyplot.xlabel('Time : ')
    pyplot.ylabel('Energy : ')
    pyplot.plot(time_list, energy_list)
    pyplot.show()

    # Part 7.) Measures the energy inaccuracy of the simulation and prints it to the screen

    initial_energy = particle1.calculate_kinetic_energy() + particle2.calculate_kinetic_energy() + lennard_jones_potential(number_particles, pair_sep)
    max_energy = max(energy_list)
    min_energy = min(energy_list)

    delta_energy = max_energy - min_energy
    energy_inaccuracy = delta_energy / initial_energy

    print("Energy inaccuracy : +/-", energy_inaccuracy, "eV ")


# Execute main method, but only when directly invoked
if __name__ == "__main__":
    main()
