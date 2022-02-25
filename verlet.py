"""
Velocity Verlet Time Integrator for simulations of N particles.

"""

import sys
import math
import numpy as np
import matplotlib.pyplot as pyplot
from particle3D import Particle3D


def pair_separation(particle, different_particle, number_particles):

    for i in range(number_particles):
        for j in range(number_particles):
            pair_sep = particle.position[i] - different_particle.position[j]
            print(pair_sep)

def mirror_image_convention(pair_sep, box_size, number_particles) :

    mic = ((pair_sep + box_size / 2) * np.linalg.norm(box_size)) - (box_size / 2)
    return mic

def periodic_boundary_conditions(particle, box_size, number_particles) :

    move = particle.update_2nd_position(dt, force)
    pbc = move * np.linalg.norm(box_size)
    return pbc

def lennard_jones_force(pair_sep, cut_off_radius, number_particles) :
    """

    :param number_particles:
    :param pair_sep:
    :param cut_off_radius:
    :return:
    """

    if pair_sep > cut_off_radius :
        lj_force = 0
        return lj_force

    else :
        for j in range(number_particles) :

            mod_pair_sep = np.linalg.norm(pair_sep)
            lj_force = - 48 * ((mod_pair_sep ** - 14) - (1 / 2) * (mod_pair_sep ** - 8)) * pair_sep
            return lj_force

def lennard_jones_potential(position1, position2, number_particles, pair_sep) :

    """


    :param position1:
    :param position2:
    :param pair_sep:
    :param number_particles:

    :return:
    """

    mod_pair_sep = np.linalg.norm(pair_sep)

    for i in range(number_particles) :
        for j in range(i + 1, number_particles) :
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
            if len(line2) != 2:
                print("Wrong number of arguments in line 2 of input data file, i.e. simulation parameters. ")

            else:
                dt = float(line2[0])  # Reads in time-step for simulation
                numstep = int(line2[1])  # Reads in number of steps for simulation

            line3 = infile.readline()  # Processes line 3 of input file
            line4 = infile.readline()  # Processes line 4 of input file
            line4 = line4.split()  # Separates the parameters in line 4
            line5 = infile.readline()   # Processes line 5 of input file
            line5 = line5.split()

            # Helpful error message if there is not 3 parameters in line 4
            if len(line4) and len(line5) != 8:
                print("Wrong number of arguments in line 5 and 6, i.e. initial conditions of particles. ")

            else:

                position1 = np.array([float(line4[2]), float(line4[3]), float(line4[4])])  # Sets up Particle 1 position
                velocity1 = np.array([float(line4[5]), float(line4[6]), float(line4[7])])  # Sets up Particle 1 velocity

                particle1 = Particle3D(str(line4[0]), float(line4[1]), position1, velocity1)  # Sets up Particle 1 as a particle3D instance

                position2 = np.array([float(line5[2]), float(line5[3]), float(line5[4])])  # Sets up Particle 2 position
                velocity2 = np.array([float(line5[5]), float(line5[6]), float(line5[7])])  # Sets up Particle 2 velocity

                particle2 = Particle3D(str(line5[0]), float(line5[1]), position2, velocity2)  # Sets up Particle 2 as a particle3D instance


    infile.close()

    # Part 2.) Specifies initial conditions

    time = 0.0
    p1_to_p2 = np.linalg.norm(particle2.position - particle1.position)
    energy = particle1.calculate_kinetic_energy() + particle2.calculate_kinetic_energy() + lennard_jones_potential( )
    outfile.write("{0:f} {1:f} {2:12.8f}\n".format(time, p1_to_p2, energy))  # Formats output file being written

    # Get initial force
    force1 = lennard_jones_force()
    force2 = - force1

    # Part 3.) Initialises data lists for plotting later

    time_list = [time]
    pos1_list = [particle1.position]
    pos2_list = [particle2.position]
    pos_list = [np.linalg.norm(particle2.position - particle1.position)]  # Position list is | r2 - r1 | from particle positions
    energy_list = [energy]

    # Part 4.) Starts a time integration loop

    for i in range(numstep):
        # Update particle position
        particle1.update_2nd_position(dt, force1)
        particle2.update_2nd_position(dt, force2)
        p1_to_p2 = np.linalg.norm(particle2.position - particle1.position)

        # Update force
        force1_new = lennard_jones_force()
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
        energy = particle1.calculate_kinetic_energy() + particle2.calculate_kinetic_energy() + lennard_jones_potential()
        outfile.write("{0:f} {1:f} {2:12.8f}\n".format(time, p1_to_p2, energy))

        # Append information to data lists
        time_list.append(time)
        pos1_list.append(particle1.position)
        pos2_list.append(particle2.position)
        pos_list.append(p1_to_p2)
        energy_list.append(energy)

    # Post-simulation:
    # Close output file
    outfile.close()

    # Part 5.) Plots particle trajectory to screen

    pyplot.title('Velocity Verlet : Position vs Time')
    pyplot.xlabel('Time (* 1.018050571e-14 s) : ')
    pyplot.ylabel('Position (Angstroms) : ')
    pyplot.plot(time_list, pos1_list)
    pyplot.plot(time_list, pos2_list)
    pyplot.show()

    # Part 6.) Plots particle energy to screen

    # Plot particle energy to screen
    pyplot.title('Velocity Verlet : Total Energy vs Time')
    pyplot.xlabel('Time (* 1.018050571e-14 s) : ')
    pyplot.ylabel('Energy (eV) : ')
    pyplot.plot(time_list, energy_list)
    pyplot.show()

    # Part 7.) Measures the energy inaccuracy of the simulation and prints it to the screen

    initial_energy = particle1.calculate_kinetic_energy() + particle2.calculate_kinetic_energy() + lennard_jones_potential()
    max_energy = max(energy_list)
    min_energy = min(energy_list)

    delta_energy = max_energy - min_energy
    energy_inaccuracy = delta_energy / initial_energy

    print("Energy inaccuracy : +/-", energy_inaccuracy, "eV ")


# Execute main method, but only when directly invoked
if __name__ == "__main__":
    main()
