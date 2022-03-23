"""
Velocity Verlet Time Integrator for simulations of N particles.

"""

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

            #print(f"separation{separation}")

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
                lj_force_matrix[i, j] = 48 * (modulus_sep_matrix**(-14) - (0.5 * modulus_sep_matrix ** (-8))) * sep_matrix[i, j]
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

    for i in range(N):
        for j in range(i + 1, N):
            
            modulus_sep_matrix = np.linalg.norm(sep_matrix[i, j])

            if modulus_sep_matrix > cut_off_radius :

                lj_potential = 4 * ((cut_off_radius ** (- 12)) - (cut_off_radius ** (- 6)))

            else :

                lj_potential = 4 * ((modulus_sep_matrix ** (- 12)) - (modulus_sep_matrix ** (- 6)))


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
                numstep = int(line2[1])  # Reads in number of steps for simulation
                number_particles = int(line2[2])

    infile.close()

    # Part 2.) Specifies initial conditions

    cut_off_radius = 3.5
    time = 0.0
    # box_size = 3
    temperature = 0.1
    rho = 1

    # p1 = Particle3D('Ar', 1, [1.222, 0, 0], [1, 0, 0])
    # p2 = Particle3D('Ar', 1, [0, 0, 0], [1, 0, 0])

    # particle_list = [p1, p2]
    particle_list = []
    N = len(particle_list)
    outfile.write(f"{str(number_particles)}\n")

    for particle in range(number_particles) :

        particle_list.append(Particle3D(label = f"n_{particle}", mass = 1, position = np.zeros(3), velocity = np.zeros(3)))

    box_size, full_lattice = md.set_initial_positions(rho, particle_list)
    box_size = box_size[0]
    md.set_initial_velocities(temperature, particle_list)
    separation_matrix = calculate_pair_separation(particle_list, box_size)
    
    for n in range(number_particles) :

        outfile.write(f"{str(particle_list[n])}")
        kinetic_energy = particle_list[n].calculate_kinetic_energy()
        potential_energy = lennard_jones_potential(particle_list, box_size, cut_off_radius, separation_matrix[n])
        force_matrix = lennard_jones_force(particle_list, box_size, cut_off_radius)
        total_energy = kinetic_energy + potential_energy

    time_list = [time]
    potential_energy_list = [potential_energy]
    kinetic_energy_list = [kinetic_energy]
    total_energy_list = [total_energy]

    for i in range(numstep) :



        for n in range(len(particle_list)) :

            # print(f"force{np.sum(force_matrix[:, n], axis=0)}")

            particle_list[n].update_2nd_position(dt, np.sum(force_matrix[:, n], axis = 0) * (-1))
            particle_list[n].position = periodic_boundary_conditions(particle_list[n], box_size)

            new_force_matrix = lennard_jones_force(particle_list, box_size, cut_off_radius)
            particle_list[n].update_velocity(dt, (-0.5) * ((np.sum(force_matrix[:, n], axis = 0)) + (np.sum(new_force_matrix[:, n], axis = 0))))

            force_matrix = new_force_matrix 

            time += dt

            kinetic_energy = particle_list[n].calculate_kinetic_energy()
            potential_energy = lennard_jones_potential(particle_list, box_size, cut_off_radius, separation_matrix[n])
            total_energy = kinetic_energy + potential_energy

            kinetic_energy_list.append(kinetic_energy)
            potential_energy_list.append(potential_energy)
            total_energy_list.append(total_energy)
            time_list.append(time)
            outfile.write(f"{str(particle_list[n])}")
     
    # Post-simulation:
    # Close output file
    outfile.close()

    # Part 5.) Plots particle trajectory to screen
    """
    pyplot.title('Position vs Time')
    pyplot.xlabel('Time : ')
    pyplot.ylabel('Separation : ')
    pyplot.plot(time_list, separation_list)
    pyplot.show()
    """
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


    # Part 7.) Measures the energy inaccuracy of the simulation and prints it to the screen
    """
    initial_energy = particle.calculate_kinetic_energy() + lennard_jones_potential(particle_list, box_size)
    max_energy = max(energy_list)
    min_energy = min(energy_list)

    delta_energy = max_energy - min_energy
    energy_inaccuracy = delta_energy / initial_energy

    print("Energy inaccuracy : +/-", energy_inaccuracy, "eV ")

    """
# Execute main method, but only when directly invoked
if __name__ == "__main__":
    main()
