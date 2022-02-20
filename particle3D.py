"""
Particle3D class to describe point particles in 3D space.

An instance describes a particle in Euclidean 3D space:
- Velocity and Position are [3] arrays

Included methods:
- time integrators
- calculations
- updates positions and velocities

Author: s1639706

"""

import math
import numpy as np

class Particle3D(object) :

    """
    Class to describe point-particles in 3D space.

        Properties:

    label: name of the particle
    mass: mass of the particle
    position: position of the particle
    velocity: velocity of the particle

        Methods:

    __init__ : initialises a particle in 3D space
    __str__ :



    """

    def __init__(self, label, mass, position, velocity) :

        """


        :param label:
        :param mass:
        :param position:
        :param velocity:
        """

        self.label = label
        self.mass = mass
        self.position = position
        self.velocity = velocity


    def __str__(self) :

        """

        :return:
        """

        xyz_string = str(self.label + "   " + str(self.position[0]) + "  " + str(self.position[1]) + "  " + str(self.pos[2]))
        return xyz_string

    def calculate_kinetic_energy(self) :

        """

        :return:
        """

        kinetic_energy = (1/2) * self.mass * (np.linalg.norm(self.velocity) ** 2)
        return kinetic_energy

    def calculate_momentum(self) :
        """

        :return:
        """

        momentum = self.mass * self.velocity
        return momentum

    def update_position(self, dt) :

        """

        :param dt:
        :return:
        """
        self.position = self.position + dt * self.velocity


    def update_2nd_position(self, dt, force) :

        """

        :param dt:
        :param force:
        :return:
        """

        self.position = self.position + dt * self.velocity + (dt ** 2) * (force / (2 * self.mass))

    def update_velocity(self, dt, force) :

        """

        :param dt:
        :param force:
        :return:
        """

        self.velocity = self.velocity + dt * (force/self.mass)

    @staticmethod

    def new_3d_particle(input_file) :

        """

        :param input_file:
        :return:
        """

        try :

            data = input_file.readline()
            lines = data.split()

            label = str(lines[0])
            mass = float(lines[1])

            x = float(lines[2])
            y = float(lines[3])
            z = float(lines[4])
            position = np.array(x, y, z)

            velocity_x = float(lines[5])
            velocity_y = float(lines[6])
            velocity_z = float(lines[7])
            velocity = np.array(velocity_x, velocity_y, velocity_z)

            return Particle3D(label, mass, position, velocity)

        except IndexError :

            print("Error: Incorrect file format")


    @staticmethod

    def calculate_system_kinetic_energy(particle_3d_list) :

        """

        :param particle_3d_list:
        :return:
        """

        system_kinetic_energy = 0

        for particle in particle_3d_list :

            kinetic_energy = particle.calculate_kinetic_energy()
            system_kinetic_energy += kinetic_energy

        return float(system_kinetic_energy)

    @staticmethod

    def calculate_centre_of_mass_velocity(particle_3d_list) :

        """

        :param particle_3d_list:
        :return:
        """
        total_mass = 0
        centre_of_mass_velocity = 0
        total = 0

        for particle in particle_3d_list :

            particle_mass = particle.mass
            total_mass += particle_mass

        for particle in particle_3d_list :

            particle_velocity = particle.velocity
            mass_x_velocity = particle_mass * particle_velocity
            total += mass_x_velocity

            centre_of_mass_velocity = total / total_mass

        return total_mass, centre_of_mass_velocity
