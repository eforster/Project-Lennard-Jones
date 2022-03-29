"""
Particle3D class to describe point particles in 3D space.

An instance describes a particle in Euclidean 3D space:
- Velocity and Position are [3] arrays

Included methods:
- time integrators
- calculations
- updates to positions and velocities

Author: E Forster, s1639706
"""

import math
import numpy as np

class Particle3D(object) :

    """
    Class to describe point-particles in 3D space.

        Properties:
    label : name of the particle
    mass : mass of the particle
    position : position of the particle
    velocity : velocity of the particle

        Methods:
    __init__ : initialises a particle in 3D space
    __str__ : sets up an x, y, z coordinate system for a particle
    calculate_kinetic_energy : computes the kinetic energy of the particle
    calculate_momentum : computes the linear momentum
    update_position : updates the position to 1st order
    update_2nd_position : updates the position to 2nd order
    update_velocity : updates the velocity

        Static Methods :
    new_3d_particle : initializes a P3D instance from a  file handle
    calculate_system_kinetic_energy : computes the total kinetic energy of a 3D particle list
    calculate_center_of_mass_velocity : computes total mass and the center-of-mass velocity of a 3D particle list
    """

    def __init__(self, label, mass, position, velocity) :

        """
        Initialises a particle in 3D space

        :param label: string with the name of the particle
        :param mass: float with the mass of the particle
        :param position: [3] float array with position
        :param velocity: [3] float array with velocity
        """

        self.label = label
        self.mass = mass
        self.position = np.array(position)
        self.velocity = np.array(velocity)

    def __str__(self) :

        """
        (x, y, z) compliant string with format :

        <label> <x> <y> <z>

        :return xyz_string: (label, x, y, z)
        """

        xyz_string = f"{self.label} {self.position[0]} {self.position[1]} {self.position[2]}\n"
        return xyz_string

    def calculate_kinetic_energy(self) :

        """
        Returns the kinetic energy of a Particle3D instance

        :return kinetic_energy: float, (1/2) * mass * (velocity ** 2)
        """

        kinetic_energy = (1 / 2) * self.mass * (np.linalg.norm(self.velocity) ** 2)
        return kinetic_energy

    def calculate_momentum(self) :
        """
        Calculates and returns the momentum of a Particle3D instance

        :return momentum: float, mass * velocity
        """

        momentum = self.mass * self.velocity
        return momentum

    def update_position(self, dt) :

        """
        Calculates and updates the new position of a Particle3D instance to 1st order

        :param dt: float, time-step
        """

        self.position = self.position + dt * self.velocity


    def update_2nd_position(self, dt, force) :

        """
        Calculates and updates the position of a Particle3D instance to 2nd order

        :param dt: float, time-step
        :param force: float, force on particle
        """

        self.position = self.position + dt * self.velocity + (dt ** 2) * (force / (2 * self.mass))


    def update_velocity(self, dt, force) :

        """
        Updates the velocity of a Particle3D instance to 1st order

        :param dt: float, time-step
        :param force: float, force on particle
        """

        self.velocity = self.velocity + dt * (force / self.mass)

    @staticmethod

    def new_3d_particle(input_file) :

        """
        Initialises a Particle3D instance when given an input file handle.

        The input file should contain one line per particle in the following format:
        label   <mass>   <x> <y> <z>   <vx> <vy> <vz>

        :param input_file: readable file handle in the above format

        :return Particle3D: Particle3D instance, label mass position velocity
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
        Returns the total kinetic energy of the system as a float

        :param particle_3d_list: list in which each item is a Particle3D instance

        :return system_kinetic_energy: sum of each particle's kinetic energy
        """

        system_kinetic_energy = 0

        for particle in particle_3d_list :

            kinetic_energy = particle.calculate_kinetic_energy()
            system_kinetic_energy += kinetic_energy

        return float(system_kinetic_energy)

    @staticmethod

    def calculate_centre_of_mass_velocity(particle_3d_list) :

        """
        Computes the total mass an centre-of-mass velocity of a list of Particle3D instances

        :param particle_3d_list: list in which each item is a Particle3D instance

        :return total_mass: the total mass of the system
        :return center_of_mass_velocity: centre-of-mass velocity
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
