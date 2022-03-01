"""
CMod Project B: auxiliary MD methods
"""
import sys
import math
import numpy as np


def set_initial_positions(rho, particles):
    # Determine number of particles
    natoms = len(particles)
    
    # Set box dimensions
    box_size = (natoms/rho)**(1./3.)
    
    # Number or particles in each direction
    ndim = math.ceil( (natoms/4.0)**(1./3.) )
    
    # Give warning if fcc lattice will not be fully occupied
    full_lattice = True
    if 4*ndim**3 != natoms:
        full_lattice = False
        print("Atoms will not fill a fcc lattice completely.\n")
        print("   Occupancy factor = {0:5.3f}".format(natoms/4*ndim**3))
            
    # Separation between particles
    delta = box_size / ndim
    print("The nearest-neighbour distance is {0:6.3f} [L]\n".format(delta*np.sqrt(2)/2))
    
    # Set particle positions
    fcc = 0.5*np.array([[0, 0, 0],
                        [0, 1, 1],
                        [1, 0, 1],
                        [1, 1, 0]],float)

    all_sites = np.zeros([4*ndim**3, 3])

    n_lattice = 0
    for i in range(ndim):
        for j in range(ndim):
            for k in range(ndim):
                all_sites[n_lattice:n_lattice+4] = np.array([i,j,k]) + fcc
                n_lattice += 4
    all_sites *= delta

    for particle, position in zip(particles,all_sites):
        particle.position = position

    # Some output
    print("{0:d} atoms placed on a face-centered cubic lattice.\n".format(natoms))
    print("Box dimensions: {0:f} {0:f} {0:f}\n".format(box_size))
    
    # Return the box size as Numpy array, and whether the lattice is fully occupied
    return box_size*np.ones(3), full_lattice
    

def set_initial_velocities(mytemp, particles, seed=None):
    # Set the random seed to ease up debugging
    try:
        prng = np.random.RandomState(seed)
    except:
        print("WARNING, {0} is not a valid random seed\n".format(seed))
        print("Continuing without setting the seed...\n")

    # Check the temperature can be a valid, positive float
    # NOTE: This should be done at parameter read
    try:
        temp = float(mytemp)
    except:
        print("ERROR: Could not convert {0} to a valid float. Exiting...\n".format(temp))
        sys.exit()
    if temp < 0.0:
        print("ERROR: Temperature must be positive. Exiting...\n")
        sys.exit()

    # Initialisation
    natoms = len(particles)
    velocities = prng.random([natoms,3])

    # Insure CoM does not move
    v_com = np.sum(velocities, axis=0)/natoms
    velocities -= v_com
    v_com = np.sum(velocities, axis=0)/natoms

    # Rescale velocities so that total e_kin = 3/2 N_atom kB T. Assumes m=1
    e_kin = np.sum(velocities**2)
    velocities *= math.sqrt(3*natoms*temp/e_kin)

    # Assign each velocity to a particle
    for particle, velocity in zip(particles, velocities):
        particle.velocity = velocity

    # Output as a sanity check
    print("Temperature = {0:f}     Total Kinetic Energy = {1:f}".format(temp, 0.5*np.sum(velocities**2)))
    print("Centre-of-mass velocity = {0}\n".format(v_com/natoms))


