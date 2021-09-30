import copy
import sys
import time
import numpy as np
from random import randrange, uniform, randint, sample
from Organisms.GA.Cluster import Cluster
from Organisms.GA.ExternalDefinitions import is_position_already_occupied_by_an_atom_in_Cluster # , InclusionRadiusOfCluster


def moveMutate(cluster_to_mutate, dist_to_move):
	'''
	This method randomly displaces all the atoms within the structure cluster_to_mutate from its original position.
	This def depends on the value of the maximum bond length of the cluster, self.r_ij.
	The def will cause all atoms in the cluster to move randomly by up to a 1/2 of self.r_ij from their original position.

	:param cluster_to_mutate: The cluster to move the atoms in the cluster.
	:type  cluster_to_mutate: GA.Cluster
	:param dist_to_move: The fistnace to move clusters by
	:type  dist_to_move: float

	:returns The newly created mutated cluster
	:rtypes  GA.Cluster
	'''
	mutant = cluster_to_mutate.deepcopy()
	# Randomly move each atom of the cluster
	for atom in mutant:
		distance_left_to_move = dist_to_move
		movement_x = uniform(-1.,1.)*distance_left_to_move
		atom.x += movement_x
		distance_left_to_move = (distance_left_to_move**2.0 - movement_x**2.0)**0.5
		movement_y = uniform(-1.,1.)*distance_left_to_move
		atom.y += movement_y
		distance_left_to_move = (distance_left_to_move**2.0 - movement_y**2.0)**0.5
		movement_z = distance_left_to_move
		atom.z += uniform(-1.,1.)*movement_z
	if not len(mutant) == len(cluster_to_mutate):
		print('Error in def moveMutate, in Types_Of_Mutations.py')
		print('The offspring contains '+str(len(mutant))+' atoms, but should contain '+str(len(cluster_to_mutate)))
		print('Check this')
		import pdb; pdb.set_trace()
		print('This program will finish without completing')
		exit()
	return mutant

def homotopMutate(cluster_to_mutate):
	'''
	This definition is designed to swap the positions of two elementally different atoms in a cluster
	
	:param cluster_to_mutate: The cluster to move the atoms in the cluster.
	:type  cluster_to_mutate: GA.Cluster

	:returns The newly created mutated cluster
	:rtypes  GA.Cluster

	'''
	# randomly select a atom in the cluster
	atom1Index = randrange(0,len(cluster_to_mutate))
	# randomly select a second atom in the cluster which is not the same as the first and is of a different element compared to the first atom. 
	while True:
		atom2Index = randrange(0,len(cluster_to_mutate))
		# If the two atoms picked are the same or are the same type of atom, try again (can make this more efficient)
		if not (cluster_to_mutate[atom1Index].symbol == cluster_to_mutate[atom2Index].symbol or
				atom1Index == atom2Index):
			break
	# Swap element types of pair in terms of positions. Note (GRW) I did originally just swap
	# the symbol variable in mutant but thought there are alot of other variables I am not thinking about
	# transferring as well which may cause issues. Swapping the positions seemed less likely to bring up
	# problems in the future.
	print('Atom ' + str(atom1Index) + ' (' + str(cluster_to_mutate[atom1Index].symbol) +
		  ') will be swapped with atom ' + str(atom2Index) + ' (' + str(cluster_to_mutate[atom2Index].symbol) + ').')
	mutant = copy.deepcopy(cluster_to_mutate)
	temp = copy.deepcopy(mutant[atom1Index].position)
	mutant[atom1Index].position = copy.deepcopy(mutant[atom2Index].position)
	mutant[atom2Index].position = copy.deepcopy(temp)
	return mutant

def randomMutate(boxtoplaceinlength, vacuumAdd, composition_constrained, cluster_makeup=None, cluster_to_mutate=None,
				 percentage_of_cluster_to_randomise=None):
	"""
	This definition provides the random method for the mutation proceedure. In this method, a cluster is
	designed by randomly placing the designed atoms of elements into a predefined unit cell.
	
	:param boxtoplaceinlength: This is the length of the box you would like to place atoms in to make a randomly constructed cluster.
	:type  boxtoplaceinlength: float
	:param vacuumAdd: The length of vacuum added around the cluster. Written in A.
	:type  vacuumAdd: float
	:param cluster_makeup: check this
	:type  cluster_makeup: {'Element': int(number of that 'element' in this cluster),...}
	:param cluster_to_mutate: If the user desired, they can tell this definition what cluster they want to mutate. Default: None.
	:type  cluster_to_mutate: ASE.Atoms
	:param percentage_of_cluster_to_randomise: This is the percentage of the number of atoms in the cluster to randomise
	:type  percentage_of_cluster_to_randomise: float
	
	:returns mutant: The description of the mutant cluster.
	:rtypes: ase.Atoms or GA.Cluster

	"""
	# Prepare the method for performing a fully ramdomly generated cluster
	mutant = None
	atoms_to_randomise = []

	np.random.seed(int(time.time()))
	if cluster_makeup is not None and (cluster_to_mutate is None or percentage_of_cluster_to_randomise is None):
		# Turn cluster_makeup into a chemical formula string
		print('Mutant transformed', file=sys.stdout)
		cluster_chemical_formula = '' 
		for element, no_of_element in cluster_makeup.items():
			if not composition_constrained:
				max_atom_change = int(np.floor(np.minimum(no_of_element, 5)))
				if max_atom_change == no_of_element:
					max_atom_change -= 1
				if max_atom_change == 0:
					actual_change = 0
				else:
					actual_change = np.random.randint(low=-max_atom_change, high=max_atom_change, size=1, dtype=int)[0]
				# print(element, actual_change, file=sys.stderr)
				cluster_chemical_formula += str(element) + str(int(no_of_element + actual_change))
			else:
				cluster_chemical_formula += str(element) + str(int(no_of_element))

		# print('genning new mutant: ', cluster_chemical_formula)
		# set up the cluster for randomizing
		mutant = Cluster(cluster_chemical_formula)
		nAtoms = len(mutant)
		atoms_to_randomise = range(nAtoms)  # randomise all atoms in the cluster.
	# preparing this method for only randomly changing the position of only a percentage (of atoms) of a cluster.
	elif cluster_makeup is None and not (cluster_to_mutate is None or percentage_of_cluster_to_randomise is None):
		mutant = copy.deepcopy(cluster_to_mutate)
		nAtoms = len(mutant)
		if len(mutant.get_chemical_symbols()) == 1:  # no swapping of unary material
			swapping_prob = 0
		else:
			swapping_prob = 0.45

		# my additional idea of swapping single atoms
		# at this point we could also swap multiple atoms but this is not composition_constrained -> use else here TODO
		if composition_constrained:
			if np.random.random() < swapping_prob:
				print('Mutant swapped', file=sys.stdout)
				symbols = dict()
				for idx, species in enumerate(mutant.get_chemical_symbols()):
					if species not in symbols:
						symbols[species] = []
						symbols[species].append(idx)
					else:
						symbols[species].append(idx)

				chosen = np.random.choice([s for s in symbols], 2)

				exchange = dict()
				for species in chosen:
					exchange[species] = sample(symbols[species], 1)

				pairs = list(exchange.items())

				swapped = dict({pairs[1][0]: exchange[pairs[0][0]], pairs[0][0]: exchange[pairs[1][0]]})
				for species in swapped:
					for idx in swapped[species]:
						mutant[idx].symbol = species

				nAtoms = len(mutant)
				atoms_to_randomise = range(nAtoms)

		else:
			# The following will pick random atoms in the cluster to randomise, original code
			print('Mutant randomized', file=sys.stdout)
			no_of_atoms_to_randomise = int(np.ceil(float(nAtoms) * (float(percentage_of_cluster_to_randomise) / 100.0)))
			all_atoms_in_cluster = [int(ia) for ia in range(nAtoms)]
			atoms_to_randomise = []
			for NOTUSED in range(no_of_atoms_to_randomise):
				index_to_randomise = randint(0, len(all_atoms_in_cluster)) - 1
				atoms_to_randomise.append(all_atoms_in_cluster.pop(index_to_randomise))

			cluster_chemical_formula = ''
			for element, no_of_element in cluster_to_mutate.get_elemental_makeup().items():
				cluster_chemical_formula += str(element) + str(int(no_of_element))
			# print('randomize mutant: ', cluster_chemical_formula)

	else:  # Error
		print('Error in MutationProcedure: def randomMutate.')
		print('cluster_makeup = ' + str(cluster_makeup))
		print('cluster_to_mutate = ' + str(cluster_to_mutate))
		print('percentage_of_cluster_to_randomise = ' + str(percentage_of_cluster_to_randomise))
		import pdb; pdb.set_trace()
		exit()
	# At this point, the method needs:
	#	* mutant: class ASE.Atoms. A framework of the cluster.
	#	* atoms_to_randomise: the index of atoms that need to be randomised, list of ints
	# setup the Atoms class to define the atoms in the cluster of mutant and define a unit cell
	# for atoms to be randomly placed in.
	# Get initial size of cell\
	mutant.set_cell([boxtoplaceinlength, boxtoplaceinlength, boxtoplaceinlength])
	mutant.center()
	# For each element, place a set number of atoms of that element into the cluster in a random position
	# within the defined cube with sides = r_ij*scale
	for index in atoms_to_randomise:
		while True:
			x_position = boxtoplaceinlength * uniform(0.25, 0.75)  # janK: change from (0, 1)
			y_position = boxtoplaceinlength * uniform(0.25, 0.75)
			z_position = boxtoplaceinlength * uniform(0.25, 0.75)
			position = [x_position, y_position, z_position]
			# check the cluster to see that this new random position will not cause this atom 
			# to be in the same place as any atom in the cluster currently
			if not is_position_already_occupied_by_an_atom_in_Cluster(position, mutant,
																	  atom_indices_to_exclude_from_comparison=[index]):
				# Found that one of the atoms in this cluster has the same position as this randomly generated one.
				# Will need to make a new randomly generated position list
				break
		mutant[index].position = position
	# Add a unit cell to the cluster. Mainly needed for VASP.
	#lengthOfCell = 2.0*InclusionRadiusOfCluster(mutant) + vacuumAdd
	#cell = [lengthOfCell,lengthOfCell,lengthOfCell]
	#mutant.set_cell(cell)
	mutant.center(vacuum=vacuumAdd) # centre the cluster within this unit cell
	# return the mutant cluster.
	#print('cluster_makeup: '+str(cluster_makeup)+'; cluster_to_mutate: '+str(cluster_to_mutate))
	if cluster_makeup is None: 
		if not len(mutant) == len(cluster_to_mutate):
			print('Error in def randomMutate, in Types_Of_Mutations.py')
			print('The offspring contains '+str(len(mutant))+' atoms, but should contain '+str(len(cluster_to_mutate)))
			print('Check this')
			import pdb; pdb.set_trace()
			print('This program will finish without completing')
			exit()
	# elif cluster_to_mutate is None:
	# 	number_of_atoms = sum(Counter(cluster_makeup).values())
	# 	pass
		#print('Atoms in mutant: '+str(len(mutant))+'; number_of_atoms = '+str(number_of_atoms))
		#import pdb; pdb.set_trace()
		# if not len(mutant) == number_of_atoms:
		# 	print('Error in def randomMutate, in Types_Of_Mutations.py')
		# 	print('The offspring contains '+str(len(mutant))+' atoms, but should contain '+str(len(cluster_to_mutate)))
		# 	print('Check this')
		# 	import pdb; pdb.set_trace()
		# 	print('This program will finish without completing')
		# 	exit()
	return mutant
		