#DiamondSquare.py
#Written by Hailey K Buckingham
#On github, buckinha/DiamondSquare
#

import random
import numpy as np

def diamond_square(desired_size, min_height, max_height, roughness, random_seed=None, AS_NP_ARRAY=False):
    """Runs a diamond square algorithm and returns an array (or list) with the landscape

        An important difference (possibly) between this, and other implementations of the 
    diamond square algorithm is how I use the roughness parameter. For each "perturbation"
    I pull a random number from a uniform distribution between min_height and max_height.
    I then take the weighted average between that value, and the average value of the 
    "neighbors", whether those be in the diamond or in the square step, as normal. The 
    weights used for the weighted sum are (roughness) and (1-roughness) for the random
    number and the average, respectively, where roughness is a float that always falls 
    between 0 and 1.
        The roughness value used in each iteration is based on the roughness parameter
    passed in, and is computed as follows:

        this_iteration_roughness = roughness**iteration_number

    where the first iteration has iteration_number = 0. Thus, the first roughness value 
    actually used (in the very first diamond and square step) is roughness**0 = 1. Thus,
    the values for those first diamond and square step entries will be entirely random.
    This effectively means that I am seeding with A 3x3 grid of random values, rather 
    than with just the four corners.
        As the process continues, the weight placed on the random number draw falls from
    the original value of 1, to roughness**1, to roughness**2, and so on, ultimately 
    approaching 0. This means that the values of new cells will slowly shift from being
    purely random, to pure averages.


    OTHER NOTES:
    Internally, all heights are between 0 and 1, and are rescaled at the end.


    PARAMETERS
    ----------
    size
        The size of the grid to be returned. If an integer is passed, the return grid will
        have sides of this length. If an array-like is passed, the first two values will be 
        used as the array shape.

    min_height
        The minimum height allowed on the landscape

    max_height
        The maximum height allowed on the landscape

    roughness
        A float with value between 0 and 1, reflecting how bumpy the landscape should be. 
        Values near 1 will result in landscapes that are extremly rough, and have almost no
        cell-to-cell smoothness. Values near zero will result in landscapes that are almost
        perfectly smooth.

    random_seed
        defaults to None. If a value is given, the algorithm will use it to seed the random
        number generator, ensuring replicability.

    AS_NP_ARRAY
        A boolean reflecting whether the landscape should be returned as a numpy array. If set
        to False, the method will return a 2-dimensional list.


    RETURNS
    -------
    A 2-D numpy array or 2-D list with a diamond-square "height-map"

    """
    #sanitize inputs
    if roughness > 1: roughness = 1.0
    if roughness < 0: roughness = 0.0
    
    #check if size is iterable (i.e, it is likely a length-2 vector, etc...)
    size = [-1,-1]
    if not hasattr(desired_size, '__iter__'):
        #it's not iterable, so it's probably an int
        size[0] = desired_size
        size[1] = desired_size
    else:
        size[0] = desired_size[0]
        size[1] = desired_size[1]


    DS_size, iterations = get_DS_size_and_iters(size)

    #create the array
    #DS_array = np.ndarray((DS_size,DS_size))
    DS_array = np.zeros((DS_size,DS_size), dtype='float')
    DS_array = DS_array - 1.0

    #seed the random number generator
    random.seed(random_seed)


    #seed the corners
    DS_array[        0,         0] = random.uniform(0, 1)
    DS_array[DS_size-1,         0] = random.uniform(0, 1)
    DS_array[        0, DS_size-1] = random.uniform(0, 1)
    DS_array[DS_size-1, DS_size-1] = random.uniform(0, 1)


    #do the algorithm
    for i in range(iterations):
        r = roughness**i
        step_size = DS_size / 2**(i)
        diamond_step(DS_array, step_size, r)
        square_step(DS_array, step_size, r)


    #rescale the array to fit the min and max heights specified
    DS_array = min_height + (DS_array * (max_height - min_height))

    #trim array, if needed

    final_array = DS_array[:size[0],:size[1]]

    if AS_NP_ARRAY:
        return final_array
    else:
        return final_array.tolist()


def get_DS_size_and_iters(requested_size, max_power_of_two=13):
    """Returns the necessary size for a square grid which is usable in a DS algorithm.

    The Diamond Square algorithm requires a grid of size n x n where n = 2**x + 1, for any 
    integer value of x greater than two. To accomodate a requested map size other than these
    dimensions, we simply create the next largest n x n grid which can entirely contain the
    requested size, and return a subsection of it.

    This method computes that size.

    PARAMETERS
    ----------
    requested_size
        A 2D list-like object reflecting the size of grid that is ultimately desired.

    max_power_of_two
        an integer greater than 2, reflecting the maximum size grid that the algorithm can EVER
        attempt to make, even if the requested size is too big. This limits the algorithm to 
        sizes that are manageable, unless the user really REALLY wants to have a bigger one.
        The maximum grid size will have an edge of size  (2**max_power_of_two + 1)

    RETURNS
    -------
    An integer of value n, as described above.
    """
    if max_power_of_two < 3: max_power_of_two = 3

    largest_edge = max(requested_size)

    for power in range(1,max_power_of_two+1):
        d = (2**power) + 1
        if largest_edge <= d:
            return d, power

    #failsafe: no values in the dimensions array were allowed, so print a warning and return
    # the maximum size.
    d = 2**max_power_of_two + 1
    print("DiamondSquare Warning: Requested size was too large. Grid of size {0} returned""".format(d))
    return d, max_power_of_two


def diamond_step(DS_array, step_size, roughness):
    """Does the diamond step for a given iteration.

    During the diamond step, the diagonally adjacent cells are filled:

    Value   None   Value   None   Value  ...

    None   FILLING  None  FILLING  None  ...
 
    Value   None   Value   None   Value  ...

    ...     ...     ...     ...    ...   ...

    So we'll step with increment step_size over BOTH axes

    """

    #calculate where all the diamond corners are (the ones we'll be filling)
    half_step = step_size/2
    x_steps = range(half_step, DS_array.shape[0], step_size)
    y_steps = x_steps[:]


    for i in x_steps:
        for j in y_steps:
            if DS_array[i,j] == -1.0:
                #print(repr((i,j)))
                DS_array[i,j] = diamond_displace(DS_array, i, j, half_step, roughness)

def square_step(DS_array, step_size, roughness):
    """Does the square step for a given iteration.

    During the diamond step, the diagonally adjacent cells are filled:

     Value    FILLING    Value    FILLING   Value   ...

    FILLING   DIAMOND   FILLING   DIAMOND  FILLING  ...
 
     Value    FILLING    Value    FILLING   Value   ...

      ...       ...       ...       ...      ...    ...

    So we'll step with increment step_size over BOTH axes

    """

    half_step = step_size/2
    steps_x = range(          0, DS_array.shape[0], half_step)
    steps_y = range(          0, DS_array.shape[0], half_step)

    for i in steps_x:
        for j in steps_y:
            if DS_array[i,j] == -1.0:
                DS_array[i,j] = square_displace(DS_array, i, j, half_step, roughness)


#defines the midpoint displacement for the diamond step
def diamond_displace(DS_array, i, j, half_step, roughness):
    ul = DS_array[i-half_step, j-half_step]
    ur = DS_array[i-half_step, j+half_step]
    ll = DS_array[i+half_step, j-half_step]
    lr = DS_array[i+half_step, j+half_step]

    ave = (ul + ur + ll + lr)/4.0

    rand_val = random.uniform(0,1)

    return (roughness*rand_val + (1.0-roughness)*ave)


#defines the midpoint displacement for the square step
def square_displace(DS_array, i, j, half_step, roughness):
    _sum = 0.0
    divide_by = 4

    #check cell "above"
    if i - half_step >= 0:
        _sum += DS_array[i-half_step, j]
    else:
        divide_by -= 1

    #check cell "below"
    if i + half_step < DS_array.shape[0]:
        _sum += DS_array[i+half_step, j]
    else:
        divide_by -= 1

    #check cell "left"
    if j - half_step >= 0:
        _sum += DS_array[i, j-half_step]
    else:
        divide_by -= 1

    #check cell "right"
    if j + half_step < DS_array.shape[0]:
        _sum += DS_array[i, j+half_step]
    else:
        divide_by -= 1


    ave = _sum / divide_by

    rand_val = random.uniform(0,1)

    return (roughness*rand_val + (1.0-roughness)*ave)

