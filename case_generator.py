import numpy as np


def generate_cases(x, valid, save_to_file=False):
    # Calculate half of x for even distribution
    half_x = x // 2
    nonSym = x - half_x

    symmetrical_numbers = []
    non_symmetrical_numbers = []
    binary_array = []

    # Helper function to check if a number is symmetrical
    def is_symmetrical(num):
        return str(num) == str(num)[::-1]

    # Generate symmetrical numbers
    while len(symmetrical_numbers) < half_x:
        num = np.random.randint(1000, 10000)  # Ensure a 4-digit number
        if is_symmetrical(num):
            symmetrical_numbers.append(num)
            binary_array.append(1)  # 1 for symmetrical

    # Generate non-symmetrical numbers
    while len(non_symmetrical_numbers) < nonSym:
        num = np.random.randint(1000, 10000)  # Ensure a 4-digit number
        if not is_symmetrical(num):
            non_symmetrical_numbers.append(num)
            binary_array.append(0)  # 0 for non-symmetrical

    # Combine and shuffle the arrays to mix symmetrical and non-symmetrical numbers
    combined_numbers = symmetrical_numbers + non_symmetrical_numbers
    combined_pairs = list(zip(combined_numbers, binary_array))
    np.random.shuffle(combined_pairs)

    # Unzip the shuffled pairs back into two lists
    shuffled_numbers, shuffled_binary_array = zip(*combined_pairs)

    if save_to_file:
        if valid:
            np.save('validationSym.npy', shuffled_numbers)
            np.save('validationBinary.npy', shuffled_binary_array)
        else:
            np.save('symmetrical_numbers.npy', shuffled_numbers)
            np.save('binary_array.npy', shuffled_binary_array)

    return list(shuffled_numbers)[:x], list(shuffled_binary_array)[:x]
