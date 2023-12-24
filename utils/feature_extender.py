import pandas as pd

def calculate_odd_even_features(N, Z):
    return {
        'is_odd_odd': int(N % 2 != 0 and Z % 2 != 0),
        'is_even_even': int(N % 2 == 0 and Z % 2 == 0),
        'is_odd_even': int(((N % 2 != 0) and (Z % 2 == 0)) or ((N % 2 == 0) and (Z % 2 != 0))),
        'is_even_odd': int(N % 2 != 0 and Z % 2 == 0),
    }
def add_individual_shell_features_extended(df, column_name, shell_count):
    """
    Add individual shell occupancy as separate columns to the DataFrame for extended shells.

    Args:
    df (DataFrame): The original DataFrame.
    column_name (str): The name of the column containing shell occupancy lists.
    shell_count (int): The number of shells to consider.

    Returns:
    DataFrame: The updated DataFrame with individual shell occupancy columns.
    """
    for i in range(shell_count):
        # Extracting the i-th element from each list in the column
        df[f'{column_name}_shell_{i+1}'] = df[column_name].apply(lambda x: x[i] if i < len(x) else 0)

    return df
def convert_nz_to_features(data):
    """
    Convert a dataset with columns N and Z to a dataset with 26 nuclear feature columns.

    Args:
    data (numpy array): A 2D numpy array where each row contains Z and N.

    Returns:
    DataFrame: A pandas DataFrame containing the 26 nuclear features.
    """
    def calculate_shell_occupancy(number, magic_numbers):
        shell_occupancy = [0] * len(magic_numbers)
        remaining = number

        for i, capacity in enumerate(magic_numbers):
            if remaining > capacity:
                shell_occupancy[i] = capacity
                remaining -= capacity
            else:
                shell_occupancy[i] = remaining
                break

        return shell_occupancy

    def calculate_atomic_features(N, Z):
        magic_numbers_protons = [2, 8, 20, 28, 50, 82, 126]
        magic_numbers_neutrons = [2, 8, 20, 28, 50, 82, 126, 184]  # Extending for the 8th shell for neutrons

        A = Z + N
        proton_distribution = calculate_shell_occupancy(Z, magic_numbers_protons)
        neutron_distribution = calculate_shell_occupancy(N, magic_numbers_neutrons)

        # Other features
        valence_protons = proton_distribution[-1]
        valence_neutrons = neutron_distribution[-1]
        N_minus_Z = N - Z
        A_two_thirds = A ** (2/3)
        A_minus_one_third = A ** (-1/3)
        is_magic_Z = 1 if Z in magic_numbers_protons else 0
        is_magic_N = 1 if N in magic_numbers_neutrons else 0
        pair_energy = (-1)**Z + (-1)**N

        features = {
            "N": N,
            "Z": Z,
            "A": A,
            "proton_distribution": proton_distribution,
            "neutron_distribution": neutron_distribution,
            "valence_protons": valence_protons,
            "valence_neutrons": valence_neutrons,
            "N_minus_Z": N_minus_Z,
            "A_two_thirds": A_two_thirds,
            "A_minus_one_third": A_minus_one_third,
            "is_magic_Z": is_magic_Z,
            "is_magic_N": is_magic_N,
            "pair_energy": pair_energy
        }
        return features

    # Convert the numpy array to a pandas DataFrame
    df = pd.DataFrame(data, columns=['N', 'Z'])

    # Apply the calculate_atomic_features function to each row
    features_df = df.apply(lambda row: calculate_atomic_features(row['N'], row['Z']), axis=1).apply(pd.Series)

    # Add individual shell features
    features_df = add_individual_shell_features_extended(features_df, 'proton_distribution', 7)
    features_df = add_individual_shell_features_extended(features_df, 'neutron_distribution', 8)

    # Apply the calculate_odd_even_features function to each row
    odd_even_features_df = df.apply(lambda row: calculate_odd_even_features(row['N'], row['Z']), axis=1).apply(pd.Series)
    features_df = pd.concat([features_df, odd_even_features_df], axis=1)

    # Drop the original distribution columns
    features_df.drop(['proton_distribution', 'neutron_distribution'], axis=1, inplace=True)

    return features_df

# Example usage
# nucleus_data_example = np.array([[20, 20], [8, 8], [50, 50]])  # Example data
# result_features_df = convert_nz_to_features(nucleus_data_example)
# result_features_df.head()
