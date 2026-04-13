import csv

def calculate_change_rate(file1, file2):
    """
        Calculate the rate of change of the type of the lattice IDs in the two CSV files containing the lattice IDs that appear only in one of the files
        :param file1: path to the first CSV file
        :param file2: path to the second CSV file
        :return: rate of change
    """
    # Read the first file
    with open(file1, mode='r', encoding='utf-8') as f1:
        reader1 = csv.reader(f1)
        headers1 = next(reader1)
        types1 = next(reader1)
        id_type_map1 = dict(zip(headers1, types1))

    # Read the second file
    with open(file2, mode='r', encoding='utf-8') as f2:
        reader2 = csv.reader(f2)
        headers2 = next(reader2)
        types2 = next(reader2)
        id_type_map2 = dict(zip(headers2, types2))

    # Get all lattice IDs (concatenation)
    all_ids = set(id_type_map1.keys()) | set(id_type_map2.keys())

    # Counting the number of changed grids
    changes = 0
    for grid_id in all_ids:
        type1 = id_type_map1.get(grid_id)
        type2 = id_type_map2.get(grid_id)
        if type1 != type2:
            changes += 1

    # Rate of change = number of grids changed / total number of grids
    total = len(all_ids)
    change_rate = changes / total if total > 0 else 0
    return change_rate


file1 = 'cluster_128_a.csv'
file2 = 'cluster_128_b.csv'
change_rate = calculate_change_rate(file1, file2)
print(f"格子类型变化率: {change_rate:.3%}")