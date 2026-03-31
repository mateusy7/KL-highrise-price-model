import re
import matplotlib.pyplot as plt

def clean_floor(level):
    # Store the processed string to return as a fallback
    original_input = str(level).strip().upper()
    
    mapping = {
        '3A': '4', '13A': '14', '23A': '24', 'G': '0', 'B': '-1',
        'MZ': '0', 'D':'0', 'UG': '0', 'P': '0', 'LG': '0'
    }
    
    # Check mapping first
    current_val = mapping.get(original_input, original_input)
    
    # Extract first number from ranges (e.g., '1-5' -> 1)
    match = re.search(r'\d+', current_val)
    
    # If a digit is found, return it as an int; otherwise, return the string
    return int(match.group()) if match else None

def plot_num_count(df, col_list):
    length = len(col_list)
    ncolumns = 3
    nrows = (length + (ncolumns-1)) // ncolumns

    fig, axs = plt.subplots(nrows, ncolumns, figsize=(12,3*nrows))
    axs = axs.flatten()

    for i in range(length):
        feature = col_list[i]
        axs[i].hist(df[feature], bins=50)
        axs[i].set_xlabel(feature)

    for j in range(i+1, len(axs)):
        axs[j].axis('off')

    fig.suptitle('Frequency Distribution')
    plt.tight_layout()
    plt.show()

def plot_category_count(df, col_list):
    length = len(col_list)
    ncolumns = 3
    nrows = (length + (ncolumns-1)) // ncolumns

    fig, axs = plt.subplots(nrows, ncolumns, figsize=(12,3*nrows))
    axs = axs.flatten()

    for i in range(length):
        feature = col_list[i]
        plot = df[feature].value_counts()
        axs[i].bar(plot.index, plot)
        axs[i].set_xlabel(feature)

        axs[i].set_xticks(range(len(plot.index)))
        axs[i].set_xticklabels(plot.index, rotation=45, ha='right', fontsize='7')
    
    for j in range(i+1, len(axs)):
        axs[j].axis('off')

    fig.suptitle('Frequency Distribution')
    plt.show()