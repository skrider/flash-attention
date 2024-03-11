import re
import matplotlib.pyplot as plt

# LLM
def parse_log(log_file):
    rowcol_pattern = re.compile(r'ROWCOL: \((\d+), (\d+)\)')
    # col_pattern = re.compile(r'COL: \((\d+)\)')
    rowcol_pairs = set()
    # cols = set()

    with open(log_file, 'r') as f:
        for line in f:
            rowcol_match = rowcol_pattern.search(line)
            if rowcol_match:
                row, col = int(rowcol_match.group(1)), int(rowcol_match.group(2))
                rowcol_pairs.add((row, col))
            # col_match = col_pattern.search(line)
            # if col_match:
            #     col = int(col_match.group(1))
            #     cols.add(col)

    mask = [[1] * (256) for _ in range(64)]

    for row, col in rowcol_pairs:
        mask[row][col] = 0

    plt.figure()
    plt.imshow(mask, cmap='binary')
    plt.savefig('mask.png', dpi=300, bbox_inches='tight')

    return mask

# Example usage
mask = parse_log('last_run.log')
# for row in mask:
#     print(row)