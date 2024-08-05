# open 'all.txt' and read each line as sgfFile
# if sgfFile doesn't have 'HA[', append to allValidFile
"""
The findSgfFiles() function recursively searches for all. sgf files in a specified path,
And store the paths of these files in a list. This function returns a list containing all found. sgf file paths
1. Search for all. sgf files in the games directory and its subdirectories.
2. Check these files one by one, filtering out games that do not contain any moves and have dates after 2000.
3. Save the file paths that meet the criteria to the allValidFile list.
4. Write all valid file paths to the allValid. txt file in the games directory.
5. Progress and total number of print processing files.
"""

import os


# find all sgf files in games/
def findSgfFiles(path):
    sgfFiles = []
    for root, dirs, files in os.walk(path):
        for file in files:
            if file.endswith('.sgf'):
                sgfFiles.append(os.path.join(root, file))
    return sgfFiles


allValidFile = []

count = 0

allSgfFiles = findSgfFiles('games')
for sgfFile in allSgfFiles:
    try:
        with open(sgfFile, 'r') as sgf:
            data = sgf.read()
        # DT[2017-08-24] date > 2000
        if 'HA[' not in data and 'DT[20' in data:
            allValidFile.append(sgfFile)
    except:
        print('Error: ' + sgfFile)

    # Every time a file is processed, the counter count is incremented by 1.
    count += 1
    if count % 10000 == 0:
        print('Processed ' + str(count) + ' files')

# write allValidFile to 'allValid.txt'
with open('games/allValid.txt', 'w') as allValid:
    for sgfFile in allValidFile:
        allValid.write(sgfFile)
        allValid.write('\n')

print('Total:', len(allValidFile))
