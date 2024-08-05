from sgfmill import sgf
from go import *
import torch
from features import getAllFeatures
# import matplotlib.pyplot as plt

colorCharToIndex = {'B': 1, 'W': -1, 'b': 1, 'w': -1}

# The preparePolicySgfFile function extracts the movement sequence of Go from the SGF file and generates corresponding
# features and strategy outputs for training machine learning models.
# The main steps include reading and parsing SGF files, extracting valid movements,
# generating features and policy outputs, checking the legality of movements, and converting data into PyTorch tensors
def preparePolicySgfFile(fileName):
    with open(fileName, 'rb') as f:
        game = sgf.Sgf_game.from_bytes(f.read())
    sequence = game.get_main_sequence()

    winnerChar = game.get_winner()

    validSequence = []
    for node in sequence:
        # print(node.get_move())
        move = node.get_move()
        if move[1]:
            validSequence.append(move)

    go = Go()

    # append go.board to inputData
    inputData = []
    policyOutput = []

    for move in validSequence:
        willPlayColor = colorCharToIndex[move[0]]
        x = move[1][0]
        y = move[1][1]
        inputData.append(getAllFeatures(go, willPlayColor))
        policyOutput.append(toDigit(x, y))

        if go.move(willPlayColor, x, y) == False:
            raise Exception('Invalid move')

    willPlayColor = -willPlayColor
    inputData.append(getAllFeatures(go, willPlayColor))
    policyOutput.append(19 * 19)  # pass

    # use torch to load data
    inputData = torch.tensor(np.array(inputData)).bool()
    policyOutput = torch.tensor(np.array(policyOutput)).long().reshape(-1)

    return inputData, policyOutput

# The prepareValueSgfFile function extracts the movement sequence of Go from the SGF file and generates
# corresponding features and value outputs for training the value network.
# The value output indicates that in the current chessboard state,
# if the winner walks, the value of that state is 1, otherwise it is 0.
def prepareValueSgfFile(fileName):
    with open(fileName, 'rb') as f:
        game = sgf.Sgf_game.from_bytes(f.read())
    sequence = game.get_main_sequence()

    winnerChar = game.get_winner()
    winner = colorCharToIndex[winnerChar]

    validSequence = []
    for node in sequence:
        # print(node.get_move())
        move = node.get_move()
        if move[1]:
            validSequence.append(move)

    go = Go()

    # append go.board to inputData

    for move in validSequence:
        willPlayColor = colorCharToIndex[move[0]]
        x = move[1][0]
        y = move[1][1]

        if go.move(willPlayColor, x, y) == False:
            raise Exception('Invalid move')

    willPlayColor = -willPlayColor
    valueInputData = np.array([getAllFeatures(go, willPlayColor)])
    valueOutput = np.array([winner == willPlayColor])

    # use torch to load data
    valueInputData = torch.tensor(valueInputData).bool()
    valueOutput = torch.tensor(valueOutput).long().reshape(-1)

    return valueInputData, valueOutput


def preparePolicyData(fileCount):
    with open('games/allValid.txt', 'r') as allValidFile:
        allValidLines = allValidFile.readlines()

    allInputData = []
    allPolicyOutput = []

    for sgfFile in allValidLines[:fileCount]:
        try:
            sgfFile = sgfFile.strip()
            inputData, policyOutput = preparePolicySgfFile(sgfFile)
            allInputData.append(inputData)
            allPolicyOutput.append(policyOutput)

        except KeyboardInterrupt:
            exit()
        except Exception as e:
            print(e)
            print('Error: ' + sgfFile)

    ansInputData = torch.cat(allInputData)
    ansPolicyOutput = torch.cat(allPolicyOutput)

    del allInputData
    del allPolicyOutput

    assert ansInputData.shape[0] == ansPolicyOutput.shape[0]

    torch.save((ansInputData, ansPolicyOutput), 'policyData.pt')


def prepareValueData(fileCount):
    with open('games/allValid.txt', 'r') as allValidFile:
        allValidLines = allValidFile.readlines()

    allValueInputData = []
    allValueOutput = []

    for sgfFile in allValidLines[:fileCount]:
        try:
            sgfFile = sgfFile.strip()
            valueInputData, valueOutput = prepareValueSgfFile(sgfFile)
            allValueInputData.append(valueInputData)
            allValueOutput.append(valueOutput)

        except KeyboardInterrupt:
            exit()
        except Exception:
            print('Error: ' + sgfFile)

    allValueInputData = torch.cat(allValueInputData)
    allValueOutput = torch.cat(allValueOutput)

    assert allValueInputData.shape[0] == allValueOutput.shape[0]

    torch.save((allValueInputData, allValueOutput), 'valueData.pt')


if __name__ == '__main__':
    preparePolicyData(2000)
    prepareValueData(2000)
