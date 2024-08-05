# Stone colour            3 Player stones; oppo. stones; empty
# Ones                    1 Constant plane
# Turns since last move   8 How many turns since a move played
# Liberties               8 Number of liberties
# Zeros                   1 Constant plane

# Extracting features from chess records for subsequent machine learning or pattern recognition tasks:
# ColorStoneFeatures (board, willPlayColor): Extract features of the position above the chessboard,
# the position of the chess piece about to land, and the position of the opponent's chess piece.
# OnesFeatures: Generate a constant planar feature that represents all positions on the chessboard.
# Liberty Features (length=8): Generate multiple features based on the qi of the chess piece,
# each feature being a Boolean array representing whether the qi of the chess piece is equal to a specific value.
# RecentOnehotFeatures (history, length=3): Generate multiple features based on recent historical records,
# each feature is a unique hot encoding array representing the position of the last few moves on the chessboard.
import numpy as np

def colorStoneFeatures(board, willPlayColor):
    # blank, this color, opponent color
    # If willPlayColor is 1 (assuming 1 represents black chess),
    # then the feature is [empty position, black chess position, white chess position].
    # If willPlayColor is -1 (assuming -1 represents white chess),
    # then the feature is [empty position, white chess position, black chess position].
    if willPlayColor == 1:
        features = [board == 0, board == 1, board == -1]
    else:
        features = [board == 0, board == -1, board == 1]
    return features


def onesFeatures():
    features = [np.ones((19, 19))]
    return features


# def turns_since_last_move_features(board):
#     features = [np.zeros(19, 19, dtype=np.int8)]
#     return features


def libertiesFeatures(liberty, length=8):
    features = []
    for i in range(1, length + 1):
        features.append(liberty == i)
    return features


def recentOnehotFeatures(history, length=3):
    features = []
    for item in history[-length:]:
        onehot = np.zeros((19, 19), dtype=np.int8)
        if item != (None, None):
            x, y = item
            onehot[x, y] = 1
        features.append(onehot)
    return features


def getAllFeatures(go, willPlayColor):
    board = go.board
    liberty = go.liberty
    history = go.history

    allFeatures = [
        colorStoneFeatures(board, willPlayColor),
        onesFeatures(),
        libertiesFeatures(liberty),
        # zeros_features(),
        recentOnehotFeatures(history)
    ]
    # combine all features
    features = []
    for feature in allFeatures:
        features += feature
    return np.array(features)
