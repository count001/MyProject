from go import *
from prepareData import *
from net import *
import sys
import csv
import scipy.stats

# use cuda if available
device = torch.device('cpu')
if torch.cuda.is_available():
    device = torch.device('cuda')


# set random seed
def setRandomSeed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)


setRandomSeed(0)


def splitData(inputData, outputData, ratio):
    length = len(inputData)
    trainLength = int(length * ratio)
    trainInputData, testInputData = inputData[:trainLength], inputData[trainLength:]
    trainOutputData, testOutputData = outputData[:trainLength], outputData[trainLength:]

    trainPermutation = torch.randperm(len(trainInputData))
    trainInputData = trainInputData[trainPermutation]
    trainOutputData = trainOutputData[trainPermutation]

    testPermutation = torch.randperm(len(testInputData))
    testInputData = testInputData[testPermutation]
    testOutputData = testOutputData[testPermutation]

    del inputData, outputData, trainPermutation, testPermutation
    return trainInputData, trainOutputData, testInputData, testOutputData

def calculateEntropy(predictions, epsilon=1e-10):
    predictions = np.clip(predictions, epsilon, 1.0)
    entropy_values = -np.sum(predictions * np.log(predictions), axis=1)
    return entropy_values

def trainPolicy(net, outputFileName, epoch=10):
    optimizer = torch.optim.SGD(net.parameters(), lr=0.01, momentum=0.9)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=20, gamma=0.1)
    loss_function = nn.CrossEntropyLoss()

    inputData, outputData = torch.load('policyData.pt')
    trainInputData, trainOutputData, testInputData, testOutputData = splitData(inputData, outputData, 0.8)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    net.to(device)

    batchSize = 100
    trainBatchCount = int(len(trainInputData) / batchSize)
    testBatchCount = int(len(testInputData) / batchSize)
    logInterval = 1000

    # Save the accuracy and loss of training and testing
    training_metrics = []

    for epoch in range(epoch):
        totalLoss = 0
        totalCorrectCount = 0
        train_entropy_values = []

        # training
        for i in range(trainBatchCount):
            inputDataBatch = trainInputData[i * batchSize:(i + 1) * batchSize]
            outputDataBatch = trainOutputData[i * batchSize:(i + 1) * batchSize].reshape(-1)
            inputDataBatch = inputDataBatch.to(device)
            outputDataBatch = outputDataBatch.to(device)

            output = net(inputDataBatch)
            correctCount = torch.sum(torch.argmax(output, dim=1) == outputDataBatch).item()
            totalCorrectCount += correctCount

            # Calculate entropy value
            # train_entropy_values.extend(calculateEntropy(output.detach().cpu().numpy()))

            loss = loss_function(output, outputDataBatch)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            totalLoss += loss.item()

            if i % logInterval == 0 and i != 0:
                correctRate = totalCorrectCount / (logInterval * batchSize)
                avgLoss = totalLoss / logInterval
                print(
                    f'train: epoch: {epoch:3}   batch: {i:>5}   correctRate: {correctRate:.2%}   avgLoss: {avgLoss:.2f}')
                totalCorrectCount = 0
                totalLoss = 0

        scheduler.step()

        # test
        totalCorrectCount = 0
        totalLoss = 0
        test_entropy_values = []

        with torch.no_grad():
            for i in range(testBatchCount):
                testInputDataBatch = testInputData[i * batchSize:(i + 1) * batchSize]
                testOutputDataBatch = testOutputData[i * batchSize:(i + 1) * batchSize].reshape(-1)
                testInputDataBatch = testInputDataBatch.to(device)
                testOutputDataBatch = testOutputDataBatch.to(device)

                output = net(testInputDataBatch)
                correctCount = torch.sum(torch.argmax(output, dim=1) == testOutputDataBatch).item()
                totalCorrectCount += correctCount

                # Calculate entropy value
                # test_entropy_values.extend(calculateEntropy(output.detach().cpu().numpy()))

                loss = loss_function(output, testOutputDataBatch)
                totalLoss += loss.item()

            testCorrectRate = totalCorrectCount / len(testInputData)
            testAvgLoss = totalLoss / len(testInputData) * batchSize
            learningRate = optimizer.param_groups[0]['lr']
            print(
                f'test: epoch: {epoch:3}               correctRate: {testCorrectRate:>2.2%}   avgLoss: {testAvgLoss:.2f}   '
                f'learningRate: {learningRate}')

        # Save the training and testing results for each round
        training_metrics.append({
            'epoch': epoch,
            'train_correct_rate': correctRate,
            'train_loss': avgLoss,
            'test_correct_rate': testCorrectRate,
            'test_loss': testAvgLoss,
            # 'train_entropy': sum(train_entropy_values) / len(train_entropy_values),
            # 'test_entropy': sum(test_entropy_values) / len(test_entropy_values)
        })

        torch.save(net.state_dict(), outputFileName)

    # 保存训练过程数据到CSV文件
    with open('training_metrics.csv', 'w', newline='') as csvfile:
        fieldnames = ['epoch', 'train_correct_rate', 'train_loss', 'test_correct_rate', 'test_loss', 'train_entropy',
                      'test_entropy']
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()
        for data in training_metrics:
            writer.writerow(data)


# valueData
def trainValue(net, outputFileName, epoch=10):
    optimizer = torch.optim.Adam(net.parameters(), lr=0.001)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=2, gamma=0.5)
    loss_function = nn.MSELoss()

    inputData, outputData = torch.load('valueData.pt')

    # selectInterval = 5
    # inputData = inputData[::selectInterval]
    # outputData = outputData[::selectInterval]

    trainInputData, trainOutputData, testInputData, testOutputData = splitData(inputData, outputData, 0.8)

    # use cuda to train
    net.to(device)

    # batch size
    batchSize = 100
    batchCount = int(len(trainInputData) / batchSize)

    logInterval = 1000

    testBatchCount = int(len(testInputData) / batchSize)
    totalLoss = 0
    totalCorrectCount = 0

    for epoch in range(epoch):
        totalLoss = 0
        totalCorrectCount = 0

        for i in range(batchCount):
            # get batch data
            inputDataBatch = trainInputData[i * batchSize:(i + 1) * batchSize]
            outputDataBatch = trainOutputData[i * batchSize:(i + 1) * batchSize].reshape(-1)

            # use cuda to train
            inputDataBatch = inputDataBatch.to(device)
            outputDataBatch = outputDataBatch.to(device)

            # forward
            output = net(inputDataBatch)
            outputInt = torch.round(output)
            correctCount = torch.sum(outputInt == outputDataBatch).item()
            totalCorrectCount += correctCount

            # backward
            loss = loss_function(output, outputDataBatch.float())
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            totalLoss += loss.item()

            # print
            if i % logInterval == 0 and i != 0:
                avgLoss = totalLoss / logInterval
                correctRate = totalCorrectCount / (logInterval * batchSize)
                print(f'epoch: {epoch:3}   batch: {i:>5}   correctRate: {correctRate:.2%}   avgLoss: {avgLoss:.2f}')
                totalCorrectCount = 0
                totalLoss = 0

        scheduler.step()

        totalCorrectCount = 0
        totalLoss = 0

        # test
        with torch.no_grad():
            for i in range(testBatchCount):
                testInputDataBatch = testInputData[i * batchSize:(i + 1) * batchSize]
                testOutputDataBatch = testOutputData[i * batchSize:(i + 1) * batchSize].reshape(-1)

                testInputDataBatch = testInputDataBatch.to(device)
                testOutputDataBatch = testOutputDataBatch.to(device)

                output = net(testInputDataBatch)
                outputInt = torch.round(output)
                correctCount = torch.sum(outputInt == testOutputDataBatch).item()
                totalCorrectCount += correctCount

                loss = loss_function(output, testOutputDataBatch)
                totalLoss += loss.item()

            correctRate = totalCorrectCount / len(testInputData)
            avgLoss = totalLoss / len(testInputData) * batchSize
            learningRate = optimizer.param_groups[0]['lr']
            print(f'epoch: {epoch:3}                  correctRate: {correctRate:>2.2%}   avgLoss: {avgLoss:.2f}   '
                  f'learningRate: {learningRate}')
        # save net
        torch.save(net.state_dict(), outputFileName)


# python3 train.py policyNet
if len(sys.argv) == 2:
    if sys.argv[1] == 'policyNet':
        net = PolicyNetwork()
        trainPolicy(net, 'policyNet.pt', 60)
    elif sys.argv[1] == 'playoutNet':
        net = PlayoutNetwork()
        trainPolicy(net, 'playoutNet.pt', 30)
    elif sys.argv[1] == 'valueNet':
        net = ValueNetwork()
        trainValue(net, 'valueNet.pt', 30)
else:
    net = PolicyNetwork()
    trainValue(net, 'policyNet.pt', 8)
