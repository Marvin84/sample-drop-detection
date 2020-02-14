import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim


from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn import metrics
from utils import *

from models import *
from config import *
import pickle

np.random.seed(47)


#######################################
# General
######################################

#function implemented by João Monteiro
def write_pickle(data, filename):
    with open(filename, "wb") as f:
        pickle.dump(data, f, protocol=2)




def compute_eer(y, y_score):
    #This code is written by João Monteiro but was not used for the paper
    fpr, tpr, thresholds = metrics.roc_curve(y, y_score, pos_label=1)
    fnr = 1 - tpr
    t = np.nanargmin(np.abs(fnr-fpr))
    eer_low, eer_high = min(fnr[t],fpr[t]), max(fnr[t],fpr[t])
    eer = (eer_low+eer_high)*0.5
    return eer


def print_results(testy, yhat_classes):
    #embed()
    accuracy = accuracy_score(testy, yhat_classes)
    print('Accuracy: %f' % accuracy)
    # precision tp / (tp + fp)
    precision = precision_score(testy, yhat_classes)
    print('Precision: %f' % precision)
    # recall: tp / (tp + fn)
    recall = recall_score(testy, yhat_classes)
    print('Recall: %f' % recall)
    # f1: 2 tp / (2 tp + fp + fn)
    f1 = f1_score(testy, yhat_classes)
    print('F1 score: %f' % f1)



def get_f1(testy, yhat_classes):

    f1 = f1_score(testy, yhat_classes)
    print('F1 score: %f' % f1)
    return f1


def save_model(model, epoch, lossValues, f1scores, preTrained=False, errors=None, isAttention=config_isAttention):

    if preTrained:
        modelName = (".").join([("_").join([config_pretrainedmodelPath, str(epoch)]), "pth"])
    else: modelName = (".").join([("_").join([config_modelPath, str(epoch)]), "pth"])
    if isAttention:
        modelName+="_att"
    if not config_isPretrained:
        modelName+="_noPre"


    torch.save(model.state_dict(), modelName)

    with open(config_lossesPath, 'wb') as lossFile:
        pickle.dump(lossValues, lossFile, protocol=2)

    with open(config_f1ScoresPath, 'wb') as f1File:
        pickle.dump(f1scores, f1File, protocol=2)

    if errors is not None:
        with open(config_errorsPath, 'wb') as errorFile:
            pickle.dump(errors, errorFile, protocol=2)







######################################
# Contaminated LibriSpeech Two-Channel
#####################################


def get_batch_from_segments_train_cnn(segments, batchSize):
    index = 0
    while True:
        try:
            data = []
            for ind, s in enumerate(segments[index*batchSize:(index+1)*batchSize]):
                tensorHyp = torch.from_numpy(s.hyp).float().view(1, 9, config_fqValues)
                tensorRef = torch.from_numpy(s.ref).float().view(1, 39, config_fqValues)
                if s.label == 1:
                    target = torch.ones(1, 1)
                else:
                    target = torch.zeros(1, 1)
                data.append([tensorHyp, tensorRef, target])

            yield data[index * batchSize: (index + 1) * batchSize]
            index += 1
        except IndexError:
            index = 0



def get_batch_from_segments_train_rnn(segments, batchSize, isLog=False):
    #the batching structure was suggested by Samuele Cornell
    index = 0
    while True:
        try:
            data = []
            for ind, s in enumerate(segments[index*batchSize:(index+1)*batchSize]):
                if isLog:
                    h = 20*np.log10(s[0])
                    r = 20*np.log10(s[1])
                else:
                    h,r = s[0], s[1]

                tensorHyp = torch.from_numpy(h).float().view(1, s[0].shape[0], s[0].shape[1])
                tensorRef = torch.from_numpy(r).float().view(1, s[1].shape[0], s[1].shape[1])
                if s[2] == 1:
                    target = torch.ones(1, 1)
                else:
                    target = torch.zeros(1, 1)
                #tensorHyp = torch.from_numpy(s.hyp).float().view(1, 39, config_freqVal)
                #tensorRef = torch.from_numpy(s.ref).float().view(1, 39, config_freqVal)
                #if s.label == 1:
                #    target = torch.ones(1, 1)
                #else:
                #    target = torch.zeros(1, 1)
                #target = torch.from_numpy(np.array(s.label)).float()
                data.append([tensorHyp, tensorRef, target])

            yield data[index * batchSize: (index + 1) * batchSize]
            index += 1
        except IndexError:
            index = 0




def get_scores_from_eval_rnn(model, segments):

    model.eval()
    mapDict = []

    with torch.no_grad():
        for s  in segments:
            tensorHyp = torch.from_numpy(s[0]).float().view(1, s[0].shape[0], s[0].shape[1])
            tensorRef = torch.from_numpy(s[1]).float().view(1, s[1].shape[0], s[0].shape[1])
            if s[2] == 1:
                target = torch.ones(1, 1)
            else:
                target = torch.zeros(1, 1)

            output = model(tensorHyp, tensorRef)
            if output.item() > 0.5:
                predicted = torch.ones(1, 1)
            else:
                predicted = torch.zeros(1, 1)

            mapDict.append([predicted.item(), output.item(), target.item()])


    predictions = [p[0] for p in mapDict]
    outputScores = [p[1] for p in mapDict]
    testLabels = [p[2] for p in mapDict]

    error = compute_eer(testLabels, outputScores)
    f1score = f1_score(testLabels, predictions)

    print("f1score: ", f1score, "and error: ", error)




def get_scores_from_dev_rnn(model, segments):


    model.eval()
    mapDict = []

    with torch.no_grad():
        for s  in segments:
            tensorHyp = torch.from_numpy(s[0]).float().view(1, s[0].shape[0], s[0].shape[1]).cuda()
            tensorRef = torch.from_numpy(s[1]).float().view(1, s[1].shape[0], s[1].shape[1]).cuda()
            if s[2] == 1:
                target = torch.ones(1, 1)
            else:
                target = torch.zeros(1, 1)

            output = model(tensorHyp, tensorRef)
            if output.item() > 0.5:
                predicted = torch.ones(1, 1)
            else:
                predicted = torch.zeros(1, 1)

            mapDict.append([predicted.item(), output.item(), target.item()])


    predictions = [p[0] for p in mapDict]
    outputScores = [p[1] for p in mapDict]
    testLabels = [p[2] for p in mapDict]

    #error = compute_eer(testLabels, outputScores)
    f1score = f1_score(testLabels, predictions)

    return f1score


def evaluate_model_with_dev_segments(modelPath, segments):

    #normalizer = Normalizer([s.ref for s in segments]+[s.hyp for s in segments])
    #print("Starting with the normalization.")
    #normalizer.normalize(segments)
    #print("Normalization done.")


    evalNet = RNNSampleLoss(num_inputs=segments[0][0].shape[1])
    evalNet.load_state_dict(torch.load(modelPath))
    evalNet.eval()


    mapDict = []
    with torch.no_grad():
        for ind, s  in enumerate(segments):

            tensorHyp = torch.from_numpy(s.hyp).float().view(1, s.hyp.shape[0], s.hyp.shape[1])
            tensorRef = torch.from_numpy(s.ref).float().view(1, s.ref.shape[0], s.ref.shape[1])
            if s.label == 1:
                target = torch.ones(1, 1)
            else:
                target = torch.zeros(1, 1)

            output = evalNet(tensorHyp, tensorRef)
            if output.item() > 0.5:
                predicted = torch.ones(1, 1)
            else:
                predicted = torch.zeros(1, 1)

            a = [predicted.item(), target.item(), output.item()]
            mapDict.append(a)


    predictions = [p[0] for p in mapDict]
    testLabels = [p[1] for p in mapDict]
    scores = [p[2] for p in mapDict]
    error = compute_eer(testLabels, scores)
    print("error: ", error)
    print_results(testLabels, predictions)


    with open(config_predictionsPath, "wb") as f:
        pickle.dump(mapDict, f, protocol=2)

def get_lr(optimizer):
    for param_group in optimizer.param_groups:
        return param_group['lr']

def train_model_with_segments_with_hdf(segments, isPretrained=True):

    net = RNNSampleLoss(num_inputs=segments[0][0].shape[1])
    net.cuda()
    criterion = nn.BCELoss().cuda()
    opt = optim.Adam(net.parameters(), lr=config_learningRate)

    divisionIndex = int(config_devPercentage * len(segments))

    devSegments = segments[:divisionIndex]
    trainSegments = segments[divisionIndex+1:]

    lossValues = []
    #netErrors = []
    evalF1Scores = []

    for epoch in range(0, config_epochs):

        np.random.shuffle(trainSegments)

        for dataSpliter in range(0, len(trainSegments), config_trainStep):

            #print("Starting with the preparation of the training set for start index", dataSpliter)
            trainS = trainSegments[dataSpliter:dataSpliter + config_trainStep]

            #print("Start training on the subset of indices." +
            #  str(dataSpliter) + "-" + str(dataSpliter + config_trainStep) + ". total segments for this subset: " + str(
            #    len(trainS)))
        # for now number of batch is number of
            for batch in get_batch_from_segments_train_rnn(trainS, config_batchSize):

                if (len(batch) == 0):
                    break
                hypX = torch.cat([d[0].cuda() for d in batch])
                refX = torch.cat([d[1].cuda() for d in batch])
                Y = torch.cat([d[2].cuda() for d in batch])

                opt.zero_grad()
                output = net(hypX, refX)
                loss = criterion(output.view(output.shape[0]), Y)
                #print("loss: ", loss.item())

                lossValues.append(loss.item())
                loss.backward()
                opt.step()


        f1score = get_scores_from_dev_rnn(net, devSegments)
        print("f1 score and the lr after the epoch ", epoch, " are: ", f1score, ", ", get_lr(opt))
        print("")
        evalF1Scores.append(f1score)
        #netErrors.append(error)

        net.train()
        save_model(net, epoch, lossValues, evalF1Scores, preTrained=isPretrained)




def train_model_with_segments(segments, net, criterion, opt):

    normalizer = Normalizer([s.ref for s in segments]+[s.hyp for s in segments])
    print("Starting with the normalization.")
    normalizer.normalize(segments)
    print("Normalization done.")
    divisionIndex = int(1-config_devPercentage * len(segments))

    trainSegments = segments[:divisionIndex]
    devSegments = segments[divisionIndex + 1:]

    trainStep = 200
    losseValues = []
    evalF1Scores = []

    np.random.shuffle(trainSegments)
    for dataSpliter in range(0, len(trainSegments), trainStep):

        print("Starting with the preparation of the training set for start index", dataSpliter)
        trainS = trainSegments[dataSpliter:dataSpliter + trainStep]

        print("Start training on the subset of indices." +
          str(dataSpliter) + "-" + str(dataSpliter + trainStep) + ". total segments for this subset: " + str(
            len(trainS)))
    # for now number of batch is number of
        for batch in get_batch_from_segments_train_rnn(trainS, config_batchSize):
            #print("got batch", len(batch))
            if (len(batch) == 0):
                break
            hypX = torch.cat([d[0].cuda() for d in batch])
            refX = torch.cat([d[1].cuda() for d in batch])
            Y = torch.cat([d[2].cuda() for d in batch])

            opt.zero_grad()
            output = net(hypX, refX)
            loss = criterion(output.view(output.shape[0]), Y)

            lossValue = loss.item()
            loss.backward()
            opt.step()

            losseValues.append(lossValue)
            print(lossValue)

    ###################################

    return net, losseValues, evalF1Scores




######################################
# Multi-device short scenarios
#####################################


def print_scores_for_multidevice(finalPredicts, targets):
    valueMaps = []
    for k in targets.keys():
        for deviceKey in targets[k]:
            valueMaps.append([finalPredicts[k][deviceKey], targets[k][deviceKey][0]["label"]])

    predicted = [e[0] for e in valueMaps]
    real = [e[1] for e in valueMaps]
    print_results(real, predicted)

def evaluate_model_with_dev_segments_multiple_devices(modelPath, targets, deviceSegments):

    predictions = dict(zip(list(deviceSegments.keys()), [{} for _ in range(len(deviceSegments.keys()))]))
    outputValues = dict(zip(list(deviceSegments.keys()), [{} for _ in range(len(deviceSegments.keys()))]))
    # segments = []



    evalNet = RNNSampleLoss(num_inputs=config_fqValues)
    evalNet.load_state_dict(torch.load(modelPath, map_location="cpu"))
    evalNet.eval()


    with torch.no_grad():
        for key, value  in deviceSegments.items():
            for deviceId, segmentArray in value.items():
                if len(deviceSegments[key][deviceId]) < 1:
                    continue
                predictions[key][deviceId] = []
                predictionsForDevice = []

                for s in segmentArray:

                    tensorHyp = torch.from_numpy(s.hyp).float().view(1, s.hyp.shape[0], s.hyp.shape[1])
                    tensorRef = torch.from_numpy(s.ref).float().view(1, s.ref.shape[0], s.ref.shape[1])
                    #if s.label == 1:
                    #    target = torch.ones(1, 1)
                    #else:
                    #    target = torch.zeros(1, 1)

                    output = evalNet(tensorHyp, tensorRef)
                    #if output.item() > 0.5:
                    #    predicted = torch.ones(1, 1)
                    #else:
                    #    predicted = torch.zeros(1, 1)

                    predictionsForDevice.append(output.item())



                p = np.mean(predictionsForDevice)
                m = np.median(predictionsForDevice)
                # majorP = np.sum([int(i<0.5) for i in predictionsForDevice])
                #print(predictionsForDevice, targets[key][deviceId][0]["label"], p, m)

                if p < 0.5:
                #if m < 0.5:
                    predictedValue = 0
                else: predictedValue = 1
                #print(predictedValue,targets[key][deviceId][0]["label"])


                predictions[key][deviceId].append(predictedValue)
                outputValues[key][deviceId] = predictionsForDevice

    mapList = []
    for key, value in predictions.items():
        for deviceId in value.keys():
            mapList.append([predictions[key][deviceId][0], targets[key][deviceId][0]["label"]])

    write_pickle(predictions, sceneName+"predictions.pickle")
    write_pickle(outputValues, sceneName+"outputs.pickle")



    predictions = [p[0] for p in mapList]
    testLabels = [p[1] for p in mapList]


    print("using epoch: ", modelPath.split(".pth")[0][-3:])
    print_results(testLabels, predictions)







def evaluate_model_with_dev_segments_chime_utt(modelPath, deviceSegments, mean, std):

    predictions = {}
    labels = {}


    evalNet = RNNSampleLoss(num_inputs=config_fqValues)
    evalNet.load_state_dict(torch.load(modelPath, map_location="cpu"))
    evalNet.eval()


    with torch.no_grad():
        for winKey in deviceSegments.keys():
            labels[winKey] = deviceSegments[winKey][0].label
            predictions[winKey] = []

            predictionsForDevice = []
            for s in deviceSegments[winKey]:
                h = (s.hyp - mean)/std
                r = (s.ref -mean)/std

                tensorHyp = torch.from_numpy(h).float().view(1, s.hyp.shape[0], s.hyp.shape[1])
                tensorRef = torch.from_numpy(r).float().view(1, s.ref.shape[0], s.ref.shape[1])

                output = evalNet(tensorHyp, tensorRef)

                predictionsForDevice.append(output.item())



            p = np.median(predictionsForDevice)
            print(predictionsForDevice)
            if p < 0.5:
                predictedValue = 0
            else: predictedValue = 1

            predictions[winKey].append(predictedValue)

    mapList = []
    for key, value in predictions.items():
        mapList.append([value[0], labels[key]])




    predictions = [p[0] for p in mapList]
    testLabels = [p[1] for p in mapList]
    return get_f1(testLabels, predictions)


    #    scores = [p[2] for p in mapDict]
    #    error = compute_eer(testLabels, scores)
    #    print("error: ", error)
    #print_results(testLabels, predictions)

    #finalPredicts = dict(zip(list(deviceSegments.keys()), [{} for _ in range(len(deviceSegments.keys()))]))
    #for key, value in deviceSegments.items():
    #    for deviceId, segmentArray in value.items():
    #        if len(predictions[key][deviceId]) > 0:
    #            finalPredicts[key][deviceId] = mode(predictions[key][deviceId])[0][0]
    # print_scores_for_multidevice(finalPredicts, targets)

    #with open(config_predictionsPath, "wb") as f:
     #   pickle.dump(mapList, f, protocol=2)










def train_model_with_segments_with_hdf_and_pretrained(modelPath, segments, startEpoch=0):

    if config_isAttention:
        modelPath+="_att"

    net = RNNSampleLoss(num_inputs=segments[0][0].shape[1])
    net.load_state_dict(torch.load(modelPath, map_location='cpu') )
    net.train()

    criterion = nn.BCELoss()
    opt = optim.Adam(net.parameters(), lr=config_learningRate)


    trainSegments = segments[:]
    #devSegments = segments[0:100]


    lossValues = []
    netErrors = []
    evalF1Scores = []


    for epoch in range(startEpoch,config_epochs):

        np.random.shuffle(trainSegments)

        for dataSpliter in range(0, len(trainSegments), config_trainStep):

            print("Starting with the preparation of the training set for start index", dataSpliter)
            trainS = trainSegments[dataSpliter:dataSpliter + config_trainStep]

            #print("Start training on the subset of indices." +
            #  str(dataSpliter) + "-" + str(dataSpliter + config_trainStep) + ". total segments for this subset: " + str(
            #    len(trainS)))
        # for now number of batch is number of
            for batch in get_batch_from_segments_train_rnn(trainS, 30):
                #print("got batch", len(batch))
                if (len(batch) == 0):
                    break
                hypX = torch.cat([d[0] for d in batch])
                refX = torch.cat([d[1] for d in batch])
                Y = torch.cat([d[2] for d in batch])

                opt.zero_grad()
                output = net(hypX, refX)
                loss = criterion(output.view(output.shape[0]), Y)
                #print("loss: ", loss.item())

                lossValues.append(loss.item())
                loss.backward()
                opt.step()


        #f1score= get_scores_from_dev_rnn(net, devSegments)
        print("lr at epoch: ", epoch, " is ",get_lr(opt))
        #evalF1Scores.append(f1score)
        #netErrors.append(error)

        #net.train()
        save_model(net, epoch, lossValues, evalF1Scores, [])





