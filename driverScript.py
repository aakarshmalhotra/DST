# -*- coding: utf-8 -*-
"""
Created on Tue Jul 30 23:36:32 2021

@author: aakarsh
"""

import torch
from torch import nn
from torchsummary import summary
import torch.optim as optim
from torch.utils.data import DataLoader
import scipy.io as io
import sys
import numpy as np
from AFLWdataLoader import loadCompleteDataset
from vgg16model import myVGG16, myMSEloss, myMSElossPose, myMSElossReconstruction
from testFunctions import testing, isBestModel
import math


SaveWeightEpochZero = 0
preTrained = True
in_channels = 3
out_channels = 3
startFrom=0

NumEpochs = 100
batch_size = 16
threadsTraining=16
TrainingSampleCount=17000#Enter total training data point here
dataset_path='/media/AFLW/'

## Things to do with weight
initialScaling=1; #to normalize each loss in same range
## Mark '1' on whih aspects to consider
NetworkDepthGate=1
TrainingSampleCountGate=1
StartingLossGate=1
AntiMomentumGate=1
RegularizerGate=1
total=NetworkDepthGate+TrainingSampleCountGate+StartingLossGate+AntiMomentumGate+RegularizerGate
## Sampling Distribution 
elementsInArray=1000
#ranger tells how much weight each aspect needs to be giiiiiven
# currently, all get equal weight
ranger=int(math.floor((elementsInArray/total)))


## Initialize network depths
NetworkDepth_taskWise=np.array([17.0 , 17.0 , 17.0 , 17.0, 26.0])
## Initialize training samples count. This get populated from dataloader
TrainingSampleCount_taskWise=np.array([0.0 , 0.0 , 0.0 , 0.0 , 0.0])
## starting loss and current loss array initialization
SLtaskWise=np.zeros([5])
CLtaskWise=np.zeros([5])
## Anti Momentum (For task stagnation cumputation)
previousLoss_taskWise=np.zeros([5])
CurrentLoss_taskWise=np.zeros([5])
Rate_j=np.zeros([5])
alpha=0.1


## Load training data
dataset_ALFW, TrainingSampleCount_taskWise, dataset_ALFW_Test, elementsInTest = loadCompleteDataset(startFrom, dataset_path, TrainingSampleCount)
train_dataloader = DataLoader(dataset_ALFW, shuffle=True, num_workers=threadsTraining, batch_size=batch_size)
print("Total number of batches in training: %d" % len(train_dataloader))
test_dataloader = DataLoader(dataset_ALFW_Test, shuffle=False, num_workers=threadsTraining, batch_size=batch_size)
print("Total number of batches in testing: %d" % len(test_dataloader))
    
## Initialize names here
CaseName='DST'
    
## History saver
IterationTaskWiseLosses=np.zeros([len(train_dataloader)*NumEpochs,6])
EpochTaskWiseLosses=np.zeros([NumEpochs,6])
EpochBitsTaskWise=np.zeros([NumEpochs,6,5])
EpochTestPerformance=np.zeros([NumEpochs,5])
EpochTaskONoff=np.zeros([NumEpochs,5])
    
if total!=0:
    ## Regularization
    regularizationBits_ON=np.array([ranger, ranger, ranger, ranger, ranger])
    print(regularizationBits_ON)
    ## Network Depth
    depthBits_ON=NetworkDepth_taskWise/max(NetworkDepth_taskWise)
    depthBits_ON=np.floor((elementsInArray/total)*depthBits_ON)
    depthBits_ON=depthBits_ON.astype(int)
    print(depthBits_ON)
    ## Training samples count
    TrainingSamplesBits_ON=TrainingSampleCount_taskWise/max(TrainingSampleCount_taskWise)
    TrainingSamplesBits_ON=np.floor((elementsInArray/total)*TrainingSamplesBits_ON)
    TrainingSamplesBits_ON=TrainingSamplesBits_ON.astype(int)
    print(TrainingSamplesBits_ON)
## Initialize Segnet Model
vgg16Model = myVGG16(preTrained, in_channels, out_channels)
vgg16Model = vgg16Model.initializeWeights(vgg16Model)
vgg16Model.cuda()
#summary(vgg16Model, input_size=(3, 224, 224))
criterion1 = nn.BCELoss()
criterion2 = myMSEloss()
criterion3 = myMSElossPose(batch_size)
criterion4 = myMSElossReconstruction()
#criterion4 = nn.BCEWithLogitsLoss()
optimizer = optim.Adam(vgg16Model.parameters(), lr=0.00005)
## Save state at epoch zero
if SaveWeightEpochZero:
    print('Saving model at epoch 0')
    torch.save(vgg16Model.state_dict(), './vgg16Model_Epoch_0_v1.pth')
else:
    print('Loading model at epoch 0')#this loads a stable model
    vgg16Model.load_state_dict(torch.load('./SelectedModel_Epoch_0.pth'))  
## For selecting best testing model
LeastError=100
BestEpoch=-1
## Training
avgLoss1=0;
avgLoss2=0;
avgLoss3=0;
avgLoss4=0;
avgLoss5=0;
avgLoss=0;
## Scaling weights based on startup
w1=1
w2=1
w3=1
w4=1
w5=1
## ON-OFF Gates for each loss.
g1=1
g2=1
g3=1
g4=1
g5=1
for epoch in range(NumEpochs):
    print("Epoch: %d" %(epoch+1))
    print('Tasks Active status---> Gender: ' + str(g1) + ', Glass: ' + str(g2) + ', Coords: ' + str(g3) + ', Pose: ' + str(g4) + ', Reconstruction: ' + str(g5))
    EpochTaskONoff[epoch,:]=[g1 , g2 , g3 , g4 , g5]
    rloss1 = 0.0
    rloss2 = 0.0
    rloss3 = 0.0
    rloss4 = 0.0
    rloss5 = 0.0
    rnet_loss = 0.0
    iteration=0
    iterationGender=0
    iterationGlasses=0
    iterationCoords=0
    iterationPose=0
    iterationReconstruct=0
    ## switch model to training mode
    vgg16Model.train()
    for i, data in enumerate(train_dataloader, 0):
        ## Load batch and convert to cuda/numpy
        img, img_flip, label_gender, label_glass, label_coords, label_presence, label_pose, imageName, TaskwiseLabelPresence = data
        TaskwiseLabelPresence = TaskwiseLabelPresence.numpy()
        img, img_flip, label_gender, label_glass, label_coords, label_presence, label_pose = img.float().cuda(), img_flip.float().cuda(), label_gender.float().cuda(), label_glass.float().cuda(), label_coords.float().cuda(), label_presence.float().cuda(), label_pose.float().cuda()
        #pairlabelTensor, combinedImages_new, combinedAnnotations = pairlabelTensor.float().cuda(), combinedImages.float().cuda(), combinedAnnotations.float().cuda()
        
        ## Find instance whole labels are present as some batches may have no labelled images
        gender_IndSelect = np.where(TaskwiseLabelPresence[:,0] == 1)
        glasses_IndSelect = np.where(TaskwiseLabelPresence[:,1] == 1)
        coords_IndSelect = np.where(TaskwiseLabelPresence[:,2] == 1)
        pose_IndSelect = np.where(TaskwiseLabelPresence[:,3] == 1)
        reconstruct_IndSelect = np.where(TaskwiseLabelPresence[:,4] == 1)
        
        ## Zero the gradients
        optimizer.zero_grad()
        ## forward + backward + optimize
        FC_output_gender, FC_output_glasses, FC_output_coords, FC_output_angle, decoderReconstruct = vgg16Model(img)
        if (gender_IndSelect[0].shape[0]!=0):
            loss1 = criterion1(FC_output_gender[gender_IndSelect[0],:], label_gender[gender_IndSelect[0],:])
            rloss1 += w1*loss1.item()
            IterationTaskWiseLosses[len(train_dataloader)*epoch + i , 0]=w1*loss1.item()
            iterationGender = iterationGender+1
        else:
            loss1 = 0.0
        if (glasses_IndSelect[0].shape[0]!=0):
            loss2 = criterion1(FC_output_glasses[glasses_IndSelect,:], label_glass[glasses_IndSelect,:])
            rloss2 += w2*loss2.item()                
            IterationTaskWiseLosses[len(train_dataloader)*epoch + i , 1]=w2*loss2.item()
            iterationGlasses = iterationGlasses+1
        else:
            loss2 = 0.0
        if (coords_IndSelect[0].shape[0]!=0):
            loss3 = criterion2(FC_output_coords[coords_IndSelect[0],:], label_coords[coords_IndSelect[0],:], label_presence[coords_IndSelect[0],:])
            rloss3 += w3*loss3.item()
            IterationTaskWiseLosses[len(train_dataloader)*epoch + i , 2]=w3*loss3.item()
            iterationCoords = iterationCoords+1
        else:
            loss3 = 0.0
        if (pose_IndSelect[0].shape[0]!=0):
            loss4 = criterion3(FC_output_angle[pose_IndSelect[0],:], label_pose[pose_IndSelect[0],:])
            rloss4 += w4*loss4.item()
            IterationTaskWiseLosses[len(train_dataloader)*epoch + i , 3]=w4*loss4.item()
            iterationPose = iterationPose+1
        else:
            loss4 = 0.0
        if (reconstruct_IndSelect[0].shape[0]!=0):
            loss5 = criterion4(decoderReconstruct[reconstruct_IndSelect[0],:,:,:], img_flip[reconstruct_IndSelect[0],:,:,:])
            rloss5 += w5*loss5.item()
            IterationTaskWiseLosses[len(train_dataloader)*epoch + i , 4]=w5*loss5.item()
            iterationReconstruct = iterationReconstruct+1
        else:
            loss5 = 0.0
        #loss1 = criterion1(FC_output_gender, label_gender)
        #loss2 = criterion1(FC_output_glasses, label_glass)
        #loss3 = criterion2(FC_output_coords, label_coords, label_presence)
        #loss4 = criterion3(FC_output_angle, label_pose)
        #loss5 = criterion1(decoderReconstruct, img_flip)
        #rloss1 += w1*loss1.item()
        #rloss2 += w2*loss2.item()
        #rloss3 += w3*loss3.item()
        #rloss4 += w4*loss4.item()
        #rloss5 += w5*loss5.item()
        NetLoss = g1*w1*loss1 + g2*w2*loss2 + g3*w3*loss3 + g4*w4*loss4 + g5*w5*loss5
        if epoch!=0: #no update in first epoch.
            NetLoss.backward()
            optimizer.step()
        rnet_loss += NetLoss.item()
        IterationTaskWiseLosses[len(train_dataloader)*epoch + i , 5]=NetLoss.item()
        iteration = iteration+1
    ## Operation after each eopch completion begins here    
    ## Initial scaling of weights
    if epoch==0:
        avgLoss1=rloss1/iterationGender;
        avgLoss2=rloss2/iterationGlasses;
        avgLoss3=rloss3/iterationCoords;
        avgLoss4=rloss4/iterationPose;
        avgLoss5=rloss5/iterationReconstruct;
        print('Initial actual loss values: ')
        SLtaskWise[0:5]=[avgLoss1 , avgLoss2 , avgLoss3 , avgLoss4 , avgLoss5]
        print(SLtaskWise)
        #youuuu may chose each loss to be in same scale
        if initialScaling==1:
            avgLoss=(avgLoss1+avgLoss2+avgLoss3+avgLoss4+avgLoss5)/5
            w1=avgLoss/avgLoss1
            w2=avgLoss/avgLoss2
            w3=avgLoss/avgLoss3
            w4=avgLoss/avgLoss4
            w5=avgLoss/avgLoss5
            SLtaskWise[0:5]=[w1*avgLoss1 , w2*avgLoss2 , w3*avgLoss3 , w4*avgLoss4 , w5*avgLoss5]
            print('Initially normalized loss values: ')
            print(SLtaskWise)
        previousLoss_taskWise=SLtaskWise.copy()
        ## put some initial values in history
        EpochTaskWiseLosses[epoch,:]=[SLtaskWise[0] , SLtaskWise[1] , SLtaskWise[2] , SLtaskWise[3] , SLtaskWise[4] , np.sum(SLtaskWise)]
        #All task remain active initially after 1st epoch
        EpochBitsTaskWise[epoch,0,:]=regularizationBits_ON
        EpochBitsTaskWise[epoch,1,:]=regularizationBits_ON
        EpochBitsTaskWise[epoch,2,:]=regularizationBits_ON
        EpochBitsTaskWise[epoch,3,:]=regularizationBits_ON
        EpochBitsTaskWise[epoch,4,:]=regularizationBits_ON
        EpochBitsTaskWise[epoch,5,:]=5*regularizationBits_ON
    else:
        # StartingLossGate: metric measuring task incompleteness
        if StartingLossGate==1: #epoch>0
            CLtaskWise[0:5]=[rloss1/iterationGender , rloss2/iterationGlasses , rloss3/iterationCoords , rloss4/iterationPose , rloss5/iterationReconstruct]
            Aj=CLtaskWise/SLtaskWise
            startingLossBits_ON_float=Aj/np.mean(Aj)
            np.clip(startingLossBits_ON_float, 0, 1, out=startingLossBits_ON_float)
            AntiMomentumAmountLearntProbability = startingLossBits_ON_float.copy() # 1 for slower task, 0.5 for faster
            startingLossBits_ON_float=np.floor((elementsInArray/total)*startingLossBits_ON_float)
            startingLossBits_ON_int=startingLossBits_ON_float.astype(int)
            print(startingLossBits_ON_int)
        # AntiMomentumGate: metric measuring task stagnation
        if AntiMomentumGate==1:
            CurrentLoss_taskWise[0:5]=[rloss1/iterationGender , rloss2/iterationGlasses , rloss3/iterationCoords , rloss4/iterationPose , rloss5/iterationReconstruct]
            if epoch==1:
                Rate_j=abs(previousLoss_taskWise-CurrentLoss_taskWise)
                Rate_j=Rate_j/previousLoss_taskWise
                Rate_j=Rate_j/AntiMomentumAmountLearntProbability
            else:
                difference=previousLoss_taskWise-CurrentLoss_taskWise
                difference=difference/previousLoss_taskWise
                difference=difference/AntiMomentumAmountLearntProbability
                difference[difference<=0] = Rate_j[difference<=0]
                Rate_j=alpha*difference + (1-alpha)*Rate_j
            AntiMomentumBits_ON_float=np.mean(Rate_j)/Rate_j
            np.clip(AntiMomentumBits_ON_float, 0, 1, out=AntiMomentumBits_ON_float)
            AntiMomentumBits_ON_float=np.floor((elementsInArray/total)*AntiMomentumBits_ON_float)
            AntiMomentumBits_ON_int=AntiMomentumBits_ON_float.astype(int)
            previousLoss_taskWise[0:5] = CurrentLoss_taskWise.copy()
            print(AntiMomentumBits_ON_int)
        ## Find total active bits (on a scale of 0-1000).
        # Always active task will have value 1000. gateDistribution_task1 array will have 1's in proportion to activation probability
        TaskWise_ON_bits=regularizationBits_ON + depthBits_ON + TrainingSamplesBits_ON + startingLossBits_ON_int + AntiMomentumBits_ON_int
        print(TaskWise_ON_bits)
        ## Switch them "ON" in distribution
        gateDistribution_task1=np.zeros([elementsInArray])
        gateDistribution_task2=np.zeros([elementsInArray])
        gateDistribution_task3=np.zeros([elementsInArray])
        gateDistribution_task4=np.zeros([elementsInArray])
        gateDistribution_task5=np.zeros([elementsInArray])
        gateDistribution_task1[0:TaskWise_ON_bits[0]]=1
        gateDistribution_task2[0:TaskWise_ON_bits[1]]=1
        gateDistribution_task3[0:TaskWise_ON_bits[2]]=1
        gateDistribution_task4[0:TaskWise_ON_bits[3]]=1
        gateDistribution_task5[0:TaskWise_ON_bits[4]]=1
        ## Select the index randomly
        TaskWiseActivationIndex=np.random.randint(elementsInArray, size=5)
        g1=gateDistribution_task1[TaskWiseActivationIndex[0]]
        g2=gateDistribution_task2[TaskWiseActivationIndex[1]]
        g3=gateDistribution_task3[TaskWiseActivationIndex[2]]
        g4=gateDistribution_task4[TaskWiseActivationIndex[3]]
        g5=gateDistribution_task5[TaskWiseActivationIndex[4]]
        ## Save active bits for history
        EpochBitsTaskWise[epoch,0,:]=regularizationBits_ON
        EpochBitsTaskWise[epoch,1,:]=depthBits_ON
        EpochBitsTaskWise[epoch,2,:]=TrainingSamplesBits_ON
        EpochBitsTaskWise[epoch,3,:]=startingLossBits_ON_int
        EpochBitsTaskWise[epoch,4,:]=AntiMomentumBits_ON_int
        EpochBitsTaskWise[epoch,5,:]=TaskWise_ON_bits
        ## Final loss for this epoch
        print('Gender loss: ' + str(rloss1/iterationGender) + ', Glass Loss: ' + str(rloss2/iterationGlasses) + ', Coords Loss: ' + str(rloss3/iterationCoords) + ', Pose Loss: ' + str(rloss4/iterationPose) + ', Reconstruction Loss: ' + str(rloss5/iterationReconstruct) + ',   Net Loss: ' + str(rnet_loss/iteration))
        EpochTaskWiseLosses[epoch,:]=[(rloss1/iterationGender) , (rloss2/iterationGlasses) , (rloss3/iterationCoords) , (rloss4/iterationPose) , (rloss5/iterationReconstruct) , (rnet_loss/iteration)]
        
    ## Perform Testing
    if (epoch+1)%1==0:
        EpochTestPerformance[epoch,:] = testing(epoch, vgg16Model, test_dataloader, elementsInTest, CaseName)
        LeastError, isBest = isBestModel(LeastError, EpochTestPerformance[epoch,:])
        if isBest:
            BestEpoch=epoch+1
        print('The award for the Best Model goes to: '+ str(BestEpoch))
io.savemat('History_'+CaseName+'.mat', dict(IterationTaskWiseLosses = IterationTaskWiseLosses, EpochTaskWiseLosses = EpochTaskWiseLosses, EpochBitsTaskWise = EpochBitsTaskWise, EpochTestPerformance = EpochTestPerformance, EpochTaskONoff = EpochTaskONoff, BestEpoch=BestEpoch))
