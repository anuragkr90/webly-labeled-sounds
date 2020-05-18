import sys
import time
import os
import argparse
import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as Fx
import torch.nn.init as init
import torch.utils.data
from torch.autograd import Variable
import torch.optim as optim
from torchvision import transforms, utils
from numpy import newaxis
import torch.nn.parallel
import torch.nn.functional
import cmetrics as metric
import random
from utils import *
from skimage.util import view_as_windows

class Logger(object):
    def __init__(self,run_number):
        self.run_number = run_number
        self.terminal = sys.stdout
        self.log = open("logfiles/logfile_" + str(run_number) + "_gpu_latest_mel.log", "a")
    def write(self, message):
        self.terminal.write(message)
        self.log.write(message)  
    def flush(self):
        pass    


def dataloader(features_path,file_list,batch_size,feat_dim,shuffle=True,last_batch=True):
    if shuffle==True:
        random.shuffle(file_list)

    if batch_size > len(file_list):
        X_train=[]
        Y_train=[]
        for item in file_list:
            filename=item[0]
            #print filename
            classname = item[1].strip()
            classname = classname.split(',')
            complete_filename = features_path + '/' + filename 
            features = np.loadtxt(complete_filename,delimiter=',')
            try:
                features.shape[1]
            except IndexError:
                features=np.array([features])
            while features.shape[0] < feat_dim:
                features=np.concatenate((features,features[0:feat_dim-features.shape[0],:]))
            features = features[newaxis,:,:]
            Y_train_this=np.zeros(args.classCount)
            classname_int=[int(i) for i in classname]
            Y_train_this[classname_int]=1.0
            X_train.append(features)
            Y_train.append(Y_train_this)
        yield np.array(X_train),np.array(Y_train)

    else:
        
        if last_batch==True:
            max_value=np.ceil(np.float(len(file_list))/batch_size) + 1
        num_batches=int(np.ceil(np.float(len(file_list))/batch_size )) 
        for i in range(num_batches):
            if i < max_value:
                mini_file_list=file_list[i*batch_size:(i+1)*batch_size]
            else:
                mini_file_list=file_list[i*batch_size:]

            X_train=[]
            Y_train=[]
            for item in mini_file_list:
                filename=item[0]
                #print filename
                classname=item[1].strip()
                classname=classname.split(',')
                complete_filename=features_path + '/' + filename
                features=np.loadtxt(complete_filename,delimiter=',')
                try:
                    features.shape[1]
                except IndexError:
                    features=np.array([features])
                while features.shape[0] < feat_dim:
                    features=np.concatenate((features,features[0:feat_dim-features.shape[0],:]))
                features = features[newaxis,:,:]
                Y_train_this=np.zeros(args.classCount)
                classname_int=[int(i) for i in classname]
                Y_train_this[classname_int] = 1.0
                X_train.append(features)
                Y_train.append(Y_train_this)

            yield np.array(X_train),np.array(Y_train)


def iterate_minibatches_torch(inputs,batchsize,targets=None,shuffle=False,last_batch=True):
    last_batch = not last_batch
    if targets is None:
        print 'No Target given to minibatch iterator. Using a dummy target with two outputs'
        targets = np.zeros((inputs.shape[0],2))

    inputs = torch.Tensor(inputs)
    targets = torch.Tensor(targets)

    dataset = torch.utils.data.TensorDataset(inputs,targets)
    dataiter = torch.utils.data.DataLoader(dataset,batch_size=batchsize,shuffle=shuffle,drop_last=last_batch)
    return dataiter

def read_file(filename):
    listname=[]
    with open(filename,'r') as f:
        lines = f.readlines()
        for line in lines:
            listname.append((line.split(' ')[0],line.split(' ')[1]))
        f.close()
    return listname



class NET(nn.Module):
    
    def __init__(self,nclass):
        super(NET,self).__init__()
        self.globalpool = Fx.avg_pool2d

        self.layer1 = nn.Sequential(nn.Conv2d(1,64,kernel_size=3,padding=1),nn.BatchNorm2d(64),nn.ReLU())
        self.layer2 = nn.Sequential(nn.Conv2d(64,64,kernel_size=3,padding=1),nn.BatchNorm2d(64),nn.ReLU())
        self.layer3 = nn.MaxPool2d((1,2)) 

        self.layer4 = nn.Sequential(nn.Conv2d(64,128,kernel_size=3,padding=1),nn.BatchNorm2d(128),nn.ReLU())
        self.layer5 = nn.Sequential(nn.Conv2d(128,128,kernel_size=3,padding=1),nn.BatchNorm2d(128),nn.ReLU())
        self.layer6 = nn.MaxPool2d((1,2)) 

        self.layer7 = nn.Sequential(nn.Conv2d(128,256,kernel_size=3,padding=1),nn.BatchNorm2d(256),nn.ReLU())
        self.layer8 = nn.Sequential(nn.Conv2d(256,256,kernel_size=3,padding=1),nn.BatchNorm2d(256),nn.ReLU())
        self.layer9 = nn.MaxPool2d((1,2))

        self.layer10 = nn.Sequential(nn.Conv2d(256,512,kernel_size=3,padding=1),nn.BatchNorm2d(512),nn.ReLU())
        self.layer11 = nn.Sequential(nn.Conv2d(512,512,kernel_size=3,padding=1),nn.BatchNorm2d(512),nn.ReLU())
        self.layer12 = nn.MaxPool2d((1,2)) 

        self.layer13 = nn.Sequential(nn.Conv2d(512,1024,kernel_size=(1,8)),nn.BatchNorm2d(1024),nn.ReLU())
        self.layer14 = nn.Sequential(nn.Conv2d(1024,1024,kernel_size=1),nn.BatchNorm2d(1024),nn.ReLU())
        self.layer15 = nn.Sequential(nn.Conv2d(1024,nclass,kernel_size=1),nn.Sigmoid())
        
        self.clamp = 1e-8
        
    # def __init__(self,nclass):
    #     super(NET,self).__init__()
    #     self.globalpool = Fx.avg_pool2d

    #     self.layer1 = nn.Sequential(nn.Conv2d(1,64,kernel_size=3,padding=1),nn.BatchNorm2d(64),nn.LeakyReLU())
    #     self.layer2 = nn.Sequential(nn.Conv2d(64,64,kernel_size=3,padding=1),nn.BatchNorm2d(64),nn.LeakyReLU())
    #     self.layer3 = nn.MaxPool2d((1,2)) 

    #     self.layer4 = nn.Sequential(nn.Conv2d(64,128,kernel_size=3,padding=1),nn.BatchNorm2d(128),nn.LeakyReLU())
    #     self.layer5 = nn.Sequential(nn.Conv2d(128,128,kernel_size=3,padding=1),nn.BatchNorm2d(128),nn.LeakyReLU())
    #     self.layer6 = nn.MaxPool2d((1,2)) 

    #     self.layer7 = nn.Sequential(nn.Conv2d(128,256,kernel_size=3,padding=1),nn.BatchNorm2d(256),nn.LeakyReLU())
    #     self.layer8 = nn.Sequential(nn.Conv2d(256,256,kernel_size=3,padding=1),nn.BatchNorm2d(256),nn.LeakyReLU())
    #     self.layer9 = nn.MaxPool2d((1,2))

    #     self.layer10 = nn.Sequential(nn.Conv2d(256,512,kernel_size=3,padding=1),nn.BatchNorm2d(512),nn.LeakyReLU())
    #     self.layer11 = nn.Sequential(nn.Conv2d(512,512,kernel_size=3,padding=1),nn.BatchNorm2d(512),nn.LeakyReLU())
    #     self.layer12 = nn.MaxPool2d((1,2)) 

    #     self.layer13 = nn.Sequential(nn.Conv2d(512,1024,kernel_size=(1,8)),nn.BatchNorm2d(1024),nn.LeakyReLU())
    #     self.layer14 = nn.Sequential(nn.Conv2d(1024,1024,kernel_size=1),nn.BatchNorm2d(1024),nn.LeakyReLU())
    #     self.layer15 = nn.Sequential(nn.Conv2d(1024,nclass,kernel_size=1),nn.Sigmoid())
        
    #     self.clamp = 1e-8



    def forward(self,x):
        out = self.layer1(x)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        out = self.layer5(out)
        out = self.layer6(out)
        out = self.layer7(out)
        out = self.layer8(out)
        out = self.layer9(out)
        out = self.layer10(out)
        out = self.layer11(out)
        out = self.layer12(out)
        out = self.layer13(out)
        out2 = self.layer14(out)
        out1 = self.layer15(out2)
        
        out = self.globalpool(out1,kernel_size=out1.size()[2:])
        out = out.view(out.size(0),-1)

        out = torch.clamp(out,min=self.clamp,max=1-self.clamp)
        return out,out2

    def xavier_init(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                init.xavier_uniform(m.weight, gain=nn.init.calculate_gain('relu'))
                m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

                
class newNET2(nn.Module):
    
    # def __init__(self,nclass,insize):
    #     super(newNET2,self).__init__()
        
    #     self.layer1 = nn.Sequential(nn.Linear(insize,2048),nn.ReLU())
    #     self.layer2 = nn.Dropout(p=0.4)
    #     self.layer3 = nn.Sequential(nn.Linear(2048,1024),nn.ReLU())
    #     self.layer4 = nn.Dropout(p=0.4)

    #     self.layer5 = nn.Sequential(nn.Linear(1024,1024),nn.ReLU())
    #     self.layer6 = nn.Sequential(nn.Linear(1024,nclass),nn.Sigmoid())
        
    #     self.clamp = 1e-8
    
    def __init__(self,nclass,insize):
        super(newNET2,self).__init__()
        
        self.layer1 = nn.Sequential(nn.Linear(insize,2048),nn.LeakyReLU())
        self.layer2 = nn.Dropout(p=0.4)
        self.layer3 = nn.Sequential(nn.Linear(2048,1024),nn.LeakyReLU())
        self.layer4 = nn.Dropout(p=0.4)

        self.layer5 = nn.Sequential(nn.Linear(1024,1024),nn.LeakyReLU())
        self.layer6 = nn.Sequential(nn.Linear(1024,nclass),nn.Sigmoid())
        
        self.clamp = 1e-8

    def forward(self,x):
        out = self.layer1(x)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        out = self.layer5(out)
        out = self.layer6(out)
        
        out = torch.clamp(out,min=self.clamp,max=1-self.clamp)
        return out
        

    def net_init(self,init_type='xavier'):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                if init_type == 'xavier':
                    init.xavier_uniform(m.weight, gain=nn.init.calculate_gain('relu'))
                elif init_type == 'kaiming':
                    init.kaiming_normal(m.weight)
                m.bias.data.zero_()


class combNET(nn.Module):
    
    def __init__(self,net1,net2):
        super(combNET,self).__init__()
        self.net1 = net1
        self.net2 = net2
        
    def forward(self,x,y):
        #print x.size(), y.size()
        out1,xxx = self.net1(x)
        out2 = self.net2(y)
        return out1, out2
        

use_cuda = torch.cuda.is_available()

def loss_div_outs(out1,out2,losstype):

    #total (not mean loss for each vec) and then mean of that
    
    if losstype == 'L1.B':
        return torch.mean(torch.sum(torch.abs(out1 - out2),1))

    elif losstype == 'L2.B':
        return torch.mean(torch.sum((out1 - out2)*(out1 - out2), dim=1))
        
    elif losstype == 'KL.B':

        fac1 = torch.log(torch.div(out1,out2 + 1e-10))
        fac1 = fac1 * out1
        fac_batch1 = torch.sum(fac1,1) - torch.sum(out1,1) + torch.sum(out2,1)

        fac2 = torch.log(torch.div(out2,out1 + 1e-10))
        fac2 = fac2 * out2
        fac_batch2 = torch.sum(fac2,1) - torch.sum(out2,1) + torch.sum(out1,1)

        fac_batch = fac_batch1 + fac_batch2

        return torch.mean(fac_batch)
             
    else:
        raise ValueError('Unknown loss type')


def setupClassifier(training_input_directory,validation_input_directory,testing_input_directory,classCount,spec_count,segment_length,learning_rate,output_path):
    
    sys.stdout = c1
    loss_fn = nn.BCELoss()
    
    if args.preload == 'AUD':
        oldnet = NET(527)
    elif args.preload == 'YT':
        oldnet = NET(40)
    
    print "Loading Feat Model..."
    if use_cuda:
        oldnet.cuda()
        oldnet = torch.nn.DataParallel(oldnet, device_ids=range(torch.cuda.device_count()))
        load_model_direct(oldnet,args.feat_model_path)
    else:
        load_model_cpu(oldnet,args.feat_model_path,False)

    #aps, aucs, aps_ranked, epo_val_loss = netVald(oldnet,validation_input_directory,loss_fn)
    if args.segmapping == 'avg':
        featMap = Fx.avg_pool2d
    elif args.segmapping == 'max':
        featMap = Fx.max_pool2d

    feat_train = netFeatures(oldnet,training_input_directory,args.train_list,'training',featMap)
    feat_vald = netFeatures(oldnet,validation_input_directory,args.val_test_list,'validation',featMap)
    feat_test = netFeatures(oldnet,testing_input_directory,args.val_test_list,'testing',featMap)

    
    if args.newnet == 'newNET':
        net1 = newNET(classCount,feat_train[0][1].shape[1])
    elif args.newnet == 'newNET2':
        net1 = newNET2(classCount,feat_train[0][1].shape[1])
        
    net2 = NET(classCount)
    net1.net_init(init_type = 'kaiming')
    net2.xavier_init()
    
    if args.preloading1 == 'Y':
        print "Loading Prev Trained Models for N1"
        load_model_cpu(net1,args.preloading_model1_path)

    
    if args.preloading2 == 'Y':
        print "Loading Prev Trained Models for N2"
        load_model_cpu(net2,args.preloading_model2_path)

    #net2.cuda()
    #net2 = torch.nn.DataParallel(net2, device_ids=range(torch.cuda.device_count()))
    #aps, aucs, aps_ranked, epo_val_loss = netVald(net2,validation_input_directory,loss_fn)
    #print aps, aucs, aps_ranked


    net = combNET(net2,net1)
    if use_cuda:
        net.cuda()
        net = torch.nn.DataParallel(net, device_ids=range(torch.cuda.device_count()))
    
    
    if args.optimizer == 'sgd':
        optimizer = optim.SGD(net.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay,momentum=0.9)#args.momentum)
    elif args.optimizer == 'adam':
        optimizer = optim.Adam(net.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)
    if args.schedular == 'Y':
        scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.87)

    epoch_best=0
    aps_best=0
    model_path=''

    for epoch in range(args.num_epochs):
        print "Current epoch is " + str(epoch)

        if args.schedular == 'Y':
            scheduler.step()

        #aps, aucs, aps_ranked, aps1,aucs1,aps_ranked1, aps2,aucs2,aps_ranked2, epo_val_loss = netFeatVald(net,feat_vald,loss_fn)
        #print aps[-1], aucs[-1], aps_ranked[-1], aps1[-1],aucs1[-1],aps_ranked1[-1], aps2[-1],aucs2[-1],aps_ranked2[-1]
        
        #train on feats
        netFeatTrain(net,feat_train,optimizer,loss_fn,epoch)

        #validation
        start_time_val = time.time()
        aps, aucs, aps_ranked, aps1,aucs1,aps_ranked1, aps2,aucs2,aps_ranked2, epo_val_loss = netFeatVald(net,feat_vald,loss_fn,epoch)

        filename = os.path.join('output', str(args.run_number),'metrics_validation_' + str(args.run_number) + '_' +  str(epoch) + '_aps.txt')
        filename1 = os.path.join('output', str(args.run_number),'metrics_validation_' + str(args.run_number) + '_' + str(epoch) + '_aps_ranked.txt')
        filename2 = os.path.join('output', str(args.run_number),'metrics_validation_' + str(args.run_number) + '_' + str(epoch) + '_aucs.txt')

        if not os.path.isdir(os.path.dirname(filename)):
            os.makedirs(os.path.dirname(filename))

        np.save(filename,np.array([aps, aps1, aps2]))
        np.save(filename1,np.array([aps_ranked, aps_ranked1, aps_ranked2]))
        np.save(filename2,np.array([aucs, aucs1, aucs2]))

        if aps_ranked[-1] > aps_best:
            aps_best = aps_ranked[-1]
            epoch_best = epoch
            if not os.path.exists('output/'+str(args.run_number)):
                os.mkdir('output/' + str(args.run_number))
            torch.save(net.state_dict(),'output/' + str(args.run_number)+'/model_path.' + str(args.run_number) + '_' +str(epoch)+  '.pkl')
            model_path='output/' + str(args.run_number) + '/model_path.' + str(args.run_number) + '_' + str(epoch) + '.pkl'

        print "Val Maps_ranked=" + str(aps_ranked[-1]), ",MAP=" + str(aps[-1]), ",MAUC=" + str(aucs[-1]),"Maps_ranked1=" + str(aps_ranked1[-1]), ",MAP1=" + str(aps1[-1]), ",MAUC1=" + str(aucs1[-1]),"Maps_ranked2=" + str(aps_ranked2[-1]), ",MAP2=" + str(aps2[-1]), ",MAUC2=" + str(aucs2[-1]),
        
        print "best MAP=" + str(aps_best), "best ep=" + str(epoch_best),
        print ", Vald Loss {}. Done in {} seconds.".format(epo_val_loss, time.time() - start_time_val)

    # testing
    start_time_test = time.time()
    if use_cuda:
        load_model_direct(net,model_path)
    else:
        load_model_cpu(net,model_path)

    aps, aucs, aps_ranked, aps1,aucs1,aps_ranked1, aps2,aucs2,aps_ranked2, epo_test_loss = netFeatVald(net,feat_test,loss_fn,epoch_best)

    filename = os.path.join('output/' + str(args.run_number),'metrics_testing_' + str(args.run_number) + '_' +  str(epoch_best) + '_aps.txt')
    filename1 = os.path.join('output/' + str(args.run_number),'metrics_testing_' + str(args.run_number) + '_' + str(epoch_best) + '_aps_ranked.txt')
    filename2 = os.path.join('output/' + str(args.run_number),'metrics_testing_' + str(args.run_number) + '_' + str(epoch_best) + '_aucs.txt')

    np.save(filename,np.array([aps, aps1, aps2]))
    np.save(filename1,np.array([aps_ranked, aps_ranked1, aps_ranked2]))
    np.save(filename2,np.array([aucs, aucs1, aucs2]))

    print "Testing Maps_ranked=" + str(aps_ranked[-1]), ",MAP=" + str(aps[-1]), ",MAUC=" + str(aucs[-1]),"Maps_ranked1=" + str(aps_ranked1[-1]), ",MAP1=" + str(aps1[-1]), ",MAUC1=" + str(aucs1[-1]),"Maps_ranked2=" + str(aps_ranked2[-1]), ",MAP2=" + str(aps2[-1]), ",MAUC2=" + str(aucs2[-1]),

    print "Testing Loss {}. Done in {} seconds".format(epo_test_loss,time.time() - start_time_test)


def netFeatTrain(net,train_data,optimizer,loss_fn,epoch):
    
    #############
    ### Training 
    #############

    net.train()
    epo_train_loss = 0
    batch_count = 0
    start_time = time.time()
    
    random.shuffle(train_data)
    
    # if stepping in div factor
    if args.div_apply[0] == 'Y':
        
        div_type = args.div_apply.split('+')[1]
        div_factor = float(args.div_apply.split('+')[2])
        if len(args.div_apply.split('+')) > 3:
            step_factors = args.div_apply.split('+')[3]
            step = int(step_factors.split('|')[0])
            factor = float(step_factors.split('|')[1])
            div_factor = div_factor * (factor ** (epoch // step))

    for batch in train_data:
        
        inputs1,inputs2,targets = batch	

        indata1 = torch.Tensor(inputs1)
        indata2 = torch.Tensor(inputs2)
        lbdata = torch.Tensor(targets)
        
        if use_cuda:
            indata1 = Variable(indata1).cuda()
            indata2 = Variable(indata2).cuda()
            lbdata = Variable(lbdata).cuda()
        else:
            indata1 = Variable(indata1)
            indata2 = Variable(indata2)
            lbdata = Variable(lbdata)

        optimizer.zero_grad()
        batch_pred1, batch_pred2 = net(indata1,indata2)
        #print batch_pred1.size(),batch_pred2.size()

        batch_train_loss1 = loss_fn(batch_pred1,lbdata)
        batch_train_loss2 = loss_fn(batch_pred2,lbdata)
        batch_train_loss = batch_train_loss1 + batch_train_loss2
        #print batch_train_loss.data[0], div_factor,
        if args.div_apply[0] == 'Y':

            batch_train_loss3 = div_factor*loss_div_outs(batch_pred1,batch_pred2,div_type)
            batch_train_loss = batch_train_loss + batch_train_loss3
            #print batch_train_loss3.data[0], batch_train_loss.data[0]

        batch_train_loss.backward()
        optimizer.step()
        
        epo_train_loss += batch_train_loss.data[0]
        batch_count += 1

    epo_train_loss = epo_train_loss/batch_count

    print "{} Training done in {} seconds. Training Loss {}".format(epoch, time.time() - start_time, epo_train_loss)



def netFeatVald(net,vald_data,loss_fn,epoch):

    ###############
    ### Validation 
    ###############
                
    epo_val_loss = 0
    val_batch_count = 0
    start_time_val = time.time()
    
    all_predictions1 = []
    all_predictions2 = []
    all_labels = []

    net.eval()
    
    if args.div_apply[0] == 'Y':
        
        div_type = args.div_apply.split('+')[1]
        div_factor = float(args.div_apply.split('+')[2])
        if len(args.div_apply.split('+')) > 3:
            step_factors = args.div_apply.split('+')[3]
            step = int(step_factors.split('|')[0])
            factor = float(step_factors.split('|')[1])
            div_factor = div_factor * (factor ** (epoch // step))

    for batch in vald_data:
        inputs1,inputs2, targets = batch
            
        indata1 = torch.Tensor(inputs1)
        indata2 = torch.Tensor(inputs2)
        lbdata = torch.Tensor(targets)
        
        if use_cuda:
            indata1 = Variable(indata1,volatile=True).cuda()
            indata2 = Variable(indata2,volatile=True).cuda()
            lbdata = Variable(lbdata,volatile=True).cuda()
        else:
            indata1 = Variable(indata1,volatile=True)
            indata2 = Variable(indata2,volatile=True)
            lbdata = Variable(lbdata,volatile=True)

        batch_pred1,batch_pred2 = net(indata1,indata2)

        batch_val_loss1 = loss_fn(batch_pred1,lbdata)
        batch_val_loss2 = loss_fn(batch_pred2,lbdata)
        batch_val_loss = batch_val_loss1 + batch_val_loss2

        if args.div_apply[0] == 'Y':

            batch_val_loss3 = loss_div_outs(batch_pred1,batch_pred2,div_type)
            batch_val_loss = batch_val_loss + div_factor*batch_val_loss3
            
        epo_val_loss += batch_val_loss.data[0]
        val_batch_count += 1
        
        inres1 = batch_pred1.data.cpu().numpy().tolist()
        inres2 = batch_pred2.data.cpu().numpy().tolist()
        
        all_predictions1.extend(inres1)
        all_predictions2.extend(inres2)
        
        all_labels.extend(lbdata.data.cpu().numpy().tolist())
            
            
        #print "{} Set-Batch Validation done in {} seconds. Validation Loss {} ".format(i, time.time() - start_time_val, epo_val_loss)

    epo_val_loss = epo_val_loss/val_batch_count
    all_predictions1 = np.array(all_predictions1)
    all_predictions2 = np.array(all_predictions2)
    all_predictions = (all_predictions1 + all_predictions2)/2.0
    all_labels = np.array(all_labels)
    
    aps1 = metric.compute_AP_all_class(all_labels,all_predictions1)
    aucs1 = metric.compute_AUC_all_class(all_labels,all_predictions1)
    aps_ranked1 = metric.compute_AP_my_all_class(all_labels,all_predictions1)

    aps2 = metric.compute_AP_all_class(all_labels,all_predictions2)
    aucs2 = metric.compute_AUC_all_class(all_labels,all_predictions2)
    aps_ranked2 = metric.compute_AP_my_all_class(all_labels,all_predictions2)
    
    aps = metric.compute_AP_all_class(all_labels,all_predictions)
    aucs = metric.compute_AUC_all_class(all_labels,all_predictions)
    aps_ranked = metric.compute_AP_my_all_class(all_labels,all_predictions)
    
    return aps, aucs, aps_ranked, aps1,aucs1,aps_ranked1, aps2,aucs2,aps_ranked2, epo_val_loss


def netFeatures(net,input_directory,input_setlist,input_fold,segMap):

    ###############
    ### get features instances 
    ###############
    
    all_batches = []
    set_file_list = []
    set_file_list = []
    net.eval()
    print input_setlist
    for i in range(len(input_setlist)):
        set_file_list.append(args.train_test_split_path + '/b_' + str(input_setlist[i]) + '_'+ input_fold +'.list')
        
        input_files = read_file(set_file_list[i])
        feat_dim = int(input_setlist[i])
        if len(input_files) == 0:
            continue
        for batch in dataloader(input_directory,input_files,args.sgd_batch_size,feat_dim,shuffle=False,last_batch=True):
            #bsize fixed to 1
            
            inputs, targets = batch
            
            if use_cuda:
                indata = torch.Tensor(inputs)
                indata = Variable(indata,volatile=True).cuda()
            else:
                indata = torch.Tensor(inputs)
                indata = Variable(indata,volatile=True)
                
            
            batch_pred, batch_feat = net(indata)
            batch_feat = segMap(batch_feat,kernel_size=batch_feat.size()[2:])
            batch_feat = batch_feat.view(batch_feat.size(0),-1)
            batch_feat = batch_feat.data.cpu().numpy()
            
            batch_tp = (inputs,batch_feat,targets)
            #print batch_feat.shape,inputs.shape,targets.shape
            all_batches.append(batch_tp)
            
    print "Total ", len(all_batches), " batches made "
    return all_batches




def netVald(net,validation_input_directory,loss_fn):

    ###############
    ### Validation 
    ###############
                
    epo_val_loss = 0
    val_batch_count = 0
    start_time_val = time.time()
    all_predictions = []
    all_labels = []
    validation_set_file_list = []
    net.eval()

    for i in range(len(args.val_test_list)):
        validation_set_file_list.append(args.train_test_split_path + '/b_' + str(args.val_test_list[i]) + '_validation.list')
        validation_files = read_file(validation_set_file_list[i])
        feat_dim=int(args.val_test_list[i])
        if len(validation_files) == 0:
            continue
        for batch in dataloader(validation_input_directory,validation_files,args.sgd_batch_size,feat_dim,shuffle=False,last_batch=True):
            inputs,targets = batch
            if targets is None:
                print 'No Target given to minibatch iterator. Using a dummy target with two outputs'
                targets = np.zeros((inputs.shape[0],2))

            indata = torch.Tensor(inputs)
            lbdata = torch.Tensor(targets)
            
            if use_cuda:
                indata = Variable(indata,volatile=True).cuda()
                lbdata = Variable(lbdata,volatile=True).cuda()
            else:
                indata = Variable(indata,volatile=True)
                lbdata = Variable(lbdata,volatile=True)

            batch_pred,xx = net(indata)

            batch_val_loss = loss_fn(batch_pred,lbdata)
            epo_val_loss += batch_val_loss.data[0]
            val_batch_count += 1

            inres = batch_pred.data.cpu().numpy().tolist()
            all_predictions.extend(inres)
            all_labels.extend(lbdata.data.cpu().numpy().tolist())
            
        print "{} Set-Batch Validation done in {} seconds. Validation Loss {} ".format(i, time.time() - start_time_val, epo_val_loss)

    epo_val_loss = epo_val_loss/val_batch_count
    all_predictions = np.array(all_predictions)
    all_labels = np.array(all_labels)
    aps = metric.compute_AP_all_class(all_labels,all_predictions)
    aucs = metric.compute_AUC_all_class(all_labels,all_predictions)
    aps_ranked = metric.compute_AP_my_all_class(all_labels,all_predictions)

    print "Val aps ranked " + str(aps_ranked[-1]), "Val APS " + str(aps[-1]), "Val AUC " + str(aucs[-1])

    return aps, aucs, aps_ranked, epo_val_loss




if __name__ == '__main__':
    print len(sys.argv )
    if len(sys.argv) < 14:
        print "Running Instructions:\n\npython cnn.py <input_dir> <classCount> <spec_count> <segment_length> <learning_rate> <momentum> <evaluate> <test> <output_dir><sel_K_segments>"
    else:
        parser = argparse.ArgumentParser()

        parser.add_argument('--training_features_directory', type=str, default='training_directory', metavar='N',
                            help='training_features_directory (default: training_directory')
        parser.add_argument('--validation_features_directory', type=str, default='validation_directory', metavar='N',
                            help='validation_features_directory (default: validation_directory')
        parser.add_argument('--testing_features_directory', type=str, default='testing_directory', metavar='N',
                            help='testing_features_directory (default: testing_directory')
        parser.add_argument('--classCount', type=int, default='18', metavar='N',
                            help='classCount - default:18')
        parser.add_argument('--spec_count', type=int, default='96', metavar='N',
                            help='spec_count - default:96')
        parser.add_argument('--segment_length', type=int, default='101', metavar='N',
                            help='segment_length (default: 101)')
        parser.add_argument('--output_path', type=str, default='outputpath', metavar='N',
                            help='output_path - default:outputpath')
        parser.add_argument('--learning_rate', type=float, default='0.001', metavar='N',
                            help='learning_rate - default:0.001')
        parser.add_argument('--num_epochs', type=int, default='1', metavar='N',
                            help='num_epochs - default:1')
        parser.add_argument('--sgd_batch_size', type=int, default='128', metavar='N',
                            help='sgd_batch_size- default:40')
        parser.add_argument('--momentum', default=0.9, type=float, metavar='M', help='momentum')
        parser.add_argument('--evaluate', dest='evaluate', action='store_true', help='evaluate model on validation set')
        parser.add_argument('--test', dest='test', action='store_true', help='evaluate model on test set')
        parser.add_argument('--val_test_list',nargs='+', help='<Required>set flag',required=True)
        parser.add_argument('--train_list',nargs='+', help='<Required>set flag',required=True)
        parser.add_argument('--train_test_split_path',type=str,default='train_test_split',metavar='N',
                           help='train_test_split_path - default train_test_split_path')
        parser.add_argument('--run_number', type=str, default='0', metavar='N',
                            help='run_number- default:0')
        parser.add_argument('--preloading1', type=str, default='N', metavar='N',
                            help='preloading- default:N')
        parser.add_argument('--preloading2', type=str, default='N', metavar='N',
                            help='preloading2- default:N')
        parser.add_argument('--feat_model_path', type=str, default='modelpath', metavar='N', help='preloading_model_path - default:modelpath')
        parser.add_argument('--segmapping', type=str, default='max', metavar='N',
                            help='segmapping - default:max')
        parser.add_argument('--weight_decay', type=float, default=0.0, metavar='N',
                            help='weight_decay - default:0.0')
        parser.add_argument('--optimizer', type=str, default='adam', metavar='N',
                            help='optimizer - default:adam')
        
        parser.add_argument('--newnet', type=str, default='newNET', metavar='N',
                            help='newnet - default:newNET')
        parser.add_argument('--preload', type=str, default='AUD', metavar='N',
                            help='preload - default:AUD')
        parser.add_argument('--schedular', type=str, default='N', metavar='N',
                            help='schedular - default:N')
        parser.add_argument('--div_apply', type=str, default='N', metavar='N',
                            help='div_apply - default:N')

        parser.add_argument('--preloading_model1_path', type=str, default='modelpath1', metavar='N', help='preloading_model1_path - default:modelpath')
        parser.add_argument('--preloading_model2_path', type=str, default='modelpath2', metavar='N', help='preloading_model2_path - default:modelpath')
        
        args = parser.parse_args()
        c1 = Logger(args.run_number)

        if not os.path.exists(args.output_path):
            os.makedirs(args.output_path)

        setupClassifier(args.training_features_directory, args.validation_features_directory, args.testing_features_directory, args.classCount, args.spec_count, args.segment_length,args.learning_rate,args.output_path)

