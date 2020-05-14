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



class newNET(nn.Module):
    
    def __init__(self,nclass,insize):
        super(newNET,self).__init__()
        
        self.layer1 = nn.Sequential(nn.Linear(insize,2048),nn.ReLU())
        self.layer2 = nn.Sequential(nn.Linear(2048,1024),nn.ReLU())
        self.layer3 = nn.Dropout(p=0.3)

        self.layer4 = nn.Sequential(nn.Linear(1024,1024),nn.ReLU())
        self.layer5 = nn.Sequential(nn.Linear(1024,nclass),nn.Sigmoid())
        
    def forward(self,x):
        out = self.layer1(x)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        out = self.layer5(out)
        
        return out
        

    def net_init(self,init_type='xavier'):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                if init_type == 'xavier':
                    init.xavier_uniform(m.weight, gain=nn.init.calculate_gain('relu'))
                elif init_type == 'kaiming':
                    init.kaiming_normal(m.weight)
                m.bias.data.zero_()


class newNET2(nn.Module):
    
    def __init__(self,nclass,insize):
        super(newNET2,self).__init__()
        
        self.layer1 = nn.Sequential(nn.Linear(insize,2048),nn.ReLU())
        self.layer2 = nn.Dropout(p=0.4)
        self.layer3 = nn.Sequential(nn.Linear(2048,1024),nn.ReLU())
        self.layer4 = nn.Dropout(p=0.4)

        self.layer5 = nn.Sequential(nn.Linear(1024,1024),nn.ReLU())
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

    elif losstype == 'L1.O1':
        #O1 target
        if use_cuda:
            out1 = Variable(torch.Tensor(out1.data.cpu().numpy())).cuda()
        else:
            out1 = Variable(torch.Tensor(out1.data.numpy()))
        
        return torch.mean(torch.sum(Fx.l1_loss(out2,out1,reduce=False),1))
        
    elif losstype == 'L1.O2':
        if use_cuda:
            out2 = Variable(torch.Tensor(out2.data.cpu().numpy())).cuda()
        else:
            out2 = Variable(torch.Tensor(out2.data.numpy()))
        return torch.mean(torch.sum(Fx.l1_loss(out1,out2,reduce=False),1))

    elif losstype == 'L2.B':
        return torch.mean(torch.sum((out1 - out2)*(out1 - out2), dim=1))
        
    elif losstype == 'L2.O1':
        if use_cuda:
            out1 = Variable(torch.Tensor(out1.data.cpu().numpy())).cuda()
        else:
            out1 = Variable(torch.Tensor(out1.data.numpy()))
        return torch.mean(torch.sum(Fx.mse_loss(out2,out1,reduce=False),1))

    elif losstype == 'L2.O2':
        if use_cuda:
            out2 = Variable(torch.Tensor(out2.data.cpu().numpy())).cuda()
        else:
            out2 = Variable(torch.Tensor(out2.data.numpy()))
        return torch.mean(torch.sum(Fx.mse_loss(out1,out2,reduce=False),1))
    
    elif losstype == 'KL.B.O1':
        out2 = out2 + 1e-10
        fac = torch.log(torch.div(out1,out2))
        fac = fac * out1
        fac_batch = torch.sum(fac,1) - torch.sum(out1,1) + torch.sum(out2,1)
        return torch.mean(fac_batch)
        
    elif losstype == 'KL.B.O2':
        out1 = out1 + 1e-10
        fac = torch.log(torch.div(out2,out1))
        fac = fac * out2
        fac_batch = torch.sum(fac,1) - torch.sum(out2,1) + torch.sum(out1,1)

        return torch.mean(fac_batch)

    elif losstype == 'KL.B':

        fac1 = torch.log(torch.div(out1,out2 + 1e-10))
        fac1 = fac1 * out1
        fac_batch1 = torch.sum(fac1,1) - torch.sum(out1,1) + torch.sum(out2,1)

        fac2 = torch.log(torch.div(out2,out1 + 1e-10))
        fac2 = fac2 * out2
        fac_batch2 = torch.sum(fac2,1) - torch.sum(out2,1) + torch.sum(out1,1)

        fac_batch = fac_batch1 + fac_batch2

        return torch.mean(fac_batch)
        
    elif losstype == 'KL.O2':
        #O2 is target, O1 is output
        #bregman -- not distributions

        if use_cuda:
            out2 = Variable(torch.Tensor(out2.data.cpu().numpy())).cuda()
        else:
            out2 = Variable(torch.Tensor(out2.data.numpy()))

        fac = Fx.kl_div(torch.log(out1),out2,reduce=False)
        fac_batch = torch.sum(fac,1) - torch.sum(out2,1) + torch.sum(out1,1) 
        return  torch.mean(fac_batch)
        
    elif losstype == 'KL.O1':
        if use_cuda:
            out1 = Variable(torch.Tensor(out1.data.cpu().numpy())).cuda()
        else:
            out1 = Variable(torch.Tensor(out1.data.numpy()))

        fac = Fx.kl_div(torch.log(out2),out1,reduce=False)
        fac_batch = torch.sum(fac,1) - torch.sum(out1,1) + torch.sum(out2,1) 
        return  torch.mean(fac_batch)

    elif losstype == 'SML1.O1':
        if use_cuda:
            out1 = Variable(torch.Tensor(out1.data.cpu().numpy())).cuda()
        else:
            out1 = Variable(torch.Tensor(out1.data.numpy()))
        
        return Fx.smooth_l1_loss(out2,out1)
        
    elif losstype == 'SML1.O2':
        if use_cuda:
            out2 = Variable(torch.Tensor(out2.data.cpu().numpy())).cuda()
        else:
            out2 = Variable(torch.Tensor(out2.data.numpy()))
        
        return Fx.smooth_l1_loss(out1,out2)

    elif losstype == 'SML1.B.O1':
        #smooth l1 becomes same as l2..all numbers between 0 and 1 so always squared
        raise ValueError('TO DO')
    elif losstype == 'SML1.B.O2':
        raise ValueError('TO DO')

    elif losstype == 'IKS.B.O1':
        #iks
        out2 = out2 + 1e-10
        fac = torch.div(out1,out2)
        fac = fac - torch.log(fac) - 1
        fac_batch = torch.sum(fac,1)
        return torch.mean(fac_batch)
        
    elif losstype == 'IKS.B.O2':

        out1 = out1 + 1e-10
        fac = torch.div(out2,out1)
        fac = fac - torch.log(fac) - 1
        fac_batch = torch.sum(fac,1)
        return torch.mean(fac_batch)

    elif losstype == 'IKS.B':
        
        fac1 = torch.div(out1,out2+1e-10)
        fac1 = fac1 - torch.log(fac1) - 1
        fac_batch1 = torch.sum(fac1,1)
        
        fac2 = torch.div(out2,out1+1e-10)
        fac2 = fac - torch.log(fac2) - 1
        fac_batch2 = torch.sum(fac2,1)

        fac_batch = fac_batch1 + fac_batch2
        return torch.mean(fac_batch)

    else:
        raise ValueError('Unknown loss type')


def setupClassifier(training_input_directory,validation_input_directory,testing_input_directory,classCount,spec_count,segment_length,output_path):

    #preload classifiers and just take avg. -- only testing

    start_time_test = time.time()
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

    #feat_train = netFeatures(oldnet,training_input_directory,args.train_list,'training',featMap)
    #feat_vald = netFeatures(oldnet,validation_input_directory,args.val_test_list,'validation',featMap)
    feat_test = netFeatures(oldnet,testing_input_directory,args.val_test_list,'testing',featMap)

    
    if args.newnet == 'newNET':
        net1 = newNET(classCount,feat_test[0][1].shape[1])
    elif args.newnet == 'newNET2':
        net1 = newNET2(classCount,feat_test[0][1].shape[1])
        
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
    epoch_best = 0
    aps, aucs, aps_ranked, aps1,aucs1,aps_ranked1, aps2,aucs2,aps_ranked2, epo_test_loss = netFeatVald(net,feat_test,loss_fn,0)

    if not os.path.exists('output/' + str(args.run_number)):
            os.makedirs('output/' + str(args.run_number))

    filename = os.path.join('output/' + str(args.run_number),'metrics_testing_' + str(args.run_number) + '_' +  str(epoch_best) + '_aps.txt')
    filename1 = os.path.join('output/' + str(args.run_number),'metrics_testing_' + str(args.run_number) + '_' + str(epoch_best) + '_aps_ranked.txt')
    filename2 = os.path.join('output/' + str(args.run_number),'metrics_testing_' + str(args.run_number) + '_' + str(epoch_best) + '_aucs.txt')

    np.save(filename,np.array([aps, aps1, aps2]))
    np.save(filename1,np.array([aps_ranked, aps_ranked1, aps_ranked2]))
    np.save(filename2,np.array([aucs, aucs1, aucs2]))

    print "Testing Maps_ranked=" + str(aps_ranked[-1]), ",MAP=" + str(aps[-1]), ",MAUC=" + str(aucs[-1]),"Maps_ranked1=" + str(aps_ranked1[-1]), ",MAP1=" + str(aps1[-1]), ",MAUC1=" + str(aucs1[-1]),"Maps_ranked2=" + str(aps_ranked2[-1]), ",MAP2=" + str(aps2[-1]), ",MAUC2=" + str(aucs2[-1]),

    print "Testing Loss {}. Done in {} seconds".format(epo_test_loss,time.time() - start_time_test)



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

    #computing geometric mean AP.
    aps_gm = metric.compute_AP_all_class(all_labels,np.sqrt(all_predictions1*all_predictions2))
    aucs_gm = metric.compute_AUC_all_class(all_labels,np.sqrt(all_predictions1*all_predictions2))
    aps_ranked_gm = metric.compute_AP_my_all_class(all_labels,np.sqrt(all_predictions1*all_predictions2))
    #print aps_gm, aucs_gm, aps_ranked_gm
    
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




if __name__ == '__main__':
    print len(sys.argv )
    if len(sys.argv) < 1:
        print "Running Instructions:\n\npython. Need something."
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
        parser.add_argument('--output_path', type=str, default='outputpath', metavar='N', help='output_path - default:outputpath')
        
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

        
        setupClassifier(args.training_features_directory, args.validation_features_directory, args.testing_features_directory, args.classCount, args.spec_count, args.segment_length,args.output_path)

