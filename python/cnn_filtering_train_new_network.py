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

class NET(nn.Module):
    
    def __init__(self,nclass):
        super(NET,self).__init__()
        self.globalpool = Fx.avg_pool2d

        self.layer1 = nn.Sequential(nn.Conv2d(1,64,kernel_size=3,padding=1),nn.BatchNorm2d(64,momentum=0.0),nn.ReLU())
        self.layer2 = nn.Sequential(nn.Conv2d(64,64,kernel_size=3,padding=1),nn.BatchNorm2d(64,momentum=0.0),nn.ReLU())
        self.layer3 = nn.MaxPool2d((1,2)) 

        self.layer4 = nn.Sequential(nn.Conv2d(64,128,kernel_size=3,padding=1),nn.BatchNorm2d(128,momentum=0.0),nn.ReLU())
        self.layer5 = nn.Sequential(nn.Conv2d(128,128,kernel_size=3,padding=1),nn.BatchNorm2d(128,momentum=0.0),nn.ReLU())
        self.layer6 = nn.MaxPool2d((1,2)) 

        self.layer7 = nn.Sequential(nn.Conv2d(128,256,kernel_size=3,padding=1),nn.BatchNorm2d(256,momentum=0.0),nn.ReLU())
        self.layer8 = nn.Sequential(nn.Conv2d(256,256,kernel_size=3,padding=1),nn.BatchNorm2d(256,momentum=0.0),nn.ReLU())
        self.layer9 = nn.MaxPool2d((1,2))

        self.layer10 = nn.Sequential(nn.Conv2d(256,512,kernel_size=3,padding=1),nn.BatchNorm2d(512,momentum=0.0),nn.ReLU())
        self.layer11 = nn.Sequential(nn.Conv2d(512,512,kernel_size=3,padding=1),nn.BatchNorm2d(512,momentum=0.0),nn.ReLU())
        self.layer12 = nn.MaxPool2d((1,2)) 

        self.layer13 = nn.Sequential(nn.Conv2d(512,1024,kernel_size=(1,8)),nn.BatchNorm2d(1024,momentum=0.0),nn.ReLU())
        self.layer14 = nn.Sequential(nn.Conv2d(1024,1024,kernel_size=1),nn.BatchNorm2d(1024,momentum=0.0),nn.ReLU())
        self.layer15 = nn.Sequential(nn.Conv2d(1024,nclass,kernel_size=1),nn.Sigmoid())
        
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
        out = self.layer14(out)
        out1 = self.layer15(out)
        
        out = self.globalpool(out1,kernel_size=out1.size()[2:])
        out = out.view(out.size(0),-1)
        return out #,out1

    def xavier_init(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                init.xavier_uniform(m.weight, gain=nn.init.calculate_gain('relu'))
                m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()


class newNET(nn.Module):
    
    def __init__(self,nclass):
        super(newNET,self).__init__()
        
        self.globalpool = Fx.avg_pool2d
        self.layer1 = nn.Sequential(nn.Conv2d(1,64,kernel_size=(1,3),padding=(0,1)),nn.BatchNorm2d(64,momentum=0.1),nn.ReLU())
        self.layer2 = nn.Sequential(nn.Conv2d(64,64,kernel_size=(1,3),padding=(0,1)),nn.BatchNorm2d(64,momentum=0.1),nn.ReLU())
        self.layer3 = nn.MaxPool2d((1,2)) 

        self.layer4 = nn.Sequential(nn.Conv2d(64,128,kernel_size=(1,3),padding=(0,1)),nn.BatchNorm2d(128,momentum=0.1),nn.ReLU())
        self.layer5 = nn.Sequential(nn.Conv2d(128,128,kernel_size=(1,3),padding=(0,1)),nn.BatchNorm2d(128,momentum=0.1),nn.ReLU())
        self.layer6 = nn.MaxPool2d((1,2)) 

        self.layer7 = nn.Sequential(nn.Conv2d(128,256,kernel_size=(1,3),padding=(0,1)),nn.BatchNorm2d(256,momentum=0.1),nn.ReLU())
        self.layer8 = nn.Sequential(nn.Conv2d(256,256,kernel_size=(1,3),padding=(0,1)),nn.BatchNorm2d(256,momentum=0.1),nn.ReLU())
        self.layer9 = nn.MaxPool2d((1,2))

        self.layer10 = nn.Sequential(nn.Conv2d(256,512,kernel_size=(1,3),padding=(0,1)),nn.BatchNorm2d(512,momentum=0.1),nn.ReLU())
        self.layer11 = nn.Sequential(nn.Conv2d(512,512,kernel_size=(1,3),padding=(0,1)),nn.BatchNorm2d(512,momentum=0.1),nn.ReLU())
        self.layer12 = nn.MaxPool2d((1,2)) 

        self.layer13 = nn.Sequential(nn.Conv2d(512,1024,kernel_size=(1,8)),nn.BatchNorm2d(1024,momentum=0.1),nn.ReLU())
        self.layer14 = nn.Sequential(nn.Conv2d(1024,1024,kernel_size=1),nn.BatchNorm2d(1024,momentum=0.1),nn.ReLU())
        self.layer15 = nn.Sequential(nn.Conv2d(1024,nclass,kernel_size=1),nn.Sigmoid())
        
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
        out = self.layer14(out)
        out1 = self.layer15(out)
        
        out = self.globalpool(out1,kernel_size=out1.size()[2:])
        out = out.view(out.size(0),-1)
        return out
        

    def xavier_init(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                init.xavier_uniform(m.weight, gain=nn.init.calculate_gain('relu'))
                m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()



use_cuda = torch.cuda.is_available()

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


def read_file(filename):
	listname=[]
        with open(filename,'r') as f:
                lines = f.readlines()
                for line in lines:
                        listname.append((line.split(' ')[0],line.split(' ')[1]))
                f.close()
        return listname

def retrieve_index(number,range_list1):
        range_list1.sort()
        for ind in range(len(range_list1)):
                if number >= range_list1[ind] and number < range_list1[ind+1]: 
                        return ind+1
                elif number < range_list1[ind]:
                        return ind
                else:
                        continue


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


def setupClassifier(training_input_directory,validation_input_directory,testing_input_directory,classCount,spec_count,segment_length,learning_rate,output_path):
        
        sys.stdout = c1
        loss_fn = nn.BCELoss()
        
        oldnet = NET(classCount)
        if use_cuda:
            oldnet.cuda()
            oldnet = torch.nn.DataParallel(oldnet, device_ids=range(torch.cuda.device_count()))
            if args.preloading:
                load_model_direct(oldnet,args.preloading_model_path)
	else:
            if args.preloading:
                load_model_cpu(oldnet,args.preloading_model_path,False)
        seg_train, seg_labels = netFilter(oldnet,training_input_directory,args.selK,args.segsize,args.hopsize)
        #aps, aucs, aps_ranked, epo_val_loss = netVald(oldnet,validation_input_directory,loss_fn)


        net = newNET(classCount)
        net.xavier_init()
        #optimizer = optim.SGD(net.parameters(), lr=args.learning_rate, weight_decay=0.0,momentum=0.0)#args.momentum)
        print args.learning_rate
        optimizer = optim.Adam(net.parameters(), lr=args.learning_rate)
	#scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.87)
        
        
	epoch_best=0
	aps_best=0
	model_path=''
        
        if use_cuda:
            net.cuda()
            net = torch.nn.DataParallel(net, device_ids=range(torch.cuda.device_count()))
            
        for epoch in range(args.num_epochs):
            print "Current epoch is " + str(epoch)
                
            #scheduler.step()
            
            #train on filtered
            netFilterTrain(net,(seg_train,seg_labels),optimizer,loss_fn,epoch)
            
            #validation
            start_time_val = time.time()
            aps, aucs, aps_ranked, epo_val_loss = netVald(net,validation_input_directory,loss_fn)

            filename = os.path.join('output_' + str(args.run_number),'metrics_validation_' + str(args.run_number) + '_' +  str(epoch) + '_aps.txt')
            filename1 = os.path.join('output_' + str(args.run_number),'metrics_validation_' + str(args.run_number) + '_' + str(epoch) + '_aps_ranked.txt')
            filename2 = os.path.join('output_' + str(args.run_number),'metrics_validation_' + str(args.run_number) + '_' + str(epoch) + '_aucs.txt')

            if not os.path.isdir(os.path.dirname(filename)):
                os.makedirs(os.path.dirname(filename))
                
            np.save(filename,aps)
            np.save(filename1,aps_ranked)
            np.save(filename2,aucs)
            if aps[-1] > aps_best:
                aps_best=aps[-1]
                epoch_best=epoch
                if not os.path.exists('output_'+str(args.run_number)):
                    os.mkdir('output_' + str(args.run_number))
                torch.save(net.state_dict(),'output_' + str(args.run_number)+'/model_path.' + str(args.run_number) + '_' +str(epoch)+  '.pkl')
                model_path='output_'+str(args.run_number) + '/model_path.' + str(args.run_number) + '_' + str(epoch) + '.pkl'

            print "{} Validation done in {} seconds. Validation Loss {}".format(epoch, time.time() - start_time_val, epo_val_loss)

            
        # testing
        start_time_test = time.time()
	if use_cuda:
            load_model_direct(net,model_path)
	else:
            load_model_cpu(net,model_path)
        
        aps, aucs, aps_ranked, epo_test_loss = netTest(net,testing_input_directory,loss_fn)

        filename = os.path.join('output_' + str(args.run_number),'metrics_testing_' + str(args.run_number) + '_' +  str(epoch_best) + '_aps.txt')
        filename1 = os.path.join('output_' + str(args.run_number),'metrics_testing_' + str(args.run_number) + '_' + str(epoch_best) + '_aps_ranked.txt')
        filename2 = os.path.join('output_' + str(args.run_number),'metrics_testing_' + str(args.run_number) + '_' + str(epoch_best) + '_aucs.txt')
        np.save(filename,aps)
        np.save(filename1,aps_ranked)
        np.save(filename2,aucs)
        print "{} Testing done in {} seconds. Testing Loss {}".format(epoch, time.time() - start_time_test,epo_test_loss)


def netFilterTrain(net,train_data,optimizer,loss_fn,epoch):
    
    #############
    ### Training 
    #############

    net.train()
    epo_train_loss = 0
    batch_count = 0
    start_time = time.time()
    
    trdata, trlabels = train_data
    
    for batch in iterate_minibatches_torch(trdata,args.sgd_batch_size,trlabels,shuffle=True,last_batch=True):
        inputs,targets=batch	
        if use_cuda:
            indata = Variable(inputs).cuda()
            lbdata = Variable(targets).cuda()
        else:
            indata = Variable(inputs)
            lbdata = Variable(targets)

        optimizer.zero_grad()
        batch_pred = net(indata)
        #print batch_pred[0:5,:], lbdata[0:5,:]

        batch_train_loss = loss_fn(batch_pred,lbdata)
        batch_train_loss.backward()
        optimizer.step()

        epo_train_loss += batch_train_loss.data[0]
        batch_count += 1

    epo_train_loss = epo_train_loss/batch_count

    print "{} Training done in {} seconds. Training Loss {}".format(epoch, time.time() - start_time, epo_train_loss)



def netTrain(net,training_input_directory,optimizer,loss_fn,epoch):
    
    #############
    ### Training 
    #############

    net.train()
    epo_train_loss = 0
    batch_count = 0
    start_time = time.time()
    

    random.shuffle(args.train_list)
    training_set_file_list = []
    for i in range(len(args.train_list)):
        training_set_file_list.append(args.train_test_split_path + '/b_' + str(args.train_list[i]) + '_training.list')
        training_files = read_file(training_set_file_list[i])
        feat_dim = int(args.train_list[i])
        if len(training_files) == 0:
            continue
        for batch in dataloader(training_input_directory,training_files,args.sgd_batch_size,feat_dim,shuffle=True,last_batch=True):
            inputs,targets=batch	
            if targets is None:
                print 'No Target given to minibatch iterator. Using a dummy target with two outputs'
                targets = np.zeros((inputs.shape[0],2))

            indata = torch.Tensor(inputs)
            lbdata = torch.Tensor(targets)

            if use_cuda:
                indata = Variable(indata).cuda()
                lbdata = Variable(lbdata).cuda()
            else:
                indata = Variable(indata)
                lbdata = Variable(lbdata)

            optimizer.zero_grad()
            batch_pred = net(indata)

            batch_train_loss = loss_fn(batch_pred,lbdata)
            batch_train_loss.backward()
            optimizer.step()

            epo_train_loss += batch_train_loss.data[0]
            batch_count += 1
    
    epo_train_loss = epo_train_loss/batch_count

    print "{} Training done in {} seconds. Training Loss {}".format(epoch, time.time() - start_time, epo_train_loss)

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

            batch_pred = net(indata)

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


def netFilter(net,training_input_directory,selK,segsize,hopsize):

    ###############
    ### Select instances 
    ###############

    
    all_predictions = []
    all_labels = []
    training_set_file_list = []
    net.eval()

    for i in range(len(args.train_list)):
        training_set_file_list.append(args.train_test_split_path + '/b_' + str(args.train_list[i]) + '_training.list')
        
        training_files = read_file(training_set_file_list[i])
        feat_dim=int(args.train_list[i])
        if len(training_files) == 0:
            continue
        for batch in dataloader(training_input_directory,training_files,1,feat_dim,shuffle=False,last_batch=True):
            #bsize fixed to 1
            
            inputs,targets = batch
                
            inputs = inputs[0,0,:,:] # now 2D
            indata = view_as_windows(inputs,window_shape=(segsize,inputs.shape[1]),step=hopsize)
            #print inputs.shape,args.train_list[i]
            segdata = torch.Tensor(indata)
            if use_cuda:
                segdata = Variable(segdata,volatile=True).cuda()
            else:
                segdata = Variable(segdata,volatile=True)

            batch_pred = net(segdata)
            
            inres = batch_pred.data.cpu().numpy()#.tolist()
            #print np.where(targets)[1]
            for cl in np.where(targets)[1]:
                cres = inres[:,cl]
                #get top k
                cres_sort_id = np.argsort(-cres)
                if cres_sort_id.shape[0] >= selK:
                    selKdata = indata[cres_sort_id[0:selK],:,:,:]
                else:
                    selKdata = indata[cres_sort_id,:,:,:]
                    
                selKlabels = np.zeros(inres.shape[1])
                selKlabels[cl] = 1.0
                selKlabels = [selKlabels] * selKdata.shape[0]
                
                all_predictions.extend(list(selKdata))
                all_labels.extend(selKlabels)
                #print len(all_predictions), len(all_labels)

    all_predictions = np.array(all_predictions)
    all_labels = np.array(all_labels)
    print "Total ", all_predictions.shape, " segments selected and labels(", all_labels.shape, ")"
    return all_predictions, all_labels


def netTest(net,testing_input_directory,loss_fn):

    ###############
    ### Testing 
    ###############
    net.eval()

    epo_test_loss = 0
    test_batch_count = 0
    start_time_test = time.time()
    all_predictions = []
    all_labels = []
    testing_set_file_list = []
    
    for i in range(len(args.val_test_list)):
        testing_set_file_list.append(args.train_test_split_path + '/b_' + str(args.val_test_list[i]) + '_testing.list')
        testing_files = read_file(testing_set_file_list[i])
        feat_dim = int(args.val_test_list[i])
        if len(testing_files) == 0:
            continue
        for batch in dataloader(testing_input_directory,testing_files,args.sgd_batch_size,feat_dim,shuffle=False,last_batch=True):
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

            batch_pred = net(indata)

            batch_test_loss = loss_fn(batch_pred,lbdata)
            epo_test_loss += batch_test_loss.data[0]
            test_batch_count += 1

            inres = batch_pred.data.cpu().numpy().tolist()
            all_predictions.extend(inres)
            all_labels.extend(lbdata.data.cpu().numpy().tolist())
        
    epo_test_loss = epo_test_loss/test_batch_count
    print "Length of all prediction - Test " + str(len(all_predictions)), len(all_labels)
    all_predictions = np.array(all_predictions)
    all_labels = np.array(all_labels)
    aps = metric.compute_AP_all_class(all_labels,all_predictions)
    aucs = metric.compute_AUC_all_class(all_labels,all_predictions)
    aps_ranked = metric.compute_AP_my_all_class(all_labels,all_predictions)

    epo_test_loss = epo_test_loss/test_batch_count

    
    print "Test APS " + str(aps[-1]), "Test aps ranked " + str(aps_ranked[-1]), "Test AUCS " + str(aucs[-1])

    return aps, aucs, aps_ranked, epo_test_loss

    
    
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
        parser.add_argument('--preloading', type=bool, default=False, metavar='N',
                            help='preloading- default:False')
        parser.add_argument('--preloading_model_path', type=str, default='modelpath', metavar='N',
                            help='preloading_model_path - default:modelpath')
        parser.add_argument('--selK', type=int, default=1, metavar='N',
                            help='classCount - default:1')
        parser.add_argument('--segsize', type=int, default=1, metavar='N',
                            help='segsize - default:1')
        parser.add_argument('--hopsize', type=int, default=1, metavar='N',
                            help='hopsize - default:1')


        args = parser.parse_args()
        c1 = Logger(args.run_number)

        if not os.path.exists(args.output_path):
            os.makedirs(args.output_path)

        setupClassifier(args.training_features_directory, args.validation_features_directory, args.testing_features_directory, args.classCount, args.spec_count, args.segment_length,args.learning_rate,args.output_path)

