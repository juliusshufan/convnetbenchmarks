import argparse
import torch
from torch.autograd import Variable
import torch.nn as nn
import torchvision.models as models
import torch.optim as optim
import time
import subprocess

# benchmark settings
parser = argparse.ArgumentParser(description='PyTorch Convnet Benchmark')
parser.add_argument('--no-cuda', action='store_true', default=False,
                   help='disable CUDA')

args = parser.parse_args()
args.cuda = not args.no_cuda and torch.cuda.is_available()

archs = {'alexnet': 128,
         'vgg11': 64,
         'inception_v3': 128,
         'resnet50': 128}
steps = 10 # nb of steps in loop to average perf
nDryRuns = 10


if args.cuda:
    import torch.backends.cudnn as cudnn
    cudnn.benchmark = True
    
    kernel = 'cudnn'
    p = subprocess.check_output('nvidia-smi --query-gpu=name --format=csv', 
                                shell=True)
    device_name = str(p).split('\\n')[1]
else:
    kernel = 'nn'
    p = subprocess.check_output('cat /proc/cpuinfo | grep name | head -n 1',
                                shell = True)
    device_name = str(p).split(':')[1][:-3]

print('Running on device: %s' % (device_name))


def main():
    for arch, batch_size in archs.items():
        t = time.time()
        if arch=='inception_v3':
            data_ = torch.randn(batch_size, 3, 299, 299)
        else:
            data_ = torch.randn(batch_size, 3, 224, 224)
        target_ = torch.arange(1, batch_size + 1).long()        
        net = models.__dict__[arch]() # no need to load pre-trained weights for dummy data
        
        optimizer = optim.SGD(net.parameters(), lr=0.01)
        criterion = nn.CrossEntropyLoss()

        if args.cuda:
            data_, target_ = data_.cuda(), target_.cuda()
            net.cuda()
            criterion = criterion.cuda()
        
        net.eval()
        
        print('ModelType: %s, Kernels: %s Input shape: %dx3x224x224 (Inception-v3 will be 3*299*299)' % (
                arch, kernel, batch_size))
        data, target = Variable(data_), Variable(target_)
        
        for i in range(nDryRuns):
            optimizer.zero_grad()   # zero the gradient buffers
            output = net(data)
            loss = criterion(output, target)
            loss.backward()
            optimizer.step()    # Does the update

        time_fwd, time_bwd, time_upt = 0, 0, 0
        
        for i in range(steps):
            optimizer.zero_grad()   # zero the gradient buffers
            t1 = time.time()
            output = net(data)
            t2 = time.time()
            loss = criterion(output, target)
            loss.backward()
            t3 = time.time()
            optimizer.step()    # Does the update
            t4 = time.time()
            time_fwd = time_fwd + (t2 - t1)
            time_bwd = time_bwd + (t3 - t2)
            time_upt = time_upt + (t4 - t3)
        
        time_fwd_avg = time_fwd / steps * 1000
        time_bwd_avg = time_bwd / steps * 1000
        time_upt_avg = time_upt / steps * 1000
        
        # update not included!
        time_total = time_fwd_avg + time_bwd_avg
    
        print("%-30s %10s %10.2f %10.2f" % (kernel, ':forward:', time_fwd_avg, batch_size*1000/time_fwd_avg))
        print("%-30s %10s %10.2f" % (kernel, ':backward:', time_bwd_avg))
        print("%-30s %10s %10.2f" % (kernel, ':update:', time_upt_avg))
        print("%-30s %10s %10.2f %10.2f" % (kernel, ':total:', time_total, batch_size*1000/time_total))
        

if __name__ == '__main__':
    main()
