import argparse
import os
import sys

if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='BRATS Pipeline')

    parser.add_argument('-config', type=str, required=True)
    parser.add_argument('-mode', type=str, default='default')
    #parser.add_argument('-gpu', type=str, default='0')

    args = parser.parse_args()
    #print 'Running Deep Learning Pipeline on gpu... %s' % args.gpu

    #os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"   # see issue #152
    #os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu
    import train

    train.pipeline(args.config, mode=args.mode)