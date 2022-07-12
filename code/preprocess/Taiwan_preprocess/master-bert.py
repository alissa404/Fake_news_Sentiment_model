from train_bert import main
import time
import os
datasets_ch = ['taiwan']
experimental_dataset = datasets_ch[0]
epochs = 1
batch_size = 32

#for experimental_model_name in model_names:
for experimental_dataset in datasets_ch:
    print('================ [{}] ================'.format(
        time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())))
    print('[Dataset]\t{}'.format(experimental_dataset))
    print('[Epoch]\t{}'.format(epochs))
    print('[Batch Size]\t{}'.format(batch_size))
    main(experimental_dataset, epochs, batch_size)