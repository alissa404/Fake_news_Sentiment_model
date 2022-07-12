from train import main
import time
import os
#os.environ["CUDA_VISIBLE_DEVICES"] = "1"

datasets_ch = ['taiwan']
model_names = ['MLP', 'BiGRU', 'EmotionEnhancedBiGRU']
experimental_dataset = datasets_ch[0]
experimental_model_name = model_names[2]

epochs = 30
batch_size = 32
l2_param = 0.01
lr_param = 0.001

#for experimental_model_name in model_names:
for experimental_dataset in datasets_ch:
    print('================ [{}] ================'.format(
        time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())))
    print('[Dataset]\t{}'.format(experimental_dataset))
    print('[Model]\t{}'.format(experimental_model_name))
    print()
    print('The hyparameters: ')
    print('[Epoch]\t{}'.format(epochs))
    print('[Batch Size]\t{}'.format(batch_size))
    print('[L2 param]\t{}'.format(l2_param))
    print('[Learning Rate]\t{}'.format(lr_param))
    print()
    main(experimental_dataset, experimental_model_name, epochs, batch_size, l2_param, lr_param)

