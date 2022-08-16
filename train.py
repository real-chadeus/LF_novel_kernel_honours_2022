import pathlib, datetime
import tensorflow as tf
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import ModelCheckpoint, LearningRateScheduler, CSVLogger
import load
import preprocessing.flatten
import preprocessing.hci_dataset_tools.file_io as hci_io
import kernel.lfi_se_net as se_net
import tensorflow.keras.losses as losses
from tqdm.keras import TqdmCallback
import os
import argparse

physical_devices = tf.config.list_physical_devices('GPU')
tf.config.experimental.set_memory_growth(device=physical_devices[0], enable=True)
print('tensorflow version: ', tf.__version__)
print("Num GPUs Available: ", len(tf.config.list_physical_devices('GPU')))
tf.compat.v1.RunOptions(report_tensor_allocations_upon_oom=True)
sintel_folders = ['../../datasets/Sintel_LF/Sintel_LFV_9x9_with_all_disp/ambushfight_1']
#tf.config.set_logical_device_configuration(
#    physical_devices[0],
#    [tf.config.LogicalDeviceConfiguration(memory_limit=8500)])
save_path = 'models/'

def train(model, args, dataset=(), epochs=10, batch_size=1, model_name='model1'):
    '''
    train function
    arg dataset: 2-tuple of data, first element = train data, second element = validation data.
                 Each is a 2-tuple of (data, labels) 
    '''
    if not os.path.exists(save_path + model_name):
        os.makedirs(save_path + model_name)

    lr = 0.0005
    loss = losses.MeanSquaredError()
    optimizer = Adam(learning_rate=lr)
    run_opts = tf.compat.v1.RunOptions(report_tensor_allocations_upon_oom = True)
    # model compile
    model.compile(optimizer=optimizer, loss=loss, 
                   metrics=[tf.keras.metrics.MeanAbsoluteError(),
                            tf.keras.metrics.MeanSquaredError(),
                            tf.keras.metrics.MeanAbsolutePercentageError()
                            ])

    # callbacks
    now = datetime.datetime.now().strftime("%Y-%m-%d_%H%M")
    output = pathlib.Path(f'../output/{now}_{args.model_name}_fl{args.frame_length}_{args.memo}')
    output.mkdir(exist_ok=True, parents=True)
    #checkpoint
    cp = ModelCheckpoint(filepath = f'{output}/weights.h5', monitor='val_loss',
            save_best_only=True, save_weights_only=True, verbose=0, mode='auto')
    logger = CSVLogger(save_path + model_name + '/history.csv', separator=',')

    def step_decay(epoch):
        # learning rate schedule
        factor = 1
        if epoch >= 10: factor = 0.1
        if epoch >= 15: factor = 0.01
        return lr * factor
   
    lr_schedule = LearningRateScheduler(step_decay, verbose=1)
    # fit model
    train = dataset[0]
    train_data = train[0]
    train_labels = train[1]
    val = dataset[1]
    model.fit(x=train_data, y=train_labels, batch_size=batch_size, 
                epochs=epochs, validation_data=val, verbose=0, 
                callbacks=[TqdmCallback(verbose=2), logger])
    model.save(save_path + model_name)


if __name__ == "__main__":
    # define model
    #input_shape = (7,512,7,512,3)
    input_shape = (3,436,3,436,3)
    model = se_net.build_model(input_shape=input_shape, summary=True, n_sais=9)
    # load datasets
    print('loading dataset...')
    hci = load.load_hci(img_shape=input_shape)
    sintel = load.load_sintel(img_shape=input_shape)
    dataset = (sintel, hci)

    # args settings
    parser = argparse.ArgumentParser()
    parser.add_argument('--memo', '-m', default='')
    parser.add_argument('--frame_length', '-fl', type=int, default=49)
    parser.add_argument('--model_name', default='STCLSTM')
    parser.add_argument('--train_list', default='../patch_data_fl5/train_data.txt')
    parser.add_argument('--valid_list', default='../patch_data_fl5/validation_data.txt')
    args=parser.parse_args()

    # start training
    train(model=model, args=args, dataset=dataset, epochs=10, model_name='model2')












