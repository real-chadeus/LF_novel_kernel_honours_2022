import pathlib, datetime
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import ModelCheckpoint, LearningRateScheduler, CSVLogger
import load
import preprocessing.flatten
import preprocessing.hci_dataset_tools.file_io as hci_io
import kernel.lfi_se_net as se_net
import tensorflow.keras.losses as losses
print('tensorflow version: ', tf.__version__)
print("Num GPUs Available: ", len(tf.config.list_physical_devices('GPU')))

def train(model, args, dataset):
    # model compile
    lr = 0.0005
    loss = losses.BinaryCrossentropy()
    optimizer = Adam(learning_rate=lr)
    model.compile(optimizer=optimizer, loss=loss)

    # callbacks
    now = datetime.datetime.now().strftime("%Y-%m-%d_%H%M")
    output = pathlib.Path(f'../output/{now}_{args.model_name}_fl{args.frame_length}_{args.memo}')
    output.mkdir(exist_ok=True, parents=True)
    #checkpoint
    cp = ModelCheckpoint(filepath = f'{output}/weights.h5', monitor='val_loss',
            save_best_only=True, save_weights_only=True, verbose=0, mode='auto')
    logger = CSVLogger(f'{output}/history.csv')

    def step_decay(epoch):
        # reduce learning rate by some factor based on number of epochs
        factor = 1
        if epoch >= 10: factor = 0.1
        if epoch >= 15: factor = 0.01
        return lr * factor
   
    lr_schedule = LearningRateScheduler(step_decay, verbose=1)
    batch_size = 32
    epochs = 10
    # fit model
    model.fit()



if __name__ == "__main__":
    # define model
    input_shape = (7,420,7,420,3)
    model = se_net.build_model()
    
    dataset = load.load_dataset()
    # args settings
    parser = argparse.ArgumentParser()
    parser.add_argument('--memo', '-m', default='')
    parser.add_argument('--frame_length', '-fl', type=int, default=49)
    parser.add_argument('--model_name', default='STCLSTM')
    parser.add_argument('--train_list', default='../patch_data_fl5/train_data.txt')
    parser.add_argument('--valid_list', default='../patch_data_fl5/validation_data.txt')
    args=parser.parse_args()

    # start train
    train(model=model, args=args, dataset=dataset)






