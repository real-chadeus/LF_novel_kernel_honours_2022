import pathlib, datetime
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import ModelCheckpoint, LearningRateScheduler, CSVLogger
import load
import preprocessing.flatten
import preprocessing.hci_dataset_tools.file_io as hci_io
import kernel.lfi_se_net as se_net
print('tensorflow version: ', tf.__version__)
print("Num GPUs Available: ", len(tf.config.list_physical_devices('GPU')))

def train(model, args, dataset):
    # model compile
    lr = 0.0005
    optimizer = Adam(learning_rate=lr)
    model.compile(optimizer=optimizer, loss=get_loss_function())

    # data generator
    if args.frame_length == 5:
        input_generator = input_generator_fl5
    elif args.frame_length == 4:
        input_generator = input_generator_fl4
    elif args.frame_length == 3:
        input_generator = input_generator_fl3
    else:
        raise Exception(f'args.frame_length must be an integer between 3 and 5, but receive {args.frame_length}.')
    train_datagen = input_generator(args.train_list)
    valid_datagen = input_generator(args.valid_list, val_mode=True)

    # callbacks
    now = datetime.datetime.now().strftime("%Y-%m-%d_%H%M")
    output = pathlib.Path(f'../output/{now}_{args.model_name}_fl{args.frame_length}_{args.memo}')
    output.mkdir(exist_ok=True, parents=True)
    cp = ModelCheckpoint(filepath = f'{output}/weights.h5', monitor='val_loss',
            save_best_only=True, save_weights_only=True, verbose=0, mode='auto')
    logger = CSVLogger(f'{output}/history.csv')
    ## learning rate
    def step_decay(epoch):
        factor = 1
        if epoch >= 10: factor = 0.1
        if epoch >= 15: factor = 0.01
        return lr * factor
    lr_schedule = LearningRateScheduler(step_decay, verbose=1)

    # START training
    batch_size = 64
    epochs = 20
    model.fit_generator(
        generator=train_datagen.flow_from_directory(batch_size),
        steps_per_epoch=len(train_datagen.data_paths) // batch_size,
        epochs=epochs,
        initial_epoch=0,
        verbose=1,
        callbacks=[cp, logger, lr_schedule],
        validation_data=valid_datagen.flow_from_directory(batch_size),
        validation_steps=len(valid_datagen.data_paths) // batch_size,
        max_queue_size=20
    )



if __name__ == "__main__":
    # define model
    model = se_net.build_model()
    print(model.summary())
    
    dataset = load.load_dataset()
    # args settings
    parser = argparse.ArgumentParser()
    parser.add_argument('--memo', '-m', default='')
    parser.add_argument('--frame_length', '-fl', type=int, default=5)
    parser.add_argument('--model_name', default='STCLSTM')
    parser.add_argument('--train_list', default='../patch_data_fl5/train_data.txt')
    parser.add_argument('--valid_list', default='../patch_data_fl5/validation_data.txt')
    args=parser.parse_args()

    # start train
    train(model=model, args=args, dataset=dataset)






