import tensorflow as tf
import os
import json
import data_preprocessing
import models
import argparse


def main(train_data_path, bg_data_path, test_data_path, save_path, img_size, batch_size, epochs):
    
    print(f'train data folder: {train_data_path}')
    print(f'test data folder: {test_data_path}')

    train_human_img_dir_path = os.path.join(train_data_path, 'images')
    train_human_ann_path = os.path.join(train_data_path, 'labels.csv')
    test_human_img_dir_path = os.path.join(test_data_path, 'images')
    test_human_ann_path = os.path.join(test_data_path, 'labels.csv')

    print("Creating dataset. Please wait...")

    train_human_dataloader = data_preprocessing.DataLoader(
                                train_human_img_dir_path,
                                train_human_ann_path,
                                img_size=img_size,
    )
    bg_dataloader = data_preprocessing.DataLoader(
        bg_data_path,
        None,
        img_size=img_size, 
                )
    test_dataloader = data_preprocessing.DataLoader(
        test_human_img_dir_path,
        test_human_ann_path,
        img_size=img_size,
    )
    
    x_human, y_human, gt_human = train_human_dataloader.create_human_data()
    x_bg, y_bg, gt_bg = bg_dataloader.create_bg_data()
    test_x, test_y, test_gt = test_dataloader.create_human_data()    
    human_data = (x_human, y_human, gt_human)
    bg_data = (x_bg, y_bg, gt_bg)
    train_dataset = data_preprocessing.DatasetCreator(human_data, bg_data).create_dataset(k_folds=False, val=False)

    print("Done!\n")
    print(f'Train input data shape : {train_dataset[0].shape}')
    print(f'Train output data shape : {train_dataset[1].shape} | {train_dataset[2].shape}')
    print(f'Test input data shape : {test_x.shape}')
    print(f'Test output data shape : {test_y.shape} | {test_gt.shape}')
    print('\nCreating model:\n')

    model = models.mobile_net2_det(img_size + (1,))
    model.summary()
    print(f'save_path: {save_path}')
    dirs = ['ieos_model']

    for i, model in zip(dirs,[model]):
        model_path = i
        model_save_name = 'model_epoch-{epoch:02d}.h5'
        csv_name = 'model_training_log.csv'
        chpkt_path = model_path + model_save_name
        csv_logger_path = model_path + csv_name
        model_checkpoint = tf.keras.callbacks.ModelCheckpoint(filepath=chpkt_path,
                                    monitor='loss',
                                    verbose=1,
                                    save_best_only=True,
                                    save_weights_only=False,
                                    mode='auto',
                                    )

        csv_logger = tf.keras.callbacks.CSVLogger(filename=csv_logger_path,
                        separator=',',
                        append=True)


        reduce_learning_rate = tf.keras.callbacks.ReduceLROnPlateau(monitor='loss',
                                            factor=0.2,
                                            patience=10,
                                            verbose=1,
                                            cooldown=0,
                                            min_lr=0.0001)
        callbacks = [model_checkpoint,
                csv_logger,
                reduce_learning_rate]        
        model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=1e-4),
                loss=[tf.keras.losses.Huber(), 'categorical_crossentropy'],
                loss_weights=[0.1, 1.]
                )
        print('\nStart fit!')
        model.fit(
            train_dataset[0], 
            [train_dataset[1], train_dataset[2]], 
            epochs=epochs,
            batch_size=batch_size,
            validation_data=(test_x, [test_y, test_gt]), validation_batch_size=16,
            callbacks=callbacks,
            )
        print('Done!')

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-c', '--cfg', dest='config', type=argparse.FileType('r'), default=None, help='cfg file in json format')
    args = parser.parse_args()
    if args.config:
        config = json.load(args.config)
    img_size = tuple(config['img_size'])
    batch_size = config['batch_size']
    epochs = config['epochs']
    main(
        config['train_data_path'], 
        config['bg_data_path'],
        config['test_data_path'], 
        config['save_path'],
        img_size,
        batch_size,
        epochs
        )