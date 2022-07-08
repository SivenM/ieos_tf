import numpy as np
import pandas as pd
import tensorflow as tf
import os


class DataLoader:
    """
    Создает датасет
    """
    def __init__(self, image_path, labels_path, num_data=None, img_size=(64, 64)):
        self.image_path = image_path
        self.labels_path = labels_path
        self.img_size = img_size
        self.image_name_list = self._get_image_names(num_data)
        if self.labels_path != None:
            self.labels_df = self._read_labels()

    def _get_image_names(self, num_data):
        if num_data != None:
            return os.listdir(self.image_path)[:num_data]
        else:
            return os.listdir(self.image_path)

    def _read_labels(self):
        return pd.read_csv(self.labels_path)

    def _encode(self):
        x = np.zeros((len(self.image_name_list),) + self.img_size + (1,), dtype="float32")
        y = np.zeros((len(self.image_name_list),) + (4,), dtype="float32")
        gt_conf = np.zeros((len(self.image_name_list),) + (2,), dtype="float32")
        names_df = self.labels_df['filename']
        for i, image_name in enumerate(self.image_name_list):
            image_path = self.image_path + '/' + image_name
            image = tf.keras.preprocessing.image.load_img(image_path,
                                                          color_mode = "grayscale",
                                                          target_size=self.img_size)
            image = tf.keras.preprocessing.image.img_to_array(image)
            index_bbox = names_df.index[names_df == image_name]
            bbox_coords = self.labels_df.iloc[index_bbox[0], 4:]
            bbox_coords = np.array([bbox_coords['xmin'],              
                           bbox_coords['ymin'],
                           bbox_coords['xmax'],
                           bbox_coords['ymax']] )
            x[i] = image
            y[i] = bbox_coords
            if self.labels_df.iloc[index_bbox[0], 3] == 'person':
                gt_conf[i] = [0, 1.]
            elif self.labels_df.iloc[index_bbox[0], 3] == 'bg':
                gt_conf[i] = [1., 0]
        return (x, y, gt_conf)

    def create_human_data(self):
        X, Y, conf = self._encode() 
        X_train = X / 255      
        return X_train, Y, conf

    def create_bg_data(self):
        x_train = np.zeros((len(self.image_name_list),) + self.img_size + (1,), dtype="float32")
        y_train = np.zeros((len(self.image_name_list),) + (4,), dtype="float32")
        conf = np.zeros((len(self.image_name_list),) + (2,), dtype="float32")
        for i, image_name in enumerate(self.image_name_list):
            image_path = self.image_path + '/' + image_name
            image = tf.keras.preprocessing.image.load_img(image_path,
                                                          color_mode = "grayscale",
                                                          target_size=self.img_size)
            image = tf.keras.preprocessing.image.img_to_array(image)
            x_train[i] = image / 255
            conf[i] = [1., 0]
        return x_train, y_train, conf

class DatasetCreator:
    """
    Создает датасет для обучения в зависимости от соотношения
    """
    def __init__(self, human_data, bg_data, test_type=None):
        """
        human_data: кортеж имеет нобор изображений, таргетов и гт боксов человека
        bg_data: кортеж имеет нобор изображений, таргетов и гт боксов фона
        """
        self.test_type = test_type
        self.human_data = human_data
        self.bg_data = bg_data
        self.num_human_data = self.human_data[0].shape[0]

    def shuffle_data(self, dataset):
        seed = 42
        for data in dataset:
            if type(data) != np.ndarray:
                data = data.numpy()
            np.random.seed(seed)
            np.random.shuffle(data)
        return dataset

    def train_val(self, data):
        fold = 5
        k=0
        num_val_samples = data[0].shape[0] // fold
        img_val = data[0][k * num_val_samples: (k+1) * num_val_samples]
        target_val= data[1][k * num_val_samples: (k+1) * num_val_samples]
        gt_val = data[2][k * num_val_samples: (k+1) * num_val_samples]

        img_train = np.concatenate([data[0][:k * num_val_samples], 
                                        data[0][(k + 1) * num_val_samples:]],
                                        axis=0)
        target_train = np.concatenate([data[1][:k * num_val_samples], 
                                        data[1][(k + 1) * num_val_samples:]],
                                        axis=0)
        gt_train = np.concatenate([data[2][:k *num_val_samples], 
                                         data[2][(k + 1) * num_val_samples:]],
                                            axis=0)
        return (img_train, target_train, gt_train), (img_val, target_val, gt_val)

    def create_dataset(self, k_folds=True, val=False):
        if k_folds:
            return [self.human_data, self.bg_data]
        else:
            human_data_shuffled = self.shuffle_data(self.human_data)
            bg_data_shuffled = self.shuffle_data(self.bg_data)
            if val:
                train_human, val_human = self.train_val(human_data_shuffled)
                train_bg, val_bg = self.train_val(bg_data_shuffled)

                train_data = (
                    np.concatenate([train_human[0], train_bg[0]], axis=0),
                    np.concatenate([train_human[1], train_bg[1]], axis=0),
                    np.concatenate([train_human[2], train_bg[2]], axis=0),
                    )
                val_data = (
                    np.concatenate([val_human[0], val_bg[0]], axis=0),
                    np.concatenate([val_human[1], val_bg[1]], axis=0),
                    np.concatenate([val_human[2], val_bg[2]], axis=0),
                    )
                #train_dataset = tf.data.Dataset.from_tensor_slices(train_data).shuffle(buffer_size=5000, seed=42).batch(32).prefetch(1)
                #val_dataset = tf.data.Dataset.from_tensor_slices(val_data).shuffle(buffer_size=50, seed=42).batch(16).prefetch(1)
                return self.shuffle_data(train_data), (val_data)
            else:
                return self.shuffle_data((
                    np.concatenate([self.human_data[0], self.bg_data[0]], axis=0),
                    np.concatenate([self.human_data[1], self.bg_data[1]], axis=0),
                    np.concatenate([self.human_data[2], self.bg_data[2]], axis=0),
                    ))