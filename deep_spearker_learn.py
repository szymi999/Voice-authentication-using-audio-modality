import random
import matplotlib.pyplot as plt
import numpy as np
from tensorflow.keras import backend as K
from tensorflow.keras.models import Model
import tensorflow.keras.layers as K_layers
import tensorflow.keras.models as K_models
import tensorflow as tf
from PIL import Image
from skimage import io
import os
import pandas as pd

# data_path = "H:/librespeech/data_numpy_trim_4_sil_01_1percent_v1.2"
data_path = "normalized_data"
input_shape = ((90,173,1))

def data_generator(batch_size=16):
    while True:
        a = []
        p = []
        n = []
        
        for _ in range(batch_size):
            x = get_random_triple(data_path)

            a.append(x[0])
            p.append(x[1]) 
            n.append(x[2]) 


        yield ([np.array(a), np.array(p), np.array(n)], np.zeros((batch_size, 1)).astype("float32"))

def get_random_triple(path):
    data_array = []


    person = random.sample(os.listdir(path), 2)

    pathanchor = path+"/"+person[0]
    filesanchor = random.sample(os.listdir(pathanchor), 2)
    
    for i in range(2):
        #print("{}/{}".format(pathanchor, filesanchor[i]))
        data_array.append(np.load("{}/{}".format(pathanchor, filesanchor[i])))

    pathnegative = path+"/"+person[1]
    filesnegative = random.sample(os.listdir(pathnegative), 1)

    for i in range(1):
        #print("{}/{}".format(pathnegative, filesnegative[i]))
        data_array.append(np.load("{}/{}".format(pathnegative, filesnegative[i])))
    
    
    return data_array 

def cosine_similarity(list_1, list_2):
    cos_sim = tf.tensordot(list_1, list_2, 0) / (tf.norm(list_1) * tf.norm(list_2))
    return cos_sim

def calc_cos_tensor(tensor1, tensor2):
    return tf.tensordot(tensor1, tensor2, axes=1) / (tf.norm(tensor1) * tf.norm(tensor2))

# def triplet_loss(y_true, y_pred, margin=0.1):
#     OUT_DIM = 100
#     anchor_out = y_pred[:, 0:OUT_DIM]
#     positive_out = y_pred[:, OUT_DIM:2*OUT_DIM]
#     negative_out = y_pred[:, 2*OUT_DIM:3*OUT_DIM]

#     cosine_loss = tf.keras.losses.CosineSimilarity(axis=1)
#     distance1 = cosine_loss(anchor_out, positive_out)
#     distance2 = cosine_loss(anchor_out, negative_out)

#     # distance1 = tf.keras.losses.cosine_similarity(anchor_out, positive_out)
#     # distance2 = tf.keras.losses.cosine_similarity(anchor_out, negative_out)

    
#     return tf.keras.backend.clip((1 + distance1) + (1 - distance2) + margin, 0., None)

def triplet_loss(y_true, y_pred, margin=0.5):
    anchor_out = y_pred[:, 0:100]
    positive_out = y_pred[:, 100:200]
    negative_out = y_pred[:, 200:300]
    
    if tf.math.count_nonzero(anchor_out) == 6400:
        for a in anchor_out:
            tf.print(a)
    
    nr_iters = [8, 4, 2, 1]
    tf_list = []
    tf_list_tmp = []
    
    for j in nr_iters:
        for i in range(j):
            if j == nr_iters[0]:
                cos1_1 = calc_cos_tensor(anchor_out[2*i], positive_out[2*i])
                cos1_2 = calc_cos_tensor(anchor_out[2*i], negative_out[2*i])
                loss1 = (1-cos1_1) - (1-cos1_2) + margin
                
                cos2_1 = calc_cos_tensor(anchor_out[2*i+1], positive_out[2*i+1])
                cos2_2 = calc_cos_tensor(anchor_out[2*i+1], negative_out[2*i+1])
                loss2 = (1-cos2_1) - (1-cos2_2) + margin
                
                tf_list_tmp.append(tf.stack([loss1, loss2], 0))
            else:
                tf_list_tmp.append(tf.concat([tf_list[2*i], tf_list[2*i+1]], 0))

        tf_list = tf_list_tmp
        tf_list_tmp = []
        
                                              
    return tf.keras.backend.mean(tf_list[0])

def model():
    input_layer = K_layers.Input(input_shape)
    x = K_layers.Conv2D(32, 5, activation="relu")(input_layer)
    x = K_layers.Conv2D(32, 5, activation="relu")(x)
    x = K_layers.MaxPool2D(2)(x)
    x = K_layers.Conv2D(64, 5, activation="relu")(x)
    x = K_layers.Conv2D(64, 5, activation="relu")(x)
    x = K_layers.MaxPool2D(2)(x)
    x = K_layers.Conv2D(128, 5, activation="relu")(x)
    x = K_layers.Flatten()(x)
    x = K_layers.Dense(100, activation="relu")(x)

    model = Model(input_layer, x)
    model.summary()


    triplet_model_a = K_layers.Input(input_shape)
    triplet_model_p = K_layers.Input(input_shape)
    triplet_model_n = K_layers.Input(input_shape)
    triplet_model_out = K_layers.Concatenate()([model(triplet_model_a), model(triplet_model_p), model(triplet_model_n)])
    triplet_model = Model([triplet_model_a, triplet_model_p, triplet_model_n], triplet_model_out)
    triplet_model.summary()

    triplet_model.compile(loss=triplet_loss, optimizer=tf.keras.optimizers.Adam(learning_rate=1e-6))

    history = triplet_model.fit(data_generator(), steps_per_epoch=1000, epochs=12)

    triplet_model.compile(loss=None, optimizer="adam")

    triplet_model.save("random_short_network.h5")
    hist_df = pd.DataFrame(history.history) 


    hist_csv_file = 'history.csv'
    with open(hist_csv_file, mode='w') as f:
        hist_df.to_csv(f)

model()


