import os
import glob 
import pandas as pd
import glob 
import numpy as np
import tensorflow as tf

target_shape = (200, 200)


def preprocess_image(filename):
    """
    Load the specified file as a JPEG image, preprocess it and
    resize it to the target shape.
    """

    image_string = tf.io.read_file(filename)
    image = tf.image.decode_jpeg(image_string, channels=3)
    image = tf.image.convert_image_dtype(image, tf.float32)
    image = tf.image.resize(image, target_shape)
    return image


def preprocess_triplets(anchor, positive, negative):
    """
    Given the filenames corresponding to the three images, load and
    preprocess them.
    """

    return (
        preprocess_image(anchor),
        preprocess_image(positive),
        preprocess_image(negative),
    )



def prepare_data (args, num_data):
    all_files = glob.glob(args.input_data_dir + os.sep + "*.*")
    D = {}  
    for f in all_files:
       fname = os.path.basename(f)
       parts = fname.split("_")
       if parts[1] not in D:
          D[parts[1]] = [f]
       else:
          D[parts[1]].append (f)

    keys = list(D.keys())
    df = pd.DataFrame(columns=['left','right','diff'])
    count = 0 
    for i in range (1, num_data):
        r = np.random.randint(95, size=[4])
        c = r[0] %3        
        if c >=1 :
            d = c-1
        else:
            d = c+1 
        data = [D[keys[c]][r[1]], D[keys[c]][r[2]], D[keys[d]][r[3]]]
        df.loc[count] = data
        count +=1
    return df 


def get_data (args):

    df = prepare_data (args, 10000)

    print("df:", df.shape, df.columns)

    anchor_images=df['left'].to_list()
    positive_images  = df['right'].to_list()
    negative_images = df['diff'].to_list()

    image_count = len(anchor_images)

    anchor_dataset = tf.data.Dataset.from_tensor_slices(anchor_images)
    positive_dataset = tf.data.Dataset.from_tensor_slices(positive_images)

    negative_dataset = tf.data.Dataset.from_tensor_slices(negative_images)
    negative_dataset = negative_dataset.shuffle(buffer_size=4096)

    dataset = tf.data.Dataset.zip((anchor_dataset, positive_dataset, negative_dataset))
    dataset = dataset.shuffle(buffer_size=1024)
    dataset = dataset.map(preprocess_triplets)

    # Let's now split our dataset in train and validation.
    train_dataset = dataset.take(round(image_count * 0.8))
    val_dataset = dataset.skip(round(image_count * 0.8))

    train_dataset = train_dataset.batch(32, drop_remainder=False)
    train_dataset = train_dataset.prefetch(8)

    val_dataset = val_dataset.batch(32, drop_remainder=False)
    val_dataset = val_dataset.prefetch(8)

    return  train_dataset, val_dataset

 
