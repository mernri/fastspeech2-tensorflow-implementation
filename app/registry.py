import os
import glob
import numpy as np
import tensorflow as tf
import time
from sklearn.model_selection import train_test_split
from tensorflow.keras.callbacks import ModelCheckpoint
from app.model import *
from app.params import *
import matplotlib.pyplot as plt


def get_latest_model_path():
    local_model_paths = glob.glob(f"{PATH_FULL_MODEL}/*")
    if not local_model_paths:
        return None
    path_to_load_from = sorted(local_model_paths, key=os.path.getmtime)[-1]
    return path_to_load_from
    
def load_data_from_directory(path, suffix, data_fraction):
    data_dict = {}

    sorted_files = sorted(os.listdir(path))
    num_files_to_load = int(len(sorted_files) * data_fraction)

    for file_name in sorted_files[:num_files_to_load]:
        if file_name.endswith(".npy") and suffix in file_name:
            sequence_id = file_name.split(f"_{suffix}")[0]
            file_path = os.path.join(path, file_name)
            data_dict[sequence_id] = np.load(file_path, allow_pickle=True)

    return data_dict

def create_tensorflow_dataset(tokens, melspecs, phone_durations, batch_size):
    features_dataset = tf.data.Dataset.from_tensor_slices({
        "tokens_input": tokens,
        "phone_durations_input": phone_durations
    })
    output_dataset = tf.data.Dataset.from_tensor_slices(melspecs)
    dataset = tf.data.Dataset.zip((features_dataset, output_dataset))
    dataset = dataset.shuffle(buffer_size=len(tokens))
    dataset = dataset.batch(batch_size)
    dataset = dataset.prefetch(tf.data.AUTOTUNE)
    return dataset

def get_test_train_val_split(data_fraction=1.0):
    # Charger les données
    tokens_data_dict = load_data_from_directory(PATH_PADDED_TOKENS, "tokens", 1.0)  # 1.0 = Charger 100% des données
    phone_durations_dict = load_data_from_directory(PATH_PADDED_DURATIONS, 'phone_durations', 1.0)
    melspec_data_dict = load_data_from_directory(PATH_PADDED_MELSPECS, "melspecs", 1.0)
    
    # Trouver les sequence_ids communs
    common_sequence_ids = set(tokens_data_dict.keys()) & set(phone_durations_dict.keys()) & set(melspec_data_dict.keys())
    
    # Mélanger et sélectionner une fraction des sequence_ids
    common_sequence_ids = list(common_sequence_ids)
    np.random.shuffle(common_sequence_ids)
    num_samples = int(len(common_sequence_ids) * data_fraction)
    selected_sequence_ids = common_sequence_ids[:num_samples]
    
    # Récupérer les données correspondantes
    tokens_data = [tokens_data_dict[seq_id] for seq_id in selected_sequence_ids]
    melspec_data = [melspec_data_dict[seq_id] for seq_id in selected_sequence_ids]
    phone_durations_data = [phone_durations_dict[seq_id] for seq_id in selected_sequence_ids]

    # Séparer les données en tain val test
    tokens_train, tokens_temp, melspec_train, melspec_temp, phone_durations_train, phone_durations_temp = train_test_split(tokens_data, melspec_data, phone_durations_data, test_size=0.2, random_state=42)
    tokens_val, tokens_test, melspec_val, melspec_test, phone_durations_val, phone_durations_test = train_test_split(tokens_temp, melspec_temp, phone_durations_temp, test_size=0.5, random_state=42)
    
    # Convertir en tensors 
    tokens_train = tf.convert_to_tensor(tokens_train, dtype=tf.int32)
    melspec_train = tf.convert_to_tensor(melspec_train, dtype=tf.float32)
    phone_durations_train = tf.convert_to_tensor(phone_durations_train, dtype=tf.float32)
    
    tokens_val = tf.convert_to_tensor(tokens_val, dtype=tf.int32)
    melspec_val = tf.convert_to_tensor(melspec_val, dtype=tf.float32)
    phone_durations_val = tf.convert_to_tensor(phone_durations_val, dtype=tf.float32)
    
    tokens_test = tf.convert_to_tensor(tokens_test, dtype=tf.int32)
    melspec_test = tf.convert_to_tensor(melspec_test, dtype=tf.float32)
    phone_durations_test = tf.convert_to_tensor(phone_durations_test, dtype=tf.float32)

    # Créer les tensorflow datasets 
    train_dataset = create_tensorflow_dataset(tokens_train, melspec_train, phone_durations_train, BATCH_SIZE)
    val_dataset = create_tensorflow_dataset(tokens_val, melspec_val, phone_durations_val, BATCH_SIZE)
    test_dataset = create_tensorflow_dataset(tokens_test, melspec_test, phone_durations_test, BATCH_SIZE)
    
    return train_dataset, val_dataset, test_dataset


def initialize_model(config):
    model = Transformer(
        num_layers=config.num_layers, 
        embedding_dim=config.embedding_dim, 
        num_heads=config.num_heads, 
        dff=config.dff, 
        input_vocab_size=config.input_vocab_size,
        conv_kernel_size=config.conv_kernel_size, 
        conv_filters=config.conv_filters, 
        rate=config.rate,
        var_conv_filters= config.var_conv_filters, 
        var_conv_kernel_size= config.var_conv_kernel_size, 
        var_rate= config.var_rate
    )
    return model


def compile_model(model, config):
    custom_loss = CustomMelspecLoss(beta=2)
    lr_schedule = CustomLearningRateScheduler(embedding_dim=config.embedding_dim, warmup_steps=config.warmup_steps)
    optimizer = tf.keras.optimizers.Adam(lr_schedule, beta_1=config.beta_1, beta_2=config.beta_2, epsilon=config.epsilon)
    
    model.compile(optimizer=optimizer, 
                  loss=custom_loss)


def train_model(model, train_dataset, val_dataset, epochs):
    batches_per_epoch = len(train_dataset)
    checkpoint_path = f"{PATH_MODEL_CHECKPOINTS}/model_at_epoch_{{epoch:02d}}.ckpt"

    checkpoint_callback = ModelCheckpoint(
        filepath=checkpoint_path,
        monitor="val_loss",  
        save_best_only=True,   
        save_weights_only=True,
        save_freq=15 * batches_per_epoch, 
        verbose=1
    )
    
    history = model.fit(train_dataset, 
                        validation_data=(val_dataset),
                        epochs=epochs,
                        callbacks=[checkpoint_callback],
                        verbose=1)
    
    plt.plot(history.history['loss'], label='Train Loss')
    plt.plot(history.history['val_loss'], label='Validation Loss')
    plt.title('Model Loss Over Time')
    plt.ylabel('Loss')
    plt.xlabel('Epoch')
    plt.legend(loc='upper right')
    plt.show()
    
    return history

def save_model_in_saved_models(model):
    model_path = f"{PATH_FULL_MODEL}/model_{int(time.time())}"
    
    tf.keras.models.save_model(model, model_path)

    print(f"Model saved at {model_path}")

def load_model_fom_saved_models(model_path):
    model = tf.keras.models.load_model(model_path)
    return model


def load_latest_checkpoint_from_dir():
    config= Config()
    checkpoints = [f for f in os.listdir(PATH_MODEL_CHECKPOINTS) if f.endswith(".ckpt.index")]
    
    if not checkpoints:
        print("aucun checkpoint trouvé")
        return None
    
    checkpoints.sort()
    latest_checkpoint_name = checkpoints[-1].replace(".index", "")
    latest_checkpoint_path = os.path.join(PATH_MODEL_CHECKPOINTS, latest_checkpoint_name)
    
    model = initialize_model(config)
    compile_model(model, config)
    
    model.load_weights(latest_checkpoint_path)
    
    input_shape = (config.embedding_dim,)
    model.build(input_shape)
    
    return model

def predict_melspec(model, input):
    phonems, durations = input
    phonems = tf.convert_to_tensor(phonems, dtype=tf.float32)
    durations = tf.convert_to_tensor(durations, dtype=tf.int32)
    
    phonems = tf.expand_dims(phonems, 0)
    durations = tf.expand_dims(durations, 0)
    
    model_input = {'tokens_input': phonems, 'phone_durations_input': durations}
    
    prediction = model.predict(model_input)
    return prediction


def evaluate_model(model, test_dataset):
    loss = model.evaluate(test_dataset)
    return loss