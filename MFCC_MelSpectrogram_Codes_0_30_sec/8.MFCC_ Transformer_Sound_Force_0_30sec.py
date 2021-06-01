'''
Purpose: Implementing TRANSFORMER Deep learning architecture for hybrid Audio and Force Data
Author: Gi-Jun Park and Binayak Bhandari modified from the original code at : https://keras.io/examples/nlp/text_classification_with_transformer/
Date: 12 March 2021
Woosong University
'''
import os
import json
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from sklearn.model_selection import train_test_split
from keras_radam.training import RAdamOptimizer
from sklearn.preprocessing import OneHotEncoder
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow_addons.optimizers import AdamW
import matplotlib.pyplot as plt
from keras.utils import plot_model
from keras.layers import Flatten
import pandas as pd
from keras.callbacks import CSVLogger  # Save history of each epochs
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay


# ============ CONSTANTS using in the program =====================
save_dir = os.path.join(os.getcwd(), 'Confusion_Matrix_AdamW_0-30sec_MFCC')
if not os.path.exists(save_dir):
    os.makedirs(save_dir)

model_name = '0_30_MFCC_RAdam'
maxlen = 130  # (orig = 323)

# ===================== AUDIO DATA====================
DATASET_PATH = "Audio_Preprocessing_CLass_MFCCs_Labels_Transformer.json"


# =============OTHER CONSTANTS =====================
random_states = [154] #, 306, 743, 877, 558
test_accuracies = []
history_loss = []
history_accuracy = []
history_val_loss = []
history_val_accuracy = []

# ================= FORCE DATA ======================
FORCE_DATA = 'Force_Data.npy'  # Shape is (1600, 130, 3)
FORCE_LABEL = 'Force_Label.npy'  # Shape is (1600, 4)

"""
## Download and prepare dataset
"""


def load_data(dataset_path):
    if dataset_path.endswith('.json'):
        with open(dataset_path, "r") as fp:
            data = json.load(fp)

        # Convert lists/dictionary into numpy array
        Sound_input = np.array(data["mfcc"])
        Sound_label = np.array(data["labels"])

        # binary encode
        onehot_encoder = OneHotEncoder(sparse=False)
        Sound_label = Sound_label.reshape(len(Sound_label), 1)
        Sound_label = onehot_encoder.fit_transform(Sound_label)
        print(Sound_label)
        print("Data successfully loaded")
        print(f'The shape of sound input is {Sound_input.shape} and shape of sound level is {Sound_label}')
        # The shape of sound input is (1600, 130, 13) and shape of sound level is [[1. 0. 0. 0.]
        return Sound_input, Sound_label
    elif dataset_path.endswith('.npy'):
        data = np.load(dataset_path)
        return data


def prepare_datasets(Sound_data, Force_data, Sound_label, Force_label, test_size, validation_size):
    # create train, validation and test split
    Sound_train, Sound_test, Force_train, Force_test, Sound_label_train, Sound_label_test, \
     Force_label_train, Force_label_test = train_test_split(Sound_data,
                                                            Force_data,
                                                            Sound_label,
                                                            Force_label,
                                                            test_size=test_size)
    Sound_test, Sound_validation, Force_test, Force_validation, \
    Sound_label_test, Sound_label_validation, Force_label_test, Force_label_validation = train_test_split(Sound_test,
                                                                                                          Force_test,
                                                                                                          Sound_label_test,
                                                                                                          Force_label_test,
                                                                                                          test_size=validation_size)

    return Sound_train, Sound_validation, Sound_test, Force_train, Force_validation, Force_test, \
           Sound_label_train, Sound_label_validation, Sound_label_test, Force_label_train, Force_label_validation, Force_label_test


"""
## Implement multi head self attention as a Keras layer
"""


class MultiHeadSelfAttention(layers.Layer):
    def __init__(self, embed_dim, num_heads=8):
        super(MultiHeadSelfAttention, self).__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        if embed_dim % num_heads != 0:
            raise ValueError(f"embedding dimension = {embed_dim} should be divisible by number of heads = {num_heads}")
        self.projection_dim = embed_dim // num_heads
        self.query_dense = layers.Dense(embed_dim)
        self.key_dense = layers.Dense(embed_dim)
        self.value_dense = layers.Dense(embed_dim)
        self.combine_heads = layers.Dense(embed_dim)

    def attention(self, query, key, value):
        score = tf.matmul(query, key, transpose_b=True)
        dim_key = tf.cast(tf.shape(key)[-1], tf.float32)
        scaled_score = score / tf.math.sqrt(dim_key)
        weights = tf.nn.softmax(scaled_score, axis=-1)
        output = tf.matmul(weights, value)
        return output, weights

    def separate_heads(self, x, batch_size):
        x = tf.reshape(x, (batch_size, -1, self.num_heads, self.projection_dim))
        return tf.transpose(x, perm=[0, 2, 1, 3])

    def call(self, inputs):
        # x.shape = [batch_size, seq_len, embedding_dim]
        batch_size = tf.shape(inputs)[0]
        query = self.query_dense(inputs)    # (batch_size, seq_len, embed_dim)
        key = self.key_dense(inputs)        # (batch_size, seq_len, embed_dim)
        value = self.value_dense(inputs)    # (batch_size, seq_len, embed_dim)
        query = self.separate_heads(
            query, batch_size)              # (batch_size, num_heads, seq_len, projection_dim)
        key = self.separate_heads(
            key, batch_size)                # (batch_size, num_heads, seq_len, projection_dim)
        value = self.separate_heads(
        value, batch_size)                  # (batch_size, num_heads, seq_len, projection_dim)
        attention, weights = self.attention(query, key, value)
        attention = tf.transpose(
            attention, perm=[0, 2, 1, 3])   # (batch_size, seq_len, num_heads, projection_dim)
        concat_attention = tf.reshape(
            attention, (batch_size, -1, self.embed_dim))  # (batch_size, seq_len, embed_dim)
        output = self.combine_heads(
            concat_attention)               # (batch_size, seq_len, embed_dim)
        return output

"""
## Implement a Transformer block as a layer
"""


class TransformerBlock(layers.Layer):
    def __init__(self, embed_dim, num_heads, ff_dim, rate=0.1):
        super(TransformerBlock, self).__init__()
        self.att = MultiHeadSelfAttention(embed_dim, num_heads)
        self.ffn = keras.Sequential(
            [layers.Dense(ff_dim, activation='relu'), layers.Dense(embed_dim), ]
        )
        self.layernorm1 = layers.LayerNormalization(epsilon=1e-6)
        self.layernorm2 = layers.LayerNormalization(epsilon=1e-6)
        self.dropout1 = layers.Dropout(rate)
        self.dropout2 = layers.Dropout(rate)

    def call(self, inputs, training):
        attn_output = self.att(inputs)
        attn_output = self.dropout1(attn_output, training=training)
        out1 = self.layernorm1(inputs + attn_output)
        ffn_output = self.ffn(out1)
        ffn_output = self.dropout2(ffn_output, training=training)
        return self.layernorm2(out1 + ffn_output)

"""
## Implement embedding layer
Two seperate embedding layers, one for tokens, one for token index (positions).
"""


class PositionEmbedding(layers.Layer):
    def __init__(self, maxlen, embed_dim):
        super(PositionEmbedding, self).__init__()
        self.pos_emb = layers.Embedding(input_dim=maxlen, output_dim=embed_dim)

    def call(self, x):
        positions = tf.range(start=0, limit=maxlen, delta=1)
        positions = self.pos_emb(positions)
        return positions


def model_configure(sound_train, force_train):

    # ================ CONSTANTS ==================
    # The shape of sound input is (1600, 130, 13) so the embed_dim_st = 13
    # The shape of force input is shape=(None, 130, 13)
    embed_dim_ST    = 15  # Embedding size for each token (n_fft/2)+1 =(2048/2)+1 =  1025 (orig = 2049)
    embed_dim_FT    = 3     # (orig = 3)
    num_heads       = 3     # Number of attention heads (orig = 3)
    ff_dim          = 512   # Hidden layer size in feed forward network inside transformer
    unit            = 512   # Number of neuron

    # Embedding dimension = 3 should be divisible by number of heads = 5
    # Transformer embedding dimension = 1025 should be divisible by number of heads = 3
    # Dimensions must be equal 323, but are 130
    # Dimensions must be equal, but are 1025 and 13 for ... with input shapes: [130,1025], [?,130,13]
    # ValueError: Dimensions must be equal, but are 13 and 4 for ...with input shapes: [130,13], [?,4]
    #

    embedding_layer_ST = PositionEmbedding(maxlen, embed_dim_ST) # maxlen = 130
    transformer_block_ST = TransformerBlock(embed_dim_ST, num_heads, ff_dim)

    embedding_layer_FT = PositionEmbedding(maxlen, embed_dim_FT)
    transformer_block_FT = TransformerBlock(embed_dim_FT, num_heads, ff_dim)

    inputs_ST = layers.Input(shape=(sound_train.shape[1:])) # First layer is shape
    print(f'Shape of inputs_Sound is {inputs_ST}') #  shape=(None, 130, 13), dtype=float32)
    masked_ST = layers.Masking(mask_value=0.)(inputs_ST)
    print(f'This is masked_ST : {masked_ST}')
    ST = embedding_layer_ST(masked_ST)
    print(f'This is ST {ST}')
    ST = transformer_block_ST(ST + masked_ST)
    ST = layers.GlobalAveragePooling1D()(ST)
    ST = layers.Dropout(0.1)(ST)
    ST = Flatten()(ST)
    # ST = layers.Dense(unit, activation='relu')(ST)
    # ST = layers.Dropout(0.3)(ST)

    inputs_FT = layers.Input(shape=(force_train.shape[1:]))  # First layer is shape
    print(f'Shape of inputs_Force is {inputs_FT}')  # shape=(None, 4), dtype=float32)
    masked_FT = layers.Masking(mask_value=0.)(inputs_FT)
    print(f'This is masked_Ft {masked_FT}')
    FT = embedding_layer_FT(masked_FT)
    print(f'This is FT {FT}')
    FT = transformer_block_FT(FT + masked_FT)
    FT = layers.GlobalAveragePooling1D()(FT)
    FT = layers.Dropout(0.1)(FT)
    FT = Flatten()(FT)
    # FT = layers.Dense(unit, activation='relu')(FT)
    # FT = layers.Dropout(0.3)(FT)

    y = layers.concatenate([ST, FT])
    print(f'y is {y}')

    y = layers.Dense(unit, activation='relu')(y)
    y = layers.Dropout(0.3)(y)
    outputs = layers.Dense(4, activation="softmax")(y)

    model = keras.Model(inputs=[inputs_ST, inputs_FT], outputs=outputs)

    model.compile(loss='categorical_crossentropy', optimizer=RAdamOptimizer(learning_rate=1e-2), metrics=['accuracy'])
    #model.compile(loss='categorical_crossentropy', optimizer=AdamW(weight_decay=0.001), metrics=['accuracy'])
    plot_model(model, to_file='Transformer_Flatten.png')
    return model


if __name__ == "__main__":
    for idx in range(len(random_states)):
        # Load Audio data
        Sound_inputs, Sound_label = load_data(DATASET_PATH)
        # Load Force data
        Force_input = load_data(FORCE_DATA)
        print(f'This is the shape of Force_input {Force_input.shape}') # ouptu (800,4) but should be (800, 130, 3)
        Force_label = load_data(FORCE_LABEL)

        # Spliting the data into train (50%) and test set (50%)
        Sound_train, Sound_validation, Sound_test, Force_train, Force_validation, Force_test, \
        Sound_label_train, Sound_label_validation, Sound_label_test, Force_label_train, Force_label_validation, Force_label_test = prepare_datasets(
                                                                            Sound_inputs,
                                                                            Force_input,
                                                                            Sound_label,
                                                                            Force_label,
                                                                            test_size=0.5,
                                                                            validation_size=0.5)
        print(f'This is the shape of Force_train shape after split {Force_train.shape}')

        # ===================== MODEL execution ======================================
        '''for idx in range(len(random_stats)):
            sound_train, sound_test, force_train, force_test, str_label, ste_label, ftr_label, fte_label = \
                train_test_split(np.load('dataset/C_Train_Sound.npy'),
                                 np.load('dataset/C_Train_Force_Ensemble.npy'),
                                 np.load('dataset/C_Label_Sound.npy'),
                                 np.load('dataset/C_Label_Force_Ensemble.npy'),
                                 test_size=0.5, shuffle=True, random_state=random_stats[idx])
    
            sound_test, sound_validation, force_test, force_validation, ste_label, stv_label, fte_label, ftv_label = \
                train_test_split(sound_test, force_test, ste_label, fte_label,
                                 test_size=0.5, shuffle=True, random_state=random_stats[idx])'''

        print(f'Sound Training sequences is {len(Sound_train)}')
        print(f'Sound Validation sequences is {len(Sound_validation)}')

        print(f'Force Training sequences is {len(Force_train)}')
        print(f'Force Validation sequences is {len(Force_validation)}')
        print(f'Shape of Force_train is {Force_train.shape}')

        """
        ## Train and Evaluate
        """
        model = model_configure(Sound_train, Force_train)
        model.summary()
        earlystop = EarlyStopping(monitor='accuracy', min_delta=0, patience=20, verbose=0, mode='auto')
        csv_logger = CSVLogger(os.path.join(save_dir, "model_history_log_" + str(idx) + ".csv"), append=True)
        history = model.fit([Sound_train, Force_train], Force_label_train,
                            batch_size=32,
                            epochs=500,
                            validation_data=([Sound_validation, Force_validation], Force_label_validation),
                            callbacks=[earlystop, csv_logger])

        # ================== Save model and weights ====================
        if not os.path.isdir(save_dir):
            os.makedirs(save_dir)
        model_path = os.path.join(save_dir, f'{model_name}_{idx + 1}.h5')
        model.save_weights(model_path)
        print('Saved trained model at %s ' % model_path)

        # ================ Evaluate on the model (Inference) ==============================
        score, acc = model.evaluate([Sound_test, Force_test], Force_label_test, batch_size=8) #Sound_label_test
        print('Test loss:', score)
        print('Test accuracy:', acc)

        # =============== Save the history of each model as csv ============================
        hist_df = pd.DataFrame(history.history)
        hist_csv_file = os.path.join(save_dir, f'{model_name}_history_{idx + 1}.csv')
        with open(hist_csv_file, mode='w') as f:
            hist_df.to_csv(f)

# exit()

# ======================== PLOTTING ==================================
plt.plot(history.history['accuracy'])
plt.plot(history.history['val_accuracy'])
plt.title('Model accuracy')
plt.ylabel('Accuracy')
plt.xlabel('Epoch')
plt.legend(['Train', 'Validation'], loc='best')
plt.show()

# ================================ CONFUSION MATRIX============================
num_classes = 4
class_names = ['Fine', 'Smooth', 'Rough', 'Coarse']
result = model.predict([Sound_test, Force_test])
confusion_mtx = confusion_matrix(Force_label_test.argmax(axis=-1), result.argmax(axis=-1))
disp = ConfusionMatrixDisplay(confusion_matrix=confusion_mtx, display_labels=class_names)
disp.plot()
plt.show()

print("Confusion matrix")
print(disp.confusion_matrix)