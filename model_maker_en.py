# ----------------------------------------------------------------------------------------#
# Bibliotecas
# ----------------------------------------------------------------------------------------#
import os
import warnings
warnings.filterwarnings('ignore')
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 
import io
import json 
import string
import unidecode
import pandas as pd
import numpy as np
import tensorflow as tf
import plotly.graph_objects as go
from IPython.core.display import display, HTML
from tensorflow import keras
from keras.preprocessing.sequence import pad_sequences
from keras.layers import Embedding, Flatten, Dense, LSTM
from keras import regularizers 

# ----------------------------------------------------------------------------------------#
# Load Data/Split & Slice
# ----------------------------------------------------------------------------------------#
imdb = keras.datasets.imdb
(train_data, train_labels), (test_data, test_labels) = imdb.load_data(num_words=10000)
print('Amostras: {}, Alvos: {}'.format(len(train_data), len(train_labels)))

# ----------------------------------------------------------------------------------------#
# Keras Tokenizer
# ----------------------------------------------------------------------------------------#

word_index = imdb.get_word_index()

word_index = {k:(v+3) for k,v in word_index.items()}
word_index["<PAD>"] = 0
word_index["<START>"] = 1
word_index["<UNK>"] = 2  
word_index["<UNUSED>"] = 3 
print('Encontrados %s tokens únicos.' % len(word_index))

train_data = keras.preprocessing.sequence.pad_sequences(train_data,
                                                        value=word_index["<PAD>"],
                                                        padding='post',
                                                        maxlen=256)

test_data = keras.preprocessing.sequence.pad_sequences(test_data,
                                                       value=word_index["<PAD>"],
                                                       padding='post',
                                                       maxlen=256)

# ----------------------------------------------------------------------------------------#
# Keras Model
# ----------------------------------------------------------------------------------------#

vocab_size = 10000

model = keras.Sequential()
model.add(keras.layers.Embedding(vocab_size, 32))
model.add(keras.layers.GlobalAveragePooling1D())
model.add(keras.layers.Dense(32, activation='relu'))
model.add(keras.layers.Dropout(0.2))
model.add(keras.layers.Dense(1, activation='sigmoid'))

opt = tf.keras.optimizers.Adam(learning_rate=0.01)
model.compile(optimizer=opt,
              loss='MeanSquaredError', #MeanSquaredError try! binary_crossentropy
              metrics=['accuracy'])

x_val = train_data[:10000]
partial_x_train = train_data[10000:]

y_val = train_labels[:10000]
partial_y_train = train_labels[10000:]

history = model.fit(partial_x_train,
                    partial_y_train,
                    epochs=40,
                    batch_size=512,
                    validation_data=(x_val, y_val),
                    verbose=1)
results = model.evaluate(test_data,  test_labels, verbose=1)
print(results)
model.save('senti_model_en')

# ----------------------------------------------------------------------------------------#
# Keras Model Logs
# ----------------------------------------------------------------------------------------#

acc = history.history['accuracy']
val_acc = history.history['val_accuracy']
loss = history.history['loss']
val_loss = history.history['val_loss']
epochs = range(1, len(acc) + 1)

fig = go.Figure(layout={'template':'plotly_dark'})

fig.add_trace(go.Scatter(x=list(epochs), y=acc,
                         line_color='rgba(0, 102, 255, 0.5)', line=dict(width=3, dash='dash'), name='Acurácia (Treinamento)', mode='lines',
                         hoverlabel=dict(namelength=-1),
                         hovertemplate='Acurácia (Treinamento): %{y:.5f} acc <extra></extra>',
                         showlegend=True))
fig.add_trace(go.Scatter(x=list(epochs), y=val_acc,
                         line_color='rgba(255, 0, 0, 0.5)', line=dict(width=3, dash='dash'), name='Acurácia (Validação)', mode='lines',
                         hoverlabel=dict(namelength=-1),
                         hovertemplate='Acurácia (Validação): %{y:.2f} acc <extra></extra>',
                         showlegend=True))

fig.update_xaxes(showgrid=False, showline=False, mirror=False)
fig.update_yaxes(showgrid=True, ticksuffix=' acc')
fig.update_layout(
    paper_bgcolor='#242424',
    plot_bgcolor='#242424',
    hovermode='x unified',
    font_family='Open Sans',
    autosize=True,
    margin=dict(l=10, r=10, b=10, t=10),
    hoverlabel=dict(bgcolor='#242424', font_size=18, font_family='Open Sans')
)

fig.show()

fig2 = go.Figure(layout={'template':'plotly_dark'})

fig2.add_trace(go.Scatter(x=list(epochs), y=loss,
                         line_color='rgba(0, 102, 255, 0.5)', line=dict(width=3, dash='dash'), name='Loss (Treinamento)', mode='lines',
                         hoverlabel=dict(namelength=-1),
                         hovertemplate='Loss (Treinamento): %{y:.5f} loss <extra></extra>',
                         showlegend=True))
fig2.add_trace(go.Scatter(x=list(epochs), y=val_loss,
                         line_color='rgba(255, 0, 0, 0.5)', line=dict(width=3, dash='dash'), name='Loss (Validação)', mode='lines',
                         hoverlabel=dict(namelength=-1),
                         hovertemplate='Loss (Validação): %{y:.2f} loss <extra></extra>',
                         showlegend=True))

fig2.update_xaxes(showgrid=False, showline=False, mirror=False)
fig2.update_yaxes(showgrid=True, ticksuffix=' loss')
fig2.update_layout(
    paper_bgcolor='#242424',
    plot_bgcolor='#242424',
    hovermode='x unified',
    font_family='Open Sans',
    autosize=True,
    margin=dict(l=10, r=10, b=10, t=10),
    hoverlabel=dict(bgcolor='#242424', font_size=18, font_family='Open Sans')
)

fig2.show()







