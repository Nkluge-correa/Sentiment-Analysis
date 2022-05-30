# ----------------------------------------------------------------------------------------#
# Bibliotecas
# ----------------------------------------------------------------------------------------#
import os
import warnings
warnings.filterwarnings('ignore')
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  
import json 
import dash
import time
import string
import unidecode
import pandas as pd
import numpy as np
import tensorflow as tf
import dash_bootstrap_components as dbc
from tensorflow import keras
from keras import regularizers 
from dash import dcc, html, Output, Input, State
from dash.dependencies import Input, Output, State
from keras.preprocessing.sequence import pad_sequences
from keras.preprocessing.text import Tokenizer, tokenizer_from_json


# ----------------------------------------------------------------------------------------#
# Load Model & Tokenizer
# ----------------------------------------------------------------------------------------#
imdb = keras.datasets.imdb
word_index = imdb.get_word_index()
model = keras.models.load_model('senti_model_en')

word_index = {k:(v+3) for k,v in word_index.items()}
word_index["<PAD>"] = 0
word_index["<START>"] = 1
word_index["<UNK>"] = 2  
word_index["<UNUSED>"] = 3 

def encode_review(text):
    texto=[]
    new_text = text.split()
    texto.append(1)
    for word in new_text:
        value = word_index.get(word)
        if value is None:
            texto.append(2)
        else:
            texto.append(value)
    sequence = [np.array(texto)]
    return sequence

# ----------------------------------------------------------------------------------------#
# Dash Back-end
# ----------------------------------------------------------------------------------------#

def textbox(text, box='other'):
    style = {
        'max-width': '55%',
        'width': 'max-content',
        'padding': '10px 15px',
        'border-radius': '25px',
    }

    if box == 'self':
        style['margin-left'] = 'auto'
        style['margin-right'] = 0

        color = 'primary'
        inverse = True

    elif box == 'other':
        style['margin-left'] = 0
        style['margin-right'] = 'auto'

        color = 'light'
        inverse = False

    else:
        raise ValueError('Incorrect option for `box`.')

    return dbc.Card(text, style=style, body=True, color=color, inverse=inverse)

conversation = html.Div(
    style={
        'width': '80%',
        'max-width': '600px',
        'height': '35vh',
        'margin': 'auto',
        'margin-top': '100px',
        'overflow-y': 'auto',
    },
    id='display-prediction',
)


controls = dbc.InputGroup(
    style={'width': '80%', 'max-width': '600px', 'margin': 'auto'},
    children=[
        dbc.Input(id='user-input', placeholder='Write a comment...', type='text'),
        dbc.InputGroup(dbc.Button('Submit', size= 'lg', id='submit')),
    ],
)

# ----------------------------------------------------------------------------------------#
# Dash app
# ----------------------------------------------------------------------------------------#

app = dash.Dash(__name__, external_stylesheets=[dbc.themes.BOOTSTRAP])

server = app.server
app.title = 'Sentiment Classifier'

# ----------------------------------------------------------------------------------------#
# Layout
# ----------------------------------------------------------------------------------------#

app.layout = dbc.Container(
    fluid=True,
    children=[
        html.H1('Sentiment Classifier  ', style={'color':'#242020',
                                                    'font-style': 'bold', 
                                                    'margin-top': '15px',
                                                    'margin-left': '15px',
                                                    'display':'inline-block'}),
        html.H1('  🤖', style={'color':'#2a9fd6',
                            'font-style': 'bold', 
                            'margin-top': '15px',
                            'margin-left': '15px',
                            'display':'inline-block'}),
        html.Hr(),
        dbc.Row([
        dbc.Col([
            dcc.Store(id='store-data', data=''),
            dcc.Loading(id='loading_0', type='circle', children=[conversation]),
            controls,
        ], md = 12),
        ]),
    ],
)

# ----------------------------------------------------------------------------------------#
# Funções
# ----------------------------------------------------------------------------------------#

@app.callback(
    Output('display-prediction', 'children'), 
    [
        Input('user-input', 'n_submit'), 
        Input('store-data', 'data')]
)
def update_display(user_input, sentiment_analysis):
    time.sleep(2)
    if user_input is None or user_input == '':
        return textbox(sentiment_analysis, box='other')
    else:
        return [
            textbox(sentiment_analysis, box='self') if i % 2 == 0 else textbox(sentiment_analysis, box='other')
            for i, sentiment_analysis in enumerate(sentiment_analysis)
        ]


@app.callback(
    [
     Output('store-data', 'data'),
     Output('user-input', 'value')
    ],

    [
     Input('submit', 'n_clicks'), 
     Input('user-input', 'n_submit')
    ],

    [
     State('user-input', 'value'), 
     State('store-data', 'data')
    ]
)

def run_senti_model(n_clicks, n_submit, user_input, sentiment_analysis):
    if n_clicks == 0:
        sentiment_analysis = []
        sentiment_analysis.append('How you feel?')
        return sentiment_analysis, ''


    if user_input is None or user_input == '':
        sentiment_analysis = []
        sentiment_analysis.append('How you feel?')
        return sentiment_analysis, ''


    texto = user_input
    texto = texto.translate(str.maketrans('', '', string.punctuation))
    texto = texto.lower()
    texto = unidecode.unidecode(texto)
    sequence = encode_review(texto)
    pad_text = keras.preprocessing.sequence.pad_sequences(sequence,
                                                            value=word_index["<PAD>"],
                                                            padding='post',
                                                            maxlen=256)

    prediction = model.predict(pad_text)

    if prediction[0][0] <= 0.5:
        pred = 1 - prediction[0][0]
        pred = '{:,.2f}'.format(pred * 100) + ' %'
        response = f'Negative Sentiment 😔 \n {pred}'

    elif prediction[0][0] >= 0.5:
        pred = '{:,.2f}'.format(prediction[0][0] * 100) + ' %'
        response = f'Positive Sentiment 😊 \n {pred}'
   
    sentiment_analysis = []
    sentiment_analysis.append(user_input)
    sentiment_analysis.append(response)

    return sentiment_analysis, ''

if __name__ == '__main__':
    app.run_server(debug=False)
