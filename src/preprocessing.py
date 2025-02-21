# Packages used in the system
# Pacotes utilizados no sistema
import pandas as pd
import numpy as np
import re
import string
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
import tensorflow as tf
from tensorflow.data import Dataset, AUTOTUNE
from tensorflow.keras.layers import TextVectorization
from tensorflow.keras.utils import pad_sequences
from sklearn.model_selection import StratifiedShuffleSplit
from transformers import DistilBertTokenizer
import pickle

import os


def rnn_preprocess(sentence):
    """
    [EN-US]
    Preprocesses the text passed as an argument.
    
    [PT-BR]
    Pré-processa o texto passado como argumento.

    Argument:
        sentence (str): text that will be preprocessed
                        (texto que será pré-processado).
    """
    stemmer = PorterStemmer()
    stopwords_en = stopwords.words('english')
    punct = string.punctuation

    # Transforming the corpus to lowercase
    # Transformando o corpus para lowercase
    sentence = sentence.lower()
    # Removing everything that is not a digit or letter
    # Retirando tudo o que não for dígito ou letra
    sentence = re.sub(r'\W+', ' ', sentence)
    clean_sentence = []

    # Going through each word in the corpus
    # Percorrendo cada palavra do corpus
    for word in sentence.split():
        # If the word is not a stopword
        # Se a palavra não for uma stopword
        if word not in stopwords_en:
            # Apply stemming to the word
            # Aplica o stemming na palavra
            stem_word = stemmer.stem(word)
            # Add the word to the final list
            # Adiciona a palavra na lista final
            clean_sentence.append(stem_word)

    # Joining each word from the final list
    # Juntando cada palavra da lista final
    clean_sentence = ' '.join(clean_sentence)
    # Removing punctuations
    # Retirando pontuações
    clean_sentence = re.sub(f'r[{re.escape(punct)}]', ' ', clean_sentence)
    # Removing whitespace
    # Retirando espaços em branco
    clean_sentence = re.sub(r'\s+', ' ', clean_sentence).strip()
    
    return clean_sentence


def rnn_tokenizer(corpus, max_tokens=None, max_len=None):
    """
    [EN-US]
    Sets the tokenizer, hyperparameters and trains the tokenizer on the received corpus
    
    [PT-BR]
    Define o tokenizador, os hiperparâmetros e treinada o tokenizador no corpus recebido

    Arguments:
        corpus (numpy.array or list or pandas.Series): corpus on which the tokenizer will be trained
                                                       (corpus em que o tokenizer será treinado).
        max_tokens (int, optional): maximum vocabulary size. Defaults to None
                                    (tamanho máximo do vocabulário. Padrão para None).
        max_len (int, optional): maximum size of tokenized output. Defaults to None
                                 (tamanho máximo do output tokenizado. Padrão para None).
    """
    # Setting the tokenizer with the specified max_tokens and max_len
    # Definindo o tokenizer com o max_tokens e max_len especificados
    vectorize_layer = tf.keras.layers.TextVectorization(
        max_tokens=max_tokens,
        output_sequence_length=max_len
    )
    # Training the tokenizer on the corpus
    # Treinando o tokenizer no corpus
    vectorize_layer.adapt(corpus)

    return vectorize_layer


if __name__ == '__main__':
    # Setting the global variable `PATH` with the path of the directory where the data will be loaded
    # Configurando a variável global `PATH` com o caminho do diretório onde os dados serão carregados
    PATH = os.path.abspath(
        os.path.join(
            os.path.dirname(__file__),
            os.pardir,
            'data'
        )
    )
    
    # Reading the dataset that will be pre-processed
    # Lendo o dataset que será pré-processado
    comics_data = pd.read_csv(os.path.join(PATH, 'raw', 'comics_corpus.csv'))

    # Preprocessing the data
    # Pré-processando os dados
    comics_data['description'] = comics_data['description'].map(rnn_preprocess)
    # Removing duplicate examples after preprocessing
    # Removendo os exemplos duplicados após o pré-processamento
    comics_data = comics_data.drop_duplicates('description')

    # Creating a dataset with only the features that will be used in the mode
    # Criando um dataset com apenas as features que serão usadas no modelo
    comics_corpus = comics_data[['description', 'y']].copy()
    # Transforming the target label y into binary
    # Transformando o target label y em binário
    comics_corpus['y'] = comics_corpus['y'].map(lambda x: 1 if x == 'action' else 0)

    # Splitting between training and the validation and testing subset
    # Dividindo entre treinamento e o subset da validação e teste
    split_train = StratifiedShuffleSplit(n_splits=1, test_size=.4, random_state=42)
    for train_index, subset_index in split_train.split(comics_corpus, comics_corpus['y']):
        train_corpus, subset_corpus = comics_corpus.iloc[train_index, :], comics_corpus.iloc[subset_index, :]
    # Splitting between validation and testing
    # Dividindo entre validação e teste
    split_test = StratifiedShuffleSplit(n_splits=1, test_size=.5, random_state=42)
    for valid_index, test_index in split_test.split(subset_corpus, subset_corpus['y']):
        valid_corpus, test_corpus = subset_corpus.iloc[valid_index, :], subset_corpus.iloc[test_index, :]

    # Setting the global variables `VOCAB_SIZE` and `MAX_LEN` to tokenize the training set
    # Definindo as variáveis globais `VOCAB_SIZE` e `MAX_LEN` para tokenizar o training set
    VOCAB_SIZE = 10000
    MAX_LEN = max([len(sentence.split()) for sentence in train_corpus['description']])

    # Training the tokenizer on the training set with the previously defined `VOCAB_SIZE` and `MAX_LEN`
    # Treinando o tokenizador no training set com o `VOCAB_SIZE` e `MAX_LEN` definidos anteriormente
    sentence_vec = rnn_tokenizer(train_corpus['description'], max_tokens=VOCAB_SIZE, max_len=MAX_LEN)

    # Applying the trained tokenizer to each subset
    # Aplicando o tokenizador treinado em cada subset
    train_tokenized = sentence_vec(train_corpus['description'])
    valid_tokenized = sentence_vec(valid_corpus['description'])
    test_tokenized = sentence_vec(test_corpus['description'])


    # Loading the trained tokenization model into the `../models/` directory for later use
    # We save the hyperparameters that were used in training and the generated vocabulary
    # Carregando o modelo de tokenização treinado no diretório `../models/` para usarmos posteriormente
    # Salvamos os hiperparâmetros que foram utilizados no treinamento e o vocabulário gerado
    pickle.dump(
        {'config': sentence_vec.get_config(), 'vocabulary': sentence_vec.get_vocabulary()},
        open('../models/vectorizer.pkl', 'wb')
    )

    # Transforming the y labels into a column vector
    # Transformando os labels y em um vetor de coluna
    labels = comics_corpus['y'].to_numpy().reshape(-1, 1)
    
    # Concatenating the corpus of each subset and the corresponding labels
    # Concatenando o corpus de cada subset e os labels correspondentes
    train_tokens = np.concatenate([train_tokenized, labels[train_index]], axis=1)
    valid_tokens = np.concatenate([valid_tokenized, labels[valid_index]], axis=1)
    test_tokens = np.concatenate([test_tokenized, labels[test_index]], axis=1)

    # Loading the dataset with initial pre-processing to disk
    # Carregando no disco o dataset com pré-processamento inicial
    comics_corpus.to_csv('../data/preprocessed/comics_corpus.csv', index=False)
    
    # Loading tokenized datasets to disk
    # Carregando no disco os datasets tokenizados
    np.save('../data/preprocessed/train_corpus.npy', train_corpus)
    np.save('../data/preprocessed/valid_corpus.npy', valid_corpus)
    np.save('../data/preprocessed/test_corpus.npy', test_corpus)

    # Tokenizing, padding and returning the tokenized corpus as pytorch tensors using the pre-trained `distilbert` tokenizer
    # Tokenizando, aplicando o padding e retornando o corpus tokenizado como tensores pytorch utilizando o tokenizer `distilbert` pré-treinado
    tokenizer = DistilBertTokenizer.from_pretrained('distilbert-base-uncased-finetuned-sst-2-english')
    comics_transformers = tokenizer(
        comics_corpus['description'].tolist(),
        return_tensors='pt',
        padding=True,
        truncation=True
    )
    # Setting the vector representations of the corpus
    # Definindo as representações vetoriais do corpus
    transformers_tokens = comics_transformers['input_ids']

    # Concatenating the tokenized corpus with the corresponding y labels
    # Concatenando o corpus tokenizado com os labels y correspondentes
    dataset_transformers = np.concatenate(
        [transformers_tokens.numpy(), labels], axis=1
    )

    # Selecting each subset with their respective indices resulting from the `stratified sampling split`
    # Selecionando cada subset com os seus respectivos índices resultantes da `stratified sampling split`
    train_transformers, valid_transformers, test_transformers = dataset_transformers[train_index], dataset_transformers[valid_index], dataset_transformers[test_index]

    # Loading each preprocessed dataset into the `../data/preprocessed/` directory
    # Carregando cada dataset pré-processado no diretório `../data/preprocessed/`
    np.save('../data/preprocessed/train_transformers.npy', train_transformers)
    np.save('../data/preprocessed/valid_transformers.npy', valid_transformers)
    np.save('../data/preprocessed/test_transformers.npy', test_transformers)
