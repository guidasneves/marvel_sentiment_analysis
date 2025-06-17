# Packages used in the system
# Pacotes utilizados no sistema
import pandas as pd
import numpy as np
import re
import string
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from tensorflow.data import Dataset
from tensorflow.keras.layers import TextVectorization
from sklearn.model_selection import StratifiedShuffleSplit
import pickle
from transformers import DistilBertTokenizer
from datasets import Dataset

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

    Return:
        clean_sentence (str): preprocessed text
                              (texto pré-processado).
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

    Return:
        vectorize_layer (tensorflow.keras.layers.TextVectorization): vectorizer layer trained on the training set corpus
                                                                     (vectorizer layer treinada sobre o training set corpus).
    """
    # Setting the tokenizer with the specified max_tokens and max_len
    # Definindo o tokenizer com o max_tokens e max_len especificados
    vectorize_layer = TextVectorization(
        max_tokens=max_tokens,
        output_sequence_length=max_len
    )
    # Training the tokenizer on the corpus
    # Treinando o tokenizer no corpus
    vectorize_layer.adapt(corpus)

    return vectorize_layer


def tensors_to_dataset(input_ids, attention_mask, labels, idx=...):
    """
    [EN-US]
    Creates a datasets.Dataset with the subset of tokenized tensors, the labels
    and the corresponding ids to select the subset.
    
    [PT-BR]
    Cria um datasets.Dataset com do subset dos tensores tokenizados, os labels
    e os ids correspondetes para selecionar o subset.

    Arguments:
        input_ids (torch.Tensor): tensor with the tokenized sentences returned by the tokenizer
                                  (tensor com as frases tokenizadas retornadas pelo tokenizador).
        attention_mask (torch.Tensor): tensor that indicates the paddings of the tokenized sentences returned by the tokenizer
                                       (tensor que indica os paddings das frases tokenizadas retornadas pelo tokenizador).
        labels (numpy.array or pandas.Series): array with the labels corresponding to the tokenized tensor
                                               (array com os labels correspondentes ao tensor tokenizado).
        idx (numpy.array or list, optional): array of indices to select the subset of tensors and the corresponding labels. Defaults to ...
                                             (array de índices para selecionar o subset dos tensores e os labels correspondentes. Padrão para ...).

    Return:
        dataset (datasets.Dataset): Dataset with values in pytorch tensor format
                                    (Dataset com os valores em formato tensores pytorch).
    """
    # Creating the dictionary with the tensors, dividing each one by the id subset argument
    # Criando o dicionário com os tensores, dividindo cada um pelo argumento do subset de ids
    dataset_dict = {
        'input_ids': input_ids[idx],
        'attention_mask': attention_mask[idx],
        'labels': labels[idx]
    }

    # # Creating the `Dataset` object from the dictionary we created above
    # Criando o objeto `Dataset` do dicionário que criamos acima 
    dataset = Dataset.from_dict(dataset_dict)
    # Transforming the Dataset format to pytorch tensor
    # Transformando o formato do Dataset para tensor pytorch
    dataset = dataset.with_format('torch')
    
    return dataset


if __name__ == '__main__':
    # Setting the global variables `PATH` and `PATH_M`,
    # with the path of the directory where the data will be loaded and the path of the model weights
    # Configurando as variáveis globais `PATH` e `PATH_M`,
    # com o path do diretório onde os dados serão carregados e o path dos pesos modelos
    PATH = os.path.abspath(
        os.path.join(
            os.path.dirname(__file__),
            os.pardir,
            'data'
        )
    )
    PATH_M = os.path.abspath(
        os.path.join(
            os.path.dirname(__file__),
            os.pardir,
            'models'
        )
    )
    
    # Reading the dataset that will be pre-processed
    # Lendo o dataset que será pré-processado
    comics_data = pd.read_csv(os.path.join(PATH, 'raw', 'comics_corpus.csv'))

    # Preprocessing the data
    # Pré-processando os dados
    comics_data_pre = comics_data.copy()
    comics_data_pre['description'] = comics_data_pre['description'].map(rnn_preprocess)
    # Removing duplicate examples after preprocessing
    # Removendo os exemplos duplicados após o pré-processamento
    comics_data_pre = comics_data_pre.drop_duplicates('description')

    # Creating a dataset with only the features that will be used in the mode
    # Criando um dataset com apenas as features que serão usadas no modelo
    comics_corpus = comics_data_pre[['description', 'y']].copy()
    # Transforming the target label y into binary
    # Transformando o target label y em binário
    comics_corpus['y'] = comics_corpus['y'].map(lambda x: 1 if x == 'action' else 0)

    # Splitting between training and the validation and testing subset
    # Dividindo entre treinamento e o subset da validação e teste
    split_train = StratifiedShuffleSplit(n_splits=1, test_size=.4, random_state=42)
    for train_index, subset_index in split_train.split(comics_corpus, comics_corpus['y']):
            train_corpus, subset_corpus = comics_corpus.iloc[train_index, :].copy(), comics_corpus.iloc[subset_index, :].copy()
    # Splitting between validation and testing
    # Dividindo entre validação e teste
    split_test = StratifiedShuffleSplit(n_splits=1, test_size=.5, random_state=42)
    for val_index, test_index in split_test.split(subset_corpus, subset_corpus['y']):
        val_corpus, test_corpus = subset_corpus.iloc[val_index, :].copy(), subset_corpus.iloc[test_index, :].copy()

    # Setting the global variables `VOCAB_SIZE` and `MAX_LEN` to tokenize the training set
    # Definindo as variáveis globais `VOCAB_SIZE` e `MAX_LEN` para tokenizar o training set
    VOCAB_SIZE = 1000
    MAX_LEN = max([len(sentence.split()) for sentence in train_corpus['description']])

    # Training the tokenizer on the training set with the previously defined `VOCAB_SIZE` and `MAX_LEN`
    # Treinando o tokenizador no training set com o `VOCAB_SIZE` e `MAX_LEN` definidos anteriormente
    sentence_vec = rnn_tokenizer(train_corpus['description'], max_tokens=VOCAB_SIZE, max_len=MAX_LEN)

    # Applying the trained tokenizer to each subset
    # Aplicando o tokenizador treinado em cada subset
    train_tokenized = sentence_vec(train_corpus['description'])
    val_tokenized = sentence_vec(val_corpus['description'])
    test_tokenized = sentence_vec(test_corpus['description'])

    # Loading the trained tokenization model into the `../models/` directory for later use
    # We save the hyperparameters that were used in training and the generated vocabulary
    # Carregando o modelo de tokenização treinado no diretório `../models/` para usarmos posteriormente
    # Salvamos os hiperparâmetros que foram utilizados no treinamento e o vocabulário gerado
    pickle.dump(
        {'config': sentence_vec.get_config(), 'vocabulary': sentence_vec.get_vocabulary()},
        open(os.path.join(PATH_M, 'vectorizer.pkl'), 'wb')
    )

    # Transforming the y labels into a column vector
    # Transformando os labels y em um vetor de coluna
    labels_train = train_corpus[['y']].copy()
    labels_val = val_corpus[['y']].copy()
    labels_test = test_corpus[['y']].copy()
    
    # Concatenating the corpus of each subset and the corresponding labels
    # Concatenando o corpus de cada subset e os labels correspondentes
    train_tokens = np.concatenate([train_tokenized, labels_train], axis=1)
    val_tokens = np.concatenate([val_tokenized, labels_val], axis=1)
    test_tokens = np.concatenate([test_tokenized, labels_test], axis=1)

    # Loading the dataset with initial pre-processing into the `../data/preprocessed/` directory
    # Carregando no diretório `../data/preprocessed/` o dataset com pré-processamento inicial
    comics_corpus.to_csv(os.path.join(PATH, 'preprocessed', 'comics_corpus.csv'), index=False)
    
    # Loading tokenized datasets into the `../data/preprocessed/` directory
    # Carregando no diretório `../data/preprocessed/` os datasets tokenizados
    np.save(os.path.join(PATH, 'preprocessed', 'train_tokens.npy'), train_tokens)
    np.save(os.path.join(PATH, 'preprocessed', 'validation_tokens.npy'), val_tokens)
    np.save(os.path.join(PATH, 'preprocessed', 'test_tokens.npy'), test_tokens)

    # Tokenizing, padding and returning the tokenized corpus as pytorch tensors using the pre-trained `distilbert` tokenizer
    # Tokenizando, aplicando o padding e retornando o corpus tokenizado como tensores pytorch utilizando o tokenizer `distilbert` pré-treinado
    tokenizer = DistilBertTokenizer.from_pretrained('distilbert-base-uncased-finetuned-sst-2-english')
    # Setting the tokenizer
    # Definindo o tokenizer
    comics_transformers = tokenizer(
        comics_data['description'].tolist(),
        return_tensors='pt',
        padding='max_length',
        truncation=True
    )
    # Accessing the vector representations of the corpus
    # Acessando as representações vetoriais do corpus
    transformers_tokens = comics_transformers['input_ids']
    transformers_attention = comics_transformers['attention_mask']

    # Selecting labels from the raw dataset
    # Selecionando os labels do dataset bruto
    labels = (comics_data['y']
              .map(lambda x: 1 if x == 'action' else 0)
              .to_numpy()
              .reshape(-1, 1))
    
    # Splitting between training and the validation and testing subset
    # Dividindo entre treinamento e o subset da validação e teste
    train_idx, subset_idx = next(split_train.split(transformers_tokens, labels))
    # Splitting between validation and testing
    # Dividindo entre validação e teste
    val_idx, test_idx = next(split_test.split(transformers_tokens[subset_idx], labels[subset_idx]))

    # Transforming split pytorch tensor subsets for type `Dataset` into dictionary format
    # Transformando os subsets de tensores pytorch divididos para o tipo `Dataset` em formato de dicionário
    train_dataset = tensors_to_dataset(transformers_tokens, transformers_attention, labels, train_idx)
    val_dataset = tensors_to_dataset(transformers_tokens, transformers_attention, labels, val_idx)
    test_dataset = tensors_to_dataset(transformers_tokens, transformers_attention, labels, test_idx)

    # Loading each pre-processed dataset and its metadata into its specific directory within the `../data/preprocessed/` directory
    # Carregando cada dataset pré-processado e seus metadados dentro de seu diretório específico dentro do diretório `../data/preprocessed/`
    train_dataset.save_to_disk('../data/preprocessed/train_dataset')
    val_dataset.save_to_disk('../data/preprocessed/validation_dataset')
    test_dataset.save_to_disk('../data/preprocessed/test_dataset')
