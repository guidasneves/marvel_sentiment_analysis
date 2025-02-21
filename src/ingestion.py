# Packages used in the system
# Pacotes utilizados no sistema
import os
from requests import get
from hashlib import md5
from time import time
from dotenv import load_dotenv
load_dotenv() # access environment variables (acessa as variáveis de ambiente)

import pandas as pd
import numpy as np
from transformers import pipeline


class MarvelIngestion(object):
    """
    [EN-US]
    The `MarvelIngestion` class is composed of the `get_params` function, which defines the variables, the hash and
    the parameters for the API request, the `__call__` method that makes the connection and the API call
    to extract the requested data.
    
    [PT-BR]
    A classe `MarvelIngestion` é composta pela função `get_params`, que define as variáveis, o hash e
    os parâmetros para a requisição à API, o método `__call__` que realiza a conexão e a chamada à API
    para extrair os dados solicitados.
    """
    def __init__(
        self,
        public_key,
        private_key,
        url='http://gateway.marvel.com/v1/public/',
        limit=100,
    ):
        """
        [EN-US]
        Initializes the class, defining the corresponding arguments.

        [PT-BR]
        Inicializa a classe, definindo os argumentos correspondentes.

        Arguments:
            public_key (str): the public key for connecting and using the APIs
                              (a public key para conectar e usar as APIs).
            private_key (str): the private key for connecting and using the APIs
                               (a private key para conectar e usar as APIs).
            url (str, optional): the Marvel Comics API’s base endpoint. Defaults to http://gateway.marvel.com/v1/public/
                                 (o endpoint base da API da Marvel Comics. Padrão para http://gateway.marvel.com/v1/public/).
            limit (int, optional): the requested result limit. Defaults to 100
                                   (o limite de resultado solicitado. Padrão para 100).
        """
        super(MarvelIngestion, self).__init__()

        self.public_key = public_key
        self.private_key = private_key
        
        self.url = url
        self.limit = limit
        # Setting the headers
        # Definindo os headers
        self.headers = {
            'Accept-Encoding': '*',
            'Accpet': '*/*',
            'Connection': 'keep-alive'
        }

    def get_params(self, offset, format_=None):
        """
        [EN-US]
        Defines the timestamp variable, the hash and the parameters for the API request.

        [PT-BR]
        Define a variável timestamp, o hash e os parâmetros para requisição à API.

        Arguments:
            offset (int): the requested offset (skipped results) of the call
                          (o deslocamento solicitado (resultados ignorados) da chamada).
            format_ (str, optional): the publication format of the comic e.g. comic, hardcover, trade paperback. Default to None
                                     (o formato de publicação dos quadrinhos, por ex. quadrinhos, capa dura, brochura comercial. Padrão para None).

        Return:
            params (dict): dictionary with the parameters for requesting the API containing
                           (dicionário com os parâmetros para solicitação da API contendo):
                               ts: a timestamp
                                   (a data/hora).
                               apikey: your public key
                                       (sua public key).
                               hash: a md5 digest of the ts parameter, your private key and your public key
                                     (um md5 digest do parâmetro ts, sua private key e sua public key.).
                               limit: the requested result limit
                                      (o limite de resultado solicitado).
                               offset: the requested offset (skipped results) of the call
                                       (o deslocamento solicitado (resultados ignorados) da chamada).
                               format: the publication format of the comic e.g. comic, hardcover, trade paperback. Default to None
                                       (o formato de publicação dos quadrinhos, por ex. quadrinhos, capa dura, brochura comercial. Padrão para None).
        """
        # Setting the timestamp
        # Definindo o timestamp
        ts = str(time())
        # Creating the hash
        # Criando o hash
        hash_ = md5(
            (
                ts + self.private_key + self.public_key
            ).encode('utf-8')
        ).hexdigest()
        # Creating the dictionary of parameters to return
        # Criando o dicionário de parâmetros para retornar
        params = {
                'ts': ts,
                'apikey': self.public_key,
                'hash': hash_,
                'limit': self.limit,
                'offset': offset,
                'format': format_
            }
        
        return params
    
    def __call__(self, endpoint, offset=0, format_=None, retries=5):
        """
        [EN-US]
        Connects to the API and makes the API call to extract the requested data.

        [PT-BR]
        Realiza a conexão com a API, realiza a chamada à API para a extração dos dados solicitados.
        
        Arguments:
            endpoint (str): endpoints to access the data, e.g. comics, characters
                            (endpoints para acessar os dados, por ex. quadrinhos, personagens.).
            offset (int, optional): the requested offset (skipped results) of the call. Default to 0
                                    (o deslocamento solicitado (resultados ignorados) da chamada. Padrão para 0).
            format_ (str, optional): the publication format of the comic e.g. comic, hardcover, trade paperback. Default to None
                                     (o formato de publicação dos quadrinhos, por ex. quadrinhos, capa dura, brochura comercial. Padrão para None).
            retries (int, optional): attempts to connect, call the API and extract the data. Default to 5
                                     (tentativas para conectar, chamar a API e extrair os dados. Padrão para 5).

        Returns:
            df (pandas.DataFrame): data extracted into JSON transformed into a pandas DataFrame
                                   (dados extraídos em JSON transformados em um DataFrame do pandas).
        """
        # Setting the key
        # Definindo a key
        key = 'name' if endpoint == 'characters' else 'title'
        # Getting the parameters
        # Obtendo os parâmetros
        params = self.get_params(offset=offset, format_=format_)
        # Making the first request to the API
        # Fazendo a primeira requisição na API
        response = get(self.url + endpoint, params=params, headers=self.headers).json()
        # Getting the total number of examples returned to loop through
        # Obtendo o número total de exemplos retornados para percorrer
        total = response['data']['total']

        examples = []
        for page in range(offset, offset + total, self.limit):
            if retries > 0:
                try:
                    results = response['data']['results']
                    
                    for i in range(len(results)):
                        description = results[i]['description']
                        # Creating each example with the object id, name and description
                        # Criando cada exemplo com o id, o nome e a descrição do objeto
                        example = [
                                results[i]['id'],
                                results[i][key],
                                description
                            ]
                        # Add examples to the list if it is not duplicated, null and with a size greater than 4,
                        # because it can return the string '#N/A' as description
                        # Adiciona à lista examples caso não seja duplicado, nulo e com o tamanho maior de 4,
                        # porque ela pode retornar a string '#N/A' como descrição
                        if example not in examples and description and len(description) > 4:
                            examples.append(example)

                    # Obtaining the parameters again to update the offset and timestamp
                    # Obtendo novamente os parâmetros para atualizar o offset e o timestamp
                    params = self.get_params(offset=page + self.limit, format_=format_)
                    # Requesting the API with a new offset
                    # Requisitando a API com um novo offset
                    response = get(self.url + endpoint, params=params, headers=self.headers).json()
                
                except:
                    retries -= 1
                    print(f'Error extracting data. Remaining attempts: {retries}')

        # Setting the features and creating the DataFrame
        # Definindo as features e criando o DataFrame
        features = ['id', key, 'description']
        df = pd.DataFrame(examples, columns=features)

        return df


if __name__ == '__main__':
    # Setting the environment variables
    # Definindo as variáveis de ambiente
    PUBLIC_KEY = str(os.environ['MARVEL_PUBLIC_KEY'])
    PRIVATE_KEY = str(os.environ['MARVEL_PRIVATE_KEY'])

    # Setting the global variable `PATH` with the path of the directory where the data will be loaded
    # Configurando a variável global `PATH` com o caminho do diretório onde os dados serão carregados
    PATH = os.path.abspath(
        os.path.join(
            os.path.dirname(__file__),
            os.pardir,
            'data'
        )
    )
    
    # Initializing the `MarvelIngestion` class
    # Inicializando a classe `MarvelIngestion`
    ingestion = MarvelIngestion(PUBLIC_KEY, PRIVATE_KEY)
    print('Extracting the dataset!')
    # Extracting the `df_comics` dataset and dropping duplicate examples
    # Extraindo o conjunto de dados `df_comics` e eliminando exemplos duplicados
    df_comics = ingestion(endpoint='comics', format_='comic', offset=0).drop_duplicates('description')
    
    # Setting the labels
    # Definindo os labels
    labels = [
        'action',
        'non-action'
    ]
    # Selecting the texts from the dataset
    # Selecionando os textos do dataset
    corpus_comics = df_comics['description'].tolist()

    # Setting the pipeline with the `zero-shot-classification` task and the `facebook/bart-large-mnli` model
    # Definindo a pipeline com a tarefa de `zero-shot-classification` e o modelo `facebook/bart-large-mnli`
    pipe_bart = pipeline(
        'zero-shot-classification',
        model='facebook/bart-large-mnli'
    )
    # Running the zero-shot learning task to label the dataset data
    # Executando a tarefa de zero-shot learning para rotular os dados do dataset
    output_bart_comics = pipe_bart(corpus_comics, labels)

    # List to store the label of each example
    # Lista para armazenar o label de cada exemplo
    labels_comic = []
    # Going through the model output
    # Percorrendo o output do modelo
    for i in range(len(output_bart_comics)):
        # Selecting the id of the label with the highest score for each example
        # Selecionando o id do label com o maior score de cada exemplo
        idx = np.argmax(output_bart_comics[i]['scores'])
        # Selecting the id from the list of labels
        # Selecionando o id na lista de labels
        label = output_bart_comics[i]['labels'][idx]
        # Adding it to a list to add to the final dataset
        # Adicionando em uma lista para adicionar ao dataset final
        labels_comic.append(label)
        print(f'Labeled examples: {i + 1}/{len(output_bart_comics)}', end='\r')
    print('Successfully labeled examples!')
    
    # Adding the labels to the final dataset
    # Adicionando os labels ao dataset final
    df_comics['y'] = labels_comic

    print('Loading the dataset.')
    # Loading the dataset into the `../data/raw/` directory
    # Carregando o dataset no diretório `../data/raw/`
    df_comics.to_csv(os.path.join(PATH, 'raw', 'comics_corpus.csv'), index=False)
    print('Done!')
