#Este arquivo
#possui o código da aula anterior onde os dados de
#treinamento pré-processados foram criados usando
#stemização e o saco de palavras (BOW).


# Bibliotecas de pré-processamento de dados de texto
import nltk

from nltk.stem import PorterStemmer
stemmer = PorterStemmer()

import json
#para armazenar estruturas de dados complexas como: listas, dicionários, classes
import pickle

import numpy as np

import random

words=[] #lista de palavras-raiz únicas nos dados
classes = [] #lista de tags únicas nos dados
#lista dos pares de (['palavras', 'da', 'frase'], 'tags')
pattern_word_tags_list = [] 
ignore_words = ['?', '!',',','.', "'s", "'m"]

train_data_file = open('intents.json').read()
intents = json.loads(train_data_file)




def get_stem_words(words, ignore_words):
    stem_words = []
    for word in words:
        if word not in ignore_words:
            w = stemmer.stem(word.lower())
            stem_words.append(w)
    return stem_words
 

#Em resumo, essa função cria um corpus para treinar um modelo de chatbot ou processamento de linguagem natural,
# coletando padrões, palavras tokenizadas e tags associadas a cada intenção.
def create_bot_corpus(words, classes, pattern_word_tags_list, ignore_words):

    for intent in intents['intents']:

        # Adicione todos os padrões e tags a uma lista
        for pattern in intent['patterns']:            
            pattern_word = nltk.word_tokenize(pattern)            
            words.extend(pattern_word)                        
            pattern_word_tags_list.append((pattern_word, intent['tag']))
              
        # Adicione todas as tags à lista classes
        if intent['tag'] not in classes:
            classes.append(intent['tag'])
            
    stem_words = get_stem_words(words, ignore_words) 
    stem_words = sorted(list(set(stem_words)))
    classes = sorted(list(set(classes)))

    return stem_words, classes, pattern_word_tags_list


#Em resumo, essa função recebe uma lista de palavras raízes (stem_words) 
#e uma lista de padrões de palavras com suas tags associadas (pattern_word_tags_list). 
#Ela cria uma representação bag-of-words para cada padrão, indicando quais palavras raízes estão presentes no padrão.
# A função retorna essas representações bag-of-words em um array NumPy.
def bag_of_words_encoding(stem_words, pattern_word_tags_list):  

    bag = []

    for word_tags in pattern_word_tags_list:
        pattern_words = word_tags[0] 
        bag_of_words = []
        stem_pattern_words= get_stem_words(pattern_words, ignore_words)
        for word in stem_words:            
            if word in stem_pattern_words:              
                bag_of_words.append(1)
            else:
                bag_of_words.append(0)

        bag.append(bag_of_words)

    return np.array(bag)





def class_label_encoding(classes, pattern_word_tags_list):

    labels = []

    for word_tags in pattern_word_tags_list:
        #Inicializa uma lista chamada labels_encoding com zeros, com o mesmo comprimento que a lista de classes.
        labels_encoding = list([0]*len(classes)) 
        #Obtém a tag associada ao padrão (word_tags[1]).
        tag = word_tags[1]
        #Encontra o índice da tag na lista de classes (tag_index = classes.index(tag)).
        tag_index = classes.index(tag)
        #Modifica a posição correspondente em labels_encoding para 1, indicando a classe da tag.
        labels_encoding[tag_index] = 1

        labels.append(labels_encoding)

    return np.array(labels)



def preprocess_train_data():
  
    #Chama a função create_bot_corpus para obter as palavras raízes 
    stem_words, tag_classes, word_tags_list = create_bot_corpus(words, classes, 
                                            pattern_word_tags_list, ignore_words)
    
    #Salva as palavras raízes em um arquivo chamado 'words.pkl' e as classes em um arquivo chamado 'classes.pkl' usando pickle.dump.
    pickle.dump(stem_words, open('words.pkl','wb'))
    pickle.dump(tag_classes, open('classes.pkl','wb'))

    #Chama as funções bag_of_words_encoding e class_label_encoding para obter as representações adequadas dos dados de treinamento.
    train_x = bag_of_words_encoding(stem_words, word_tags_list)
    train_y = class_label_encoding(tag_classes, word_tags_list)
    
    return train_x, train_y

# preprocess_train_data()


