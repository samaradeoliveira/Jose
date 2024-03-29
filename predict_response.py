# Bibliotecas de pré-processamento de dados de texto
#importar nltk

#importar o json

#importar o pickle

#inportar o numpy

#importar o random


#ignorando simbolos


#descomentar o tensorflow e análisar
#import tensorflow
#from data_preprocessing import get_stem_words


#carregando modelo na variável model



# Carregue os arquivos de dados




#analisar a função
def preprocess_user_input(user_input):

    input_word_token_1 = nltk.word_tokenize(user_input)
    input_word_token_2 = get_stem_words(input_word_token_1, ignore_words) 
    input_word_token_2 = sorted(list(set(input_word_token_2)))

    bag=[]
    bag_of_words = []
   
    # Codificação dos dados de entrada 
    for word in words:            
        if word in input_word_token_2:              
            bag_of_words.append(1)
        else:
            bag_of_words.append(0) 
    bag.append(bag_of_words)
  
    return np.array(bag)



#analisar a função
def bot_class_prediction(user_input):

    inp = preprocess_user_input(user_input)
    prediction = model.predict(inp)
    predicted_class_label = np.argmax(prediction[0])

    return predicted_class_label



#analisar a função
def bot_response(user_input):

   predicted_class_label =  bot_class_prediction(user_input)
   predicted_class = classes[predicted_class_label]

   for intent in intents['intents']:
    if intent['tag']==predicted_class:
        bot_response = random.choice(intent['responses'])
        return bot_response


#criar o print do robo para começar a conversar 


#criar um while para poder começar. 

