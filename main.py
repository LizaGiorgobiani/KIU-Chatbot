import os
import json #to read intents
import random #for picking a random response from the intents file

import nltk # for tokenization and lemmatization
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from pyexpat.errors import messages
from torch.utils.data import DataLoader, TensorDataset

#nltk.download('punkt_tab')
#nltk.download('wordnet')


class ChatbotModel(nn.Module):

    def __init__(self,input_size, output_size):
        super(ChatbotModel,self).__init__()

        #this is going to be the first hidden layer of neural networks and we are feeding the data to 128 neurons
        self.fc1 = nn.Linear(input_size, 128)
        #second hidden layer feeding the 128 neurons to 64 neurons
        self.fc2 = nn.Linear(128, 64)
        #last layer we are going to turn the inputs to outputs
        self.fc3 = nn.Linear(64, output_size)
        #activation function to break linearity
        self.relu = nn.ReLU()
        #regularization
        self.dropout = nn.Dropout(0.5)

    #forward propagation when we get input how we get the output
    def forward(self, x):
        #multiply by weight add bias and apply relu
        x = self.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.relu(self.fc2(x))
        x = self.dropout(x)
        # we want to have output probabilities so not gonna apply relu
        x = self.fc3(x)

        return x



class ChatbotAssistant:
    def __init__(self, intents_path, function_mappings = None):
        self.model = None
        self.intents_path = intents_path

        self.documents = []
        self.vocabulary = []
        self.intents = []
        self.intents_responses = {}

        self.function_mappings = function_mappings

        self.X = None # matrix
        self.y = None #output vector

    @staticmethod
    def tokenize_and_lemmatize(text):
        lemmatizer = nltk.WordNetLemmatizer()
        #take a text and extract words and reduce them down to word stem and also ignore the casing we are going to lower them
        words = nltk.word_tokenize(text)
        words = [lemmatizer.lemmatize(word.lower()) for word in words]

        return words

    #turn tokenized words to numerical representation and we are using bag of words strategy

    def bag_of_words(self,words):
        return [1 if word in words else 0 for word in self.vocabulary]

    def parse_intents(self):
        lemmatizer = nltk.WordNetLemmatizer()

        if os.path.exists(self.intents_path):
            with open(self.intents_path, 'r', encoding= 'utf-8') as f:
                intents_data = json.load(f)


            #dictionary which contains tag patterns
            #going to the intent saving the tag to intents list and to intents_respones we add the repsonses so basically tags are mapped to the responses
            for intent in intents_data['intents']:
                if intent['tag'] not in self.intents:
                    self.intents.append(intent['tag'])
                    self.intents_responses[intent['tag']] = intent['responses']
                #we are going to encode the patterns as well
                for pattern in intent['patterns']:
                    pattern_words = self.tokenize_and_lemmatize(pattern)
                    self.vocabulary.extend(pattern_words)
                    self.documents.append((pattern_words,intent['tag']))

                #using set to eliminate duplicates and sorted for it to become a list
                self.vocabulary = sorted(set(self.vocabulary))


    def prepare_data(self):
        bags = []
        indices = []

        for document in self.documents:
            words = document[0]
            bag = self.bag_of_words(words) # turn the words into 0 and 1

            intent_index = self.intents.index(document[1])

            bags.append(bag)
            indices.append(intent_index)

        #in the X array we will have bag of words then we have ys which are vector of correct predictions
        self.X = np.array(bags)
        self.y = np.array(indices)



    def train_model(self, batch_size, lr, epochs):
        X_tensor = torch.tensor(self.X, dtype=torch.float32) # bag of word representation
        y_tensor = torch.tensor(self.y, dtype = torch.long) # classification

        dataset = TensorDataset(X_tensor, y_tensor)
        loader = DataLoader(dataset, batch_size = batch_size, shuffle= True)

        self.model = ChatbotModel(self.X.shape[1],len(self.intents))
        #loss function
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(self.model.parameters() , lr = lr)

        for epoch in range(epochs):
            running_loss = 0.0

            for batch_X, batch_y in loader:
                optimizer.zero_grad()
                outputs = self.model(batch_X)
                #wanna know how wrong these outputs are batch_y are the correct outputs
                loss = criterion(outputs, batch_y)
                #we backpropagate this to see where it went wrong
                loss.backward()
                #this step depends on the learning rate
                optimizer.step()
                running_loss += loss
            print(f"Epoch{epoch + 1}: Loss: {running_loss/len(loader):.4f}")

    def save_model(self,model_path, dimensions_path):
        torch.save(self.model.state_dict(), model_path)

        with open(dimensions_path, 'w') as f :
            json.dump({'input_size': self.X.shape[1], 'output_size': len(self.intents) }, f)

    def load_model(self, model_path, dimensions_path):
        with open(dimensions_path, 'r') as f:
            dimensions = json.load(f)
        self.model = ChatbotModel(dimensions['input_size'], dimensions['output_size'])
        self.model.load_state_dict(torch.load(model_path, weights_only=True))

    def process_message(self, input_message):
        # take the input message and split it into tokens reduce them to the word stem
        words = self.tokenize_and_lemmatize(input_message)
        bag = self.bag_of_words(words)

        bag_tensor = torch.tensor([bag], dtype=torch.float32)

        self.model.eval()
        with torch.no_grad():
            predictions = self.model(bag_tensor)

        predicted_class_index = torch.argmax(predictions, dim=1).item()
        predicted_intent = self.intents[predicted_class_index]

        if self.function_mappings:
            if predicted_intent in self.function_mappings:
                self.function_mappings[predicted_intent]()

        if self.intents_responses[predicted_intent]:
            return random.choice(self.intents_responses[predicted_intent])

        else:
            return None



if __name__ == '__main__':
        assistant = ChatbotAssistant('intents.json')
        assistant.parse_intents()
        assistant.prepare_data()

        assistant.train_model(batch_size=8, lr=0.001, epochs=100)

        assistant.save_model('chatbot_model.pth', 'dimensions.json')

        # assistant = ChatbotAssistant('intents.json')
        # assistant.parse_intents()
        assistant.load_model('chatbot_model.pth', 'dimensions.json')

        while True:
            message = input('Enter your question: ')

            if message == '/quit':
                break

            print(assistant.process_message(message))





