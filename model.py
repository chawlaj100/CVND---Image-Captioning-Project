import torch
import torch.nn as nn
import torchvision.models as models


class EncoderCNN(nn.Module):
    def __init__(self, embed_size):
        super(EncoderCNN, self).__init__()
        resnet = models.resnet50(pretrained=True)
        for param in resnet.parameters():
            param.requires_grad_(False)
        
        modules = list(resnet.children())[:-1]
        self.resnet = nn.Sequential(*modules)
        self.embed = nn.Linear(resnet.fc.in_features, embed_size)

    def forward(self, images):
        features = self.resnet(images)
        features = features.view(features.size(0), -1)
        features = self.embed(features)
        return features
    

class DecoderRNN(nn.Module):
    def __init__(self, embed_size, hidden_size, vocab_size, num_layers=1):
        super(DecoderRNN, self).__init__()
        # Defining the variables for the Decoder
        self.embed_size = embed_size
        self.hidden_size = hidden_size
        self.vocab_size = vocab_size
        self.num_layers = num_layers
        
        # Embedding for converting the probablities/vocab to a feature vector
        self.embedding = nn.Embedding(self.vocab_size, self.embed_size)
        
        # LSTM layer
        self.lstm = nn.LSTM(self.embed_size, self.hidden_size, self.num_layers, batch_first=True)
        
        # Defining the hidden layer in between LSTM and the Dense layer
        self.hidden = (torch.zeros(1, 1, self.hidden_size), torch.zeros(1, 1, self.hidden_size))
        
        # Fully connected Dense layer
        self.fc = nn.Linear(self.hidden_size, self.vocab_size)
       
    
    def forward(self, features, captions):
        # To extract the caption excluding the 'end' character
        captions = self.embedding(captions[:,:-1])
        
        # Concatenating the feature vector of Encoder with the caption
        concatenated_embedding = torch.cat((features.unsqueeze(1), captions), 1)
        
        # Passing the entire vector to our LSTM
        output, self.hidden = self.lstm(concatenated_embedding)
        
        # Passing the output of LSTM to Dense layer
        output = self.fc(output)
        return output

    def sample(self, inputs, states=None, max_len=20):
        " accepts pre-processed image tensor (inputs) and returns predicted sentence (list of tensor ids of length max_len) "
        caption_list = []
        for l in range(max_len):
            # Passing the inputs and hidden states to our Decoder network for generating output
            out, states = self.lstm(inputs, states)
            out = self.fc(out.squeeze(1))
            
            # Selecting the keyword with maximum prob
            out = out.max(1)[1]
            
            # Appending the keyword having maximum value/prob to our list 
            caption_list.append(out.item())
            
            # We use output of one timestamp as input to other so creating embedding of the output keyword for input to other timestamp.
            inputs = self.embedding(out).unsqueeze(1)
        return caption_list
