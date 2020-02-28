import torch
import torch.nn as nn
from data import get_model_data, read_data, split_data, RED_WINE_PATH, WHITE_WINE_PATH



class NeuralNet(nn.Module):
    def __init__(self,input_size, hidden_size,number_of_classes):
        super().__init__()
        self.input_size = input_size
        self.hidden_size = input_size
        self.number_of_classes = number_of_classes

        # Hidden layer
        self.fc1 = nn.Linear(self.input_size,self.hidden_size)
        self.relu = nn.ReLU()
        # Output layer
        self.fc2 = nn.Linear(self.hidden_size, self.number_of_classes)
        self.softmax = nn.Softmax()


    def forward(self,x):
        hidden = self.fc1(x)
        relu = self.relu(hidden)
        output = self.fc2(relu)
        output = self.softmax(output)
        return output

        


if __name__ == "__main__":
    data = read_data(RED_WINE_PATH)
    x,y = get_model_data(data)
    x_train,x_test,y_train, y_test = split_data(x.values,y.values)
    input_size = x_train.shape[1]
    hidden_size = 100
    no_of_classes = 6
    x_train,x_test,y_train,y_test = torch.FloatTensor(x_train),torch.FloatTensor(x_test), torch.as_tensor(y_train),torch.as_tensor(y_test)
    y_train, y_test = torch.argmax(y_train, dim=1), torch.argmax(y_test, dim=1)


    
    model = NeuralNet(input_size,hidden_size,no_of_classes)

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(),lr = 0.01)


    model.eval()
    y_pred = model(x_test)
    before_train = criterion(y_pred.squeeze(), y_test)
    print('Test loss before training' , before_train.item())

    model.train()
    epoch = 1000
    print(y_train.shape)

    for epoch in range(epoch):

        optimizer.zero_grad()

        y_pred = model(x_train)
        loss = criterion(y_pred.squeeze(),y_train)
        
        if epoch % 100 == 0:
            print('Epoch {}: train loss: {}'.format(epoch, loss.item()))    # Backward pass
        loss.backward()
        optimizer.step()

    i = 200
    print(model(x_test[i]).argmax())
    print(y_test[i])
    

    


