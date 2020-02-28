import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
import torch
from torch.utils.data import Dataset, DataLoader

RED_WINE_PATH = "data/winedata/winequality_red.csv"
WHITE_WINE_PATH = "data/winedata/winequality_white.csv"
def read_data(path):
    return pd.read_csv(path)

def get_model_data(data : pd.DataFrame()):
    if "quality" not in data.columns:
        raise Exception("Sorry quality feature is not in this dataset") 
    else:
        target = data['quality']
        inputs = data.drop(columns=['quality'])
        return(inputs,pd.get_dummies(target))
def split_data(x,y):
    return train_test_split(x,y,test_size=0.2, random_state =42)

def scale_data(x):
    scaler = MinMaxScaler()
    scaler.fit(x)
    return scaler.transform(x)


class WineData(Dataset):

    def __init__(self, data : pd.DataFrame):
        """Initializes an instance of WineData dataset. 
        Data is split into x and y and then converted from 
        dataframes and series to a torch tensor. 
        
        Arguments:
            data {pd.DataFrame} -- the raw pandas dataframe we will convert to a x and y 
            torch tensor. 
        """        
        self.data = data
        self.x_data, self.y_data = self.get_x_y(self.data)
        self.x_data, self.y_data = self.df_to_torch(self.x_data),self.df_to_torch(self.y_data)


    def __getitem__(self,index):
        return self.x_data[index], self.y_data[index]
    
    def __len__(self):
        return len(self.x_data)

    @staticmethod
    def df_to_torch(df):
        return torch.tensor(df.values)

    @staticmethod
    def read_data(path):
        return pd.read_csv(path)

    @staticmethod
    def get_x_y(data):
        if "quality" not in data.columns:
            raise BaseException("Column `quality` could not be found in your dataset")
        else:
            y = data['quality']
            x = data.drop(columns=['quality'])
            return (x,y)
            
    @staticmethod
    def train_test_splitter(data, test_size = 0.1):
        """ Static helper method for splitting a provided 
        wine dataframe into a train and test with a split of `test_size`
        and stratified on the `data['quality']` column.
        
        Keyword Arguments:
            test_size {float} -- [description] (default: {0.1})
        
        Raises:
            ValueError: If the quality column is not present a value error is raised.
        
        Returns:
            [type] -- [description]
        """        
        if 'quality' not in data.columns:
            raise ValueError("The given dataframe must contain the `quality` column for the wine dataset")
        return train_test_split(data,test_size=test_size, stratify= data['quality'])



if __name__ == "__main__":
    df = pd.DataFrame()
    # data = WineData.read_data(WHITE_WINE_PATH)
    train_data, test_data= WineData.train_test_splitter(df)
    

