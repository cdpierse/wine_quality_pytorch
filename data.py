import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
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


if __name__ == "__main__":
    data = read_data(RED_WINE_PATH)
    x,y = get_model_data(data)
    x = scale_data(x)
    print(split_data(x,y)) 

