from sklearn.model_selection import train_test_split
from structure import structure
from preprocess import create_dataset
from model import Model

if __name__ == "__main__":
    print("Downloading files")
    structure()
    print("Creating dataset")
    X, y, chars = create_dataset()
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.05)

    model = Model(chars)
    model.fit(
        X_train,
        y_train,
        X_test, y_test
    )
