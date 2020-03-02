import torch
import torch.nn as nn
from ignite.engine import Events, create_supervised_trainer, create_supervised_evaluator
from ignite.metrics import Accuracy, Loss
from ignite.contrib.handlers.tqdm_logger import ProgressBar
from data import WineData, RED_WINE_PATH, WHITE_WINE_PATH
from torch.utils.data import Dataset, DataLoader


class NeuralNet(nn.Module):
    def __init__(self, input_size, hidden_size, number_of_classes):
        super().__init__()
        self.input_size = input_size
        self.hidden_size = input_size
        self.number_of_classes = number_of_classes
        print(f"number of classs is {number_of_classes}")
        # Hidden layer
        self.fc1 = nn.Linear(self.input_size, self.hidden_size)
        self.relu = nn.ReLU()
        # Output layer
        self.fc2 = nn.Linear(self.hidden_size, self.hidden_size)
        self.fc3 = nn.Linear(self.hidden_size, number_of_classes)

    def forward(self, x):
        # l1
        x = self.fc1(x)
        x = self.relu(x)

        # # l2
        x = self.fc2(x)
        x = self.relu(x)

        # l3
        x = self.fc3(x)

        # # output
        # x = torch.log_softmax(x, dim=1)
        return x


def run():
    data = WineData.read_data(WHITE_WINE_PATH)
    train_data, test_data = WineData.train_test_splitter(data)
    wd = WineData(train_data)
    weights = wd.class_weights
    classes = wd.number_of_classes
    model = NeuralNet(wd.x_data.shape[1], 150, classes)
    train_loader = DataLoader(dataset=wd, batch_size=124, shuffle=True, num_workers=2)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    criterion = torch.nn.CrossEntropyLoss(weight=weights)

    max_epochs = 300

    trainer = create_supervised_trainer(model, optimizer, criterion)
    evaluator = create_supervised_evaluator(
        model, metrics={"accuracy": Accuracy(), "nll": Loss(criterion)}
    )

    @trainer.on(Events.EPOCH_COMPLETED)
    def log_training_loss(trainer):
        print(f"Epoch[{trainer.state.epoch}] Loss:[{round(trainer.state.output,2)}]")

    @trainer.on(Events.EPOCH_COMPLETED)
    def log_training_results(trainer):
        evaluator.run(train_loader)
        metrics = evaluator.state.metrics
        print(
            "Training Results - Epoch: {}  Avg accuracy: {:.2f} Avg loss: {:.2f}".format(
                trainer.state.epoch, metrics["accuracy"], metrics["nll"]
            )
        )

    trainer.run(train_loader, max_epochs=max_epochs)

if __name__ == "__main__":
    run()
