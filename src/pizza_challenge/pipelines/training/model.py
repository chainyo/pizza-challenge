from typing import Dict, Tuple

import pytorch_lightning as pl

import torch
import torchmetrics
from torch import nn

from transformers import AutoModel, BertTokenizerFast


class RequestClassifier(pl.LightningModule):
    """
    Simple classifier class to say if the request deserve a pizza or not
    """
    def __init__(
        self, 
        max_seq_len: int = 400, 
        batch_size: int = 256, 
        learning_rate: float = 1e-3,
        n_classes: int = None,
        dataset: Dict[str, list] = None
    ) -> None:
        """
        Initialize the model with the parameters given and add new layers for our downstream task

        Parameters
        ----------
        max_seq_len: int
            Maximum length of the sequence used to pad the input, default 400
        batch_size: int
            Batch size for training, default is 256
        learning_rate: float
            Learning rate for the optimizer of the model, default 1e-3
        n_classes: int
            Number of classes in the dataset, default is None
        dataset: Dict[list]
            Dictionary containing the training, validation and testing data, default is None
        """
        super().__init__()
        self.max_seq_len = max_seq_len
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        if n_classes is not None:
            self.n_classes = n_classes
        else:
            raise ValueError("n_classes must be specified.")
        if dataset is not None:
            self.X_train = dataset["X_train"]
            self.X_val = dataset["X_val"]
            self.X_test = dataset["X_test"]
            self.y_train = dataset["y_train"]
            self.y_val = dataset["y_val"]
            self.y_test = dataset["y_test"]
        else:
            raise ValueError("dataset must be specified.")

        # Training metrics
        self.loss = nn.CrossEntropyLoss()
        self.train_accuracy = torchmetrics.Accuracy()
        self.val_accuracy = torchmetrics.Accuracy()
        self.test_accuracy = torchmetrics.Accuracy()
        # self.confusion_matrix = torchmetrics.ConfusionMatrix(num_classes=self.n_classes)
        # self.precision = torchmetrics.Precision(num_classes=self.n_classes, average=None)
        # self.recall = torchmetrics.Recall(num_classes=self.n_classes)
        # self.f1_score = torchmetrics.F1Score(num_classes=self.n_classes)

        self.model = AutoModel.from_pretrained("bert-base-uncased", num_labels=self.n_classes)
        self.model.eval() # Set model to evaluation mode
        for param in self.model.parameters():
            param.requires_grad = False # Freeze all the weights and prevent the existing layers from training

        self.classification_layers = nn.Sequential(
            nn.Linear(768, 512),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(512, self.n_classes),
            nn.LogSoftmax(dim=1)
        )

    
    def forward(self, encode_id: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of the model

        Parameters
        ----------
        encode_id: torch.Tensor
            Tensor of shape (batch_size, max_seq_len) containing the encoded ids of the input sequence
        mask: torch.Tensor
            Tensor of shape (batch_size, max_seq_len) containing the mask of the input sequence
        
        Returns
        -------
        torch.Tensor
            Tensor of shape (batch_size, 2) containing the logits of the model
        """
        output = self.model(encode_id, attention_mask=mask)
        logits = self.classification_layers(output["last_hidden_state"])
        return logits[:, -1]

    
    def prepare_data(self) -> None:
        """
        Load the data for the model and prepare it for training, validation and testing.
        """
        self.tokenizer = BertTokenizerFast.from_pretrained("bert-base-uncased")

        # Tokenize and encode the input sequences for training, validation and testing
        tokens_train = self.tokenizer.batch_encode_plus(
            self.X_train, 
            max_length=self.max_seq_len, 
            pad_to_max_length=True, 
            truncation=True, 
            return_token_type_ids=False, 
            return_tensors="pt"
        )
        tokens_val = self.tokenizer.batch_encode_plus(
            self.X_val, 
            max_length=self.max_seq_len, 
            pad_to_max_length=True, 
            truncation=True, 
            return_token_type_ids=False, 
            return_tensors="pt"
        )
        tokens_test = self.tokenizer.batch_encode_plus(
            self.X_test, 
            max_length=self.max_seq_len, 
            pad_to_max_length=True, 
            truncation=True, 
            return_token_type_ids=False, 
            return_tensors="pt"
        )

        self.train_seq = torch.tensor(tokens_train["input_ids"])
        self.val_seq = torch.tensor(tokens_val["input_ids"])
        self.test_seq = torch.tensor(tokens_test["input_ids"])

        self.train_mask = torch.tensor(tokens_train["attention_mask"])
        self.val_mask = torch.tensor(tokens_val["attention_mask"])
        self.test_mask = torch.tensor(tokens_test["attention_mask"])

        self.train_labels = torch.tensor(self.y_train)
        self.val_labels = torch.tensor(self.y_val)
        self.test_labels = torch.tensor(self.y_test)

    
    def train_dataloader(self) -> torch.utils.data.DataLoader:
        """
        Create a dataloader for the training set.

        Returns
        -------
        torch.utils.data.DataLoader
            Dataloader for the training set with the batch size specified in the constructor.
        """
        return torch.utils.data.DataLoader(
            torch.utils.data.TensorDataset(self.train_seq, self.train_mask, self.train_labels),
            batch_size=self.batch_size,
            shuffle=False
        )


    def val_dataloader(self) -> torch.utils.data.DataLoader:
        """
        Create a dataloader for the validation set.

        Returns
        -------
        torch.utils.data.DataLoader
            Dataloader for the validation set with the batch size specified in the constructor.
        """
        return torch.utils.data.DataLoader(
            torch.utils.data.TensorDataset(self.val_seq, self.val_mask, self.val_labels),
            batch_size=self.batch_size,
            shuffle=False
        )

    
    def test_dataloader(self) -> torch.utils.data.DataLoader:
        """
        Create a dataloader for the testing set.

        Returns
        -------
        torch.utils.data.DataLoader
            Dataloader for the testing set with the batch size specified in the constructor.
        """
        return torch.utils.data.DataLoader(
            torch.utils.data.TensorDataset(self.test_seq, self.test_mask, self.test_labels),
            batch_size=self.batch_size,
            shuffle=False
        )

    
    def training_step(self, batch: Tuple[torch.Tensor, torch.Tensor, torch.Tensor], batch_idx: int) -> Dict:
        """
        Training step of the model.

        Parameters
        ----------
        batch: Tuple[torch.Tensor, torch.Tensor, torch.Tensor]
            Tuple containing the encoded ids, mask and labels of the batch
        batch_idx: int
            Index of the batch

        Returns
        -------
        Dict
            Dict containing the loss and the accuracy of the model for the training set
        """
        encode_id, mask, labels = batch
        outputs = self(encode_id, mask)
        preds = torch.argmax(outputs, dim=1)
        self.train_accuracy(preds, labels)
        loss = self.loss(outputs, labels)
        self.log("train_accuracy", self.train_accuracy, prog_bar=True, on_step=False, on_epoch=True)
        self.log("train_loss", loss, prog_bar=True, on_step=False, on_epoch=True)
        return {"loss": loss, "train_accuracy": self.train_accuracy}


    def validation_step(self, batch: Tuple[torch.Tensor, torch.Tensor, torch.Tensor], batch_idx: int) -> Dict:
        """
        Validation step of the model.

        Parameters
        ----------
        batch: Tuple[torch.Tensor, torch.Tensor, torch.Tensor]
            Tuple containing the encoded ids, mask and labels of the batch
        batch_idx: int
            Index of the batch

        Returns
        -------
        Dict
            Dict containing the loss and the accuracy of the model for the validation set
        """
        encode_id, mask, labels = batch
        outputs = self(encode_id, mask)
        preds = torch.argmax(outputs, dim=1)
        self.val_accuracy(preds, labels)
        loss = self.loss(outputs, labels)
        self.log("val_accuracy", self.val_accuracy, prog_bar=True, on_step=False, on_epoch=True)
        self.log("val_loss", loss, prog_bar=True, on_step=False, on_epoch=True)
        return {"val_loss": loss, "val_accuracy": self.val_accuracy}

    
    def test_step(self, batch: Tuple[torch.Tensor, torch.Tensor, torch.Tensor], batch_idx: int) -> Dict:
        """
        Test step of the model.

        Parameters
        ----------
        batch: Tuple[torch.Tensor, torch.Tensor, torch.Tensor]
            Tuple containing the encoded ids, mask and labels of the batch
        batch_idx: int
            Index of the batch

        Returns
        -------
        Dict
            Dict containing the loss and the accuracy of the model for the test set
        """
        encode_id, mask, labels = batch
        outputs = self(encode_id, mask)
        preds = torch.argmax(outputs, dim=1)
        self.test_accuracy(preds, labels)
        loss = self.loss(outputs, labels)
        self.log("test_accuracy", self.test_accuracy, prog_bar=True, on_step=True, on_epoch=True)
        self.log("test_loss", loss, prog_bar=True, on_step=True, on_epoch=True)
        return {"test_loss": loss, "test_accuracy": self.test_accuracy}  


    def train_epoch_end(self, outs) -> None:
        """
        End of the training epoch. Reset the training accuracy.
        """
        self.train_accuracy.reset()

    
    def val_epoch_end(self, outs) -> None:
        """
        End of the validation epoch. Compute the validation accurac and reset the validation metrics.
        """
        self.val_accuracy.reset()

    
    def configure_optimizers(self) -> torch.optim.AdamW:
        """
        Configure the optimizer.

        Returns
        -------
        torch.optim.AdamW
            Optimizer for the model
        """
        return torch.optim.AdamW(self.parameters(), lr=self.learning_rate)
