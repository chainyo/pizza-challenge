from typing import Dict, List, Tuple, Union

import pytorch_lightning as pl

import torch
import torchmetrics
from torch import nn

from transformers import AutoModel, BertTokenizerFast


class Model(nn.Module):
    def __init__(self, n_classes: int = None) -> None:
        """
        Initialize the model and it's layers. BERT is used as the backbone of the model and the classification layers 
        are added to the model.

        Parameters
        ----------
        n_classes: int
            Number of classes in the dataset. Default: None

        Raises
        ------
        ValueError
            If n_classes is not specified. 
        """
        super().__init__()
        if n_classes is not None:
            self.n_classes = n_classes
        else:
            raise ValueError("n_classes must be specified.")

        self.model = AutoModel.from_pretrained("bert-base-uncased", num_labels=self.n_classes)
        if self.training is True:
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


    def forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
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
        output = self.model(input_ids, attention_mask)
        logits = self.classification_layers(output["last_hidden_state"])
        return logits[:, -1]


class RequestClassifier(pl.LightningModule):
    def __init__(
        self, 
        n_classes: int,
        learning_rate: float = 1e-3,
    ) -> None:
        """
        Instantiate the model and it's layers. 

        Parameters
        ----------
        n_classes: int
            Number of classes in the dataset.
        learning_rate: float
            Learning rate for the optimizer. Default: 1e-3
        """
        super().__init__()
        self.model = Model(n_classes)
        self.learning_rate = learning_rate
        self.criterion = nn.CrossEntropyLoss(weight=torch.tensor([0.68, 2.0])) # Because the dataset is imbalanced
        self.train_accuracy = torchmetrics.Accuracy()
        self.val_accuracy = torchmetrics.Accuracy()
        self.test_accuracy = torchmetrics.Accuracy()
        self.save_hyperparameters()

    
    def forward(self, *args, **kwargs) -> torch.Tensor:
        return self.model(*args, **kwargs)

    
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
        loss = self.criterion(outputs, labels)
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
        loss = self.criterion(outputs, labels)
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
        loss = self.criterion(outputs, labels)
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


class ClassifierDataLoader(pl.LightningDataModule):
    """"""
    def __init__(
        self, 
        dataset: Dict[str, list] = None,
        batch_size: int = 128,
        max_seq_len: int = 512,
    ) -> None:
        """
        Instantiate the dataloader for the model. 

        Parameters
        ----------
        dataset: Dict[str, list]
            Dictionnary containing the dataset already split in train, validation and test.
        batch_size: int
            Batch size for the dataloaders. Default: 128
        max_seq_len: int
            Maximum sequence length of the dataset. Default: 512
        """
        super().__init__()
        self.tokenizer = BertTokenizerFast.from_pretrained("bert-base-uncased")
        self.dataset = dataset
        self.batch_size = batch_size
        self.max_seq_len = max_seq_len


    def setup(self, stage: str = None) -> None:
        """
        Setup the data for the dataloader.

        Parameters
        ----------
        stage: str
            Stage of the training. Either "train" or "val" or "test".
            If None, the data is loaded for all three stages. Default: None
        """
        if stage == "train" or stage is None:
            tokens_train = self.tokenize_data(self.dataset["X_train"])
            self.train_seq = torch.tensor(tokens_train["input_ids"])
            self.train_mask = torch.tensor(tokens_train["attention_mask"])
            self.train_labels = torch.tensor(self.dataset["y_train"])
        if stage == "val" or stage is None:
            tokens_val = self.tokenize_data(self.dataset["X_val"])
            self.val_seq = torch.tensor(tokens_val["input_ids"])
            self.val_mask = torch.tensor(tokens_val["attention_mask"])
            self.val_labels = torch.tensor(self.dataset["y_val"])
        if stage == "test" or stage is None:
            tokens_test = self.tokenize_data(self.dataset["X_test"])
            self.test_seq = torch.tensor(tokens_test["input_ids"])
            self.test_mask = torch.tensor(tokens_test["attention_mask"])
            self.test_labels = torch.tensor(self.dataset["y_test"])

    
    def tokenize_data(self, sample: Union[str, List[str]]) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Tokenize a sample or a list of samples using the tokenizer.

        Parameters
        ----------
        sample: Union[str, List[str]]
            Sample or list of samples to tokenize.
        
        Returns
        -------
        Tuple[torch.Tensor, torch.Tensor]
            Tuple containing the encoded ids and mask of the sample or list of samples.
        """
        if isinstance(sample, list):
            tokens = self.tokenizer.batch_encode_plus(
                sample, 
                max_length=self.max_seq_len, 
                pad_to_max_length=True, 
                truncation=True, 
                return_token_type_ids=False, 
                return_tensors="pt"
            )
        elif isinstance(sample, str):
            tokens = self.tokenizer.encode_plus(
                sample, 
                max_length=self.max_seq_len, 
                pad_to_max_length=True, 
                truncation=True, 
                return_token_type_ids=False, 
                return_tensors="pt"
            )
        return tokens


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
