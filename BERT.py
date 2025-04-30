import torch
from transformers import BertTokenizer, BertForSequenceClassification
from torch.utils.data import DataLoader, TensorDataset
from sklearn.metrics import accuracy_score
from torch.optim import AdamW
from transformers import get_linear_schedule_with_warmup


class BERTModel:
    def __init__(self, model_name='bert-base-uncased', num_labels=2, weight_decay=0.01):
        # Initialize the BERT model and tokenizer with increased dropout for regularization
        self.model = BertForSequenceClassification.from_pretrained(
            model_name, num_labels=num_labels,
            hidden_dropout_prob=0.5,  # Increased dropout rate to 0.5 for better regularization
            output_attentions=False, output_hidden_states=False
        )
        self.tokenizer = BertTokenizer.from_pretrained(model_name)

        # Move the model to GPU if available
        self.model.to('cuda' if torch.cuda.is_available() else 'cpu')

        # Initialize optimizer with weight decay for regularization
        self.optimizer = AdamW(self.model.parameters(), lr=2e-5, weight_decay=weight_decay)  # Fine-tuned learning rate

    def train(self, train_data, val_data, epochs=15, batch_size=32, max_grad_norm=1.0):
        train_texts, train_labels = train_data
        val_texts, val_labels = val_data

        # Tokenize the data
        train_encodings = self.tokenizer(train_texts, truncation=True, padding=True, max_length=512)
        val_encodings = self.tokenizer(val_texts, truncation=True, padding=True, max_length=512)

        train_dataset = TensorDataset(torch.tensor(train_encodings['input_ids']),
                                      torch.tensor(train_encodings['attention_mask']),
                                      torch.tensor(train_labels))

        val_dataset = TensorDataset(torch.tensor(val_encodings['input_ids']),
                                    torch.tensor(val_encodings['attention_mask']),
                                    torch.tensor(val_labels))

        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

        # Set up the learning rate scheduler
        total_steps = len(train_loader) * epochs
        scheduler = get_linear_schedule_with_warmup(self.optimizer,
                                                    num_warmup_steps=0,
                                                    num_training_steps=total_steps)

        # Early stopping initialization
        best_val_accuracy = 0.0
        patience = 5  # Increased patience for early stopping
        epochs_without_improvement = 0

        # Training loop
        for epoch in range(epochs):
            self.model.train()
            total_loss = 0
            for batch in train_loader:
                self.optimizer.zero_grad()
                input_ids, attention_mask, labels = [item.to('cuda' if torch.cuda.is_available() else 'cpu') for item in
                                                     batch]
                outputs = self.model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
                loss = outputs.loss
                total_loss += loss.item()
                loss.backward()

                # Gradient clipping to prevent exploding gradients
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_grad_norm)

                self.optimizer.step()
                scheduler.step()  # Update the learning rate

            avg_train_loss = total_loss / len(train_loader)
            print(f"Epoch {epoch + 1}/{epochs} - Training Loss: {avg_train_loss:.4f}")
            val_accuracy = self.evaluate(val_loader)  # Validate after each epoch

            # Early stopping based on validation accuracy improvement
            if val_accuracy > best_val_accuracy:
                best_val_accuracy = val_accuracy
                epochs_without_improvement = 0
            else:
                epochs_without_improvement += 1
                if epochs_without_improvement >= patience:
                    print(f"Early stopping triggered after {epoch + 1} epochs.")
                    break

    def evaluate(self, val_loader):
        self.model.eval()
        val_preds = []
        val_labels = []
        with torch.no_grad():
            for batch in val_loader:
                input_ids, attention_mask, labels = [item.to('cuda' if torch.cuda.is_available() else 'cpu') for item in
                                                     batch]
                outputs = self.model(input_ids=input_ids, attention_mask=attention_mask)
                logits = outputs.logits
                preds = torch.argmax(logits, dim=-1)
                val_preds.extend(preds.cpu().numpy())
                val_labels.extend(labels.cpu().numpy())

        val_accuracy = accuracy_score(val_labels, val_preds)
        print(f"Validation Accuracy: {val_accuracy:.4f}")
        return val_accuracy

    def test(self, test_data, batch_size=32):
        test_texts, test_labels = test_data
        test_encodings = self.tokenizer(test_texts, truncation=True, padding=True, max_length=512)
        test_dataset = TensorDataset(torch.tensor(test_encodings['input_ids']),
                                     torch.tensor(test_encodings['attention_mask']),
                                     torch.tensor(test_labels))
        test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

        self.model.eval()
        test_preds = []
        test_labels = []
        with torch.no_grad():
            for batch in test_loader:
                input_ids, attention_mask, labels = [item.to('cuda' if torch.cuda.is_available() else 'cpu') for item in
                                                     batch]
                outputs = self.model(input_ids=input_ids, attention_mask=attention_mask)
                logits = outputs.logits
                preds = torch.argmax(logits, dim=-1)
                test_preds.extend(preds.cpu().numpy())
                test_labels.extend(labels.cpu().numpy())

        test_accuracy = accuracy_score(test_labels, test_preds)
        print(f"Test Accuracy: {test_accuracy:.4f}")
        return test_accuracy
