import torch
from transformers import DistilBertTokenizer, DistilBertForSequenceClassification


class DistilBERTModel:
    def __init__(self, model_path, vocab_path, device='cpu'):
        self.device = torch.device(device)
        self.tokenizer = DistilBertTokenizer.from_pretrained(vocab_path)
        self.model = torch.load(model_path, map_location=self.device)  # Ensure model loads to the right device
        self.model.eval()  # Set the model to evaluation mode

    def predict(self, text, threshold=0.5):
        # Prepare the input data
        inputs = self.tokenizer.encode_plus(
            text,
            add_special_tokens=True,
            max_length=256,  # Ensure your model max_length during training is the same
            padding='max_length',
            truncation=True,
            return_tensors="pt"
        )

        input_ids = inputs["input_ids"].to(self.device)
        attention_mask = inputs["attention_mask"].to(self.device)

        # Model inference
        with torch.no_grad():
            logits = self.model(input_ids, attention_mask=attention_mask)[0]

        # Convert logits to probabilities using sigmoid
        probs = torch.sigmoid(logits).cpu().numpy()
        predictions = (probs > threshold).astype(int)

        return predictions[0], probs[0]
