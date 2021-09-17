from torch.nn import Module, Linear
from transformers import AutoModel, AutoTokenizer


class Bert(Module):
    """
    A simple fine-tuned Bert-mini to train and test on personal computer's cpu
    """

    def __init__(self, model_path):
        super().__init__()
        self.model = AutoModel.from_pretrained(model_path)
        self.tokenizer = AutoTokenizer.from_pretrained(model_path)
        self.linear = Linear(128, 2)

    def forward(self, tokenize_ids, device):
        attention_mask = (tokenize_ids != 0).to(device)
        predict = self.model(input_ids=tokenize_ids.to(device), attention_mask=attention_mask)
        cls_feature = predict[0][:, 0]
        return self.linear(cls_feature)
