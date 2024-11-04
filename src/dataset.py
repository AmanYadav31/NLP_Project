from torch.utils.data import Dataset, DataLoader
import pandas as pd

class SingleStage(Dataset):
    def __init__(self, csv_file, tokenizer, max_length=128):
        self.data = pd.read_csv(csv_file)
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        prompt = self.data.loc[idx, 'Prompt']
        story = self.data.loc[idx, 'Story']

        # Encode the input prompt and outline text
        input_text = str(prompt) + " <SEP> " + str(story)
        inputs = self.tokenizer.encode_plus(input_text, add_special_tokens=True, max_length=self.max_length, padding="max_length", truncation=True, return_tensors="pt")

        return inputs

class PromptToOutline(Dataset):
    def __init__(self, csv_file, tokenizer, max_length=128):
        self.data = pd.read_csv(csv_file)
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        prompt = self.data.loc[idx, 'Prompt']
        outline = self.data.loc[idx, 'Outline']

        # Encode the input prompt and outline text
        input_text = str(prompt) + " <SEP> " + str(outline)
        inputs = self.tokenizer.encode_plus(input_text, add_special_tokens=True, max_length=self.max_length, padding="max_length", truncation=True, return_tensors="pt")

        return inputs
    
class DoubleStage(Dataset):
    def __init__(self, csv_file, tokenizer, max_length=128):
        self.data = pd.read_csv(csv_file)
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        prompt = self.data.loc[idx, 'Prompt']
        outline = self.data.loc[idx, 'Outline']
        story = self.data.loc[idx, 'Story']

        # Encode the input prompt and outline text
        input_text = str(prompt) + " <S> " + str(outline) + " <SEP> " + str(story)
        inputs = self.tokenizer.encode_plus(input_text, add_special_tokens=True, max_length=self.max_length, padding="max_length", truncation=True, return_tensors="pt")

        return inputs
