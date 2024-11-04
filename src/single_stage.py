import yaml
import torch
import pandas as pd
from transformers import GPT2LMHeadModel, GPT2Tokenizer
from torch.utils.data import  DataLoader
from dataset import *
from utils import *
from tqdm import tqdm
import os 

with open('config.yaml',) as file:
    config = yaml.full_load(file)
    
training_args = config['training_args']

#checkpoint directory
checkpoint_dir = os.path.join(config['model']['save_path'], 'single_stage')
model_path = os.path.join(checkpoint_dir, 'model.safetensors')

#Model
decoder_name = config['model']['decoder']
tokenizer = GPT2Tokenizer.from_pretrained(decoder_name,padding_side="left")

if os.path.exists(model_path):
    print("Loading model from checkpoint...")
    model = GPT2LMHeadModel.from_pretrained(checkpoint_dir, use_safetensors=True)
else:
    print("Initializing new model...")
    raise FileNotFoundError(f"Checkpoint not found in {checkpoint_dir}. Ensure the checkpoint exists before resuming.")
    
#model = GPT2LMHeadModel.from_pretrained(decoder_name)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model.to(device)
tokenizer.add_special_tokens({'pad_token': '[PAD]'})
model.resize_token_embeddings(len(tokenizer),mean_resizing=False)
model.config.pad_token_id = tokenizer.pad_token_id

#dataset
train_dataset = SingleStage(config['dataset']['train_path'], tokenizer)
train_data_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
val_dataset = SingleStage(config['dataset']['val_path'], tokenizer)
val_data_loader = DataLoader(val_dataset, batch_size=64, shuffle=True)

optimizer = torch.optim.Adam(model.parameters(), lr=training_args['learning_rate'])

lowest_val_loss = float('inf')
for epoch in range(training_args['epochs']):
    # Training
    model.train()
    train_loss = 0
    train_tokens = []
    for batch in tqdm(train_data_loader):
        input_ids = batch['input_ids'].to(device)
        input_ids = input_ids.squeeze(1)
        #attention_mask = batch['attention_mask'].to(device)
        attention_mask = (input_ids != tokenizer.pad_token_id).long().to(device)
        #print("Attention mask shape",attention_mask.shape)
        #print("input id shape",input_ids.shape)
        outputs = model(input_ids, attention_mask=attention_mask, labels=input_ids)
        loss = outputs.loss
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        train_loss += loss.item()
        generated_ids = model.generate(input_ids, repetition_penalty=2.0,max_length=config['dataset']['max_story_length'], num_return_sequences=1,attention_mask=attention_mask,pad_token_id=tokenizer.pad_token_id)
        generated_text = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)
        generated_tokens = [tokenizer.tokenize(text) for text in generated_text]
        for tokens in generated_tokens:
            train_tokens.extend(tokens)

    train_avg_loss = train_loss / len(train_data_loader)
    train_perplexity = calculate_perplexity(train_avg_loss)
    train_distinct_1 = calculate_distinct_n(train_tokens, 1)
    train_distinct_2 = calculate_distinct_n(train_tokens, 2)
    
    # Validation
    model.eval()
    val_loss = 0
    val_tokens = []
    for batch in tqdm(val_data_loader):
        input_ids = batch['input_ids'].to(device)
        input_ids = input_ids.squeeze(1)
        attention_mask = (input_ids != tokenizer.pad_token_id).long().to(device)

        outputs = model(input_ids, attention_mask=attention_mask, labels=input_ids)
        loss = outputs.loss
        
        
        val_loss += loss.item()
        generated_ids = model.generate(input_ids, repetition_penalty=2.0,max_length=config['dataset']['max_story_length'], num_return_sequences=1,attention_mask=attention_mask,pad_token_id=tokenizer.pad_token_id)
        generated_text = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)
        generated_tokens = [tokenizer.tokenize(text) for text in generated_text]
        for tokens in generated_tokens:
            val_tokens.extend(tokens)

    val_avg_loss = val_loss / len(val_data_loader)
    val_perplexity = calculate_perplexity(val_avg_loss)
    val_distinct_1 = calculate_distinct_n(val_tokens, 1)
    val_distinct_2 = calculate_distinct_n(val_tokens, 2)
    
    #Early stopping
    if val_avg_loss < lowest_val_loss:
        model.save_pretrained(config['model']['save_path']+'single_stage')
        

    print(f"Epoch {epoch+1} train perplexity: {train_perplexity:.3f} train distinct-1: {train_distinct_1:.3f} train distinct-2: {train_distinct_2:.3f}")
    print(f"Epoch {epoch+1} val perplexity: {val_perplexity:.3f} val distinct-1: {val_distinct_1:.3f} val distinct-2: {val_distinct_2:.3f}")
    print(f"Epoch {epoch+1} done.")

