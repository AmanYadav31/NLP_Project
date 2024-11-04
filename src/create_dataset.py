import numpy as np  # linear algebra
import pandas as pd
import os
import csv
from tqdm import tqdm
from rake_nltk import Rake
from sumy.parsers.plaintext import PlaintextParser
from sumy.nlp.tokenizers import Tokenizer
from sumy.summarizers.lex_rank import LexRankSummarizer
import nltk

nltk.download('stopwords')
nltk.download('punkt_tab')
# Using raw string to handle Windows path with backslashes
for dirname, _, filenames in os.walk(r'/home/btech/2021/shreya.malik21b/nlp_project/NLP_Project/writingPrompts/'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

DIR = r"/home/btech/2021/shreya.malik21b/nlp_project/NLP_Project/writingPrompts/"
data = [os.path.join(DIR,"train"), os.path.join(DIR,"test"), os.path.join(DIR,"valid")]
print(data)

NUM_WORDS = 1000

# Function to create combined dataset
def create_combined_data(name_id, output_file):
    combined_data = []
    # Open the source and target files with utf-8 encoding
    with open(data[name_id] + ".wp_source", encoding='utf-8') as fp, \
         open(data[name_id] + ".wp_target", encoding='utf-8') as ft:
        
        stories = ft.readlines()
        prompts = fp.readlines()
        assert len(prompts) == len(stories), "Source and target files should have the same number of lines."
        
        for i in range(len(stories)):
            prompt = prompts[i].rstrip()
            story = " ".join(stories[i].split()[:NUM_WORDS])
            combined_data.append([prompt, story])
    
    # Write combined data to CSV file
    with open(output_file, 'w', newline='', encoding='utf-8') as csvfile:
        csv_writer = csv.writer(csvfile)
        csv_writer.writerow(['Prompt', 'Story'])
        for row in combined_data:
            csv_writer.writerow(row)
    print(f'CSV file created: {output_file}')

# Create train, test, and validation CSV files
create_combined_data(0, 'combined_traindata.csv')
create_combined_data(1, 'combined_testdata.csv')
create_combined_data(2, 'combined_valdata.csv')

# Processing keywords and summaries
csv_file = 'combined_traindata.csv'
r = Rake()
triplets = []
summarizer = LexRankSummarizer()

with open(csv_file, 'r', newline='', encoding='utf-8') as csvfile:
    csv_reader = csv.reader(csvfile)
    next(csv_reader)  # Skip header row
    for row in tqdm(csv_reader):
        prompt, story = row[0], row[1]

        # Extract keywords using Rake
        r.extract_keywords_from_text(story)
        keywords = r.get_ranked_phrases()[:10]  # Top 10 keywords

        # Generate summary using LexRank
        parser = PlaintextParser.from_string(story, Tokenizer("english"))
        abstract_sentences = summarizer(parser.document, 3)  # Get top 3 sentences
        abstract = " ".join([str(sentence) for sentence in abstract_sentences])

        # Add prompt, keywords, and summary to triplets
        triplet = (prompt, " ".join(keywords), abstract)
        triplets.append(triplet)

# Write new data with keywords and summaries to CSV
new_csv_file = 'new_traindata.csv'
with open(new_csv_file, 'w', newline='', encoding='utf-8') as new_csvfile:
    csv_writer = csv.writer(new_csvfile)
    csv_writer.writerow(['Prompt', 'Outline', 'Story'])
    for triplet in triplets:
        csv_writer.writerow(triplet)

print(f"New CSV file created: {new_csv_file}")

# Processing keywords and summaries
csv_file = 'combined_valdata.csv'
r = Rake()
triplets = []
summarizer = LexRankSummarizer()

with open(csv_file, 'r', newline='', encoding='utf-8') as csvfile:
    csv_reader = csv.reader(csvfile)
    next(csv_reader)  # Skip header row
    for row in tqdm(csv_reader):
        prompt, story = row[0], row[1]

        # Extract keywords using Rake
        r.extract_keywords_from_text(story)
        keywords = r.get_ranked_phrases()[:10]  # Top 10 keywords

        # Generate summary using LexRank
        parser = PlaintextParser.from_string(story, Tokenizer("english"))
        abstract_sentences = summarizer(parser.document, 3)  # Get top 3 sentences
        abstract = " ".join([str(sentence) for sentence in abstract_sentences])

        # Add prompt, keywords, and summary to triplets
        triplet = (prompt, " ".join(keywords), abstract)
        triplets.append(triplet)

# Write new data with keywords and summaries to CSV
new_csv_file = 'new_valdata.csv'
with open(new_csv_file, 'w', newline='', encoding='utf-8') as new_csvfile:
    csv_writer = csv.writer(new_csvfile)
    csv_writer.writerow(['Prompt', 'Outline', 'Story'])
    for triplet in triplets:
        csv_writer.writerow(triplet)

print(f"New CSV file created: {new_csv_file}")

# Processing keywords and summaries
csv_file = 'combined_testdata.csv'
r = Rake()
triplets = []
summarizer = LexRankSummarizer()

with open(csv_file, 'r', newline='', encoding='utf-8') as csvfile:
    csv_reader = csv.reader(csvfile)
    next(csv_reader)  # Skip header row
    for row in tqdm(csv_reader):
        prompt, story = row[0], row[1]

        # Extract keywords using Rake
        r.extract_keywords_from_text(story)
        keywords = r.get_ranked_phrases()[:10]  # Top 10 keywords

        # Generate summary using LexRank
        parser = PlaintextParser.from_string(story, Tokenizer("english"))
        abstract_sentences = summarizer(parser.document, 3)  # Get top 3 sentences
        abstract = " ".join([str(sentence) for sentence in abstract_sentences])

        # Add prompt, keywords, and summary to triplets
        triplet = (prompt, " ".join(keywords), abstract)
        triplets.append(triplet)

# Write new data with keywords and summaries to CSV
new_csv_file = 'new_testdata.csv'
with open(new_csv_file, 'w', newline='', encoding='utf-8') as new_csvfile:
    csv_writer = csv.writer(new_csvfile)
    csv_writer.writerow(['Prompt', 'Outline', 'Story'])
    for triplet in triplets:
        csv_writer.writerow(triplet)

print(f"New CSV file created: {new_csv_file}")
