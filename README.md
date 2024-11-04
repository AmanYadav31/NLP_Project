# Consistency-Enhanced-Story-Generation

<div dir ='rtl'>
Aman Yadav(2101032)
</div dir='ltr'>
<div dir ='rtl'>
Pari tiwari(2101139)
</div dir='ltr'>
<div dir ='rtl'>
Shreya Malik(2101192)
</div dir='ltr'>


# Writing Prompts Data Processing

The code combines source and target data, extracts keywords, and generates abstracts for the stories. It is designed to assist in data preprocessing for natural language processing tasks.

## Getting Started

### Prerequisites

Before running the code, ensure to run the requirements.txt file:

```bash
pip install -r requirements.txt
```

### Usage

1. **Combining Source and Target Data**

   - Modify the `DIR` and `TARGET_DIR` variables in the code to specify the directory paths for your data files.
   - Run the code to combine source and target data for training, testing, and validation datasets.
   - CSV files named `combined_traindata.csv`, `combined_testdata.csv`, and `combined_valdata.csv` will be created.

2. **Extracting Keywords and Generating Abstracts**

   - The code uses the RAKE algorithm for keyword extraction and the LexRankSummarizer for abstract generation.
   - Ensure you have the CSV files (`combined_traindata.csv`, etc.) in the same directory as the code.
   - Run the code to process the data and generate (prompt, outline, story) triplets.
   - A new CSV file named `new_traindata.csv` will be created, containing the processed data.

3. **Adjust Hyperparameters:**
   - You can modify the hyperparameters in the config.yaml file as needed.
    
4. **To Run single stage story generation**
   - Update src/config.yml accordingly
   - Run *python single_stage.py* to train and validate the model.
   
5. **To Run double stage story generation**
   - Update src/config.yml accordingly
   - Run *python double_stage.py* to train and validate the model.

### Testing the Model

To test the models and view the results, open and run the outputs.ipynb file.

You can find the saved models by accessing the link : https://drive.google.com/file/d/1GpN81VY6iskawkvX57ryiiyWqi3s2BQd/view?usp=sharing
   

