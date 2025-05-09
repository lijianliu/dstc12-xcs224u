# Dialog System Technology Challenge 12 - Controllable Conversation Theme Detection track

Task description: [proposal paper](https://drive.google.com/file/d/1TkVYdcQenWH_MVXHGEX3O7BGMyTj5Xd_/view)

DSTC12 home: https://dstc12.dstc.community/tracks

## *** To participate in the track, please fill out the [registration form](https://forms.gle/URgeLuSUL95BHgWb6) ***

## *** Final results submission ***
Testset: [AppenTravel](/dstc12-data/AppenTravel)

Submit the final results via [this form](https://form.jotform.com/250896633081058) (1 submission per team)

## 1. Motivation  
Theme detection plays a critical role in analyzing conversations, particularly in domains like customer support, sales, and marketing. Automatically identifying and categorizing themes can significantly reduce the time spent on manual analysis of lengthy conversations.  

While theme detection is similar to dialogue intent detection, the key difference lies in its purpose. Intent detection is used in downstream dialogue systems where responses are predefined based on intent categories. In contrast, theme detection provides a summary of the conversation from the customer’s perspective, allowing for various surface forms and user-driven customizations.  

![Task diagram](/img/DSTC12_task_large.png)

### **Challenges for Participants:**  
- Designing a system that can **capture nuanced themes** beyond predefined intent categories.  
- Handling **multiple valid surface forms** for theme labels while maintaining consistency.  
- Balancing **style following and accuracy** in theme detection without compromising clarity.  

## 2. Proposed Task  
The goal of this track is to develop a **Controllable Theme Detection** system capable of:  
- Clustering unlabeled utterances into themes.  
- Generating concise and natural language labels for each theme.  
- Adapting theme granularity based on **user preferences** to support different levels of detail.  

To achieve this, participants will be provided with:  
- A dataset of raw conversation utterances.  
- User preferences indicating whether certain utterances should belong to the same theme.  
- A **Theme Label Writing Guideline** to ensure high-quality and consistent labels.   

## 3. Datasets  
This track builds upon the **NatCS dataset**, which contains customer support dialogues across multiple domains.  

### **Dataset Overview:**  
- **Training Domain:** Banking  
- **Development Domain:** Finance  
- **Testing Domain:** Undisclosed (for zero-shot evaluation)  

Participants will receive:  
1. **Utterances and dialogue context** – Some utterances may lack explicit details and require surrounding dialogue context for accurate labeling.  
2. **User preferences for clustering** – A set of utterance pairs with binary labels indicating whether they belong to the same theme.  
3. **Gold standard theme labels** – Available for training and development but **hidden for the test set** to assess generalization.  

### **Challenges for Participants:**  
- **Extracting meaningful themes** from natural conversations that may include ambiguity or informal language.  
- **Leveraging user preferences** to fine-tune theme granularity effectively.  
- **Handling out-of-domain test cases** with minimal prior knowledge.  

## 4. Evaluation  
The evaluation consists of two key components:  

### **4.1 Clustering Metrics**  
- **Normalized Mutual Information (NMI)** – Measures the agreement between predicted and reference clusters while ignoring permutations.  
- **Accuracy (ACC) Score** – Evaluates how well predicted clusters align with reference clusters using the Hungarian algorithm.  

### **4.2 Label Generation Metrics**  
- **Cosine Similarity** – Measures the semantic similarity between predicted and reference labels using Sentence-BERT embeddings.  
- **ROUGE Score** – Analyzes n-gram overlap, useful for comparing short, concise word sequences.  
- **LLM-Based Scoring** – Assesses label quality based on compliance with the provided **Theme Label Writing Guideline**.  
- **Human Evaluation (if applicable)** – Top-performing models may be manually reviewed to assess real-world usability.  

## 5. Theme Label Writing Guideline  
To maintain consistency and quality, participants must follow specific guidelines when generating theme labels.  

### **5.1 Exclusion of Unnecessary Words**  
- Labels should be **concise (2–5 words long)** and avoid:  
  - Articles (e.g., *the, a*).  
  - Auxiliary verbs (e.g., *is, have*).  
  - Pronouns (e.g., *he, she, it*).  
  - Demonstratives (e.g., *this, that*).  
  - Overly specific or context-sensitive words.  

### **5.2 Theme Labels as Verb Phrases**  
- Labels should be action-based, starting with a verb.  
- The verb should be in its **citation form** (e.g., *sign up* rather than *signing up*).  
- Avoid noun phrases and general claims (e.g., *report defective product* instead of *defective product*).  

### **5.3 Balancing Generality and Specificity**  
- Labels should provide **enough detail to be useful** but not be overly specific.  
- Example:  
  - ✅ **Good:** *schedule appointments*  
  - ❌ **Too General:** *ask about appointments*  
  - ❌ **Too Specific:** *schedule appointment for elderly parent*

--------------------

## Getting started
Setting up environment and installing packages:
```
conda create -n dstc12 python=3.11
conda activate dstc12
pip install -r requirements.txt --no-cache-dir
. ./set_paths.sh
```

## Getting familiar with the baseline code

Running theme detection
```
python scripts/run_theme_detection.py <dataset_file> <preferences_file> <result_dataset_with_predictions_file>
```

e.g. for Banking:

```
python scripts/run_theme_detection.py \
    dstc12-data/AppenBanking/all.jsonl \
    dstc12-data/AppenBanking/preference_pairs.json \
    appen_banking_predicted.jsonl
```

Running evaluation:

```
python scripts/run_evaluation.py <ground_truth_file> <predictions_file>
```

## Running the LLM
Some parts of logic used in this baseline use an LLM being run locally:

* theme labeling in `run_theme_detection.py`
* evaluation of theme labels against the Theme Label Guideline

We use `lmsys/vicuna-13b-v1.5` by default which we tested on 4x Nvidia V100's (16GB each). Please feel free to use a locally run model or an API that works best for you. In case of any questions, please feel free to contact the organizers e.g. via Github issues.

## Security

See [CONTRIBUTING](CONTRIBUTING.md#security-issue-notifications) for more information.

## License

This library is licensed under the CC-BY-NC-4.0 License.


