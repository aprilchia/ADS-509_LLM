import pandas as pd
from datasets import Dataset, DatasetDict, Features, ClassLabel, Value


def train_prep(df: pd.DataFrame, test1: float=0.3, test2: float=0.5):
    # Label mapping
    label2id = {label: idx for idx, label in enumerate(df['source'].unique())}
    id2label = {idx: label for label, idx in label2id.items()}

    # Prepare dataset
    dataset = Dataset.from_dict({
        "text": df['comment_text'].tolist(),
        "label": [label2id[source] for source in df['source']]
    })

    labels = list(label2id.keys())
    class_features = Features({
        'text': Value('string'),
        'label': ClassLabel(names=labels) # This creates the id2label mapping
    })

    # Cast your dataset to these features
    dataset = dataset.cast(class_features)

    # Split dataset 3-ways and combine back into  one
    split_dataset = dataset.train_test_split(test_size=test1, seed=10)
    test_valid = split_dataset['test'].train_test_split(test_size=test2, seed=10)

    final_dataset = DatasetDict({
        "train": split_dataset['train'],
        "test": test_valid['test'],
        "valid": test_valid['train']
    })

    return final_dataset, label2id, id2label