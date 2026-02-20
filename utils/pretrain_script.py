import pandas as pd
from datasets import Dataset, DatasetDict, Features, ClassLabel, Value


def train_prep(df: pd.DataFrame, label_column: str, test1: float=0.3, test2: float=0.5):
    """Prepare a labeled DataFrame for transformer fine-tuning.

    Converts a pandas DataFrame into a Hugging Face DatasetDict with train,
    validation, and test splits. Creates bidirectional label mappings and
    casts the dataset with ClassLabel features for proper label encoding.

    The default split produces roughly 70% train / 15% validation / 15% test.
    The test set is first carved out, then split in half to form valid and test.

    Args:
        df: DataFrame containing at least 'comment_text' and the label column.
        label_column: Name of the column holding string class labels.
        test1: Fraction of data reserved for the initial test+valid pool.
            Defaults to 0.3 (30%).
        test2: Fraction of the test+valid pool used as the final test set.
            Defaults to 0.5 (50% of 30% = 15% overall).

    Returns:
        tuple:
            - DatasetDict: Keys 'train', 'valid', 'test' with encoded features.
            - dict: label2id mapping from label string to integer index.
            - dict: id2label mapping from integer index to label string.
    """
    # Label mapping
    label2id = {label: idx for idx, label in enumerate(df[label_column].unique())}
    id2label = {idx: label for label, idx in label2id.items()}

    # Prepare dataset
    dataset = Dataset.from_dict({
        "text": df['comment_text'].tolist(),
        "label": [label2id[label] for label in df[label_column]]
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