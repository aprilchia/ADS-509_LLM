import numpy as np

def get_match(row):
    labels = [row['openai_label'], row['claude_labels'], row['gemini_label']]
    # yes if any 2 labels agree, no if all 3 are different
    return 'yes' if len(set(labels)) < 3 else 'no'

def get_label(row):
    labels = [row['openai_label'], row['claude_labels'], row['gemini_label']]
    counts = np.unique(labels, return_counts=True)
    idx = np.argmax(counts[1])
    label = str(counts[0][idx])
    return label