import numpy as np

def get_match(row):
    """Check whether at least 2 of 3 LLM labels agree (majority match).

    Intended for use with DataFrame.apply() on a row that has 'openai_label',
    'claude_labels', and 'gemini_label' columns.

    Args:
        row: A pandas Series (DataFrame row) with keys 'openai_label',
            'claude_labels', and 'gemini_label'.

    Returns:
        str: 'yes' if at least 2 labels match, 'no' if all 3 are different.
    """
    labels = [row['openai_label'], row['claude_labels'], row['gemini_label']]
    # yes if any 2 labels agree, no if all 3 are different
    return 'yes' if len(set(labels)) < 3 else 'no'

def get_label(row):
    """Return the majority-vote label from 3 LLM annotations.

    Intended for use with DataFrame.apply() on rows where get_match returned
    'yes'. If all 3 labels are unique (no majority), returns the first label
    alphabetically due to numpy's behavior with ties.

    Args:
        row: A pandas Series (DataFrame row) with keys 'openai_label',
            'claude_labels', and 'gemini_label'.

    Returns:
        str: The label that appears most frequently among the three LLMs.
    """
    labels = [row['openai_label'], row['claude_labels'], row['gemini_label']]
    counts = np.unique(labels, return_counts=True)
    idx = np.argmax(counts[1])
    label = str(counts[0][idx])
    return label