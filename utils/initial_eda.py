import pandas as pd
import textstat
import matplotlib.pyplot as plt

def initial_eda(main, comments, source):
    print(f"=== EDA Report for {source} ===\n\n")
    print("=== Original Post Columns ===")
    print(list(main.columns))

    print("\n=== Comments Columns ===")
    print(list(comments.columns))

    print(f"\nShape of Main df: {main.shape}")
    print(f"Shape of Comments df: {comments.shape}")

    print("\n=== Missing Values ===")
    print(pd.DataFrame(comments.isna().sum()).rename(columns={0: "NA Values"}), "\n")
    print(pd.DataFrame(main.isna().sum()).rename(columns={0: "NA Values"}))

    comments = comments.fillna("")
    main = main.fillna("")

    comments.groupby("thread_link").size().hist(bins=40, edgecolor='black')
    plt.title("Number of Comments Distribution")
    plt.show()

    comments['comment_length'] = comments['comment_text'].apply(len)
    comments['comment_length'].hist(bins=100, edgecolor='black')
    plt.title('Comment Length Distribution')
    plt.show()

    print(f"Average Comment Length: {comments['comment_length'].mean():.2f}")

    syls = comments_df['comment_text'].fillna("").apply(textstat.syllable_count)

    # lexicon defaults to removing punct

    lexicon_count = comments_df['comment_text'].apply(textstat.lexicon_count)

    sentence_count = comments_df['comment_text'].apply(textstat.sentence_count)

    # this is words with 3 or more syllables

    poly_count = comments_df['comment_text'].apply(textstat.polysyllabcount)

    mono_count = comments_df['comment_text'].apply(textstat.monosyllabcount)

    data_list = [syls, lexicon_count, sentence_count, poly_count, mono_count]
    labels = ['Syllables', 'Lexicon Count', 'Sentence Count', 'Polysyllables', 'Monosyllables']

    fig, ax = plt.subplots(nrows = 2, ncols = 3)

    for idx, axis in enumerate(ax.flat[:-1]):
        axis.hist(data_list[idx], bins=50, edgecolor='black')
        axis.set_xlabel(labels[idx])
    plt.tight_layout()
    plt.show()