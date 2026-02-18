## Column Standardization

**Main Post Dataframe**

- unique id -> `post_id`
- title of post -> `post_title`
- original author or channel of post -> `post_author`
- time of post -> `created_at`
- number of comments on post -> `comment_count`
- original source of post (reddit, bluesky etc) -> `source`

**Comments Dataframe**

- id linking to main post -> `post_id`
- the commenter -> `username`
- date comment posted -> `created_at`
- text of comment itself `comment_text`

## EDA

- In main notebook, include summary/basic stats for each source, then 1-2 interesting visuals
- Keep full EDA for each in separate appendix

## Fine-Tuning Labels

- Randomly sample 100 comments from all sources to independently label and compare
- Use language models to label the rest
    - Used 10 true labels in the prompt as positive examples, and used 10 negative examples with correct label
- Use 3 LM models to label the rest and compare results
- Spot check 100 LLM labeled samples
- Kept majority vote labels and set aside the rest

### Labels

**Argumentative**
- Makes specific claims, predictions, or assertions supported by reasoning
- Uses evidence, anecdotes, or scenarios to build a case
- The key distinction from Opinion: there's an attempt to *persuade* or *explain why*, not just state a position

**Informational**
- Shares facts, data, links, or context relevant to the discussion
- Low emotional affect — the comment is trying to *inform*, not convince or react
- Includes answering another commenter's question with factual content
- The key distinction from Argumentative: presenting information without advocating for a position

**Opinion**
- States a value judgment, stance, or take without substantial reasoning
- "This is good/bad/wrong/overrated" — the comment *asserts* but doesn't *argue*
- The key distinction from Argumentative: no real attempt to persuade or support the claim
- The key distinction from Expressive: the comment is making a point, not just reacting

**Expressive**
- Emotional reactions, sarcasm, jokes, venting, exclamations
- The comment is primarily *expressing feeling* rather than making a point
- Includes performative agreement/disagreement ("THIS," "lol exactly," "what a joke")
- The key distinction from Opinion: no identifiable stance being taken, just affect

**Neutral**
- Clarifying or rhetorical questions, meta-commentary, off-topic remarks
- Comments that don't clearly fit the other four categories
- Includes simple factual questions directed at other commenters

# Preprocessing Steps

## Fine-tuning models
- Converting emoji's to written text to preserve the emotion conveyed
- Converting web address to a URL token because that information could convey something about the nature of the comment

## Traditional machine learning
- Removing all punctuation except ! and ? because those convey meaning and change the nature of a comment
- Removing stop words using the NLTK set
- Lemmatizing text prior, useful for feature reduction in TF-IDF without losing meaning