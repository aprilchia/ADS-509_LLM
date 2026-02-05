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

## Fine-Tuning Labels

- Randomly sample 100 comments from all sources to independently label
- Compare then use LM to come up with the rest
- Spot check 100 LLM labeled samples
- Do another round if need be

### Labels

**Argumentative**
- Comment makes specific claims or predictions
- Uses anecdotes or scenarios to make their case without relying on ad hominems

**Informational**
- Sharing information relevant to the main post or discussion
- Low on emotional affect
- Referring or linking other sources
- Responding to other commenter questions with more information

**Interrogative**
- Asking questions meant for others to respond
- Not rhetorical or clarifying questions

**Affective**
- Opinion based
- Emotional affect
- Sarcasm/jokes

**Neutral**
- Clarifying questions
- meta or off-topic
- Random comments from the peanut gallery that don't fall neatly into the other categories