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