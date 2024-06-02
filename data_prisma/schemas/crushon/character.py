from pydantic import BaseModel


class Creator(BaseModel):
    id: str

    name: str


class Tag(BaseModel):
    slug: str

    label: str

    description: str

    nsfw: bool


class Message(BaseModel):
    role: int

    content: str


class ExampleConversation(BaseModel):
    messages: list[Message]


class CrushonCharacterProfile(BaseModel):
    id: str

    name: str

    avatar: str

    creator: Creator

    definition_visibility: int

    description: str

    greeting: str

    personality: str

    scenario: str

    example_conversation: str

    new_example_conversation: ExampleConversation

    is_ban: bool

    visibility: int

    is_favorite: bool

    is_nsfw: bool

    likes: int

    share_count: int

    messages: int

    thumbs_up_count: int

    is_thumbs_up: bool

    tags: list[Tag]

    review_state: int

    review_msg: str

    update_at: int

    thumbnail_avatar: str

    flying_nsfw: bool
