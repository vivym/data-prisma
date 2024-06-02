from enum import Enum

from pydantic import BaseModel, ConfigDict, Field

from .character import Creator, Tag


class Role(Enum):
    USER = 1

    CHARACTER = 2

    SYSTEM = 3


class Character(BaseModel):
    model_config: ConfigDict = ConfigDict(populate_by_name=True)

    id: str

    avatar: str

    name: str

    greeting: str

    visibility: int

    flying_nsfw: bool = Field(alias="flyingNsfw")

    nsfw: bool


class User(BaseModel):
    id: str

    name: str

    image: str


class Message(BaseModel):
    model_config: ConfigDict = ConfigDict(populate_by_name=True, use_enum_values=True)

    id: str

    role: Role

    content: str

    is_edited: bool = Field(alias="isEdited")

    channel: str

    channel_name: str = Field(alias="channelName")


class Emoji(BaseModel):
    model_config: ConfigDict = ConfigDict(populate_by_name=True, use_enum_values=True)

    id: int # TODO: make this an enum

    count: int

    has_interacted: bool = Field(alias="hasInteracted")


class Content(BaseModel):
    message: Message

    emojis: list[Emoji]


class CharacterProfile(BaseModel):
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


class CrushonMemory(BaseModel):
    model_config: ConfigDict = ConfigDict(populate_by_name=True)

    id: str

    title: str

    contents: list[Content]

    visibility: int

    created_at: int = Field(alias="createdAt")

    character: Character

    last_talk_time: int = Field(alias="lastTalkTime")

    user: User

    total_memories: int = Field(alias="totalMemories")

    memories: int

    is_like: bool = Field(alias="isLike")

    is_favorite: bool = Field(alias="isFavorite")

    total_emoji_count: int = Field(alias="totalEmojiCount")

    like_count: int = Field(alias="likeCount")

    favorite_count: int = Field(alias="favoriteCount")

    view_count: int = Field(alias="viewCount")

    share_count: int = Field(alias="shareCount")

    ban_types: list[str] = Field(alias="banTypes")

    is_own: bool = Field(alias="isOwn")

    profile: dict | None

    is_anonymity: bool = Field(alias="isAnonymity")

    review_state: int = Field(alias="reviewState")

    review_msg: str | None = Field(alias="reviewMsg")

    self_like_count: int = Field(alias="selfLikeCount")

    character_profile: CharacterProfile | None = None
