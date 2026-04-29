from uuid import UUID

from pydantic import BaseModel


class CurrentUser(BaseModel):
    user_id: UUID
    email: str
