from typing import List
from pydantic import BaseModel, Field


class StartupReview(BaseModel):
    startup_name: str = Field(description="Name of the startup")
    funding:str = Field(description="How much the startup has raised")
    summary: str = Field(description="What the startup does")
    founders:str = Field(description="Names of the founders")

