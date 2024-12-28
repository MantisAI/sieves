from outlines import generate, models
from pydantic import BaseModel

model = models.transformers("gpt2")
generator = generate.choice(model, ["skirt", "dress", "pen", "jacket"])
answer = generator("Pick the odd word out: skirt, dress, pen, jacket")


class User(BaseModel):
    name: str
    last_name: str
    id: int


generator = generate.json(model, User)
result = generator("Create a user profile with the fields name, last_name and id")
print(result)

x = 3
