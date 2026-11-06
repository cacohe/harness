class Skill:
    """A skill to be used by the harness."""

    name: str
    description: str
    content: str

    def __init__(self, name: str, description: str, content: str):
        self.name = name
        self.description = description
        self.content = content
