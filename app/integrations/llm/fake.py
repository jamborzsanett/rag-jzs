class FakeLLM:
    def __init__(self, answer: str = "Ez egy teszt válasz.", error: Exception | None = None) -> None:
        self.answer = answer
        self.error = error
        self.calls: list[tuple[str, str]] = []

    async def generate(self, system: str, prompt: str) -> str:
        self.calls.append((system, prompt))
        if self.error is not None:
            raise self.error
        return self.answer
