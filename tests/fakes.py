from pathlib import Path


class FakeEvaluator:
    """Drop-in for core.evaluate.Evaluator that returns canned transcripts.

    Files inside an 'original' directory are transcribed as the reference,
    everything else as the hypothesis.
    """

    def __init__(self, endpoint, defaults):
        self.name = endpoint.get("name") or "fake"
        self.model = endpoint["model"]
        self.ref_text = endpoint.get("ref", "hello world")
        self.hyp_text = endpoint.get("hyp", "hello world")
        self.hyp_empty = endpoint.get("hyp_empty", False)
        self.latency = endpoint.get("latency", 0.1)

    def transcribe(self, path):
        if Path(path).parent.name == "original":
            return self.ref_text, self.latency
        text = "" if self.hyp_empty else self.hyp_text
        return text, self.latency