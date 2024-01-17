import time
import torch
from typing import List
from sympy.parsing.sympy_parser import parse_expr
from prompting.rewards import BaseRewardModel, BatchRewardOutput, RewardModelTypeEnum


class FloatDiffModel(BaseRewardModel):
    @property
    def name(self) -> str:
        return 'float_diff'

    @property
    def model_type(self) -> RewardModelTypeEnum:
        return RewardModelTypeEnum.WEIGHTED_REWARD

    def __init__(self, **kwargs):
        super().__init__()

    @staticmethod
    def extract_number(text):
        # loop over all words reversed and try to cast as a float, break when you find the first one
        for word in text.split()[::-1]:
            try:
                # Convert the string to a float
                return parse_expr(word.replace('$', ''))
            except ValueError:
                continue

    @staticmethod
    def math_score(reference, completion):
        # Extract all the digits and numerical expressions from the completion and take only the last one (assuming it's the answer)

        # Convert the string to a float
        pred = FloatDiffModel.extract_number(completion)
        if pred is None:
            return 0.0

        # Convert reference to float (this is okay because we already checked that the reference is a float)
        ref = float(reference)
        if pred == ref:
            return 1.0

        # Compute the difference
        diff = abs(ref - pred)/(ref + 1e-6)
        # Make sure the difference is between 0 and 1
        diff = min(abs(diff), 1)

        return 1.0 - diff


    def reward(self, reference: str, completions: List[str]) -> BatchRewardOutput:
        """Compute difference scores given a completion and reference pair."""
        rewards = []
        timings = []

        for completion in completions:
            t0 = time.time()
            reward = self.math_score(reference, completion)
            timings.append(time.time() - t0)
            rewards.append(reward)

        output = BatchRewardOutput(
            rewards = torch.FloatTensor(rewards),
            timings = torch.FloatTensor(timings),
            extra_info = {'type': 'math', },
        )
        return output
