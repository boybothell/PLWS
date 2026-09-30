"""vLLM processor that applies the post-lock CORE bias schedule."""

from __future__ import annotations

from vllm import SamplingParams
from vllm.v1.sample.logits_processor import AdapterLogitsProcessor

from plws.schedule_bias import (
    IncrementalSequenceCounter,
    continuation_bias,
    count_bias,
    penalized_token_ids,
)


class ScheduledCoreBiasLogitsProcessor(AdapterLogitsProcessor):
    """Add a time-varying bias to CORE token ids during one continuation."""

    @classmethod
    def validate_params(cls, params: SamplingParams):
        extra = params.extra_args or {}
        schedule = extra.get("bias_schedule")
        if schedule is None:
            return None
        if schedule not in {"front", "back", "count"}:
            raise ValueError(f"unknown bias_schedule {schedule!r}")
        horizon = extra.get("bias_horizon")
        peak = extra.get("bias_peak")
        sequences = extra.get("bias_token_seqs")
        if schedule != "count" and (
            not isinstance(horizon, int) or horizon <= 0
        ):
            raise ValueError(f"bias_horizon must be a positive int, got {horizon!r}")
        if schedule == "count":
            n_pre = extra.get("bias_n_pre")
            if isinstance(n_pre, bool) or not isinstance(n_pre, int) or n_pre < 0:
                raise ValueError(f"bias_n_pre must be an int >= 0, got {n_pre!r}")
        if isinstance(peak, bool) or not isinstance(peak, (int, float)) or peak <= 0:
            raise ValueError(f"bias_peak must be a positive number, got {peak!r}")
        if not isinstance(sequences, list) or not sequences:
            raise ValueError("bias_token_seqs must be a non-empty list of token-id lists")
        for sequence in sequences:
            if not isinstance(sequence, list) or not sequence:
                raise ValueError(f"bad bias token sequence {sequence!r}")
            if not all(isinstance(token_id, int) and token_id >= 0 for token_id in sequence):
                raise ValueError(f"bad bias token sequence {sequence!r}")
        return None

    def is_argmax_invariant(self) -> bool:
        return False

    def new_req_logits_processor(self, params: SamplingParams):
        extra = params.extra_args or {}
        schedule = extra.get("bias_schedule")
        if schedule is None:
            return None
        peak = float(extra["bias_peak"])
        sequences = extra["bias_token_seqs"]
        if schedule == "count":
            n_pre = int(extra["bias_n_pre"])
            counter = IncrementalSequenceCounter(sequences)

            def processor(past_tokens_ids, logits):
                # past_tokens_ids is the continuation only. The lock prefix
                # is already in the prompt, so its count arrives as n_pre.
                bias = count_bias(counter.update(past_tokens_ids), n_pre, peak)
                if bias == 0.0:
                    return logits
                token_ids = penalized_token_ids(list(past_tokens_ids), sequences)
                if not token_ids:
                    return logits
                logits[token_ids] += bias
                return logits

            return processor
        horizon = int(extra["bias_horizon"])

        def processor(past_tokens_ids, logits):
            bias = continuation_bias(
                schedule,
                len(past_tokens_ids),
                horizon=horizon,
                peak=peak,
            )
            if bias == 0.0:
                return logits
            token_ids = penalized_token_ids(list(past_tokens_ids), sequences)
            if not token_ids:
                return logits
            if bias == float("-inf"):
                logits[token_ids] = float("-inf")
            else:
                logits[token_ids] += bias
            return logits

        return processor
