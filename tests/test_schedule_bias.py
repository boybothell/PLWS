from plws.schedule_bias import (
    HORIZON,
    PEAK,
    IncrementalSequenceCounter,
    continuation_bias,
    count_bias,
    count_completed_sequences,
    encode_ban_sequences,
    penalized_token_ids,
)


def test_front_fades_to_zero_and_back_ends_at_negative_infinity() -> None:
    assert continuation_bias("front", 0) == -PEAK
    assert continuation_bias("back", 0) == 0.0
    assert continuation_bias("front", HORIZON // 2) == -PEAK / 2
    assert continuation_bias("back", HORIZON // 2) == -PEAK / 2
    assert continuation_bias("front", HORIZON - 1) == -PEAK / HORIZON
    assert continuation_bias("back", HORIZON - 1) == -PEAK * (HORIZON - 1) / HORIZON
    assert continuation_bias("front", HORIZON) == 0.0
    assert continuation_bias("front", HORIZON + 100) == 0.0
    assert continuation_bias("back", HORIZON) == float("-inf")
    assert continuation_bias("back", HORIZON + 100) == float("-inf")


def test_count_bias_uses_the_same_trace_as_its_scale() -> None:
    assert count_bias(0, 10) == 0.0
    assert count_bias(11, 10) == -PEAK / 2
    assert count_bias(1, 0) == -PEAK / 2
    assert count_bias(1000, 0) == -PEAK * 1000 / 1001


def test_completed_core_sequences_count_each_end_position_once() -> None:
    sequences = [[7], [3, 4, 5]]
    assert count_completed_sequences([], sequences) == 0
    assert count_completed_sequences([3, 4], sequences) == 0
    assert count_completed_sequences([1, 3, 4, 5], sequences) == 1
    assert count_completed_sequences([7, 3, 4, 5, 7], sequences) == 3
    assert count_completed_sequences([7, 7], [[7], [7, 7]]) == 2


def test_incremental_counter_matches_full_scan_on_append_and_rollback() -> None:
    sequences = [[7], [3, 4, 5], [9, 5], [7, 7]]
    counter = IncrementalSequenceCounter(sequences)
    tokens: list[int] = []
    for token_id in [7, 3, 4, 5, 7, 7, 2, 9, 5]:
        tokens.append(token_id)
        assert counter.update(tokens) == count_completed_sequences(tokens, sequences)
        # Repeated calls for the same decoding position must not double count.
        assert counter.update(tokens) == count_completed_sequences(tokens, sequences)

    rolled_back = [7, 3, 4]
    assert counter.update(rolled_back) == count_completed_sequences(
        rolled_back, sequences
    )
    changed_branch = [7, 3, 4, 5, 9, 5]
    assert counter.update(changed_branch) == count_completed_sequences(
        changed_branch, sequences
    )


def test_incremental_counter_handles_multi_token_jump() -> None:
    sequences = [[7], [3, 4, 5]]
    counter = IncrementalSequenceCounter(sequences)
    tokens = [7, 3, 4, 5, 7]
    assert counter.update(tokens) == 3


def test_penalized_ids_fire_once_and_only_on_a_matched_prefix() -> None:
    sequences = [[7], [3, 4, 5], [9, 5]]
    assert penalized_token_ids([], sequences) == [7]
    assert penalized_token_ids([3], sequences) == [7]
    assert penalized_token_ids([1, 3, 4], sequences) == [7, 5]
    assert penalized_token_ids([9], sequences) == [7, 5]
    assert penalized_token_ids([3, 4], sequences) == [7, 5]


class _Tokenizer:
    def encode(self, text: str, add_special_tokens: bool = False) -> list[int]:
        del add_special_tokens
        table = {
            "Wait": [1],
            " Wait": [2],
            "Alternatively": [3, 4],
            " Alternatively": [9, 9, 9],
            "Hmmmm": [5],
            " Hmmmm": [5, 6],
            "": [],
        }
        return list(table.get(text, [8]))


def test_encode_ban_sequences_matches_vllm_bad_words_filter() -> None:
    sequences = encode_ban_sequences(
        ["Wait", "Wait", " Wait", "", "Alternatively", " Hmmmm"],
        _Tokenizer(),
    )
    # Spaced "Alternatively" and "Hmmmm" change the token count, so vLLM
    # drops them. "Wait" / " Wait" stay because they are the same length.
    assert sequences == [[1], [2], [3, 4], [5]]
