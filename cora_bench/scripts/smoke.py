"""Run a small BM25 packing example without a GPU, credentials, or paid APIs."""

import json

from cora_bench.budgets.token_budget import Tokenizer
from cora_bench.config.schema import RunConfig
from cora_bench.experiment.packing import SourceUnionEnforcer, validate_source_union
from cora_bench.methods.registry import get_selector


def main():
    tokenizer = Tokenizer()
    context = (
        "The library opens at nine. " * 70
        + "Mira placed the red key in the kitchen. "
        + "The park closes at sunset. " * 70
    )
    config = RunConfig(
        experiment_name="smoke",
        dataset="synthetic",
        method="bm25",
        reader="none",
        chunk_size_tokens=64,
        chunk_overlap_tokens=16,
    )
    selector = get_selector("bm25", config, tokenizer)
    selector.enforcer = SourceUnionEnforcer(tokenizer)
    selector.enforcer.prepare(context)
    result = selector.select("Where did Mira place the red key?", context, 128)
    validate_source_union(context, result.text, selector.enforcer.last_spans, tokenizer)
    print(
        json.dumps(
            {
                "source_tokens": tokenizer.count(context),
                "selected_tokens": tokenizer.count(result.text),
                "budget": 128,
                "selected_text": result.text,
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
