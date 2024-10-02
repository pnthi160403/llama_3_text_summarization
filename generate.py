import torch
from .search_algos import (
    TYPE_SEARCH,
    DIVERSE_BEAM_SEARCH,
    SearchNode,
)
import sys
# length penalty
def sequence_length_penalty(length: int, alpha: float=0.6) -> float:
    return ((5 + length) / (5 + 1)) ** alpha

# beam search
def generate(
    model,
    config,
    beam_size,
    tokenizer_src,
    tokenizer_tgt,
    src,
    top_cosine_similarity_indices=None,
):
    # Search Module
    # special token id
    device = config["device"]
    max_len = config["max_len"]
    n_gram = config["n_gram_search"]

    sos_token_id = tokenizer_src.token_to_id("<s>")
    eos_token_id = tokenizer_src.token_to_id("</s>")
    pad_token_id = tokenizer_src.token_to_id("<pad>")
    sep_token_id = tokenizer_src.token_to_id("<sep>")
    special_tokens = {
        "<s>": sos_token_id,
        "</s>": eos_token_id,
        "<pad>": pad_token_id,
        "<sep>": sep_token_id,
    }
    type_search = config["type_search"]
    vocab_size = tokenizer_tgt.get_vocab_size()
    search_module = TYPE_SEARCH[type_search](
        special_tokens=special_tokens,
        vocab_size=vocab_size,
        num_groups=config["num_groups_search"],
        diversity_strength=config["diversity_strength_search"],
        diversity_discount=config["diversity_discount_search"],
        candidate_multiple=config["candidate_multiple_search"],
        n_gram=n_gram,
        device=device,
        type_diversity_function=config["type_diversity_function"],
        top_cosine_similarity_indices=top_cosine_similarity_indices,
    )

    sos_token = torch.tensor([tokenizer_tgt.token_to_id("<s>")], dtype=torch.int64)
    sep_token = torch.tensor([tokenizer_tgt.token_to_id("<sep>")], dtype=torch.int64)

    enc_input_tokens = tokenizer_src.encode(src).ids
    initial_candidate = torch.cat(
        [
            sos_token,
            torch.tensor(enc_input_tokens, dtype=torch.int64),
            sep_token,
        ],
        dim=0,
    ).to(device)
  
    candidates = [SearchNode(
        eos_token_id=eos_token_id,
        pad_token_id=pad_token_id,
        sep_token_id=sep_token_id,
        initial_candidate=initial_candidate,
        device=device,
        tokenizer=tokenizer_src,
        max_len=max_len,
        n_gram=n_gram,
        step=initial_candidate.shape[0],
    )] * beam_size

    for step in range(initial_candidate.shape[0] - 1, max_len - 1):
        if all([candidate.stop_search() for candidate in candidates]):
            break
        new_candidates = []
        lprobs = []
        scores = None
        indices = []
        mask_stop_search = None
        # mask (batch_size, beam_size)
        for input_beam in range(beam_size):
            candidate = candidates[input_beam]
            if candidate.stop_search():
                # lprob (1, vocab_size)
                lprob = torch.zeros((1, vocab_size), dtype=torch.float32).to(device)
            else:
                if (step == initial_candidate.shape[0] - 1):
                    logits, loss = model(
                        inputs_ids=candidate.tgt.unsqueeze(0),
                    )
                else:
                    logits, loss = model(
                        inputs_ids=candidate.tgt[-1:].unsqueeze(0),
                        start_pos=step,
                    )
                # lprob (1, vocab_size)
                lprob = torch.nn.functional.log_softmax(logits[:, -1], dim=1)
                lprob = lprob / sequence_length_penalty(len(candidate.tgt), alpha=0.6)
            lprobs.append(lprob)

            if scores is None and candidate.scores is not None:
                scores = [candidate.scores.unsqueeze(0)]
            elif candidate.scores is not None:
                scores.append(candidate.scores.unsqueeze(0))

            if mask_stop_search is None:
                mask_stop_search = [torch.tensor(1- candidate.stop_search()).to(device).unsqueeze(0)]
            elif mask_stop_search is not None:
                mask_stop_search.append(torch.tensor(1 - candidate.stop_search()).to(device).unsqueeze(0))

            indices.append(candidate.indices.unsqueeze(0))
        
        # lprobs (batch_size, beam_size, vocab_size)
        lprobs = torch.cat(lprobs, dim=0).unsqueeze(0)
        
        # scores (batch_size, beam_size, step)
        if scores is not None:
            scores = torch.cat(scores, dim=0).unsqueeze(0)
        
        # indices (batch_size, beam_size, step)
        indices = torch.cat(indices, dim=0).unsqueeze(0)

        # mask_stop_search (batch_size, input_beam_size)
        if mask_stop_search is not None:
            mask_stop_search = torch.cat(mask_stop_search, dim=0).unsqueeze(0)

        scores, indices, beams = search_module.step(
            step=step,
            lprobs=lprobs,
            scores=scores,
            prev_indices=indices,
            mask_stop_search=mask_stop_search,
            original_batch_idxs=torch.tensor([0]).to(device),
        )

        for output_beam in range(config["candidate_multiple_search"] * beam_size):
            input_beam = beams[0][output_beam]
            # copy candidate
            candidate = candidates[input_beam].copy()
            candidate.step(
                score=scores[0][output_beam],
                indice=indices[0][output_beam],
            )
            new_candidates.append(candidate)

        # del all elements in candidates from memory
        del candidates
        
        # update candidates
        candidates = new_candidates

        # sort by score
        if config["type_search"] != DIVERSE_BEAM_SEARCH:
            candidates = sorted(new_candidates, key=lambda x: x.scores[-1], reverse=True)
            candidates = candidates[:beam_size]
            
    return sorted(new_candidates, key=lambda x: x.scores[-1], reverse=True)[:beam_size]