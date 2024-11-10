import torch
import torch.nn as nn
import sys

class Search(nn.Module):
    def __init__(
        self,
        special_tokens: dict,
        vocab_size: int,
    ):
        super().__init__()
        self.special_tokens = special_tokens
        self.vocab_size = vocab_size

    def step(
        self,
        step: int,
        lprobs: torch.Tensor,
        scores: torch.Tensor,
    ):
        raise NotImplementedError
        
class BeamSearch(Search):
    def __init__(
        self,
        special_tokens: dict,
        vocab_size: int,
        candidate_multiple: int=2,
        **kwargs,
    ):
        super().__init__(
            special_tokens=special_tokens,
            vocab_size=vocab_size,
        )
        self.candidate_multiple = candidate_multiple

    def step(
        self,
        step: int,
        lprobs: torch.Tensor,
        scores: torch.Tensor=None,
        mask_stop_search: torch.Tensor=None,
        **kwargs,
    ):
        # lprobs (batch_size, input_beam_size, vocab_size)
        # scores (batch_size, input_beam_size, step + 1)
        # mask_stop_search (batch_size, input_beam_size)

        bsz, beam_size, vocab_size = lprobs.size()
        if step == 0:
            # at the first step all hypotheses are equally likely, so use
            # only the first beam
            lprobs = lprobs[:, 0:1, :].contiguous()
        else:
            # make probs contain cumulative scores for each hypothesis
            assert scores is not None
            lprobs = lprobs + scores[:, :, step].unsqueeze(-1)

        top_prediction = torch.topk(
            lprobs.view(bsz, -1),
            k=self.candidate_multiple * beam_size,
        )
        # scores_buf (batch_size, candidate_multiple * beam_size)
        scores_buf = top_prediction[0]
        # indices_buf (batch_size, candidate_multiple * beam_size)
        indices_buf = top_prediction[1]
        # Project back into relative indices and beams
        # beams_buf (batch_size, candidate_multiple * beam_size)
        beams_buf = torch.div(indices_buf, vocab_size, rounding_mode="trunc")
        indices_buf = indices_buf.fmod(vocab_size)
        mask = torch.gather(
            input=mask_stop_search,
            index=beams_buf,
            dim=-1,
        )
        indices_buf = indices_buf.masked_fill_(
            mask=mask == 0,
            value=self.special_tokens["<pad>"],
        )
        # At this point, beams_buf and indices_buf are single-dim and contain relative indices
        return scores_buf, indices_buf, beams_buf

HAMMING_CUMULATIVE_TYPE_DIVERSITY = "Hamming_Cumulative"
N_GRAM_TYPE_DIVERSITY = "N_Gram"
NEURAL_EMBEDDING_TYPE_DIVERSITY = "Neural_Embedding"

class DiverseBeamSearch(Search):
    def __init__(
        self,
        special_tokens: dict,
        vocab_size: int,
        num_groups: int,
        diversity_strength: float,
        diversity_discount: float=0.5,
        candidate_multiple: int=1,
        n_gram: int=1,
        device: str="cpu",
        type_diversity_function: str=HAMMING_CUMULATIVE_TYPE_DIVERSITY,
        top_cosine_similarity_indices: torch.Tensor=None,
        **kwargs,
    ):
        super().__init__(
            special_tokens=special_tokens,
            vocab_size=vocab_size,
        )
        self.TYPE_DIVERSITY_FUNCTION = {
            HAMMING_CUMULATIVE_TYPE_DIVERSITY: self.calc_overlap_type_hamming_cumulative,
            N_GRAM_TYPE_DIVERSITY: self.calc_overlap_type_n_gram,
            NEURAL_EMBEDDING_TYPE_DIVERSITY: self.calc_overlap_type_hamming_cumulative,
        }
        self.num_groups = num_groups
        self.diversity_strength = -diversity_strength
        self.beam = BeamSearch(
            special_tokens=special_tokens,
            vocab_size=vocab_size,
            candidate_multiple=candidate_multiple,
        )
        self.diversity_discount = diversity_discount
        self.candidate_multiple = candidate_multiple
        self.n_gram = n_gram
        self.device = device
        self.type_diversity_function = type_diversity_function
        if self.type_diversity_function == N_GRAM_TYPE_DIVERSITY and self.n_gram <= 1:
            raise ValueError("N-gram diversity requires n_gram > 1")
        self.group_overlap = None
        self.top_cosine_similarity_indices = top_cosine_similarity_indices

    def transform_tensor(
        self,
        tensor: torch.Tensor,
        n_gram: int=1,
    ):
        if tensor.size(-1) < n_gram:
            return None
        transformed_tensor = tensor.unfold(dimension=-1, size=n_gram, step=1)
        return transformed_tensor

    def calc_overlap_type_n_gram(
        self,
        indices: torch.Tensor,
        **kwargs,
    ):
        # indices (batch_size, mini_beam_size, num_groups, step + 2)
        bsz, mini_beam_size, num_groups, step = indices.size()
        if step < self.n_gram:
            return None

        # indices_n_gram (batch_size, mini_beam_size, num_groups, step + 2) -> (batch_size, mini_beam_size, num_groups, step + 3 - n_gram, n_gram)
        indices_n_gram = self.transform_tensor(
            tensor=indices,
            n_gram=self.n_gram,
        )
        if indices_n_gram is None:
            return None
        # (batch_size, mini_beam_size, num_groups, step + 3 - n_gram, n_gram)
        mask_indices_n_gram = (indices_n_gram != self.special_tokens["<pad>"]).type(torch.int64)
        
        # last_indices_n_gram (batch_size, mini_beam_size, num_groups, n_gram)
        last_indices_n_gram = indices_n_gram[:, :, :, -1, :]
        # (batch_size, mini_beam_size, num_groups, n_gram) -> (batch_size, mini_beam_size, num_groups, step + 3 - n_gram, n_gram)
        last_indices_n_gram = last_indices_n_gram.unsqueeze(-2).repeat(1, 1, 1, indices_n_gram.size(-2), 1)
        # mask_last_indices_n_gram (batch_size, mini_beam_size, num_groups, step + 3 - n_gram, n_gram)
        mask_last_indices_n_gram = (last_indices_n_gram != self.special_tokens["<pad>"]).type(torch.int64)

        # overlap (batch_size, mini_beam_size, num_groups, num_groups, step + 3 - n_gram, n_gram)
        overlap = (last_indices_n_gram.unsqueeze(2) == indices_n_gram.unsqueeze(3)).int()
        # overlap_mask (batch_size, mini_beam_size, num_groups, num_groups, step + 3 - n_gram, n_gram)
        overlap_mask = mask_last_indices_n_gram.unsqueeze(2) & mask_indices_n_gram.unsqueeze(3)

        # overlap = overlap * overlap_mask
        overlap = overlap.masked_fill_(
            mask=overlap_mask == 0,
            value=0,
        )
        
        # overlap (batch_size, mini_beam_size, num_groups, num_groups, step + 3 - n_gram)
        overlap = torch.sum(overlap, dim=-1)
        # overlap (batch_size, mini_beam_size, num_groups, num_groups, step + 3 - n_gram)
        overlap = (overlap == self.n_gram).int()
        # overlap (batch_size, mini_beam_size, num_groups, num_groups)
        overlap = torch.sum(overlap, dim=-1)
        # overlap (batch_size, num_groups, num_groups)
        overlap = torch.sum(overlap, dim=1)
        return overlap

    def calc_overlap_type_hamming_cumulative(
        self,
        indices: torch.Tensor,
        **kwargs,
    ):
        # last_indices (batch_size, mini_beam_size, num_groups)
        last_indices = indices[:, :, :, -1]
        # mask_last_indices (batch_size, mini_beam_size, num_groups)
        mask_last_indices = (last_indices != self.special_tokens["<pad>"]).type(torch.int64)

        # overlap (batch_size, mini_beam_size, num_groups, num_groups)
        overlap = (last_indices.unsqueeze(2) == last_indices.unsqueeze(3)).int()
        # overlap_mask (batch_size, mini_beam_size, num_groups, num_groups)
        overlap_mask = mask_last_indices.unsqueeze(2) & mask_last_indices.unsqueeze(3)
        # overlap = overlap * overlap_mask
        overlap = overlap.masked_fill_(
            mask=overlap_mask == 0,
            value=0,
        )
        # overlap (batch_size, num_groups, num_groups)
        overlap = torch.sum(
            input=overlap,
            dim=1,
        )
        return overlap

    def step(
        self,
        step: int,
        lprobs: torch.Tensor,
        mask_stop_search: torch.Tensor,
        scores: torch.Tensor=None,
        prev_indices: torch.Tensor=None,
        original_batch_idxs: torch.Tensor=None,
        **kwargs,
    ):
        # lprobs: (batch_size, input_beam_size, vocab_size)
        # mask_stop_search: (batch_size, input_beam_size)
        # scores: (batch_size, input_beam_size, step + 1)
        # prev_indices: (batch_size, input_beam_size, step + 1)
        # original_batch_idxs: (batch_size,)
        bsz, input_beam_size, vocab_size = lprobs.size()
        if input_beam_size % self.num_groups != 0:
            raise ValueError(
                "DiverseBeamSearch requires --beam to be divisible by the number of groups"
            )
        mini_beam_size = input_beam_size // self.num_groups
        if input_beam_size % self.num_groups != 0:
            raise ValueError(
                "DiverseBeamSearch requires --beam to be divisible by the number of groups"
            )
        
        # (batch_size, mini_beam_size, num_groups)
        mask_stop_search = mask_stop_search.view(bsz, mini_beam_size, self.num_groups)

        # indices (batch_size, mini_beam_size, num_groups, step + 2)
        indices = None
        
        # diversity_buf (batch_size, vocab_size)
        diversity_buf = torch.zeros(bsz, vocab_size, dtype=torch.float32, device=lprobs.device)

        scores_G, beams_G = [], []

        for g in range(self.num_groups):
            # lpobbs_g: (batch_size, mini_beam_size, vocab_size)
            lprobs_g = lprobs[:, g :: self.num_groups, :]
            # scores_g: (batch_size, mini_beam_size, step + 1)
            scores_g = scores[:, g :: self.num_groups, :]
            # mask_stop_search_g: (batch_size, mini_beam_size)
            mask_stop_search_g = mask_stop_search[:, :, g]
            # prev_indices_g: (batch_size, mini_beam_size, step + 1)
            prev_indices_g = prev_indices[:, g :: self.num_groups, :]

            diversity_buf.zero_()
            # apply diversity penalty
            if g > 0:
                # HAMMING_CUMULATIVE_TYPE_DIVERSITY
                if self.type_diversity_function is HAMMING_CUMULATIVE_TYPE_DIVERSITY:
                    if self.group_overlap is not None:
                        # penatly_val (batch_size, g)
                        penalty_val = 1 + self.group_overlap[original_batch_idxs, g, :g]
                        # penatly_val (batch_size, 1, g)
                        penalty_val = penalty_val.unsqueeze(1)
                    else:
                        penalty_val = torch.ones(bsz, 1, 1)
                    # indices_ (batch_size, mini_beam_size, g)
                    indices_ = indices[:, :, :g, -1]
                    # penalty_val (batch_size, 1, g)
                    # diversity_buf (batch_size, vocab_size)
                    diversity_buf.scatter_add_(
                        index=indices_.reshape(bsz, -1),
                        src=penalty_val.expand(indices_.size()).reshape(bsz, -1).to(diversity_buf),
                        dim=1,
                    )
                # N_GRAM_TYPE_DIVERSITY
                elif self.type_diversity_function is N_GRAM_TYPE_DIVERSITY and step + 1 >= self.n_gram:
                    # prev_indices_reshape (batch_size, mini_beam_size, num_groups, step + 1)
                    prev_indices_reshape = prev_indices.view(bsz, mini_beam_size, self.num_groups, -1)
                    # prev_indices_cut_n_gram (batch_size, mini_beam_size, num_groups, step + 3 - n_gram, n_gram - 1)
                    prev_indices_cut_n_gram = self.transform_tensor(
                        tensor=prev_indices_reshape,
                        n_gram=self.n_gram - 1,
                    )
                    # last_prev_cut_n_gram_g_gr (batch_size, mini_beam_size, g, n_gram - 1)
                    last_prev_cut_n_gram_g_gr = prev_indices_cut_n_gram[:, :, :g, -1, :]
                    # last_prev_cut_n_gram_g (batch_size, mini_beam_size, n_gram - 1)
                    last_prev_cut_n_gram_g = prev_indices_cut_n_gram[:, :, g, -1, :]
                    # overlap_n_gram (batch_size, mini_beam_size, g, n_gram - 1)
                    overlap_n_gram = (last_prev_cut_n_gram_g.unsqueeze(-2) == last_prev_cut_n_gram_g_gr).int()
                    # overlap_n_gram (batch_size, mini_beam_size, g)
                    penalty_val = (torch.sum(overlap_n_gram, dim=-1) == self.n_gram - 1).int()
                    # last_indices_g_gr (batch_size, mini_beam_size, g)
                    last_indices_g_gr = indices[:, :, :g, -1]
                    # diversity_buf (batch_size, vocab_size)
                    diversity_buf.scatter_add_(
                        index=last_indices_g_gr.reshape(bsz, -1),
                        src=penalty_val.reshape(bsz, -1).to(diversity_buf),
                        dim=1,
                    )
                # NEURAL_EMBEDDING_TYPE_DIVERSITY
                elif self.type_diversity_function is NEURAL_EMBEDDING_TYPE_DIVERSITY:
                    # last_indices_g_gr (batch_size, mini_beam_size, g)
                    last_indices_g_gr = indices[:, :, :g, -1]
                    # top k cosine similarity indices
                    k = self.top_cosine_similarity_indices.size(-1)
                    # expand last_indices_g_gr to (batch_size, mini_beam_size, g, k)
                    last_indices_g_gr = last_indices_g_gr.unsqueeze(-1).expand(-1, -1, -1, k)
                    # last_indices_g_gr_similarities (batch_size, mini_beam_size, g, k)
                    last_indices_g_gr_similarities = self.top_cosine_similarity_indices[last_indices_g_gr]
                    # last_indices_g_gr_similarities (batch_size, mini_beam_size, g * k)
                    last_indices_g_gr_similarities = last_indices_g_gr_similarities.view(bsz, mini_beam_size, -1).contiguous()
                    # penalty_val
                    if self.group_overlap is not None:
                        # penalty_val (batch_size, g)
                        penalty_val = 1 + self.group_overlap[original_batch_idxs, g, :g]
                        # penalty_val (batch_size, mini_beam_size, g, k)
                        penalty_val = penalty_val.unsqueeze(1).unsqueeze(-1).repeat(1, last_indices_g_gr_similarities.size(1), 1, last_indices_g_gr_similarities.size(-1))
                        # penalty_val (batch_size, mini_beam_size, g * k)
                        penalty_val = penalty_val.view(bsz, mini_beam_size, -1).contiguous()
                    else:
                        penalty_val = torch.ones(last_indices_g_gr_similarities.size())
                    # penalty_val = torch.ones(last_indices_g_gr_similarities.size())

                    # diversity_buf (batch_size, vocab_size)
                    diversity_buf.scatter_add_(
                        index=last_indices_g_gr_similarities.reshape(bsz, -1),
                        src=penalty_val.reshape(bsz, -1).to(diversity_buf),
                        dim=-1,
                    )
                # lprobs_g (batch_size, mini_beam_size, vocab_size)
                lprobs_g = torch.add(
                    lprobs_g,
                    other=diversity_buf.unsqueeze(1),
                    alpha=self.diversity_strength,
                )
            else:
                lprobs_g = lprobs_g.contiguous()

            # scores_buf (batch_size, mini_beam_size)
            # indices_buf (batch_size, mini_beam_size)
            # beams_buf (batch_size, mini_beam_size)
            scores_buf, indices_buf, beams_buf = self.beam.step(
                step=step,
                lprobs=lprobs_g,
                scores=scores_g,
                mask_stop_search=mask_stop_search_g,
            )
            prev_indices_buf = prev_indices_g.view(bsz * mini_beam_size, -1).contiguous()[beams_buf.view(-1).contiguous()]
            # (batch_size, mini_beam_size, step + 1)
            prev_indices_buf = prev_indices_buf.view(bsz, mini_beam_size, -1).contiguous()
            # (batch_size, mini_beam_size, step + 2)
            prev_indices_buf = torch.cat([
                prev_indices_buf,
                indices_buf.unsqueeze(-1),
            ], dim=-1)
            # indices (batch_size, mini_beam_size, num_groups, step + 2)
            if indices is None:
                indices = prev_indices_buf.unsqueeze(2)
            else:
                indices = torch.cat([
                    indices,
                    prev_indices_buf.unsqueeze(2),
                ], dim=2)
            
            beams_buf = beams_buf * self.num_groups + g
            scores_G.append(scores_buf.clone())
            beams_G.append(beams_buf.clone())
            
        # interleave results from different groups
        scores_buf = torch.stack(scores_G, dim=2).view(bsz, -1)
        indices_buf = indices[:, :, :, -1].view(bsz, -1)
        beams_buf = torch.stack(beams_G, dim=2).view(bsz, -1)

        # find num of overlapped tokens for each group pair
        # overlap (batch_size, num_groups, num_groups)
        overlap = self.TYPE_DIVERSITY_FUNCTION[self.type_diversity_function](
            indices=indices,
        )
        # then discount it for next timestamp
        # self.group_overlap (batch_size, num_groups, num_groups)
        if overlap is not None:
            if self.group_overlap is None:
                self.group_overlap = overlap
            else:
                self.group_overlap[original_batch_idxs] = self.group_overlap[original_batch_idxs] + overlap
            self.group_overlap = self.group_overlap * self.diversity_discount

        # scores_buf (batch_size, input_beam_size)
        # indices_buf (batch_size, input_beam_size)
        # beams_buf (batch_size, input_beam_size)
        return scores_buf, indices_buf, beams_buf

class SearchNode():
    def __init__(
        self,
        eos_token_id: int,
        pad_token_id: int,
        sep_token_id: int,
        initial_candidate: torch.Tensor,
        device: str,
        tokenizer,
        max_len: int=200,
        n_gram: int=1,
        step: int=0,
    ):
        self.eos_token_id = eos_token_id
        self.pad_token_id = pad_token_id
        self.sep_token_id = sep_token_id
        self.tokenizer = tokenizer
        self.device = device
        self.max_len = max_len
        self.n_gram = n_gram
        self.tgt = torch.tensor(initial_candidate).to(self.device)
        self.last_token = self.sep_token_id
        self.num_steps = step
        self.scores = torch.tensor([0] * self.num_steps).to(self.device)
        self.indices = torch.tensor(initial_candidate).to(self.device)

    def copy(self):
        new_item = SearchNode(
            eos_token_id=self.eos_token_id,
            pad_token_id=self.pad_token_id,
            sep_token_id=self.sep_token_id,
            initial_candidate=self.tgt,
            device=self.device,
            tokenizer=self.tokenizer,
            max_len=self.max_len,
            n_gram=self.n_gram,
            step=self.num_steps,
        )
        new_item.tgt = self.tgt
        new_item.last_token = self.last_token
        new_item.num_steps = self.num_steps
        new_item.scores = self.scores
        new_item.indices = self.indices
        return new_item

    def stop_search(self):
        return 1 if len(self.tgt) >= self.max_len or self.last_token == self.eos_token_id else 0
    
    def step(
        self,
        score: float,
        indice: int,
    ):
        self.num_steps += 1
        # add score in time step
        self.scores = torch.cat([
            self.scores,
            torch.tensor([score]).to(self.device),
        ], dim=-1)

        indice = indice if self.stop_search() == 0 else self.pad_token_id
        self.last_token = indice if indice != self.pad_token_id else self.last_token
        # if not stop search, update tgt and tgt_attention_mask
        if indice == self.eos_token_id or self.stop_search() == 0:
            self.tgt = torch.cat([
                self.tgt,
                torch.tensor([indice]).to(self.device)
            ], dim=-1)

        # update indices
        self.indices = torch.cat([
            self.indices,
            torch.tensor([indice]).to(self.device)
        ], dim=-1)

BEAM_SEARCH = "beam_search"
DIVERSE_BEAM_SEARCH = "diverse_beam_search"

TYPE_SEARCH = {
    BEAM_SEARCH: BeamSearch,
    DIVERSE_BEAM_SEARCH: DiverseBeamSearch,
}
