# -*- coding: utf-8 -*-
"""
Process influence (I_proc) scaffold.

This module defines a minimal interface and a greedy inducer for sentence dependencies:
for each sentence s_j (j>i), find a minimal parent set P_j ⊆ {1..j-1} s.t. P_j ⟶ s_j (LLM entailment).

You need to implement EntailmentModel.entails(prefix_sentences, target_sentence) -> bool or prob.

This is optional and used to compute:
  - coverage influence: I_proc_cov(i) = |{ j>i : i ∈ P_j }|
  - local entailment delta: I_proc_local(i) = Σ_j [ Entail(s_j|prefix) - Entail(s_j|prefix\{s_i}) ]·w_j
"""

from __future__ import annotations
from dataclasses import dataclass
from typing import List, Dict, Any, Optional, Tuple
import math

class EntailmentModel:
    def entails(self, prefix: List[Dict[str,Any]], target: Dict[str,Any]) -> float:
        """Return probability that prefix entails target (0..1)."""
        raise NotImplementedError

@dataclass
class ProcessInfluence:
    entailment: EntailmentModel

    def minimal_parents(self, sentences: List[Dict[str,Any]], j: int, thresh: float=0.7) -> List[int]:
        """Greedy backward elimination to approximate a minimal parent set for s_j."""
        assert 1 <= j <= len(sentences)
        prefix = [s for s in sentences if s["sid"] < j]
        tgt = next(s for s in sentences if s["sid"] == j)
        if not prefix:
            return []
        # start with all sids as candidate parents
        S = [s["sid"] for s in prefix]
        # while entailment still holds after removing some, try to remove
        changed = True
        while changed and S:
            changed = False
            for sid in list(S):
                trial = [s for s in prefix if s["sid"] in S and s["sid"] != sid]
                p = self.entailment.entails(trial, tgt)
                if p >= thresh:
                    S.remove(sid)
                    changed = True
        return S

    def coverage_influence(self, sentences: List[Dict[str,Any]]) -> Dict[int,int]:
        cover = {s["sid"]: 0 for s in sentences}
        for j in range(2, len(sentences)+1):
            Pj = self.minimal_parents(sentences, j)
            for sid in Pj:
                cover[sid] += 1
        return cover

    def local_entailment_delta(self, sentences: List[Dict[str,Any]], w_decay: float=0.98) -> Dict[int,float]:
        """Sum over j: [Entail(prefix, s_j) - Entail(prefix\\{s_i}, s_j)] * w_j."""
        out = {s["sid"]: 0.0 for s in sentences}
        for j in range(2, len(sentences)+1):
            prefix = [s for s in sentences if s["sid"] < j]
            tgt = next(s for s in sentences if s["sid"] == j)
            base = self.entailment.entails(prefix, tgt)
            # distance-based weight
            wj = w_decay ** (len(sentences) - j)
            for sid in [s["sid"] for s in prefix]:
                trial = [s for s in prefix if s["sid"] != sid]
                p2 = self.entailment.entails(trial, tgt)
                out[sid] += max(0.0, base - p2) * wj
        return out
