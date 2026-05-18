# RAG Evaluation Results

## Aggregate Metrics

| Config | Hit@1 | Hit@5 | Hit@10 | MRR | Recall@10 | Faithfulness | Relevance | Cite Acc. |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| fixed_dense | 0.184 | 0.429 | 0.429 | 0.272 | 0.429 | 0.816 | 0.837 | 0.816 |
| fixed_hybrid | 0.320 | 0.460 | 0.460 | 0.387 | 0.460 | 1.000 | 0.820 | 0.800 |

## Notes
- Hit@k: fraction of questions where ground-truth chunk is in top-k
- MRR: Mean Reciprocal Rank
- Faithfulness / Relevance: LLM-as-judge binary scores
- Cite Acc.: fraction of cited chunk IDs present in retrieved set