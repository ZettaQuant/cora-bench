# Third-party assets

Dataset and model owners retain their licenses. This repository does not redistribute model weights or raw benchmark source contexts.

- BABILong: pinned upstream commit `7a6efee29f5cac03c3c410e6799c80fd2ffe3610`. Adapted generator, task prompts, and metric source retain attribution. See `cora_bench/experiment/upstream/babilong/LICENSE`.
- LOONG: pinned upstream `MozerWang/Loong`, commit `6d2115b8b48a3d19412ccb52a0c9c4ee37869af4`; license and source checksums are in `cora_bench/experiment/upstream/loong/`.
- LongMemEval: pinned upstream `xiaowu0162/LongMemEval`, commit `9e0b455f4ef0e2ab8f2e582289761153549043fc`; license and source checksums are in `cora_bench/experiment/upstream/longmemeval/`.
- NoLiMa: pinned upstream `adobe-research/NoLiMa`, commit `cb14780b249fecf2851127b2101a062c1b2c6430`; the included source and needle templates carry Adobe Research's non-commercial research terms. See the bundled license. A future license for CoRA-Bench's own code does not override those terms.
- Public selector checkpoints are identified in `protocol.json`; obtain them from their owners under their respective model licenses. Hosted readers and judges remain subject to their providers' terms.

Public release is pending selection of a license for original CoRA-Bench code and review of the bundled third-party notices. No statement here replaces an upstream license.
