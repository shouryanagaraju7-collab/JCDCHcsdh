# JCDCHcsdh

Gate-level quantum circuits and Clifford+T resource estimates for **QARMA-64**, the tweakable block cipher behind pointer authentication in Armv8.3-A. Every number in this repo comes from a circuit that was actually built, simulated, and checked against the published QARMA-64 test vectors — nothing is estimated by formula alone.

## What's here

1. **Minimum-Toffoli S-box circuits.** An exhaustive meet-in-the-middle search over left cosets of the affine group AGL(4,2) proves σ1 needs exactly **5 Toffoli gates with no ancilla**, and that σ0/σ2 (odd permutations) need **4 Toffoli + 1 C³X**, with no smaller ancilla-free circuit possible.
2. **Gidney AND-gadget optimization**, applied two ways:
   - To the C³X gate's own construction (21T → 11T).
   - Systematically across every Toffoli in the cipher, using a borrowed clean ancilla — but only where a compute/uncompute pair is *provably clean* (checked by simulating every input, not by reading the circuit).
3. **A complete, verified QARMA-64 encryption circuit** — S-boxes, MixColumns, tweak schedule, key specialization — reproducing all nine published test vectors across every S-box/round/mode combination.
4. **A working Grover key-search oracle**, cost out under both the textbook and AND-gadget compilation modes, plus NIST `maxdepth`-limited attack scenarios.

## Findings

|                             | **Textbook (7T Toffoli)**        | **AND-gadget everywhere**                |
| --------------------------- | -------------------------------- | ---------------------------------------- |
| One S-box (σ1)              | 35 T, T-depth 20                 | **20 T**, T-depth 10                     |
| One S-box (σ0/σ2)           | 49 T                             | **24 T**                                 |
| Full encryption (σ1, r=7)   | 192 qubits, 8,960 T, T-depth 320 | 208 qubits, **5,120 T**, T-depth **160** |
| vs. RevKit baseline         | —                                | **8.8× fewer T gates**                   |
| Grover, one iteration (s=3) | 512 qubits, 58,198 T             | 560 qubits, **31,988 T**                 |
| Full attack                 | 2^81.11 gates, 2^79.48 T         | 2^81.19 gates, **2^78.62 T**             |

**The one counterintuitive result worth knowing before you read the tables:** the AND-gadget version has *more* total gates and slightly *more* full depth than the textbook version, despite cutting T-count nearly in half. Every gadget trades a T gate for extra Clifford gates plus a mid-circuit measurement, and Clifford gates dominate the raw gate count. **T-count and T-depth improve substantially; total gate count and depth do not** — which mode is "better" depends entirely on whether your cost model is T-gate-distillation-bound (favor AND-gadget) or gate-count/depth-bound (favor textbook).

## Repository structure

```text
├── main.tex                         # Main LaTeX source of the research paper
├── main.bib                         # Bibliography database
├── llncs.cls                        # Springer LNCS document class
├── splncs04.bst                     # Springer LNCS bibliography style
│
├── tables/                          # LaTeX source files for paper tables
│   ├── ...
│   └── ...
│
├── supplementary_code/              # Supplementary source/code
│   ├── ...
│   └── ...
│
└── QARMA64_verification_code.zip    # QARMA-64 verification code
```
## Scope and honest limitations

- **Proven vs. heuristic.** Only the S-box Toffoli counts are proven minimal, within the stated circuit class (4 wires, no ancilla, one C³X for odd S-boxes). CNOT/NOT counts and the MixColumns circuit are the best found by heuristic search, not certified optimal.
- **Logical-level estimates only.** These exclude error correction, qubit routing, and magic-state distillation overhead, consistent with standard practice in this literature.
- **Zero-T uncomputation assumption.** It requires mid-circuit measurement and classical feed-forward within the coherence budget, which is a genuine hardware assumption. On architectures without adaptive control, read the textbook columns instead.
- **Q1 model only.** Grover search assumes only classical query access to the cipher. In the Q2 model (superposition access to the keyed oracle), Even-Mansour-style constructions including QARMA can fall to Simon's-algorithm-based attacks running in polynomial time. This is a categorically different threat that this resource estimate does not address.

## Building on this

- **QARMAv2** revises the design; the S-box search and gadget-application tooling here should carry over with minimal changes.
- **AND-gadget technique** applies equally to the Grover comparator tree's internal structure (already done here) and could be pushed further into the diffusion operator's multi-controlled-Z.
- **Further T-depth reduction** avenues not yet pursued here: T-depth-1 constructions (Selinger) and depth-optimal meet-in-the-middle synthesis (Amy et al.).

## References

Built on Avanzi's QARMA specification, Jones' measurement-based Toffoli construction, and Gidney's temporary-AND gadget (implemented and cross-checked against [Qualtran](https://github.com/quantumlib/Qualtran)'s reference `And` bloq). S-box starting circuits synthesized with [LIGHTER-R](https://github.com/vdasu/lighter-r); baseline comparison against [RevKit](https://github.com/msoeken/revkit). Full citations in the accompanying paper.
