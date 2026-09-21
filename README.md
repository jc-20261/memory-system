

\## Subsystems



\### Memory generation



`Memory generation/mgen.py` produces memories from an activity prompt.

Each memory goes through:



1\. NL narrative generation

2\. Memory sketch (objects + action plot)

3\. Enhanced sketch (recursive decomposition)

4\. Generalization (reusable template names)

5\. State annotation (preconditions, changes, sensory, natural processes)

6\. JSON encoding

7\. JSON review



The demo `.txt` files in the same folder are the static parts of the

prompts (instructions plus worked examples). The dynamic part is the

activity and the memory being processed.



\### Search



`Explore/Explorer.py` loads all pickles, builds a reverse index over

every term in every memory, and accepts natural language queries. The

query processor calls DeepSeek once to produce a structured parse, from

which weighted search terms are generated. The engine matches those

terms against the index using a chain of multipliers: full-path

matching, component matching, token matching, verb matching,

measurement, position units, spatial inference, and goal-state bonuses.



Results are ranked by total score. Each result can be drilled into to

see the individual matched terms and their source contexts.



\### Planning



Given a query with a goal state, the planner:



1\. Retrieves seed actions by verb (once per target action).

2\. Chains backward from each seed's preconditions and old states.

3\. Runs a general backward chaining pass from the goals directly.

4\. For multi-target queries, combines per-action plans by Cartesian

&#x20;  product.

5\. Validates each candidate in two world states: one built from the

&#x20;  plan's own memories, one from the query's objects.

6\. Ranks by a weighted score with a coverage bonus for multi-target

&#x20;  plans that address every target action.



Traces are written to `plan\_trace\_simple.txt` and `plan\_trace\_full.txt`.



\### 3D module



Each memory can be represented as an assembly of primitive elements

(spheres, boxes, cylinders, and so on) with transforms. The module

supports:



\- Assembly generation from a memory JSON via LLM

\- Rendering (matplotlib, trimesh, or a software rasteriser)

\- Shape similarity via Chamfer distance on sampled point clouds

\- Spatial relation extraction using 13 AABB-based predicates

\- Relation map similarity for structural comparison across memories



\## Memory corpus



The `Memories/memories\_final.jsonl` file contains 98 memory encodings.

Each is a full JSON object with `object\_templates`, `action\_templates`,

`memory`, and `goal\_state` sections.



The pickles in `Explore/pickle\_memories/` are runtime objects converted

from those JSONs. They are the primary input to the search and planning

runtime.



The corpus is fixed. To add new memories, run the generation pipeline

and convert the new JSONs to pickles.



\## Known limitations



\- \*\*Pickles are Python-version-specific.\*\* Generated under Python 3.12.

&#x20; If you use an older Python, regenerate the pickles with

&#x20; `Explore/post processing and diagnostic scripts/convert\_pickle.py`.

\- \*\*The planner produces some false-success plans.\*\* A plan can succeed

&#x20; because its remaining subgoals were visited in dead branches rather

&#x20; than because they were actually satisfied. Workarounds are documented

&#x20; in the source.

\- \*\*The 3D module is not wired into search or planning.\*\* It runs

&#x20; independently and shares only the memory JSONs.



\## License



No license has been assigned. Add a `LICENSE` file if you intend the

code to be reused. Common choices are MIT, Apache 2.0, or GPL v3.



\## Citation



If you use this system in academic work, cite the archived version:



