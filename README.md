Memory System
A memory system that encodes natural language narratives as structured memory objects, indexes them for hybrid search, plans over them via backward chaining, and represents them geometrically as 3D assemblies.

This repository contains the code, the memory corpus, the object and action registries, the pickle artifacts, and the embedding caches needed to run the system out of the box.

Contents
Overview

Requirements

Installation

Running

Repository layout

Subsystems

Memory corpus

Known limitations

License

Citation

Overview
The system has four subsystems:

Memory generation (Memory generation/) - an LLM pipeline that turns an activity prompt into a natural language narrative, an annotated action timeline with state changes, and a final JSON memory encoding. Objects and actions are registered in template storages for reuse.

Search (Explore/) - a hybrid search engine over the memory corpus. Builds a reverse index from each memory's objects, actions, attributes, preconditions, state changes, and goal states. Supports exact, fuzzy, component, token, measurement, spatial, and goal-state matching.

Planning (Explore/) - an action-centric planner. Retrieves candidate actions by verb, chains backward from goals through preconditions and old states, and validates candidate plans against plan-context and query-context world states.

3D module (3d/) - represents each memory as an assembly of primitive elements, renders it, compares assemblies by shape similarity, and extracts a spatial relation map.

Requirements
Python 3.10 or newer

A DeepSeek API key (for LLM-driven stages)

Disk space: the repository includes roughly 350 MB of pickles, embeddings, and memory JSONs

Python packages (see requirements.txt):

text
openai>=1.0
numpy
matplotlib
rank-bm25
sentence-transformers
model2vec
scikit-learn
The fuzzy-matching features degrade gracefully if sentence-transformers or model2vec are absent; the rest of the system still runs.

Installation
text
git clone https://github.com/jc-20261/memory-system.git
cd memory-system
pip install -r requirements.txt
Set your API key. Either create a .env file at the repo root:

text
DEEPSEEK_API_KEY=your-key-here
or set it as an environment variable:

text
set DEEPSEEK_API_KEY=your-key-here
on Windows, or

text
export DEEPSEEK_API_KEY=your-key-here
on macOS and Linux.

Running
The search and planning CLI is the main entry point.

text
cd Explore
python Explorer.py
The default query mode is llm. Type a natural language query at the prompt:

text
[llm]> how to peel an apple
Type plan at the result prompt to generate plans for the last query. Type manual to switch to manual query terms, github to print the repository URL, or quit to exit.

To run the memory generation pipeline:

text
cd "Memory generation"
python mgen.py
Repository layout
text
memory-system/
├── Explore/                      search and planning
│   ├── Explorer.py               CLI entry point
│   ├── pickle_loader.py          loads memories and storages
│   ├── pickle_search_index.py    reverse index
│   ├── pickle_search_engine.py   matching and ranking
│   ├── pickle_query_processor.py query parsing and term generation
│   ├── planner*.py               planning module
│   ├── memory_model.py           runtime classes
│   ├── embedding_cache.py        embedding group cache
│   ├── pickle_memories/          one .pkl per memory
│   ├── pickle_storage/           object, action, verb storage pickles
│   └── data/
│       ├── object_storage.json
│       ├── action_storage_linked.json
│       ├── verb_storage.json
│       ├── alias_expansion.json
│       └── embeddings/           precomputed .npz groups
│
├── Memory generation/            mgen pipeline
│   ├── mgen.py
│   └── *_demo.txt                prompt demonstrations
│
├── 3d/                          3D module
│   ├── 3d_assembly.py
│   ├── 3d_elements_storage.py
│   ├── 3d_shape_similarity.py
│   ├── 3d_relation_map.py
│   ├── 3d_relation_predicates.py
│   ├── 3d_render_assembly.py
│   ├── 3d_scene_operations.py
│   ├── 3d_llm_assembly_generation.py
│   └── data/                    elements storage, demos, pointcloud cache
│
├── Memories/                     memory corpus
│   ├── memories_final.jsonl      the 98 memory encodings
│   └── records.jsonl             pipeline records from the generation run
│
└── runtime/                      created on first run
    ├── traces/
    ├── snapshots/
    └── renders/
Subsystems
Memory generation
Memory generation/mgen.py produces memories from an activity prompt. Each memory goes through:

NL narrative generation

Memory sketch (objects + action plot)

Enhanced sketch (recursive decomposition)

Generalization (reusable template names)

State annotation (preconditions, changes, sensory, natural processes)

JSON encoding

JSON review

The demo .txt files in the same folder are the static parts of the prompts (instructions plus worked examples). The dynamic part is the activity and the memory being processed.

Search
Explore/Explorer.py loads all pickles, builds a reverse index over every term in every memory, and accepts natural language queries. The query processor calls DeepSeek once to produce a structured parse, from which weighted search terms are generated. The engine matches those terms against the index using a chain of multipliers: full-path matching, component matching, token matching, verb matching, measurement, position units, spatial inference, and goal-state bonuses.

Results are ranked by total score. Each result can be drilled into to see the individual matched terms and their source contexts.

Planning
Given a query with a goal state, the planner:

Retrieves seed actions by verb (once per target action).

Chains backward from each seed's preconditions and old states.

Runs a general backward chaining pass from the goals directly.

For multi-target queries, combines per-action plans by Cartesian product.

Validates each candidate in two world states: one built from the plan's own memories, one from the query's objects.

Ranks by a weighted score with a coverage bonus for multi-target plans that address every target action.

Traces are written to plan_trace_simple.txt and plan_trace_full.txt.

3D module
Each memory can be represented as an assembly of primitive elements (spheres, boxes, cylinders, and so on) with transforms. The module supports:

Assembly generation from a memory JSON via LLM

Rendering (matplotlib, trimesh, or a software rasteriser)

Shape similarity via Chamfer distance on sampled point clouds

Spatial relation extraction using 13 AABB-based predicates

Relation map similarity for structural comparison across memories

Memory corpus
The Memories/memories_final.jsonl file contains 98 memory encodings. Each is a full JSON object with object_templates, action_templates, memory, and goal_state sections.

The pickles in Explore/pickle_memories/ are runtime objects converted from those JSONs. They are the primary input to the search and planning runtime.

The corpus is fixed. To add new memories, run the generation pipeline and convert the new JSONs to pickles.

Known limitations
Hardcoded paths in some files. The 3d/ module and Memory generation/mgen_objecttemplatesUsed.py still reference the original absolute path used during development. Replace the original base path with a relative path if you run them from elsewhere.

Pickles are Python-version-specific. Generated under Python 3.12. If you use an older Python, regenerate the pickles with Explore/post processing and diagnostic scripts/convert_pickle.py.

The planner produces some false-success plans. A plan can succeed because its remaining subgoals were visited in dead branches rather than because they were actually satisfied. Workarounds are documented in the source.

The 3D module is not wired into search or planning. It runs independently and shares only the memory JSONs.

License
This project is licensed under the MIT License. See LICENSE for the full text.

Citation
If you use this system in academic work, cite the archived version:

text
Jack Cheng. Using LLM to build a cognitive system: a preliminary study Zenodo, 2026.
https://doi.org/10.5281/zenodo.22870202