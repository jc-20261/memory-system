# Memory System

A memory system that encodes natural language narratives as structured memory objects, indexes them for hybrid search, plans over them via backward chaining, and represents them geometrically as 3D assemblies.

This repository contains the code, the memory corpus, the object and action registries, the pickle artifacts, and the embedding caches needed to run the system out of the box.

## Overview

The system has four subsystems:

1. **Memory generation** — an LLM pipeline that turns an activity prompt into a natural language narrative, an annotated action timeline with state changes, and a final JSON memory encoding.

2. **Search** — a hybrid search engine over the memory corpus. Builds a reverse index from each memory's objects, actions, attributes, preconditions, state changes, and goal states.

3. **Planning** — an action-centric planner. Retrieves candidate actions by verb, chains backward from goals, and validates candidate plans against two world states.

4. **3D module** — represents each memory as an assembly of primitive elements and extracts a spatial relation map.

## Requirements

- Python 3.10 or newer
- A DeepSeek API key
- Disk space: roughly 350 MB of pickles, embeddings, and memory JSONs

Python packages:

    openai>=1.0
    numpy
    matplotlib
    rank-bm25
    sentence-transformers
    model2vec
    scikit-learn

## Installation

    git clone https://github.com/jc-20261/memory-system.git
    cd memory-system
    pip install -r requirements.txt

Set your API key as an environment variable:

    set DEEPSEEK_API_KEY=your-key-here

on Windows, or

    export DEEPSEEK_API_KEY=your-key-here

on macOS and Linux.

## Running

The search and planning CLI is the main entry point.

    cd Explore
    python Explorer.py

The default query mode is LLM. Type a natural language query at the prompt. Type `plan` at the result prompt to generate plans for the last query. Type `manual` to switch to manual query terms, `github` to print the repository URL, or `quit` to exit.

Search and planning are based on an initial batch of 98 memories and are for demonstrating basic functionalities. 

To run the memory generation pipeline:

    cd "Memory generation"
    python mgen.py

You can examine the memories file and records file in the Memories folder to see the initial batch of 98 memories generated, as well as all output from all stages of memory generation. if you wish to generate memories, you can set parameters in the mgen.py file on how many to generate and the conditions. The code is not currently optimized with respect to token usage and each memory might output ~200k tokens in total. 

## Repository layout

    memory-system/
    Explore/                      search and planning
        Explorer.py               CLI entry point
        pickle_loader.py          loads memories and storages
        pickle_search_index.py    reverse index
        pickle_search_engine.py   matching and ranking
        pickle_query_processor.py query parsing and term generation
        planner*.py               planning module
        memory_model.py           runtime classes
        embedding_cache.py        embedding group cache
        pickle_memories/          one .pkl per memory
        pickle_storage/           object, action, verb storage pickles
        data/
            object_storage.json
            action_storage_linked.json
            verb_storage.json
            alias_expansion.json
            embeddings/           precomputed .npz groups

    Memory generation/            mgen pipeline
        mgen.py
        *_demo.txt                prompt demonstrations

    3d/                          3D module
        3d_assembly.py
        3d_elements_storage.py
        3d_shape_similarity.py
        3d_relation_map.py
        3d_relation_predicates.py
        3d_render_assembly.py
        3d_scene_operations.py
        3d_llm_assembly_generation.py
        data/                     elements storage, demos, pointcloud cache

    Memories/                     memory corpus
        memories_final.jsonl      the 98 memory encodings
        records.jsonl             pipeline records from the generation run

    runtime/                      created on first run
        traces/
        snapshots/
        renders/

## Subsystems

### Memory generation

`Memory generation/mgen.py` produces memories from an activity prompt. Each memory goes through narrative generation, memory sketch, enhanced sketch, generalization, state annotation, JSON encoding, and JSON review.

The demo `.txt` files in the same folder are the static parts of the prompts.

### Search

`Explore/Explorer.py` loads all pickles, builds a reverse index over every term in every memory, and accepts natural language queries. The engine matches query terms against the index using a chain of multipliers: full-path matching, component matching, token matching, verb matching, measurement, position units, spatial inference, and goal-state bonuses.

Results are ranked by total score. Each result can be drilled into to see the individual matched terms and their source contexts.

### Planning

Given a query with a goal state, the planner retrieves seed actions by verb, chains backward from each seed's preconditions and old states, runs a general backward chaining pass from the goals, combines per-action plans for multi-target queries, validates each candidate in two world states, and ranks by a weighted score with a coverage bonus.

Traces are written to `plan_trace_simple.txt` and `plan_trace_full.txt`.

### 3D module 

Each memory can be represented as an assembly of primitive elements with transforms. The module supports assembly generation from a memory JSON via LLM, rendering, shape similarity via Chamfer distance, spatial relation extraction, and relation map similarity.

This module is under development. Token usage is very high for memory generation. You can check out scene renders from: https://github.com/jc-20261/memory-system-paper/tree/main/figures/3D%20frames%20for%20peeling%20apple%20memory

## Memory corpus

The `Memories/memories_final.jsonl` file contains 98 memory encodings. The pickles in `Explore/pickle_memories/` are runtime objects converted from those JSONs. They are the primary input to the search and planning runtime.

## Known limitations

- Hardcoded paths in the 3d module and mgen_objecttemplatesUsed.py still reference the original absolute path used during development.
- Pickles were generated under Python 3.12. Older Python versions may need to regenerate them.
- The planner produces some false-success plans. Workarounds are documented in the source.
- The 3D module is not wired into search or planning. It shares only the memory JSONs.

## License

This project is licensed under the MIT License. See LICENSE for the full text.

## Citation

If you use this system in academic work, cite the archived version:

    Jack Cheng. Using LLM to build a cognitive system: a preliminary study. Zenodo, 2026.
    https://doi.org/10.5281/zenodo.22870201  

