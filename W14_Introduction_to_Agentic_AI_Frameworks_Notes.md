# Week 14 — Introduction to Agentic AI Frameworks

**Source:** `Week 14_Introduction-to-Agentic-AI-Frameworks.pdf`  
**Instructor:** Prof. Yueming Xing  

Study notes distilled from the slides. Use alongside the PDF for diagrams and exact wording.

---

## 1. What is an AI agent?

An **AI agent** is not the same thing as a plain **language model**:

| Aspect | Standard LLM | Agent |
|--------|----------------|--------|
| Behavior | One-shot: input → output | **Iterative**: perceive → reason → act → update memory |
| World | No real “environment” | Interacts with environment (tools, APIs, user) |
| Theory tie-in | Generation | Aligns with **classical AI** and **RL**, amplified by LLM reasoning |

### Formal view (four components)

An agent is often summarized as:

**Agent = (Perception, Memory, Policy, Action)**

- **Perception** — Reads environment state and inputs.  
- **Memory** — Stores and retrieves prior context.  
- **Policy** — Chooses actions given state (and goals).  
- **Action** — Executes tools, APIs, writes outputs.  

**Goal (informal):** Maximize utility through **repeated** interaction with the environment.

**Takeaway:** Modern stacks combine **LLM reasoning**, **tools**, **memory**, and **planning loops**. None of these alone is a full agent; **orchestration** matters.

---

## 2. From LLMs to agentic systems

- **Traditional LLM:** Single forward pass; no iteration, tools, or persistent workflow.  
- **Agentic loop:** **Observe → Plan → Act → Reflect** (repeat until done or stuck).  
- **Autonomous framing:** Goal → workflow → outcome with less step-by-step human steering.

This reframes the problem from **“generate text”** to **“solve tasks over multiple reasoning steps.”**  
Slides emphasize this as a **core interview topic** in AI engineering.

---

## 3. Core components of agentic AI

**Message:** Agentic AI is a **system**, not “just a bigger model.” “How does an agent work?” expects discussion of the **full stack**, not only the LLM.

| Component | Role |
|-----------|------|
| **LLM core** | Interprets goals, plans steps, language outputs each turn |
| **Planning** | Decomposes goals into subgoals; orders tasks; handles dependencies |
| **Memory** | Short-term context + longer / episodic knowledge across turns |
| **Tool use** | Search, calculators, APIs, code execution, etc. |
| **Environment interface** | How state changes are observed; user/system I/O |
| **Reflection loop** | Critique outputs; revise plan when results are weak or wrong |

### Agent architecture (conceptual pipeline)

Flow: **User goal** → **Planner** (with **Memory**) → **Tool executor** → **Reflection** → back to planning as needed.

- **Planner** — “Strategic brain”: what next, which tools, whether the goal is met.  
- **Reflection** — When results fall short, triggers **another planning cycle** → **self-correction** on open-ended, multi-step tasks.

---

## 4. Planning

Planning is a defining difference from single-shot LLM inference:

- **Decomposition** — Break task into subgoals.  
- **Dependencies** — Which steps must precede others.  
- **Sequencing** — Efficient order of execution.  
- **Replanning** — Revise when execution fails or context shifts.

**Policy view:** At time \(t\), choose action \(a_t\) given state \(s_t\) (often written \(\pi(a_t \mid s_t)\)): map state → beneficial action, grounded in goal and memory.

**Links:** RL, search, symbolic planning — good **cross-domain interview** material.

**Bottleneck:** **Planning quality** often limits performance more than raw model size; strong model + weak plan can lose to modest model + strong loop. **Tree-of-Thought**, hierarchical planning, etc., are active research directions.

---

## 5. Chain-of-thought (CoT)

- Prompt the model to emit **intermediate reasoning** before the final answer.  
- Helps **math**, **logic**, **multi-step planning**; reasoning is **in the token stream** (inspectable).

**Formal sketch:** \(x \rightarrow r_1 \rightarrow r_2 \rightarrow \cdots \rightarrow y\) instead of \(x \rightarrow y\) only.

**Critical caveat (interviews):** CoT traces are **not guaranteed** to reflect true internal computation — they can be **plausible narratives**. Still, CoT often **improves quality** and underpins **ReAct**, **Tree-of-Thought**, **Self-Consistency**, etc.

---

## 6. ReAct (Reason + Act)

- Interleaves **reasoning** with **real actions**: web search, DB, code, APIs.  
- Cycle: **Thought** (what to do) → **Action** (tool call) → **Observation** (result) → repeat.

**Why it matters:** Pure CoT can **hallucinate** with confident reasoning. ReAct **grounds** steps in **retrieved evidence** (“I searched X, found Y, therefore Z”).  
Very common in **agent interviews** and system design discussions.

---

## 7. Tool use

LLMs are limited by **parametric knowledge** (weights), weak precise arithmetic, no live web, no cross-session persistence by default.

**Pattern:** At step \(t\), agent emits tool query \(q_t\); environment returns observation → next reasoning step.

**Typical tools:** Search, calculator, databases (relational / vector), APIs (weather, finance, calendars), code sandboxes, etc.

**Production:** Tool routing and **error handling** are among the highest-impact engineering levers.

---

## 8. Retrieval-augmented generation (RAG)

**Idea:** At inference time, **retrieve** relevant documents and **inject** them into context instead of relying only on weights.

**Two-stage view:**

1. `Context = Retriever(query)`  
2. `Output = LLM(query, Context)`

**Why RAG:**

- Grounds answers → **less hallucination** on facts  
- **Updates** via index/corpus changes without full retrain  
- **Domain adaptation** via specialized corpora  
- Often **cheaper/faster** than fine-tuning for frequent knowledge updates  

**Engineering depth:** Chunking, **embedding** models, **reranking** — standard for enterprise agents.

---

## 9. Memory systems

Memory turns a **stateless** chat model into a **persistent** agent.

| Type | Idea |
|------|------|
| **Short-term** | What fits in active context **now**; session-scoped; lost when session ends |
| **Long-term** | Survives across sessions (DBs, KV stores, etc.); needs explicit read/write |
| **Episodic** | Records of past interactions/outcomes — “what worked or failed before” |
| **Vector memory** | Embeddings in a vector DB; **semantic** recall (meaning, not keywords) |

**Interview distinction:** The **context window is not “memory”** in the strong sense. Real memory needs **persistent storage** and **explicit retrieval** outside parameters.

---

## 10. Vector embeddings and semantic retrieval

**Embedding:** \(z = f(x)\) maps text (word, sentence, doc, query) to a **dense vector**; **similar meaning → nearby vectors**.

**Similarity:** Often dot product or cosine similarity, e.g. \(\mathrm{sim}(x_i, x_j) = z_i^\top z_j\) (or normalized cosine).

**For agents:** Store embeddings of memories, tool outputs, docs; at query time embed query and run **nearest-neighbor** search → relevant past context. Powers RAG and memory stacks. Tools mentioned: **FAISS**, **Pinecone**, **Weaviate**, **Chroma**.

---

## 11. Reflection and self-correction

**Loop:** Generate output → **critique vs. goal** → revise → repeat until satisfactory.

Strong for **code**, **math**, **multi-step argument** where errors compound. Can be implemented as a **critic** model, a **self-eval** prompt step, or **reward model** feedback.

**Interview link:** Connect to **Constitutional AI**, **RLHF** as examples of **alignment / meta-evaluation** thinking.

---

## 12. Multi-agent systems

One agent may be insufficient for hard tasks. **Multiple specialized agents** (planner, researcher, coder, reviewer) mirror human teams.

| Role (examples) | Function |
|-----------------|----------|
| **Planner** | Decompose goal, assign work, monitor progress (“PM”) |
| **Researcher** | Search, read, synthesize evidence |
| **Coder** | Implement, run, debug code |
| **Reviewer** | Quality, correctness, alignment with goal |

**Challenges:** Coordination, **communication protocols**, task allocation, avoiding redundant context. **LangGraph**, **AutoGen** mentioned as orchestration aids; workflow design remains partly **craft**.

---

## 13. Agent orchestration frameworks (high level)

| Framework | Positioning (from slides) |
|-----------|---------------------------|
| **LangChain** | Chains, agents, tools, memory; flexible; can get complex; good for **prototyping** and straightforward pipelines |
| **LangGraph** | **Graph** of nodes/edges; **stateful, cyclic** workflows; explicit state; strong for multi-step / multi-agent |
| **AutoGen** | **Microsoft**; **multi-agent conversation**; human-in-the-loop; code execution workflows |
| **CrewAI** | **Role-based** “crews”; roles, goals, backstories; low boilerplate for team-style agents |

**Interview tip:** Compare **graph state (LangGraph)** vs **conversation coordination (AutoGen)** vs **role crews (CrewAI)**.

---

## 14. Evaluation challenges

Agents run **trajectories**, not single strings — harder than static test-set accuracy.

| Metric | Notes |
|--------|--------|
| **Task success** | Did the agent finish the goal? Hard to define for open-ended tasks |
| **Latency** | End-to-end time across many steps |
| **Hallucination rate** | Errors can accumulate over the trajectory |
| **Robustness** | Input variation, tool failures, adversarial conditions |

**Research:** Interactive envs, trajectory scoring, partial credit. Examples cited: **AgentBench**, **WebArena**, **ToolBench** — space still **unsettled**. Robust **eval pipelines** matter as much as building the agent.

---

## 15. Failure modes

More autonomy → more ways to fail:

| Mode | Description |
|------|-------------|
| **Hallucination** | Wrong facts propagate and compound across steps |
| **Infinite loops** | Repeated actions without progress; token/compute drain |
| **Tool misuse** | Wrong tool, bad args, misread observations → cascading errors |
| **Reasoning drift** | Long runs: lose track of true user goal; optimize wrong proxies |

Address at **system design**, not only post-hoc patches.

---

## 16. Mathematical view (RL connection)

**Trajectory:** \((s_t, a_t, r_t, s_{t+1})\) — observe state, act, reward, next state.

**Objective:** Maximize **expected discounted return**, e.g. \(\max \mathbb{E}[\sum_t \gamma^t r_t]\) (discount \(\gamma\) trades off near vs. future reward).

**Concepts:** Long-horizon consequences, **exploration vs. exploitation**, **reward shaping** (designing rewards is as important as model choice). Bridges LLM agents to **classic RL theory**.

---

## 17. Interview-style Q&A (from slides)

1. **LLM vs. agent?**  
   LLM: single-pass generation. Agent: iterate with perception, planning, tools, memory, reflection.

2. **Why memory — isn’t the context window enough?**  
   Context is limited and usually session-scoped. Real memory needs **persistent external storage** and **retrieval** across sessions and tasks.

3. **RAG vs. fine-tuning for knowledge updates?**  
   RAG updates the **index/corpus** without retraining; cheaper/faster; provides **explicit evidence** and can reduce factual hallucination.

4. **ReAct vs. CoT?**  
   CoT: reasoning in tokens only. ReAct: **interleaved** reasoning and **external** actions, grounding in real observations.

5. **Critical failure modes?**  
   Hallucination compounding, infinite loops, tool misuse, reasoning drift — all worsened by long autonomous runs.

---

## 18. Future directions (slides)

- **Autonomous workflows** — Long-horizon, branching projects with less human micromanagement  
- **Multimodal agents** — Text, images, video, audio, code for richer perception and tools  
- **Self-improving systems** — Learn from trajectories; update policies/memory from success/failure  
- **AI operating systems** — Orchestrate **fleets** of agents (scheduling, resources, inter-agent comms)

**Closing theme:** The hard problems are **reliable orchestration**, **persistent memory**, **long-horizon planning**, and **alignment** under real-world complexity — not raw “intelligence” alone.

---

*End of notes.*
