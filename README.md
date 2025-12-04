# 🧠 Askle: Your AI Research Agent for Biomedical Discovery

> *Askle is a POC demo demo of **ModiX**, the agentic biomedical Q&A module within ModiScopeAI's core MVP platform.

*By Tiange (Alex) Cui*

---

## 🚀 Why Askle?

In the fast-paced world of life sciences, keeping up with the ever-expanding universe of research literature is a constant challenge. PubMed alone publishes thousands of new articles daily.

**What if you could ask a research question and instantly get a grounded, referenced summary** — synthesizing the *latest* studies, *guidelines*, and *expert consensus*?

That’s what **Askle** delivers. built on open LLMs, such as **Google’s Gemini large language model**, Askle is a powerful, automated AI agent designed to help medical professionals and researchers get **trustworthy, up-to-date answers**, without combing through papers one by one.

- Saves hours of manual research
- Provides instant, grounded answers
- Scales affordably with no external API costs
- Highly modular and developer-friendly

Askle is your AI co-pilot for research discovery. Powered by LLM, built for clarity.

<img width="397" height="314" alt="Askle" src="https://github.com/user-attachments/assets/68ee5c48-994a-4f2a-b5dc-30405447e719" />

---

## 🔍 How Askle Works

Askle is built around four modular phases:

1.  **Query Understanding & Retrieval**
2.  **Data Acquisition & Storage**
3.  **RAG Indexing & Embedding**
4.  **Answer Synthesis & Referencing**

### 🛠️ Pipeline
<img width="409" height="409" alt="workflow" src="https://github.com/user-attachments/assets/5cee568f-853b-4a44-9be2-c9da9441d204" />

### 📚 Phase 1: Understand the Question
Askle uses **Few-shot prompting** and **Structured Output (JSON)** from Gemini to transform user questions into powerful search queries:
```python
prompt = f"""You are an expert at extracting relevant keywords from text for search queries.
    Your output should be a JSON object with a single key "keywords" and its value as a list of strings.
    Here are a few examples:

    User Question: "What are the latest treatments for rheumatoid arthritis?"
    {{"keywords": ["latest treatments", "rheumatoid arthritis"]}}
    User Question: "Find me research articles on the impact of exercise on cognitive function in older adults."
    {{"keywords": ["impact of exercise", "cognitive function", "older adults"]}}
    User Question: "Tell me about the genetic basis of Alzheimer's disease."
    {{"keywords": ["genetic basis", "Alzheimer's disease"]}}
    User Question: "{sanitized_question}"
    {{
      "keywords":"""
        response = client.models.generate_content(
        model='gemini-2.0-flash',
        contents= prompt,
        config={'response_mime_type': 'application/json'}
    )
```

### 📚 Phase 2: Retrieve Literature
Askle searches PubMed and downloads both journal abstracts and guidelines, utilizing Function Calling and Document Understanding.
```python
conn = sqlite3.connect(DATABASE_NAME)
  create_papers_table(conn)

  for i, paper in enumerate(most_recent_papers_data):
      download_paper(paper, conn)

  for i, paper in enumerate(most_relevant_papers_data):
      download_paper(paper, conn)
```

### 📚 Phase 3: Vector Embedding & RAG
(Gemini's) Embedding model is used with ChromaDB for vector storage and retrieval:
```python
rag_collection = process_papers_for_rag_persistent(
        client=chroma_client,
        download_dir=RAG_DIR,
        collection_name=CHROMA_COLLECTION)
```

### 📚 Phase 4: Final Answer Generation
All retrieved information is compiled into a long-context prompt for LLM (eg. Gemini), producing a final answer with references:
```python
response = client.models.generate_content(
        model=GENERATIVE_MODEL,
        config=low_temp_config,
        contents=synthesis_prompt)
```

| Concept | Used in Askle? | How? |
| :--- | :--- | :--- |
| Structured Output (JSON) | ✅ | Extracted keywords for search |
| Few-shot Prompting | ✅ | Clarified user queries |
| Document Understanding | ✅ | Parsed PubMed abstracts & PDFs |
| Function Calling | ✅ | Downloaded literature from URLs |
| Long Context Window | ✅ | Summarized abstracts + context |
| Embeddings | ✅ | Chunked text + Gemini vector model |
| Vector Search/Store (RAG) | ✅ | ChromaDB used for retrieval |
| Grounding | ✅ | Source tags replaced with numbered references |
| GenAI Evaluation | ✅ | Ratings recorded (1-5 + hallucination check) |
| Agents | ⏳ In progress | Framework supports modular agent design, <br> a novel agentic framework **FlowLDP** has been developed |
| Image/Video/Audio Understanding | ❌ | Not yet implemented |
| MLOps with GenAI | ⚙️ Planned | Modular pipeline supports future CI/CD |

---

## To Test:

1. Create new env.<br>
``` diff
+ conda create -n askle python==3.10
```
3. Install [Docker](https://docs.docker.com/desktop/setup/install/mac-install/)
4. Makre sure docker-compose is working <br>
```diff
+ docker-compose -v
```
6. Install requirements<br>
```diff
+ pip install --no-cache-dir -r ./backend/requirements.txt
```
8. Run Docker (Desktop)
9. Build the app<br>
```diff
+ docker-compose up --build
```
11. ☕
12. Open browser, check the website at:<br>
```diff
+ http://localhost:3000/
```

---

## To Develop:
Please create on your own branch and pull from `main` periodically to get the latest update. <br>
```diff 
+ git checkout -b new-branch-name
```
<br>
<br>

![framework](https://github.com/user-attachments/assets/cca0bad7-9456-4564-8258-3cda6ae1e6ec)

---

## ⚠️ Limitations & What's Next
API Rate Limits: Scaling may require better batching or quota management.

No PMC full-text: Limited access to non-open papers may affect context quality.

Grounding edge cases: Some citation replacements may fail if metadata isn't matched.

## Future Potential:

Add image/video understanding for radiology or pathology workflows
Apply Agentic AI to chain multiple intelligent tools
Integrate MLOps pipelines for live deployment, retraining, and monitoring

---

Interested the code? Stay tuned for future open-source release of ModiX and ModiScope on my [GitHub](https://github.com/AlexTRee/)! <br>
🚀 For questions and suggestions, please feel free to connect me on [LinkedIn](https://www.linkedin.com/in/tiangecui/)

---
<sup><i>© Tiange Cui</i> 2025. All Rights Reserved. Askle is Open Source for Non-Commercial Use Only.</sup>
