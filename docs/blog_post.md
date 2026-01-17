from sieves.tasks import DistillationFramework

# `sieves`: A Unified Interface for Structured Document AI

Document AI is the process of turning unstructured documents - PDFs, emails, social media feeds - into structured data. Files are ingested, texts chunked, structured entities extracted, and perhaps a specialized model distilled for more efficient processing. Think invoices, support emails, or tweets at scale.

## Isn't this solved already?

![xkcd #927 – Standards](https://imgs.xkcd.com/comics/standards_2x.png)
_<sup>There's an xkcd for everything, including the current language model tooling landscape.</sup>_

The tools for these steps exist. They are excellent, and they are built with different paradigms and goals in mind. **[Outlines](https://github.com/dottxt-ai/outlines)** provides robust constrained generation. **[DSPy](https://github.com/stanfordnlp/dspy)** focuses on declarative prompt optimization. **[LangChain](https://github.com/langchain-ai/langchain)** has broad API support and a large ecosystem. **[GLiNER](https://github.com/fastino-ai/GLiNER2)** and **[Transformers](https://github.com/huggingface/transformers)** zero-shot classification pipelines offer specialized, high-performance inference.

Choosing one creates path dependency early in the application design. If you build your application logic around LangChain's abstractions, switching to a specialized model like GLiNER for efficiency is painful. You must migrate prompt signatures, integrations, and orchestration logic.

Too much developer time is lost to this friction, largely on integration and rework. You piece together disparate tools for ingestion, chunking, and prediction. You reinvent the same boilerplate for every project. This is a pattern we repeatedly encountered while working on document AI problems.

The missing piece was a stable, schema-first layer that sits above these tools and allows them to be composed without entangling application logic with a specific backend or execution framework.

To mitigate this, we built `sieves`. `sieves` is a framework-agnostic abstraction for building these workflows. It replaces imperative glue code with declarative design. Its modularity allows complex pipelines to be built from simple, reusable tasks.

It allows you to mix and match language model frameworks, so you can build a pipeline like this:

[![](https://mermaid.ink/img/pako:eNqlWGmPo0qy_SslS3daulR1si_1dGfGNjbGmMWYzai-JIvNvoOBVv_3oaretHTv7ac3S36AzCAy4kRkJsTh28ovg3D1uvrll29xEXevT9--dFGYhxZsYuhlYftlEb0VT09fikVxUzZB2CyiL10Di7aCTVh0X54_nldNnMNm2pZZ-SeNt-L79--__PJWvBW3rHz4EWy6p5P-Pu_pSSzuYdvFZfH3b09xfn99eltFXVe1rwA0YVW2cVc208tifFH7eo-7qPf6Nmz8sugW21_9MgcsTmM0i9EoCAifheyNe4E3jnshSe72wlK090JCj_NZjvH9EH9bPT9l0Auzd1e__vrD_6-_vr69FXzpZ3Fxf1danL-rdO_9x-sThqLPT9H_3hf37RJiXHTvKmXxtnr6_hnPNuqLdLHw53DgADvY_F9B9ABHKZxhSYz6W_sbzqJ_GX4j_4D1n7Y_oW6jchmFP4H6rwHNYNvGt9iHv89-sKB8_cg3qIr7_3iwDWnyObY2qv5AJeFerpemXMxoZ96XnuUvl62xXcsfchKW7HsHYBvZ2jkAgPchD-aLAebJAcV56W-1B6ttzexoXHElaWhkkGd5FC-7iA6LuqylJKNCE2gzEhQuLifiDMbLAZSbGyjvPM4NcGobMREnDXT6DGq9AOMkDqcJADRW0OCgn809Dyo9JjrmBvqR9A-bMbra-xrmzebhHwABb1U_uY6F-aTKJ6DdDOBhLXC1A8COBmC5GdBsae43OnadDTCRHmADD0zFDEjGAU08gGFvgN4sQMcTAKdVLJ2I-TE5Zj2l8JiMh3OSukee2BaXtjjx6FrgaSVI1CyaGDDm7OmQoJPC9008sUAxsFnl2VROxt1xxxN6mihn1NioqSVfkhua8wdgmg4IktkVhc170gV8l-3Olk4WKg63WOhdNrEgmut0jybb43lJZ3HUz7SE2TXeYY1beU5INMXAAYbLXIZ0d_k0c_ZUQ3Bh5zm1LgIUJl973AzZ1M-79X_etlokad2e0o6uypFItw5Cs5I5lEIJW30ME3LTHH0o7rNXRHSewn2V97tdUdA04BbdaNFVNpGRtn6VeI9BxQ9MJxpXrZJ4WjfvOtwjfb_InISi7XBfY1NrwvLAIoBD0DjTHe063KB163p1eJCqNgoSWWBUIuCNDqvGgbZo2Ug9k2A9QC4RBusaZw4jxHpA1U6_7gcD1kxgh5xTsLnT6juYdIU66bY6c8wOKF2p3ozGpenJ3kC-i3MNS6rTWrFD7NIRGU5gFMZpF7Xbj_58yq7N4BAEYqCEd4iNa5NOLFOPSJ7VdddQKCQRKWMavpGvBYN7NyfS7gmQ3YirHoIb-qGNsfsSN6y-shP8xgC_rSfsUNo5PPH9srS7oaaibkIqva7rK2-ZAR67nCPsMTaR8GWrS_h8dLeeGiM83jjh1TrZLdIOqUuJO1YOnMxbvKpnTNVODRNwFFfttnfBDYrjKU5VpxoJz3nAOqsfA10FplXFLu9RM8I-Ck86GGXI2qc8G6KJlsGpNsedfTk_cP3aXBkhao51qx_QVHqwZbIeZgERdyKTAuog0JWl12W1TzG_OAJHvcTeljZH6Xghu2RnR6fUoxzpcOOlB4LDzEGgSA6DXN9qzNuVTrMGyMMeXI8JXSsgoqs3ZjM4xmzUsOrGE4045NGBsOSSw3auSaVE28Cjr5Z7xBCb4gFSlfSI2uIfUebjZulyo33zkhzVxvM9S8-aQeSmFz88x-sKQkHuW0tCO62NPa1EZljfDjydY5samIKNnJGrokfG3OeJYdumcReZqM_JnPSTS4FvsJY9F1KvnhjPEfLOvkADozEF5-Iqu5xmjvU6og3azt6aHVOyu72zMxDNY70WlXwmC4C8JsimmyYjYh9wPOPZ1FAtm3rCOZv47qTidiEdaMzZYSK1JzGU8rQOjeqrqCU0cO9ouHYbhW5K4V6n_YgPLNC0WUFuwtxbWsEimM2g9jGd87EVm31wT5txeYOVOFtqy4HjGpIQid4LIWawfidfdMp1PGU6cces66VauepOXGDo9UKjsKGnrtmSJL9hMYdlWacl85LNWSJsM35dMOSVKLvtw-1C9wJxBS_Nc6EOR2EvO3FHbdXxbIcplp7lx_BoZIieDic9bRTWcqN9ENISr6kSn132ys5L3FjBHW8NNISHhAjdCJNZRSwp_FRjCOIccGp5w8qFdzajAG9ZMblDGsRcpFX1Fu5uvWW1NznVc0UOK0y74jYKxSS6mMQoHk5GTaO9utmEru0LiB8m4RiiYowsi7VWepr0nBtF326Ex4UcyPYnh_e6CNVsddtbfdibUYowmVHZCO0QTeZgcewC7c7wqNrOy4G6ip24lCyYtOzpLLfstT14p74TA7TC57Ci2ttVlahrOJFX_SKwLgcci7uBEXW9fks6UDMDj9jqQn_EaIGMQzxz4jY3gkHJq3U0-NRAudfryAFiGmTL9fKaubT7ZrTnEWE3IcUTrkpcsVKTg1C8J3sVl4OTGXvOOvEu_WN2N1UMEALfHi0KdgLZiOYJH4ZbNRWaZD5I69iAB2T3hhrN9RCSg9Pfk0zkDiih9jkvQ7uvKW3aU4zePx4xFKII7gVdoO-qlftRyrT0koFWPjcHQuZ9yIGcCUnGv0xRIamSh0vpfvamVutPFlk91lQVhVdn0KRGzikYWCbeF4uNADPjZY_Qeq1epnYQkQujV1Vl9xUdU067qzmkqKwKyjTnlQWpZmxt2seaWjdT5YbXukRPTp6KXM6RluntiVMoNkDdO8Xpds-u6Xyc5PnsXbe8fdUTaGGGSEipqYLpLBeyKGM5myqBlCdCdRRzqGZn5oQCW7LNrt5D2z_yjjS3anosT_FdCG3HFUYVzRTtjs4qMieN5VFUG9inpcZZEzubiA1lRh60zOGDiY59O8pNeFFY6dIun0HIbrPU326IPJ25MTgTim8ptF3f94LRk_JVCaYhLKiBmRoAMhJ0LXUAebdpd0kEjxmckzafrcxkD2KMKvde2q21x8FIT8pWOV6Yyjkycl2ssfoMW68jR7OlrJhVd7xmM3ab03AjUQzP8j5qtoBHxrRh-8licget0takj4xKTJaEX3d65eaFQW-wwKW4bV8oJ4XwcoS8GdA3ELexp4vQgnhzwIn-EHE9NS4nuT34cn1e_9eNXDPUZuapG-cdQ9_efgh32d5IL_05327_WFn_rhj-rK-NdwpzK5s8bNqfFNk49S-V2cpO__epwPJhoAmOxT-oAPozKrCY_UQpZPHSx_9TFiAW7xF-RL0bFxX_51QsLPzlrHRh8NJ5Bfr13nbLFP8D7Sc1-1v92_Lkda0EnOBforkiUGd9cWZNaEx8xqakESnbbGLxRZxxBUsef2n_RMR-IHkKf0D5DPLpBIv7NloC-GOc_3-Ef6CZTy8vf_1B0j4Hv1v7D9GS0Y_7T7Ozel7dmzhYvXZNHz6vlu2Rw_fh6oMpL7jeGfTb6pNLNenbauG_y5wKFm5Z5v-c1pT9PVq93mDWLqO-WmhXyMfw3sD8h3Rhzwvh3pZ90a1eCZJhP6ysXr-txtXrC0bSX2mMJVCM4Bbyi1H082paveIk95XhWJLGOIxEaZz6_ryaPxxjX1EOJygOpVmUonCKxZ9XYfBOsOXP3wAffwO-_wOtfTs1?type=png)](https://mermaid.live/edit#pako:eNqlWGmPo0qy_SslS3daulR1si_1dGfGNjbGmMWYzai-JIvNvoOBVv_3oaretHTv7ac3S36AzCAy4kRkJsTh28ovg3D1uvrll29xEXevT9--dFGYhxZsYuhlYftlEb0VT09fikVxUzZB2CyiL10Di7aCTVh0X54_nldNnMNm2pZZ-SeNt-L79--__PJWvBW3rHz4EWy6p5P-Pu_pSSzuYdvFZfH3b09xfn99eltFXVe1rwA0YVW2cVc208tifFH7eo-7qPf6Nmz8sugW21_9MgcsTmM0i9EoCAifheyNe4E3jnshSe72wlK090JCj_NZjvH9EH9bPT9l0Auzd1e__vrD_6-_vr69FXzpZ3Fxf1danL-rdO_9x-sThqLPT9H_3hf37RJiXHTvKmXxtnr6_hnPNuqLdLHw53DgADvY_F9B9ABHKZxhSYz6W_sbzqJ_GX4j_4D1n7Y_oW6jchmFP4H6rwHNYNvGt9iHv89-sKB8_cg3qIr7_3iwDWnyObY2qv5AJeFerpemXMxoZ96XnuUvl62xXcsfchKW7HsHYBvZ2jkAgPchD-aLAebJAcV56W-1B6ttzexoXHElaWhkkGd5FC-7iA6LuqylJKNCE2gzEhQuLifiDMbLAZSbGyjvPM4NcGobMREnDXT6DGq9AOMkDqcJADRW0OCgn809Dyo9JjrmBvqR9A-bMbra-xrmzebhHwABb1U_uY6F-aTKJ6DdDOBhLXC1A8COBmC5GdBsae43OnadDTCRHmADD0zFDEjGAU08gGFvgN4sQMcTAKdVLJ2I-TE5Zj2l8JiMh3OSukee2BaXtjjx6FrgaSVI1CyaGDDm7OmQoJPC9008sUAxsFnl2VROxt1xxxN6mihn1NioqSVfkhua8wdgmg4IktkVhc170gV8l-3Olk4WKg63WOhdNrEgmut0jybb43lJZ3HUz7SE2TXeYY1beU5INMXAAYbLXIZ0d_k0c_ZUQ3Bh5zm1LgIUJl973AzZ1M-79X_etlokad2e0o6uypFItw5Cs5I5lEIJW30ME3LTHH0o7rNXRHSewn2V97tdUdA04BbdaNFVNpGRtn6VeI9BxQ9MJxpXrZJ4WjfvOtwjfb_InISi7XBfY1NrwvLAIoBD0DjTHe063KB163p1eJCqNgoSWWBUIuCNDqvGgbZo2Ug9k2A9QC4RBusaZw4jxHpA1U6_7gcD1kxgh5xTsLnT6juYdIU66bY6c8wOKF2p3ozGpenJ3kC-i3MNS6rTWrFD7NIRGU5gFMZpF7Xbj_58yq7N4BAEYqCEd4iNa5NOLFOPSJ7VdddQKCQRKWMavpGvBYN7NyfS7gmQ3YirHoIb-qGNsfsSN6y-shP8xgC_rSfsUNo5PPH9srS7oaaibkIqva7rK2-ZAR67nCPsMTaR8GWrS_h8dLeeGiM83jjh1TrZLdIOqUuJO1YOnMxbvKpnTNVODRNwFFfttnfBDYrjKU5VpxoJz3nAOqsfA10FplXFLu9RM8I-Ck86GGXI2qc8G6KJlsGpNsedfTk_cP3aXBkhao51qx_QVHqwZbIeZgERdyKTAuog0JWl12W1TzG_OAJHvcTeljZH6Xghu2RnR6fUoxzpcOOlB4LDzEGgSA6DXN9qzNuVTrMGyMMeXI8JXSsgoqs3ZjM4xmzUsOrGE4045NGBsOSSw3auSaVE28Cjr5Z7xBCb4gFSlfSI2uIfUebjZulyo33zkhzVxvM9S8-aQeSmFz88x-sKQkHuW0tCO62NPa1EZljfDjydY5samIKNnJGrokfG3OeJYdumcReZqM_JnPSTS4FvsJY9F1KvnhjPEfLOvkADozEF5-Iqu5xmjvU6og3azt6aHVOyu72zMxDNY70WlXwmC4C8JsimmyYjYh9wPOPZ1FAtm3rCOZv47qTidiEdaMzZYSK1JzGU8rQOjeqrqCU0cO9ouHYbhW5K4V6n_YgPLNC0WUFuwtxbWsEimM2g9jGd87EVm31wT5txeYOVOFtqy4HjGpIQid4LIWawfidfdMp1PGU6cces66VauepOXGDo9UKjsKGnrtmSJL9hMYdlWacl85LNWSJsM35dMOSVKLvtw-1C9wJxBS_Nc6EOR2EvO3FHbdXxbIcplp7lx_BoZIieDic9bRTWcqN9ENISr6kSn132ys5L3FjBHW8NNISHhAjdCJNZRSwp_FRjCOIccGp5w8qFdzajAG9ZMblDGsRcpFX1Fu5uvWW1NznVc0UOK0y74jYKxSS6mMQoHk5GTaO9utmEru0LiB8m4RiiYowsi7VWepr0nBtF326Ex4UcyPYnh_e6CNVsddtbfdibUYowmVHZCO0QTeZgcewC7c7wqNrOy4G6ip24lCyYtOzpLLfstT14p74TA7TC57Ci2ttVlahrOJFX_SKwLgcci7uBEXW9fks6UDMDj9jqQn_EaIGMQzxz4jY3gkHJq3U0-NRAudfryAFiGmTL9fKaubT7ZrTnEWE3IcUTrkpcsVKTg1C8J3sVl4OTGXvOOvEu_WN2N1UMEALfHi0KdgLZiOYJH4ZbNRWaZD5I69iAB2T3hhrN9RCSg9Pfk0zkDiih9jkvQ7uvKW3aU4zePx4xFKII7gVdoO-qlftRyrT0koFWPjcHQuZ9yIGcCUnGv0xRIamSh0vpfvamVutPFlk91lQVhVdn0KRGzikYWCbeF4uNADPjZY_Qeq1epnYQkQujV1Vl9xUdU067qzmkqKwKyjTnlQWpZmxt2seaWjdT5YbXukRPTp6KXM6RluntiVMoNkDdO8Xpds-u6Xyc5PnsXbe8fdUTaGGGSEipqYLpLBeyKGM5myqBlCdCdRRzqGZn5oQCW7LNrt5D2z_yjjS3anosT_FdCG3HFUYVzRTtjs4qMieN5VFUG9inpcZZEzubiA1lRh60zOGDiY59O8pNeFFY6dIun0HIbrPU326IPJ25MTgTim8ptF3f94LRk_JVCaYhLKiBmRoAMhJ0LXUAebdpd0kEjxmckzafrcxkD2KMKvde2q21x8FIT8pWOV6Yyjkycl2ssfoMW68jR7OlrJhVd7xmM3ab03AjUQzP8j5qtoBHxrRh-8licget0takj4xKTJaEX3d65eaFQW-wwKW4bV8oJ4XwcoS8GdA3ELexp4vQgnhzwIn-EHE9NS4nuT34cn1e_9eNXDPUZuapG-cdQ9_efgh32d5IL_05327_WFn_rhj-rK-NdwpzK5s8bNqfFNk49S-V2cpO__epwPJhoAmOxT-oAPozKrCY_UQpZPHSx_9TFiAW7xF-RL0bFxX_51QsLPzlrHRh8NJ5Bfr13nbLFP8D7Sc1-1v92_Lkda0EnOBforkiUGd9cWZNaEx8xqakESnbbGLxRZxxBUsef2n_RMR-IHkKf0D5DPLpBIv7NloC-GOc_3-Ef6CZTy8vf_1B0j4Hv1v7D9GS0Y_7T7Ozel7dmzhYvXZNHz6vlu2Rw_fh6oMpL7jeGfTb6pNLNenbauG_y5wKFm5Z5v-c1pT9PVq93mDWLqO-WmhXyMfw3sD8h3Rhzwvh3pZ90a1eCZJhP6ysXr-txtXrC0bSX2mMJVCM4Bbyi1H082paveIk95XhWJLGOIxEaZz6_ryaPxxjX1EOJygOpVmUonCKxZ9XYfBOsOXP3wAffwO-_wOtfTs1)

```mermaid
%%{init: {'themeVariables': {
  'nodeBorder': 'transparent',
  'primaryColor': 'transparent'
}}}%%

flowchart LR
    Ingestion@{ img: "https://repository-images.githubusercontent.com/826168160/d3c8a8f9-af99-449f-856b-4ab9c897cce2", label: "**Ingestion**:\nDocling", pos: "t", w: 100, h: 100, constraint: "on" }
    Chunking@{ img: "https://avatars.githubusercontent.com/u/205278415?s=280&v=4", label: "**Chunking**:\nChonkie", pos: "t", w: 0, h: 100, constraint: "on" }
    Classification@{ img: "data:image/png;base64,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", label: "**Classification**:\nTransformers", pos: "t", w: 250, h: 100, constraint: "on" }
    NER@{ img: "https://avatars.githubusercontent.com/u/161639825?s=200&v=4", label: "**NER**:\nGliNER2", pos: "t", w: 0, h: 100, constraint: "on" }
    InformationExtraction@{ img: "https://encrypted-tbn0.gstatic.com/images?q=tbn:ANd9GcShzp30XASXzPGrU2z1yjrI5WUriI-Iz2N1jw&s", label: "**Information extraction**:\n LangChain", pos: "t", h: 100, constraint: "on" }

    Ingestion --> Chunking --> Classification --> NER --> InformationExtraction
```
## Declarative Document AI

`sieves` decouples the business logic - the "what" - from the execution framework - the "how."

You focus on the data you need and the **structure it should conform to**, while the framework handles execution. Because `sieves` provides a unified interface over multiple backends for structured generation, you can swap execution engines by changing a single parameter. The rest of your pipeline remains untouched.

`sieves` acts as a cohesive layer for the entire document AI lifecycle:

1.  **Ingestion**: Standardized parsing of PDFs, images, and Office docs via **[docling](https://github.com/DS4SD/docling)**.
2.  **Preprocessing**: Built-in text chunking and windowing via **[chonkie](https://github.com/chonkie-inc/chonkie)**.
3.  **Task Library**: A collection of ready-to-use tasks like NER, classification, and summarization. Skip the prompt engineering and focus on the schema.
4.  **Prediction**: Structured generation across `outlines`, `dspy`, `langchain`, `gliner2`, and `transformers` zero-shot classification pipelines.
5.  **Persistence**: Save and load pipelines with their configurations to ensure reproducibility across environments.

## Evaluation and Optimization

A production pipeline is not static. You need to know if it works and how to make it better. `sieves` builds these needs into the core workflow.

**Evaluation** is a first-class citizen. You can measure pipeline performance against ground-truth data using deterministic metrics or LLM-based judging. This allows you to track regression as you update models or prompts.

If your documents contain ground truth/gold data, you can evaluate your pipeline like so:
```python
results = list(pipeline(docs))
eval_report = pipeline.evaluate(results)
# Inspect the evaluation results for pipeline components:
for task_id in eval_report.reports:
    print(eval_report[task_id].summary())
# In the case of our case study (see below) this prints something like
# 'Task: crisis_label_classifier | Metrics: F1 (Macro): 0.4874 | Failures: 48'
# 'Task: crisis_type_classifier | Metrics: F1 (Macro): 0.9563 | Failures: 16'
# 'Task: location_extractor | Metrics: Accuracy: 0.1900 | Failures: 89'
```

**Optimization** is integrated via DSPy's MIPROv2. If your extraction precision is low, `sieves` can automatically optimize your prompts and few-shot examples.

In order to optimize your tasks to have to
- provide your task with fewshot examples
- initialize an optimizer
- run `task.optimize()`

Use it like this:
```python
task = Classification(..., fewshot_examples=...)
optimizer = Optimizer(...)
best_prompt, best_examples = task.optimize(optimizer)
```

**Distillation** completes the cycle. Once you have a high-performing pipeline using a large LLM, you can distill that logic into a specialized local model using **SetFit** or **Model2Vec**. This reduces costs and latency without a total rewrite of your application.

In order to e.g. distill a classification task with `setfit`, you call the tasks `.distill()` method:
```python
task = Classification(...)
# This distills a `setfit` model from the results in `docs` and stores it at `output_path`.
task.distill(..., data=docs, framework=DistillationFramework.setfit, output_path=...)
```

## Small Abstractions

We built `sieves` around three objects:

- **Doc**: The atomic unit of data. It holds text, metadata, and the pipeline's history.
- **Task**: A reusable unit of work. It encapsulates logic and schema validation.
- **Pipeline**: A sequence of tasks. It manages execution, caching, and conditional logic.

Tasks are portable. A sentiment analysis task defined for a prototype with GPT-4 will work with a local Llama model. The schema is the contract.

## Case Study: Filtering the Noise

In our **[Crisis Tweet Case Study](https://sieves.ai/demos/crisis_tweets)**, we used the CrisisNLP dataset to solve a common engineering problem: noise. Social media text is informal and often irrelevant to emergency response.

Running complex extraction on every noisy tweet wastes money, time, and introduces avoidable errors.
To address this, we built a multi-stage pipeline. A classifier identifies if a tweet is crisis-related. A "gatekeeper" condition determines if the expensive extraction tasks should run.

This kind of conditional execution is straightforward only when classification, extraction, and orchestration are expressed as first-class pipeline components. Without explicit task boundaries and structured results, this logic quickly devolves into ad-hoc control flow and duplicated checks scattered across the codebase.

```python
def related_to_crisis(doc: Doc) -> bool:
    result = doc.results.get("crisis_label_classifier")
    return result and result.label != 'irrelevant'

pipeline = (
    crisis_label_classifier +
    tasks.InformationExtraction(
        task_id="location_extractor",
        entity_type=Country,
        model=model,
        condition=related_to_crisis
    )
)
```

Filtering the noise before the later extraction stage can significantly improve the reliability of your pipeline. Don't ask your model to extract entities from irrelevant text, and you'll end up with a more resilient, accurate, and cheaper system.

## What v1.0 Means

We are releasing v1.0 to signal API stability. We have used `sieves` in production for complex, multi-stage document processing. The abstractions are robust enough to handle the rapid evolution of the underlying LLM ecosystem.

We are committed to stability. The underlying models and frameworks will change. Your `sieves` pipelines should remain a stable part of your infrastructure.

We also presented an earlier version of `sieves` at PyData Amsterdam 2025, where we walked through the design rationale, core abstractions, and examples. The recording is available [here](https://www.youtube.com/watch?v=5i8tEvrYEyQ).

## When to Use sieves

`sieves` is for teams building document-centric NLP pipelines.

- **Good fits**: Structured data extraction, multi-stage processing, and moving from prototypes to production without backend lock-in.
- **Poor fits**: Chatbots, RAGs or simple one-off LLM calls where a single prompt suffices.

***

*`sieves` is open-source under a MIT license and available on **[GitHub](https://github.com/MantisAI/sieves)**. Read the documentation and see the full case study at **[sieves.ai](https://sieves.ai)**.*
