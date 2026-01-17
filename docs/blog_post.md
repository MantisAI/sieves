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

[![](https://mermaid.ink/img/pako:eNqlWAmP47iV_isFA5MBRtVN3UcFs4lt2bIs67Csy0IBAXXYum_Jkhr931dVNZn0dHp3kyxhWOTj4-P7yEfpffyy8ssgXL2sfvrpS1zE3cuX1-Lp6ecuCvPQgk0MvSxsf355ehcvHVUT57CZtmVWNov4566BRVvBJiy6n59_0ykWi5uyCcL_XSO9f9_91vv1tfj69aefXovX4paVDz-CTfd00j8G-hlsWz68PX0zalOOT7c4y16-kT23XVOm4beiP79ZfLMhFvew7eKy-OuXpzi_vzy9rqKuq9oXAJqwKtu4K5vp04JyUft8j7uo9_o2bPyy6BYzn_0yByxOYzSL0SgICJ-F7I37BG8c94kkudsnlqK9TyT0OJ_lGN8P8dfV81MGvTB7m-qXX36f_5dfXl5fC770s7i4vyktk7-pdG_1x8sThqLPT9Fvz2X6BROMi-5NpSxeV28r9YZnG_VFulj4ZzhwgB1s_icQPcBRCmdYEqP-0v6Ks-ifhl_J73z9u-0PV7dRubTCH7j6rzn6tnnxLfbhH1c_WLx8eV9vUBX3P3uwDWnyObY2qv5AJeFerpeiXMxoZ96XmuUvf1tju5bf5SQs2bcKwDaytXMAAG9NHswXA8yTA4rzUt9qD1bbmtnRuOJK0tDIIM_yKF52ER0WdVlLSUaFJtBmJChcXE7EGYyXAyg3N1DeeZwb4NQ2YiJOGuj0GdR6AcZJHE4TAGisoMFBP5t7HlR6THTMDfQj6R82Y3S19zXMm83DPwAC3qp-ch0L80mVT0C7GcDDWtzVDgA7GoDlZkCzpbnf6Nh1NsBEeoANPDAVMyAZBzTxAIa9AXqzAB1PAJxWsXQi5sfkmPWUwmMyHs5J6h55Yltc2uLEo2uBp5UgUbNoYsCYs6dDgk4K3zfxxALFwGaVZ1M5GXfHHU_oaaKcUWOjppZ8SW5ozh-AaTogSGZXFDZviy7gu2x3tnSyUHG4xULvsokF0VynezTZHs_LchZH_UxLmF3jHda4leeERFMMHGC4zGVId5dPM2dPNQQXdp5T6yJAYfK1x82QTf28W__nZatFktbtKe3oqhyJdOsgNCuZQymUsNXHMCE3zdGH4j57RUTnKdxXeb_bFQVNA27RjRZdZRMZaetXifcYVPzAdKJx1SqJp3XzrsM90veLzEko2g73NTa1JiwPLAI4BI0z3dGuww1at65XhwepaqMgkQVGJQLe6LBqHGiLlo3UMwnWA-QSYbCuceYwQqwHVO30634wYM0Edsg5BZs7rb6DSVeok26rM8fsgNKV6s1oXJqe7A3kuzjXsKQ6rRU7xC4dkeEERmGcdlG7_ejPp-zaDA5BIAZKeIfYuDbpxDL1iORZXXcNhUISkTKm4Rv5WjC4d3Mi7Z4A2Y246iG4oR_aGLsvccPqKzvBbwzw23rCDqWdwxPfL1u7G2oq6iak0uu6vvKWGeCxyznCHmMTCV9CXcLno7v11Bjh8cYJr9bJbpF2SF1K3LFy4GTeMqt6xlTt1DABR3HVbnsX3KA4nuJUdaqR8JwHrLP6MdBVYFpV7PIeNSPso_Ckg1GGrH3KsyGaaBmcanPc2ZfzA9evzZURouZYt_oBTaUHWybrYRYQcScyKaAOAl1Zel1W-xTziyNw1EvsbWlzlI4Xskt2dnRKPcqRDjdeeiA4zBwEiuQwyPWtxrxd6TRrgDzswfWY0LUCIrp6YzaDY8xGDatuPNGIQx4dCEsuOWznmlRKtA08-mq5RwyxKR4gVUmPqC3-EWU-bpYuN9o3L8lRbTzfs_SsGURuevHDc7yuIBTkvrUktNPa2NNKZIb17cDTObapgSnYyBm5KnpkzH2eGLZtGneRifqczEk_uRT4BmvZcyH16onxHCHv7As0MBpTcC6usstp5livI9qg7eyt2TElu9s7OwPRPNZrUclnsgDIa4JsumkyIvYBxzOeTQ3VsqknnLOJ704qbhfSgcacHSZSexJDKU_r0Ki-ilpCA_eOhmu3UeimFO512o_4wAJNmxXkJsy9pRUsgtkMah_TOR9bsdkH97QZlzdYibOlthw4riEJkei9EGIG63fyRadcx1OmE3fMul6qlavuxAWGXi80Cht66potSfIbFnNYlnVaMi_ZnCXCNuPXBUNeibLbPtwudC8QV_DSPBfqcBT2shN31FYdz3aYYulZfgyPRobo6XDS00ZhLTfaByEt8Zoq8dllr-y8xI0V3PHWQEN4SIjQjTCZVcSSwk81hiDOAaeWN6xceGczCvCWFZM7pEHMRVpVb-Hu1ltWe5NTPVfksMK0K26jUEyii0mM4uFk1DTaq5tN6Nq-gPhhEo4hKsbIsllrpadJz7lR9O1GeFzIgWx_cnivi1DNVre91Ye9GaUIkxmVjdAO0WQOFscu0O4Mj6rtvByoq9iJS8qCSUtMZ7llr-3BO_WdGKAVPocV1d6uqkRdw4m86heBdTngWNwNjKjr9VvSgZoZeMRWF_ojRgtkHOKZE7e5EQxKXq2jwacGyr1eRw4Q0yBbrpfXzKXdN6M9jwi7CSmecFXiipWaHITiPdmruByczNhz1ol36R-zu6ligBD49mhRsBPIRjRP-DDcqqnQJPNBWscGPCC7N9RoroeQHJz-nmQid0AJtc95Gdp9TWnTnmL0_vGIoRBFcC_oAn1XrdyPUqallxVo5XNzIGTehxzImZBk_MsUFZIqebiU7mdvarX-ZJHVY01VUXh1Bk1q5JyCgWXifbHYCDAzXmKE1mv1MrWDiFwYvaoqu6_omHLaXc0hRWVVUKY5ryxINWNr0z7W1LqZKje81iV6cvJU5HKOtExvT5xCsQHq3ilOt3t2TefjJM9n77rl7aueQAszREJKTRVMZ7mQRRnL2VQJpDwRqqOYQzU7MycU2JJtdvUe2v6Rd6S5VdNjeYrvQmg7rjCqaKZod3RWkTlpLI-i2sA-LTnOmtjZRGwoM_KgZQ4fTHTs21FuwovCSpd2-QxCdpul_nZD5OnMjcGZUHxLoe36vheMnpSvSjANYUENzNQAkJGga6kDyLtNu0sieMzgnLT5bGUmexBjVLn30m6tPQ5GelK2yvHCVM6RketijdVn2HodOZotZcWsuuM1m7HbnIYbiWJ4lvdRswU8MqYN208WkztolbYmfWRUYrIk_LrTKzcvDHqDBS7FbftCOSmElyPkzYC-gbiNPV2EFsSbA070h4jrqXE5ye3Bl-vz-v9dyDVDbWaeunHeMfTt7btwl-2N9NKf8-32-8z6D8nwR35tvLGVW9nkYdP-IMnGqX8pzVZ2-r9PBZYPA01wLP5OBdAfUYHF7IeXQhYvdfw_ZQFi8YbwHfVuXFT8H1OxsPCXs9KFwafOK9DP97Zbhvjv3n5Qs7_Uvy49L2sl4AT_Es0VgTrrizNrQmPiMzYljUjZZhOLn8QZV7Dk8af2n4jY7548hb-78gHy6QSL-zZaAHyP8_9GyMcLvcsy-GNgHzvwDkReCHcbnaAH8oUPZ_gQ-qCBD5AvVsESHmHX_gb2Hwp_y8p7-bn6IIzfgPl20t8gyO9jrND_9yF8x5SfPn36r9955kfjD-H7LlqC4v35ww1-7_nWx9Xz6t7EwWqh6H34vFpCfkG9NFfvFw2Lo28XEK-rD37YpK-r1-LrMqaChVuW-d-HNWV_j1YvN5i1S6uvFioZ8jG8N_AfKmERhM227Itu9UK9W1i9fFmNqxcMYz-zS8wTHEZQDIlj3PNqWsQc_RlHOZYlaZKgKJL5-rya3-dEP3M0iS-_5aRgBIMzzPMqDN7uC-SPO5T3q5Sv_w0kkagn?type=png)](https://mermaid.live/edit#pako:eNqlWAmP47iV_isFA5MBRtVN3UcFs4lt2bIs67Csy0IBAXXYum_Jkhr931dVNZn0dHp3kyxhWOTj4-P7yEfpffyy8ssgXL2sfvrpS1zE3cuX1-Lp6ecuCvPQgk0MvSxsf355ehcvHVUT57CZtmVWNov4566BRVvBJiy6n59_0ykWi5uyCcL_XSO9f9_91vv1tfj69aefXovX4paVDz-CTfd00j8G-hlsWz68PX0zalOOT7c4y16-kT23XVOm4beiP79ZfLMhFvew7eKy-OuXpzi_vzy9rqKuq9oXAJqwKtu4K5vp04JyUft8j7uo9_o2bPyy6BYzn_0yByxOYzSL0SgICJ-F7I37BG8c94kkudsnlqK9TyT0OJ_lGN8P8dfV81MGvTB7m-qXX36f_5dfXl5fC770s7i4vyktk7-pdG_1x8sThqLPT9Fvz2X6BROMi-5NpSxeV28r9YZnG_VFulj4ZzhwgB1s_icQPcBRCmdYEqP-0v6Ks-ifhl_J73z9u-0PV7dRubTCH7j6rzn6tnnxLfbhH1c_WLx8eV9vUBX3P3uwDWnyObY2qv5AJeFerpeiXMxoZ96XmuUvf1tju5bf5SQs2bcKwDaytXMAAG9NHswXA8yTA4rzUt9qD1bbmtnRuOJK0tDIIM_yKF52ER0WdVlLSUaFJtBmJChcXE7EGYyXAyg3N1DeeZwb4NQ2YiJOGuj0GdR6AcZJHE4TAGisoMFBP5t7HlR6THTMDfQj6R82Y3S19zXMm83DPwAC3qp-ch0L80mVT0C7GcDDWtzVDgA7GoDlZkCzpbnf6Nh1NsBEeoANPDAVMyAZBzTxAIa9AXqzAB1PAJxWsXQi5sfkmPWUwmMyHs5J6h55Yltc2uLEo2uBp5UgUbNoYsCYs6dDgk4K3zfxxALFwGaVZ1M5GXfHHU_oaaKcUWOjppZ8SW5ozh-AaTogSGZXFDZviy7gu2x3tnSyUHG4xULvsokF0VynezTZHs_LchZH_UxLmF3jHda4leeERFMMHGC4zGVId5dPM2dPNQQXdp5T6yJAYfK1x82QTf28W__nZatFktbtKe3oqhyJdOsgNCuZQymUsNXHMCE3zdGH4j57RUTnKdxXeb_bFQVNA27RjRZdZRMZaetXifcYVPzAdKJx1SqJp3XzrsM90veLzEko2g73NTa1JiwPLAI4BI0z3dGuww1at65XhwepaqMgkQVGJQLe6LBqHGiLlo3UMwnWA-QSYbCuceYwQqwHVO30634wYM0Edsg5BZs7rb6DSVeok26rM8fsgNKV6s1oXJqe7A3kuzjXsKQ6rRU7xC4dkeEERmGcdlG7_ejPp-zaDA5BIAZKeIfYuDbpxDL1iORZXXcNhUISkTKm4Rv5WjC4d3Mi7Z4A2Y246iG4oR_aGLsvccPqKzvBbwzw23rCDqWdwxPfL1u7G2oq6iak0uu6vvKWGeCxyznCHmMTCV9CXcLno7v11Bjh8cYJr9bJbpF2SF1K3LFy4GTeMqt6xlTt1DABR3HVbnsX3KA4nuJUdaqR8JwHrLP6MdBVYFpV7PIeNSPso_Ckg1GGrH3KsyGaaBmcanPc2ZfzA9evzZURouZYt_oBTaUHWybrYRYQcScyKaAOAl1Zel1W-xTziyNw1EvsbWlzlI4Xskt2dnRKPcqRDjdeeiA4zBwEiuQwyPWtxrxd6TRrgDzswfWY0LUCIrp6YzaDY8xGDatuPNGIQx4dCEsuOWznmlRKtA08-mq5RwyxKR4gVUmPqC3-EWU-bpYuN9o3L8lRbTzfs_SsGURuevHDc7yuIBTkvrUktNPa2NNKZIb17cDTObapgSnYyBm5KnpkzH2eGLZtGneRifqczEk_uRT4BmvZcyH16onxHCHv7As0MBpTcC6usstp5livI9qg7eyt2TElu9s7OwPRPNZrUclnsgDIa4JsumkyIvYBxzOeTQ3VsqknnLOJ704qbhfSgcacHSZSexJDKU_r0Ki-ilpCA_eOhmu3UeimFO512o_4wAJNmxXkJsy9pRUsgtkMah_TOR9bsdkH97QZlzdYibOlthw4riEJkei9EGIG63fyRadcx1OmE3fMul6qlavuxAWGXi80Cht66potSfIbFnNYlnVaMi_ZnCXCNuPXBUNeibLbPtwudC8QV_DSPBfqcBT2shN31FYdz3aYYulZfgyPRobo6XDS00ZhLTfaByEt8Zoq8dllr-y8xI0V3PHWQEN4SIjQjTCZVcSSwk81hiDOAaeWN6xceGczCvCWFZM7pEHMRVpVb-Hu1ltWe5NTPVfksMK0K26jUEyii0mM4uFk1DTaq5tN6Nq-gPhhEo4hKsbIsllrpadJz7lR9O1GeFzIgWx_cnivi1DNVre91Ye9GaUIkxmVjdAO0WQOFscu0O4Mj6rtvByoq9iJS8qCSUtMZ7llr-3BO_WdGKAVPocV1d6uqkRdw4m86heBdTngWNwNjKjr9VvSgZoZeMRWF_ojRgtkHOKZE7e5EQxKXq2jwacGyr1eRw4Q0yBbrpfXzKXdN6M9jwi7CSmecFXiipWaHITiPdmruByczNhz1ol36R-zu6ligBD49mhRsBPIRjRP-DDcqqnQJPNBWscGPCC7N9RoroeQHJz-nmQid0AJtc95Gdp9TWnTnmL0_vGIoRBFcC_oAn1XrdyPUqallxVo5XNzIGTehxzImZBk_MsUFZIqebiU7mdvarX-ZJHVY01VUXh1Bk1q5JyCgWXifbHYCDAzXmKE1mv1MrWDiFwYvaoqu6_omHLaXc0hRWVVUKY5ryxINWNr0z7W1LqZKje81iV6cvJU5HKOtExvT5xCsQHq3ilOt3t2TefjJM9n77rl7aueQAszREJKTRVMZ7mQRRnL2VQJpDwRqqOYQzU7MycU2JJtdvUe2v6Rd6S5VdNjeYrvQmg7rjCqaKZod3RWkTlpLI-i2sA-LTnOmtjZRGwoM_KgZQ4fTHTs21FuwovCSpd2-QxCdpul_nZD5OnMjcGZUHxLoe36vheMnpSvSjANYUENzNQAkJGga6kDyLtNu0sieMzgnLT5bGUmexBjVLn30m6tPQ5GelK2yvHCVM6RketijdVn2HodOZotZcWsuuM1m7HbnIYbiWJ4lvdRswU8MqYN208WkztolbYmfWRUYrIk_LrTKzcvDHqDBS7FbftCOSmElyPkzYC-gbiNPV2EFsSbA070h4jrqXE5ye3Bl-vz-v9dyDVDbWaeunHeMfTt7btwl-2N9NKf8-32-8z6D8nwR35tvLGVW9nkYdP-IMnGqX8pzVZ2-r9PBZYPA01wLP5OBdAfUYHF7IeXQhYvdfw_ZQFi8YbwHfVuXFT8H1OxsPCXs9KFwafOK9DP97Zbhvjv3n5Qs7_Uvy49L2sl4AT_Es0VgTrrizNrQmPiMzYljUjZZhOLn8QZV7Dk8af2n4jY7548hb-78gHy6QSL-zZaAHyP8_9GyMcLvcsy-GNgHzvwDkReCHcbnaAH8oUPZ_gQ-qCBD5AvVsESHmHX_gb2Hwp_y8p7-bn6IIzfgPl20t8gyO9jrND_9yF8x5SfPn36r9955kfjD-H7LlqC4v35ww1-7_nWx9Xz6t7EwWqh6H34vFpCfkG9NFfvFw2Lo28XEK-rD37YpK-r1-LrMqaChVuW-d-HNWV_j1YvN5i1S6uvFioZ8jG8N_AfKmERhM227Itu9UK9W1i9fFmNqxcMYz-zS8wTHEZQDIlj3PNqWsQc_RlHOZYlaZKgKJL5-rya3-dEP3M0iS-_5aRgBIMzzPMqDN7uC-SPO5T3q5Sv_w0kkagn)

<!--
%%{init:{
  'themeVariables': {
    'primaryColor': 'transparent',
    'nodeBorder': 'transparent',
    'nodeBkg': 'transparent'
  }
}}%%

flowchart LR
    classDef transparentBox fill:transparent,stroke:transparent;

    Ingestion@{ img: "https://repository-images.githubusercontent.com/826168160/d3c8a8f9-af99-449f-856b-4ab9c897cce2", label: "**Ingestion**:\nDocling", pos: "t", w: 100, h: 100, constraint: "on" }
    Chunking@{ img: "https://avatars.githubusercontent.com/u/205278415?s=280&v=4", label: "**Chunking**:\nChonkie", pos: "t", w: 0, h: 100, constraint: "on" }
    Classification@{ img: "data:image/png;base64,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", label: "**Classification**:\nTransformers", pos: "t", w: 250, h: 100, constraint: "on" }
    NER@{ img: "https://avatars.githubusercontent.com/u/161639825?s=200&v=4", label: "**NER**:\nGliNER2", pos: "t", w: 0, h: 100, constraint: "on" }
    InformationExtraction@{ img: "https://encrypted-tbn0.gstatic.com/images?q=tbn:ANd9GcShzp30XASXzPGrU2z1yjrI5WUriI-Iz2N1jw&s", label: "**Information extraction**:\n LangChain", pos: "t", h: 100, constraint: "on" }
    Distillation@{ img: "https://github.com/MinishLab/model2vec/raw/main/assets/images/model2vec_logo.png", label: "**Distillation**:\n Model2Vec", pos: "t", h: 100, constraint: "on" }

    Ingestion -> Chunking -> Classification -> NER -> InformationExtraction -> Distillation
-->

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
