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

`sieves` lets you mix and match language model frameworks within a single pipeline. That means you can assign the best tool for each task instead of being locked into one framework. For example, you might use a fast local model for document classification, a lightweight frame like `gliner2` for NER, and a more expressive LLM framework like `langchain` for challenging information extraction. With `sieves`, you can define and orchestrate this heterogeneous pipeline without rewriting your business logic:
[![](https://mermaid.ink/img/pako:eNqlWHmvo0iS_ypPlnpaal5Vch9v1DtrGxtjzGHMZfT-SQ6b-wYDpfrui1_NlLprandnZ5FlMiIj48pIiB9fVn4ZhKu31S-_fImLuHv78l68vPzaRWEeWrCJoZeF7a9vLx_sZaJq4hw207bMymZh_9o1sGgr2IRF9-vr32WKReOmbILwf5ZI7z9OP2e_vhdfv_7yy3vxXtyy8uFHsOleTvq3hX4G25YPby9_WLUpx5dbnGVvf-C9tl1TpuEfWX99anzqEIt72HZxWfznl5c4v7-9vK-irqvaNwCasCrbuCub6dMS5SL2-R53Ue_1bdj4ZdEtaj77ZQ5YnMZoFqNREBA-C9kb9wneOO4TSXK3TyxFe59I6HE-yzG-H-Lvq9eXDHph9jT122_f7f_229v7e8GXfhYX96fQYvwp0j3Hj7cXDEVfX6K_3xfzS0wwLrqnSFm8r56ZesazjfoiXTT8czhwgB1s_rsgeoCjFM6wJEb9rf0dZ9G_DL-TP_j6D93fXN1G5UKFP3H1X3P0uXnxLfbhn7MfLF6-feQbVMX9rx5sQ5p8ja2Nqj9QSbiX6-VSLma0M-_LyPKXv62xXcsffBKW7HMAsI1s7RwAwJPkwXwxwDw5oDgv4632YLWtmR2NK64kDY0M8iyP4mUX0WFRl7WUZFRoAm1GgsLF5UScwXg5gHJzA-Wdx7kBTm0jJuKkgU6fQa0XYJzE4TQBgMYKGhz0s7nnQaXHRMfcQD-S_mEzRld7X8O82Tz8AyDgreon17Ewn1T5BLSbATysxV3tALCjAVhuBjRbmvuNjl1nA0ykB9jAA1MxA5JxQBMPYNgboDcL0PEEwGkVSydifkyOWU8pPCbj4Zyk7pEntsWlLU48uhZ4WgkSNYsmBow5ezok6KTwfRNPLFAMbFZ5NpWTcXfc8YSeJsoZNTZqasmX5Ibm_AGYpgOCZHZFYfNMuoDvst3Z0slCxeEWC73LJhZEc53u0WR7PC_pLI76mZYwu8Y7rHErzwmJphg4wHCZy5DuLp9mzp5qCC7sPKfWRYDC5GuPmyGb-nm3_vevrRZJWrentKOrciTSrYPQrGQOpVDCVh_DhNw0Rx-K--wVEZ2ncF_l_W5XFDQNuEU2WmSVTWSkrV8l3mNQ8QPTicZVqySe1s27DvdI3y88J6FoO9zX2NSasDywCOAQNM50R7sON2jdul4dHqSqjYJEFhiVCHijw6pxoC1aNlLPJFgPkEuEwbrGmcMIsR5QtdOv-8GANRPYIecUbO60-g4mXaFOuq3OHLMDSleqN6NxaXqyN5Dv4lzDkuq0VuwQu3REhhMYhXHaRe32oz-fsmszOASBGCjhHWLj2qQTy9Qjkmd13TUUCklEypiGb-RrweDezYm0ewJkN-Kqh-CGfmhj7L7EDauv7AS_McBv6wk7lHYOT3y_bO1uqKmom5BKr-v6yltmgMcu5wh7jE0kfCl1CZ-P7tZTY4THGye8Wie7RdohdSlxx8qBk3mLVfWMqdqpYQKO4qrd9i64QXE8xanqVCPhOQ9YZ_VjoKvAtKrY5T1qRthH4UkHowxZ-5RnQzTRMjjV5rizL-cHrl-bKyNEzbFu9QOaSg-2TNbDLCDiTmRSQB0EurL0uqz2KeYXR-Col9jb0uYoHS9kl-zs6JR6lCMdbrz0QHCYOQgUyWGQ61uNebvSadYAediD6zGhawVEdPXGbAbHmI0aVt14ohGHPDoQllxy2M41qZRoG3j01XKPGGJTPECqkh5RW_wjynzcLF1utG9ekqPaeL5n6VkziNz04ofneF1BKMh9a0lop7Wxp5XIDOvbgadzbFMDU7CRM3JV9MiY-zwxbNs07iIT9TmZk35yKfAN1rLnQurVE-M5Qt7ZF2hgNKbgXFxll9PMsV5HtEHb2VuzY0p2t3d2BqJ5rNeiks9kAZDXBNl002RE7AOOZzybGqplU084ZxPfnVTcLqQDjTk7TKT2JIZSntahUX0VtYQG7h0N126j0E0p3Ou0H_GBBZo2K8hNmHtLK1gEsxnUPqZzPrZisw_uaTMuT7ASZ0ttOXBcQxIi0XshxAzW7-SLTrmOp0wn7ph1vVQrV92JCwy9XmgUNvTUNVuS5Dcs5rAs67RkXrI5S4Rtxq8LhrwSZbd9uF3oXiCu4KV5LtThKOxlJ-6orTqe7TDF0rP8GB6NDNHT4aSnjcJabrQPQlriNVXis8te2XmJGyu4462BhvCQEKEbYTKriCWFn2oMQZwDTi1PWLnwzmYU4C0rJndIg5iLtKrewt2tt6z2Jqd6rshhhWlX3EahmEQXkxjFw8moabRXN5vQtX0B8cMkHENUjJFls9ZKT5Oec6Po243wuJAD2f7k8F4XoZqtbnurD3szShEmMyoboR2iyRwsjl2g3RkeVdt5OVBXsROXlgWTlprOcste24N36jsxQCt8DiuqvV1VibqGE3nVLwLrcsCxuBsYUdfrt6QDNTPwiK0u9EeMFsg4xDMnbnMjGJS8WkeDTw2Ue72OHCCmQbZcL6-ZS7tvRnseEXYTUjzhqsQVKzU5CMV7sldxOTiZseesE-_SP2Z3U8UAIfDt0aJgJ5CNaJ7wYbhVU6FJ5oO0jg14QHZvqNFcDyE5OP09yUTugBJqn_MytPua0qY9xej94xFDIYrgXtAF-q5auR-lTEsvGWjlc3MgZN6HHMiZkGT8yxQVkip5uJTuZ29qtf5kkdVjTVVReHUGTWrknIKBZeJ9segIMDNeaoTWa_UytYOIXBi9qiq7r-iYctpdzSFFZVVQpjmvLEg1Y2vTPtbUupkqN7zWJXpy8lTkco60TG9PnEKxAereKU63e3ZN5-Mkz2fvuuXtq55ACzNEQkpNFUxnuZBFGcvZVAmkPBGqo5hDNTszJxTYkm129R7a_pF3pLlV02N5iu9CaDuuMKpopmh3dFaROWksj6LawD4tPc6a2NlEbCgz8qBlDh9MdOzbUW7Ci8JKl3Z5DUJ2m6X-dkPk6cyNwZlQfEuh7fq-F4yelK9KMA1hQQ3M1ACQkaBrqQPIu027SyJ4zOCctPlsZSZ7EGNUuffSbq09DkZ6UrbK8cJUzpGR62KN1WfYeh05mi1lxay64zWbsduchhuJYniW91GzBTwypg3bTxaTO2iVtiZ9ZFRisiT8utMrNy8MeoMFLsVt-0I5KYSXI-TNgL6BuI09XYQWxJsDTvSHiOupcTnJ7cGX6_P6_32Ra4bazDx147xj6NvbD-Yu2xvppT_n2-2PnfWfmuFv_bXxRCu3ssnDpv1Jk41T_1Kbrez0_zsUWF4MNMGx-AcUQH8GBRa137wUsngZ4_8uChCLZ4QfUe_GRcT_ORQLC385K10YfOq8Av18b7tlif_h7Tdo9rf692Xmba0EnOBforkiUGd9cWZNaEx8xqakESnbbGLxkzjjCpY8_tL-ExD77slL-N2Vb0G-nGBx30ZLAD_G-b9H-APMfPn06T--g7RvxJ_2_oO1ZPTj_tPsrF5X9yYOVguc7cPX1VIeOXySqw9Qvvj1BOvvq29YqknfVwuEXtZUsHDLMv_Hsqbs79Hq7QazdqH6aoFdIR_DewPz79wFKC-ofVv2Rbd6w5gPHau3L6txoTD2M7tUCMFhBMWQOMa9rqaFzdGfcZRjWZImCYoima-vq_nDKvqZo0l8-eFLaREsSWCvqzB4omv52xeHjw8PX_8LJ_Fgyg?type=png)](https://mermaid.live/edit#pako:eNqlWHmvo0iS_ypPlnpaal5Vch9v1DtrGxtjzGHMZfT-SQ6b-wYDpfrui1_NlLprandnZ5FlMiIj48pIiB9fVn4ZhKu31S-_fImLuHv78l68vPzaRWEeWrCJoZeF7a9vLx_sZaJq4hw207bMymZh_9o1sGgr2IRF9-vr32WKReOmbILwf5ZI7z9OP2e_vhdfv_7yy3vxXtyy8uFHsOleTvq3hX4G25YPby9_WLUpx5dbnGVvf-C9tl1TpuEfWX99anzqEIt72HZxWfznl5c4v7-9vK-irqvaNwCasCrbuCub6dMS5SL2-R53Ue_1bdj4ZdEtaj77ZQ5YnMZoFqNREBA-C9kb9wneOO4TSXK3TyxFe59I6HE-yzG-H-Lvq9eXDHph9jT122_f7f_229v7e8GXfhYX96fQYvwp0j3Hj7cXDEVfX6K_3xfzS0wwLrqnSFm8r56ZesazjfoiXTT8czhwgB1s_rsgeoCjFM6wJEb9rf0dZ9G_DL-TP_j6D93fXN1G5UKFP3H1X3P0uXnxLfbhn7MfLF6-feQbVMX9rx5sQ5p8ja2Nqj9QSbiX6-VSLma0M-_LyPKXv62xXcsffBKW7HMAsI1s7RwAwJPkwXwxwDw5oDgv4632YLWtmR2NK64kDY0M8iyP4mUX0WFRl7WUZFRoAm1GgsLF5UScwXg5gHJzA-Wdx7kBTm0jJuKkgU6fQa0XYJzE4TQBgMYKGhz0s7nnQaXHRMfcQD-S_mEzRld7X8O82Tz8AyDgreon17Ewn1T5BLSbATysxV3tALCjAVhuBjRbmvuNjl1nA0ykB9jAA1MxA5JxQBMPYNgboDcL0PEEwGkVSydifkyOWU8pPCbj4Zyk7pEntsWlLU48uhZ4WgkSNYsmBow5ezok6KTwfRNPLFAMbFZ5NpWTcXfc8YSeJsoZNTZqasmX5Ibm_AGYpgOCZHZFYfNMuoDvst3Z0slCxeEWC73LJhZEc53u0WR7PC_pLI76mZYwu8Y7rHErzwmJphg4wHCZy5DuLp9mzp5qCC7sPKfWRYDC5GuPmyGb-nm3_vevrRZJWrentKOrciTSrYPQrGQOpVDCVh_DhNw0Rx-K--wVEZ2ncF_l_W5XFDQNuEU2WmSVTWSkrV8l3mNQ8QPTicZVqySe1s27DvdI3y88J6FoO9zX2NSasDywCOAQNM50R7sON2jdul4dHqSqjYJEFhiVCHijw6pxoC1aNlLPJFgPkEuEwbrGmcMIsR5QtdOv-8GANRPYIecUbO60-g4mXaFOuq3OHLMDSleqN6NxaXqyN5Dv4lzDkuq0VuwQu3REhhMYhXHaRe32oz-fsmszOASBGCjhHWLj2qQTy9Qjkmd13TUUCklEypiGb-RrweDezYm0ewJkN-Kqh-CGfmhj7L7EDauv7AS_McBv6wk7lHYOT3y_bO1uqKmom5BKr-v6yltmgMcu5wh7jE0kfCl1CZ-P7tZTY4THGye8Wie7RdohdSlxx8qBk3mLVfWMqdqpYQKO4qrd9i64QXE8xanqVCPhOQ9YZ_VjoKvAtKrY5T1qRthH4UkHowxZ-5RnQzTRMjjV5rizL-cHrl-bKyNEzbFu9QOaSg-2TNbDLCDiTmRSQB0EurL0uqz2KeYXR-Col9jb0uYoHS9kl-zs6JR6lCMdbrz0QHCYOQgUyWGQ61uNebvSadYAediD6zGhawVEdPXGbAbHmI0aVt14ohGHPDoQllxy2M41qZRoG3j01XKPGGJTPECqkh5RW_wjynzcLF1utG9ekqPaeL5n6VkziNz04ofneF1BKMh9a0lop7Wxp5XIDOvbgadzbFMDU7CRM3JV9MiY-zwxbNs07iIT9TmZk35yKfAN1rLnQurVE-M5Qt7ZF2hgNKbgXFxll9PMsV5HtEHb2VuzY0p2t3d2BqJ5rNeiks9kAZDXBNl002RE7AOOZzybGqplU084ZxPfnVTcLqQDjTk7TKT2JIZSntahUX0VtYQG7h0N126j0E0p3Ou0H_GBBZo2K8hNmHtLK1gEsxnUPqZzPrZisw_uaTMuT7ASZ0ttOXBcQxIi0XshxAzW7-SLTrmOp0wn7ph1vVQrV92JCwy9XmgUNvTUNVuS5Dcs5rAs67RkXrI5S4Rtxq8LhrwSZbd9uF3oXiCu4KV5LtThKOxlJ-6orTqe7TDF0rP8GB6NDNHT4aSnjcJabrQPQlriNVXis8te2XmJGyu4462BhvCQEKEbYTKriCWFn2oMQZwDTi1PWLnwzmYU4C0rJndIg5iLtKrewt2tt6z2Jqd6rshhhWlX3EahmEQXkxjFw8moabRXN5vQtX0B8cMkHENUjJFls9ZKT5Oec6Po243wuJAD2f7k8F4XoZqtbnurD3szShEmMyoboR2iyRwsjl2g3RkeVdt5OVBXsROXlgWTlprOcste24N36jsxQCt8DiuqvV1VibqGE3nVLwLrcsCxuBsYUdfrt6QDNTPwiK0u9EeMFsg4xDMnbnMjGJS8WkeDTw2Ue72OHCCmQbZcL6-ZS7tvRnseEXYTUjzhqsQVKzU5CMV7sldxOTiZseesE-_SP2Z3U8UAIfDt0aJgJ5CNaJ7wYbhVU6FJ5oO0jg14QHZvqNFcDyE5OP09yUTugBJqn_MytPua0qY9xej94xFDIYrgXtAF-q5auR-lTEsvGWjlc3MgZN6HHMiZkGT8yxQVkip5uJTuZ29qtf5kkdVjTVVReHUGTWrknIKBZeJ9segIMDNeaoTWa_UytYOIXBi9qiq7r-iYctpdzSFFZVVQpjmvLEg1Y2vTPtbUupkqN7zWJXpy8lTkco60TG9PnEKxAereKU63e3ZN5-Mkz2fvuuXtq55ACzNEQkpNFUxnuZBFGcvZVAmkPBGqo5hDNTszJxTYkm129R7a_pF3pLlV02N5iu9CaDuuMKpopmh3dFaROWksj6LawD4tPc6a2NlEbCgz8qBlDh9MdOzbUW7Ci8JKl3Z5DUJ2m6X-dkPk6cyNwZlQfEuh7fq-F4yelK9KMA1hQQ3M1ACQkaBrqQPIu027SyJ4zOCctPlsZSZ7EGNUuffSbq09DkZ6UrbK8cJUzpGR62KN1WfYeh05mi1lxay64zWbsduchhuJYniW91GzBTwypg3bTxaTO2iVtiZ9ZFRisiT8utMrNy8MeoMFLsVt-0I5KYSXI-TNgL6BuI09XYQWxJsDTvSHiOupcTnJ7cGX6_P6_32Ra4bazDx147xj6NvbD-Yu2xvppT_n2-2PnfWfmuFv_bXxRCu3ssnDpv1Jk41T_1Kbrez0_zsUWF4MNMGx-AcUQH8GBRa137wUsngZ4_8uChCLZ4QfUe_GRcT_ORQLC385K10YfOq8Av18b7tlif_h7Tdo9rf692Xmba0EnOBforkiUGd9cWZNaEx8xqakESnbbGLxkzjjCpY8_tL-ExD77slL-N2Vb0G-nGBx30ZLAD_G-b9H-APMfPn06T--g7RvxJ_2_oO1ZPTj_tPsrF5X9yYOVguc7cPX1VIeOXySqw9Qvvj1BOvvq29YqknfVwuEXtZUsHDLMv_Hsqbs79Hq7QazdqH6aoFdIR_DewPz79wFKC-ofVv2Rbd6w5gPHau3L6txoTD2M7tUCMFhBMWQOMa9rqaFzdGfcZRjWZImCYoima-vq_nDKvqZo0l8-eFLaREsSWCvqzB4omv52xeHjw8PX_8LJ_Fgyg)

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
    Classification@{ img: "data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAVcAAACTCAMAAAAN4ao8AAAA/1BMVEX///8AAAD/zST/zyX/nQD/zCPw8PCUlJTY2Njr6+vMzMxISEh6enqoqKjl5eU/Pz+dnZ2MjIz/xSH/oBf/ogD29vaysrIjIyP/tRz/qRn/xyIvLy//0iN0dHRQUFD/pRi3t7f/ux4cHBxhYWFqamrBwcH/3afpuyZXV1c4ODj/sBv/wV///PH/1JT/89z/68oUFBR1YzT/y4b/8db/ynz/47X/riv/vFT/uUn/tD3/26O1ky3zwyXUqykaJjxHQjkZJD3CnSsnLD0AGD6NdjOlhy7/xm8LHj0yNDuriy8/NT1zOD8kMjxEJED3RkjNQ0TBOkVMSjf0mDH/UUX/djzZIGBcAAAG2ElEQVR4nO2aC1ebSBiGIUAkF0jCJQISEnJRQ6K1Wq2t1rZpbXe3rnv9/79lZ74ZEmyz9Wyqa/S8zzkVSGaGycPwfTMURQEAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAACPhKPtF5PJZO94+tAdeUpM90503WOwvy+fPXRvngzbnh6mkaFpmuEEnn66/9AdehpM9NBhTkscpjbwvO2H7tITYPpKD6RUgRaF+uuH7tXj56WeFq1ysUaoH8+/9+0ilRXPYvfaVftuOvw4OPxGK4n15jG2rRaprXaWIVW+qz4/Ava9jGvVYilX7GiRd5qXuAuvTaq7dWe9Xn8mXsREajtnOyRWOz97E/NtoOfTrZ66yWBaDtimP1jpLANWe1St3l231519PSOtFxczLlYrvX33+T03bHiTYrky87qx+mlqqtr50a4+Kl7rDrMYn72bfXhPgj/MZh9pwGZeceW18Fo2TVupWj2f7/csqy1HoWmaLDu1rZEvq5hty+pRqqqYDVUd2iZ9XGF18jK2afpK2zJZCbOi+D2rXeYVLWs+svkZ5IE8MdXlbbfXOQ1OPLr7d959pECgGZdnJLikOXpx3bXwaqlqwv6pdUVpiZDb5z+8wnbKHToe8WLmlvhy6M/LqUxEWSQw2RYrY7GhrJqsRH0kKw8ojAvzG+IEI7k/5HG6pVRqopFk1cnJ/XOSibC6UxKJS4tjEWhLkb5XKHfDKw+2alX+aI4vvMqfq1bEoXrA/+wWvZb7eZVd3hYbxlz/Ji8hr8OBbITieD0v3VMo91EZU5k3sraJcOoF+TIrnw/kO4b3qVDwhlc2UoZ9xWfbjm0PxQglkQPT3mUbiwbXbtn3N+gCVK0tPsibPo+zaqfHD6m1Bq/UGW+Q+YNRhTzumjTWWUTgI7hum4m4cjSn2B1s8QnKuOL7bXGmtWSaT161N29iplSLz98bt3sdstWCUt7o8EFXET+Pb8bs0Kc7ld/MA34rtyyTh8waxQ2lyr5s8kbGQlyDtLO2WnKH61XE1I5F4105bPt0hqYIPj6/Zg0eAZrN6roGgqkux2v8/PPzN+fGzuVPn8+1W70WJkzmxsIrFdgkrxQjko28oPRaF9r43I3ubea1T8ctMSR5ZXbNyL9JltuKqNYRXin10YS60ar6ytrC44DB81X888Xs4mo8m83eslDAn74Y3otCwZteZSa2N2oUQnOvJGFMXit5COxQWek1kQMwvwrMa0LHLRkrN8VZhFde6KDPOKDlSFNEbjZiN2XbA/P+Da3IaZh1M8NIo52Lq1++XH25uriMnbQUhd2s8Ijga6/i9hPpqCaEfuVVsfMkRmNMep1PY2W0aIjhSU3xIHLTq60uOBBeZWcG+cejexe0Ii+8btANu64bXf56ff3b9e9/lFLXDbth0PWOCuVueuUhk+7lTpW+6X3rlX1iiZ/Pg7D0OszXwaYItI3c81KvvMlmVWAWvbLutId0p2zep5sfYOK5Yey4YRSG8Z9/XV9f/x0ZbuC4XaPUdb3CRGuJ16G4ie2lXismTdvNmpAhvc5v5ZYYx9/3yvMVZbmq7SsFrxWzx+8Be5D3ZO3Y1oPMdeIgjFO2MdLUibXAjbSuwzZBpi/+32CJV5atG4rIUL2vvfpynPKUw4VJr/wa8FTOhzqve4vXugjlI9H03OumDMaWuq5PyF57RuwwiaGhhaFGRG6gOVmchk7s6ItIsMQrH3MDca9/m7e47cSyhnKOKb2KkFzbysPuLV4pwA5pheYXvPKrMm5adVU2un7s6d1Ui0LH6RqOSysvI+S7RpppWup6i5XsEq9+npVpaM69bon4Ol8qUWJq5ArypZeYqo0LXmkI9m94VUbF3LeIr/OFXnLfglYkzJyMzQbYCDWYRjaV1TI3KkUO/yQMnMIM1m8kNdKmjGpJImaOlQ7L0/WKWUtqFaWcJDXKzsOkJoLigGeWXZGxO0lNPg0zO+zjrVb55sdWLSH/A3EW3iTNz+w6M92vU0xusxMreSN8KSs7tIa8ClkcCB3mkz9xdQ3NcVN6WqgFGTu4MYNdyven5v7yr//l4/ts5H/mtBsEjhaJlazjsmzVlU8HIi0NguKEAPwHTkLNCNJS7pXJ7MqnA1qQasbt4xUs5Vi8OEDPW7Wsm6aBK57D8Dc0Us/D+xkr8uyV7mX0pksU6J7O3yVK2YERpZmnT6B1dZ59CunNLN3bm+4fTacT+ZrWySGs/iBH23uHh9u5xiN2sHcMqQAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA4A75BzD5f9bJecWCAAAAAElFTkSuQmCC", label: "**Classification**:\nTransformers", pos: "t", w: 250, h: 100, constraint: "on" }
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
