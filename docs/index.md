# What is `sieves`?

`sieves` is a library for zero-shot document AI with structured generation. It facilitates the rapid prototyping of
document AI pipelines with validated output. No training required.

It bundles common NLP utilities, document parsing, and text chunking capabilities together with ready-to-use tasks like
classification and information extraction, all organized in an observable pipeline architecture. It's particularly
valuable for rapid prototyping scenarios where structured output is needed but training data is scarce.

`sieves` is built around three key components that you should be familiar with in order to use it:

1. **`Pipeline`**: The main orchestrator that runs your NLP tasks sequentially (define with `Pipeline([...])` or chain with `+`)
2. **`Task`**: Pre-built or custom NLP operations (classification, extraction, etc.)
3. **`Doc`**: The fundamental data structure for document processing

There are two more key components that you only need to know if you're a maintainer or want to create your own tasks for
`sieves`:

1. **`ModelWrapper`**: Backend implementations that power the tasks (outlines, dspy, langchain, etc.)
2. **`Bridge`**: Connectors between Tasks and Model wrappers

---

# Next Steps

- [Install `sieves`](setup.md)
- Dive into our guides, starting with the [Getting Started Guide](guides/getting_started.md)
- Understand different [model configurations](guides/models.md)
- Learn about custom task creation

Consult the API reference for each component you're working with if you have specific question. They contain detailed
information about parameters, configurations, and best practices.

---

# Essential Links

- [GitHub Repository](https://github.com/mantisai/sieves)
- [PyPI Package](https://pypi.org/project/sieves/)
- [Issue Tracker](https://github.com/mantisai/sieves/issues)

For any feedback, feature requests, contributions etc. use our [GitHub issue tracker](https://github.com/MantisAI/sieves/issues).

---

`sieves` is maintained by [Mantis](https://mantisnlp.com), an AI consultancy. We help our clients to solve business problems related to
natural human language and speech. If that's something you're interested in - [drop us a line](https://mantisnlp.com/contact/#cta)!
