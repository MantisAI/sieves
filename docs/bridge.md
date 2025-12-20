# Bridge

A `Bridge` acts as an intermediary between a `Task` and a `ModelWrapper`. It is responsible for translating task-specific requirements (like prompts and schemas) into a format the model wrapper understands, and conversely, integrating model outputs back into the document's results.

---

::: sieves.tasks.predictive.bridges.Bridge
::: sieves.tasks.predictive.bridges.GliNERBridge
