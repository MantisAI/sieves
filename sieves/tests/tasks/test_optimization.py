"""Tests for task optimization."""
import dspy
import pydantic
import pytest

from sieves import ModelSettings
from sieves.model_wrappers import ModelType
from sieves.tasks.optimization import Optimizer
from sieves.tasks.predictive import (
    classification,
    sentiment_analysis,
    ner,
    pii_masking,
    information_extraction,
    summarization,
    translation,
    question_answering,
)

from sieves.tests.conftest import make_model


@pytest.fixture(scope="module")
def optimizer(request) -> Optimizer:
    """Return model and optimizer to use for optimization.

    :return: model and optimizer to use for optimization.
    """
    model = make_model(ModelType.dspy)
    optimizer = Optimizer(
        model,
        val_frac=.25,
        shuffle=True,
        dspy_init_kwargs=dict(
            auto=None, num_candidates=1, max_bootstrapped_demos=1, max_labeled_demos=1, max_errors=10, num_threads=1
        ),
        dspy_compile_kwargs=dict(num_trials=1, minibatch=False),
    )

    return optimizer


def test_optimization_classification(optimizer) -> None:
    """Tests optimization for classification tasks."""
    examples_single_label = [
        classification.FewshotExampleSingleLabel(text='Apple', label='fruit', confidence=1.),
        classification.FewshotExampleSingleLabel(text='Broccoli', label='vegetable', confidence=1.),
        classification.FewshotExampleSingleLabel(text='Melon', label='fruit', confidence=1.),
        classification.FewshotExampleSingleLabel(text='Carrot', label='vegetable', confidence=1.),
        classification.FewshotExampleSingleLabel(text='Tomato', label='fruit', confidence=1.),
        classification.FewshotExampleSingleLabel(text='Pepper', label='vegetable',
                                                 confidence=1.),
        classification.FewshotExampleSingleLabel(text='Kiwi', label='fruit', confidence=1.),
        classification.FewshotExampleSingleLabel(text='Onion', label='vegetable',
                                                 confidence=1.),
    ]
    examples_multi_label = [
        classification.FewshotExampleMultiLabel(
            text='Ghostbusters',
            confidence_per_label={'comedy': 0.9, 'scifi': 0.8}
        ),
        classification.FewshotExampleMultiLabel(
            text='The Martian',
            confidence_per_label={'comedy': 0.4, 'scifi': 1.0}
        ),
        classification.FewshotExampleMultiLabel(
            text='Galaxy Quest',
            confidence_per_label={'comedy': 1.0, 'scifi': 0.9}
        ),
        classification.FewshotExampleMultiLabel(
            text='Back to the Future',
            confidence_per_label={'comedy': 0.8, 'scifi': 0.9}
        ),
        classification.FewshotExampleMultiLabel(
            text='Superbad',
            confidence_per_label={'comedy': 1.0, 'scifi': 0.0}
        ),
        classification.FewshotExampleMultiLabel(
            text='Blade Runner 2049',
            confidence_per_label={'comedy': 0.05, 'scifi': 1.0}
        ),
        classification.FewshotExampleMultiLabel(
            text='Guardians of the Galaxy',
            confidence_per_label={'comedy': 0.75, 'scifi': 0.9}
        ),
        classification.FewshotExampleMultiLabel(
            text='Interstellar',
            confidence_per_label={'comedy': 0.05, 'scifi': 1.0}
        ),
    ]

    task_single_label = classification.Classification(
        multi_label=False,
        labels=["fruit", "vegetable"],
        fewshot_examples=examples_single_label,
        model=optimizer.model,
        model_settings=ModelSettings(),
    )
    task_multi_label = classification.Classification(
        multi_label=True,
        labels=["comedy", "scifi"],
        fewshot_examples=examples_multi_label,
        model=optimizer.model,
        model_settings=ModelSettings(),
    )

    # Test evaluation.
    assert task_single_label._evaluate_optimization_example(
        truth=dspy.Example(text="", reasoning="", label="fruit", confidence=.7),
        pred=dspy.Prediction(text="", reasoning="", label="fruit", confidence=.1),
        trace=None,
        model=optimizer.model,
    ) == .4
    assert task_single_label._evaluate_optimization_example(
        truth=dspy.Example(text="", reasoning="", label="fruit", confidence=.7),
        pred=dspy.Prediction(text="", reasoning="", label="vegetable", confidence=.1),
        trace=None,
        model=optimizer.model,
    ) == 0
    assert task_multi_label._evaluate_optimization_example(
        truth=dspy.Example(text="", reasoning="", confidence_per_label={"comedy": .4, "scifi": .2}),
        pred=dspy.Prediction(text="", reasoning="", confidence_per_label={"comedy": .1, "scifi": .3}),
        trace=None,
        model=optimizer.model,
    ) == .8
    assert task_multi_label._evaluate_optimization_example(
        truth=dspy.Example(text="", reasoning="", confidence_per_label={"comedy": .4, "scifi": .2}),
        pred=dspy.Prediction(text="", reasoning="", confidence_per_label={"comedy": .4, "scifi": .2}),
        trace=None,
        model=optimizer.model,
    ) == 1

    # Smoke-test optimization.
    best_prompt, best_examples = task_single_label.optimize(optimizer, verbose=False)
    assert task_single_label._custom_prompt_instructions == best_prompt
    assert task_single_label._bridge._prompt_instructions == best_prompt
    assert isinstance(task_single_label._fewshot_examples, list)

    best_prompt, best_examples = task_multi_label.optimize(optimizer, verbose=False)
    assert task_multi_label._custom_prompt_instructions == best_prompt
    assert task_multi_label._bridge._prompt_instructions == best_prompt
    assert isinstance(task_multi_label._fewshot_examples, list)


def test_optimization_sentiment_analysis(optimizer) -> None:
    """Tests optimization for sentiment analysis task."""
    examples = [
        sentiment_analysis.FewshotExample(
            text='Great product, excellent quality and fast shipping!',
            sentiment_per_aspect={'overall': 0.95, 'quality': 0.9, 'delivery': 0.95}
        ),
        sentiment_analysis.FewshotExample(
            text='Terrible quality, arrived damaged and late.',
            sentiment_per_aspect={'overall': 0.1, 'quality': 0.05, 'delivery': 0.15}
        ),
        sentiment_analysis.FewshotExample(
            text='Decent product but shipping was slow.',
            sentiment_per_aspect={'overall': 0.5, 'quality': 0.6, 'delivery': 0.3}
        ),
        sentiment_analysis.FewshotExample(
            text='Amazing quality! Worth every penny.',
            sentiment_per_aspect={'overall': 0.95, 'quality': 1.0, 'delivery': 0.5}
        ),
        sentiment_analysis.FewshotExample(
            text='Not great, not terrible. Just okay.',
            sentiment_per_aspect={'overall': 0.5, 'quality': 0.5, 'delivery': 0.5}
        ),
        sentiment_analysis.FewshotExample(
            text='Quick delivery but product quality is poor.',
            sentiment_per_aspect={'overall': 0.4, 'quality': 0.2, 'delivery': 0.8}
        ),
    ]

    task = sentiment_analysis.SentimentAnalysis(
        aspects=('quality', 'delivery'),
        model=optimizer.model,
        fewshot_examples=examples,
        model_settings=ModelSettings(),
    )

    # Test evaluation: perfect match
    assert task._evaluate_optimization_example(
        truth=dspy.Example(text='', reasoning='', sentiment_per_aspect={'overall': 0.8, 'quality': 0.7, 'delivery': 0.9}),
        pred=dspy.Prediction(text='', reasoning='', sentiment_per_aspect={'overall': 0.8, 'quality': 0.7, 'delivery': 0.9}),
        trace=None,
        model=optimizer.model
    ) == 1.0

    # Test evaluation: partial match (MAE-based accuracy)
    # |0.8-0.6| = 0.2, |0.7-0.5| = 0.2, |0.9-0.8| = 0.1
    # Accuracy per aspect: (1-0.2) + (1-0.2) + (1-0.1) = 0.8 + 0.8 + 0.9 = 2.5
    # Average: 2.5 / 3 ≈ 0.833
    score = task._evaluate_optimization_example(
        truth=dspy.Example(text='', reasoning='', sentiment_per_aspect={'overall': 0.8, 'quality': 0.7, 'delivery': 0.9}),
        pred=dspy.Prediction(text='', reasoning='', sentiment_per_aspect={'overall': 0.6, 'quality': 0.5, 'delivery': 0.8}),
        trace=None,
        model=optimizer.model,
    )
    assert abs(score - 0.833) < 0.01

    # Smoke-test optimization
    best_prompt, best_examples = task.optimize(optimizer, verbose=False)
    assert task._custom_prompt_instructions == best_prompt
    assert task._bridge._prompt_instructions == best_prompt
    assert isinstance(task._fewshot_examples, list)


def test_optimization_ner(optimizer) -> None:
    """Tests optimization for NER task."""
    examples = [
        ner.FewshotExample(
            text='John Smith visited Paris last week.',
            entities=[
                ner.EntityWithContext(text='John Smith', context='visited Paris', entity_type='PERSON'),
                ner.EntityWithContext(text='Paris', context='John Smith visited', entity_type='LOCATION'),
            ]
        ),
        ner.FewshotExample(
            text='Apple CEO Tim Cook announced new products.',
            entities=[
                ner.EntityWithContext(text='Tim Cook', context='Apple CEO', entity_type='PERSON'),
            ]
        ),
        ner.FewshotExample(
            text='The meeting in London was attended by Sarah Johnson.',
            entities=[
                ner.EntityWithContext(text='London', context='meeting in', entity_type='LOCATION'),
                ner.EntityWithContext(text='Sarah Johnson', context='attended by', entity_type='PERSON'),
            ]
        ),
        ner.FewshotExample(
            text='Berlin is the capital of Germany.',
            entities=[
                ner.EntityWithContext(text='Berlin', context='capital of Germany', entity_type='LOCATION'),
            ]
        ),
        ner.FewshotExample(
            text='Maria Rodriguez traveled to Tokyo.',
            entities=[
                ner.EntityWithContext(text='Maria Rodriguez', context='traveled to Tokyo', entity_type='PERSON'),
                ner.EntityWithContext(text='Tokyo', context='Maria Rodriguez traveled', entity_type='LOCATION'),
            ]
        ),
        ner.FewshotExample(
            text='The conference in New York was successful.',
            entities=[
                ner.EntityWithContext(text='New York', context='conference in', entity_type='LOCATION'),
            ]
        ),
    ]

    task = ner.NER(
        entities=['PERSON', 'LOCATION'],
        model=optimizer.model,
        fewshot_examples=examples,
        model_settings=ModelSettings(),
    )

    # Test evaluation: perfect match (F1 = 1.0)
    assert task._evaluate_optimization_example(
        truth=dspy.Example(text='', entities=[
            {'text': 'Alice', 'entity_type': 'PERSON'},
            {'text': 'Boston', 'entity_type': 'LOCATION'},
        ]),
        pred=dspy.Prediction(text='', entities=[
            {'text': 'Alice', 'entity_type': 'PERSON'},
            {'text': 'Boston', 'entity_type': 'LOCATION'},
        ]),
        trace=None,
        model=optimizer.model,
    ) == 1.0

    # Test evaluation: partial match (precision=0.5, recall=0.5, F1=0.5)
    # True: {('Alice', 'PERSON'), ('Boston', 'LOCATION')}
    # Pred: {('Alice', 'PERSON'), ('Chicago', 'LOCATION')}
    # Intersection: {('Alice', 'PERSON')} = 1
    # Precision: 1/2 = 0.5, Recall: 1/2 = 0.5, F1: 0.5
    assert task._evaluate_optimization_example(
        truth=dspy.Example(text='', entities=[
            {'text': 'Alice', 'entity_type': 'PERSON'},
            {'text': 'Boston', 'entity_type': 'LOCATION'},
        ]),
        pred=dspy.Prediction(text='', entities=[
            {'text': 'Alice', 'entity_type': 'PERSON'},
            {'text': 'Chicago', 'entity_type': 'LOCATION'},
        ]),
        trace=None,
        model=optimizer.model
    ) == 0.5

    # Test evaluation: no entities in truth (empty case)
    assert task._evaluate_optimization_example(
        truth=dspy.Example(text='', entities=[]),
        pred=dspy.Prediction(text='', entities=[]),
        trace=None,
        model=optimizer.model
    ) == 1.0

    # Smoke-test optimization
    best_prompt, best_examples = task.optimize(optimizer, verbose=False)
    assert task._custom_prompt_instructions == best_prompt
    assert task._bridge._prompt_instructions == best_prompt
    assert isinstance(task._fewshot_examples, list)


def test_optimization_pii_masking(optimizer) -> None:
    """Tests optimization for PII masking task."""
    examples = [
        pii_masking.FewshotExample(
            text='Please contact John Doe at john.doe@email.com for more information.',
            masked_text='Please contact [MASKED] at [MASKED] for more information.',
            pii_entities=[
                pii_masking.PIIEntity(entity_type='NAME', text='John Doe'),
                pii_masking.PIIEntity(entity_type='EMAIL', text='john.doe@email.com'),
            ]
        ),
        pii_masking.FewshotExample(
            text='Send the report to alice.smith@company.com by Friday.',
            masked_text='Send the report to [MASKED] by Friday.',
            pii_entities=[
                pii_masking.PIIEntity(entity_type='EMAIL', text='alice.smith@company.com'),
            ]
        ),
        pii_masking.FewshotExample(
            text='Call Bob Johnson at the office tomorrow.',
            masked_text='Call [MASKED] at the office tomorrow.',
            pii_entities=[
                pii_masking.PIIEntity(entity_type='NAME', text='Bob Johnson'),
            ]
        ),
        pii_masking.FewshotExample(
            text='Sarah Miller will attend the meeting.',
            masked_text='[MASKED] will attend the meeting.',
            pii_entities=[
                pii_masking.PIIEntity(entity_type='NAME', text='Sarah Miller'),
            ]
        ),
        pii_masking.FewshotExample(
            text='Email the document to michael.brown@org.net and copy jane.white@org.net.',
            masked_text='Email the document to [MASKED] and copy [MASKED].',
            pii_entities=[
                pii_masking.PIIEntity(entity_type='EMAIL', text='michael.brown@org.net'),
                pii_masking.PIIEntity(entity_type='EMAIL', text='jane.white@org.net'),
            ]
        ),
        pii_masking.FewshotExample(
            text='The meeting is scheduled for 2pm.',
            masked_text='The meeting is scheduled for 2pm.',
            pii_entities=[]
        ),
    ]

    task = pii_masking.PIIMasking(
        pii_types=['NAME', 'EMAIL'],
        model=optimizer.model,
        fewshot_examples=examples,
        model_settings=ModelSettings(),
    )

    # Test evaluation: perfect match (F1 = 1.0)
    assert task._evaluate_optimization_example(
        truth=dspy.Example(text='', reasoning='', masked_text='', pii_entities=[
            {'entity_type': 'NAME', 'text': 'Alice'},
            {'entity_type': 'EMAIL', 'text': 'alice@test.com'},
        ]),
        pred=dspy.Prediction(text='', reasoning='', masked_text='', pii_entities=[
            {'entity_type': 'NAME', 'text': 'Alice'},
            {'entity_type': 'EMAIL', 'text': 'alice@test.com'},
        ]),
        trace=None,
model=optimizer.model
    ) == 1.0

    # Test evaluation: partial match (precision=0.5, recall=0.5, F1=0.5)
    # True: {('NAME', 'Alice'), ('EMAIL', 'alice@test.com')}
    # Pred: {('NAME', 'Alice'), ('EMAIL', 'bob@test.com')}
    # Intersection: {('NAME', 'Alice')} = 1
    # Precision: 1/2 = 0.5, Recall: 1/2 = 0.5, F1: 0.5
    assert task._evaluate_optimization_example(
        truth=dspy.Example(text='', reasoning='', masked_text='', pii_entities=[
            {'entity_type': 'NAME', 'text': 'Alice'},
            {'entity_type': 'EMAIL', 'text': 'alice@test.com'},
        ]),
        pred=dspy.Prediction(text='', reasoning='', masked_text='', pii_entities=[
            {'entity_type': 'NAME', 'text': 'Alice'},
            {'entity_type': 'EMAIL', 'text': 'bob@test.com'},
        ]),
        trace=None,
model=optimizer.model
    ) == 0.5

    # Test evaluation: no PII (empty case)
    assert task._evaluate_optimization_example(
        truth=dspy.Example(text='', reasoning='', masked_text='', pii_entities=[]),
        pred=dspy.Prediction(text='', reasoning='', masked_text='', pii_entities=[]),
        trace=None,
model=optimizer.model
    ) == 1.0

    # Smoke-test optimization
    best_prompt, best_examples = task.optimize(optimizer, verbose=False)
    assert task._custom_prompt_instructions == best_prompt
    assert task._bridge._prompt_instructions == best_prompt
    assert isinstance(task._fewshot_examples, list)


def test_optimization_information_extraction(optimizer) -> None:
    """Tests optimization for information extraction task."""
    # Define entity type for extraction
    class Person(pydantic.BaseModel, frozen=True):
        """Person entity."""
        name: str
        age: int
        occupation: str

    examples = [
        information_extraction.FewshotExampleMulti(
            text='Alice Johnson is a 28-year-old software engineer.',
            entities=[Person(name='Alice Johnson', age=28, occupation='software engineer')]
        ),
        information_extraction.FewshotExampleMulti(
            text='Bob Smith, age 35, works as a teacher.',
            entities=[Person(name='Bob Smith', age=35, occupation='teacher')]
        ),
        information_extraction.FewshotExampleMulti(
            text='The team includes Sarah Lee (42, doctor) and Mike Brown (30, lawyer).',
            entities=[
                Person(name='Sarah Lee', age=42, occupation='doctor'),
                Person(name='Mike Brown', age=30, occupation='lawyer'),
            ]
        ),
        information_extraction.FewshotExampleMulti(
            text='Emma Davis is 25 and works as a designer.',
            entities=[Person(name='Emma Davis', age=25, occupation='designer')]
        ),
        information_extraction.FewshotExampleMulti(
            text='Dr. John Williams, a 50-year-old researcher, published a new paper.',
            entities=[Person(name='Dr. John Williams', age=50, occupation='researcher')]
        ),
        information_extraction.FewshotExampleMulti(
            text='The company hired Lisa Chen (27) as an analyst.',
            entities=[Person(name='Lisa Chen', age=27, occupation='analyst')]
        ),
    ]

    task = information_extraction.InformationExtraction(
        entity_type=Person,
        model=optimizer.model,
        fewshot_examples=examples,
        model_settings=ModelSettings(),
    )

    # Test evaluation: perfect match (F1 = 1.0)
    assert task._evaluate_optimization_example(
        truth=dspy.Example(text='', reasoning='', entities=[
            {'name': 'Alice', 'age': 30, 'occupation': 'engineer'},
            {'name': 'Bob', 'age': 25, 'occupation': 'designer'},
        ]),
        pred=dspy.Prediction(text='', reasoning='', entities=[
            {'name': 'Alice', 'age': 30, 'occupation': 'engineer'},
            {'name': 'Bob', 'age': 25, 'occupation': 'designer'},
        ]),
        trace=None,
model=optimizer.model
    ) == 1.0

    # Test evaluation: partial match (precision=0.5, recall=0.5, F1=0.5)
    # True entities (as sorted tuples): {('age',25),('name','Alice'),('occupation','engineer'), ('age',30),
    # ('name','Bob'),('occupation','designer')}
    # Pred entities: one correct, one different
    # This should give F1 of 0.5
    assert task._evaluate_optimization_example(
        truth=dspy.Example(text='', reasoning='', entities=[
            {'name': 'Alice', 'age': 30, 'occupation': 'engineer'},
            {'name': 'Bob', 'age': 25, 'occupation': 'designer'},
        ]),
        pred=dspy.Prediction(text='', reasoning='', entities=[
            {'name': 'Alice', 'age': 30, 'occupation': 'engineer'},
            {'name': 'Charlie', 'age': 35, 'occupation': 'teacher'},
        ]),
        trace=None,
model=optimizer.model
    ) == 0.5

    # Test evaluation: no entities (empty case)
    assert task._evaluate_optimization_example(
        truth=dspy.Example(text='', reasoning='', entities=[]),
        pred=dspy.Prediction(text='', reasoning='', entities=[]),
        trace=None,
model=optimizer.model
    ) == 1.0

    # Smoke-test optimization
    best_prompt, best_examples = task.optimize(optimizer, verbose=False)
    assert task._custom_prompt_instructions == best_prompt
    assert task._bridge._prompt_instructions == best_prompt
    assert isinstance(task._fewshot_examples, list)


def test_optimization_summarization(optimizer) -> None:
    """Tests optimization for summarization task using LLM-based evaluator."""
    examples = [
        summarization.FewshotExample(
            text='The European Space Agency launched a new satellite yesterday to monitor climate change. '
                 'The satellite will collect data on temperature, sea levels, and atmospheric conditions over the '
                 'next decade.',
            n_words=30,
            summary='ESA launched a climate monitoring satellite to track temperature, sea levels, and atmosphere for '
                    '10 years.'
        ),
        summarization.FewshotExample(
            text='Local farmers are adopting sustainable practices to reduce water usage and improve soil health. '
                 'New irrigation systems have reduced water consumption by 40% while maintaining crop yields.',
            n_words=30,
            summary='Farmers use sustainable methods and new irrigation, cutting water use 40% with same yields.'
        ),
        summarization.FewshotExample(
            text='The company announced record profits of $5 billion this quarter, driven by strong sales in Asia '
                 'and Europe. CEO Jane Smith credited the success to innovative product launches and effective '
                 'marketing.',
            n_words=30,
            summary='Company posts $5B profit from Asian/European sales, CEO cites innovation and marketing.'
        ),
        summarization.FewshotExample(
            text='Researchers discovered a new species of deep-sea fish near the Mariana Trench. '
                 'The bioluminescent creature can survive at depths of 8,000 meters and uses light to attract prey.',
            n_words=30,
            summary='Scientists find new bioluminescent deep-sea fish at 8,000m depth that uses light for hunting.'
        ),
        summarization.FewshotExample(
            text='The city council approved a $200 million infrastructure plan to repair roads, bridges, and public transit. '
                 'Construction will begin next spring and is expected to create 5,000 jobs over three years.',
            n_words=30,
            summary='Council approves $200M infrastructure plan for roads, bridges, transit; 5,000 jobs starting spring.'
        ),
        summarization.FewshotExample(
            text='Scientists developed a new vaccine that shows 95% effectiveness against multiple virus strains. '
                 'Clinical trials involved 50,000 participants across 20 countries and demonstrated strong safety profiles.',
            n_words=30,
            summary='New vaccine shows 95% effectiveness across virus strains in 50,000-participant global trials.'
        ),
    ]

    task = summarization.Summarization(
        n_words=30,
        model=optimizer.model,
        fewshot_examples=examples,
        model_settings=ModelSettings(),
    )

    # Test LLM-based evaluation (no hardcoded scores since it uses LLM)
    # Just verify it runs and returns a score between 0 and 1
    score = task._evaluate_optimization_example(
        truth=dspy.Example(text='', n_words=30, summary='Short summary about climate.'),
        pred=dspy.Prediction(text='', n_words=30, summary='Brief summary on climate change.'),
        trace=None,
model=optimizer.model
    )
    assert 0.5 <= score <= 1.0

    # Smoke-test optimization
    best_prompt, best_examples = task.optimize(optimizer, verbose=False)
    assert task._custom_prompt_instructions == best_prompt
    assert task._bridge._prompt_instructions == best_prompt
    assert isinstance(task._fewshot_examples, list)


def test_optimization_translation(optimizer) -> None:
    """Tests optimization for translation task using LLM-based evaluator."""
    examples = [
        translation.FewshotExample(
            text='Hello, how are you today?',
            to='Spanish',
            translation='Hola, ¿cómo estás hoy?'
        ),
        translation.FewshotExample(
            text='The weather is beautiful this morning.',
            to='Spanish',
            translation='El clima está hermoso esta mañana.'
        ),
        translation.FewshotExample(
            text='I would like to order a coffee, please.',
            to='Spanish',
            translation='Me gustaría pedir un café, por favor.'
        ),
        translation.FewshotExample(
            text='Where is the nearest train station?',
            to='Spanish',
            translation='¿Dónde está la estación de tren más cercana?'
        ),
        translation.FewshotExample(
            text='Thank you very much for your help.',
            to='Spanish',
            translation='Muchas gracias por tu ayuda.'
        ),
        translation.FewshotExample(
            text='I am learning Spanish and enjoying it.',
            to='Spanish',
            translation='Estoy aprendiendo español y lo disfruto.'
        ),
    ]

    task = translation.Translation(
        to='Spanish',
        model=optimizer.model,
        fewshot_examples=examples,
        model_settings=ModelSettings(),
    )

    # Test LLM-based evaluation
    score = task._evaluate_optimization_example(
        truth=dspy.Example(text='Good morning.', to='Spanish', translation='Buenos días.'),
        pred=dspy.Prediction(text='Good morning.', to='Spanish', translation='Buen día.'),
        trace=None,
model=optimizer.model
    )
    assert 0.7 <= score <= 1.0

    # Smoke-test optimization
    best_prompt, best_examples = task.optimize(optimizer, verbose=False)
    assert task._custom_prompt_instructions == best_prompt
    assert task._bridge._prompt_instructions == best_prompt
    assert isinstance(task._fewshot_examples, list)


def test_optimization_question_answering(optimizer) -> None:
    """Tests optimization for question answering task using LLM-based evaluator."""
    questions = ['What is the main topic?', 'Who are the key people mentioned?']

    examples = [
        question_answering.FewshotExample(
            text='Albert Einstein developed the theory of relativity in 1915. His work revolutionized modern physics.',
            questions=questions,
            answers=['The theory of relativity', 'Albert Einstein']
        ),
        question_answering.FewshotExample(
            text='Marie Curie won two Nobel Prizes for her research on radioactivity. She was the first woman to win a Nobel Prize.',
            questions=questions,
            answers=['Research on radioactivity and Nobel Prizes', 'Marie Curie']
        ),
        question_answering.FewshotExample(
            text='Shakespeare wrote Romeo and Juliet in 1597. The play is one of the most famous love stories in '
                 'literature.',
            questions=questions,
            answers=['Romeo and Juliet play', 'Shakespeare']
        ),
        question_answering.FewshotExample(
            text='The Amazon rainforest is home to millions of species. Deforestation threatens this biodiversity.',
            questions=questions,
            answers=['Amazon rainforest biodiversity and deforestation', 'No specific people mentioned']
        ),
        question_answering.FewshotExample(
            text='Neil Armstrong became the first person to walk on the moon in 1969. This historic event was part of '
                 'the Apollo 11 mission.',
            questions=questions,
            answers=['First moon landing', 'Neil Armstrong']
        ),
        question_answering.FewshotExample(
            text='Leonardo da Vinci painted the Mona Lisa during the Renaissance. He was also an inventor and scientist.',
            questions=questions,
            answers=['The Mona Lisa painting', 'Leonardo da Vinci']
        ),
    ]

    task = question_answering.QuestionAnswering(
        questions=questions,
        model=optimizer.model,
        fewshot_examples=examples,
        model_settings=ModelSettings(),
    )

    # Test LLM-based evaluation
    score = task._evaluate_optimization_example(
        truth=dspy.Example(text='', reasoning='', questions=questions, answers=['Climate change', 'Scientists']),
        pred=dspy.Prediction(text='', reasoning='', questions=questions, answers=['Global warming', 'Researchers']),
        trace=None,
        model=optimizer.model
    )
    assert 0.5 <= score <= 1.0

    # Smoke-test optimization
    best_prompt, best_examples = task.optimize(optimizer, verbose=False)
    assert task._custom_prompt_instructions == best_prompt
    assert task._bridge._prompt_instructions == best_prompt
    assert isinstance(task._fewshot_examples, list)
