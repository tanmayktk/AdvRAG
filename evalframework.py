from ragas.testset.generator import TestsetGenerator
from ragas.testset.evolutions import simple, reasoning, multi_context
from langchain_community.document_loaders import PyPDFLoader, DirectoryLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings.huggingface import HuggingFaceEmbeddings
from datasets import Dataset
from ragas import evaluate
from ragas.metrics import (
        answer_relevancy,
        faithfulness,
        context_recall,
        context_precision,
        answer_correctness
    )
    


def eval_framework(filepath, customchain, myllm, testsize ):

    # Load your doc
    loader = DirectoryLoader(filepath, glob="*.pdf", loader_cls=PyPDFLoader)
    documents = loader.load()

    # Document Splitter
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size = 1000,
        chunk_overlap = 200
    )
    documents = text_splitter.split_documents(documents)
    #
    generator_llm = myllm
    critic_llm = myllm
    embeddings = HuggingFaceEmbeddings()

    #
    generator = TestsetGenerator.from_langchain(
        generator_llm,
        critic_llm,
        embeddings
    )
    #
    testset = generator.generate_with_langchain_docs(documents, test_size=testsize, distributions={simple: 0.5, reasoning: 0.25, multi_context: 0.25})

    test_df = testset.to_pandas()
    test_questions = test_df["question"].values.tolist()
    test_groundtruths = test_df["ground_truth"].values.tolist()

    # Generate responses using our Advanced RAG pipeline using the questions we’ve generated.
    adv_answers = []
    adv_contexts = []

    for question in test_questions:
        response = customchain.invoke({"query" : question})
        adv_answers.append(response["result"])
        adv_contexts.append([context.page_content for context in response['source_documents']])

    #wrap into huggingface dataset
    response_dataset_advanced_retrieval = Dataset.from_dict({
        "question" : test_questions,
        "answer" : adv_answers,
        "contexts" : adv_contexts,
        "ground_truth" : test_groundtruths
    })

    metrics = [
        faithfulness,
        answer_relevancy,
        context_recall,
        context_precision,
        answer_correctness,
    ]

    advanced_retrieval_results = evaluate(response_dataset_advanced_retrieval, metrics, llm=myllm, embeddings=embeddings, raise_exceptions=False)
    
    return advanced_retrieval_results
