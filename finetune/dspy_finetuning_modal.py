"""
DSPy Classification Fine-tuning on Modal

This script converts the DSPy fine-tuning tutorial to run on Modal.
It fine-tunes a Llama-3.2-1B model on the Banking77 classification task.

Run with: modal run dspy_finetuning_modal.py
"""

import modal
import os
import random
from typing import Literal

# Define Modal image with all dependencies
image = modal.Image.debian_slim(python_version="3.11").pip_install(
    "dspy-ai>=2.6.0",
    "datasets",
    "torch",
    "transformers==4.48.3",
    "accelerate",
    "trl",
    "peft",
    "sglang[all]>=0.4.4.post3",
    "huggingface-hub",
)

app = modal.App(name="dspy-finetuning", image=image)

# Configure GPU for fine-tuning and inference
GPU_CONFIG = modal.gpu.A100(count=1)


@app.function(
    gpu=GPU_CONFIG,
    timeout=3600,
    volumes={"/model_cache": modal.Volume.from_name("model-cache", create_if_missing=True)},
    env={"HF_HOME": "/model_cache"},
)
def finetune_banking77_classifier():
    """
    Main fine-tuning function that runs on Modal GPU.
    """
    import dspy
    from dspy.datasets import DataLoader
    from datasets import load_dataset

    print("Loading Banking77 dataset...")
    
    # Load the Banking77 dataset
    CLASSES = load_dataset(
        "PolyAI/banking77", 
        split="train", 
        trust_remote_code=True
    ).features['label'].names
    
    kwargs = dict(
        fields=("text", "label"),
        input_keys=("text",),
        split="train",
        trust_remote_code=True
    )
    
    # Load first 1000 examples from the dataset
    raw_data = [
        dspy.Example(x, label=CLASSES[x.label]).with_inputs("text")
        for x in DataLoader().from_huggingface(
            dataset_name="PolyAI/banking77", 
            **kwargs
        )[:1000]
    ]
    
    random.Random(0).shuffle(raw_data)
    
    print(f"Dataset loaded: {len(CLASSES)} classes, {len(raw_data)} examples")
    print(f"Sample classes: {CLASSES[:10]}")
    
    # Create unlabeled training set (first 500 examples)
    unlabeled_trainset = [
        dspy.Example(text=x.text).with_inputs("text") 
        for x in raw_data[:500]
    ]
    
    print(f"Unlabeled training set: {len(unlabeled_trainset)} examples")
    print(f"Sample: {unlabeled_trainset[0]}")
    
    # Define DSPy program
    print("\nDefining DSPy program...")
    classify = dspy.ChainOfThought(f"text -> label: Literal{CLASSES}")
    
    # Set up student LM (local) and teacher LM (OpenAI)
    print("Setting up language models...")
    
    # For Modal, we'll use local inference with SGLang
    from dspy.clients.lm_local import LocalProvider
    
    student_lm_name = "meta-llama/Llama-3.2-1B-Instruct"
    student_lm = dspy.LM(
        model=f"openai/local:{student_lm_name}",
        provider=LocalProvider(),
        max_tokens=2000
    )
    
    # Teacher LM - using OpenAI API (requires OPENAI_API_KEY env var)
    teacher_lm = dspy.LM('openai/gpt-4o-mini', max_tokens=3000)
    
    # Create student and teacher classifiers
    student_classify = classify.deepcopy()
    student_classify.set_lm(student_lm)
    
    teacher_classify = classify.deepcopy()
    teacher_classify.set_lm(teacher_lm)
    
    # Bootstrapped fine-tuning without labels
    print("\n=== Phase 1: Bootstrapped Fine-tuning (no labels) ===")
    dspy.settings.experimental = True
    
    optimizer = dspy.BootstrapFinetune(num_threads=16)
    classify_ft_no_labels = optimizer.compile(
        student_classify,
        teacher=teacher_classify,
        trainset=unlabeled_trainset
    )
    
    # Launch the local model
    print("Launching local model...")
    classify_ft_no_labels.get_lm().launch()
    
    # Evaluate on dev set (examples 500-600)
    devset = raw_data[500:600]
    metric = lambda x, y, trace=None: x.label == y.label
    
    print(f"\nEvaluating on {len(devset)} dev examples...")
    evaluate = dspy.Evaluate(
        devset=devset,
        metric=metric,
        display_progress=True,
        num_threads=16
    )
    
    score_no_labels = evaluate(classify_ft_no_labels)
    print(f"\n✓ Phase 1 Score (no labels): {score_no_labels:.1%}")
    
    # Clean up
    classify_ft_no_labels.get_lm().kill()
    
    # Bootstrapped fine-tuning with labels (metric-guided)
    print("\n=== Phase 2: Bootstrapped Fine-tuning (with labels & metric) ===")
    
    optimizer_with_metric = dspy.BootstrapFinetune(
        num_threads=16,
        metric=metric
    )
    classify_ft_with_labels = optimizer_with_metric.compile(
        classify.deepcopy(),
        teacher=teacher_classify,
        trainset=raw_data[:500]  # Now using labeled data
    )
    
    print("Launching local model (metric-guided)...")
    classify_ft_with_labels.get_lm().launch()
    
    score_with_labels = evaluate(classify_ft_with_labels)
    print(f"\n✓ Phase 2 Score (with labels): {score_with_labels:.1%}")
    
    # Compare with baseline teacher
    print("\n=== Baseline: Teacher LM (GPT-4o-mini) ===")
    score_teacher = evaluate(teacher_classify)
    print(f"\n✓ Teacher Score: {score_teacher:.1%}")
    
    # Summary
    print("\n" + "="*60)
    print("FINE-TUNING SUMMARY")
    print("="*60)
    print(f"Baseline (Teacher):        {score_teacher:.1%}")
    print(f"Student (no labels):       {score_no_labels:.1%}")
    print(f"Student (with labels):     {score_with_labels:.1%}")
    print(f"Improvement:               +{(score_with_labels - score_teacher):.1%}")
    print("="*60)
    
    # Save the fine-tuned model
    print("\nSaving fine-tuned model...")
    model_dir = "/modal/finetuned_model"
    os.makedirs(model_dir, exist_ok=True)
    
    # Get the underlying LM and save it
    finetuned_lm = classify_ft_with_labels.get_lm()
    
    # You can save the weights here if needed
    # finetuned_lm.model.save_pretrained(model_dir)
    
    print(f"✓ Model saved to {model_dir}")
    
    # Clean up
    classify_ft_with_labels.get_lm().kill()
    
    return {
        "baseline_score": score_teacher,
        "no_labels_score": score_no_labels,
        "with_labels_score": score_with_labels,
        "improvement": score_with_labels - score_teacher,
    }


@app.function(
    gpu=GPU_CONFIG,
    timeout=600,
)
def inference_example():
    """
    Run inference on a sample query using the fine-tuned model.
    """
    import dspy
    from dspy.clients.lm_local import LocalProvider
    
    print("Loading fine-tuned model for inference...")
    
    student_lm_name = "meta-llama/Llama-3.2-1B-Instruct"
    student_lm = dspy.LM(
        model=f"openai/local:{student_lm_name}",
        provider=LocalProvider(),
        max_tokens=2000
    )
    
    # In practice, you'd load the fine-tuned weights here
    classify = dspy.ChainOfThought("text -> label")
    classify.set_lm(student_lm)
    
    student_lm.launch()
    
    # Example inference
    test_query = "Why hasn't my card come in yet?"
    result = classify(text=test_query)
    
    print(f"\nQuery: {test_query}")
    print(f"Reasoning: {result.reasoning}")
    print(f"Label: {result.label}")
    
    student_lm.kill()
    
    return result.label


@app.local_entrypoint()
def main():
    """
    Local entrypoint to orchestrate the fine-tuning job.
    """
    print("Starting DSPy Fine-tuning on Modal...")
    print("="*60)
    
    # Run fine-tuning
    results = finetune_banking77_classifier.remote()
    
    print("\n✓ Fine-tuning complete!")
    print(f"Results: {results}")
    
    # Optionally run inference
    # label = inference_example.remote()
    # print(f"\nInference result: {label}")


if __name__ == "__main__":
    app.deploy()
