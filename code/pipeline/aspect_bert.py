from transformers import TrainingArguments, Trainer, AutoTokenizer, AutoModelForSequenceClassification, EarlyStoppingCallback
from datasets import load_from_disk
from huggingface_hub import login
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, classification_report
from torch.utils.data import WeightedRandomSampler
from collections import Counter
import os
import logging
import torch
from torch.nn.functional import sigmoid
import numpy as np
import json
import pandas as pd
import wandb
import argparse

logging.basicConfig(level = logging.INFO)


def classify_aspect(threshold:float, lr:float, batch:int, size:str, train_domain:str, test_domain:str, 
                    seed:int, hd:float, wd:float, wr:float):    
    # wandb
    os.environ["WANDB__SERVICE_WAIT"]= "300"
    if "comment" in train_domain:
        os.environ["WANDB_PROJECT"] = "nb3_aspect_comment" 
    else:
        os.environ["WANDB_PROJECT"] = "nb3_aspect_sent" 

    os.environ["WANDB_API_KEY"] = ""
    os.environ["WANDB_WATCH"] = "true"
    os.environ["WANDB_NAME"] = f"{train_domain}_{test_domain}_{size}_{seed}" # run name
    run = wandb.init()


    if "true" in explore.lower():
        explore = True
    else:
        explore=False

    # finding aspects automatically
    aspects = []
    true_direc = f"../data/aspect_ds/{test_domain}" # tidligere _json til slutt
            
    with open(f"../data/{true_direc}/full_ds.json", "r", encoding="utf-8") as readfile:
        data = json.load(readfile)
        for sample in data:
            for aspect in sample["labels"]:
                aspects.append(aspect)
    aspect_list = sorted(list(set(aspects)))   

    id2label = {i: label for i, label in enumerate(aspect_list)}
    label2id = {label: i for i, label in enumerate(aspect_list)}
    
    logging.info(f"Seed: {seed}")
    logging.info(f"Train dataset: {train_domain}")
    logging.info(f"Dev/test dataset: {test_domain}")
    logging.info(f"Aspects used: {aspect_list}")

    # Load model and tokenizer
    model = AutoModelForSequenceClassification.from_pretrained(f"ltg/norbert3-{size}", problem_type="multi_label_classification",
                                                                num_labels=len(aspect_list), trust_remote_code=True,
                                                                label2id=label2id, id2label=id2label)
    tokenizer = AutoTokenizer.from_pretrained(f"ltg/norbert3-{size}", use_fast=True)
    predicting = False

    def tokenize_function(batch):
        """
        Tokenizes input text and encodes multi-label aspect categories.
        """
        tokenized_batch = tokenizer(batch["text"], padding="max_length", truncation=True, max_length=512)
        labels = []
        for i in range(len(batch["text"])):
            # One-hot encoding for each aspect
            aspect_vector = [1 if aspect in batch["labels"][i] else 0 for aspect in aspect_list]
            labels.append(aspect_vector)
        
        # Add the labels to the tokenized batch 
        tokenized_batch["labels"] = torch.tensor(labels, dtype=torch.float32)
        
        return tokenized_batch


    def compute_metrics(p):
        preds = p.predictions
        labels = p.label_ids

        # Apply sigmoid to logits to get probabilities
        probs = sigmoid(torch.from_numpy(preds))  # Convert logits to probabilities
        preds = (probs > threshold).numpy().astype(int) # threshold for binary classification
        accuracy = accuracy_score(labels, preds)
        precision, recall, f1, _ = precision_recall_fscore_support(labels, preds, average='macro')
        # writing predictions
        if predicting:
            true_path = f"{true_direc}/test.json"
            key = "test"
        else:
            true_path = f"{true_direc}/dev.json"
            key = "dev"
        
        if not os.path.isdir(f"{training_args.output_dir}/preds/"):
            os.mkdir(f"{training_args.output_dir}/preds/")

        with open(f"{training_args.output_dir}/preds/preds_{key}.csv", "w", encoding="utf-8") as pred_file,\
            open(true_path, "r", encoding="utf-8") as true_file:
            
            loaded_true = json.load(true_file)
            pred_df = pd.DataFrame({"text":[ordbok["text"] for ordbok in loaded_true], "true":["dummy" for i in loaded_true], "pred":["dummy" for i in loaded_true]})

            # converting one-hot-vector to aspects
            rows = []
            for pred, true_label, sample in zip(preds, labels, loaded_true):
                converted_pred = [aspect_list[i] for i, val in enumerate(pred) if val == 1]
                converted_true = [aspect_list[i] for i, val in enumerate(true_label) if val == 1]
                rows.append({
                    "id":str(sample["id"]),
                    "text": str(sample["text"]),
                    "true": converted_true,
                    "pred": converted_pred
                })
            pred_df = pd.DataFrame(rows)
            pred_df.to_csv(pred_file, sep="\t") # writing predictions

        # calculating metrics
        report = classification_report(y_true=labels, y_pred=preds, target_names=aspect_list, output_dict=True)

        custom_report = {"f1-score":report["macro avg"]["f1-score"], "precision":report["macro avg"]["precision"], 
                         "recall":report["macro avg"]["recall"], "accuracy":accuracy}
        label_report = {key:val for key,val in report.items() if key not in custom_report and key not in exclude}

        custom_report.update(label_report)
        print(custom_report)
      
        return custom_report     


    # datasets
    train_dir = f"../data/hf_arrow_aspect/{train_domain}"
    test_dir = f"../data/hf_arrow_aspect/{test_domain}"
    train = load_from_disk(f"{train_dir}/train")
    dev = load_from_disk(f"{test_dir}/dev")
    test = load_from_disk(f"{test_dir}/test")

    # tokenize data
    train_dataset = train.map(tokenize_function, batched=True)
    dev_dataset = dev.map(tokenize_function, batched=True)
    test_dataset = test.map(tokenize_function, batched=True)

    # training arguments
    training_args = TrainingArguments(
    output_dir=f"./nb3_aspect_results/{train_domain}_{test_domain}_{size}/{seed}",
    eval_strategy="epoch",
    save_strategy="no",
    load_best_model_at_end=False,
    seed=seed,
    warmup_ratio=wr,
    per_device_train_batch_size=int(batch),
    per_device_eval_batch_size=int(batch),
    num_train_epochs=10,
    learning_rate=lr,
    weight_decay=wd,
    disable_tqdm=False,
    report_to="wandb"
)

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=dev_dataset,
        compute_metrics=compute_metrics,
    )

    training_args.output_dir = f"./nb3_aspect_results/{train_domain}_{test_domain}_{size}/{seed}"

    trainer = Trainer( # might add custom compute_metrics
        model=model,
        args=training_args,
        train_dataset=train_dataset, # tokenized dataset
        eval_dataset=dev_dataset,
        compute_metrics=compute_metrics
    )
    # Train
    logging.info("TRAINING MODEL...")
    trainer.train()

    # Evaluate
    logging.info("EVALUATING...")
    eval_results = trainer.evaluate()
    eval_results["seed"] = seed
    eval_results["model_size"] = size
    trainer.save_metrics("dev", eval_results)
   
    # test/predict
    logging.info("PREDICTING...")
    predicting = True
    predict_output = trainer.predict(test_dataset)
    preds = predict_output.predictions
    labels = predict_output.label_ids
    metrics = predict_output.metrics
    metrics["seed"] = seed
    metrics["model_size"] = size
    trainer.save_metrics("test", metrics)

    # log to wandb
    print("test-metrics:")
    print(metrics)
    run.log({"test_highlights/f1":metrics["test_f1-score"]})
    run.log({"test_highlights/recall":metrics["test_recall"]})
    run.log({"test_highlights/precision":metrics["test_precision"]})

    # expand results files
    for split in ["dev_results.json", "test_results.json"]:
        with open(f"{training_args.output_dir}/{split}", "r") as readfile:
            results = json.load(readfile)
            info = {"args_info":
                    {"ds":f"{train_domain}_{test_domain}", 
                    "epochs":training_args.num_train_epochs, 
                    "seed":seed,
                    "epochs":training_args.num_train_epochs,
                    "weight_decay":training_args.weight_decay,
                    "warmup ratio":training_args.warmup_ratio,
                    "hidden_dropout": hd,
                    "batch_size":training_args.per_device_train_batch_size, 
                    "threshold":threshold, 
                    "model":model.config._name_or_path,
                    "lr":lr}}
            results.update(info)
        with open(f"{training_args.output_dir}/{split}", "w") as writefile:
            json.dump(results, writefile, ensure_ascii=False, indent=4)
            print(f"\nExpanded {training_args.output_dir}/{split} with argument info")

    os.makedirs("models/aspect/", exist_ok=True)
    trainer.save_model(output_dir=f"models/aspect/{train_domain}_{seed}")
    tokenizer.save_pretrained(f"models/aspect/{train_domain}_{seed}")


def main():
    parser = argparse.ArgumentParser(description="arguments for aspect modeling")
    parser.add_argument("--threshold", type=float, required=True, help="the threshold in classification, e.g. 0.5")
    parser.add_argument("--lr", type=float, required=True, help="the learning rate, e.g 1e-5")
    parser.add_argument("--batch", type=int, required=True, help="train and eval batch size")
    parser.add_argument("--size", type=str, required=True, help="the size of the model, small, base or large")
    parser.add_argument("--train_domain", type=str, required=True, help="the domain to train on")
    parser.add_argument("--test_domain", type=str, required=True, help="the domain to test on")
    parser.add_argument("--seed", type=int, required=True, help="the seed")
    parser.add_argument("--hd", type=float, required=True, help="hidden dropout, e.g 0.2 (default 0.1)")
    parser.add_argument("--wd", type=float, required=True, help="weight decay, e.g 0.02")
    parser.add_argument("--wr", type=float, required=True, help="warmup ration, e.g 0.1")

    args = parser.parse_args()

    classify_aspect(threshold=args.threshold, lr=args.lr, batch=args.batch, size=args.size, hd=args.hd,
                    wd=args.wd, wr=args.wr, train_domain=args.train_domain, test_domain=args.test_domain, 
                    seed=args.seed)

if __name__ == "__main__":
    main()