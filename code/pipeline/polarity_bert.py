from transformers import TrainingArguments, Trainer, AutoTokenizer, AutoModelForSequenceClassification
from datasets import load_from_disk
from huggingface_hub import login
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, classification_report
import os
import logging
import torch
from torch.nn.functional import sigmoid
import numpy as np
import json
import pandas as pd
import wandb
import argparse

# other 
logging.basicConfig(level = logging.INFO)

def classify_aspect(threshold:float, lr:float, batch:int, size:str, train_domain:str, test_domain:str, gold:str, 
                    seed:int, hd:float, wd:float, wr:float):    
    # wandb
    os.environ["WANDB__SERVICE_WAIT"]= "300"
    os.environ["WANDB_PROJECT"] = "absa_polarity" 
    os.environ["WANDB_API_KEY"] = " "
    os.environ["WANDB_WATCH"] = "true"
    os.environ["WANDB_NAME"] = f"{gold}_{train_domain}_{test_domain}_{size}_{seed}" # run name
    run = wandb.init()
  
    # if gold is True, we predict polarity for gold aspects
    if gold.lower() == "false": # convert string input to bool
        gold = False
    else:
        gold = True

    if "true" in explore.lower():
        explore = True
    else:
        explore = False

    
    if "titles" in test_domain:
        aspect_descriptions = {
            "treat":"behandling", "staff":"ansatte", "avail":"tilgjengelighet", "org":"organisering", "env":"miljø og fasiliteter",
            "pip":"pasientinvolvering", "oits":"utfall av behandling/opphold", "gen":"generelt", "NO-PRED":"ukjent aspekt"} # there might be aspects that have not been predicted in the last step
    else:
        aspect_descriptions = {
            "med":"medisinering", "ppr":"relasjon mellom pasient og ansatt", "cp":"kompetanse", "tshp":"tid brukt med ansatte", 
            "isp":"informasjonsdeling med pasienter", "wtc":"ventetid i klinikk", "wtp":"ventetid til time",
            "td":"telefon og digital kommunikasjon", "gd":"geografisk distanse", "wol":"arbeidsmengde", 
            "sct":"stabilitet og kontinuitet i behandling", "incc":"intern organisering", "excos":"eksternt samarbeid med andre tjenester", 
            "slohs":"organisering av helsetjenester på systemnivå", "ppe":"fysisk og psykososialt miljø", "ftc":"tvungen behandling", "dur":"varighet av behandlinger og opphold", 
            "sr":"struktur og rutiner", "act":"aktiviteter", "qfm":"kvalitet på mat og måltidsrutiner", "iop":"interaksjon med andre pasienter", "lang":"språk", 
            "pip":"pasientinvolvering", "oits":"utfall av behandling/opphold", "gen":"generelt", "NO-PRED":"ukjent aspekt"}
    
    aspects = []
    if gold:
        domain = test_domain
        true_direc = f"../data/pipe_gold_ds/{domain}"
    else:
        domain = f"{train_domain}_{test_domain}"
        true_direc = f"../data/pipe_pred_ds/best_seeds/{domain}" # predicting polarity for the best seeds used for aspect classification
    
    label_list = ["pos", "neg", "mixed"]

    id2label = {i: label for i, label in enumerate(label_list)}
    label2id = {label: i for i, label in enumerate(label_list)}
    
    logging.info(f"Seed: {seed}")
    logging.info(f"Train dataset: {train_domain}")
    logging.info(f"Dev/test dataset: {test_domain}")
    logging.info(f"Labels used: {label_list}")

    # Load model and tokenizer
    model = AutoModelForSequenceClassification.from_pretrained(f"ltg/norbert3-{size}", problem_type="single_label_classification",
                                                                num_labels=3, trust_remote_code=True, id2label=id2label, label2id=label2id)
    tokenizer = AutoTokenizer.from_pretrained(f"ltg/norbert3-{size}", use_fast=True)
    predicting = False


    def tokenize_function(batch):
        """
        Tokenizes input text and encodes multi-label aspect categories.
        """
        aspects = []
        labels = []
        texts = []
        has_gold_label = [] # correctly predicted aspects

        for i, text in enumerate(batch["text"]):
            if batch["labels"][i] == "NO-GOLD": # if an aspect prediction has no gold we ignore it
                aspects.append(aspect_descriptions[batch["aspect"][i]])
                texts.append(text)
                labels.append(-100) # placeholder, wont be used in loss
                has_gold_label.append(False)
            else:
                aspects.append(aspect_descriptions[batch["aspect"][i]])
                labels.append(label2id[batch["labels"][i]])
                texts.append(text)
                has_gold_label.append(True)
        
        # Tokenize as pairs: (sentence/comment, aspect)
        tokenized_batch = tokenizer(
            texts,
            aspects,
            padding="max_length",
            truncation=True,
            max_length=512,
        )
        
        tokenized_batch["labels"] = torch.tensor(labels, dtype=torch.long)
        tokenized_batch["has_gold_label"] = torch.tensor(has_gold_label, dtype=torch.bool)
        return tokenized_batch


    def compute_metrics(p):
        preds = p.predictions.argmax(-1)
        labels = p.label_ids
        has_gold_label = p.label_ids != -100 # true where label is not the ignore index 
 
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
            
            data = json.load(true_file)
            loaded_true = [item for item in data if item["aspect"] != "no-asp"] # ignore neutral which is already classified (no-asp)
            pred_df = pd.DataFrame({"text":[ordbok["text"] for ordbok in loaded_true], "true":["dummy" for i in loaded_true], "pred":["dummy" for i in loaded_true]})

            # converting labels to string
            rows = []
            for pred, true_label, sample in zip(preds, labels, loaded_true): 
                converted_pred = id2label[pred]
                # Handle NO-GOLD labels
                if true_label == -100:
                    converted_true = "NO-GOLD"
                    has_gold = False
                else:
                    converted_true = label_map[true_label]
                    has_gold = True
                
                rows.append({
                    "id": sample["id"],
                    "aspect": sample["aspect"],
                    "text": str(sample["text"]),
                    "true": converted_true,
                    "pred": converted_pred,
                    "has_gold_label": has_gold
                })
        
            pred_df = pd.DataFrame(rows)
            pred_df.to_csv(pred_file, sep="\t")

        # Calculate metrics only on samples with gold labels
        gold_rows = [r for r in rows if r["has_gold_label"]]
        no_gold_rows = [r for r in rows if not r["has_gold_label"]]
        
        if gold_rows:
            gold_preds = np.array([label2id[r["pred"]] for r in gold_rows])
            gold_labels_arr = np.array([label2id[r["true"]] for r in gold_rows])
            
            # Calculate metrics on gold samples
            exclude = ["micro avg"]
            report = classification_report(
                y_true=gold_labels_arr, 
                y_pred=gold_preds, 
                target_names=["pos", "neg", "mixed"], 
                output_dict=True
            )
            custom_report = {
                "f1-score": report["macro avg"]["f1-score"], 
                "precision": report["macro avg"]["precision"], 
                "recall": report["macro avg"]["recall"], 
                "accuracy": accuracy_score(gold_labels_arr, gold_preds)
            }
            label_report = {key: val for key, val in report.items() if key not in custom_report and key not in exclude}
            custom_report.update(label_report)
        
        # Diagnostic info on NO_GOLD samples
        if no_gold_rows:
            no_gold_preds = [row["pred"] for row in no_gold_rows]
            print(f"\nNO_GOLD predictions (diagnostic):")
            print(f"Total NO_GOLD samples: {len(no_gold_rows)}")
            print(f"Prediction distribution: {pd.Series(no_gold_preds).value_counts().to_dict()}")

        print(custom_report)
        return custom_report     

    # datasets
    train_dir = f"../data/hf_arrow_gold_pipe/{train_domain}" 
    if gold:
        test_dir = f"../data/hf_arrow_gold_pipe/{domain}" # domain changes automatically based on 'gold' arg
    else:
        test_dir = f"../data/hf_arrow_pred_pipe/best_seeds/{domain}" 
    

    train = load_from_disk(f"{train_dir}/train").filter(lambda x: x["aspect"] != "no-asp") # neutral already classified in last step
    dev = load_from_disk(f"{test_dir}/dev").filter(lambda x: x["aspect"] != "no-asp")
    test = load_from_disk(f"{test_dir}/test").filter(lambda x: x["aspect"] != "no-asp")

    # tokenize data
    train_dataset = train.map(tokenize_function, batched=True)
    dev_dataset = dev.map(tokenize_function, batched=True)
    test_dataset = test.map(tokenize_function, batched=True)

    # training arguments
    if gold:
        out_dir = f"./nb3_polarity_gold_results/{train_domain}_{test_domain}_{size}/{seed}"
    else:
        out_dir = f"./nb3_polarity_pred_results/{train_domain}_{test_domain}_{size}/{seed}"
    
    training_args = TrainingArguments(
        output_dir=out_dir,
        eval_strategy="epoch",
        save_strategy="no",
        load_best_model_at_end=False,
        seed=seed,
        warmup_ratio=wr,
        per_device_train_batch_size=int(batch),
        per_device_eval_batch_size=int(batch),
        num_train_epochs=10,
        learning_rate = lr,
        weight_decay=wd,
        disable_tqdm = False,
        report_to="wandb"
    )

    trainer = Trainer( 
        model=model,
        args=training_args,
        train_dataset=train_dataset, 
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

    # expand results file
    with open(f"{training_args.output_dir}/test_results.json", "r") as readfile:
        results = json.load(readfile)
        info = {"args_info":
                    {"ds":f"{train_domain}_{test_domain}", "epochs":training_args.num_train_epochs, 
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

    with open(f"{training_args.output_dir}/test_results.json", "w") as writefile:
        json.dump(results, writefile, ensure_ascii=False, indent=4)
        print(f"\nExpanded {training_args.output_dir}/test_results.json with argument info")

    os.makedirs(f"/models/polarity_{gold}", exist_ok=True)
    tokenizer.save_pretrained(f"models/polarity_{gold}/{train_domain}_{seed}")
    trainer.save_model(output_dir=f"models/polarity_{gold}/{train_domain}_{seed}")


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
    parser.add_argument("--gold", type=str, required=True, help="If True, predictions are done on gold aspects. Otherwise, predicted aspects from same train-test combination.")
    args = parser.parse_args()

    classify_aspect(threshold=args.threshold, lr=args.lr, batch=args.batch, size=args.size, hd=args.hd,
                    wd=args.wd, wr=args.wr, train_domain=args.train_domain, test_domain=args.test_domain, gold=args.gold, 
                    seed=args.seed)

if __name__ == "__main__":
    main()