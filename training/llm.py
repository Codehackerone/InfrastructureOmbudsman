from transformers import (
    LlamaTokenizerFast,
    TrainingArguments,
    Trainer,
    LlamaForSequenceClassification,
    BitsAndBytesConfig,
    AutoTokenizer,
    pipeline,
    AutoModelForSequenceClassification,
)
from peft import (
    LoraConfig,
    get_peft_config,
    get_peft_model,
    get_peft_model_state_dict,
    prepare_model_for_kbit_training,
    set_peft_model_state_dict,
    PeftType,
    PromptEncoderConfig,
    PeftModel,
    PeftModelForSequenceClassification,
    AutoPeftModelForSequenceClassification,
    PeftConfig
)
import numpy as np
import pandas as pd
import sklearn.metrics
import bitsandbytes as bnb
import datasets
import evaluate
import torch
from huggingface_hub import login
import os
import random
import spacy
from tqdm import tqdm
import seaborn as sns
import matplotlib.pyplot as plt
import json

login(token='Insert Huggingface Token')

# Custom Trainer for Unbalanced Dataset. Use default Trainer if not balanced.
class InfraTrainer(Trainer):
    def compute_loss(self, model, inputs, return_outputs=False):
        labels = inputs.pop("labels")
        labels = labels.type(torch.LongTensor)
        labels = labels.to('cuda')
        # forward pass
        outputs = model(**inputs)
        logits = outputs.get("logits")
        # compute custom loss (we have 2 labels with greater weight on positive)
        loss_fct = torch.nn.CrossEntropyLoss(weight=torch.tensor([0.5567, 4.9114], device=0))
        loss = loss_fct(logits.view(-1, self.model.config.num_labels), labels.view(-1))
        return (loss, outputs) if return_outputs else loss
class LLMUtils:
    def __init__(self):
        self.metric = evaluate.load("f1")
        self.cf_metric = evaluate.load("BucketHeadP65/confusion_matrix")
        self.classification_report_metric = evaluate.load("bstrai/classification_report")
        self.nlp = spacy.load("en_core_web_sm")
    def ner_mask(self, x):
        '''
        '''
        doc = self.nlp(x['comment'])
        masked = x['comment']
        for e in reversed(doc.ents): #reversed to not modify the offsets of other entities when substituting
            if e.label_ in ['LOC', 'GPE']:
                start = e.start_char
                end = start + len(e.text)
                masked = masked[:start] + '<LOCATION>' + masked[end:]
        x['comment'] = masked
        return x

class LLMTrain():
    def __init__(
            self, 
            data,
            config='llm-config.json',
            mask=True,
            sample=None,
            base_model="meta-llama/Llama-2-7b-hf",
            exp="LLAMA2_nomask"
        ):
        self.metric = evaluate.load("f1")
        self.cf_metric = evaluate.load("BucketHeadP65/confusion_matrix")
        self.classification_report_metric = evaluate.load("bstrai/classification_report")
        self.nlp = spacy.load("en_core_web_sm")
        self.device_map = {"": 0}
        self.sample = sample
        with open(config) as cfile:
            config = json.loads(cfile.read())
            for k, v in config.items():
                setattr(self, k, v)
        self.mask = mask
        self.base_model = base_model
        self.exp = exp
        self.output_dir = os.path.join(os.getcwd(), f"{base_model}_{exp}")
        self.datafile = data
        
    def tokenize_function(self, examples):
        '''
        '''
        return self.tokenizer(examples["comment"], padding="max_length", truncation=True, max_length=512)
    def compute_metrics(self, eval_pred):
        '''
        '''
        logits, labels = eval_pred
        predictions = np.argmax(logits, axis=-1)
        f1 = self.metric.compute(predictions=predictions, references=labels, average="weighted")
        return f1
    def ner_mask(self, x):
        '''
        '''
        doc = self.nlp(x['comment'])
        masked = x['comment']
        for e in reversed(doc.ents): #reversed to not modify the offsets of other entities when substituting
            if e.label_ in ['LOC', 'GPE']:
                start = e.start_char
                end = start + len(e.text)
                masked = masked[:start] + '<LOCATION>' + masked[end:]
        x['comment'] = masked
        return x
    def load_dataset(self, tokenize=True):
        '''
        '''
        # Load Dataset
        df = pd.read_csv(self.datafile)
        df['labels'] = df['label'].astype('int')
        dt = datasets.Dataset.from_pandas(df)
        if self.sample:
            dt = dt.select(range(self.sample))
        if self.mask:
            dt = dt.map(self.ner_mask)
        dt = dt.train_test_split(test_size=0.3, seed=42)
        self.train_data = dt['train'].shuffle(seed=42)
        self.test_data = dt['test'].shuffle(seed=42)
        if tokenize:
            print(f'Tokenizing dataset')
            self.small_train = self.train_data.map(self.tokenize_function, batched=True, remove_columns=['id', 'comment', 'Unnamed: 0', 'label'])
            self.small_test = self.test_data.map(self.tokenize_function, batched=True)
    def load_qlora(self):
        model_id = '/home/mac9908/InfrastructureOmbudsman/LLAMA2_mask'
        self.model = AutoPeftModelForSequenceClassification.from_pretrained(model_id)
        print(type(self.model))
        self.model = self.model.merge_and_unload()
        self.model.save_pretrained('LLAMA2_mask_merged')
        print(type(self.model))
    
    def load_trainer(self):
        # Load tokenizer and model with QLoRA configuration
        compute_dtype = getattr(torch, self.bnb_4bit_compute_dtype)

        bnb_config = BitsAndBytesConfig(
            load_in_4bit=self.use_4bit,
            bnb_4bit_quant_type=self.bnb_4bit_quant_type,
            bnb_4bit_compute_dtype=compute_dtype,
            bnb_4bit_use_double_quant=self.use_nested_quant,
        )

        # Check GPU compatibility with bfloat16
        if compute_dtype == torch.float16 and self.use_4bit:
            major, _ = torch.cuda.get_device_capability()
            if major >= 8:
                print("=" * 80)
                print("Your GPU supports bfloat16: accelerate training with bf16=True")
                print("=" * 80)

        self.model = AutoModelForSequenceClassification.from_pretrained(
            self.base_model,
            quantization_config=bnb_config,
            device_map=self.device_map,
        )
        # for name, module in model.named_modules():
        #   print(name)
        self.model.config.use_cache = False
        self.model.config.pretraining_tp = 1

        
        # # Load LLaMA tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(self.base_model, trust_remote_code=True)
        self.tokenizer.pad_token = self.tokenizer.eos_token
        self.model.config.pad_token_id = self.model.config.eos_token_id
        # tokenizer.padding_side = "right" # Fix weird overflow issue with fp16 training

        self.load_dataset()
        # Load LoRA configuration
        peft_config = LoraConfig(
            lora_alpha=self.lora_alpha,
            lora_dropout=self.lora_dropout,
            r=self.lora_r,
            bias="none",
            task_type="SEQ_CLS",
        )

        self.model.gradient_checkpointing_enable()
        self.model = prepare_model_for_kbit_training(self.model)
        self.model = get_peft_model(self.model, peft_config)

        # Set training parameters
        self.training_arguments = TrainingArguments(
            output_dir=self.output_dir,
            num_train_epochs=self.num_train_epochs,
            per_device_train_batch_size=self.per_device_train_batch_size,
            gradient_accumulation_steps=self.gradient_accumulation_steps,
            optim=self.optim,
            save_strategy='epoch',
            logging_steps=self.logging_steps,
            learning_rate=self.learning_rate,
            weight_decay=self.weight_decay,
            evaluation_strategy='epoch',
            fp16=self.fp16,
            bf16=self.bf16,
            max_grad_norm=self.max_grad_norm,
            max_steps=self.max_steps,
            warmup_ratio=self.warmup_ratio,
            group_by_length=self.group_by_length,
            lr_scheduler_type=self.lr_scheduler_type,
            load_best_model_at_end=True,
            metric_for_best_model='eval_f1',
            report_to="wandb"
        )
        self.trainer = InfraTrainer(
            model=self.model,
            train_dataset=self.small_train,
            eval_dataset=self.small_test,
            tokenizer=self.tokenizer,
            args=self.training_arguments,
            compute_metrics=self.compute_metrics,
        )
    def train(self):
        self.load_trainer()
        # Train model
        self.trainer.train()
        # Save trained model
        self.trainer.save_model(f'{self.exp}')
        self.tokenizer.save_pretrained(f'{self.exp}')
    def predict(self, x):
        x['predict'] = self.pipe(x['comment'])[0]['label']
        return x
    def do_eval(self, data):
        pipe = pipeline(
              task='text-classification',
              model=self.model,
              tokenizer=self.tokenizer,
              padding="max_length",
              truncation=True,
              max_length=512
        )
        def predict(x):
            x['predict'] = pipe(x['comment'])[0]['label']
            return x
        res = data.map(predict)
        res = res.to_pandas()
        res['predict'] = res['predict'].map({'LABEL_0': 0, 'LABEL_1': 1})
        print(f"Precision Score: {sklearn.metrics.precision_score(res['label'], res['predict'])}")
        print(f"Accuracy Score: {sklearn.metrics.accuracy_score(res['label'], res['predict'])}")
        print(f"Balanced Accuracy Score: {sklearn.metrics.balanced_accuracy_score(res['label'], res['predict'])}")
        print(f"F1: {sklearn.metrics.f1_score(res['label'], res['predict'])}")
        print(f"Classification Report:\n {sklearn.metrics.classification_report(res['label'], res['predict'])}")
        cm = sklearn.metrics.confusion_matrix(res['label'], res['predict'], labels=[0, 1])
        disp = sklearn.metrics.ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=[0,1])
        disp.plot()
        plt.show()
        return res        