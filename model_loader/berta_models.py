from transformers import pipeline, AutoTokenizer, AutoModelForTokenClassification

def load_deberta_ner_model():
    model_name = "kamalkraj/deberta-v3-base-ner"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForTokenClassification.from_pretrained(model_name)
    ner_pipeline = pipeline("ner", model=model, tokenizer=tokenizer, aggregation_strategy="simple")
    return ner_pipeline
