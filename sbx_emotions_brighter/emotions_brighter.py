from sparv.api import Annotation, Output, annotator, get_logger
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch
import os

logger = get_logger(__name__)

@annotator("Classify each sentence with emotions",
           language=["swe"])

def emotions(
    sentence: Annotation = Annotation("<sentence>"),
    out: Output = Output("<sentence>:sbx_emotions_brighter.emotion",description="Emotions found in sentence"),
    ):  
    
    #load model
    model_path = 'sbx/KB-bert-base-swedish-cased_emotions_brighter'

    tokenizer = AutoTokenizer.from_pretrained(model_path)
    model = AutoModelForSequenceClassification.from_pretrained(model_path)

    annotations= sentence.create_empty_attribute()

    #get predictions per senctence
    for i,sent in enumerate(sentence.read()):
        predictions = get_preds_single_sent(sent,tokenizer, model)
        annotations[i]=predictions
    
    # write predictions to out
    out.write([d for d in annotations])


def get_preds_single_sent(sent,tokenizer,model):

    label_list = ['anger', 'disgust', 'fear', 'joy', 'sadness', 'surprise']

    #threshold to filter out lower scores
    threshold=0.1

    inputs = tokenizer(
        sent,
        return_tensors="pt",
        max_length= 512,
        padding='max_length',
        truncation = True
    )

    model.eval()

    with torch.no_grad():
        logits = model(**inputs).logits  
        probs = torch.sigmoid(logits).squeeze().tolist()  

    # Map labels to probabilities
    label_scores = list(zip(label_list, probs))

    predicted_labels = sorted(
        [(label,round(score,2)) for label,score in label_scores if score > threshold],
        key=lambda x:x[1],
        reverse=True)

    labels_str = (
        "|"
        if not predicted_labels
        else "|" + "|".join(f"{lbl},{score:.2f}" for lbl, score in predicted_labels) + "|")
    
    return labels_str
    






