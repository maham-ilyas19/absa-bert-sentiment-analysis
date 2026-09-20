
# 11. Save Model and Tokenizer
model.save_pretrained("./Artifacts/absa_model")
tokenizer.save_pretrained("./Artifacts/absa_model")
label_encoder.classes_.dump("./Artifacts/label_encoder_classes.npy")

# 12. Inference Example
def predict_sentiment(review, aspect, model, tokenizer, label_encoder):
    model.eval()
    text = f"{review} [SEP] {aspect}"
    encoding = tokenizer(
        text,
        max_length=128,
        padding="max_length",
        truncation=True,
        return_tensors="pt"
    )
    input_ids = encoding["input_ids"].to(model.device)
    attention_mask = encoding["attention_mask"].to(model.device)

    with torch.no_grad():
        outputs = model(input_ids, attention_mask=attention_mask)
        logits = outputs.logits
        prediction = torch.argmax(logits, dim=1).cpu().numpy()[0]

    return label_encoder.inverse_transform([prediction])[0]

# Test inference
sample_review = "The prices were a bit high, but the food was not worth it."
sample_aspect = "food"
predicted_sentiment = predict_sentiment(sample_review, sample_aspect, model, tokenizer, label_encoder)
print(f"Predicted sentiment for '{sample_aspect}' in '{sample_review}': {predicted_sentiment}")
