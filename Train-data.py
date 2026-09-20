training_args = TrainingArguments(
    output_dir="./Artifacts/absa_results",
    num_train_epochs=8,
    per_device_train_batch_size=8,
    per_device_eval_batch_size=8,
    warmup_steps=50,
    weight_decay=0.01,
    learning_rate=2e-5,
    logging_dir="./Artifacts/absa_logs",
    logging_steps=10,
    eval_strategy="epoch",
    save_strategy="epoch",
    load_best_model_at_end=True,
    metric_for_best_model="f1_macro",
    save_total_limit=1,
    report_to="none" # added this line to not report to wandb
)

# 8. Initialize Trainer
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=val_dataset,
    compute_metrics=compute_metrics
)

# 9. Train Model
trainer.train()

# 10. Evaluate on Test Set
test_results = trainer.evaluate(test_dataset)
print("Test Results:", test_results)

