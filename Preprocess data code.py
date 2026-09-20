# 2. Preprocess Data
data = []
for review_entry in dataset:
    review = review_entry["review"]
    for aspect_entry in review_entry["aspects"]:
        data.append({
            "review": review,
            "aspect": aspect_entry["aspect"],
            "sentiment": aspect_entry["sentiment"]
        })

print(data)
df = pd.DataFrame(data)

# Encode sentiments
label_encoder = LabelEncoder()
df["label"] = label_encoder.fit_transform(df["sentiment"])

# Split data: 80% train, 10% val, 10% test
train_df, temp_df = train_test_split(df, test_size=0.2, random_state=42, stratify=df["label"])
val_df, test_df = train_test_split(temp_df, test_size=0.5, random_state=42, stratify=temp_df["label"])
