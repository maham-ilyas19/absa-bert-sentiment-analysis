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
