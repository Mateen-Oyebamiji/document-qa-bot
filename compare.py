import csv

A = ("fixed_500_50", "vector")      # app today
B = ("semantic_p75", "hybrid")      # best setup
K = 5

rows = list(csv.DictReader(open("evaluation/results/per_question.csv", encoding="utf-8")))


def hits(setup):
    return {
        row["id"]: (row["page_rank"] != "" and int(row["page_rank"]) <= K, row)
        for row in rows
        if (row["chunker"], row["retriever"]) == setup
    }


def top(row, n=5):
    return " ".join(row["retrieved_pages"].split()[:n])


a, b = hits(A), hits(B)

for label, wanted in [(f"Fixed by {B}", (False, True)), (f"Broken by {B}", (True, False))]:
    print(f"\n{label}:")
    for qid in a:
        if (a[qid][0], b[qid][0]) == wanted:
            row = b[qid][1]
            print(f"  {qid} {row['question']}")
            print(f"      gold {row['gold_pages']} | A top 5: {top(a[qid][1])} | B top 5: {top(row)}")

print("\nMissed by both:")
for qid in a:
    if not a[qid][0] and not b[qid][0]:
        print(f"  {qid} {a[qid][1]['question']} (gold {a[qid][1]['gold_pages']})")
