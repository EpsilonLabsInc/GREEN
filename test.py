import tempfile
import os
from green_score import GREEN

refs = [
    "Interstitial opacities without changes.",
    "Interval development of segmental heterogeneous airspace opacities throughout the lungs . No significant pneumothorax or pleural effusion . Bilateral calcified pleural plaques are scattered throughout the lungs . The heart is not significantly enlarged .",
    "Lung volumes are low, causing bronchovascular crowding. The cardiomediastinal silhouette is unremarkable. No focal consolidation, pleural effusion, or pneumothorax detected. Within the limitations of chest radiography, osseous structures are unremarkable.",
]
hyps = [
    "Interstitial opacities at bases without changes.",
    "Interval development of segmental heterogeneous airspace opacities throughout the lungs . No significant pneumothorax or pleural effusion . Bilateral calcified pleural plaques are scattered throughout the lungs . The heart is not significantly enlarged .",
    "Endotracheal and nasogastric tubes have been removed. Changes of median sternotomy, with continued leftward displacement of the fourth inferiomost sternal wire. There is continued moderate-to-severe enlargement of the cardiac silhouette. Pulmonary aeration is slightly improved, with residual left lower lobe atelectasis. Stable central venous congestion and interstitial pulmonary edema. Small bilateral pleural effusions are unchanged.",
]

# Use temp directory that works on both Mac and Linux
with tempfile.TemporaryDirectory() as tmpdir:
    green_scorer = GREEN(use_azure=True, output_dir=tmpdir, verbose=True)

    mean, std, green_score_list, summary, result_df = green_scorer(refs, hyps)

    print(green_score_list)
    print(summary)

    for index, row in result_df.iterrows():
        print(f"Row {index}:\n")
        for col_name in result_df.columns:
            print(f"{col_name}: {row[col_name]}\n")
        print("-" * 80)

    # Save to temp file, then it gets auto-deleted when exiting the context
    tmp_csv = os.path.join(tmpdir, "test_green_scores.csv")
    result_df.to_csv(tmp_csv, index=False)
    print(f"Saved to {tmp_csv} (will be deleted)")
# tmpdir and all files inside are automatically deleted here
