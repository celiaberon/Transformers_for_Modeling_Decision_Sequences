from gitingest import ingest

repo_url = "https://github.com/celiaberon/Transformers_for_Modeling_Decision_Sequences"
output_file = "gitingest.txt"

# Ingest the repo
summary, tree, content = ingest(repo_url)

# Save to a single .txt file
with open(output_file, "w", encoding="utf-8") as f:
    f.write(summary + "\n\n")
    f.write(tree + "\n\n")
    f.write(content)