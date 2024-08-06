from transformers import pipeline

# Specify the model name explicitly
summarizer = pipeline("summarization", model="facebook/bart-large-cnn")

def summarize_object_attributes(object_texts):
    summaries = {}
    for filename, text in object_texts.items():
        try:
            # Generate the summary
            summary = summarizer(text, max_length=50, min_length=25, do_sample=False)
            summaries[filename] = summary[0]['summary_text']
        except Exception as e:
            print(f"Error summarizing text for {filename}: {e}")
            summaries[filename] = "Error summarizing text"

    return summaries

# Example usage
object_texts = {
    "example_file.txt": "This is an example text that needs to be summarized."
}

object_summaries = summarize_object_attributes(object_texts)
print(object_summaries)