# daily-llama 🦙

### Retrieval augmented generation framework based on news contents.

This project presents a retrieval-augmented generation framework tailored for news content. By employing the FAISS index for efficient similarity search, relevant documents are retrieved. Leveraging the LLAMA-2 model, these retrieved documents serve as a foundation for generating comprehensive and contextually accurate answers.

![daily-llama](images/daily-llama.png)

### Usage info
```bash
python3 daily-llama.py --dataset_path data/news-small.json --query "What happended to Dimuth in this week?"
```
