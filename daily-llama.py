from model.generator import DailyLLAMA

model  = DailyLLAMA(source_data_path='/home/ransaka/daily-llama/data/news-small.json', source_column='title', content_column='content')
resp = model("What happened to rupee in this week?")
print(resp)
