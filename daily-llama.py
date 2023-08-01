from model.generator import DailyLLAMA

model  = DailyLLAMA(source_data_path='/home/ransaka/daily-llama/data/news-small.json', source_column='title')
resp = model("What Dimuth tells about his cricket career?")
print(resp)
