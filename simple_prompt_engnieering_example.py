import time
import psutil
import random
from llama_cpp import Llama
import pandas as pd
import unicodedata
process = psutil.Process()
file_path = "D:\camera detection\models\llama-2-7b-chat.ggmlv3.q2_K.bin"
llm = Llama(
    model_path=file_path, n_ctx=1000, n_parts=0, seed=-1, n_threads=4, verbose=True
)
# df = pd.read_csv("/Users/shadialarbed/Downloads/sharjah_reviews_googlegmps.csv")
data = {
    "Client's Name": ["carolina"],
    "reviews": ["The salon experience was amazing. Staff was friendly and professional."],
    "Special Service/Offer Based on Positive Sentiment": ["50% off spa package"],
    "Salon Phone Number": ["(123) 456-7890"],
    "Special Code": ["SPECIAL123"],
    "Salon Address": ["jpt"],
    "salon_name": ["XYZ"],
    "Sentiment": ["negative"],
}
df1 = pd.DataFrame(data)
print(df1["reviews"])
# Extract data from DataFrame
client_name = df1["Client's Name"][0]
salon_address = df1["Salon Address"][0]
salon_name = df1["salon_name"][0]
sentiment = df1["Sentiment"][0]
last_visit = "1 month"
# for j in range(1000):
#     i = random.randint(500, 1000)
#     def custom_prompt2(
#         client_name,
#         salon_name,
#         salon_address,
#         last_visit
#     ):
#         prompt3 = f"""
#         write an email, from salon management in {salon_name} to {client_name}.
#         include these information:
#         Location: {salon_address}, Last Visit: {last_visit}, review: {df1["reviews"]}.
#         Express enthusiasm to serve them again and thank them for their feedback.
#         Write it in 200 words only.
#         """
#         return prompt3
    
    
prompt3 = f"""
write an email, from salon management in {salon_name} to {client_name}.
include these information:
Location: {salon_address}, Last Visit: {last_visit}, review: {df1["reviews"]}.
Express enthusiasm to serve them again and thank them for their feedback.
Write it in 200 words only.
        """
# prompt = custom_prompt2(client_name,salon_name,salon_address,last_visit)
# prompt = "generate an appolgy email to my boss for not comming today"
em = llm.create_completion(prompt3,max_tokens=512,temperature=0.4,top_k=40)
# print(em)
print(em['choices'][0]['text'])