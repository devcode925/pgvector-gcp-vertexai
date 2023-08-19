
## https://www.linkedin.com/pulse/llmlarge-language-models-numeric-data-think-stats-chuck-hernandez%3FtrackingId=Ur%252FigJ6jRMihj378PKV%252Fjg%253D%253D/?trackingId=Ur%2FigJ6jRMihj378PKV%2Fjg%3D%3D

LLMs are a powerful tool for understanding and analyzing text data. However, LLMs can also be used to understand and analyze numeric data. In this article, I will discuss how LLMs can be used to answer questions about numeric data, and I will share some code snippets of how I have used an LLM to analyze marketing data.
Highlights:
LLM’s for Enterprise specific text or document data
LLM’s for Statistics 
LangChain use in Inference

Back in June I attended a Data and AI conference hosted by Databricks. It was a great conference with lots of workshops and updated information about what is going on in AI and Data. 
LLM’s for the Enterprise was a big focal point and looked to be the immediate future for companies that want to leverage AI. For me one of the best discoveries was the use of Vector databases, to store and search documents or text. This would allow enterprises to now search their own data on their own systems using one of the open-source Foundational Models like Llama2 or PaLM 2, and Falcon-40B.  These models are more than powerful and accurate enough to help enterprises search their data troves of information with accuracy, security and privacy.  For example, a company could use an LLM to search for customer records in their own data on their own private clouds.
This is big news, Enterprises won’t have to train models from scratch, they can use pre-trained LLM models and now use vector databases to store and access their own data. Saving the cost of training machine learning models and getting quick accurate access to their documents/data.
LLM for Statistics
One use case I had trouble finding an answer for was, how do I use LLM’s to help understand my statistical data? Rightfully so, right? Language not Numeric is the first word for these models. Still, I wanted to try. The data I had in mind was numeric data for marketing metrics. There's a myriad of marketing data exports that could use some explanation. Could a LLM answer questions such as “What day and time will a photo post get the most engagement?” Based on my statistical data?
From my testing I believe the answer is yes. The trick is to have the LLM use our numeric data either by search and/or context. Which should allow us to ask questions about what’s in our numeric data, returning to us answers we can understand.
Data
The data is Facebook metrics. From https://data.world/uci/facebook-metrics/workspace/data-dictionary
This is what it looks like;
“139441,Photo,2,12,4,3,0,2752,5091,178,109,159,3078,1640,119,4,79,17,100” 
Hundreds of rows numeric CSV data. Each column does have meaning, it’s just needs to be conveyed to the LLM model. 
The Platform
After looking for just the right combination of technologies and platform, I went with Google's Cloud Platform. This made it easy for me to have the vector data, and my standard relational database together. I want to store the original data for context and historical reasons. Plus, the process would work for doing a purely text or document project, that would respond to human language queries about the data stored in the PostgreSQL database.
The main reason for using Postgres on Google was to allow me to use their newly released pgvector extension. The extension allows me to create vector data from our data and store it in Postgres with the columns it’s related to. Basically, allowing me to provide “context” to my statistical data. This blog was a huge help for me to set up the necessary tools and customize some of the routines to add my new processes.  
The Process
First, I created a Postgres database and loaded it with CSV data using a Panda’s Dataframe. Then created a vector from the data in each row, adding context to each element. 
Here’s an example of the type context created for each column.
string += f"Page Total likes is {df.page_total_likes[row_index]}.The type of Post is {df.post_type[row_index]}."
Context added information, such as why the column is important, and descriptions of the columns use. These rows and columns were then converted to vector embeddings. I did use the LangChain “chunk” process to keep the size down.
From there, I created a table to hold the embeddings in my Postgres database.
Using the Google example, the user query, generates its vector embeddings and use pgvector vector similarity search operators to find the closest matching row after applying the relevant SQL filters.
Since I am using numeric data, there’s probably not going to be a match on the SQL filters, just the vector embeddings should return something. 
Then I used the MapReduceChain from LangChain framework to generate a summarized context using an LLM model (Google PaLM2 in this case).
The last step is to pass the context to an LLM prompt to answer the user query. The LLM model returns a well-formatted natural sounding response back to us.
If there are no results, we display no results. We can control the “Similarity threshold”, which greatly affects whether you get an answer or not. “.5 - .7” was the sweet spot for me.
LangChain for Inference
I believe this is where I made the most customizations to handle numeric data.
map_prompt_template = "

              You will be given a question about the data in the vector embeddings.

              the data in the embeddings are Facebook statistics for Posts.

               Select the right column described in the question and do the analysis.

                You should use SUM() and AVG(), MEAN() when necessary. Always return the row_id. The question will be enclosed in triple backticks (```).

              Using this description only, extract the number for the column referenced and return your answer.

              ```{text}```

              SUMMARY:

              """

map_prompt = PromptTemplate(template=map_prompt_template, input_variables=["text"])

combine_prompt_template = """

  You will be given a question about the Facebook statistics about POSTs.
  
  All information for answering the question is in the vector embeddings and fb_stats database. (```) and a question enclosed in
  
  double backticks(``).
  
  Select the right column described in the question and do the analysis.
  
  You should use SUM() and AVG(), MEAN() when necessary.
  
  You should only use the information in the description. always return the row_id.
  
  Your answer should include the explanation for your answer. Your answer should be less than 200 words.
  
  Your answer should be in a numbered list format.
  
  Description:
  
           ```{text}```
    
  
          Question:
  
          ``{user_query}``
  
         Answer:
  
           """
  
  combine_prompt = PromptTemplate(
  
     template=combine_prompt_template, input_variables=["text", "user_query"]
  )"
The results were encouraging. Here’s some of the questions I was able to ask, and the answers the LLM returned.
Question: What day and time will a photo post get the most engagement?
Answer: The day and time that a photo post will get the most engagement is on a Saturday at 10:00 AM. This is because the average engagement for a photo post on a Saturday at 10:00 AM is 10.00, which is higher than the average engagement for a photo post on any other day or time.
Question: What type of posts will get the most likes?
Answer: The most likes are for posts with the word "love".
Next up is applying more numeric data, like the cost of marketing campaigns and expected results. I also want to apply the solution to more types of data, such as financial data and customer sales data.
The end goal for me is to have an LLM take questions and give answers for business users based on their own numeric and text data. The business users know what questions to ask. And it looks like a LLM using the business’s data can give the answers they need.
References:
https://cloud.google.com/blog/products/databases/using-pgvector-llms-and-langchain-with-google-cloud-databases
https://www.pinecone.io/learn/vector-database/
https://towardsdatascience.com/document-oriented-agents-a-journey-with-vector-databases-llms-langchain-fastapi-and-docker-be0efcd229f4
https://api.python.langchain.com/en/latest/_modules/langchain/llms/vertexai.html#VertexAI
https://www.youtube.com/watch?v=3fsn19OI_C8
https://colab.research.google.com/gist/PubliusAu/1d3f3e6d1497e80f72f3e887e8a2bdc1/zero-shot-anomaly-detection-using-llms-california-medium-home-values-arize.ipynb
https://www.pinecone.io/learn/series/langchain/langchain-tools/
https://towardsdatascience.com/what-i-learned-pushing-prompt-engineering-to-the-limit-c40f0740641f
