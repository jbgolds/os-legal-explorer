# Goal of this project
I want to create a very interactive website using d3.js, htmx, and a python backend preferrably something simple, like fastapi. 

- There will be no accounts, just a frontend with some different views, probably tabs work for now.
- We will get more into the visualizations later.


I have a local postgres database running with a copy `[courtlistenerof](http://courtlistener.com)'s dumps of all of their data. With specifically the Court Opinions text, I will be mappings of citations found in these cases to one another, while staying consistent with courtlistener's data to hopefully integrate one day. I've built a pipeline already to use LLM's to extract the citation's treatment and other metadata out of all citations in the opinion's text. I also have a neo4j database where I'd like to build a graph database of these citation representations. I have models built out for all three of these datatypes to use in python. I have a VPS with 64GB of RAM and 2TB SSD, 16 dedicated server cores, so we have plenty of processing power.

Needs to be done:
- connect the pipes and start getting citations loaded in! 
- build out a python web backend to manage two things:
    1. the pipeline between the csv export of the database table, calling LLM, resolving opinion cluster ID, and entering into neo4j.
    2. web app using a python + htmx + d3.js + tailwindcss to create a nice looking website to display our rich graphs of information.
    

