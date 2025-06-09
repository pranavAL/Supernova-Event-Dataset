# Supernova Event Dataset: Official Code Repository

Official repository for the research paper **"Supernova Event Dataset: Interpreting Large Language Model's Personality through Critical Event Analysis"** by

[Pranav Agarwal](https://pranaval.github.io/) and [Ioana Ciucă](https://www.iciuca.com/)

In this work, we interpret the personality traits of Large Language Models (LLMs) using our proposed Supernova Event Dataset, which includes Wikipedia articles consisting of historical events, biographies, news events, and scientific discoveries. We benchmark models based on their identification and ranking of key life or discovery events, a complex task requiring causal reasoning. A second LLM acts as a judge to infer each model’s personality based on its event selection and interpretation. Our analysis show distinct traits—like emotional reasoning in Orca 2 and analytical framing in Qwen 2.5—enhancing interpretability and trust.

[Paper]()  [Webpage]() [Hugging Face Dataset](https://huggingface.co/datasets/SupernovaEvent/SupernovaEventDataset)

<p align="center">
  <img width="100%" src="assets/intro_fig-1.png">
</p>

## Dependencies
Create a virtual environment and install all the required files
```
python3.8 -m venv myenv
source myenv/bin/activate
pip install -r requirements.txt
```

## Dataset
The Supernova Event Dataset consists of of 592 Wikipedia articles across 3 domains:
* Biographies (192 articles): Life stories of celebrities.
* News events (200 articles): Major recent events.
* Historical events (200 articles): Improtant historical events.

For the event of Scientific Discovery, we used Google's Gemini 2.5 Pro with Deep Research and sampled 25 fully formed encyclopedic articles that cover historical context, methodology, publication trail, significance, and legacy of a scientific discovery. 

### Dataset Extraction
```
tar -zxvf Dataset/biographies.tar.xz
tar -zxvf Dataset/historical-events.tar.xz
tar -zxvf Dataset/major-news-events.tar.xz
```

### Extract and Rank Critical Events

#### Biographies
```
python biography_dataset.py --model model_name
```

#### Historical Events
```
python history_dataset.py --model model_name
```

#### News Events
```
python news_dataset.py --model model_name
```

#### Movies Scrips
```
python movies_dataset.py --model model_name
```

#### Save the Model personality
The above code extracts independently the results for each categories. To save the personality for each model for event in the given categories in a single csv file, run
```
python extract_personality.py
```

#### Create the radar plot and Semantic Space Mapping 
Based on the csv file from above, the radar plot and semantic space mapping can be created using
```
python plot_personality.py
```

## Citation
If you use this code in your research, please cite our paper:
```bibtex
@article{agarwal2025supernova,
  title={Supernova Event Dataset: Interpreting Large Language Model’s Personality through Critical Event Analysis},
  author={Agarwal, Pranav and Ciucă, Ioana},
  journal={},
  year={2025}
}
```
