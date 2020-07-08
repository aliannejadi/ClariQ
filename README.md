# ClariQ

## Introduction

The challenge is organized as part of the Search-oriented Conversational AI (SCAI) EMNLP
workshop in 2020. The main aim of the conversational systems is to return
an appropriate answer in response to the user requests. However, some user
requests might be ambiguous. In Information Retrieval (IR) settings such a situation is handled mainly
through the diversification of search result page. It is however much more challenging in dialogue settings. 
Hence, we aim to study the following situation for dialogue settings:

* a user is asking an ambiguous question (where ambiguous question is a
question to which one can return > 1 possible answers);
* the system must identify that the question is ambiguous, and, instead of
trying to answer it directly, ask a good clarifying question.

The main research questions we aim to answer as part of the challenge are
the following:

* RQ1: When to ask clarifying questions during dialogues?
* RQ2: How to generate the clarifying questions?

## Challenge Design

The ClariQ challenge is run in two stages. At Stage 1 (described below)
participants are provided a static dataset consisting mainly of an initial user
request, clarifying question and user answer, which is suitable for initial training,
validating and testing. At Stage 2, we bring a human
in the loop. Namely, the TOP-N systems, resulted from Stage 1, are exposed
to the real users.

### Stage 1: initial dataset

Taking inspiration from [Qulac](https://github.com/aliannejadi/qulac) [[1]](#ref1) dataset,
 we have crowdsourced a new dataset to study clarifying questions that is suitable for conversational settings. 
Namely, the collected dataset consists of:

* **User Request:** an initial user request in the conversational form, e.g.,
"What is Fickle Creek Farm?", with a label reflects if clarification is needed
ranged from 1 to 4;
* **Clarification questions:** a set of possible clarifying questions, e.g., "Do
you want to know the location of fickle creek farm?";
* **User Answers:** each questions is supplied with a user answer, e.g., "No, I
want to find out where can i purchase fickle creek farm products."

For training, the collected dataset is split into training (187 topics) and validation
(50 topics) sets. For testing, the participants are supplied with: (1) a set of user
requests in conversational form and (2) a set a set of questions (i.e., question
bank) which contains all the questions that we have collected for the collection.
Therefore to answer our research questions, we suggest the following
two tasks:

* To answer RQ1: Given a user request, return a score [1 −4] indicating the
necessity of asking clarifying questions.
* To answer RQ2: Given a user request which needs clarification, return the
most suitable clarifying question. Here participants are able to choose: (1)
either select the clarifying question from the provided question bank (all
clarifying questions we collected), aiming to maximize the precision, (2) or
choose not to ask any question (by choosing `Q0001` from the question bank.)

### Stage 2: human-in-the-loop
To be announced.

## ClariQ Dataset
We have extended the [Qulac](https://github.com/aliannejadi/qulac) [[1]](#ref1) dataset and base the competition mostly
 on the training data that [Qulac](https://github.com/aliannejadi/qulac) provides. 
 In addition, we have added some new topics, questions, and answers in the training set. 
 The test set is completely unseen and newly collected. 
 Like Qulac, ClariQ consists of single-turn conversations (`initial_request`, followed by clarifying `question` and `answer`).
 In addition, it comes with synthetic multi-turn conversations (up to three turns). ClariQ features approximately 18K single-turn conversations, as well as 1.8 million multi-turn conversations. 
Below, we provide a short summary of the data characteristics, for the training set:

### ClariQ Train
Feature  						| Value
--------------------------------| -----
\# train (dev) topics			| 187 (50)
\# faceted topics 				| 141
\# ambiguous topics 			| 57
\# single topics				| 39
\# facets 						| 891
\# total questions              | 3,929
\# single-turn conversations    | 11,489
\# multi-turn conversations     | ~ 1 million 
\# documents                    | ~ 2 million

Below, we provide a brief overview of the structure of the data, as well as a guideline on how to submit the runs.

## Files
Below we list the files in the repository:

* `./data/train.tsv` and `./data/dev.tsv` are TSV files consisting of topics (queries), facets, clarifying questions, user's answers, and labels for how much clarification is needed (`clarification needs`).
* `./data/test.tsv` is a TSV file consisting of test topic ID's, as well as queries (text).
* `./data/question_bank.tsv` is a TSV file containing all the questions in the collection, as well as their ID's. Participants' models should select questions from this file.
* `./data/top10k_docs_dict.pkl.tar.gz` is a `dict` containing the top 10,000 document ID's retrieved from ClueWeb09 and ClueWeb12 collections for each topic. This may be used by the participants who wish to leverage documents content in their models. 
* `./data/single_turn_train_eval.pkl` is a `dict` containing the performance of each topic after asking a question and getting the answer. The evaluation tool that we provide uses this file to evaluate the selected questions.
* `./src/clariq_eval_tool.py` is a python script to evaluate the runs. The participants may use this tool to evaluate their models on the `dev` set. We would use the same tool to evaluate the submitted runs on the `test` set.
* `./sample_runs/` contains some sample runs and baselines. Among them, we have included the two oracle models `BestQuestion` and `WorstQuestion`, as well as `NoQuestion`, the model choosing no question. Participants may check these files as sample run files. Also, they could test the evaluation tool using these files.

## File Format

### `train.tsv`, `dev.tsv`:

`train.tsv` and `dev.tsv` have the same format. They contain the topics, facets, questions, answers, and clarification need labels. These are considered to be the main files, containing the labels of the training set. Note that the `clarification needs` labels are already explicitly included in the files. Regarding the `question relevance` labels for each topic, these labels can be extracted inderictly: each row only contains the questions that are considered to be relevant to a topic. Therefore, any other question is deemed irrelevant while computing `Recall@k`. 
In the `train.tsv` and `dev.tsv` files, you will find these fields:


* `topic_id`: the ID of the topic (`initial_request`).
* `initial_request`: the query (text) that initiates the conversation.
* `topic_desc`: a full description of the topic as it appears in the TREC Web Track data.
* `clarification_need`: a label from 1 to 4, indicating how much it is needed to clarify a topic. If an `initial_request` is self-contained and would not need any clarification, the label would be 1. While if a `initial_request` is absolutely ambiguous, making it impossible for a search engine to guess the user's right intent before clarification, the label would be 4.
* `facet_id`: the ID of the facet.
* `facet_desc`: a full description of the facet (information need) as it appears in the TREC Web Track data.
* `question_id`: the ID of the question as it appears in `question_bank.tsv`.
* `question`: a clarifying question that the system can pose to the user for the current topic and facet.
* `answer`: an answer to the clarifying question, assuming that the user is in the context of the current row (i.e., the user's initial query is ``initial_request``, their information need is `facet_desc`, and `question` has been posed to the user).

Below, you can find a few example rows of `train.tsv`:


topic\_id | initial\_request | topic\_desc | clarification\_need | facet\_id | facet\_desc | question\_id | question | answer 
---------|---------|--------------|----------------------------|-------|-----------|--------|-----|---
14	 | I'm interested in dinosaurs |	I want to find information about and pictures of dinosaurs. | 	4 | 	F0159	| Go to the Discovery Channel's dinosaur site, which has pictures of dinosaurs and games. | 	Q00173 | 	are you interested in coloring books | 	no i just want to find the discovery channels website
14	| I'm interested in dinosaurs | 	I want to find information about and pictures of dinosaurs.	| 4 | F0159	| Go to the Discovery Channel's dinosaur site, which has pictures of dinosaurs and games. | 	Q03021	| which dinosaurs are you interested in | 	im not asking for that i just want to go to the discovery channel dinosaur page

### `test.tsv`:
`test.tsv` only contains the list of test topics, as well as their ID's. Below we see some sample rows:

topic\_id | initial\_request
------|--------
201	 | I would like to know more about raspberry pi
202	 | Give me information on uss carl vinson.

### `question_bank.tsv`: 
`question_bank.tsv` constitutes of all the questions in the collection. So, all the questions that participants may re-rank and select for the test set are also included in this question bank. The TSV file has two columns, `question_id`, which is a unique ID to the question, and `question`, which is the text of the question. Below we see some example rows of the file:

question\_id | question
------|--------
Q00001 | 
Q02318 |	what kind of medium do you want this information to be in
Q02319	 | what kind of penguin are you looking for
Q02320	| what kind of pictures are you looking for

**Note:** Question id `Q00001` is reserved for cases when a model predicts that asking clarifying questions is not required. Therefore, selecting `Q00001` means selecting no question.

### `single_turn_train_eval.pkl`
`single_turn_train_eval.pkl` is a `dict` of pre-computed document relevance results after asking each question.  The document relevance performance is calculated as follows:

* For a facet, the selected question and its corresponding answer are added to the document retrieval system.
* The document retrieval model [[1]](#ref1) , then re-ranks the documents with the given question and answer.
* The performance of the newly-ranked document is then computed as follows. For every given facet, the effect of asking the question can be determined using the pre-computed `dict`. Below we see the structure of the `dict`:
	
		{ <evaluation_metric>: 
			[ 
			  <facet_id>: 
			  {
		  	    <question_id> : 
			  	 {
			  	   'no_answer': <float>,
			  	   'with_answer': <float>
			  	 }
			  	 , ... , 
			  	 'MAX': 
			  	  {
			  	    'no_answer': <float>,
			  	    'with_answer: <float>
			  	  },
			  	 'MIN':
			  	  {
			  	    'no_answer: <float>,
			  	    'with_answer: <float>
			  	  } 
			  }
		  ]
		  ...
		}	
	

As we see, one has first to identify the `evaluation_metric` they are interested in, followed by a `facet_id` and `question_id`. Notice that here we report the retrieval performance for both with and without considering the answer to the question. Furthermore, we also include two other values, namely, `MAX` and `MIN`. These refer to the maximum and minimum performance that the retrieval model achieves by asking the "best" and "worst" questions among the candidate questions. Below we see a sample of the data:

	{ 'NDCG20: 
		[ 
		  'F0513': 
		  {
	  	    'Q00045' : 
		  	 {
		  	   'no_answer': 0.2283394055312402,
		  	   'with_answer': 0.2233114358097999
		  	 }
		  	 , ... , 
		  	 'MAX': 
		  	  {
		  	    'no_answer': 0.30202557044031736,
		  	    'with_answer: 0.28863807501469424
		  	  },
		  	 'MIN':
		  	  {
		  	    'no_answer: 0.16989316652772574,
		  	    'with_answer: 0.054861833842573086
		  	  } 
		  }
	  ]
	  ...
	}

Notice that this `dict` contains the following evaluation metrics: 

* nDCG@{1, 3, 5, 10, 20}
* Precision@{1, 3, 5, 10, 20}
* MRR@100

**Note**: If a question is selected for a topic, that is not among the candidate questions (thus not appearing in `single_turn_train_eval.pkl`, the document relevance is assumed to be equal to `MIN` for the facet. 

### `top10k_docs_dict.pkl.tar.gz`
`top10k_docs_dict.pkl.tar.gz` is a `dict` consisting of a `list` of document ID's for a given `topic_id`. In case one plans to use the contents of a document in their model, and does not have access to ClueWeb09 or ClueWeb12 data collections, this `dict` is useful for having the list of top 10,000 documents as an initial ranking. The participants can use this list for two purposes:

* To get access to the full text of the documents listed in this `dict`. For this, we suggest using the [ChatNoir](http://chatnoir.eu)'s API [[2]](#ref2) . Upon request, we provide the participants with an API key, using which they can get access by providing a document's ID. Sample codes will be added soon.
**Note**: The ClueWeb document ID should be translated into a UUID used by [ChatNoir](http://chatnoir.eu). ChatNoir provides a simple JavaScript for this purpose: [https://github.com/chatnoir-eu/webis-uuid](https://github.com/chatnoir-eu/webis-uuid).
More information on how to use `ChatNoir`'s API: [https://www.chatnoir.eu/doc/api/#retrieving-full-documents](https://www.chatnoir.eu/doc/api/#retrieving-full-documents)
* Get a pre-build index to re-run document retrieval. We advise the participants to contact us if they require access to pre-build index files to re-run document retrieval. We recommend viewing the `QL.py` in [Qulac](https://github.com/aliannejadi/qulac)'s repository for more information on how the pre-build index files could be used. 

## ClariQ Evaluation Script
We provide an evaluation script, called `clariq_eval_tool.py` to evaluate submitted runs. We strongly recommend participants to evaluate their models on the `dev` set using this script before submitting their runs. `clariq_eval_tool.py` can be used to evaluate three subtasks:

* **Predicting clarification need:** A submitted run would be evaluated against the gold labels. Precision, recall, and F1-measure would be reported.
* **Question relevance:** Recall@k would be reported for this task. Among the top k results, we would measure the model's performance in terms of retrieving the maximum number of relevant questions to a topic. Relevant questions are listed in `train.tsv` and `dev.tsv` for each topic.
* **Document relevance:** As mentioned earlier, we also evaluate the quality of the predictions in terms of how they affect document retrieval performance. To evaluate this, we select the top-ranked question and measure how much it would affect the performance of document retrieval after being asked and answered. The performance would be reported in P@k, nDCG@k, and MRR@100.

Below, we see all the possible commands that one can pass to `clariq_eval_tool.py`:

	usage: clariq_eval_tool.py [-h] --eval_task EVAL_TASK
	                           [--experiment_type EXPERIMENT_TYPE]
	                           [--data_dir DATA_DIR] --run_file RUN_FILE
	                           [--out_file OUT_FILE] [--multi_turn]
And here is the full description if one passes `-h` argument:

	optional arguments:
	  -h, --help            show this help message and exit
	  --eval_task EVAL_TASK
	                        Defines the evaluation task. Possible values: clarific
	                        ation_need|document_relevance|question_relevance
	  --experiment_type EXPERIMENT_TYPE
	                        Defines the experiment type. The run file will be
	                        evaluated on the data that you specify here. Possible
	                        values: train|dev|test. Default value: dev
	  --data_dir DATA_DIR   Path to the data directory.
	  --run_file RUN_FILE   Path to the run file.
	  --out_file OUT_FILE   Path to the evaluation output json file.
	  --multi_turn          Determines if the results are on multi-turn
	                        conversations. Conversation is assumed to be single-
	                        turn if not specified.
As the description above is self-contained in most cases, we only add some additional remarks below:

* `--data_dir` should point to the directory where all the contents of the `data` directory are stored.
* `--run_file` is the full path to the run file (see notes on the format below).
* `--out_file` is the full path to the file where detailed evaluation results (per facet) will be stored. If not specified, the output will be stored. 

### Requirements
- pandas 
- sklearn

### Examples

Below, we give some examples of how to use the script and what to expect as output:

	python ./src/clariq_eval_tool.py --eval_task document_relevance \
	                                 --data_dir ./data/ \
	                                 --experiment_type dev \
	                                 --run_file ./sample_runs/dev_best_q \
	                                 --out_file ./sample_runs/dev_best_q.eval

Would produce the output below:

    NDCG1: 0.2942708333333333
    NDCG3: 0.25778462800726465
    NDCG5: 0.24697827353140434
    NDCG10: 0.22726519398403755
    NDCG20: 0.19582055938247206
    P1: 0.36875
    P3: 0.29583333333333334
    P5: 0.2675
    P10: 0.22625
    P20: 0.16125
    MRR100: 0.45411771321729144


An example on question relevance:

	python ./src/clariq_eval_tool.py --eval_task question_relevance \
	                                 --data_dir ./data/ \
	                                 --experiment_type dev \
	                                 --run_file ./sample_runs/dev_bm25 \
	                                 --out_file ./sample_runs/dev_bm25_question_relevance.eval

Would produce the output below:

	Recall5: 0.3245570421150917
	Recall10: 0.5638042646208281
	Recall20: 0.6674997108155003
	Recall30: 0.6912818698329535


## Run file format
Each run consists of two separate files: 

* Ranked list of questions for each topic;
* Predicted `clarification_need` label for each topic. 

Below we explain how each file should be formatted.

### Question ranking
This file is supposed to contain a ranked list of questions per topic. The number of questions per topic could be any number, but we evaluate only the top 30 questions. We follow the traditional TREC run format. Each line of the file should be formatted as follows:

    <topic_id> 0 <question_id> <ranking> <relevance_score> <run_id>

Each line represents a relevance prediction. `<relevance_score>` is the relevance score that a model predicts for a given `<topic_id>` and `<question_id>`. `<run_id>` is a string indicating the ID of the submitted run. `<ranking>` denotes the ranking of the `<question_id>` for `<topic_id>`. Practically, the ranking is computed by sorting the questions for each topic by their relevance scores.
Here are some example lines:

	170 0 Q00380 1 6.53252 sample_run
	170 0 Q02669 2 6.42323 sample_run
	170 0 Q03333 3 6.34980 sample_run
	171 0 Q03775 1 4.32344 sample_run
	171 0 Q00934 2 3.98838 sample_run
	171 0 Q01138 3 2.34534 sample_run

This run file will be used to evaluate both question relevance and document relevance. Sample runs can found in `./sample_runs/` directory.

### Clarification need
This file is supposed to contain the predicted `clarification_need` labels. Therefore, the file format is simply the `topic_id` and the predicted label. Sample lines can be found below:

    171 1
    170 3
    182 4

## Run Submission
Please send two files per run as described above to `clariq@convai.io`, indicating your team's name, as well as your run ID.  You'll also need to share your GitHub repository with us.

## Sample Baseline Code
A sample Colab Notebook of a simple baseline model can be found [here](https://colab.research.google.com/drive/1g_Sc9j5fYT1hiOxif6BVH5NHNt-icxtT?usp=sharing). The baseline model ranks the questions using a BM25 ranker.
The same baseline can also be found in the repo under `./src/clariq_baseline_bm25.ipynb`. It is a very simple baseline,
ranking the questions simply by their BM25 relevance score compared to the `original_request`.

## Questions
Please contact us via `clariq@convai.io` should you have any questions, comments, or concerns regarding the challenge.

## Acknowledgments
The challenge is organized as a joint effort by the University of Amsterdam, Microsoft, Google, University of Glasgow, and MIPT. We would like to thank Microsoft for their generous support of data annotation costs. 
We would also like to thank the [Webis Group](https://webis.de/) for giving us access to ChatNoir search API.
Thanks to the crowd workers for their invaluable help in annotating ClariQ.

## References

- <a name="ref1">[1]</a>: "Asking Clarifying Questions in Open-Domain Information-Seeking Conversations", M. Aliannejadi, H. Zamani, F. Crestani, and W. B. Croft, International ACM SIGIR Conference on Research and Development in Information Retrieval (SIGIR), Paris, France, 2019
- <a name="ref2">[2]</a>: "Elastic ChatNoir: Search Engine for the ClueWeb and the Common Crawl", J. Bevendorff, B. Stein,  M. Hagen, Martin Potthast, Advances in Information Retrieval. 40th European Conference on IR Research (ECIR 2018), Grenoble, France
