# ClariQ

## Introduction

The main aim of the conversational systems is to return
an appropriate answer in response to the user requests. However, some user
requests might be ambiguous. In Information Retrieval (IR) settings such a situation is handled mainly
through the diversification of search result page. It is however much more challenging in dialogue settings. 

We release the ClariQ dataset [[3](#ref3), [4](#ref4)], aiming to study the following situation for dialogue settings:

* a user is asking an ambiguous question (where ambiguous question is a
question to which one can return > 1 possible answers);
* the system must identify that the question is ambiguous, and, instead of
trying to answer it directly, ask a good clarifying question.

The main research questions we aim to answer as part of the challenge are
the following:

* RQ1: When to ask clarifying questions during dialogues?
* RQ2: How to generate the clarifying questions?

## ConvAI3 Data Challenge

ClariQ was collected as part of the ConvAI3 (http://convai.io) challenge which was co-organized with the SCAI workshop (https://scai-workshop.github.io/2020/).
The challenge ran in two stages. At Stage 1 (described below)
participants were provided with a static dataset consisting mainly of an initial user
request, clarifying question and user answer, which is suitable for initial training,
validating and testing. At Stage 2, we brought a human
in the loop. Namely, the top 3 systems, resulted from Stage 1, were invited to develop systems that were exposed
to human annotators.

### Stage 1: initial dataset

Taking inspiration from [Qulac](https://github.com/aliannejadi/qulac) [[1]](#ref1) dataset,
 we have crowdsourced a new dataset to study clarifying questions that is suitable for conversational settings. 
Namely, the collected dataset consists of:

* **User Request:** an initial user request in the conversational form, e.g.,
"What is Fickle Creek Farm?", with a label reflects if is needed
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
The second stage of the ClariQ data challenge enables the top-performing teams of the first stage to evaluate their models with the help of human evaluators. To do so, we ask the teams to generate their responses in a given conversation and pass the results to human evaluators. We instruct the human evaluators to read and understand the context of the conversation and write a response to the system. The evaluator assumes that they are part of the conversation. We evaluate the performance of a system in two respects: (i) How much the conversation can help a user find the information they are looking for and (ii) How natural and realistic does the conversation appear to a human evaluator. 


## ClariQ Dataset
We have extended the [Qulac](https://github.com/aliannejadi/qulac) [[1]](#ref1) dataset and base the competition mostly
 on the training data that [Qulac](https://github.com/aliannejadi/qulac) provides. 
 In addition, we have added some new topics, questions, and answers in the training set. 
 The test set is completely unseen and newly collected. 
 Like Qulac, ClariQ consists of single-turn conversations (`initial_request`, followed by clarifying `question` and `answer`).
 In addition, it comes with synthetic multi-turn conversations (up to three turns). ClariQ features approximately 18K single-turn conversations, as well as 1.8 million multi-turn conversations. 
Below, we provide a short summary of the data characteristics, for the training set:

### ClariQ Train
Feature  			| Value
--------------------------------| -----
\# train (dev) topics		| 187 (50)
\# faceted topics 		| 141
\# ambiguous topics 		| 57
\# single topics		| 39
\# facets 			| 891
\# total questions              | 3,929
\# single-turn conversations    | 11,489
\# multi-turn conversations     | ~ 1 million 
\# documents                    | ~ 2 million

Below, we provide a brief overview of the structure of the data.

## Files
Below we list the files in the repository:

* `./data/train.tsv` and `./data/dev.tsv` are TSV files consisting of topics (queries), facets, clarifying questions, user's answers, and labels for how much clarification is needed (`clarification needs`).
* `./data/test.tsv` is a TSV file consisting of test topic ID's, as well as queries (text).
* `./data/test_with_labels.tsv` is a TSV file consiting of test topic ID's with the labels. It can be used with the evaluation script.
* `./data/multi_turn_human_generated_data.tsv` is a TSV file containing the human-generated multi turn conversations which is the result of of the human-in-the-loop process.
* `./data/question_bank.tsv` is a TSV file containing all the questions in the collection, as well as their ID's. Participants' models should select questions from this file.
* `./data/top10k_docs_dict.pkl.tar.gz` is a `dict` containing the top 10,000 document ID's retrieved from ClueWeb09 and ClueWeb12 collections for each topic. This may be used by the participants who wish to leverage documents content in their models. 
* `./data/single_turn_train_eval.pkl` is a `dict` containing the performance of each topic after asking a question and getting the answer. The evaluation tool that we provide uses this file to evaluate the selected questions.
* `./data/multi_turn_train_eval.pkl.tar.gz.**` and `./data/multi_turn_dev_eval.pkl.tar.gz` are `dict`s that contain the performance of each conversation after asking a question from the `question_bank` and getting the answer from the user. The evaluation tool that we provide uses this file to evaluate the selected questions. Notice that these `dict`s are built based on the synthetic multi-turn conversations.
* `./data/dev_synthetic.pkl.tar.gz` and `./data/train_synthetic.pkl.tar.gz` are two compressed `pickle` files that contain `dict`s of synthetic multi-turn conversations. We have generated these conversations following the method explained in [[1]](#ref1). 
* `./src/clariq_eval_tool.py` is a python script to evaluate the runs. The participants may use this tool to evaluate their models on the `dev` set. We would use the same tool to evaluate the submitted runs on the `test` set.
* `./sample_runs/` contains some sample runs and baselines. Among them, we have included the two oracle models `BestQuestion` and `WorstQuestion`, as well as `NoQuestion`, the model choosing no question. Participants may check these files as sample run files. Also, they could test the evaluation tool using these files.

## File Format

### `train.tsv`, `dev.tsv`:

`train.tsv` and `dev.tsv` have the same format. They contain the topics, facets, questions, answers, and clarification need labels. These are considered to be the main files, containing the labels of the training set. Note that the `clarification needs` labels are already explicitly included in the files. Regarding the `question relevance` labels for each topic, these labels can be extracted inderictly: each row only contains the questions that are considered to be relevant to a topic. Therefore, any other question is deemed irrelevant while computing `Recall@k`. 
In the `train.tsv` and `dev.tsv` files, you will find these fields:


* `topic_id`: the ID of the topic (`initial_request`).
* `initial_request`: the query (text) that initiates the conversation.
* `topic_desc`: a full description of the topic as it appears in the TREC Web Track data.
* `clarification_need`: a label from 1 to 4, indicating how much it is needed to clarify a topic. If an `initial_request` is self-contained and would not need any clarification, the label would be 1. While if a `initial_request` is absolutely ambiguous, making it impossible for a search engine to guess the user's right intent before clarification, the label would be 4. Labels 2 and 3 represent other levels of clarification need, where clarification is still needed but not as much as label 4.
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

### `dev_synthetic.pkl.tar.gz` and `train_synthetic.pkl.tar.gz`:
These files contain `dict`s of synthetically built multi-turn conversations (up to three turns). We follow the same approach explained in [[1]](#ref1) to generate these conversations. The format of these files is very similar to the format of the test file that will be fed to the system (see below), except for having the current question and answer of a conversation. Each record in this `dict` is identified by its topic, facet, conversation context, question, and answer. Below we see the `dict` structure:

	{<record_id>: {'topic_id': <int>,
	  'facet_id': <str>,
	  'initial_request': <str>,
	  'question': <str>,
	  'answer': <str>,
	  'conversation_context': [{'question': <str>,
	   'answer': <str>},
	  {'question': <str>,
	   'answer': <str>}],
	  'context_id': <int>},
	  ...
	  }
	  
where
 - `<record_id>` is an `int` indicating the ID of the current conversation record. While in the `dev` set there exists multiple `<record_id>` values per `<context_id>`, in the `test` file there would be only one. We include current questions and answers from the synthetic multi-turn data in the `synthetic_dev.pkl` file for training purposes.
 - `'topic_id'`, `'facet_id'`, and `'initial_request'` indicate the topic, facet, and initial request of the current conversation, according to the single turn dataset.
 - `'question'`: current clarifying question that is being posed to the user.
 - `'answer'`: user's answer to the clarifying question.
 - `'conversation_context'` identifies the context of the current conversation. A context consists of previous turns in a conversation. As we see, it is a list of `'question'` and `'answer'` items. This list tells us which questions have been asked in the conversation so far, and what has been the answer to them.
 - `'context_id'` is the ID of the conversation context. Basically, participants should predict the next utternace for each `context_id`.
 
 Some example records can be seen below:
 
	{2287: {'topic_id': 8,
	  'facet_id': 'F0968',
	  'initial_request': 'I want to know about appraisals.',
	  'question': 'are you looking for a type of appraiser',
	  'answer': 'im looking for nearby companies that do home appraisals',
	  'conversation_context': [],
	  'context_id': 968},
	 2288: {'topic_id': 8,
	  'facet_id': 'F0969',
	  'initial_request': 'I want to know about appraisals.',
	  'question': 'are you looking for a type of appraiser',
	  'answer': 'yes jewelry',
	  'conversation_context': [],
	  'context_id': 969},
	 1570812: {'topic_id': 293,
	 'facet_id': 'F0729',
	 'initial_request': 'Tell me about the educational advantages of social networking sites.',
	 'question': 'which social networking sites would you like information on',
	 'answer': 'i don have a specific one in mind just overall educational benefits to social media sites',
	 'conversation_context': [{'question': 'what level of schooling are you interested in gaining the advantages to social networking sites',
	   'answer': 'all levels'},
	  {'question': 'what type of educational advantages are you seeking from social networking',
	   'answer': 'i just want to know if there are any'}],
	 'context_id': 976573}
	

### `single_turn_train_eval.pkl` and `multi_turn_****_eval.pkl.tar.gz`:
These files are `dict`s of pre-computed document relevance results after asking each question.  The document relevance performance is calculated as follows:

* For a context, the selected question and its corresponding answer are added to the document retrieval system.
* The document retrieval model [[1]](#ref1) , then re-ranks the documents with the given question and answer.
* The performance of the newly-ranked document is then computed as follows. For every given facet, the effect of asking the question can be determined using the pre-computed `dict`. Below we see the structure of the `dict`:
	
		{ <evaluation_metric>: 
			[ 
			  <context_id>: 
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
	

As we see, one has first to identify the `evaluation_metric` they are interested in, followed by a `context_id` and `question_id`. Notice that here we report the retrieval performance for both with and without considering the answer to the question. Furthermore, we also include two other values, namely, `MAX` and `MIN`. These refer to the maximum and minimum performance that the retrieval model achieves by asking the "best" and "worst" questions among the candidate questions. Below we see a sample of the data:

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

**Note**: The `context_id` in the multi-turn dictionaries is an `int`. The multi-turn `dict`s also contain single-turn dialogs. For those, the `context_id` equals the `facet_id` after removing the initial `F` and casting to `int`. On the other hand, for the single-turn `dict`, the `context_id` is actually `facet_id`.

### `top10k_docs_dict.pkl.tar.gz`
`top10k_docs_dict.pkl.tar.gz` is a `dict` consisting of a `list` of document ID's for a given `topic_id`. In case one plans to use the contents of a document in their model, and does not have access to ClueWeb09 or ClueWeb12 data collections, this `dict` is useful for having the list of top 10,000 documents as an initial ranking. The participants can use this list for two purposes:

* To get access to the full text of the documents listed in this `dict`. For this, we suggest using the [ChatNoir](http://chatnoir.eu)'s API [[2]](#ref2) . Upon request, we provide the participants with an API key, using which they can get access by providing a document's ID. Sample codes will be added soon.
**Note**: The ClueWeb document ID should be translated into a UUID used by [ChatNoir](http://chatnoir.eu). ChatNoir provides a simple JavaScript for this purpose: [https://github.com/chatnoir-eu/webis-uuid](https://github.com/chatnoir-eu/webis-uuid).
More information on how to use `ChatNoir`'s API: [https://www.chatnoir.eu/doc/api/#retrieving-full-documents](https://www.chatnoir.eu/doc/api/#retrieving-full-documents)
* Get a pre-build index to re-run document retrieval. We advise the participants to contact us if they require access to pre-build index files to re-run document retrieval. We recommend viewing the `QL.py` in [Qulac](https://github.com/aliannejadi/qulac)'s repository for more information on how the pre-build index files could be used. 

### `train.qrel` & `dev.qrel`
These files contain the relevance assessments of ClueWeb09 and ClueWeb12 collections for every facet in the train and dev sets, respectively.
They follow the conventional TREC format for qrel files, that is:

    <facet_id> 0 <document_id> <relevance_score>
    
Some sample lines of `train.qrel` file is shown below:

    F0001 0 clueweb09-en0038-74-08250 1
    F0001 0 clueweb09-enwp01-17-11113 1
    F0002 0 clueweb09-en0001-02-21241 1
    F0002 0 clueweb09-en0006-52-11056 1


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

    NDCG1: 0.3541666666666667
    NDCG3: 0.33374776946106466
    NDCG5: 0.3064048059484046
    NDCG10: 0.26443649709165346
    NDCG20: 0.22765633337753358
    P1: 0.41875
    P3: 0.37916666666666665
    P5: 0.32875
    P10: 0.256875
    P20: 0.186875
    MRR100: 0.4882460524507918


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
To evaluate a run using the evaluation script, each file should be formatted as follows. The following files can be evaluated using the script:

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

## Multi-turn Input/Output
Each team in the second stage must submit a system that accepts the conversation in the following format, and produces output as described.
### Input format
	{<record_id>: {'topic_id': <int>,
	  'facet_id': <str>,
	  'initial_request': <str>,
	  'conversation_context': [{'question': <str>,
	   'answer': <str>},
	  {'question': <str>,
	   'answer': <str>}],
	  'context_id': <int>},
	  ...
	  }
	  
where
 - `<record_id>` is an `int` indicating the ID of the current conversation record. While in the `dev` set there exists multiple `<record_id>` values per `<context_id>`, in the `test` file there would be only one. We include current questions and answers from the synthetic multi-turn data in the `synthetic_dev.pkl` file for training purposes.
 - `'topic_id'`, `'facet_id'`, and `'initial_request'` indicate the topic, facet, and initial request of the current conversation, according to the single turn dataset.
 - `'conversation_context'` identifies the context of the current conversation. A context consists of previous turns in a conversation. As we see, it is a list of `'question'` and `'answer'` items. This list tells us which questions have been asked in the conversation so far, and what has been the answer to them. For the `train` and `dev` sets, these `str` values can be mapped to the `question_bank` question values. Here, we do not refer to questions by ID's, as the second stage aims to evaluate **machine-generated** questions as well.
 - `'context_id'` is the ID of the conversation context. Basically, participants should predict the next utternace for each `context_id`. Therefore, even in cases of `train` and `dev` sets where multiple records exists for single `context_id`, one prediction must be provided. Some example data can be found below:
 
 		{2287: {'topic_id': 8,
		  'facet_id': 'F0968',
		  'initial_request': 'I want to know about appraisals.',
		  'conversation_context': [],
		  'context_id': 968},
		 2288: {'topic_id': 8,
		  'facet_id': 'F0969',
		  'initial_request': 'I want to know about appraisals.',
		  'conversation_context': [],
		  'context_id': 969},
		 1570812: {'topic_id': 293,
		 'facet_id': 'F0729',
		 'initial_request': 'Tell me about the educational advantages of social networking sites.',
		 'conversation_context': [{'question': 'what level of schooling are you interested in gaining the advantages to social networking sites',
		   'answer': 'all levels'},
		  {'question': 'what type of educational advantages are you seeking from social networking',
		   'answer': 'i just want to know if there are any'}],
		 'context_id': 976573}


### Output format
The system output should be submitted in a single file per set (dev and test) in the following format:

	<context_id> 0 “<question_text>” <ranking> <relevance_score> <run_id>

Participants may submit more than one response per `context_id`, however, we only evaluate the first response in the ranked list per `context_id`.
`<question_text>` must be quoted. Empty string (`""`) value for `<question_text>` indicates that a system asks no question for a given context (i.e., `Q00001`). This could be the case where the system predicts that no further improvement can be achieved by asking clarifying questions, or no further clarification is required. We mark empty question as the end of a conversation, and count the number of turns based on that.

Notice that `<question_text>` must be an `str` of the question. As participants are allowed to select a question from the `question_bank` or **generate** clarifying questions, we only take full text strings as input. In case a question is selected from the `question_bank`, simply quote the text of the question. An example generated output can be found below:

	784 0 "are you looking for reviews related to the pampered chef" 0 13 bestq_multi_turn
	785 0 "" 0 13 bestq_multi_turn
	813 0 "are you interested in a current map of the united states" 0 17 bestq_multi_turn
	820 0 "are you looking for a specific type of solar panels" 0 10 bestq_multi_turn
	841 0 "" 0 15 bestq_multi_turn
	

## Baselines
### BM25 Ranker
 - **Single turn**:
A sample Colab Notebook of a simple baseline model can be found [here](https://colab.research.google.com/drive/1g_Sc9j5fYT1hiOxif6BVH5NHNt-icxtT?usp=sharing). The baseline model ranks the questions using a BM25 ranker.
The same baseline can also be found in the repo under `./src/clariq_baseline_bm25.ipynb`. It is a very simple baseline,
ranking the questions simply by their BM25 relevance score compared to the `original_request`.
 - **Multi turn**:
A simple BM25 baseline for multi-turn question selection can be found in the repo under `./src/clariq_baseline_bm25_multi_turn.ipynb`.

### BERT-based Ranker
We have trained a BERT-based model for the `question_relevance` task. The model fine-tunes BERT for retrieve relevant questions to a given topic. The model is tested on two different evaluation setups, i.e., question reranking and question ranking. The reranking model takes the top 30 predictions of BM25 and reranks them, while the full ranking model ranks all the questions available in the question bank. The results of the two models can be found in the [leaderboard](http://convai.io/). Special thanks to [Gustavo Penha](https://guzpenha.github.io/guzblog/), who kindly developed the models based on the [Transformer Rankers](https://guzpenha.github.io/transformer_rankers/) library, and shared the code in a [Google Colab Notebook](https://colab.research.google.com/drive/1RHHbh5KQY-QDA7kV7wyHFJ7B_w5RRHzP?usp=sharing).


## Citing

	@inproceedings{aliannejadi2021building,
	    title={Building and Evaluating Open-Domain Dialogue Corpora with Clarifying Questions},
	    author={Mohammad Aliannejadi and Julia Kiseleva and Aleksandr Chuklin and Jeff Dalton and Mikhail Burtsev},
	    year={2021},
	    booktitle={{EMNLP}}	 
	}

## Acknowledgments
The challenge is organized as a joint effort by the University of Amsterdam, Microsoft, Google, University of Glasgow, and MIPT. We would like to thank Microsoft for their generous support of data annotation costs. 
We would also like to thank the [Webis Group](https://webis.de/) for giving us access to ChatNoir search API.
We appreciate [Gustavo Penha](https://guzpenha.github.io/guzblog/)'s efforts in development of BERT-based baselines for the task.
Thanks to the crowd workers for their invaluable help in annotating ClariQ.

## References

- <a name="ref1">[1]</a>: "Asking Clarifying Questions in Open-Domain Information-Seeking Conversations", M. Aliannejadi, H. Zamani, F. Crestani, and W. B. Croft, International ACM SIGIR Conference on Research and Development in Information Retrieval (SIGIR), Paris, France, 2019
- <a name="ref2">[2]</a>: "Elastic ChatNoir: Search Engine for the ClueWeb and the Common Crawl", J. Bevendorff, B. Stein,  M. Hagen, Martin Potthast, Advances in Information Retrieval. 40th European Conference on IR Research (ECIR 2018), Grenoble, France
- <a name="ref3">[3]</a>: "ConvAI3: Generating Clarifying Questions for Open-Domain Dialogue Systems (ClariQ)", M. Aliannejadi, J. Kiseleva, A. Chuklin, J. Dalton, M. Burtsev, arXiv, 2009.11352, 2020
- <a name="ref4">[4]</a>: "Building and Evaluating Open-Domain Dialogue Corpora with Clarifying Questions", M. Aliannejadi, J. Kiseleva, A. Chuklin, J. Dalton, M. Burtsev, EMNLP 2021

