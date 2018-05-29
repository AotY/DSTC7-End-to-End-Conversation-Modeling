# DSTC7: End-to-End Conversation Modeling
## Moving beyond Chitchat
This [DST7](http://workshop.colips.org/dstc7/) track proposes an end-to-end conversational modeling task, where the goal is to generate conversational responses that go beyond chitchat, by injecting informational responses that are grounded in external knowledge (e.g.,Foursquare, or possibly also Wikipedia, Goodreads, or TripAdvisor). There is no specific or predefined goal (e.g., booking a flight, or reserving a table at a restaurant), so this task does not constitute what is commonly called either goal-oriented, task-oriented, or task-completion dialog, but target human-human dialogs where the underlying goal is often ill-defined or not known in advance, even at work and other productive environments (e.g.,brainstorming meetings).

A full description of the task and the dataset is available [here](http://workshop.colips.org/dstc7/proposals/DSTC7-MSR_end2end.pdf)

## Task, Data and Evaluate
We extend the knowledge-grounded setting, with each system input consisting of two parts: 
* Conversational input, from Reddit
* Contextually-relevant “facts”, from WWW

The data extraction scripts are available [here](https://github.com/DSTC-MSR/DSTC7-End-to-End-Conversation-Modeling/tree/master/data_extraction). 
We will evaluate response quality using both automatic and human evaluation on two criteria .
* Appropriateness
* Informativeness & Utility

A baseline model is provided [here](https://github.com/DSTC-MSR/DSTC7-End-to-End-Conversation-Modeling/tree/master/baseline). 

## Contact Information
You can get the latest updates and participate in discussions on DSTC mailing list. To join the mailing list, send an email to listserv@lists.research.microsoft.com with "subscribe DSTC" in the body of the message (without the quotes). 
