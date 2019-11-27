# Active-learning-through-label-error-statistical-methods

Abstract: Clustering-based active learning splits data into a number of blocks and queries the labels of the most critical instances. An active learner must decide how to choose these critical instances and how to split the blocks. In this paper, we present theoretical and practical statistical methods for analyzing the relationship between the label error and the neighbor radius, and design new split and selection strategies to handle these two issues. First, we define statistical functions for the label error based on a single instance and instance pairs. Second, we build practical statistical models, calculate empirical label errors, and guide the block splitting process. Third, using these practical models, we develop a center-and-edge instance selection strategy for choosing critical instances. Fourth, we design a new algorithm called active learning through label error statistical methods (ALSE). Learning experiments were performed with 20 datasets from various domains. The results of significance tests verify the effectiveness of ALSE and its superiority over state-of-the-art active learning algorithms.

Highlights：
We define two label error statistics functions and build clustering-based practical statistical models to guide block splitting.
We propose a center-and-edge instance selection strategy to choose critical instances.
We design an algorithm called active learning through label error statistical methods (ALSE).
Results of significance test verify the superiority of ALSE to state-of-the-art algorithms.

@article{Wang2019Active,

author = "Min Wang and Ke Fu and Fan Min and Xiu-Yi Jia",

title = "Active learning through label error statistical methods",

year = "2019",

journal = "Knowledge-Based Systems",

pages = "105140",

issn = "0950-7051",

doi = "https://doi.org/10.1016/j.knosys.2019.105140"

}

python 3.6
