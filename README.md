# ccn-reviewer-assignment
Algorithm used for assigning reviewers for the Cognitive Computational Neuroscience conference 2022 and 2023

This algorithm was based on ideas and code from an earlier version, which you can read about here: [https://github.com/titipata/paper-reviewer-matcher/tree/master](https://github.com/titipata/paper-reviewer-matcher/tree/master)

The main updates from the earlier version are added constraints to the reviewer-paper matching optimization related to:
1. Conflicts of interest related to papers authored by a specific reviewer and all of the papers authors by the reviewer's co-authors
2. Diversification of the paper assignment, such that a paper is less likely to be reviewed by co-authors

The assignments can be obtained using the following command:

```sh
python ccn_paper_reviewer_matching_2023.py
```

Note that the python file above should be edited to reflect the correct paths to the needed files, and the correct maximum and minumum desired numbers of assignments per submission and per reviewer.
Example input files are provided in the same repository. Note that for significantly larger files, the algorithm takes a long time to run (possibly 4-5 hours). 
