from glob import glob
from ortools.linear_solver import pywraplp
import numpy as np
import pandas as pd
import scipy.sparse as sp
from paper_reviewer_matcher import (
    preprocess, compute_affinity,
    create_lp_matrix, linprog,
    create_assignment
)   

def assign_articles_to_reviewers_diverse(article_df, reviewer_df ,people_df, coi_df):

    """
    Perform reviewer-assignment from dataframe of article, reviewer, and people

    Parameters
    ==========
    article_df: a dataframe that has columns `PaperID`, `Title`, `Abstract`, and `PersonIDList`
        where PersonIDList contains string of simicolon separated list of PersonID
    reviewer_df: a dataframe that has columns `PersonID` and `Abstract`
    people_df:  dataframe that has columns `PersonID`, `FullName`

    We assume `PersonID` is an integer

    Output
    ======
    article_assignment_df: an assigned reviewers dataframe, each row of article will have 
        list of reviewers in `ReviewerIDList` column and their name in reviewer_names
    """
    
    person_id_map = {}
    for index, person_id in enumerate(reviewer_df["PersonID"]):
        person_id_map[person_id] = index

    paper_id_map = {}
    for index, paper_id in enumerate(article_df["PaperID"]):
        paper_id_map[paper_id] = index

    solver = pywraplp.Solver.CreateSolver('SCIP')

    min_reviewers_per_paper=12
    max_reviewers_per_paper=13
    min_papers_per_reviewer=7
    max_papers_per_reviewer=9


    # calculate affinity matrix
    papers = list((article_df['Title'] + ' ' + article_df['Abstract']).map(preprocess))
    reviewers = list(reviewer_df['Abstract'].map(preprocess))
    affinities = compute_affinity(
        papers, reviewers,
        n_components=10, min_df=3, max_df=0.8,
        weighting='tfidf', projection='pca'
    )
    num_articles, num_reviewers = affinities.shape

    # who authored what paper
    authors = np.zeros((num_articles, num_reviewers))
    for paper in range(len(article_df)):
        for author in article_df.PersonIDList[paper].split(';'):
            if int(author) in person_id_map.keys():
                person_index = person_id_map[int(author)]
                authors[paper, person_index] = 1.0
    np.save('authors_mat.npy', authors)
    
    # create assignment constraints
    X = np.zeros((num_articles, num_reviewers), dtype="object")
    
    for article in range(num_articles):
        for reviewer in range(num_reviewers):
            X[article, reviewer] = solver.IntVar(0.0, 1.0, 'x_{}_{}'.format(article, paper))

    for article in range(num_articles):
        solver.Add(sum(X[article, :]) <= max_reviewers_per_paper)
        solver.Add(sum(X[article, :]) >= min_reviewers_per_paper)

    for reviewer in range(num_reviewers):
        solver.Add(sum(X[:, reviewer]) <= max_papers_per_reviewer)
        solver.Add(sum(X[:, reviewer]) >= min_papers_per_reviewer)

    # add conflict of interests with own paper
    coauthors_df = pd.DataFrame([[int(r.PaperID), int(co_author)]
                                for _, r in article_df.iterrows()
                                for co_author in r.PersonIDList.split(';')],
                                columns = ['PaperID', 'PersonID'])
    article_df['paper_id'] = list(range(len(article_df)))
    reviewer_df['person_id'] = list(range(len(reviewer_df)))
    coi_coauthors_df = coauthors_df.merge(article_df[['PaperID', 'paper_id']], 
                                on='PaperID').merge(reviewer_df[['PersonID', 'person_id']], 
                                on='PersonID')[['paper_id', 'person_id']]
    for i, j in zip(coi_coauthors_df.paper_id.tolist(), coi_coauthors_df.person_id.tolist()):
        solver.Add(X[i, j] <= 0.0)

    # add conflict of interests with other papers authored by coauthors
    coauthor_papers_coi = pd.DataFrame(columns=['person_id', 'PaperID'])
    coauthor_mat = np.matmul(authors.T, authors)
    for author in range(0,authors.shape[1]):
        coauthors = coauthor_mat[author,:]
        coauthors[author] = 0 # self
        coauthors_ids = np.where(coauthors > 0)[0]

        author_papers = reviewer_df.PaperIDs[author].split(';')
    
        # for each id in coauthors_ids, find which papers the person authored (from people file), and add them as constraints for author
        for c_id in coauthors_ids:
            coauthor_papers = reviewer_df.PaperIDs[c_id].split(';')
            for paper in coauthor_papers:
                if paper not in author_papers:
                     coauthor_papers_coi = coauthor_papers_coi.append({'person_id':author, 'PaperID':int(paper)}, ignore_index=True)
    coauthor_papers_coi = coauthor_papers_coi.merge(article_df[['PaperID', 'paper_id']], on='PaperID')
    for i, j in zip(coauthor_papers_coi.paper_id.tolist(), coauthor_papers_coi.person_id.tolist()):
        solver.Add(X[i, j] <= 0.0)

    # adding coi from coi_df (e.g. from same email domain)
    coi_df_indexed = coi_df.merge(article_df[['PaperID', 'paper_id']], 
                                on='PaperID').merge(reviewer_df[['PersonID', 'person_id']], 
                                on='PersonID')[['person_id','paper_id']]
    for i, j in zip(coi_df_indexed.paper_id.tolist(), coi_df_indexed.person_id.tolist()):
        solver.Add(X[i, j] <= 0.0)

    # add constraint to avoid coauthors reviewing the same paper
    Z = np.zeros(num_articles, dtype="object")
    for reviewed_article in range(num_articles):
        Z[reviewed_article] = solver.NumVar(1.0, num_reviewers, 'z_{}'.format(reviewed_article))
        for authored_article in range(num_articles):
            num_people = sum(X[reviewed_article, a] * authors[authored_article, a] for a in range(num_reviewers))
            solver.Add(Z[reviewed_article] >= num_people)

    # coef trades off importance of TF-IDF similarity of reviewer-papers (when coef = 0),
    # and importance of avoiding coauthors reviewing same paper (when coef is large)
    coef = 150.0
    solver.Maximize(sum(sum(X * affinities)) - coef * sum(Z) / num_articles) 

    result_status = solver.Solve()
    if result_status != pywraplp.Solver.OPTIMAL:
        print("The final solution might not converged")
    get_solution = np.vectorize(lambda x: x.SolutionValue())
    final_assignment = get_solution(X)

    print("Total coauthor penalty", sum(get_solution(Z)) / num_articles, coef)
    print("Total affinity", sum(sum(get_solution(X) * affinities)))
    print('Objective value =', solver.Objective().Value())

    reviewer_ids = list(reviewer_df.PersonID)
    reviewer_name_dict = {r['PersonID']: r['FullName'] for _, r in people_df.iterrows()} # map reviewer id to reviewer name
    assignments = []
    for i in range(len(final_assignment)):
        assignments.append([i, 
                            [reviewer_ids[b_] for b_ in np.nonzero(final_assignment[i])[0]], 
                            [reviewer_name_dict[reviewer_ids[b_]] for b_ in np.nonzero(final_assignment[i])[0]]])
    assignments_df = pd.DataFrame(assignments, columns=['paper_id', 'ReviewerIDList', 'reviewer_names'])
    assignments_df['ReviewerIDList'] = assignments_df.ReviewerIDList.map(lambda e: ';'.join(str(e_) for e_ in e))
    assignments_df['reviewer_names'] = assignments_df.reviewer_names.map(lambda x: ';'.join(x))
    article_assignment_df = article_df.merge(assignments_df, on='paper_id').drop('paper_id', axis=1)
    return article_assignment_df, final_assignment


if __name__ == '__main__':
    CCN_PATH = '/Users/mtoneva/Desktop/Service/CCN/Submissions2023/*.csv'
    article_path = '/Users/mtoneva/Desktop/Service/CCN/Submissions2023/CCN2023_Articles_2023-Apr-18-100533.csv'
    reviewer_path = '/Users/mtoneva/Desktop/Service/CCN/Submissions2023/reviewers.csv'
    people_path = '/Users/mtoneva/Desktop/Service/CCN/Submissions2023/CCN2023_People_2023-Apr-18-100533.csv'
    coi_path = '/Users/mtoneva/Desktop/Service/CCN/Submissions2023/CCN2023_COI_2023-Apr-18-100533.csv'
    
    # there is a problem when encoding lines in the given CSV so we have to use ISO-8859-1 instead
    article_df = pd.read_csv(article_path,sep=';') # has columns `PaperID`, `Title`, `Abstract`, `PersonIDList`
    reviewer_df = pd.read_csv(reviewer_path, encoding="ISO-8859-1") # has columns `PersonID`, `Abstract`
    people_df = pd.read_csv(people_path,sep=';', encoding="ISO-8859-1") # has columns `PersonID`, `FullName`
    coi_df = pd.read_csv(coi_path,sep=';')

    article_assignment_df, solution = assign_articles_to_reviewers_diverse(article_df, reviewer_df, people_df, coi_df)
    np.save('article_assignment_submissions2023.npy', solution)
    article_assignment_df.to_csv('/Users/mtoneva/Desktop/Service/CCN/Submissions2023/article_assignment_submissions2023.csv', index=False)
