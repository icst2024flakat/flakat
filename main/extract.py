import pandas as pd
import git
import shutil
import os
import pprint
import javalang
import argparse
from tqdm import tqdm
from parsing import Project
from parsing import UnitTest
from extract_utils import read_test_file_content
from extract_utils import extract_raw_test_methods
from extract_utils import extract_method_body
from extract_utils import cols


pp = pprint.PrettyPrinter(indent=4)

# output: 
#   csv format
#   cols: id, project url, raw, category, is_flaky

# global variables
projects = {} # key: url, value: Project
repo_dir = './repos'
output_file = 'data/extracted-all-projects.csv'

def run_extract():
    missing_proj = set()
    missing_commits = {}
    missing_files = set()
    missing_tests = 0
    # delete output file
    if os.path.exists(output_file):
        os.remove(output_file)
    # debug repo limit
    i = 0
    limit = float('inf')
    counter = 0
    invalid_flaky = 0
    # load input csv file
    df = pd.read_csv('data/input/extracted-all-projects.csv')
    print('raw data dimensions={}'.format(df.shape))
    for index, row in df.iterrows():
        project_url = row['Project URL']
        if project_url not in projects:
            i += 1
            if i > limit:
                break
            # create new project
            projects[project_url] = Project(
                url=project_url
            )
            
        project = projects[project_url]
        sha = row['SHA Detected']
        module = row['Module Path']
        full_name = row['Fully-Qualified Test Name (packageName.ClassName.methodName)']
        category = row['Category']
        # create new flaky test
        flaky_test = UnitTest(
                project_url=project_url,
                sha=sha,
                module=module,
                full_name=full_name,
                category=category,
                is_flaky=True,
            )
        project.add_test(flaky_test)
    # sanity check
    for url in projects:
        for sha in projects[url].flaky_commits:
            for t in projects[url].flaky_commits[sha]:
                if projects[url].flaky_commits[sha][t].is_flaky:
                    counter += 1
    print('# flaky tests={}'.format(counter))
    counter = 0

    for url in projects:
       counter += len(projects[url].tests)
    print('# tests={}'.format(counter))
    counter = 0

    i = 0
    # go through each project
    for url, project in projects.items():
        print('\nproject: {}'.format(project.name))
        local_project_dir = '{}/{}'.format(repo_dir, project.name)
        if (os.path.isdir(local_project_dir)):
            # use local dir
            repo = git.Repo(local_project_dir)
        else:
            # clone from remote
            try:
                repo = git.Repo.clone_from(project.url, local_project_dir)
            except git.exc.GitCommandError as e:
                print('pull error: {}'.format(project.name))
                missing_proj.add(project)
                missing_tests += len(project.tests)
                continue
        # go through each commit with flaky test(s)
        for sha in project.flaky_commits:
            print('checkout commit: {}'.format(sha))
            try:
                repo.git.checkout(sha)
            except git.exc.GitCommandError as e:
                print('Commit not found: {}, project: {}'.format(sha, project.name))
                if project not in missing_commits:
                    missing_commits[project.name] = set()
                missing_commits[project.name].add(sha)
                missing_tests += len(project.flaky_commits[sha])
                continue
            raw_test_method_contents = {} # key: test_file_path, value: dict{key: test method name, value: raw test method}
            
            # go through each flaky test case for commit
            for _id in project.flaky_commits[sha]:
                flaky_test_obj = project.flaky_commits[sha][_id]
                test_file_path = flaky_test_obj.get_test_file_path(local_project_dir)

                if test_file_path not in raw_test_method_contents:
                    # extract raw method contents from file
                    file_content = read_test_file_content(test_file_path=test_file_path)
                    if file_content == None:
                        missing_files.add(test_file_path)
                        missing_tests += 1
                        continue
                    # store raw method contents in cache
                    raw_test_method_contents[test_file_path] = extract_raw_test_methods(file_content)
                
                raw_tests = raw_test_method_contents[test_file_path]
                if flaky_test_obj.name not in raw_tests:
                    print('{} missing test: {}'.format(test_file_path, flaky_test_obj.name))
                    missing_tests += 1

                for method_name in raw_tests:
                    if method_name == flaky_test_obj.name:
                        ut = flaky_test_obj
                    else:
                        ut = UnitTest(
                            project_url=project.url,
                            sha=sha,
                            module=flaky_test_obj.module,
                            full_name=UnitTest.build_full_name(test_file_path, method_name),
                            category=None,
                            is_flaky=False
                        )
                        project.add_test(ut)
                    ut.set_raw(raw_tests[method_name])

        # clean up
        # shutil.rmtree(local_project_dir)
        
        output_list = [] 
        for _id in project.tests:
            test = project.tests[_id]
            if (test.raw == None or len(test.raw) == 0):
                # print('{} has invalid raw: {}'.format(test.full_name, test.raw))
                if test.is_flaky:
                    invalid_flaky += 1
                continue
            elif test.is_flaky:
                counter += 1
            output_list.append(test.to_row())
        out_df = pd.DataFrame(output_list, columns=cols)
        try:
            out_df.to_csv(output_file, 
                mode='a', index=False, header=(i==0), encoding='utf-8-sig')
        except UnicodeEncodeError as e:
            print(e)
            # print("error encoding: {}".format(output_list))
        # debug repo limit
        i += 1
        if i >= limit:
            break
    print('accounted flaky tests={}'.format(counter))
    print('invalid flaky tests={}'.format(invalid_flaky))
    return missing_proj, missing_commits, missing_files, missing_tests


if __name__ == '__main__':
    missing_proj, missing_commits, missing_files, missing_tests = run_extract()
    # counter = 0
    # for p in projects:
    #     for t in projects[p].tests:
    #         if projects[p].tests[t].is_flaky:
    #             counter+=1
    # print('is flaky: {}'.format(counter))
    df = pd.read_csv(output_file)
    print("output shape: {}".format(df.shape))
    is_flaky = df['is_flaky'] == True
    print("is flaky: {}".format(df[is_flaky].shape))
    print("missing proj: " + str(len(missing_proj)))
    print("missing commits: " + str(sum([len(missing_commits[k]) for k in missing_commits])))
    print("missing files: " + str(len(missing_files)))
    print("missing tests: " + str(missing_tests))