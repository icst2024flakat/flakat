import javalang
import os

class UnitTest:
    """
    Class for holding data related to unit tests in Java
    """
    def __init__(self, full_name, is_flaky, 
                raw=None, project_name=None,  
                project_url=None, sha=None, module=None,
                category=None, javalang_tree=None,
                zss_tree=None, apted_tree=None,
                invoked_methods=None,
                dft_list = None,
                num_nodes=0,):
        self.project_url = project_url
        if project_url:
            self.project_name = project_url.split('/')[-1]
        else:
            self.project_name = project_name
        self.sha = sha
        self.module = module
        self.full_name = full_name
        if ' ' in self.full_name:
            self.full_name = self.full_name.split(' ')[0]
        self.category = category
        self.name = full_name.split('.')[-1]
        if ' ' in self.name:
            self.name = self.name.split(' ')[0]
        self.category = category
        self.is_flaky = is_flaky
        self.set_raw(raw)

        self.javalang_tree = javalang_tree
        self.zss_tree = zss_tree
        self.apted_tree = apted_tree
        if invoked_methods:
            self.invoked_methods = invoked_methods
            self.invoked_methods_set = set(invoked_methods)
        else:
            self.invoked_methods = []
            self.invoked_methods_set = set()
        self.num_nodes = num_nodes
        self.dft_list = dft_list

    def __lt__(self, other):
        return self.num_nodes < other.num_nodes

    @property
    def _id(self):
        """
        Returns unique id for this unit test. Combination of first 5 characters of 
        commit id + ":" + full name of unit test
        """
        # if self.sha:
        #     return self.project_name+':'+self.full_name+':'+self.sha[:4]
        # else:
        return self.project_name+':'+self.full_name+':'+str(self.is_flaky)

    def get_test_file_path(self, local_proj_path):
        """
        Returns the file path for this test based on full test name
        """
        try:
            file_path = self.full_name.split('.')[:-1]
            f_name = '/'.join(file_path) + '.java'
            prefix_list = [
                'test',
                'test/src',
                'src/test',
                'src/test/java',
                # 'src/test/scala',
                'test/scala/com',
                self.module+'/java',
                self.module+'/test',
                self.module+'/test/src',
                self.module+'/src/test',
                self.module+'/src/test/java',
                # self.module+'/src/test/scala',
                'modules/'+self.module+'/src/test/java',
                'cukes-samples/'+self.module+'/src/test/java',
            ]
            for prefix in prefix_list:
                _path = self.validate_path(local_proj_path, prefix, f_name)
                if _path:
                    return _path
            return self.full_name
        except Exception as e:
            return "invalid"

    def validate_path(self, local_proj_path, prefix, path):
        full_path = '{}/{}/{}'.format(local_proj_path, prefix, path)
        if os.path.exists(full_path):
            return full_path
        return None

    def set_raw(self, raw):
        """
        Sets the raw method content for this unit test if its syntax valid
        Returns True if successfully updated, False otherwise
        """
        self.raw = raw
        if raw == None:
            return False
        # replace any non-printable characters from raw string
        import string
        printable = set(string.printable)
        self.raw = ''.join([x if x in printable else '?' for x in self.raw])
        self.raw = self.raw.replace('\n', ' ')
        try:
            parsable_str = UnitTest.build_parseable_string(self)
            javalang.parse.parse(parsable_str)
            return True
        except javalang.parser.JavaSyntaxError as e:
            print('\ninvalid raw {}'.format(self.full_name))
            print(UnitTest.build_parseable_string(self))
            self.raw = None
            return False
        # except:
        #     raise
    @staticmethod
    def build_parseable(source_code):
        """
        Returns a Java class definition (a dummy class) wrapping the method definition  
        of the unit test which is parsable by javalang.
        """
        return 'public class MyTest {{ {} }}'.format(source_code)

    @staticmethod
    def build_parseable_string(test_instance):
        """
        Returns a Java class definition (a dummy class) wrapping the method definition  
        of the unit test which is parsable by javalang.
        """
        return 'public class MyTest {{ {} }}'.format(test_instance.raw)
    
    @staticmethod
    def build_full_name(file_path, name):
        """
        Returns the fully qualified test name based on file path and method
        name
        """
        file_path = file_path.replace('.java', '.'+name)
        if './repos/' in file_path:
            file_path = file_path.replace('./repos/', '')
        return file_path.replace('/', '.')
    
    def to_row(self):
        """
        Returns tuple with object attributes in following order:
        (id, project url, full test name, sha, raw, category, is_flaky)
        """
        return (
            self._id,
            self.project_url,
            self.raw,
            self.category,
            self.is_flaky
        )

class Project:
    """
    Class for holding data related to projects
    """
    def __init__(self, url=None, name=None):
        if url:
            self.name = url.split('/')[-1]
            self.url = url
        elif name:
            self.name = name
        self.flaky_commits = {} # key: SHA id, value: dict{key: test id, value: UnitTest}
        self.tests = {} # key: test id, value: UnitTest

    def add_flaky_test(self, test):
        """
        Adds a flaky unit test to this project based on commit id 
        and unit test id
        """
        if test.is_flaky == False:
            return

        if test.sha not in self.flaky_commits:
            self.flaky_commits[test.sha] = {}
        commit = self.flaky_commits[test.sha]
        if test._id not in commit:
            commit[test._id] = test    
    
    def add_test(self, test):
        """
        Adds a unit test to this project based on unit test id.
        Also calls add_flaky_test if test is flaky
        """
        if test._id not in self.tests:
            self.tests[test._id] = test
        # else:
        #     print('duplicate! {}'.format(test.name))
        if test.is_flaky:
            self.add_flaky_test(test)
        

class Node:
    def __init__(self, name, children):
        self.name = name
        self.children = children
        import hashlib
        hash_obj = hashlib.sha384(self.name.encode('utf-8'))
        self.digest = hash_obj.hexdigest()

    def __str__(self):
        return self.name