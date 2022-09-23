import numpy as np
import pandas as pd
import unittest

from censyn.checks import ConsistencyCheck, LessThanConsistencyCheck, AndConsistencyCheck, AllConsistencyCheck, \
    AnyConsistencyCheck, BooleanDataCalculator, NumericDataCalculator, StringDataCalculator, rename_variables

test_df = pd.DataFrame({'name': ['Bob', 'Alice', 'Joe', 'Chris'],
                        'age': [28, 43, 36, -20],
                        'edu': ['BS', 'some college', 'HS', None],
                        'year': [2002, 2008, 2005, 2001],
                        'year_2': [2005, 2009, 2007, 2008],
                        'sex': ['m', 'f', 'm', None],
                        'id': ['00101A', '00101B', '00111C', '01021A'],
                        'b_1': [True, True, False, True],
                        'b_2': [1, 2, 0, 0],
                        'b_3': [1, 2, 0, 3],
                        'occ': ['manager', 'accountant', 'sales clerk', 'engineer 1']
                        })


class TestChecks(unittest.TestCase):

    def test_rename_variables_empty(self) -> None:
        params = [
            ("VAR1 + _VAR2", "VAR1 + _VAR2")
        ]
        for expr, test_expr in params:
            with self.subTest():
                res_expr = rename_variables(expression=expr, rename={}, validate=True)
                self.assertEqual(res_expr, test_expr, msg=f"result '{res_expr}' not equal test expr '{test_expr}'.")

    def test_rename_variables(self) -> None:
        params = [
            ("VAR1 + VAR1", "VAR_1 + VAR_1"),
            ("VAR1 + _VAR2", "VAR_1 + VAR_2"),
        ]
        rename_d = {"VAR1": "VAR_1",
                    "_VAR2": "VAR_2"}
        for expr, test_expr in params:
            with self.subTest():
                res_expr = rename_variables(expression=expr, rename=rename_d, validate=True)
                self.assertEqual(res_expr, test_expr, msg=f"result '{res_expr}' not equal test expr '{test_expr}'.")

    def test_rename_variables_invalid(self) -> None:
        params = [
            {"VAR1": "min"},
            {"if": "VAR1"},
            {"VAR2": None},
            {None: "VAR2"},
            {"VAR3": ["VAR3"]},
            {"VAR3": ("VAR3", "VAR_3")},
            {("VAR3", "VAR_3"): "VAR_3"},
            {"VAR4": "4VAR"},
            {"4VAR": "VAR4"},
            {"VAR4": ""},
            {"": "VAR4"},
            {"VAR5": "VAR 5"},
            {"VAR 5": "VAR5"},
        ]
        for rename_d in params:
            with self.assertRaises(ValueError):
                rename_variables(expression="", rename=rename_d, validate=True)

    def test_BooleanScalarData(self) -> None:
        params = [
            ('true', True),
            ('True', True),
            ('TRUE', True),
            ('false', False),
            ('False', False),
            ('FALSE', False),
            ('True or False and False', True),
            ('True and True or True and False', True),
            ('not False and True or False', True),
            ('None or True', True),
            ('None and True', False),
            ('isnull(True)', False),
            ('isnull(False)', False),
            ('notnull(True)', True),
            ('notnull(False)', True),
            ('isnull("Hello")', False),
            ('isnull("")', False),
            ('isnull(None)', True),
            ('notnull("Hello")', True),
            ('notnull("")', True),
            ('notnull(None)', False),
            ('isnull(1)', False),
            ('notnull(2)', True),
            ('startswith("Hello", "He")', True),
            ('startswith("  Hello  ", "He")', False),
            ('startswith("Hello", "he")', False),
            ('startswith("Hello", "")', True),
            ('endswith("Hello", "lo")', True),
            ('endswith("Hello", "klo")', False),
            ('contains("Hello", "ll")', True),
            ('contains("Hello", "ee")', False),
            ('isin("ab", ["abc", 1, "bc", concat("a", "b"), "a"])', True),
            ('isin("ab", ["abc", 1 + 2, "bc", "a"])', False),
            ('isin(3, ["abc", 1 + 2, "bc", "a"])', True),
            ('isin("ab", [])', False),
            ('isin("ab", ["ab"])', True),
            ('1 <= 2', True),
            ('1 < 2', True),
            ('1 >= 2', False),
            ('1 > 2', False),
            ('1 == 2', False),
            ('1 != 2', True),
            ('2 <= 2', True),
            ('2 < 2', False),
            ('2 >= 2', True),
            ('2 > 2', False),
            ('2 == 2', True),
            ('2 != 2', False),
            ('"abc" <= "def"', True),
            ('"abc" < "def"', True),
            ('"abc" >= "def"', False),
            ('"abc" > "def"', False),
            ('"abc" == "def"', False),
            ('"abc" != "def"', True),
            ('"abc" == "abc"', True),
            ('"abc" != "abc"', False),
            ('None == "abc"', False),
            ('None != "abc"', True),
            ('"abc" == None', False),
            ('"abc" != None', True),
            ('None == None', True),
            ('None != None', False),
            ('if( True then False else True)', False),
        ]
        bool_df = pd.DataFrame()
        for expr, res in params:
            with self.subTest():
                calculator = BooleanDataCalculator(feature_name='test', expression=expr)
                dep = calculator.compute_variables()
                self.assertEqual(len(dep), 0, msg=f"Expression: '{expr}'")
                cur_res = calculator.execute(in_df=bool_df)
                self.assertEqual(cur_res, res, msg=f"expr='{expr}'")
                self.assertTrue(bool_df.empty, msg=f"expr='{expr}': DataFrame shape {bool_df.shape} is not empty")

    def test_BooleanSeriesData(self) -> None:
        params = [
            ('b_1 or b_2', ['b_1', 'b_2'], [True, True, False, True]),
            ('b_1 and b_3 or b_2 and b_3', ['b_1', 'b_2', 'b_3'], [True, False, False, True]),
            ('not b_1 or b_2', ['b_1', 'b_2'], [False, True, True, True]),
            ('b_1 or None', ['b_1'], [True, True, False, False]),
            ('None and b_1', ['b_1'], [False, False, False, False]),
            ('none or b_1', ['none', 'b_1'], [True, True, False, False]),
            ('b_1 or none', ['b_1', 'none'], [True, True, False, False]),
            ('b_1 and none', ['b_1', 'none'], [False, False, False, False]),
            ('isnull(b_3)', ['b_3'], [False, False, True, False]),
            ('isnull(s_2)', ['s_2'], [False, True, False, False]),
            ('notnull(s_2)', ['s_2'], [True, False, True, True]),
            ('isnull(n_3)', ['n_3'], [False, True, True, False]),
            ('notnull(n_3)', ['n_3'], [True, False, False, True]),
            ('startswith(s_1, "a")', ['s_1'], [True, True, False, False]),
            ('endswith(s_1, "def")', ['s_1'], [True, False, True, True]),
            ('contains(s_1, "cd")', ['s_1'], [True, False, True, False]),
            ('startswith(s_2, "a")', ['s_2'], [True, None, False, False]),
            ('endswith(s_2, "def")', ['s_2'], [False, None, True, False]),
            ('contains(s_2, "b")', ['s_2'], [True, None, False, False]),
            ('isin(n_1, [0, 1, 10, 11, count(n_3)])', ['n_1', 'n_3'], [True, True, True, False]),
            ('isin(s_1, ["abcdef", 1, concat("def", "abc"), "abcabc"])', ['s_1'], [True, True, False, False]),
            ('isin(s_2, ["abcdef", None, "def", "abcabc"])', ['s_2'], [False, True, True, False]),
            ('isin(n_3, ["abcdef", 1, "abc", "abcabc"])', ['n_3'], [True, False, False, False]),
            ('isin(n_3, ["abcdef", 1, Nan, s_1, "abcabc"])', ['s_1', 'n_3'], [True, True, True, False]),
            ('sort(b_2, True)', ['b_2'], [False, False, True, True]),
            ('any(b_1)', ['b_1'], [True, True, True, True]),
            ('any(b_1 and False)', ['b_1'], [False, False, False, False]),
            ('all(b_1)', ['b_1'], [False, False, False, False]),
            ('all(b_1 or True)', ['b_1'], [True, True, True, True]),
            ('if( True then b_1 else b_2)', ['b_1', 'b_2'], [True, True, False, False]),
            ('if( b_1 then b_1 elseif b_2 then b_2 else False)', ['b_1', 'b_2'], [True, True, False, True]),
        ]
        bool_df = pd.DataFrame({'b_1': [True, True, False, False],
                                'b_2': [False, True, False, True],
                                'b_3': [True, False, None, True],
                                'none': [None, None, None, None],
                                's_1': ['abcdef', 'abcabc', 'defabcdef', 'def'],
                                's_2': ['abc', None, 'def', ''],
                                'n_1': [1, 2, 0, 3],
                                'n_3': [1, np.nan, np.nan, 2],
                                })
        copy_df = bool_df.copy()
        f_name = "test"
        for expr, dependencies, res in params:
            with self.subTest():
                calculator = BooleanDataCalculator(feature_name=f_name, expression=expr)
                dep = calculator.compute_variables()
                self.assertEqual(len(dep), len(dependencies), msg=f"Expression: '{expr}'")
                self.assertEqual(set(dep), set(dependencies), msg=f"Expression: '{expr}'")
                cur_res = calculator.execute(in_df=bool_df)
                pd.testing.assert_series_equal(cur_res, pd.Series(name=f_name, data=res))
                self.assertTrue(bool_df.equals(copy_df), msg=f"Expression: '{expr}'")

    def test_BooleanComparisonSeriesData(self) -> None:
        params = [
            ('n_1 <= n_2', ['n_1', 'n_2'], [True, False, True, False]),
            ('n_1 < n_2', ['n_1', 'n_2'], [False, False, True, False]),
            ('n_1 >= n_2', ['n_1', 'n_2'], [True, True, False, True]),
            ('n_1 > n_2', ['n_1', 'n_2'], [False, True, False, True]),
            ('n_1 == n_2', ['n_1', 'n_2'], [True, False, False, False]),
            ('n_1 == 2', ['n_1'], [False, True, False, False]),
            ('2 == n_1', ['n_1'], [False, True, False, False]),
            ('n_1 == n_2', ['n_1', 'n_2'], [True, False, False, False]),
            ('n_1 != n_2', ['n_1', 'n_2'], [False, True, True, True]),
            ('n_1 <= 2', ['n_1'], [True, True, True, False]),
            ('n_1 < 2', ['n_1'], [True, False, True, False]),
            ('n_1 >= 2', ['n_1'], [False, True, False, True]),
            ('n_1 > 2', ['n_1'], [False, False, False, True]),
            ('n_1 == 2', ['n_1'], [False, True, False, False]),
            ('n_1 != 2', ['n_1'], [True, False, True, True]),
            ('n_1 >= n_3', ['n_1', 'n_3'], [True, False, False, True]),
            ('n_1 <= n_3', ['n_1', 'n_3'], [True, False, False, False]),
            ('n_3 == n_3', ['n_3'], [True, False, False, True]),
            ('Nan <= n_2', ['n_2'], [False, False, False, False]),
            ('s_1 <= s_2', ['s_1', 's_2'], [False, False, False, True]),
            ('s_1 < s_2', ['s_1', 's_2'], [False, False, False, False]),
            ('s_1 >= s_2', ['s_1', 's_2'], [True, True, True, True]),
            ('s_1 > s_2', ['s_1', 's_2'], [True, True, True, False]),
            ('s_1 == s_2', ['s_1', 's_2'], [False, False, False, True]),
            ('s_1 != s_2', ['s_1', 's_2'], [True, True, True, False]),
            ('s_1 == s_3', ['s_1', 's_3'], [False, False, False, True]),
            ('s_1 != s_3', ['s_1', 's_3'], [True, True, True, False]),
        ]
        bool_df = pd.DataFrame({'s_1': ['abcdef', 'abcabc', 'defabcdef', 'def'],
                                's_2': ['abc', '', 'def', 'def'],
                                's_3': ['abc', '', None, 'def'],
                                'n_1': [1, 2, 0, 3],
                                'n_2': [1, 0, 3, 2],
                                'n_3': [1, np.nan, np.nan, 2],
                                })
        copy_df = bool_df.copy()
        f_name = "test"
        for expr, dependencies, res in params:
            with self.subTest():
                calculator = BooleanDataCalculator(feature_name=f_name, expression=expr)
                dep = calculator.compute_variables()
                self.assertEqual(len(dep), len(dependencies), msg=f"Expression: '{expr}'")
                self.assertEqual(set(dep), set(dependencies), msg=f"Expression: '{expr}'")
                cur_res = calculator.execute(in_df=bool_df)
                pd.testing.assert_series_equal(cur_res, pd.Series(name=f_name, data=res))
                self.assertTrue(bool_df.equals(copy_df), msg=f"Expression: '{expr}'")

    def test_BooleanGroupByFunctionSeriesData(self) -> None:
        params = [
            ('any(groupby(n_1), "b_1")', ['n_1', 'b_1'],
             [True, True, True, True, True, True, True, False, False, True]),
            ('any(series_groupby(b_1, by= n_1))', ['n_1', 'b_1'],
             [True, True, True, True, True, True, True, False, False, True]),
            ('all(groupby(n_1), "b_1")', ['n_1', 'b_1'],
             [False, False, False, False, True, True, True, False, False, True]),
            ('all(series_groupby(b_1, by= n_1))', ['n_1', 'b_1'],
             [False, False, False, False, True, True, True, False, False, True]),
            ('is_increasing(series_groupby(year, by= n_1))', ['n_1', 'year'],
             [False, False, False, False, True, True, True, False, False, True]),
            ('is_decreasing(series_groupby(year, by= n_1))', ['n_1', 'year'],
             [False, False, False, False, False, False, False, True, True, True]),
            ('apply(groupby(["n_1"]), expr= "sum(n_1) > 5" )', ['n_1'],
             [False, False, False, False, True, True, True, True, True, False]),
            ('apply(groupby(["n_1"]), boolean_expr= "sum(n_1) > 5" )', ['n_1'],
             [False, False, False, False, True, True, True, True, True, False]),
        ]
        string_df = pd.DataFrame({'n_1': [1, 1, 1, 1, 2, 2, 2, 3, 3, 4],
                                  'sex': ['m', 'f', 'm', 'm', 'f', 'm', 'm', 'f', 'm', 'm'],
                                  'year': [2002, 2003, 2005, 2001, 2002, 2003, 2004, 2005, 2004, 2001],
                                  'b_1': [True, True, False, False, True, True, True, False, False, True],
                                  })
        copy_df = string_df.copy()
        f_name = "test"
        for expr, dependencies, res in params:
            with self.subTest():
                calculator = BooleanDataCalculator(feature_name=f_name, expression=expr)
                dep = calculator.compute_variables()
                self.assertEqual(len(dep), len(dependencies), msg=f"Expression: '{expr}'")
                self.assertEqual(set(dep), set(dependencies), msg=f"Expression: '{expr}'")
                cur_res = calculator.execute(in_df=string_df)
                pd.testing.assert_series_equal(cur_res, pd.Series(name=f_name, data=res))
                self.assertTrue(string_df.equals(copy_df), msg=f"Expression: '{expr}'")

    def test_NumericScalarData(self) -> None:
        params = [
            ('2 + 3 * 2', 8),
            ('2 * 5 - 3', 7),
            ('2 * - 3 + 9', 3),
            ('2 + 6 / 3', 4),
            ('(2 + 3) * 4', 20),
            ('abs(5 - 12)', 7),
            ('pow(3, 3)', 27),
            ('round(1234.567, 1)', 1234.6),
            ('round(1234.567, 0)', 1235),
            ('round(1234.567, -1)', 1230),
            ('length("Hello")', 5),
            ('length("")', 0),
            ('find("Hello", "ll")', 2),
            ('find("Hello", "low")', -1),
            ('find("Hello", "")', 0),
            ('int("1")', 1),
            ('int(1.0)', 1),
            ('fillna(Nan, 2)', 2),
            ('fillna(3.5, 2)', 3.5),
            ('min(3, 6, 10, -1, 0.5)', -1),
            ('max(3, 6, 10, -1, 0.5)', 10),
            ('min(3)', 3),
            ('max(4)', 4),
            ('ifthenelse( TRUE, 1, 2)', 1),
            ('ifthenelse( False, 1, 2)', 2),
            ('if( True then 1 else 2)', 1),
            ('if( False then 1 else 2)', 2),
        ]
        numeric_df = pd.DataFrame()
        for expr, res in params:
            with self.subTest():
                calculator = NumericDataCalculator(feature_name='test', expression=expr)
                dep = calculator.compute_variables()
                self.assertEqual(len(dep), 0, msg=f"Expression: '{expr}'")
                cur_res = calculator.execute(in_df=numeric_df)
                self.assertEqual(cur_res, res, msg=f"expr='{expr}'")
                self.assertTrue(numeric_df.empty, msg=f"expr='{expr}': DataFrame shape {numeric_df.shape} is not empty")

    def test_NumericFunctionSeriesData(self) -> None:
        params = [
            ('n_1 + n_2 * n_2', ['n_1', 'n_2'], [2, 2, 9, 7]),
            ('n_1 * n_3 + n_2', ['n_1', 'n_2', 'n_3'], [2, np.nan, np.nan, 8]),
            ('n_4 + n_5', ['n_4', 'n_5'], [4, np.nan, 8, 16]),
            ('n_4 * 2 - n_6', ['n_4', 'n_6'], [-8, -10, -16, -12]),
            ('abs(n_4 * 2 - n_6)', ['n_4', 'n_6'], [8, 10, 16, 12]),
            ('pow(3, n_2)', ['n_2'], [3, 1, 27, 9]),
            ('pow(n_2, 3)', ['n_2'], [1, 0, 27, 8]),
            ('pow(n_2, n_1)', ['n_1', 'n_2'], [1, 0, 1, 8]),
            ('round(n_8, 0)', ['n_8'], [100., 15., 2., 36.]),
            ('round(n_8, 2)', ['n_8'], [100.12, 15.22, 2.12, 36.22]),
            ('length(s_2)', ['s_2'], [7, 7, 4, 5]),
            ('find(s_2, "hic")', ['s_2'], [1, 1, -1, -1]),
            ('find(s_2, "e")', ['s_2'], [5, 5, -1, 4]),
            ('n_4 - int(s_1)', ['n_4', 's_1'], [-1, -1, 1, 4]),
            ('n_4 + fillna(n_5, 0)', ['n_4', 'n_5'], [4, 2, 8, 16.0]),
            ('rowid()', [], [1, 2, 3, 4]),
            ('sort(n_2, True)', ['n_2'], [0, 1, 2, 3]),
            ('sort(n_2, False)', ['n_2'], [3, 2, 1, 0]),
            ('sort(n_5, True)', ['n_5'], [2, 4, 8, np.nan]),
            ('sort(n_5, False)', ['n_5'], [8, 4, 2, np.nan]),
            ('sort(n_2 + 2, True)', ['n_2'], [2, 3, 4, 5]),
            ('min(n_6, n_7)', ['n_6', 'n_7'], [1, 1, 2, 3]),
            ('min(n_6, n_7, 1)', ['n_6', 'n_7'], [1, 1, 1, 1]),
            ('min(n_1, n_2, n_6, n_7)', ['n_1', 'n_2', 'n_6', 'n_7'], [1, 0, 0, 2]),
            ('min(n_1, n_2, n_6, n_7, 0, 1)', ['n_1', 'n_2', 'n_6', 'n_7'], [0, 0, 0, 0]),
            ('max(n_6, n_7)', ['n_6', 'n_7'], [12, 14, 24, 28]),
            ('max(n_6, n_7, 26)', ['n_6', 'n_7'], [26, 26, 26, 28]),
            ('max(n_1, n_2, n_6, n_7)', ['n_1', 'n_2', 'n_6', 'n_7'], [12, 14, 24, 28]),
            ('max(n_1, n_2, n_6, n_7, 0, 26)', ['n_1', 'n_2', 'n_6', 'n_7'], [26, 26, 26, 28]),
            ('max(2, 3, 4)', [], [4, 4, 4, 4]),
            ('max(min(n_6, n_7), min(n_4 * 6, n_6))', ['n_4', 'n_6', 'n_7'], [12, 12, 24, 28]),
            ('mean(n_4)', ['n_4'], [4.0, 4.0, 4.0, 4.0]),
            ('median(n_4)', ['n_4'], [3.0, 3.0, 3.0, 3.0]),
            ('std(n_4)', ['n_4'], [2.82843, 2.82843, 2.82843, 2.82843]),
            ('sem(n_4)', ['n_4'], [1.41421, 1.41421, 1.41421, 1.41421]),
            ('min(n_4)', ['n_4'], [2, 2, 2, 2]),
            ('max(n_4)', ['n_4'], [8, 8, 8, 8]),
            ('sum(n_4)', ['n_4'], [16, 16, 16, 16]),
            ('count(n_5)', ['n_5'], [3, 3, 3, 3]),
            ('size(n_4)', ['n_4'], [4, 4, 4, 4]),
            ('unique(n_4, n_5)', ['n_4', 'n_5'], [1, 1, 1, 1]),
            ('unique(n_7, s_1)', ['n_7', 's_1'], [2, 2, 1, 1]),
            ('unique(n_4, n_5)', ['n_4', 'n_5'], [1, 1, 1, 1]),
            ('unique(n_7, s_1, s_2)', ['n_7', 's_1', 's_2'], [2, 2, 1, 1]),
            ('ifthenelse(True, n_6, 0)', ['n_6'], [12, 14, 24, 28]),
            ('ifthenelse(False, n_6, 0)', ['n_6'], [0, 0, 0, 0]),
            ('ifthenelse(True, n_6 + 2, n_6 + 4)', ['n_6'], [14, 16, 26, 30]),
            ('ifthenelse( n_4 <= 2, 1, n_6)', ['n_4', 'n_6'], [1, 1, 24, 28]),
            ('ifthenelse( n_4 <= 2, 1, ifthenelse( n_4 <= 6, 2, n_6))', ['n_4', 'n_6'], [1, 1, 2, 28]),
            ('if( b_1 then n_2 elseif b_2 then 3 else n_6)',  ['b_1', 'b_2', 'n_2', 'n_6'], [1, 0, 24, 3]),
            ('if(True then n_6 + 2 else n_6 + 4)', ['n_6'], [14, 16, 26, 30]),
            ('if(True then n_6 + 2 elseif False then n_4 else n_6 + 4)', ['n_4', 'n_6'], [14, 16, 26, 30]),
        ]
        numeric_df = pd.DataFrame({'n_1': [1, 2, 0, 3],
                                   'n_2': [1, 0, 3, 2],
                                   'n_3': [1, np.nan, np.nan, 2],
                                   'n_4': [2, 2, 4, 8],
                                   'n_5': [2, np.nan, 4, 8],
                                   'n_6': [12, 14, 24, 28],
                                   'n_7': [1, 1, 2, 3],
                                   'n_8': [100.123123, 15.22234, 2.12323, 36.22303],
                                   'b_1': [True, True, False, False],
                                   'b_2': [False, True, False, True],
                                   's_1': ['3', '3', '3', '4'],
                                   's_2': ['chicken', 'chicken', 'bird', 'snake'],
                                   })
        copy_df = numeric_df.copy()
        f_name = "test"
        for expr, dependencies, res in params:
            with self.subTest():
                calculator = NumericDataCalculator(feature_name=f_name, expression=expr)
                dep = calculator.compute_variables()
                self.assertEqual(len(dep), len(dependencies), msg=f"Expression: '{expr}'")
                self.assertEqual(set(dep), set(dependencies), msg=f"Expression: '{expr}'")
                cur_res = calculator.execute(in_df=numeric_df)
                pd.testing.assert_series_equal(cur_res, pd.Series(name=f_name, data=res))
                self.assertTrue(numeric_df.equals(copy_df), msg=f"Expression: '{expr}'")

    def test_NumericGroupByFunctionSeriesData(self) -> None:
        params = [
            ('cumcount(groupby(n_1), "n_2")', ['n_1', 'n_2'], [0.0, 1.0, 2.0, 3.0, 0.0, 1.0, 2.0, 0.0, 1.0, 0.0]),
            ('cumcount(series_groupby(n_2, by= n_1))', ['n_1', 'n_2'],
             [0.0, 1.0, 2.0, 3.0, 0.0, 1.0, 2.0, 0.0, 1.0, 0.0]),
            ('cummax(groupby(n_1), "n_2")', ['n_1', 'n_2'], [0.0, 0.0, 3.0, 3.0, 1.0, 2.0, 3.0, 4.0, 5.0, 0.0]),
            ('cummax(series_groupby(n_2, by= n_1))', ['n_1', 'n_2'],
             [0.0, 0.0, 3.0, 3.0, 1.0, 2.0, 3.0, 4.0, 5.0, 0.0]),
            ('cummin(groupby(n_1), "n_2")', ['n_1', 'n_2'], [0.0, 0.0, 0.0, 0.0, 1.0, 1.0, 1.0, 4.0, 4.0, 0.0]),
            ('cummin(series_groupby(n_2, by= n_1))', ['n_1', 'n_2'],
             [0.0, 0.0, 0.0, 0.0, 1.0, 1.0, 1.0, 4.0, 4.0, 0.0]),
            ('cumprod(groupby(n_1), "n_2")', ['n_1', 'n_2'], [0.0, 0.0, 0.0, 0.0, 1.0, 2.0, 6.0, 4.0, 20.0, 0.0]),
            ('cumprod(series_groupby(n_2, by= n_1))', ['n_1', 'n_2'],
             [0.0, 0.0, 0.0, 0.0, 1.0, 2.0, 6.0, 4.0, 20.0, 0.0]),
            ('cumsum(groupby(n_1), "n_2")', ['n_1', 'n_2'], [0.0, 0.0, 3.0, 5.0, 1.0, 3.0, 6.0, 4.0, 9.0, 0.0]),
            ('cumsum(series_groupby(n_2, by= n_1))', ['n_1', 'n_2'],
             [0.0, 0.0, 3.0, 5.0, 1.0, 3.0, 6.0, 4.0, 9.0, 0.0]),
            ('ngroup(groupby(n_1))', ['n_1'], [0.0, 0.0, 0.0, 0.0, 1.0, 1.0, 1.0, 2.0, 2.0, 3.0]),
            ('ngroup(series_groupby(n_1, by= n_1))', ['n_1'], [0.0, 0.0, 0.0, 0.0, 1.0, 1.0, 1.0, 2.0, 2.0, 3.0]),
            ('size(groupby(n_1))', ['n_1'], [4.0, 4.0, 4.0, 4.0, 3.0, 3.0, 3.0, 2.0, 2.0, 1.0]),
            ('size(series_groupby(n_1, by= n_1))', ['n_1'], [4.0, 4.0, 4.0, 4.0, 3.0, 3.0, 3.0, 2.0, 2.0, 1.0]),
            ('count(groupby(n_1), "n_2")', ['n_1', 'n_2'], [4.0, 4.0, 4.0, 4.0, 3.0, 3.0, 3.0, 2.0, 2.0, 1.0]),
            ('count(series_groupby(n_2, by=n_1))', ['n_1', 'n_2'], [4.0, 4.0, 4.0, 4.0, 3.0, 3.0, 3.0, 2.0, 2.0, 1.0]),
            ('prod(groupby(n_1), "n_2")', ['n_1', 'n_2'], [0.0, 0.0, 0.0, 0.0, 6.0, 6.0, 6.0, 20.0, 20.0, 0.0]),
            ('prod(series_groupby(n_2, by=n_1))', ['n_1', 'n_2'], [0.0, 0.0, 0.0, 0.0, 6.0, 6.0, 6.0, 20.0, 20.0, 0.0]),
            ('sum(groupby(n_1), "n_2")', ['n_1', 'n_2'], [5.0, 5.0, 5.0, 5.0, 6.0, 6.0, 6.0, 9.0, 9.0, 0.0]),
            ('sum(series_groupby(n_2, by=n_1))', ['n_1', 'n_2'], [5.0, 5.0, 5.0, 5.0, 6.0, 6.0, 6.0, 9.0, 9.0, 0.0]),
            ('mean(groupby(n_1), "n_2")', ['n_1', 'n_2'], [1.25, 1.25, 1.25, 1.25, 2.0, 2.0, 2.0, 4.5, 4.5, 0.0]),
            ('mean(series_groupby(n_2, by=n_1))', ['n_1', 'n_2'],
             [1.25, 1.25, 1.25, 1.25, 2.0, 2.0, 2.0, 4.5, 4.5, 0.0]),
            ('median(groupby(n_1), "n_2")', ['n_1', 'n_2'], [1.0, 1.0, 1.0, 1.0, 2.0, 2.0, 2.0, 4.5, 4.5, 0.0]),
            ('median(series_groupby(n_2, by=n_1))', ['n_1', 'n_2'], [1.0, 1.0, 1.0, 1.0, 2.0, 2.0, 2.0, 4.5, 4.5, 0.0]),
            ('std(groupby(n_1), "n_2")', ['n_1', 'n_2'], [1.5, 1.5, 1.5, 1.5, 1.0, 1.0, 1.0, 0.70711, 0.70711, np.nan]),
            ('std(series_groupby(n_2, by=n_1))', ['n_1', 'n_2'],
             [1.5, 1.5, 1.5, 1.5, 1.0, 1.0, 1.0, 0.70711, 0.70711, np.nan]),
            ('min(groupby(n_1), "n_2")', ['n_1', 'n_2'], [0.0, 0.0, 0.0, 0.0, 1.0, 1.0, 1.0, 4.0, 4.0, 0.0]),
            ('min(series_groupby(n_2, by=n_1))', ['n_1', 'n_2'], [0.0, 0.0, 0.0, 0.0, 1.0, 1.0, 1.0, 4.0, 4.0, 0.0]),
            ('max(groupby(n_1), "n_2")', ['n_1', 'n_2'], [3.0, 3.0, 3.0, 3.0, 3.0, 3.0, 3.0, 5.0, 5.0, 0.0]),
            ('max(series_groupby(n_2, by=n_1))', ['n_1', 'n_2'], [3.0, 3.0, 3.0, 3.0, 3.0, 3.0, 3.0, 5.0, 5.0, 0.0]),
            ('size(groupby(n_1 + n_2))', ['n_1', 'n_2'], [2.0, 2.0, 3.0, 2.0, 2.0, 3.0, 1.0, 1.0, 1.0, 3.0]),
            ('min(groupby(n_1 + n_2), "n_2")', ['n_1', 'n_2'], [0.0, 0.0, 0.0, 1.0, 1.0, 0.0, 3.0, 4.0, 5.0, 0.0]),
            ('max(groupby(n_1 + n_2), "n_2")', ['n_1', 'n_2'], [0.0, 0.0, 3.0, 2.0, 2.0, 3.0, 3.0, 4.0, 5.0, 3.0]),
            ('size(groupby(sex))', ['sex'], [7.0, 3.0, 7.0, 7.0, 3.0, 7.0, 7.0, 3.0, 7.0, 7.0]),
            ('min(groupby(sex), "n_2")', ['sex', 'n_2'], [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]),
            ('min(series_groupby(n_2, by=sex))', ['sex', 'n_2'], [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]),
            ('max(groupby(sex), "n_2")', ['sex', 'n_2'], [5.0, 4.0, 5.0, 5.0, 4.0, 5.0, 5.0, 4.0, 5.0, 5.0]),
            ('max(series_groupby(n_2, by=sex))', ['sex', 'n_2'], [5.0, 4.0, 5.0, 5.0, 4.0, 5.0, 5.0, 4.0, 5.0, 5.0]),
            ('size(groupby(b_1))', ['b_1'], [6.0, 6.0, 4.0, 4.0, 6.0, 6.0, 4.0, 4.0, 6.0, 6.0]),
            ('min(groupby(b_1), "n_2")', ['b_1', 'n_2'], [0.0, 0.0, 2.0, 2.0, 0.0, 0.0, 2.0, 2.0, 0.0, 0.0]),
            ('max(groupby(b_1), "n_2")', ['b_1', 'n_2'], [5.0, 5.0, 4.0, 4.0, 5.0, 5.0, 4.0, 4.0, 5.0, 5.0]),
            ('size(groupby(n_1 <= 1))', ['n_1'], [4.0, 4.0, 4.0, 4.0, 6.0, 6.0, 6.0, 6.0, 6.0, 6.0]),
            ('min(groupby(n_1 <= 1), "n_2")', ['n_1', 'n_2'], [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]),
            ('max(groupby(n_1 <= 1), "n_2")', ['n_1', 'n_2'], [3.0, 3.0, 3.0, 3.0, 5.0, 5.0, 5.0, 5.0, 5.0, 5.0]),
            ('size(groupby(["n_1", "sex"]))', ['n_1', 'sex'],
             [3.0, 1.0, 3.0, 3.0, 1.0, 2.0, 2.0, 1.0, 1.0, 1.0]),
            ('count(groupby(["n_1", "sex"]), "n_2")', ['n_1', 'n_2', 'sex'],
             [3.0, 1.0, 3.0, 3.0, 1.0, 2.0, 2.0, 1.0, 1.0, 1.0]),
            ('sum(groupby(["n_1", "sex"]), "n_2")', ['n_1', 'n_2', 'sex'],
             [5.0, 0.0, 5.0, 5.0, 1.0, 5.0, 5.0, 4.0, 5.0, 0.0]),
            ('median(groupby(["n_1", "sex"]), "n_2")', ['n_1', 'n_2', 'sex'],
             [2.0, 0.0, 2.0, 2.0, 1.0, 2.5, 2.5, 4.0, 5.0, 0.0]),
            ('min(groupby(["n_1", "sex"]), "n_2")', ['n_1', 'n_2', 'sex'],
             [0.0, 0.0, 0.0, 0.0, 1.0, 2.0, 2.0, 4.0, 5.0, 0.0]),
            ('max(groupby(["n_1", "sex"]), "n_2")', ['n_1', 'n_2', 'sex'],
             [3.0, 0.0, 3.0, 3.0, 1.0, 3.0, 3.0, 4.0, 5.0, 0.0]),
            ('apply(groupby(["n_1"]), expr="sum(n_2)" )', ['n_1', 'n_2'],
             [5.0, 5.0, 5.0, 5.0, 6.0, 6.0, 6.0, 9.0, 9.0, 0.0]),
            ('apply(groupby(["n_1"]), numeric_expr="sum(n_2)" )', ['n_1', 'n_2'],
             [5.0, 5.0, 5.0, 5.0, 6.0, 6.0, 6.0, 9.0, 9.0, 0.0]),
            ('apply(groupby(["n_1"]), expr="if( sum(n_2) < 6 then 1 else Nan)" )', ['n_1', 'n_2'],
             [1.0, 1.0, 1.0, 1.0, np.nan, np.nan, np.nan, np.nan, np.nan, 1.0]),
        ]
        numeric_df = pd.DataFrame({'n_1': [1, 1, 1, 1, 2, 2, 2, 3, 3, 4],
                                   'n_2': [0, 0, 3, 2, 1, 2, 3, 4, 5, 0],
                                   'sex': ['m', 'f', 'm', 'm', 'f', 'm', 'm', 'f', 'm', 'm'],
                                   'year': [2002, 2003, 2005, 2001, 2002, 2003, 2001, 2005, 2004, 2001],
                                   'b_1': [True, True, False, False, True, True, False, False, True, True],
                                   })
        copy_df = numeric_df.copy()
        f_name = "test"
        for expr, dependencies, res in params:
            with self.subTest():
                calculator = NumericDataCalculator(feature_name=f_name, expression=expr)
                dep = calculator.compute_variables()
                self.assertEqual(len(dep), len(dependencies), msg=f"Expression: '{expr}'")
                self.assertEqual(set(dep), set(dependencies), msg=f"Expression: '{expr}'")
                cur_res = calculator.execute(in_df=numeric_df)
                pd.testing.assert_series_equal(cur_res, pd.Series(name=f_name, data=res))
                self.assertTrue(numeric_df.equals(copy_df), msg=f"Expression: '{expr}'")

    def test_StringScalarData(self) -> None:
        params = [
            ('concat("Hello", " World")', 'Hello World'),
            ('concat("start", "")', 'start'),
            ('concat("", "end")', 'end'),
            ('strip("   test   ")', 'test'),
            ('strip("   test   ", "")', '   test   '),
            ('strip("012test961", "0123")', 'test96'),
            ('lstrip("   test   ")', 'test   '),
            ('lstrip("012test961", "0123")', 'test961'),
            ('rstrip("   test   ")', '   test'),
            ('rstrip("012test961", "0123")', '012test96'),
            ('replace("str_Test", "str_", "S")', 'STest'),
            ('replace("str_Test", "s", "S")', 'Str_TeSt'),
            ('replace("Test_str", "str", "_")', 'Test__'),
            ('slice("Test_str", 0, 4)', 'Test'),
            ('slice("Test_str", -3, 8)', 'str'),
            ('slice("Test_str", -3)', 'str'),
            ('slice("Test_str", -3, 4)', ''),
            ('get("Test_str", -3)', 's'),
            ('get("Test_str", 0)', 'T'),
            ('get("Test_str", 4)', '_'),
            ('padleft("1", 5, "0")', '00001'),
            ('padleft("1234567", 5, "0")', '1234567'),
            ('padright("1", 5, "0")', '10000'),
            ('padright("1234567", 5, "0")', '1234567'),
            ('str(123)', '123'),
            ('str("123")', '123'),
            ('fillna("1", "123")', '1'),
            ('fillna(None, "123")', '123'),
            ('ifthenelse(True, "1", "2")', '1'),
            ('ifthenelse(False, "1", "2")', '2'),
            ('if( True then "A" else "B")', 'A'),
        ]
        string_df = pd.DataFrame()
        for expr, res in params:
            with self.subTest():
                calculator = StringDataCalculator(feature_name='test', expression=expr)
                dep = calculator.compute_variables()
                self.assertEqual(len(dep), 0, msg=f"Expression: '{expr}'")
                cur_res = calculator.execute(in_df=string_df)
                self.assertEqual(cur_res, res, msg=f"expr='{expr}'")
                self.assertTrue(string_df.empty, msg=f"expr='{expr}': DataFrame shape {string_df.shape} is not empty")

    def test_StringFunctionSeriesData(self) -> None:
        params = [
            ('concat(name, sex)', ['name', 'sex'], ['Bobm', 'Alicef', 'Joem', np.nan]),
            ('concat(name, fillna(sex, "x"))', ['name', 'sex'], ['Bobm', 'Alicef', 'Joem', 'Chrisx']),
            ('concat(name, "1")', ['name'], ['Bob1', 'Alice1', 'Joe1', 'Chris1']),
            ('concat(name, str(year))', ['name', 'year'], ['Bob2002', 'Alice2008', 'Joe2005', 'Chris2001']),
            ('concat("__" , name)', ['name'], ['__Bob', '__Alice', '__Joe', '__Chris']),
            ('lstrip(id, "0")', ['id'], ['101A', '101B', '111C', '1021A']),
            ('rstrip(id, "A")', ['id'], ['00101', '00101B', '00111C', '01021']),
            ('rstrip(id, "AB")', ['id'], ['00101', '00101', '00111C', '01021']),
            ('replace(id, "00", "99")', ['id'], ['99101A', '99101B', '99111C', '01021A']),
            ('replace(id, "00", "")', ['id'], ['101A', '101B', '111C', '01021A']),
            ('replace(id, "A", "")', ['id'], ['00101', '00101B', '00111C', '01021']),
            ('slice(id, 2, -1)', ['id'], ['101', '101', '111', '021']),
            ('slice(id, -2)', ['id'], ['1A', '1B', '1C', '1A']),
            ('slice(name, 0, 4)', ['name'], ['Bob', 'Alic', 'Joe', 'Chri']),
            ('get(id, -1)', ['id'], ['A', 'B', 'C', 'A']),
            ('get(id, 1)', ['id'], ['0', '0', '0', '1']),
            ('padleft( name, 5, "-")', ['name'], ['--Bob', 'Alice', '--Joe', 'Chris']),
            ('padright(name, 6, "-")', ['name'], ['Bob---', 'Alice-', 'Joe---', 'Chris-']),
            ('ifthenelse( b_1, "1", "2")', ['b_1'], ['1', '1', '2', '1']),
            ('str(year)', ['year'], ['2002', '2008', '2005', '2001']),
            ('str(name)', ['name'], ['Bob', 'Alice', 'Joe', 'Chris']),
            ('fillna(sex, "x")', ['sex'], ['m', 'f', 'm', 'x']),
            ('sort(name, True)', ['name'], ['Alice', 'Bob', 'Chris', 'Joe']),
            ('sort(name, False)', ['name'], ['Joe', 'Chris', 'Bob', 'Alice']),
            ('sort(sex, True)', ['sex'], ['f', 'm', 'm', None]),
            ('sort(sex, False)', ['sex'], ['m', 'm', 'f', None]),
            ('ifthenelse(b_1, name, id)', ['b_1', 'name', 'id'], ['Bob', 'Alice', '00111C', 'Chris']),
            ('ifthenelse(True, name, "name")', ['name'], ['Bob', 'Alice', 'Joe', 'Chris']),
            ('ifthenelse(False, name, "name")', ['name'], ["name", "name", "name", "name"]),
            ('if( b_1 then name elseif b_2 then sex else None)',  ['b_1', 'b_2', 'name', 'sex'],
             ['Bob', 'Alice', None, 'Chris']),
            ]
        string_df = pd.DataFrame({'name': ['Bob', 'Alice', 'Joe', 'Chris'],
                                  'sex': ['m', 'f', 'm', None],
                                  'year': [2002, 2008, 2005, 2001],
                                  'id': ['00101A', '00101B', '00111C', '01021A'],
                                  'b_1': [True, True, False, True],
                                  'b_2': [False, True, False, True],
                                  })
        copy_df = string_df.copy()
        f_name = "test"
        for expr, dependencies, res in params:
            with self.subTest():
                calculator = StringDataCalculator(feature_name=f_name, expression=expr)
                dep = calculator.compute_variables()
                self.assertEqual(len(dep), len(dependencies), msg=f"Expression: '{expr}'")
                self.assertEqual(set(dep), set(dependencies), msg=f"Expression: '{expr}'")
                cur_res = calculator.execute(in_df=string_df)
                pd.testing.assert_series_equal(cur_res, pd.Series(name=f_name, data=res))
                self.assertTrue(string_df.equals(copy_df), msg=f"Expression: '{expr}'")

    def test_StringGroupByFunctionSeriesData(self) -> None:
        params = [
            ('apply(groupby(["sex"]), expr="str(if( b_1 then year else 2000))" )', ['sex', 'b_1', 'year'],
             ['2002', '2003', '2000', '2000', '2002', '2003', '2000', '2000', '2004', '2001']),
            ('apply(groupby(["sex"]), string_expr="str(if( b_1 then year else 2000))" )', ['sex', 'b_1', 'year'],
             ['2002', '2003', '2000', '2000', '2002', '2003', '2000', '2000', '2004', '2001']),
        ]
        string_df = pd.DataFrame({'n_1': [1, 1, 1, 1, 2, 2, 2, 3, 3, 4],
                                  'sex': ['m', 'f', 'm', 'm', 'f', 'm', 'm', 'f', 'm', 'm'],
                                  'year': [2002, 2003, 2005, 2001, 2002, 2003, 2001, 2005, 2004, 2001],
                                  'b_1': [True, True, False, False, True, True, False, False, True, True],
                                  })
        copy_df = string_df.copy()
        f_name = "test"
        for expr, dependencies, res in params:
            with self.subTest():
                calculator = StringDataCalculator(feature_name=f_name, expression=expr)
                dep = calculator.compute_variables()
                self.assertEqual(len(dep), len(dependencies), msg=f"Expression: '{expr}'")
                self.assertEqual(set(dep), set(dependencies), msg=f"Expression: '{expr}'")
                cur_res = calculator.execute(in_df=string_df)
                pd.testing.assert_series_equal(cur_res, pd.Series(name=f_name, data=res))
                self.assertTrue(string_df.equals(copy_df), msg=f"Expression: '{expr}'")

    def test_PegConsistencyCheck(self) -> None:
        self.CheckCount(ConsistencyCheck(expression="TRUE"), count=0)
        self.CheckCount(ConsistencyCheck(expression="False"), count=4)
        self.CheckCount(ConsistencyCheck(expression="not ( sex == 'f' and year < 2003 )"), count=0)
        self.CheckCount(ConsistencyCheck(expression="not sex == 'f'"), count=1)
        self.CheckCount(ConsistencyCheck(expression="- abs( age) <= 0"), count=0)
        self.CheckCount(ConsistencyCheck(expression="100 - age > 0"), count=0)
        self.CheckCount(ConsistencyCheck(expression="sex=='m'"), count=2)
        self.CheckCount(ConsistencyCheck(expression='sex != "f"'), count=1)
        self.CheckCount(ConsistencyCheck(expression="notnull( sex )"), count=1)
        self.CheckCount(ConsistencyCheck(expression="ifthenelse(b_2 == 1 , FALSE, TRUE)"), count=1)

    def test_LessThanConsistencyCheck(self) -> None:
        indexes = LessThanConsistencyCheck(feat_1='year', op_1='   ', feat_2='year_2', op_2='').execute(test_df)
        self.assertEqual(len(indexes), 0)
        indexes = LessThanConsistencyCheck(feat_1='year', op_1='add 3', feat_2='year_2', op_2='').execute(test_df)
        self.assertEqual(len(indexes), 3)
        indexes_n = LessThanConsistencyCheck(feat_1='year', op_1='add 3', feat_2='year_2', op_2='',
                                             negate=True).execute(test_df)
        self.assertEqual(len(indexes_n), 1)
        indexes = LessThanConsistencyCheck(feat_1='age', op_1=['abs'], feat_2='age',
                                           op_2=['add 10', 'subtract 5']).execute(test_df)
        self.assertEqual(len(indexes), 1)
        indexes_n = LessThanConsistencyCheck(feat_1='age', op_1=['abs'], feat_2='age', op_2=['add 10', 'subtract 5'],
                                             negate=True).execute(test_df)
        self.assertEqual(len(indexes_n), 3)

        indexes = LessThanConsistencyCheck(feat_1='year_2', op_1='sub year', feat_2='b_3', op_2='').execute(test_df)
        self.assertEqual(len(indexes), 3)

    def test_AndConsistencyCheck(self) -> None:
        indexes = AndConsistencyCheck(feat_1='edu', op_1='notnull', feat_2='sex', op_2='equal "m"').execute(test_df)
        self.assertEqual(len(indexes), 2)
        indexes_n = AndConsistencyCheck(feat_1='edu', op_1='notnull', feat_2='sex', op_2='equal "m"',
                                        negate=True).execute(test_df)
        self.assertEqual(len(indexes_n), 2)

    def test_AllConsistencyCheck(self) -> None:
        indexes = AllConsistencyCheck(features=['b_1', 'b_2', 'b_3']).execute(test_df)
        self.assertEqual(len(indexes), 2)
        indexes_n = AllConsistencyCheck(features=['b_1', 'b_2', 'b_3'], negate=True).execute(test_df)
        self.assertEqual(len(indexes_n), 2)

    def test_AnyConsistencyCheck(self) -> None:
        indexes = AnyConsistencyCheck(features=['b_1', 'b_2', 'b_3']).execute(test_df)
        self.assertEqual(len(indexes), 1)
        indexes_n = AnyConsistencyCheck(features=['b_1', 'b_2', 'b_3'], negate=True).execute(test_df)
        self.assertEqual(len(indexes_n), 3)

    def CheckCount(self, check: ConsistencyCheck, count: int):
        indexes = check.execute(test_df)
        self.assertEqual(count, len(indexes))


if __name__ == '__main__':
    unittest.main()
