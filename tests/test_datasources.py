import json
import os
import unittest
from json.decoder import JSONDecodeError
from pathlib import Path
from shutil import copyfile
from typing import List, Union

import pandas as pd
from jsonschema.exceptions import ValidationError as JSONValidationError

import censyn.datasources as ds
import tests
from censyn.features import Feature


class TestDataSource(unittest.TestCase):
    # Bunch of useful paths.
    _parent = Path(__file__).parent.parent
    _assets_path = _parent / 'tests' / 'assets'
    _conf_path = _parent / 'conf'
    _data_path = _parent / 'data'

    def get_data_file(self, file_name: str) -> Union[None, Path]:
        if tests.nas_connection_available():
            nas_file = tests.get_nas_file(file_name)
            if nas_file:
                if os.path.exists(str(nas_file)):
                    return nas_file
        local_file = self._data_path / file_name
        return local_file if os.path.exists(local_file) else None

    def test_csv_datasource(self) -> None:
        """Test csv file as data source."""
        table_soc_cha_15_name = 'ACS_15_5YR_DP02_socCh.csv'
        table_1_path = str(self._assets_path / table_soc_cha_15_name)

        df = ds.DelimitedDataSource(path_to_file=table_1_path).to_dataframe()
        self.validate(df, 2, 611)

    @unittest.skipIf(not tests.nas_connection_available(), tests.NAS_UNAVAILABLE_MESSAGE)
    def test_parquet_datasource_nas(self) -> None:
        """Test Parquet file as data source."""
        # Parquet table file
        file_name = tests.get_nas_file('personsCA2016.parquet')

        df = ds.ParquetDataSource(path_to_file=str(file_name)).to_dataframe()
        self.validate(df, 376035, 284)

    def test_parquet_datasource(self) -> None:
        """Test Parquet file as data source."""
        data_file = self.get_data_file('personsIL2016.parquet')
        if not data_file:
            raise unittest.SkipTest(tests.NAS_UNAVAILABLE_MESSAGE)
        df = ds.ParquetDataSource(path_to_file=data_file).to_dataframe()
        self.validate(df, 126334, 284)

    def test_load_csv_datasource(self) -> None:
        """Test CSV file as data source."""
        data_file = str(self._assets_path / 'ACS_15_5YR_DP02_socCh.csv')
        df = ds.load_data_source(file_names=[str(data_file)])
        self.validate(df, 2, 611)

    def test_load_parquet_datasource(self) -> None:
        """Test Parquet file as data source."""
        data_file = self.get_data_file('personsIL2016.parquet')
        if not data_file:
            raise unittest.SkipTest(tests.NAS_UNAVAILABLE_MESSAGE)
        df = ds.load_data_source(file_names=[str(data_file)])
        self.validate(df, 126334, 284)

    def test_load_datasource_with_features(self) -> None:
        json_path = str(self._conf_path / 'features_PUMS-P.json')       # Load the features
        features = ds.FeatureJsonFileSource(json_path).feature_definitions
        feature_names = [f.feature_name for f in features]
        data_file = self.get_data_file('personsIL2016.parquet')
        if not data_file:
            raise unittest.SkipTest(tests.NAS_UNAVAILABLE_MESSAGE)
        df = ds.load_data_source(file_names=[str(data_file)], feature_names=feature_names)
        self.validate(df, 126334, 128)
        feature_names = [f.feature_name for f in features if f.feature_name in df.columns]
        for i in range(len(feature_names)):
            self.assertEqual(df.columns[i], feature_names[i])

    def validate(self, df: pd.DataFrame, rows: int, columns: int):
        """Validate a DataFrame's rows and columns"""
        # Validate DataFrame
        self.assertIsInstance(df, pd.DataFrame)

        # Validate the number of elements
        number_rows, number_columns = df.shape
        self.assertEqual(rows, number_rows)
        self.assertEqual(columns, number_columns)

    def test_pipe_delimited(self) -> None:
        """Tests a CSV, but with pipes."""
        mdf_unip_zip_file = 'mdf_unit.zip'
        test_file_location = str(self._assets_path / mdf_unip_zip_file)

        header = [
            'SCHEMA_TYPE_CODE', 'SCHEMA_BUILD_ID', 'TABBLKST', 'TABBLKCOU', 'ENUMDIST', 'EUID', 'RTYPE', 'GQTYPE',
            'TEN', 'VACS', 'FINAL_POP', 'HHT', 'HHT2', 'NPF', 'CPLT', 'UPART', 'MULTG', 'HHLDRAGE', 'HHSPAN', 'HHRACE',
            'PAOC', 'P18', 'P60', 'P65', 'P75'
        ]
        data = ds.DelimitedDataSource(path_to_file=test_file_location, header=header, row_count=1000,
                                      file_name_in_zip='MDF_UNIT.txt')
        df = data.to_dataframe()
        self.validate(df, 37, len(header))

        expected_json_feature_definitions = 284

        features = ds.FeatureJsonFileSource(path_to_file=test_file_location,
                                            file_name_in_zip='features.json').feature_definitions
        self.assertEqual(expected_json_feature_definitions, len(features))
        for feature in features:
            self.assertIsInstance(feature, Feature)
            
    def test_pickle_datasource(self) -> None:
        self._example_pickle = Path(__file__).parent.parent / 'tests' / 'assets' / 'example.pickle'
        expected_columns_shape = (20, 2)
        expected_all_shape = (20, 9)
        header = ['Name', 'College']
        
        pickle_datasource_columns = ds.PickleDataSource(path_to_file=self._example_pickle, columns=header)
        pickle_datasource_all = ds.PickleDataSource(path_to_file=self._example_pickle)
        
        self.assertEqual(expected_columns_shape, pickle_datasource_columns.to_dataframe().shape)
        self.assertEqual(expected_all_shape, pickle_datasource_all.to_dataframe().shape)

    def test_save_datasource(self) -> None:
        """Test saving a data source."""
        file_name = str(self._assets_path / 'ACS_15_5YR_DP02_socCh.csv')
        df = ds.load_data_source(file_names=[str(file_name)])
        self.validate(df, 2, 611)

        invalid_path = str(Path(self._assets_path / 'invalid' / 'test.parquet'))
        self.assertFalse(ds.validate_data_source(invalid_path))
        invalid_ds = str(Path(self._assets_path / 'test.paquet'))
        self.assertFalse(ds.validate_data_source(invalid_ds))
        save_ds = Path(self._assets_path / 'save_test.parquet')
        self.assertFalse(save_ds.exists())
        ds.save_data_source(file_name=str(save_ds), in_df=df)
        self.assertTrue(save_ds.exists())
        os.remove(save_ds)


class TestFeatureDefinitions(unittest.TestCase):
    """Tests the FeatureJsonDataSource class."""

    # Bunch of useful paths.
    _parent = Path(__file__).parent.parent
    _assets_path = _parent / 'tests' / 'assets'
    _conf_path = _parent / 'conf'
    _data_path = _parent / 'data'
    _json_definition_path = _conf_path / 'features_PUMS-P.json'
    _schema_path = _conf_path / 'features_PUMS-P_schema.json'

    def setUp(self) -> None:
        """Keep tabs on any files these tests create."""
        self._created_files: List[Path] = []

    def tearDown(self) -> None:
        """Remove any files that a test creates."""
        for path in self._created_files:
            if path.exists():
                path.unlink()

    def test_feature_import_valid(self) -> None:
        """Tests a variety of possible valid inputs to the FeatureJsonDataSource."""
        with open(self._json_definition_path) as w:
            decoded_json = json.loads(w.read())

        json_source = ds.FeatureJsonFileSource(path_to_file=self._json_definition_path)
        features = json_source.feature_definitions
        self.assertEqual(len(features), len(decoded_json))

        # Retrieve printable string for features as stored
        for feature in features:
            fstr = feature.__str__()
            self.assertIsNotNone(fstr)
            self.assertTrue(feature.feature_name in fstr)

        # Ensure that json root property names match feature names
        for fname, fdname in zip(
                sorted([feature.feature_name for feature in features]),
                sorted(decoded_json.keys())):
            self.assertEqual(fname, fdname)

        # Load copy with alternative schema
        alt_schema = ds.FeatureJsonFileSource(path_to_file=self._json_definition_path, path_to_schema=self._schema_path)
        alt_features = alt_schema.feature_definitions
        self.assertEqual(len(features), len(alt_features))
        for fname, afname in zip(
                sorted([feature.feature_name for feature in features]),
                sorted([feature.feature_name for feature in alt_features])):
            self.assertEqual(fname, afname)

    def test_feature_import_invalid(self) -> None:
        """Tests a variety of possible bad inputs to the FeatureJsonDataSource."""
        malformed_json: Path = self._assets_path / 'ACS_15_5YR_DP02_socCh.json'
        copyfile(self._assets_path / 'ACS_15_5YR_DP02_socCh.csv', malformed_json)
        self._created_files.append(malformed_json)

        bad_extension: Path = Path('30b78f72-22a9-40df-b83e-4cc344faf971.txt')
        alt_json_path = self._assets_path / 'features_copy.json'

        # wrong extension
        with self.assertRaises(ValueError):
            bad_extension.touch()
            ds.FeatureJsonFileSource(path_to_file=str(bad_extension), path_to_schema=str(self._schema_path))
        bad_extension.unlink()

        # Try loading a file that doesn't exist
        with self.assertRaises(FileNotFoundError):
            ds.FeatureJsonFileSource(path_to_file='30b78f72-22a9-40df-b83e-4cc344faf971.json')

        # Open FeatureJsonFileSource without corresponding schema
        with self.assertRaises(FileNotFoundError):
            ds.FeatureJsonFileSource(path_to_file=alt_json_path)

        # Try alternative way of validating schema
        with self.assertRaises(FileNotFoundError):
            ds.FeatureJsonFileSource(path_to_file=self._json_definition_path,
                                     path_to_schema='not_the_right_schema.json')

        # Load something that is clearly not a json file
        with self.assertRaises(JSONDecodeError):
            ds.FeatureJsonFileSource(path_to_file=malformed_json, path_to_schema=self._schema_path)

        # Load a schema file that exists, but does not contain a valid schema
        with self.assertRaises(JSONDecodeError):
            ds.FeatureJsonFileSource(path_to_file=self._json_definition_path, path_to_schema=malformed_json)

        # Try to load an invalid JSON file
        with self.assertRaises(JSONValidationError):
            ds.FeatureJsonFileSource(path_to_file=self._schema_path, path_to_schema=self._schema_path)

        # Use an invalid schema
        with self.assertRaises(JSONValidationError):
            ds.FeatureJsonFileSource(path_to_file=self._json_definition_path, path_to_schema=self._json_definition_path)


if __name__ == '__main__':
    unittest.main()
