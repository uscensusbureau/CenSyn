import dill
import os
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np
import pandas as pd

from censyn.experiments.experiment_generator import ExperimentGenerator
from censyn.features import Feature, FeatureType
from censyn.features import ModelChoice, ModelSpec
from censyn.models import NoopModel, DecisionTreeModel, DecisionTreeRegressorModel
from censyn.encoder import IdentityEncoder, NumericalEncoder
from censyn.experiments import Experiment

test_df = pd.DataFrame({'NAME': ['Bob', 'Alice', 'Joe', 'Chris'],
                        'AGE': [28, 43, 12, 7],
                        'ICB': [2, 3, 1, 0]})

report_cfg = {
    "report_level": "FULL"
}


class TestExperiment(unittest.TestCase):
    def setUp(self) -> None:
        name_model_spec = ModelSpec(model=ModelChoice.DecisionTreeModel,
                                    model_params={'max_depth': 10, 'criterion': 'squared_error',
                                                  'min_impurity_decrease': 1e-5})
        income_model_spec = ModelSpec(model=ModelChoice.NoopModel)
        age_model_spec_1 = ModelSpec(model=ModelChoice.DecisionTreeRegressorModel,
                                     model_params={'max_depth': 10, 'criterion': 'squared_error',
                                                   'min_impurity_decrease': 1e-5})

        mapping = {'Bob': 1, 'Alice': 2, 'Joe': 3, 'Chris': 4}

        self._income_f = Feature(feature_name='ICB', feature_type=FeatureType.integer,
                                 encoder=IdentityEncoder(column='income_bracket', indicator=True, inplace=False),
                                 model_type=income_model_spec)
        self._name_f = Feature(feature_name='NAME', feature_type=FeatureType.obj,
                               encoder=NumericalEncoder(column='name', mapping=mapping, alpha=0.5,
                                                        indicator=True, inplace=False),
                               dependencies=['ICB'],
                               model_type=name_model_spec)
        self._age_f = Feature(feature_name='AGE', feature_type=FeatureType.integer,
                              encoder=IdentityEncoder(column='age', indicator=True, inplace=True),
                              model_type=age_model_spec_1,
                              dependencies=['ICB', 'NAME'])

        # This is to test latter to make sure that we got the expected modal types as well as the dependencies
        # are resolved correctly.
        self._model_expectation = {
            'NAME': DecisionTreeModel,
            'ICB': NoopModel,
            'AGE': DecisionTreeRegressorModel
        }
        self._feature_dependency_expectation = {
            'NAME': ['ICB'],
            'ICB': [],
            'AGE': ['ICB', 'NAME']
        }

    def test_dependencies(self) -> None:
        models = {
            'FEAT1': DecisionTreeModel(target_feature=Feature(feature_name='FEAT1',
                                                              feature_type=FeatureType.integer,
                                                              dependencies=['FEAT2'],
                                                              model_type=ModelSpec(model=ModelChoice.DecisionTreeModel),
                                                              encoder=IdentityEncoder(column='FEAT1', indicator=True,
                                                                                      inplace=False))),
            'FEATNO_1': NoopModel(target_feature=Feature(feature_name='FEATNO_1',
                                                         feature_type=FeatureType.integer,
                                                         dependencies=[],
                                                         model_type=ModelSpec(model=ModelChoice.NoopModel),
                                                         encoder=IdentityEncoder(column='FEATNO_1', indicator=True,
                                                                                 inplace=False))),
            'FEAT2': DecisionTreeModel(target_feature=Feature(feature_name='FEAT2',
                                                              feature_type=FeatureType.integer,
                                                              dependencies=['FEAT3', 'FEAT5'],
                                                              model_type=ModelSpec(model=ModelChoice.DecisionTreeModel),
                                                              encoder=IdentityEncoder(column='FEAT2', indicator=True,
                                                                                      inplace=False))),
            'FEAT3': DecisionTreeModel(target_feature=Feature(feature_name='FEAT3',
                                                              feature_type=FeatureType.integer,
                                                              dependencies=['FEAT2'],
                                                              model_type=ModelSpec(model=ModelChoice.DecisionTreeModel),
                                                              encoder=IdentityEncoder(column='FEAT3', indicator=True,
                                                                                      inplace=False))),
            'FEATNO_2': NoopModel(target_feature=Feature(feature_name='FEATNO_2',
                                                         feature_type=FeatureType.obj,
                                                         dependencies=[],
                                                         model_type=ModelSpec(model=ModelChoice.NoopModel),
                                                         encoder=IdentityEncoder(column='FEATNO_2', indicator=True,
                                                                                 inplace=False))),
            'FEAT4': DecisionTreeModel(target_feature=Feature(feature_name='FEAT4',
                                                              feature_type=FeatureType.obj,
                                                              dependencies=[],
                                                              model_type=ModelSpec(model=ModelChoice.DecisionTreeModel),
                                                              encoder=IdentityEncoder(column='FEAT4', indicator=True,
                                                                                      inplace=False))),
            'FEAT5': DecisionTreeModel(target_feature=Feature(feature_name='FEAT5',
                                                              feature_type=FeatureType.integer,
                                                              dependencies=['FEAT1'],
                                                              model_type=ModelSpec(model=ModelChoice.DecisionTreeModel),
                                                              encoder=IdentityEncoder(column='FEAT5', indicator=True,
                                                                                      inplace=False)))
        }

        experiment = Experiment(experiment_name="test", experiment_models=models)
        with self.assertRaises(ValueError):
            experiment.verify_dependencies(bootstrap_f=['FEAT1'])

        with self.assertRaises(ValueError):
            experiment.verify_dependencies(bootstrap_f=['FEAT3'])

        experiment.verify_dependencies(bootstrap_f=['FEAT1', 'FEAT3'])
        self.assertEqual(6, len(experiment.experiment_models['FEAT4'].target_feature.train_dependencies))

        data_df = pd.DataFrame(data=[[0, np.nan, 1, 2, '2', '3', np.nan],
                                     [0, np.nan, 1, 0, '2', '0', np.nan],
                                     [0, np.nan, 1, 0, '5', '0', np.nan],
                                     [0, np.nan, 1, 1, '2', '0', np.nan],
                                     [0, np.nan, 1, 0, '6', '0', np.nan]],
                               columns=["FEAT1", "FEATNO_1", "FEAT2", "FEAT3", 'FEATNO_2', "FEAT4", "FEAT5"])
        experiment.bootstrap_features = ['FEAT1', 'FEAT3']
        out_df = experiment.run(data_df=data_df, external_synth_df=pd.DataFrame(), processes=1)
        self.assertEqual(data_df.shape, out_df.shape)

    def test_experiment_generator(self) -> None:
        # I am not sure about how far we should go with testing this.
        expected_experiments = 1
        experiment_generator = ExperimentGenerator(feature_spec={self._name_f.feature_name: self._name_f,
                                                   self._income_f.feature_name: self._income_f,
                                                   self._age_f.feature_name: self._age_f})

        all_experiments = experiment_generator.generate()
        self.assertEqual(len(all_experiments), expected_experiments, f'Did not get the expected number of '
                                                                     f'experiments {expected_experiments}')

        for idx, experiment in enumerate(all_experiments):
            self.assertIsInstance(experiment, Experiment)
            experiment_models = experiment.experiment_models
            for key, model in experiment_models.items():
                self.assertIsInstance(model, self._model_expectation[key])
                self.assertEqual(model.target_feature.dependencies, self._feature_dependency_expectation[key])

            with TemporaryDirectory() as temp_dir:
                file_path = Path(temp_dir).joinpath('experiment.json')
                experiment.pickle_experiment_models(file_path=str(file_path))
                self.assertTrue(os.path.isfile(file_path))

                with open(file_path, 'rb') as dill_file:
                    un_pickled_models = dill.load(dill_file)

                    for key, model in un_pickled_models.items():
                        self.assertIsInstance(model, self._model_expectation[key])
                        self.assertEqual(model.target_feature.dependencies, self._feature_dependency_expectation[key])
