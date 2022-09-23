import json
import logging
import sys
from pathlib import Path
from typing import Dict, List, Union

from jsonschema import validate as json_validate
from jsonschema.exceptions import ValidationError as JSONValidationError

from .filesource import FileSource
from censyn.encoder import Binner
from censyn.features import Feature, FeatureType, ModelChoice, ModelSpec
from censyn.utils import get_class


class JsonFileSource(FileSource):
    """Abstract class for handling JSON files."""

    @FileSource.validate_extension(extensions='.json')
    def __init__(self, *args, path_to_schema: str = None, **kwargs) -> None:
        """
        Construct a JsonFileSource.

        If path_to_file is a zip, path_to_schema should either be left None or set to the name of the schema filename
        inside the zip. If not given, it will be assumed to be in the same directory with '_schema' appended to the name
        of the definitions file, e.g. if the definitions file is './conf/features.json', the schema will be
        './conf/features_schema.json'.

        :param: path_to_schema: The path to the json schema used to validate the json.
        """
        super().__init__(*args, **kwargs)
        logging.debug(msg=f"Instantiate JsonFileSource with path_to_schema {path_to_schema}.")

        self._schema = None
        self._path_to_schema = path_to_schema

    @property
    def schema(self) -> Dict:
        """Returns the schema, reading it first if necessary."""
        logging.debug(msg=f"JsonFileSource.schema().")
        if self._schema is None:
            schema_file = None
            try:
                schema_file = self.zipped_file.open(self.path_to_schema) if self.zipped_file else \
                    open(self.path_to_schema)
                read_schema = json.loads(schema_file.read())
                if '$schema' not in read_schema:
                    raise JSONValidationError(f'Decoded json [ {self.path_to_schema} ] not labeled as a json schema!')
                self._schema = read_schema
            except IOError as e:
                raise type(e)(f'{str(e)} Unable to open schema file [ {self.path_to_schema} ]!\n'
                              ).with_traceback(sys.exc_info()[2])
            except json.decoder.JSONDecodeError as e:
                raise json.decoder.JSONDecodeError(
                    msg=f'{str(e)} Unable to decode file [ {self.path_to_file} ] as json',
                    pos=e.pos, doc=e.doc).with_traceback(sys.exc_info()[2])
            finally:
                if schema_file:
                    schema_file.close()
        return self._schema

    @property
    def path_to_schema(self) -> Path:
        """Gets the path to the schema file, constructing it first if necessary."""
        logging.debug(msg=f"JsonFileSource.path_to_schema().")
        if self._path_to_schema is None:
            if self.zipped_file:
                zipped_path: Path = Path(self.file_name_in_zip)
                self._path_to_schema = f'{zipped_path.stem}_schema{zipped_path.suffix}'
            else:
                json_path: Path = Path(self.path_to_file)
                schema_name = f'{json_path.stem}_schema{json_path.suffix}'
                self._path_to_schema = json_path.parent / schema_name
        return self._path_to_schema


class FeatureJsonFileSource(JsonFileSource):
    """Extracts List[Feature] from feature definition JSON."""

    def __init__(self, *args, **kwargs) -> None:
        """
        Sets up to read features and reads the feature definitions.

        :raise: json.decoder.JSONDecodeError if the json cannot be read.
        """
        super().__init__(*args, **kwargs)
        logging.debug(msg=f"Instantiate FeatureJsonFileSource.")

        try:
            self._feature_json = json.loads(self.file_handle.read())
        except json.decoder.JSONDecodeError as e:
            raise json.decoder.JSONDecodeError(
                msg=f'{str(e)} Unable to decode file [ {self.path_to_file} ] as json',
                pos=e.pos, doc=e.doc).with_traceback(sys.exc_info()[2])
        try:
            json_validate(self._feature_json, self.schema)
        except json.decoder.JSONDecodeError as e:
            raise json.decoder.JSONDecodeError(
                msg=f'{str(e)} Unable to validate definition file against schema!\n',
                pos=e.pos, doc=e.doc).with_traceback(sys.exc_info()[2])
        self._feature_definitions = self.compile_features(self._feature_json)

    def compile_features(self, feature_dict: dict) -> List[Feature]:
        """
        Compile a list of Feature class instances from a JSON definition dictionary

        :param: feature_dict: A dictionary of column definitions, one per root entry, identified by PUMS data dictionary
               column name. Each such entry contains type, field length, density, binning strategy, and bin definitions
        :return: A list of features, List[Feature], where each Feature instance corresponds to an entry in the JSON
                 definition dictionary
        """
        logging.debug(msg=f"FeatureJsonFileSource.compile_features(). feature_dict {feature_dict}")
        # There's no special complex type that needs to be extracted from the JSON, so parsing with default loader is
        # fine for now However, there's a case to be made for directly serializing Features into JSON, but right now
        # they do not support all the information that the spreadsheet extracted version does at the moment
        return [self._create_feature(name=f_name, feature_dict=f_dict) for (f_name, f_dict) in feature_dict.items()]

    @staticmethod
    def _create_feature(name: str, feature_dict: dict) -> Feature:
        """
        Create a Feature from the JSON definition dictionary.

        :param: name: Name of the feature
        :param: feature_dict: JSON dictionary of the feature
        :return: The created Feature
        """
        logging.debug(msg=f"FeatureJsonFileSource._create_feature(). name {name} and feature_dict {feature_dict}")
        # Note: Existence and error checking for JSON are handled in validation
        f_type = FeatureType[feature_dict['feature_type']]
        f_format = feature_dict.get('feature_format', None)
        if f_format:
            if isinstance(f_format, List):
                f_format = " ".join(f_format)

        # No binning strategy is required since we've manually defined all the bins we need
        binner: Union[Binner, None] = None
        if feature_dict.get('bins') is not None:
            is_numeric = f_type == FeatureType.integer or f_type == FeatureType.floating_point
            binner = Binner(is_numeric=is_numeric, mapping=feature_dict['bins'])

        model_type = None
        if feature_dict.get('model_info') is not None:
            model_info = feature_dict.get('model_info')
            model_type = ModelSpec(model=ModelChoice[model_info['model_name']],
                                   model_params=model_info.get('model_params'))

        encoder = None
        if feature_dict.get('encoder') is not None:
            encoder_info = feature_dict.get('encoder')
            encoder_class = encoder_info['encoder_type']
            encoder_param = encoder_info.get('encoder_param')
            encoder = get_class(encoder_class, encoder_param)

        dependencies = feature_dict.get('dependencies')
        exclude_dependencies = feature_dict.get('exclude_dependencies')
        return Feature(name, f_type, feature_format=f_format, binner=binner, model_type=model_type, encoder=encoder,
                       dependencies=dependencies, exclude_dependencies=exclude_dependencies)

    @property
    def feature_json(self) -> Dict:
        """returns a json dictionary of Features."""
        return self._feature_json

    @property
    def feature_definitions(self) -> List[Feature]:
        """returns a parsed list of Features."""
        return self._feature_definitions
