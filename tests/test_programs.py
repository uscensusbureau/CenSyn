import os
import shutil
import unittest
from pathlib import Path
from typing import Union

import pandas as pd

import tests
import censyn.results as res
from censyn.__main__ import command_line_start
from censyn.programs.censyn import Censyn
from censyn.datasources import load_data_source, FeatureJsonFileSource


class TestPrograms(unittest.TestCase):
    out_path = './tests/output'
    _convert = None
    _synthesizer = None

    @classmethod
    def setUpClass(cls) -> None:
        isdir = os.path.isdir(cls.out_path)
        if not isdir:
            os.makedirs(cls.out_path, exist_ok=True)

    @classmethod
    def tearDownClass(cls) -> None:
        isdir = os.path.isdir(cls.out_path)
        if isdir:
            shutil.rmtree(cls.out_path, ignore_errors=True)

    @classmethod
    def runConvert(cls) -> Censyn:
        if not cls._convert:
            data_file = "data/personsIL2016.parquet"
            if not cls.get_data_file(data_file):
                raise unittest.SkipTest(f"Can not find {data_file}")
            cls._convert = Censyn('--convert_config_file ./tests/conf/convert.cfg')
            cls._convert.execute()
        return cls._convert

    @classmethod
    def runSynthesis(cls) -> Censyn:
        if not cls._synthesizer:
            cls._synthesizer = Censyn('--synthesize_config_file ./tests/conf/synthesize.cfg')
            cls._synthesizer.execute()
        return cls._synthesizer

    @staticmethod
    def get_data_file(file_name: str) -> Union[None, Path]:
        if tests.nas_connection_available():
            nas_file = tests.get_nas_file(file_name)
            if nas_file:
                if os.path.exists(str(nas_file)):
                    return nas_file
        return file_name if os.path.exists(file_name) else None

    def validateMarginalResult(self, eval_r: res.Result, raw: float, calc: float,
                               bl_raw: float, bl_calc: float) -> None:
        self.assertTrue(isinstance(eval_r, res.ListResult))
        test = 0
        for cur_r in eval_r.value:
            if isinstance(cur_r, res.FloatResult):
                pass
            elif isinstance(cur_r, res.MappingResult):
                if cur_r.description.startswith('Summary'):
                    self.assertAlmostEqual(cur_r.value['Raw Marginal Score'], raw, 5)
                    self.assertAlmostEqual(cur_r.value['Calculated Marginal Score'], calc, 3)
                    test += 1
                if cur_r.description.startswith('Baseline'):
                    self.assertAlmostEqual(cur_r.value['Baseline Raw Marginal Score'], bl_raw, 5)
                    self.assertAlmostEqual(cur_r.value['Baseline Calculated Marginal Score'], bl_calc, 3)
                    test += 1
                elif cur_r.description == 'frequent_itemset':
                    pass
            else:
                pass
        self.assertEqual(2, test)

    def validateCellDifferenceResult(self, eval_r: res.Result) -> None:
        self.assertTrue(isinstance(eval_r, res.ListResult))
        test = 0
        for cur_r in eval_r.value:
            if isinstance(cur_r, res.FloatResult):
                pass
            elif isinstance(cur_r, res.MappingResult):
                if cur_r.description.startswith('Summary'):
                    self.assertEqual(cur_r.value['Total Differences'], 90326)
                    self.assertAlmostEqual(cur_r.value['Total Differences %'], 22.67235, 3)
                    self.assertAlmostEqual(cur_r.value['Average number of differences per row'], 27.88700, 3)
                    test += 1
                else:
                    pass
            else:
                pass
        self.assertEqual(1, test)

    def test_command_line_start(self) -> None:
        command_line_start()

    def test_no_process(self) -> None:
        """Create a Censyn with no process. Execute will print a help while results will throw a runtime error."""
        with self.assertRaises(RuntimeError):
            process = Censyn()
            process.execute()
            process.results()

    def test_convert(self) -> None:
        self.runConvert()
        data_path = "./tests/output/small_personsIL2016.parquet"
        out_df = pd.read_parquet(path=data_path)
        shape = out_df.shape
        self.assertEqual(3239, shape[0])
        self.assertEqual(284, shape[1])

    def test_synthesis(self) -> None:
        self.runConvert()
        syn_process = self.runSynthesis()
        syn_results = syn_process.results()
        syn_res = syn_results["Synthesized features"]
        self.assertEqual(syn_res.value, ['DDRS', 'DEAR', 'DEYE', 'DOUT', 'DPHY', 'DRAT', 'DRATX', 'DREM',
                                         'ENG', 'FER', 'GCL', 'GCM', 'GCR', 'HINS1', 'HINS2', 'HINS3',
                                         'HINS4', 'HINS5', 'HINS6', 'HINS7', 'INTP', 'JWMNP', 'JWRIP',
                                         'JWTR', 'LANP', 'LANX', 'MAR', 'MARHD', 'MARHM', 'MARHT', 'MARHW',
                                         'MARHYP', 'MIG', 'MIL', 'MLPA', 'MLPB', 'MLPCD', 'MLPE', 'MLPFG',
                                         'MLPH', 'MLPI', 'MLPJ', 'MLPK', 'NWAB', 'NWAV', 'NWLA', 'NWLK',
                                         'NWRE', 'OIP', 'PAP', 'RELP', 'RETP', 'SCH', 'SCHG', 'SCHL',
                                         'SEMP', 'SEX', 'SSIP', 'SSP', 'WAGP', 'WKHP', 'WKL', 'WKW', 'WRK',
                                         'YOEP', 'ANC', 'ANC1P', 'ANC2P', 'DECADE', 'DIS', 'DRIVESP', 'ESP',
                                         'ESR', 'FOD1P', 'FOD2P', 'HICOV', 'HISP', 'INDP', 'JWAP', 'JWDP',
                                         'MIGPUMA', 'MIGSP', 'MSP', 'NAICSP', 'NATIVITY', 'NOP', 'OC', 'OCCP', 'PAOC',
                                         'PERNP', 'PINCP', 'POBP', 'POVPIP', 'POWPUMA', 'POWSP', 'PRIVCOV',
                                         'PUBCOV', 'QTRBIR', 'RAC1P', 'RAC2P', 'RAC3P', 'RACAIAN', 'RACASN',
                                         'RACBLK', 'RACNH', 'RACNUM', 'RACPI', 'RACSOR', 'RACWHT', 'RC',
                                         'SCIENGP', 'SCIENGRLP', 'SFN', 'SFR', 'SOCP', 'VPS', 'WAOB', 'PUMA',
                                         'AGEP', 'CIT', 'CITWP', 'COW'])
        self.assertIsNone(syn_results.get("Consistency check MLPJ_AGE", None))
        self.assertIsNone(syn_results.get("Consistency check MLPI_AGE", None))
        self.assertIsNone(syn_results.get("Consistency check MLPK_AGE", None))
        syn_res = syn_results.get("Consistency check MLPH_AGE", None)
        self.assertEqual(1, len(syn_res.value))
        syn_res = syn_results.get("Consistency check MLPFG_AGE", None)
        self.assertEqual(6, len(syn_res.value))
        syn_res = syn_results.get("Consistency check MLPE_AGE", None)
        self.assertEqual(11, len(syn_res.value))
        syn_res = syn_results.get("Consistency check MLPCD_AGE", None)
        self.assertEqual(3, len(syn_res.value))
        syn_res = syn_results.get("Consistency check MLPB_AGE", None)
        self.assertEqual(1, len(syn_res.value))

        eval_process = Censyn('--eval_config_file ./tests/conf/eval.cfg')
        eval_process.execute()
        eval_results = eval_process.results()
        self.validateMarginalResult(eval_r=eval_results['marginal_metric'], raw=0.1493947, calc=925.302604,
                                    bl_raw=0.1182594, bl_calc=940.870294)
        self.validateCellDifferenceResult(eval_r=eval_results['cell_difference_metric'])

    def test_partialSynthesis(self) -> None:
        self.runConvert()
        syn_process = Censyn('--synthesize_config_file ./tests/conf/synthesize_partial.cfg')
        syn_process.execute()
        syn_results = syn_process.results()
        for k, result in syn_results.items():
            if isinstance(result, res.FloatResult):
                pass
            elif isinstance(result, res.StrResult):
                if result.metric_name == "Synthesized features":
                    self.assertEqual(result.value, ['AGEP',  'SCHL', 'RAC1P'])
            elif isinstance(result, res.ModelResult):
                pass
            elif isinstance(result, res.IndexResult):
                if result.metric_name == "Consistency check MLPK_AGE":
                    self.assertEqual(0, len(result.value))
                elif result.metric_name == "Consistency check MLPJ_AGE":
                    self.assertEqual(0, len(result.value))
                elif result.metric_name == "Consistency check MLPI_AGE":
                    self.assertEqual(0, len(result.value))
                elif result.metric_name == "Consistency check MLPH_AGE":
                    self.assertEqual(5, len(result.value))
                elif result.metric_name == "Consistency check MLPFG_AGE":
                    self.assertEqual(6, len(result.value))
                elif result.metric_name == "Consistency check MLPE_AGE":
                    self.assertEqual(5, len(result.value))
                elif result.metric_name == "Consistency check MLPCD_AGE":
                    self.assertEqual(3, len(result.value))
                elif result.metric_name == "Consistency check MLPB_AGE":
                    self.assertEqual(0, len(result.value))
            else:
                pass

        eval_process = Censyn('--eval_config_file ./tests/conf/eval_partial.cfg')
        eval_process.execute()
        eval_results = eval_process.results()
        self.validateMarginalResult(eval_r=eval_results['marginal_metric AGEP'], raw=0.0871171, calc=956.441407,
                                    bl_raw=0.1848742, bl_calc=907.562889)
        self.validateMarginalResult(eval_r=eval_results['marginal_metric SCHL'], raw=0.1476000, calc=926.199989,
                                    bl_raw=0.2389660, bl_calc=880.516983)
        self.validateMarginalResult(eval_r=eval_results['marginal_metric RAC1P'], raw=0.0032289, calc=998.385541,
                                    bl_raw=0.1180232, bl_calc=940.988393)

    def test_external_synthesis(self) -> None:
        self.runConvert()
        syn_process = Censyn('--synthesize_config_file ./tests/conf/synthesize_external.cfg')
        syn_process.execute()
        syn_results = syn_process.results()
        for k, result in syn_results.items():
            if isinstance(result, res.FloatResult):
                pass
            elif isinstance(result, res.StrResult):
                if result.metric_name == "Synthesized features":
                    self.assertEqual(result.value, ['DDRS', 'DEAR', "DEYE", "DOUT", "DPHY", "DRAT", "DRATX", "DREM",
                                                    "ENG", "FER", "GCL", "GCM", "GCR", "HINS1", "HINS2", "HINS3",
                                                    "HINS4", "HINS5", "HINS6", "HINS7", "INTP", "JWMNP", "JWRIP",
                                                    "JWTR", "LANP", "LANX", "MAR", "MARHD", "MARHM", "MARHT", "MARHW",
                                                    "MARHYP", "MIG", "MIL", "MLPA", "MLPB", "MLPCD", "MLPE", "MLPFG",
                                                    "MLPH", "MLPI", "MLPJ", "MLPK", "NWAB", "NWAV", "NWLA",
                                                    "NWLK", "NWRE", "OIP", "PAP",
                                                    "RELP", "RETP", "SCH", "SCHG", "SCHL", "SEMP", "SEX",
                                                    "SSIP", "SSP", "WAGP", "WKHP",
                                                    "WKL", "WKW", "WRK", "YOEP", "ANC", "ANC1P", "ANC2P",
                                                    "DECADE", "DIS", "DRIVESP", "ESP",
                                                    "ESR", "FOD1P", "FOD2P", "HICOV", "HISP", "INDP", "JWAP",
                                                    "JWDP",  "MIGPUMA", "MIGSP", "MSP", "NAICSP",
                                                    "NATIVITY", "NOP", "OC", "OCCP", "PAOC", "PERNP", "PINCP",
                                                    "POBP", "POVPIP", "POWPUMA", "POWSP",
                                                    "PRIVCOV", "PUBCOV", "QTRBIR", "RAC1P", "RAC2P", "RAC3P",
                                                    "RACAIAN", "RACASN", "RACBLK",
                                                    "RACNH",  "RACNUM", "RACPI", "RACSOR", "RACWHT", "RC",
                                                    "SCIENGP", "SCIENGRLP", "SFN",
                                                    "SFR", "SOCP", "VPS", "WAOB"])
            elif isinstance(result, res.ModelResult):
                pass
            elif isinstance(result, res.IndexResult):
                pass
            else:
                pass

        eval_process = Censyn('--eval_config_file ./tests/conf/eval_external.cfg')
        eval_process.execute()
        eval_results = eval_process.results()
        self.validateMarginalResult(eval_r=eval_results['marginal_metric'], raw=0.2448868, calc=877.556599,
                                    bl_raw=0.1182594, bl_calc=940.870294)

    def test_external_synthesis_error(self) -> None:
        with self.assertRaises(ValueError):
            self.runConvert()
            syn_process = Censyn('--synthesize_config_file ./tests/conf/synthesize_external_error.cfg')
            syn_process.execute()

    def test_external_synthesis_error_1(self) -> None:
        with self.assertRaises(ValueError):
            self.runConvert()
            syn_process = Censyn('--synthesize_config_file ./tests/conf/synthesize_external_error_1.cfg')
            syn_process.execute()

    def test_external_synthesis_error_2(self) -> None:
        with self.assertRaises(ValueError):
            self.runConvert()
            syn_process = Censyn('--synthesize_config_file ./tests/conf/synthesize_external_error_2.cfg')
            syn_process.execute()

    def test_join(self) -> None:
        # make sure synthesize data is generated.
        data_file = "data/housesIL2016.parquet"
        if not self.get_data_file(data_file):
            raise unittest.SkipTest(f"Can not find {data_file}")

        self.runConvert()
        self.runSynthesis()

        # Join ground truth data
        join_process = Censyn('--join_config_file ./tests/conf/join_gt.cfg')
        join_process.execute()

        # Join synthesize data
        join_process = Censyn('--join_config_file ./tests/conf/join.cfg')
        join_process.execute()
        join_results = join_process.results()
        for k, result in join_results.items():
            if isinstance(result, res.FloatResult):
                pass
            else:
                pass

        eval_process = Censyn('--eval_config_file ./tests/conf/eval_join.cfg')
        eval_process.execute()
        eval_results = eval_process.results()
        eval_r = eval_results['join_marginal_metric']
        self.assertTrue(isinstance(eval_r, res.ListResult))
        self.validateMarginalResult(eval_r=eval_r.value[0], raw=0.9652049, calc=517.397534,
                                    bl_raw=0.3929016, bl_calc=803.549195)

    def test_join_persons(self) -> None:
        self.runConvert()

        # Join data
        join_process = Censyn('--join_config_file ./tests/conf/join_persons.cfg')
        join_process.execute()

        join_results = join_process.results()
        names_res = join_results.get('Names', None)
        self.assertIsNotNone(names_res)
        names_str = (f'"RT_1", "SERIALNO", "SPORDER_1", "PUMA", "ST", "ADJINC_1", "PWGTP_1", "AGEP_1", "CIT_1", '
                     f'"CITWP_1", "COW_1", "DDRS_1", "DEAR_1", "DEYE_1", "DOUT_1", "DPHY_1", "DRAT_1", "DRATX_1", '
                     f'"DREM_1", "ENG_1", "FER_1", "GCL_1", "GCM_1", "GCR_1", "HINS1_1", "HINS2_1", "HINS3_1", '
                     f'"HINS4_1", "HINS5_1", "HINS6_1", "HINS7_1", "INTP_1", "JWMNP_1", "JWRIP_1", "JWTR_1", '
                     f'"LANP_1", "LANX_1", "MAR_1", "MARHD_1", "MARHM_1", "MARHT_1", "MARHW_1", "MARHYP_1", '
                     f'"MIG_1", "MIL_1", "MLPA_1", "MLPB_1", "MLPCD_1", "MLPE_1", "MLPFG_1", "MLPH_1", "MLPI_1", '
                     f'"MLPJ_1", "MLPK_1", "NWAB_1", "NWAV_1", "NWLA_1", "NWLK_1", "NWRE_1", "OIP_1", "PAP_1", '
                     f'"RELP_1", "RETP_1", "SCH_1", "SCHG_1", "SCHL_1", "SEMP_1", "SEX_1", "SSIP_1", "SSP_1", '
                     f'"WAGP_1", "WKHP_1", "WKL_1", "WKW_1", "WRK_1", "YOEP_1", "ANC_1", "ANC1P_1", "ANC2P_1", '
                     f'"DECADE_1", "DIS_1", "DRIVESP_1", "ESP_1", "ESR_1", "FOD1P_1", "FOD2P_1", "HICOV_1", "HISP_1", '
                     f'"INDP_1", "JWAP_1", "JWDP_1", "MIGPUMA_1", "MIGSP_1", "MSP_1", "NAICSP_1", "NATIVITY_1", '
                     f'"NOP_1", "OC_1", "OCCP_1", "OCCPINC_1", "PAOC_1", "PERNP_1", "PINCP_1", "POBP_1", "POVPIP_1", '
                     f'"POWPUMA_1", "POWSP_1", "PRIVCOV_1", "PUBCOV_1", "QTRBIR_1", "RAC1P_1", "RAC2P_1", "RAC3P_1", '
                     f'"RACAIAN_1", "RACASN_1", "RACBLK_1", "RACNH_1", "RACNUM_1", "RACPI_1", "RACSOR_1", "RACWHT_1", '
                     f'"RC_1", "SCIENGP_1", "SCIENGRLP_1", "SFN_1", "SFR_1", "SOCP_1", "VPS_1", "WAOB_1", "RT_2", '
                     f'"SPORDER_2", "ADJINC_2", "PWGTP_2", "AGEP_2", "CIT_2", "CITWP_2", "COW_2", "DDRS_2", "DEAR_2", '
                     f'"DEYE_2", "DOUT_2", "DPHY_2", "DRAT_2", "DRATX_2", "DREM_2", "ENG_2", "FER_2", "GCL_2", '
                     f'"GCM_2", "GCR_2", "HINS1_2", "HINS2_2", "HINS3_2", "HINS4_2", "HINS5_2", "HINS6_2", "HINS7_2", '
                     f'"INTP_2", "JWMNP_2", "JWRIP_2", "JWTR_2", "LANP_2", "LANX_2", "MAR_2", "MARHD_2", "MARHM_2", '
                     f'"MARHT_2", "MARHW_2", "MARHYP_2", "MIG_2", "MIL_2", "MLPA_2", "MLPB_2", "MLPCD_2", "MLPE_2", '
                     f'"MLPFG_2", "MLPH_2", "MLPI_2", "MLPJ_2", "MLPK_2", "NWAB_2", "NWAV_2", "NWLA_2", "NWLK_2", '
                     f'"NWRE_2", "OIP_2", "PAP_2", "RELP_2", "RETP_2", "SCH_2", "SCHG_2", "SCHL_2", "SEMP_2", '
                     f'"SEX_2", "SSIP_2", "SSP_2", "WAGP_2", "WKHP_2", "WKL_2", "WKW_2", "WRK_2", "YOEP_2", "ANC_2", '
                     f'"ANC1P_2", "ANC2P_2", "DECADE_2", "DIS_2", "DRIVESP_2", "ESP_2", "ESR_2", "FOD1P_2", "FOD2P_2", '
                     f'"HICOV_2", "HISP_2", "INDP_2", "JWAP_2", "JWDP_2", "MIGPUMA_2", "MIGSP_2", "MSP_2", '
                     f'"NAICSP_2", "NATIVITY_2", "NOP_2", "OC_2", "OCCP_2", "OCCPINC_2", "PAOC_2", "PERNP_2", '
                     f'"PINCP_2", "POBP_2", "POVPIP_2", "POWPUMA_2", "POWSP_2", "PRIVCOV_2", "PUBCOV_2", "QTRBIR_2", '
                     f'"RAC1P_2", "RAC2P_2", "RAC3P_2", "RACAIAN_2", "RACASN_2", "RACBLK_2", "RACNH_2", "RACNUM_2", '
                     f'"RACPI_2", "RACSOR_2", "RACWHT_2", "RC_2", "SCIENGP_2", "SCIENGRLP_2", "SFN_2", "SFR_2", '
                     f'"SOCP_2", "VPS_2", "WAOB_2", "RT_3", "SPORDER_3", "ADJINC_3", "PWGTP_3", "AGEP_3", "CIT_3", '
                     f'"CITWP_3", "COW_3", "DDRS_3", "DEAR_3", "DEYE_3", "DOUT_3", "DPHY_3", "DRAT_3", "DRATX_3", '
                     f'"DREM_3", "ENG_3", "FER_3", "GCL_3", "GCM_3", "GCR_3", "HINS1_3", "HINS2_3", "HINS3_3", '
                     f'"HINS4_3", "HINS5_3", "HINS6_3", "HINS7_3", "INTP_3", "JWMNP_3", "JWRIP_3", "JWTR_3", '
                     f'"LANP_3", "LANX_3", "MAR_3", "MARHD_3", "MARHM_3", "MARHT_3", "MARHW_3", "MARHYP_3", "MIG_3", '
                     f'"MIL_3", "MLPA_3", "MLPB_3", "MLPCD_3", "MLPE_3", "MLPFG_3", "MLPH_3", "MLPI_3", "MLPJ_3", '
                     f'"MLPK_3", "NWAB_3", "NWAV_3", "NWLA_3", "NWLK_3", "NWRE_3", "OIP_3", "PAP_3", "RELP_3", '
                     f'"RETP_3", "SCH_3", "SCHG_3", "SCHL_3", "SEMP_3", "SEX_3", "SSIP_3", "SSP_3", "WAGP_3", '
                     f'"WKHP_3", "WKL_3", "WKW_3", "WRK_3", "YOEP_3", "ANC_3", "ANC1P_3", "ANC2P_3", "DECADE_3", '
                     f'"DIS_3", "DRIVESP_3", "ESP_3", "ESR_3", "FOD1P_3", "FOD2P_3", "HICOV_3", "HISP_3", "INDP_3", '
                     f'"JWAP_3", "JWDP_3", "MIGPUMA_3", "MIGSP_3", "MSP_3", "NAICSP_3", "NATIVITY_3", "NOP_3", '
                     f'"OC_3", "OCCP_3", "OCCPINC_3", "PAOC_3", "PERNP_3", "PINCP_3", "POBP_3", "POVPIP_3", '
                     f'"POWPUMA_3", "POWSP_3", "PRIVCOV_3", "PUBCOV_3", "QTRBIR_3", "RAC1P_3", "RAC2P_3", "RAC3P_3", '
                     f'"RACAIAN_3", "RACASN_3", "RACBLK_3", "RACNH_3", "RACNUM_3", "RACPI_3", "RACSOR_3", "RACWHT_3", '
                     f'"RC_3", "SCIENGP_3", "SCIENGRLP_3", "SFN_3", "SFR_3", "SOCP_3", "VPS_3", "WAOB_3"')
        self.assertEqual(names_res.value, names_str)

        data_df = load_data_source(["./tests/output/flat_personsIL2016.parquet"])
        self.assertEqual(data_df.shape[0], 1449)
        self.assertEqual(data_df.shape[1], 381)
        feature_def = FeatureJsonFileSource("./tests/output/features_flat-PUMS-P.json").feature_definitions
        self.assertEqual(len(feature_def), 381)


if __name__ == '__main__':
    unittest.main()
