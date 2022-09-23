import sys
from pathlib import Path

from censyn.datasources import datasource
from censyn.evaluator import evaluator
from censyn.metrics import metrics
from censyn.report import report


def main():
    """
    1) Use DataSource to get two dataframes from two files.
    2) Instantiate new Evaluator. Load an Evaluator with two dataframes.
    3) Create Metrics and register them w/ the Evaluator.
    4) Get results by calling Evaluator.evaluate
    5) Instantiate a Report object with evaluation results
    6) call produce_report() from Report
    """

    table_socCha_15_name = 'ACS_15_5YR_DP02_socCh.csv'
    table_socCha_16_name = 'ACS_16_5YR_DP02_socCh.csv'

    assets_path = Path(__file__).resolve().parent.parent / 'tests' / 'assets'

    table_1_full_file_name = str(assets_path / table_socCha_15_name)
    table_2_full_file_name = str(assets_path / table_socCha_16_name)

    df_1 = datasource.DelimitedDataSource(path_to_file=table_1_full_file_name).to_dataframe()
    df_2 = datasource.DelimitedDataSource(path_to_file=table_2_full_file_name).to_dataframe()

    marginal_metric = metrics.MarginalMetric(df_1, df_2, name='main_compare', sample_ratio=1.0)

    evaluation_metrics = [marginal_metric]

    evaluator_instance = evaluator.Evaluator(df_1, df_2, evaluation_metrics)

    results = evaluator_instance.evaluate()

    console_report = report.ConsoleReport()
    for k, v in results.items():
        console_report.add_result(key=k, value=v)

    console_report.produce_report()


def command_line_start() -> None:
    args = sys.argv[1:]
    if len(args) == 1 and args[0] == 'start':
        main()


if __name__ == '__main__':
    main()
