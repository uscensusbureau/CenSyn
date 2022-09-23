# This is a example file it will be remove before this gets merged.

from censyn import asserts
from censyn.datasources import ParquetDataSource


def example_run():
    df = ParquetDataSource(path_to_file='/Users/micahheineck/Documents/Knexus/CenSyn/data/synthetic-persons-IL-2016.parquet').to_dataframe()

    example_to_run = asserts.ExampleAssert().assert_list

    results = [item.execute(df=df) for item in example_to_run]

    print(results)


example_run()
