# CenSyn Evaluation Framework v0.8.1

<li>Lead Privacy Researcher: Christine Task</li>
<li>Lead Developer: Jeffrey Hodges</li>
<li>Software Engineer: Damon Streat</li>
<li>Data Analyst: Ashley Simpson</li>
<li>Technical Writer: David Lee</li>

## Building / Installing the Evaluation Framework
We recommend an [Anaconda Python Environment](https://www.anaconda.com/distribution/). This package should work with any Python 3.7 setup, but Anaconda's is by far the easiest to set up.

### Python Environment

To create a new Anaconda environment, enter the command ("censyn" can be any name you prefer):
```bash
conda create --name censyn python=3.8
```

And activate it with:
```bash
conda activate censyn
```

### Building and Installing
Run in the CenSyn root directory, after satisfying requirements.txt:
```bash
pip install .
```

### Version Check and Command Line Help
Displaying the help information will also display the version information (at the end). 
Call `censyn` at the command line with the `-h` flag to display argument information and current version.
```bash
censyn -h
```

## Running The Synthesizer
The following command, with the default configuration, will run a synthesis on the data, and create a Report named `synthesis_report.txt` and a `synthetic.parquet` in the `output` directory.
```bash
censynthesize --synthesize_config_file conf/synthesize.cfg
```

## Running The Evaluation Framework
The following command, with the default configuration, will run an evaluation of the two data sets from using the Marginal Metric evaluation method, and create a Report named `report.txt` in the `output` directory.
```bash
censyn --eval_config_file conf/eval.cfg
```

