# Pizza Challenge

All is explained in the associated [documentation]().

## Creata your training environment

Clone the project locally and create a new `conda` environment.

```bash
git clone https://github.com/ChainYo/pizza-challenge.git
cd pizza-challenge

conda create -n pizza-challenge python=3.8
conda activate pizza-challenge

poetry install
```

Then you can run the training pipeline bye using this command:

```bash
kedro run
```

If you want to change some training parameters, you need to edit the `conf/base/parameters.yml` file.


## Kedro Commands

- Run pipeline: 

```bash
kedro run
```

- Build requirements file:

```bash
kedro build-reqs
```

- Run tests

```bash
kedro test
```
