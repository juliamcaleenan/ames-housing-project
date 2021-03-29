<img src="http://imgur.com/1ZcRyrc.png" align="left" height="55px">

# Regression and Classification with the Ames Housing Data

### Third project for General Assembly Data Science Immersive course

This project uses the [Ames housing data made available on kaggle](https://www.kaggle.com/c/house-prices-advanced-regression-techniques).

### Project Aim
#### Part 1: Estimating the value of homes from fixed characteristics
Build a reliable estimator for the price of the house given characteristics of the house that cannot be renovated. Some examples include:
- The neighbourhood
- Square feet
- Bedrooms, bathrooms
- Basement and garage space

Train a model on pre-2010 data and evaluate its performance on the 2010 houses.

#### Part 2: Determine any value of changeable property characteristics unexplained by the fixed ones
Some examples of things that **ARE renovateable:**
- Roof and exterior features
- "Quality" metrics, such as kitchen quality
- "Condition" metrics, such as condition of garage
- Heating and electrical components

#### Part 3: What property characteristics predict an "abnormal" sale?
Determine which features predict "abnormal" sales e.g. foreclosures.

## Files in this repository
- Jupyter Notebook files:
  - [Data cleaning and EDA](data_cleaning_EDA.ipynb)
  - [Modelling part 1 and part 2](modelling_parts_1_and_2.ipynb)
  - [Modelling part 3](modelling_part_3.ipynb)
- Python file:
  - [Useful functions](useful_functions.py)
